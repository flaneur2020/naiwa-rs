//! Built-in Multi-Token Prediction (MTP) speculative decoding driver for Qwen3.8-Flash-Next.
//!
//! Pure Rust/Tensor implementation (no CUDA graph capture, no `speculative_rejection.cu`):
//! - Chunk-by-chunk MTP KV cache warm-up during prefill (O(chunk) VRAM, never stores full prompt hidden).
//! - Fast GPU-argmax greedy drafting & verification (copies only `k+1` u32 token IDs per step).
//! - Exact stochastic Leviathan rejection sampling on CPU F32 when `temperature > 0`.
//! - Bit-exact KV + GDN + PLE rollback on partial draft rejection.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use naiwa_models::{MtpForwardInput, Qwen4ExpModel};

use crate::block_manager::BlockManager;
use crate::sampler::{logits_to_filtered_probs, sample_categorical, Sampler};

/// Draft sampling policy for Built-in MTP.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MtpDraftSampling {
    #[default]
    Auto,
    Greedy,
    Probabilistic,
}

/// Built-in MTP configuration.
#[derive(Clone, Debug)]
pub struct MtpConfig {
    /// Number of draft tokens proposed per target verify step (default 2).
    pub n_predict: usize,
    /// Draft sampling policy.
    pub draft_sampling: MtpDraftSampling,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            n_predict: 2,
            draft_sampling: MtpDraftSampling::Auto,
        }
    }
}

/// Cumulative MTP acceptance counters (exposed on `/metrics`).
#[derive(Default)]
pub struct MtpStats {
    pub steps: AtomicU64,
    pub drafted_tokens: AtomicU64,
    pub accepted_tokens: AtomicU64,
}

impl MtpStats {
    pub fn record(&self, drafted: usize, accepted: usize) {
        self.steps.fetch_add(1, Ordering::Relaxed);
        self.drafted_tokens
            .fetch_add(drafted as u64, Ordering::Relaxed);
        self.accepted_tokens
            .fetch_add(accepted as u64, Ordering::Relaxed);
    }
}

/// One proposed draft token plus optional draft distribution (for stochastic rejection sampling).
#[derive(Clone)]
pub struct DraftProposal {
    pub token_id: u32,
    /// Filtered draft probabilities (populated only when stochastic verification is active).
    pub probs: Option<HashMap<u32, f32>>,
}

/// Per-sequence MTP draft state maintained between target forward passes.
#[derive(Default)]
pub struct MtpSeqState {
    /// Draft tokens queued for the next target verify pass (`[d_0, ..., d_{k-1}]`).
    pub pending_drafts: Vec<DraftProposal>,
}

/// Outcome of verifying `k` draft tokens against target `logits[0..k+1]`.
pub struct VerifyOutcome {
    /// Accepted draft tokens (`0..=k`).
    pub accepted_drafts: Vec<u32>,
    /// Bonus or correction token (`logits[accepted]`).
    pub bonus_token: u32,
    /// Total tokens kept from the target verify pass (`accepted_drafts.len() + 1`).
    pub keep_rows: usize,
}

/// Warm up the MTP KV cache on a prefill chunk (`start..start+len`) where the shifted next tokens
/// `shifted_tokens` (length `len`) are already known.
///
/// When `propose_first_draft` is `true` (the final prompt chunk), returns `(first_draft, chain_hidden)`
/// for the last row `len - 1`; otherwise returns `None`.
#[allow(clippy::too_many_arguments)]
pub fn mtp_warmup_chunk(
    model: &Qwen4ExpModel,
    cache: &naiwa_models::HybridCache,
    bm: &BlockManager,
    block_table: &[u32],
    spec_hidden: &Tensor,
    shifted_tokens: &[u32],
    start_pos: usize,
    propose_first_draft: bool,
    sampler: &mut Sampler,
    mtp_cfg: &MtpConfig,
    device: &Device,
) -> Result<Option<(DraftProposal, Tensor)>> {
    let len = shifted_tokens.len();
    debug_assert_eq!(len, spec_hidden.dim(1)?);
    let end_kv_len = start_pos + len;

    let embeds = model.embed_tokens_host(shifted_tokens, &[1, len])?;
    let pos_vec: Vec<u32> = (start_pos as u32..end_kv_len as u32).collect();
    let pos_t = Tensor::from_slice(&pos_vec, (1, len), device)?;

    let meta = bm.build_meta(&[block_table], &[end_kv_len], &[len], false, device)?;
    let spans = [(0usize, len)];
    let kv_lens = [end_kv_len];

    let out = model.forward_mtp(
        &MtpForwardInput {
            input_embeds: &embeds,
            target_hidden: spec_hidden,
            position_ids: &pos_t,
            paged_meta: &meta,
            chunk_spans: Some((&spans, &kv_lens)),
        },
        cache,
    )?;

    if !propose_first_draft {
        return Ok(None);
    }

    let last_logits_hidden = out.logits_hidden.i((0, len - 1))?.unsqueeze(0)?; // [1, hidden]
    let logits = if sampler.is_greedy() || mtp_cfg.draft_sampling == MtpDraftSampling::Greedy {
        model.lm_head_forward_raw(&last_logits_hidden)?
    } else {
        model.lm_head_forward(&last_logits_hidden)?
    };
    let proposal = sample_draft(&logits, sampler, mtp_cfg)?;
    let last_chain = out.chain_hidden.narrow(1, len - 1, 1)?.contiguous()?; // [1, 1, hc*hidden]
    Ok(Some((proposal, last_chain)))
}

/// Propose `k` total drafts given the already-computed first draft `d_0` and `chain_hidden`
/// (produced by `mtp_warmup_chunk` at `base_pos = P - 1`), chaining `k - 1` single-token MTP calls
/// at positions `base_pos + 1 .. base_pos + k`.
#[allow(clippy::too_many_arguments)]
pub fn mtp_chain_drafts(
    model: &Qwen4ExpModel,
    cache: &naiwa_models::HybridCache,
    bm: &mut BlockManager,
    block_table: &mut Vec<u32>,
    first_draft: DraftProposal,
    mut chain_hidden: Tensor,
    base_pos: usize,
    k: usize,
    sampler: &mut Sampler,
    mtp_cfg: &MtpConfig,
    device: &Device,
) -> Result<Vec<DraftProposal>> {
    let mut drafts = Vec::with_capacity(k);
    drafts.push(first_draft);

    for step in 1..k {
        let pos = base_pos + step;
        let kv_len = pos + 1;
        if !bm.ensure_capacity(block_table, kv_len) {
            break;
        }
        let prev_tok = [drafts.last().unwrap().token_id];
        let embeds = model.embed_tokens_host(&prev_tok, &[1, 1])?;
        let pos_t = Tensor::from_slice(&[pos as u32], (1, 1), device)?;
        let meta = bm.build_meta(&[block_table.as_slice()], &[kv_len], &[1], true, device)?;

        let out = model.forward_mtp(
            &MtpForwardInput {
                input_embeds: &embeds,
                target_hidden: &chain_hidden,
                position_ids: &pos_t,
                paged_meta: &meta,
                chunk_spans: None,
            },
            cache,
        )?;
        let logits_h = out.logits_hidden.reshape((1, model.config().hidden_size))?;
        let logits = if sampler.is_greedy() || mtp_cfg.draft_sampling == MtpDraftSampling::Greedy {
            model.lm_head_forward_raw(&logits_h)?
        } else {
            model.lm_head_forward(&logits_h)?
        };
        let proposal = sample_draft(&logits, sampler, mtp_cfg)?;
        drafts.push(proposal);
        chain_hidden = out.chain_hidden;
    }

    Ok(drafts)
}

/// Verify `drafts` (`len = k`) against target `logits` (`[k + 1, vocab]` F32).
pub fn verify_drafts(
    logits: &Tensor,
    drafts: &[DraftProposal],
    sampler: &mut Sampler,
) -> Result<VerifyOutcome> {
    let k = drafts.len();
    if sampler.is_greedy() {
        // Fast GPU path: single argmax kernel, copy only k+1 u32s to CPU!
        let target_ids = Sampler::sample_batch_greedy(logits)?;
        let mut accepted_drafts = Vec::with_capacity(k);
        for (i, d) in drafts.iter().enumerate() {
            if target_ids[i] == d.token_id {
                accepted_drafts.push(d.token_id);
            } else {
                break;
            }
        }
        let accepted = accepted_drafts.len();
        let bonus_token = target_ids[accepted];
        return Ok(VerifyOutcome {
            accepted_drafts,
            bonus_token,
            keep_rows: accepted + 1,
        });
    }

    // Stochastic Leviathan rejection sampling on CPU F32
    let logits_f32 = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let params = sampler.params().clone();
    let mut accepted_drafts = Vec::with_capacity(k);
    for (i, d) in drafts.iter().enumerate() {
        let p_target_pairs = logits_to_filtered_probs(
            &logits_f32[i],
            params.temperature,
            params.top_k,
            params.top_p,
        );
        let p_t = p_target_pairs
            .iter()
            .find(|&&(id, _)| id == d.token_id)
            .map_or(0.0f32, |&(_, p)| p);
        let p_d = d
            .probs
            .as_ref()
            .and_then(|m| m.get(&d.token_id).copied())
            .unwrap_or(1.0f32);
        let accept_prob = if p_d <= 0.0 { 1.0 } else { (p_t / p_d).min(1.0) };
        if sampler.next_uniform() < accept_prob {
            accepted_drafts.push(d.token_id);
        } else {
            // Sample correction from normalized max(0, p_target - p_draft)
            let residual: Vec<(u32, f32)> = p_target_pairs
                .iter()
                .filter_map(|&(id, pt)| {
                    let pd = d
                        .probs
                        .as_ref()
                        .and_then(|m| m.get(&id).copied())
                        .unwrap_or(0.0f32);
                    let diff = (pt - pd).max(0.0);
                    (diff > 0.0).then_some((id, diff))
                })
                .collect();
            let bonus_token = if residual.is_empty() {
                sampler.sample_from_f32(&logits_f32[i])
            } else {
                let sum: f32 = residual.iter().map(|(_, p)| *p).sum::<f32>().max(1e-30);
                let norm: Vec<(u32, f32)> = residual.into_iter().map(|(id, p)| (id, p / sum)).collect();
                let u = sampler.next_uniform();
                let mut acc = 0.0f32;
                let mut chosen = norm[0].0;
                for &(id, p) in &norm {
                    acc += p;
                    if u <= acc {
                        chosen = id;
                        break;
                    }
                }
                chosen
            };
            let keep_rows = accepted_drafts.len() + 1;
            return Ok(VerifyOutcome {
                accepted_drafts,
                bonus_token,
                keep_rows,
            });
        }
    }

    // All k drafts accepted: sample bonus token from logits[k]
    let bonus_token = sampler.sample_from_f32(&logits_f32[k]);
    let keep_rows = accepted_drafts.len() + 1;
    Ok(VerifyOutcome {
        accepted_drafts,
        bonus_token,
        keep_rows,
    })
}

fn sample_draft(
    logits: &Tensor,
    sampler: &mut Sampler,
    mtp_cfg: &MtpConfig,
) -> Result<DraftProposal> {
    let greedy_draft = sampler.is_greedy() || mtp_cfg.draft_sampling == MtpDraftSampling::Greedy;
    if greedy_draft {
        let token_id = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
        let probs = if sampler.is_greedy() {
            None
        } else {
            let mut m = HashMap::with_capacity(1);
            m.insert(token_id, 1.0f32);
            Some(m)
        };
        return Ok(DraftProposal { token_id, probs });
    }
    let row = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let p = sampler.params();
    let pairs = logits_to_filtered_probs(&row, p.temperature, p.top_k, p.top_p);
    let mut rng = rand::rngs::StdRng::seed_from_u64(
        sampler.next_uniform().to_bits() as u64 ^ 0x9E37_79B9_7F4A_7C15,
    );
    let token_id = sample_categorical(&pairs, &mut rng);
    let probs = pairs.into_iter().collect::<HashMap<u32, f32>>();
    Ok(DraftProposal {
        token_id,
        probs: Some(probs),
    })
}

use rand::SeedableRng;
