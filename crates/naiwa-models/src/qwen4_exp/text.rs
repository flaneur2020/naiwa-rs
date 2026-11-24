//! Top-level Qwen3.8-Flash-Next (`qwen4exp`) text model forward pass, MTP drafting, and rollback.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use naiwa_attn::{GatedDeltaNet, PagedAttnMeta, RotaryEmbedding};
use naiwa_quant::{GgufArchive, QLinear, RoutedExpertsProvider};

use crate::cache::{HybridCache, RecurrentVerifyStash};
use crate::probe;
use crate::qwen4_exp::{
    config::Qwen4ExpConfig,
    hyper::GatedResidual,
    loader::{load_qwen4_exp_from_archive, open_gguf_archive},
    moe::SparseMoeBlock,
    mtp::{MtpForwardInput, MtpForwardOutput, Qwen4ExpMtpHead},
    ple::PleLayer,
    qsa::{QsaAttention, QsaMode, QsaStep},
};

pub(crate) enum Mixer {
    Attention(QsaAttention),
    Linear(GatedDeltaNet),
}

pub(crate) struct DecoderLayer {
    pub(crate) mixer: Mixer,
    pub(crate) ple: Option<PleLayer>,
    pub(crate) attn_hc: GatedResidual,
    pub(crate) mlp_hc: GatedResidual,
    pub(crate) moe: SparseMoeBlock,
}

/// Batch mode for [`Qwen4ExpModel::forward`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardKind {
    /// Prefill (full or chunked prompt processing).
    Prefill,
    /// Single-token autoregressive decode (`seq_len == 1` per sequence).
    Decode,
    /// Multi-token MTP verification (`seq_len == k + 1` per sequence).
    Verify,
}

/// Inputs to one target-model forward pass (`Prefill`, `Decode`, or `Verify`).
pub struct ForwardInput<'a> {
    /// Flattened/rectangular token IDs on host (length `batch * seq_len`).
    pub tokens_host: &'a [u32],
    /// `[batch, seq_len]` U32 token IDs on device.
    pub input_ids: &'a Tensor,
    /// `[batch, seq_len]` U32 1D text positions on device.
    pub position_ids: &'a Tensor,
    /// Recurrent state slot index in [`HybridCache`] for each logical sequence.
    pub cache_slots: &'a [usize],
    /// `(start, len)` span of each logical sequence on the flattened token axis (`0..batch*seq_len`).
    pub seq_spans: &'a [(usize, usize)],
    /// Post-step KV length of each logical sequence.
    pub kv_lens: &'a [usize],
    /// Paged attention metadata for all `FullAttention` layers.
    pub paged_meta: &'a PagedAttnMeta,
    /// Step kind (`Prefill`, `Decode`, or `Verify`).
    pub kind: ForwardKind,
    /// Optional flattened indices (`0..batch*seq_len`) at which to compute logits.
    /// `None` computes logits for every token row.
    pub logit_indices: Option<&'a [u32]>,
    /// Capture the pre-final-mixer 4-stream residual (`[batch, seq_len, hc * hidden]`) for MTP.
    pub capture_spec_hidden: bool,
    /// Capture GDN + PLE rollback stashes so a rejected verify suffix can be undone bit-exactly.
    pub capture_verify_stash: bool,
}

/// Outputs of [`Qwen4ExpModel::forward`].
pub struct ForwardOutput {
    /// Logits `[num_selected_rows, vocab_size]` in F32 (or `[batch * seq_len, vocab_size]`).
    pub logits: Tensor,
    /// Pre-final-mixer 4-stream residual `[batch, seq_len, hc * hidden]` when requested.
    pub spec_hidden: Option<Tensor>,
    /// Recurrent rollback stash when `capture_verify_stash == true`.
    pub verify_stash: Option<RecurrentVerifyStash>,
}

/// Complete single-device Qwen3.8-Flash-Next (`qwen4exp`) text model.
pub struct Qwen4ExpModel {
    cfg: Qwen4ExpConfig,
    embed_tokens: QLinear,
    layers: Vec<DecoderLayer>,
    final_mixer: GatedResidual,
    lm_head: QLinear,
    rotary_emb: Arc<RotaryEmbedding>,
    mtp: Option<Qwen4ExpMtpHead>,
    device: Device,
    dtype: DType,
}

impl Qwen4ExpModel {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        cfg: Qwen4ExpConfig,
        embed_tokens: QLinear,
        layers: Vec<DecoderLayer>,
        final_mixer: GatedResidual,
        lm_head: QLinear,
        rotary_emb: Arc<RotaryEmbedding>,
        mtp: Option<Qwen4ExpMtpHead>,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            cfg,
            embed_tokens,
            layers,
            final_mixer,
            lm_head,
            rotary_emb,
            mtp,
            device,
            dtype,
        }
    }

    /// Open GGUF shards/components and load the model directly onto `device`.
    pub fn load_gguf<P: AsRef<Path>>(
        paths: &[P],
        device: &Device,
        dtype: DType,
        moe_provider: Option<Arc<dyn RoutedExpertsProvider>>,
        load_mtp: bool,
    ) -> anyhow::Result<(Self, Arc<GgufArchive>)> {
        let archive = open_gguf_archive(paths)?;
        let model = load_qwen4_exp_from_archive(
            Arc::clone(&archive),
            device,
            dtype,
            moe_provider,
            load_mtp,
        )?;
        Ok((model, archive))
    }

    /// Load from an already-opened [`GgufArchive`].
    pub fn load_from_archive(
        archive: Arc<GgufArchive>,
        device: &Device,
        dtype: DType,
        moe_provider: Option<Arc<dyn RoutedExpertsProvider>>,
        load_mtp: bool,
    ) -> anyhow::Result<Self> {
        load_qwen4_exp_from_archive(archive, device, dtype, moe_provider, load_mtp)
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn mtp_kv_layer_idx(&self) -> Option<usize> {
        self.mtp.as_ref().map(Qwen4ExpMtpHead::kv_layer_idx)
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding_forward(input_ids, self.dtype)
    }

    /// Project collapsed hidden states `[.., hidden]` through `lm_head` -> `[.., vocab]` in F32.
    pub fn lm_head_forward(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden)?.to_dtype(DType::F32)
    }

    /// Run one target-model forward pass (`Prefill`, `Decode`, or `Verify`).
    pub fn forward(
        &self,
        input: &ForwardInput<'_>,
        cache: &mut HybridCache,
    ) -> Result<ForwardOutput> {
        let xs = self.embed_tokens(input.input_ids)?; // [batch, seq_len, hidden]
        let (batch, seq_len, _hidden) = xs.dims3()?;
        let n_tokens = batch * seq_len;

        let cos_sin = self.rotary_emb.compute_cos_sin(input.position_ids, self.dtype)?;

        let rectangular = matches!(input.kind, ForwardKind::Decode | ForwardKind::Verify);
        let mode = if rectangular {
            QsaMode::Sparse {
                max_blocks: self.cfg.max_position_embeddings / self.cfg.qsa.compress_ratio,
            }
        } else {
            let max_kv = input.kv_lens.iter().copied().max().unwrap_or(0);
            QsaMode::for_max_kv_len(&self.cfg.qsa, max_kv)
        };

        #[cfg(feature = "cuda")]
        let layout = if self.device.is_cuda() {
            Some(if rectangular {
                naiwa_attn::qwen4_exp::TokenLayout::rectangular(batch, seq_len)
            } else {
                naiwa_attn::qwen4_exp::TokenLayout::from_host(
                    input.seq_spans,
                    input.kv_lens,
                    n_tokens,
                    &self.device,
                )?
            })
        } else {
            None
        };

        let qsa_step = QsaStep {
            mode,
            #[cfg(feature = "cuda")]
            layout: layout.as_ref(),
        };

        let mut verify_stash = input.capture_verify_stash.then(|| RecurrentVerifyStash {
            slots: input.cache_slots.to_vec(),
            gdn_layers: Vec::new(),
            ple: None,
        });

        let mut res = xs.repeat((1, 1, self.cfg.hc_count))?;
        let mut pending_norm: Option<Tensor> = None;

        for (i, layer) in self.layers.iter().enumerate() {
            if i == 0 {
                probe::boundary(&res, i, "next_model");
            }
            probe::boundary(&res, i, "layer_start");

            if let Some(ple) = &layer.ple {
                let ple_states = cache
                    .ple_states_mut()
                    .ok_or_else(|| candle_core::Error::msg("HybridCache missing PLE states"))?;
                let mut ple_stash = None;
                res = ple.forward(
                    &res,
                    input.tokens_host,
                    input.seq_spans,
                    input.cache_slots,
                    ple_states,
                    verify_stash.as_ref().map(|_| &mut ple_stash),
                )?;
                if let (Some(vs), Some(ps)) = (verify_stash.as_mut(), ple_stash) {
                    vs.ple = Some(ps);
                }
                pending_norm = None;
            }

            let xn = match pending_norm.take() {
                Some(xn) => xn,
                None => layer.attn_hc.norm(&res)?,
            };
            probe::boundary(&xn, i, "attn_norm_end");
            let h = layer.attn_hc.mix(&xn)?;
            probe::boundary(&h, i, "attn_mix_end");
            let inject = layer.attn_hc.inject(&xn)?;
            probe::boundary(&h, i, "mixer_start");

            let out = match &layer.mixer {
                Mixer::Attention(attn) => {
                    let kv = cache.kv_layer(i).ok_or_else(|| {
                        candle_core::Error::Msg(format!("HybridCache missing KV cache for layer {i}"))
                    })?;
                    attn.forward(&h, &cos_sin, kv, input.paged_meta, &qsa_step)?
                }
                Mixer::Linear(gdn) => {
                    let layer_slots = cache.gdn_layer_mut(i).ok_or_else(|| {
                        candle_core::Error::Msg(format!("HybridCache missing GDN state for layer {i}"))
                    })?;
                    let mut seq_refs = HybridCache::collect_gdn_mut(layer_slots, input.cache_slots)?;
                    if let Some(vs) = verify_stash.as_mut() {
                        let mut stashes = Vec::with_capacity(input.cache_slots.len());
                        let y = gdn.forward_batch(
                            &h,
                            input.seq_spans,
                            &mut seq_refs,
                            Some(&mut stashes),
                        )?;
                        vs.gdn_layers.push((i, stashes));
                        y
                    } else {
                        gdn.forward_batch(&h, input.seq_spans, &mut seq_refs, None)?
                    }
                }
            };
            probe::boundary(&out, i, "mixer_end");

            let (next_res, xn) = layer
                .attn_hc
                .combine(&res, &out, &inject, Some(&layer.mlp_hc))?;
            res = next_res;
            let xn = match xn {
                Some(xn) => xn,
                None => layer.mlp_hc.norm(&res)?,
            };
            probe::boundary(&xn, i, "mlp_norm_end");
            let h = layer.mlp_hc.mix(&xn)?;
            probe::boundary(&h, i, "mlp_mix_end");
            let inject = layer.mlp_hc.inject(&xn)?;
            probe::boundary(&h, i, "moe_start");

            let out = layer.moe.forward(&h)?;
            probe::boundary(&out, i, "moe_end");

            let next_hc = match self.layers.get(i + 1) {
                Some(next) if next.ple.is_none() => Some(&next.attn_hc),
                Some(_) => None,
                None => Some(&self.final_mixer),
            };
            let (next_res, xn) = layer.mlp_hc.combine(&res, &out, &inject, next_hc)?;
            res = next_res;
            pending_norm = xn;
            probe::boundary(&res, i, "layer_end");
        }

        let spec_hidden = input.capture_spec_hidden.then(|| res.clone());

        probe::boundary(&res, self.layers.len(), "tail_start");
        let xn = match pending_norm {
            Some(xn) => xn,
            None => self.final_mixer.norm(&res)?,
        };
        let flat_xn = xn.reshape((n_tokens, self.cfg.hc_hidden_size()))?;
        let selected_xn = match input.logit_indices {
            Some(indices) => {
                let idx_t = Tensor::from_slice(indices, (indices.len(),), &self.device)?;
                flat_xn.index_select(&idx_t, 0)?
            }
            None => flat_xn,
        };
        let xs_final = self.final_mixer.mix(&selected_xn)?;
        probe::boundary(&xs_final, self.layers.len(), "head_start");
        let logits = self.lm_head_forward(&xs_final)?;
        probe::boundary(&logits, self.layers.len(), "head_end");

        Ok(ForwardOutput {
            logits,
            spec_hidden,
            verify_stash,
        })
    }

    /// Run one Built-in MTP drafter step (warm-up chunk or chained draft token).
    pub fn forward_mtp(
        &self,
        input: &MtpForwardInput<'_>,
        cache: &HybridCache,
    ) -> Result<MtpForwardOutput> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("MTP head is not loaded"))?;
        let kv = cache.kv_layer(mtp.kv_layer_idx()).ok_or_else(|| {
            candle_core::Error::msg("HybridCache missing KV cache for MTP head")
        })?;
        mtp.forward(input, kv)
    }

    /// Bit-exact rollback of GDN + PLE recurrent states after partial MTP speculative rejection.
    ///
    /// `rows` contains `(batch_idx, keep_tokens)` pairs for sequences that rejected a suffix.
    pub fn rollback_recurrent_prefixes(
        &self,
        cache: &mut HybridCache,
        stash: &RecurrentVerifyStash,
        rows: &[(usize, usize)],
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        for (layer_idx, layer_stashes) in &stash.gdn_layers {
            let Mixer::Linear(gdn) = &self.layers[*layer_idx].mixer else {
                continue;
            };
            let layer_slots = cache.gdn_layer_mut(*layer_idx).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "HybridCache missing GDN state for layer {layer_idx}"
                ))
            })?;
            for &(batch_idx, keep) in rows {
                let slot = stash.slots[batch_idx];
                gdn.replay_prefix(&mut layer_slots[slot], &layer_stashes[batch_idx], keep)?;
            }
        }
        if let (Some(ple_stash), Some(ple_cfg)) = (&stash.ple, &self.cfg.ple) {
            cache.rollback_ple(ple_stash, &stash.slots, rows, ple_cfg)?;
        }
        Ok(())
    }
}
