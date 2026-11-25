//! Continuous-batching + chunked-prefill + Built-in MTP async inference engine.
//!
//! Runs a dedicated synchronous CUDA worker thread owning `Qwen4ExpModel`, `HybridCache`, and
//! `BlockManager`, driven by async HTTP handlers over `tokio::sync::mpsc` channels.

use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Tensor;
use naiwa_models::{ForwardInput, ForwardKind, HybridCache, Qwen4ExpModel};
use tokio::sync::mpsc;

use crate::block_manager::{BlockManager, DEFAULT_BLOCK_SIZE};
use crate::mtp::{
    mtp_chain_drafts, mtp_warmup_chunk, verify_drafts, MtpConfig, MtpSeqState, MtpStats,
};
use crate::sampler::{Sampler, SamplingParams};
use crate::tokenizer::{GgufTokenizer, StreamDecoder};

pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 4096;
pub const DEFAULT_MAX_PREFILL_CHUNK_TOKENS: usize = 2048;
pub const DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL: usize = 4;

/// Runtime configuration for the inference engine.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub max_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_prefill_chunk_tokens: usize,
    pub max_decode_steps_before_prefill: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub default_seed: u64,
    pub mtp: Option<MtpConfig>,
    /// Optional live prompt-chunk ceiling updated by `naiwa-offload::PrefillStreams`.
    pub prompt_step_ceiling: Option<Arc<AtomicUsize>>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_seqs: 4,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_prefill_chunk_tokens: DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
            max_decode_steps_before_prefill: DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL,
            num_blocks: 128,
            block_size: DEFAULT_BLOCK_SIZE,
            default_seed: 0,
            mtp: None,
            prompt_step_ceiling: None,
        }
    }
}

/// Reason a sequence finished generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }

    pub fn as_anthropic_str(self) -> &'static str {
        match self {
            Self::Stop => "end_turn",
            Self::Length => "max_tokens",
        }
    }
}

/// Streamed events emitted from the engine worker back to an HTTP request handler.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token {
        id: u32,
        text: String,
    },
    Done {
        finish_reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
        ttft_ms: f64,
        total_ms: f64,
    },
    Error(String),
}

/// Request submitted to [`EngineHandle::generate`].
pub struct GenerationRequest {
    pub prompt_tokens: Vec<u32>,
    pub sampling: SamplingParams,
}

/// Lightweight Prometheus-compatible counters maintained by the engine.
#[derive(Default)]
pub struct EngineMetrics {
    pub requests_total: AtomicU64,
    pub requests_active: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub completion_tokens_total: AtomicU64,
    pub kv_blocks_used: AtomicU64,
    pub kv_blocks_total: AtomicU64,
    pub mtp: MtpStats,
}

impl EngineMetrics {
    /// Render lightweight Prometheus text exposition for `GET /metrics`.
    pub fn render_prometheus(&self) -> String {
        let reqs = self.requests_total.load(Ordering::Relaxed);
        let active = self.requests_active.load(Ordering::Relaxed);
        let ptok = self.prompt_tokens_total.load(Ordering::Relaxed);
        let ctok = self.completion_tokens_total.load(Ordering::Relaxed);
        let used = self.kv_blocks_used.load(Ordering::Relaxed);
        let total = self.kv_blocks_total.load(Ordering::Relaxed);
        let mtp_steps = self.mtp.steps.load(Ordering::Relaxed);
        let mtp_draft = self.mtp.drafted_tokens.load(Ordering::Relaxed);
        let mtp_acc = self.mtp.accepted_tokens.load(Ordering::Relaxed);
        format!(
            "# HELP naiwa_requests_total Total inference requests received\n\
             # TYPE naiwa_requests_total counter\n\
             naiwa_requests_total {reqs}\n\
             # HELP naiwa_requests_active Currently running or queued requests\n\
             # TYPE naiwa_requests_active gauge\n\
             naiwa_requests_active {active}\n\
             # HELP naiwa_prompt_tokens_total Total prompt tokens processed\n\
             # TYPE naiwa_prompt_tokens_total counter\n\
             naiwa_prompt_tokens_total {ptok}\n\
             # HELP naiwa_completion_tokens_total Total completion tokens generated\n\
             # TYPE naiwa_completion_tokens_total counter\n\
             naiwa_completion_tokens_total {ctok}\n\
             # HELP naiwa_kv_blocks_used Currently allocated paged KV blocks\n\
             # TYPE naiwa_kv_blocks_used gauge\n\
             naiwa_kv_blocks_used {used}\n\
             # HELP naiwa_kv_blocks_total Total paged KV blocks in pool\n\
             # TYPE naiwa_kv_blocks_total gauge\n\
             naiwa_kv_blocks_total {total}\n\
             # HELP naiwa_mtp_steps_total Total Built-in MTP verify steps\n\
             # TYPE naiwa_mtp_steps_total counter\n\
             naiwa_mtp_steps_total {mtp_steps}\n\
             # HELP naiwa_mtp_drafted_tokens_total Total MTP drafted tokens\n\
             # TYPE naiwa_mtp_drafted_tokens_total counter\n\
             naiwa_mtp_drafted_tokens_total {mtp_draft}\n\
             # HELP naiwa_mtp_accepted_tokens_total Total MTP accepted draft tokens\n\
             # TYPE naiwa_mtp_accepted_tokens_total counter\n\
             naiwa_mtp_accepted_tokens_total {mtp_acc}\n"
        )
    }
}

struct SubmittedRequest {
    prompt_tokens: Vec<u32>,
    sampling: SamplingParams,
    tx: mpsc::UnboundedSender<StreamEvent>,
    created_at: Instant,
}

/// Cloneable async handle to the running inference engine.
#[derive(Clone)]
pub struct EngineHandle {
    model_id: String,
    tokenizer: Arc<GgufTokenizer>,
    metrics: Arc<EngineMetrics>,
    tx: mpsc::UnboundedSender<SubmittedRequest>,
    max_model_len: usize,
}

impl EngineHandle {
    /// Spawn the synchronous GPU worker thread and return an async [`EngineHandle`].
    pub fn spawn(
        model_id: String,
        model: Qwen4ExpModel,
        tokenizer: Arc<GgufTokenizer>,
        cfg: EngineConfig,
    ) -> Result<Self> {
        let has_mtp = cfg.mtp.is_some() && model.has_mtp();
        if cfg.mtp.is_some() && !model.has_mtp() {
            anyhow::bail!("MTP enabled in EngineConfig, but model has no loaded MTP head");
        }
        let max_model_len = model.config().max_position_embeddings;
        let cache = HybridCache::new(
            model.config(),
            has_mtp,
            cfg.num_blocks,
            cfg.block_size,
            cfg.max_seqs,
            model.dtype(),
            model.device(),
        )
        .context("failed to allocate HybridCache")?;
        model
            .warm_aux_caches(&cache)
            .context("failed to pre-allocate QSA aux caches")?;
        let block_manager = BlockManager::new(cfg.num_blocks, cfg.block_size);

        let metrics = Arc::new(EngineMetrics::default());
        metrics
            .kv_blocks_total
            .store(cfg.num_blocks as u64, Ordering::Relaxed);

        let (tx, rx) = mpsc::unbounded_channel();
        let tok_worker = Arc::clone(&tokenizer);
        let met_worker = Arc::clone(&metrics);
        let worker_cfg = cfg.clone();

        std::thread::Builder::new()
            .name("naiwa-engine".to_string())
            .spawn(move || {
                let mut worker = EngineWorker {
                    model,
                    cache,
                    bm: block_manager,
                    tokenizer: tok_worker,
                    cfg: worker_cfg,
                    metrics: met_worker,
                    waiting: VecDeque::new(),
                    prefilling: Vec::new(),
                    decoding: Vec::new(),
                    decode_steps_since_prefill: 0,
                };
                worker.run(rx);
            })
            .context("failed to spawn naiwa-engine thread")?;

        Ok(Self {
            model_id,
            tokenizer,
            metrics,
            tx,
            max_model_len,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn tokenizer(&self) -> &Arc<GgufTokenizer> {
        &self.tokenizer
    }

    pub fn metrics(&self) -> &Arc<EngineMetrics> {
        &self.metrics
    }

    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    /// Submit a generation request and receive a stream of [`StreamEvent`]s.
    pub fn generate(
        &self,
        req: GenerationRequest,
    ) -> Result<mpsc::UnboundedReceiver<StreamEvent>> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        self.metrics.requests_active.fetch_add(1, Ordering::Relaxed);
        self.tx
            .send(SubmittedRequest {
                prompt_tokens: req.prompt_tokens,
                sampling: req.sampling,
                tx,
                created_at: Instant::now(),
            })
            .map_err(|_| anyhow::anyhow!("inference engine worker has exited"))?;
        Ok(rx)
    }
}

struct ActiveSequence {
    prompt_tokens: Vec<u32>,
    prompt_cursor: usize,
    kv_len: usize,
    cache_slot: usize,
    block_table: Vec<u32>,
    sampler: Sampler,
    decoder: StreamDecoder,
    tx: mpsc::UnboundedSender<StreamEvent>,
    created_at: Instant,
    first_token_at: Option<Instant>,
    generated_tokens: Vec<u32>,
    emitted_text: String,
    last_token: u32,
    mtp: MtpSeqState,
}

impl ActiveSequence {
    fn is_client_gone(&self) -> bool {
        self.tx.is_closed()
    }

    fn emit_token(&mut self, tokenizer: &GgufTokenizer, token_id: u32) -> Result<Option<FinishReason>> {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        let params = self.sampler.params();
        let is_eos = !params.ignore_eos
            && (tokenizer.is_stop_token(token_id) || params.stop_token_ids.contains(&token_id));
        if is_eos {
            let tail = self.decoder.flush(tokenizer)?;
            if !tail.is_empty() {
                self.emitted_text.push_str(&tail);
                let _ = self.tx.send(StreamEvent::Token {
                    id: token_id,
                    text: tail,
                });
            }
            return Ok(Some(FinishReason::Stop));
        }

        self.generated_tokens.push(token_id);
        self.last_token = token_id;
        let delta = self.decoder.push(tokenizer, token_id)?;
        if !delta.is_empty() {
            self.emitted_text.push_str(&delta);
            // Check string stop sequences
            if !params.stop.is_empty() {
                for stop in &params.stop {
                    if !stop.is_empty() && self.emitted_text.ends_with(stop.as_str()) {
                        let keep_delta_len = delta.len().saturating_sub(stop.len());
                        if keep_delta_len > 0 {
                            let _ = self.tx.send(StreamEvent::Token {
                                id: token_id,
                                text: delta[..keep_delta_len].to_string(),
                            });
                        }
                        return Ok(Some(FinishReason::Stop));
                    }
                }
            }
            let _ = self.tx.send(StreamEvent::Token {
                id: token_id,
                text: delta,
            });
        }
        if self.generated_tokens.len() >= params.max_tokens {
            let tail = self.decoder.flush(tokenizer)?;
            if !tail.is_empty() {
                let _ = self.tx.send(StreamEvent::Token {
                    id: token_id,
                    text: tail,
                });
            }
            return Ok(Some(FinishReason::Length));
        }
        Ok(None)
    }

    fn finish(self, reason: FinishReason, metrics: &EngineMetrics) {
        let total_ms = self.created_at.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = self
            .first_token_at
            .map_or(total_ms, |t| t.duration_since(self.created_at).as_secs_f64() * 1000.0);
        let prompt_tokens = self.prompt_tokens.len();
        let completion_tokens = self.generated_tokens.len();
        metrics
            .prompt_tokens_total
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        metrics
            .completion_tokens_total
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        metrics.requests_active.fetch_sub(1, Ordering::Relaxed);
        let _ = self.tx.send(StreamEvent::Done {
            finish_reason: reason,
            prompt_tokens,
            completion_tokens,
            ttft_ms,
            total_ms,
        });
    }

    fn fail(self, msg: String, metrics: &EngineMetrics) {
        metrics.requests_active.fetch_sub(1, Ordering::Relaxed);
        let _ = self.tx.send(StreamEvent::Error(msg));
    }
}

struct EngineWorker {
    model: Qwen4ExpModel,
    cache: HybridCache,
    bm: BlockManager,
    tokenizer: Arc<GgufTokenizer>,
    cfg: EngineConfig,
    metrics: Arc<EngineMetrics>,
    waiting: VecDeque<SubmittedRequest>,
    prefilling: Vec<ActiveSequence>,
    decoding: Vec<ActiveSequence>,
    decode_steps_since_prefill: usize,
}

impl EngineWorker {
    fn run(&mut self, mut rx: mpsc::UnboundedReceiver<SubmittedRequest>) {
        loop {
            // 1. Drain any newly arrived requests (blocking if idle)
            if self.waiting.is_empty() && self.prefilling.is_empty() && self.decoding.is_empty() {
                match rx.blocking_recv() {
                    Some(req) => self.waiting.push_back(req),
                    None => break,
                }
            }
            while let Ok(req) = rx.try_recv() {
                self.waiting.push_back(req);
            }

            // 2. Drop disconnected clients early
            self.reap_disconnected();

            // 3. Admit waiting requests up to max_seqs
            self.admit_waiting();

            // 4. Pick step: chunked prefill vs decode/MTP
            let has_prefill = !self.prefilling.is_empty();
            let has_decode = !self.decoding.is_empty();

            let do_prefill = has_prefill
                && (!has_decode
                    || self.decode_steps_since_prefill >= self.cfg.max_decode_steps_before_prefill);

            if do_prefill {
                self.decode_steps_since_prefill = 0;
                self.step_prefill();
            } else if has_decode {
                self.decode_steps_since_prefill += 1;
                if self.cfg.mtp.is_some() {
                    self.step_decode_mtp();
                } else {
                    self.step_decode_standard();
                }
            }

            self.metrics
                .kv_blocks_used
                .store(self.bm.used_count() as u64, Ordering::Relaxed);
        }
    }

    fn reap_disconnected(&mut self) {
        let mut i = 0;
        while i < self.prefilling.len() {
            if self.prefilling[i].is_client_gone() {
                let mut seq = self.prefilling.swap_remove(i);
                self.bm.free_all(&mut seq.block_table);
                self.cache.free_slot(seq.cache_slot);
                self.metrics.requests_active.fetch_sub(1, Ordering::Relaxed);
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.decoding.len() {
            if self.decoding[i].is_client_gone() {
                let mut seq = self.decoding.swap_remove(i);
                self.bm.free_all(&mut seq.block_table);
                self.cache.free_slot(seq.cache_slot);
                self.metrics.requests_active.fetch_sub(1, Ordering::Relaxed);
            } else {
                i += 1;
            }
        }
    }

    fn admit_waiting(&mut self) {
        while self.prefilling.len() + self.decoding.len() < self.cfg.max_seqs {
            let Some(req) = self.waiting.front() else {
                break;
            };
            if req.prompt_tokens.is_empty() {
                let req = self.waiting.pop_front().unwrap();
                self.metrics.requests_active.fetch_sub(1, Ordering::Relaxed);
                let _ = req.tx.send(StreamEvent::Error("empty prompt".to_string()));
                continue;
            }
            let max_kv = self.bm.num_blocks() * self.bm.block_size();
            if req.prompt_tokens.len() >= max_kv {
                let req = self.waiting.pop_front().unwrap();
                self.metrics.requests_active.fetch_sub(1, Ordering::Relaxed);
                let _ = req.tx.send(StreamEvent::Error(format!(
                    "prompt length {} exceeds KV cache capacity {max_kv}",
                    req.prompt_tokens.len()
                )));
                continue;
            }
            let first_chunk_len = req.prompt_tokens.len().min(self.effective_prefill_chunk());
            let mut block_table = Vec::new();
            if !self.bm.ensure_capacity(&mut block_table, first_chunk_len) {
                break; // wait for running seqs to free KV blocks
            }
            let slot = match self.cache.alloc_slot() {
                Ok(Some(s)) => s,
                Ok(None) => {
                    self.bm.free_all(&mut block_table);
                    break;
                }
                Err(e) => {
                    self.bm.free_all(&mut block_table);
                    let req = self.waiting.pop_front().unwrap();
                    self.metrics.requests_active.fetch_sub(1, Ordering::Relaxed);
                    let _ = req.tx.send(StreamEvent::Error(format!("cache alloc failed: {e}")));
                    continue;
                }
            };
            let req = self.waiting.pop_front().unwrap();
            let sampler = Sampler::new(req.sampling, self.cfg.default_seed);
            self.prefilling.push(ActiveSequence {
                prompt_tokens: req.prompt_tokens,
                prompt_cursor: 0,
                kv_len: 0,
                cache_slot: slot,
                block_table,
                sampler,
                decoder: StreamDecoder::new(),
                tx: req.tx,
                created_at: req.created_at,
                first_token_at: None,
                generated_tokens: Vec::new(),
                emitted_text: String::new(),
                last_token: 0,
                mtp: MtpSeqState::default(),
            });
        }
    }

    fn effective_prefill_chunk(&self) -> usize {
        let base = self
            .cfg
            .max_prefill_chunk_tokens
            .min(self.cfg.max_num_batched_tokens)
            .max(1);
        if let Some(ceiling) = &self.cfg.prompt_step_ceiling {
            let live = ceiling.load(Ordering::Relaxed);
            if live > 0 {
                return base.min(live).max(1);
            }
        }
        base
    }

    fn step_prefill(&mut self) {
        if self.prefilling.is_empty() {
            return;
        }
        // Process the front prefilling sequence chunk (keeps QSA/GDN/PLE chunking super clean and
        // maximizes prefill-stream batch length for that prompt!).
        let chunk_budget = self.effective_prefill_chunk();
        let mut seq = self.prefilling.remove(0);

        let start = seq.prompt_cursor;
        let remaining = seq.prompt_tokens.len() - start;
        let chunk_len = remaining.min(chunk_budget);
        let end = start + chunk_len;
        let is_final_chunk = end == seq.prompt_tokens.len();

        if !self.bm.ensure_capacity(&mut seq.block_table, end) {
            // Put back and let decode finish to free blocks
            self.prefilling.insert(0, seq);
            if !self.decoding.is_empty() {
                if self.cfg.mtp.is_some() {
                    self.step_decode_mtp();
                } else {
                    self.step_decode_standard();
                }
            }
            return;
        }

        let dev = self.model.device().clone();
        let chunk_ids = &seq.prompt_tokens[start..end];
        let pos_vec: Vec<u32> = (start as u32..end as u32).collect();
        let mtp_enabled = self.cfg.mtp.is_some();

        let step_res = (|| -> candle_core::Result<_> {
            let input_ids = Tensor::from_slice(chunk_ids, (1, chunk_len), &dev)?;
            let position_ids = Tensor::from_slice(&pos_vec, (1, chunk_len), &dev)?;
            let meta = self.bm.build_meta(
                &[seq.block_table.as_slice()],
                &[end],
                &[chunk_len],
                false,
                &dev,
            )?;
            let last_idx = [(chunk_len - 1) as u32];
            self.model.forward(
                &ForwardInput {
                    tokens_host: chunk_ids,
                    input_ids: &input_ids,
                    position_ids: &position_ids,
                    cache_slots: &[seq.cache_slot],
                    seq_spans: &[(0, chunk_len)],
                    kv_lens: &[end],
                    paged_meta: &meta,
                    kind: ForwardKind::Prefill,
                    logit_indices: is_final_chunk.then_some(&last_idx[..]),
                    capture_spec_hidden: mtp_enabled,
                    capture_verify_stash: false,
                },
                &mut self.cache,
            )
        })();

        let out = match step_res {
            Ok(o) => o,
            Err(e) => {
                self.bm.free_all(&mut seq.block_table);
                self.cache.free_slot(seq.cache_slot);
                seq.fail(format!("prefill forward failed: {e}"), &self.metrics);
                return;
            }
        };

        seq.prompt_cursor = end;
        seq.kv_len = end;

        if !is_final_chunk {
            // Warm up MTP KV cache for this intermediate chunk using known next tokens prompt[start+1..end+1]
            if let (Some(mtp_cfg), Some(spec_h)) = (&self.cfg.mtp, &out.spec_hidden) {
                let shifted = &seq.prompt_tokens[start + 1..end + 1];
                if let Err(e) = mtp_warmup_chunk(
                    &self.model,
                    &self.cache,
                    &self.bm,
                    &seq.block_table,
                    spec_h,
                    shifted,
                    start,
                    false,
                    &mut seq.sampler,
                    mtp_cfg,
                    &dev,
                ) {
                    self.bm.free_all(&mut seq.block_table);
                    self.cache.free_slot(seq.cache_slot);
                    seq.fail(format!("MTP chunk warmup failed: {e}"), &self.metrics);
                    return;
                }
            }
            self.prefilling.insert(0, seq);
            return;
        }

        // Final chunk: sample the first generated token!
        let first_tok = match seq.sampler.sample_row(&out.logits) {
            Ok(t) => t,
            Err(e) => {
                self.bm.free_all(&mut seq.block_table);
                self.cache.free_slot(seq.cache_slot);
                seq.fail(format!("sampling failed: {e}"), &self.metrics);
                return;
            }
        };

        let finish = match seq.emit_token(&self.tokenizer, first_tok) {
            Ok(f) => f,
            Err(e) => {
                self.bm.free_all(&mut seq.block_table);
                self.cache.free_slot(seq.cache_slot);
                seq.fail(format!("decode token failed: {e}"), &self.metrics);
                return;
            }
        };
        if let Some(reason) = finish {
            self.bm.free_all(&mut seq.block_table);
            self.cache.free_slot(seq.cache_slot);
            seq.finish(reason, &self.metrics);
            return;
        }

        // If MTP is enabled, warm up final chunk & propose initial k drafts!
        if let (Some(mtp_cfg), Some(spec_h)) = (self.cfg.mtp.clone(), &out.spec_hidden) {
            let mut shifted = Vec::with_capacity(chunk_len);
            if chunk_len > 1 {
                shifted.extend_from_slice(&seq.prompt_tokens[start + 1..end]);
            }
            shifted.push(first_tok);
            let warmup = mtp_warmup_chunk(
                &self.model,
                &self.cache,
                &self.bm,
                &seq.block_table,
                spec_h,
                &shifted,
                start,
                true,
                &mut seq.sampler,
                &mtp_cfg,
                &dev,
            );
            match warmup {
                Ok(Some((d0, chain_h))) => {
                    let remaining_budget = seq
                        .sampler
                        .params()
                        .max_tokens
                        .saturating_sub(seq.generated_tokens.len());
                    let k = mtp_cfg.n_predict.min(remaining_budget.saturating_sub(1));
                    if k == 0 {
                        seq.mtp.pending_drafts.clear();
                    } else {
                        match mtp_chain_drafts(
                            &self.model,
                            &self.cache,
                            &mut self.bm,
                            &mut seq.block_table,
                            d0,
                            chain_h,
                            end - 1,
                            k,
                            &mut seq.sampler,
                            &mtp_cfg,
                            &dev,
                        ) {
                            Ok(drafts) => seq.mtp.pending_drafts = drafts,
                            Err(e) => {
                                self.bm.free_all(&mut seq.block_table);
                                self.cache.free_slot(seq.cache_slot);
                                seq.fail(format!("MTP draft chain failed: {e}"), &self.metrics);
                                return;
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    self.bm.free_all(&mut seq.block_table);
                    self.cache.free_slot(seq.cache_slot);
                    seq.fail(format!("MTP final warmup failed: {e}"), &self.metrics);
                    return;
                }
            }
        }

        self.decoding.push(seq);
    }

    /// Standard batched single-token decode step across all active decoding sequences.
    fn step_decode_standard(&mut self) {
        if self.decoding.is_empty() {
            return;
        }
        let dev = self.model.device().clone();

        // Ensure every active sequence has KV capacity for +1 token (finish with Length if KV full).
        let mut ready_indices = Vec::with_capacity(self.decoding.len());
        let mut oom_indices = Vec::new();
        for (idx, seq) in self.decoding.iter_mut().enumerate() {
            let next_len = seq.kv_len + 1;
            if next_len > self.model.config().max_position_embeddings
                || !self.bm.ensure_capacity(&mut seq.block_table, next_len)
            {
                oom_indices.push(idx);
            } else {
                ready_indices.push(idx);
            }
        }
        for idx in oom_indices.into_iter().rev() {
            let mut seq = self.decoding.remove(idx);
            self.bm.free_all(&mut seq.block_table);
            self.cache.free_slot(seq.cache_slot);
            seq.finish(FinishReason::Length, &self.metrics);
        }
        if self.decoding.is_empty() {
            return;
        }

        let b = self.decoding.len();
        let tokens_host: Vec<u32> = self.decoding.iter().map(|s| s.last_token).collect();
        let pos_vec: Vec<u32> = self.decoding.iter().map(|s| s.kv_len as u32).collect();
        let slots: Vec<usize> = self.decoding.iter().map(|s| s.cache_slot).collect();
        let spans: Vec<(usize, usize)> = (0..b).map(|i| (i, 1)).collect();
        let kv_lens: Vec<usize> = self.decoding.iter().map(|s| s.kv_len + 1).collect();
        let query_lens = vec![1usize; b];
        let bt_refs: Vec<&[u32]> = self
            .decoding
            .iter()
            .map(|s| s.block_table.as_slice())
            .collect();

        let fwd = (|| -> candle_core::Result<_> {
            let input_ids = Tensor::from_slice(&tokens_host, (b, 1), &dev)?;
            let position_ids = Tensor::from_slice(&pos_vec, (b, 1), &dev)?;
            let meta = self
                .bm
                .build_meta(&bt_refs, &kv_lens, &query_lens, true, &dev)?;
            self.model.forward(
                &ForwardInput {
                    tokens_host: &tokens_host,
                    input_ids: &input_ids,
                    position_ids: &position_ids,
                    cache_slots: &slots,
                    seq_spans: &spans,
                    kv_lens: &kv_lens,
                    paged_meta: &meta,
                    kind: ForwardKind::Decode,
                    logit_indices: None,
                    capture_spec_hidden: false,
                    capture_verify_stash: false,
                },
                &mut self.cache,
            )
        })();

        let out = match fwd {
            Ok(o) => o,
            Err(e) => {
                let msg = format!("decode forward failed: {e}");
                for mut seq in self.decoding.drain(..) {
                    self.bm.free_all(&mut seq.block_table);
                    self.cache.free_slot(seq.cache_slot);
                    seq.fail(msg.clone(), &self.metrics);
                }
                return;
            }
        };

        // Fast path: if all sequences are greedy, run ONE GPU argmax(1) across the whole batch
        let all_greedy = self.decoding.iter().all(|s| s.sampler.is_greedy());
        let next_tokens: Vec<u32> = if all_greedy {
            match Sampler::sample_batch_greedy(&out.logits) {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("batch greedy sample failed: {e}");
                    for mut seq in self.decoding.drain(..) {
                        self.bm.free_all(&mut seq.block_table);
                        self.cache.free_slot(seq.cache_slot);
                        seq.fail(msg.clone(), &self.metrics);
                    }
                    return;
                }
            }
        } else {
            let mut v = Vec::with_capacity(b);
            for (i, seq) in self.decoding.iter_mut().enumerate() {
                let row = out.logits.narrow(0, i, 1);
                match row.and_then(|r| seq.sampler.sample_row(&r)) {
                    Ok(t) => v.push(t),
                    Err(_) => v.push(self.tokenizer.eos_token_id()),
                }
            }
            v
        };

        let mut finished = Vec::new();
        for (i, &tok) in next_tokens.iter().enumerate() {
            let seq = &mut self.decoding[i];
            seq.kv_len += 1;
            match seq.emit_token(&self.tokenizer, tok) {
                Ok(Some(reason)) => finished.push((i, Ok(reason))),
                Ok(None) => {}
                Err(e) => finished.push((i, Err(e.to_string()))),
            }
        }
        for (idx, res) in finished.into_iter().rev() {
            let mut seq = self.decoding.remove(idx);
            self.bm.free_all(&mut seq.block_table);
            self.cache.free_slot(seq.cache_slot);
            match res {
                Ok(reason) => seq.finish(reason, &self.metrics),
                Err(msg) => seq.fail(msg, &self.metrics),
            }
        }
    }

    /// Built-in MTP speculative verify + refresh + draft step.
    fn step_decode_mtp(&mut self) {
        if self.decoding.is_empty() {
            return;
        }
        let mtp_cfg = self.cfg.mtp.clone().expect("MTP config");
        let dev = self.model.device().clone();

        // Step one sequence per MTP verify call (or fall back to standard decode if no pending drafts).
        // When max_seqs > 1, round-robin through active decoding sequences.
        let mut seq = self.decoding.remove(0);

        if seq.mtp.pending_drafts.is_empty() {
            // Final single-token step (e.g., remaining_budget == 1)
            self.decoding.insert(0, seq);
            self.step_decode_standard();
            return;
        }

        let drafts = std::mem::take(&mut seq.mtp.pending_drafts);
        let k = drafts.len();
        let verify_len = k + 1;
        let prev_kv_len = seq.kv_len;
        let verify_end_kv = prev_kv_len + verify_len;

        if verify_end_kv > self.model.config().max_position_embeddings
            || !self.bm.ensure_capacity(&mut seq.block_table, verify_end_kv)
        {
            self.bm.free_all(&mut seq.block_table);
            self.cache.free_slot(seq.cache_slot);
            seq.finish(FinishReason::Length, &self.metrics);
            return;
        }

        // 1. Target parallel verify over [last_token, d_0, ..., d_{k-1}]
        let mut verify_tokens = Vec::with_capacity(verify_len);
        verify_tokens.push(seq.last_token);
        for d in &drafts {
            verify_tokens.push(d.token_id);
        }
        let pos_vec: Vec<u32> = (prev_kv_len as u32..verify_end_kv as u32).collect();

        let fwd = (|| -> candle_core::Result<_> {
            let input_ids = Tensor::from_slice(&verify_tokens, (1, verify_len), &dev)?;
            let position_ids = Tensor::from_slice(&pos_vec, (1, verify_len), &dev)?;
            let meta = self.bm.build_meta(
                &[seq.block_table.as_slice()],
                &[verify_end_kv],
                &[verify_len],
                false,
                &dev,
            )?;
            self.model.forward(
                &ForwardInput {
                    tokens_host: &verify_tokens,
                    input_ids: &input_ids,
                    position_ids: &position_ids,
                    cache_slots: &[seq.cache_slot],
                    seq_spans: &[(0, verify_len)],
                    kv_lens: &[verify_end_kv],
                    paged_meta: &meta,
                    kind: ForwardKind::Verify,
                    logit_indices: None,
                    capture_spec_hidden: true,
                    capture_verify_stash: true,
                },
                &mut self.cache,
            )
        })();

        let out = match fwd {
            Ok(o) => o,
            Err(e) => {
                self.bm.free_all(&mut seq.block_table);
                self.cache.free_slot(seq.cache_slot);
                seq.fail(format!("MTP verify forward failed: {e}"), &self.metrics);
                return;
            }
        };

        // 2. Verify drafts against target logits[0..k+1]
        let outcome = match verify_drafts(&out.logits, &drafts, &mut seq.sampler) {
            Ok(o) => o,
            Err(e) => {
                self.bm.free_all(&mut seq.block_table);
                self.cache.free_slot(seq.cache_slot);
                seq.fail(format!("MTP verify sampling failed: {e}"), &self.metrics);
                return;
            }
        };

        self.metrics
            .mtp
            .record(k, outcome.accepted_drafts.len());

        // 3. Commit accepted prefix or roll back rejected suffix bit-exactly
        let keep = outcome.keep_rows;
        seq.kv_len = prev_kv_len + keep;
        if keep < verify_len {
            self.bm.truncate(&mut seq.block_table, seq.kv_len);
            if let Some(stash) = &out.verify_stash {
                if let Err(e) = self
                    .model
                    .rollback_recurrent_prefixes(&mut self.cache, stash, &[(0, keep)])
                {
                    self.bm.free_all(&mut seq.block_table);
                    self.cache.free_slot(seq.cache_slot);
                    seq.fail(format!("MTP recurrent rollback failed: {e}"), &self.metrics);
                    return;
                }
            }
        }

        // 4. Emit accepted drafts + bonus token
        let mut emitted = Vec::with_capacity(keep);
        emitted.extend_from_slice(&outcome.accepted_drafts);
        emitted.push(outcome.bonus_token);

        for &tok in &emitted {
            match seq.emit_token(&self.tokenizer, tok) {
                Ok(Some(reason)) => {
                    self.bm.free_all(&mut seq.block_table);
                    self.cache.free_slot(seq.cache_slot);
                    seq.finish(reason, &self.metrics);
                    return;
                }
                Ok(None) => {}
                Err(e) => {
                    self.bm.free_all(&mut seq.block_table);
                    self.cache.free_slot(seq.cache_slot);
                    seq.fail(format!("MTP emit token failed: {e}"), &self.metrics);
                    return;
                }
            }
        }

        // 5. Refresh MTP KV cache on [prev_kv_len .. prev_kv_len + keep] using true target hidden states
        //    and chain next k drafts!
        let remaining_budget = seq
            .sampler
            .params()
            .max_tokens
            .saturating_sub(seq.generated_tokens.len());
        let next_k = mtp_cfg.n_predict.min(remaining_budget.saturating_sub(1));
        if next_k > 0 {
            if let Some(spec_h) = &out.spec_hidden {
                let res = (|| -> candle_core::Result<()> {
                    let keep_hidden = spec_h.narrow(1, 0, keep)?.contiguous()?;
                    if let Some((d0, chain_h)) = mtp_warmup_chunk(
                        &self.model,
                        &self.cache,
                        &self.bm,
                        &seq.block_table,
                        &keep_hidden,
                        &emitted,
                        prev_kv_len,
                        true,
                        &mut seq.sampler,
                        &mtp_cfg,
                        &dev,
                    )? {
                        let next_drafts = mtp_chain_drafts(
                            &self.model,
                            &self.cache,
                            &mut self.bm,
                            &mut seq.block_table,
                            d0,
                            chain_h,
                            seq.kv_len - 1,
                            next_k,
                            &mut seq.sampler,
                            &mtp_cfg,
                            &dev,
                        )?;
                        seq.mtp.pending_drafts = next_drafts;
                    }
                    Ok(())
                })();
                if let Err(e) = res {
                    self.bm.free_all(&mut seq.block_table);
                    self.cache.free_slot(seq.cache_slot);
                    seq.fail(format!("MTP refresh/draft failed: {e}"), &self.metrics);
                    return;
                }
            }
        }

        self.decoding.push(seq);
    }
}
