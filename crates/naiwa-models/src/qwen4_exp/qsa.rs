//! Block-sparse QSA (Query-Selected Attention) layer for Qwen3.8-Flash-Next.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Result, Tensor, D};
use naiwa_attn::{PagedAttention, PagedAttnMeta, RotaryEmbedding};
use naiwa_quant::QLinear;

use crate::qwen4_exp::config::QsaConfig;

/// Per-token aux cache row width: `[indexer_key | cos | sin]`.
pub(crate) fn aux_dim(qsa: &QsaConfig, half_rot: usize) -> usize {
    qsa.head_dim + 2 * half_rot
}

/// Execution mode for QSA layers decided once per step on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QsaMode {
    /// Every query sees <= budget + ratio - 1 tokens; dense attention keeps them all.
    Dense,
    /// Block-sparse attention; `max_blocks` bounds complete blocks for any query.
    Sparse { max_blocks: usize },
}

impl QsaMode {
    pub(crate) fn for_max_kv_len(qsa: &QsaConfig, max_kv_len: usize) -> Self {
        if max_kv_len <= qsa.max_selected_tokens() {
            Self::Dense
        } else {
            Self::Sparse {
                max_blocks: max_kv_len / qsa.compress_ratio,
            }
        }
    }
}

/// Per-step QSA state shared across all full-attention layers.
pub(crate) struct QsaStep<'a> {
    pub mode: QsaMode,
    #[cfg(feature = "cuda")]
    pub layout: Option<&'a naiwa_attn::qwen4_exp::TokenLayout>,
}

#[derive(Clone)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct QsaAuxCache {
    raw: Tensor,
    block_keys: Tensor,
}

pub(crate) struct QsaAttention {
    pub(crate) q_proj: QLinear,
    pub(crate) k_proj: QLinear,
    pub(crate) v_proj: QLinear,
    pub(crate) o_proj: QLinear,
    /// Already-shifted `(1 + w)` weights in model dtype `[head_dim]`.
    pub(crate) q_norm_weight: Tensor,
    pub(crate) k_norm_weight: Tensor,
    /// Separate GGUF indexer projections (`indexer.q_proj` and `indexer.k_proj`), avoiding concat overhead.
    pub(crate) index_q_proj: QLinear,
    pub(crate) index_k_proj: QLinear,
    /// Already-shifted `(1 + w)` weight in model dtype `[qsa.head_dim]`.
    pub(crate) index_q_norm_weight: Tensor,
    /// Already-shifted `(1 + w)` weight in F32 `[qsa.head_dim]` for fused `qsa_finalize`.
    pub(crate) index_k_norm_f32: Tensor,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) half_rot: usize,
    pub(crate) qsa: QsaConfig,
    pub(crate) rotary_emb: Arc<RotaryEmbedding>,
    pub(crate) paged_attn: PagedAttention,
    pub(crate) eps: f64,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    aux_cache: Mutex<Option<QsaAuxCache>>,
}

impl QsaAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        q_proj: QLinear,
        k_proj: QLinear,
        v_proj: QLinear,
        o_proj: QLinear,
        q_norm_weight: Tensor,
        k_norm_weight: Tensor,
        index_q_proj: QLinear,
        index_k_proj: QLinear,
        index_q_norm_weight: Tensor,
        index_k_norm_f32: Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        half_rot: usize,
        qsa: QsaConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        eps: f64,
    ) -> Self {
        let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm_weight,
            k_norm_weight,
            index_q_proj,
            index_k_proj,
            index_q_norm_weight,
            index_k_norm_f32,
            num_heads,
            num_kv_heads,
            head_dim,
            half_rot,
            qsa,
            rotary_emb,
            paged_attn: PagedAttention::new(num_heads, num_kv_heads, head_dim, softmax_scale),
            eps,
            aux_cache: Mutex::new(None),
        }
    }

    #[cfg(feature = "cuda")]
    fn aux_cache(
        &self,
        key_cache: &Tensor,
        block_size: usize,
        dtype: DType,
    ) -> Result<QsaAuxCache> {
        if !block_size.is_multiple_of(self.qsa.compress_ratio) {
            candle_core::bail!(
                "QSA paged block_size {block_size} must be divisible by compress_ratio {}",
                self.qsa.compress_ratio
            );
        }
        let mut guard = self.aux_cache.lock().expect("QSA aux cache poisoned");
        let num_blocks = key_cache.dim(0)?;
        if let Some(existing) = guard.as_ref() {
            if existing.raw.dim(0)? == num_blocks * block_size
                && existing.raw.device().same_device(key_cache.device())
            {
                return Ok(existing.clone());
            }
        }
        let cache = QsaAuxCache {
            raw: Tensor::zeros(
                (num_blocks * block_size, aux_dim(&self.qsa, self.half_rot)),
                dtype,
                key_cache.device(),
            )?,
            block_keys: Tensor::zeros(
                (
                    num_blocks * block_size / self.qsa.compress_ratio,
                    self.qsa.head_dim,
                ),
                dtype,
                key_cache.device(),
            )?,
        };
        *guard = Some(cache.clone());
        Ok(cache)
    }

    /// Forward pass for `[b_sz, seq_len, hidden]` using paged KV cache `(key_cache, value_cache)`.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &(Tensor, Tensor),
        meta: &PagedAttnMeta,
        step: &QsaStep<'_>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let n_tokens = b_sz * seq_len;

        let q_gate = self.q_proj.forward(x)?.reshape((
            b_sz,
            seq_len,
            self.num_heads,
            self.head_dim * 2,
        ))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        let k = self
            .k_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;

        // Q/K per-head RMSNorm + partial NeoX RoPE (in BHSD layout)
        let (q, k) = self.rotary_emb.forward_qk_norm(
            cos_sin,
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &self.q_norm_weight,
            &self.k_norm_weight,
            self.eps,
            self.eps,
        )?;
        // Convert Q and K to `[n_tokens, heads, head_dim]`
        let q_tok = q
            .transpose(1, 2)?
            .contiguous()?
            .reshape((n_tokens, self.num_heads, self.head_dim))?;
        let k_tok = k
            .transpose(1, 2)?
            .contiguous()?
            .reshape((n_tokens, self.num_kv_heads, self.head_dim))?;
        let v_tok = v
            .contiguous()?
            .reshape((n_tokens, self.num_kv_heads, self.head_dim))?;

        // Indexer query & raw key
        let index_q = self.index_q_proj.forward(x)?.reshape((
            b_sz,
            seq_len,
            self.qsa.n_heads,
            self.qsa.head_dim,
        ))?;
        let index_q = candle_nn::ops::rms_norm(
            &index_q.contiguous()?,
            &self.index_q_norm_weight,
            self.eps as f32,
        )?;
        let index_q = self.rotary_emb.apply_bshd(&index_q, &cos_sin.0, &cos_sin.1)?;
        let raw_key = self.index_k_proj.forward(x)?; // [b, s, qsa.head_dim]

        let (key_cache, value_cache) = kv_cache;
        let y_tok = self.forward_paged(
            &q_tok,
            &k_tok,
            &v_tok,
            &index_q,
            &raw_key,
            cos_sin,
            key_cache,
            value_cache,
            meta,
            step,
            n_tokens,
        )?;

        let y = y_tok.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        self.o_proj.forward(&y.broadcast_mul(&gate)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_paged(
        &self,
        q_tok: &Tensor,
        k_tok: &Tensor,
        v_tok: &Tensor,
        index_q: &Tensor,
        raw_key: &Tensor,
        cos_sin: &(Tensor, Tensor),
        key_cache: &Tensor,
        value_cache: &Tensor,
        meta: &PagedAttnMeta,
        step: &QsaStep<'_>,
        n_tokens: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if q_tok.device().is_cuda() {
            use naiwa_attn::qwen4_exp as kernels;

            let layout = step
                .layout
                .ok_or_else(|| candle_core::Error::msg("QSA requires a CUDA TokenLayout"))?;
            let block_size = value_cache.dim(3)?;
            let kv_lens = match &layout.kv_lens {
                Some(l) => l.clone(),
                None => meta.context_lens.clone(),
            };
            let paged = kernels::PagedView {
                block_tables: &meta.block_tables,
                kv_lens: &kv_lens,
                block_size,
            };
            let shape = kernels::QsaShape {
                ratio: self.qsa.compress_ratio,
                topk: self.qsa.block_topk(),
                half_rot: self.half_rot,
                n_index_heads: self.qsa.n_heads,
            };

            let aux = self.aux_cache(key_cache, block_size, q_tok.dtype())?;
            kernels::qsa_aux_write(
                &raw_key.reshape((n_tokens, self.qsa.head_dim))?,
                &cos_sin.0.reshape((n_tokens, self.half_rot))?,
                &cos_sin.1.reshape((n_tokens, self.half_rot))?,
                &meta.slot_mapping,
                &aux.raw,
            )?;
            kernels::qsa_finalize(kernels::QsaFinalizeArgs {
                aux: &aux.raw,
                block_keys: &aux.block_keys,
                layout,
                paged: &paged,
                norm_weight: &self.index_k_norm_f32,
                shape: &shape,
                eps: self.eps,
            })?;

            match step.mode {
                QsaMode::Dense => {
                    return self.paged_attn.forward(
                        q_tok,
                        k_tok,
                        v_tok,
                        key_cache,
                        value_cache,
                        meta,
                    );
                }
                QsaMode::Sparse { max_blocks } => {
                    self.paged_attn.write_cache(
                        k_tok,
                        v_tok,
                        key_cache,
                        value_cache,
                        &meta.slot_mapping,
                    )?;
                    let index_q_flat =
                        index_q.reshape((n_tokens, self.qsa.n_heads, self.qsa.head_dim))?;
                    let (selected, n_selected) = kernels::qsa_select(
                        &index_q_flat,
                        &aux.block_keys,
                        layout,
                        &paged,
                        &shape,
                        max_blocks,
                    )?;
                    return kernels::qsa_attention(kernels::QsaAttentionArgs {
                        q: q_tok,
                        key_cache,
                        value_cache,
                        layout,
                        paged: &paged,
                        selected: &selected,
                        n_selected: &n_selected,
                        shape: &shape,
                        n_kv_heads: self.num_kv_heads,
                        scale: self.paged_attn.softmax_scale,
                    });
                }
            }
        }

        // Non-CUDA fallback: dense paged attention
        let _ = (index_q, raw_key, cos_sin, step, n_tokens);
        self.paged_attn
            .forward(q_tok, k_tok, v_tok, key_cache, value_cache, meta)
    }
}
