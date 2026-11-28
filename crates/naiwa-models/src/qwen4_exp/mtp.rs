//! Built-in Multi-Token Prediction (MTP) draft head for Qwen3.8-Flash-Next (`blk.<block>.nextn.*`).

use std::sync::Arc;

use candle_core::{Result, Tensor};
use naiwa_attn::{PagedAttnMeta, RotaryEmbedding};
use naiwa_quant::QLinear;

use crate::qwen4_exp::{
    config::QsaConfig,
    hyper::GatedResidual,
    moe::SparseMoeBlock,
    qsa::{QsaAttention, QsaMode, QsaStep},
};

/// Input to one MTP drafter step (either prompt warm-up or chained single-token draft).
pub struct MtpForwardInput<'a> {
    /// `[batch, rows, hidden]` embeddings of the shifted next tokens.
    pub input_embeds: &'a Tensor,
    /// `[batch, rows, hc * hidden]` pre-final-mixer target (or previous chained) residuals.
    pub target_hidden: &'a Tensor,
    /// `[batch, rows]` 1D text positions.
    pub position_ids: &'a Tensor,
    /// Paged attention metadata for the MTP layer's KV cache.
    pub paged_meta: &'a PagedAttnMeta,
    /// Per-sequence `(start, len)` spans on flattened token axis, and post-step `kv_lens`
    /// when running a multi-token warm-up chunk; `None` for rectangular single-token draft.
    pub chunk_spans: Option<(&'a [(usize, usize)], &'a [usize])>,
}

/// Output of one MTP drafter forward step.
pub struct MtpForwardOutput {
    /// `[batch, rows, hidden]` collapsed hidden states ready for `lm_head`.
    pub logits_hidden: Tensor,
    /// `[batch, rows, hc * hidden]` pre-mixer 4-stream residual for the next chained draft.
    pub chain_hidden: Tensor,
}

pub(crate) struct Qwen4ExpMtpHead {
    /// Already-shifted `(1 + w)` weights in model dtype.
    pub(crate) pre_fc_norm_embedding: Tensor,
    pub(crate) pre_fc_norm_hidden: Tensor,
    pub(crate) fc_embedding: QLinear,
    pub(crate) fc_hidden: QLinear,
    pub(crate) attn: QsaAttention,
    pub(crate) attn_hc: GatedResidual,
    pub(crate) mlp_hc: GatedResidual,
    pub(crate) moe: SparseMoeBlock,
    pub(crate) mixer: GatedResidual,
    pub(crate) rotary_emb: Arc<RotaryEmbedding>,
    pub(crate) qsa: QsaConfig,
    pub(crate) kv_layer_idx: usize,
    pub(crate) hc: usize,
    pub(crate) hidden: usize,
    pub(crate) eps: f64,
}

impl Qwen4ExpMtpHead {
    pub(crate) fn kv_layer_idx(&self) -> usize {
        self.kv_layer_idx
    }

    pub(crate) fn forward(
        &self,
        input: &MtpForwardInput<'_>,
        kv_cache: &(Tensor, Tensor),
    ) -> Result<MtpForwardOutput> {
        let (batch, rows, _) = input.input_embeds.dims3()?;
        let norm_emb = candle_nn::ops::rms_norm(
            &input.input_embeds.contiguous()?,
            &self.pre_fc_norm_embedding,
            self.eps as f32,
        )?;
        let embeds = self.fc_embedding.forward(&norm_emb)?;

        let norm_hid = candle_nn::ops::rms_norm(
            &input.target_hidden.contiguous()?,
            &self.pre_fc_norm_hidden,
            self.eps as f32,
        )?
        .reshape((batch, rows * self.hc, self.hidden))?;
        let hidden = self
            .fc_hidden
            .forward(&norm_hid)?
            .reshape((batch, rows, self.hc, self.hidden))?;
        let res = hidden.broadcast_add(&embeds.unsqueeze(2)?)?.reshape((
            batch,
            rows,
            self.hc * self.hidden,
        ))?;

        let cos_sin = self.rotary_emb.compute_cos_sin(input.position_ids, res.dtype())?;
        #[cfg(feature = "cuda")]
        let n_tokens = batch * rows;
        let max_kv = match input.chunk_spans {
            Some((_, kv_lens)) => kv_lens.iter().copied().max().unwrap_or(0),
            None => input.paged_meta.max_context_len,
        };
        let mode = QsaMode::for_max_kv_len(&self.qsa, max_kv);

        #[cfg(feature = "cuda")]
        let layout = if !res.device().is_cuda() {
            None
        } else if let Some((spans, kv_lens)) = input.chunk_spans {
            if spans.len() == 1 && batch == 1 && spans[0] == (0, rows) {
                Some(naiwa_attn::qwen4_exp::TokenLayout::rectangular(1, rows))
            } else {
                Some(naiwa_attn::qwen4_exp::TokenLayout::from_host(
                    spans,
                    kv_lens,
                    n_tokens,
                    res.device(),
                )?)
            }
        } else {
            Some(naiwa_attn::qwen4_exp::TokenLayout::rectangular(batch, rows))
        };

        let step = QsaStep {
            mode,
            #[cfg(feature = "cuda")]
            layout: layout.as_ref(),
            #[cfg(not(feature = "cuda"))]
            _marker: std::marker::PhantomData,
        };

        let xn = self.attn_hc.norm(&res)?;
        let (h, inject) = self.attn_hc.mix_and_inject(&xn)?;
        let out = self.attn.forward(&h, &cos_sin, kv_cache, input.paged_meta, &step)?;
        let (res, xn) = self
            .attn_hc
            .combine(&res, &out, &inject, Some(&self.mlp_hc))?;
        let xn = match xn {
            Some(xn) => xn,
            None => self.mlp_hc.norm(&res)?,
        };
        let (h, inject) = self.mlp_hc.mix_and_inject(&xn)?;
        let out = self.moe.forward(&h)?;
        let (res, xn) = self
            .mlp_hc
            .combine(&res, &out, &inject, Some(&self.mixer))?;
        let xn = match xn {
            Some(xn) => xn,
            None => self.mixer.norm(&res)?,
        };
        Ok(MtpForwardOutput {
            logits_hidden: self.mixer.mix(&xn)?,
            chain_hidden: res,
        })
    }
}
