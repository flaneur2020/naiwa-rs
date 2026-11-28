//! Hyper-Connection low-rank gated residual mixer (`GatedResidual`) for Qwen3.8-Flash-Next.

use candle_core::{DType, Result, Tensor, D};
use naiwa_attn::grouped_rms_norm;
use naiwa_quant::QLinear;

use crate::probe;

/// Low-rank gated mixer over `hc` parallel residual streams (replaces standard RMSNorm + residual).
pub(crate) struct GatedResidual {
    /// Already-shifted `(1 + w)` F32 norm weight of shape `[hc * hidden]`.
    pub(crate) norm_weight: Tensor,
    pub(crate) down: QLinear,
    pub(crate) up: QLinear,
    pub(crate) inject: Option<QLinear>,
    pub(crate) hc: usize,
    pub(crate) hidden: usize,
    pub(crate) eps: f64,
}

impl GatedResidual {
    pub(crate) fn norm(&self, res: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if res.device().is_cuda() {
            return naiwa_attn::qwen4_exp::hc_norm(res, &self.norm_weight, self.hc, self.eps);
        }
        grouped_rms_norm(res, &self.norm_weight, self.hc, self.eps)
    }

    /// Block input: mean over streams of `xn * sigmoid(up(silu(down(xn) / hc)))`.
    pub(crate) fn mix(&self, xn: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if xn.device().is_cuda() && matches!(xn.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            if let (Some(w_down), Some(w_up)) = (self.down.dense_weight(), self.up.dense_weight()) {
                let width = xn.dim(D::Minus1)?;
                let tokens = xn.elem_count() / width.max(1);
                if tokens > 0
                    && tokens <= naiwa_attn::qwen4_exp::HC_FUSED_MAX_TOKENS
                    && w_down.dtype() == xn.dtype()
                    && w_up.dtype() == xn.dtype()
                {
                    let (lo, _) = naiwa_attn::qwen4_exp::hc_down_inject_act(xn, w_down, None, self.hc)?;
                    probe::boundary(&lo, usize::MAX, "mix_down");
                    probe::boundary(&lo, usize::MAX, "mix_act");
                    let out = naiwa_attn::qwen4_exp::hc_up_mix(xn, &lo, w_up, self.hc)?;
                    probe::boundary(&out, usize::MAX, "mix_up");
                    return Ok(out);
                }
            }
        }
        let lo = self.down.forward(xn)?;
        probe::boundary(&lo, usize::MAX, "mix_down");
        let lo = candle_nn::ops::silu(&(lo / self.hc as f64)?)?;
        probe::boundary(&lo, usize::MAX, "mix_act");
        let gate = self.up.forward(&lo)?;
        probe::boundary(&gate, usize::MAX, "mix_up");
        #[cfg(feature = "cuda")]
        if xn.device().is_cuda() {
            return naiwa_attn::qwen4_exp::hc_mix(xn, &gate, self.hc);
        }
        let mut shape = xn.dims().to_vec();
        let last = shape.len() - 1;
        shape[last] = self.hc;
        shape.push(self.hidden);
        let gated = (xn * candle_nn::ops::sigmoid(&gate)?)?.reshape(shape)?;
        gated
            .to_dtype(DType::F32)?
            .mean(D::Minus2)?
            .to_dtype(xn.dtype())
    }

    /// Fused `(mix(xn), inject(xn))` — on CUDA for `tokens <= 8`, reads `xn` once across
    /// `down + silu + inject` (1 launch instead of 4) and fuses `up + hc_mix` (1 launch instead of 2).
    pub(crate) fn mix_and_inject(&self, xn: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if xn.device().is_cuda() && matches!(xn.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            if let (Some(w_down), Some(w_up), Some(w_inj)) = (
                self.down.dense_weight(),
                self.up.dense_weight(),
                self.inject.as_ref().and_then(|l| l.dense_weight()),
            ) {
                let width = xn.dim(D::Minus1)?;
                let tokens = xn.elem_count() / width.max(1);
                if tokens > 0
                    && tokens <= naiwa_attn::qwen4_exp::HC_FUSED_MAX_TOKENS
                    && w_down.dtype() == xn.dtype()
                    && w_up.dtype() == xn.dtype()
                    && w_inj.dtype() == xn.dtype()
                {
                    let (lo, inj) = naiwa_attn::qwen4_exp::hc_down_inject_act(
                        xn,
                        w_down,
                        Some(w_inj),
                        self.hc,
                    )?;
                    probe::boundary(&lo, usize::MAX, "mix_down");
                    probe::boundary(&lo, usize::MAX, "mix_act");
                    let out = naiwa_attn::qwen4_exp::hc_up_mix(xn, &lo, w_up, self.hc)?;
                    probe::boundary(&out, usize::MAX, "mix_up");
                    return Ok((out, inj.expect("w_inj was Some")));
                }
            }
        }
        let h = self.mix(xn)?;
        let inject = self.inject(xn)?;
        Ok((h, inject))
    }

    /// Raw per-stream injection logits `[.., hc]`.
    pub(crate) fn inject(&self, xn: &Tensor) -> Result<Tensor> {
        self.inject
            .as_ref()
            .expect("final hyper-connection mixer has no injection")
            .forward(xn)
    }

    /// `res + out * 2 * sigmoid(inject / hc)` per stream, optionally fused with `next.norm()`.
    pub(crate) fn combine(
        &self,
        res: &Tensor,
        out: &Tensor,
        inject: &Tensor,
        next: Option<&GatedResidual>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let next = next.filter(|n| n.norm_weight.device().same_device(res.device()));
        #[cfg(feature = "cuda")]
        if res.device().is_cuda() {
            return naiwa_attn::qwen4_exp::hc_combine(
                res,
                out,
                inject,
                next.map(|n| &n.norm_weight),
                self.hc,
                self.eps,
            );
        }
        let dims = res.dims().to_vec();
        let mut stream_shape = dims.clone();
        let last = stream_shape.len() - 1;
        stream_shape[last] = self.hc;
        stream_shape.push(self.hidden);
        let weights =
            (candle_nn::ops::sigmoid(&(inject.to_dtype(DType::F32)? / self.hc as f64)?)? * 2.0)?;
        let injection = out
            .to_dtype(DType::F32)?
            .unsqueeze(D::Minus2)?
            .broadcast_mul(&weights.unsqueeze(D::Minus1)?)?;
        let res = (res.to_dtype(DType::F32)?.reshape(stream_shape)? + injection)?
            .reshape(dims)?
            .to_dtype(res.dtype())?;
        let xn = next.map(|n| n.norm(&res)).transpose()?;
        Ok((res, xn))
    }
}
