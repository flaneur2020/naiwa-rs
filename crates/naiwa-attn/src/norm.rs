//! RMSNorm variants (`RmsNorm`, `GemmaRmsNorm`, and `grouped_rms_norm`).

use candle_core::{DType, Module, Result, Tensor, D};

/// Standard RMSNorm: `y = x * rsqrt(mean(x^2) + eps) * weight`.
#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        rms_norm_with_weight(xs, &self.weight, self.eps)
    }
}

/// Gemma/Qwen3.8-style centered RMSNorm: `y = x * rsqrt(mean(x^2) + eps) * (1 + original_weight)`.
///
/// Note: llama.cpp GGUF checkpoints store shifted norm weights as `1 + w` already; when loading
/// from GGUF, either subtract 1 or construct via [`GemmaRmsNorm::from_shifted_weight`].
#[derive(Clone, Debug)]
pub struct GemmaRmsNorm {
    eps: f64,
    original_weight: Tensor,
    /// Precomputed `1.0 + original_weight` in the weight's dtype.
    weight: Tensor,
}

impl GemmaRmsNorm {
    /// Construct from HF-style unshifted weight `w` (effective multiplier is `1 + w`).
    pub fn from_unshifted(original_weight: Tensor, eps: f64) -> Result<Self> {
        let weight = (&original_weight + 1.0)?;
        Ok(Self {
            eps,
            original_weight,
            weight,
        })
    }

    /// Construct from GGUF-style pre-shifted weight `w_gguf = 1 + w_hf` directly (effective
    /// multiplier is `w_gguf`, avoiding unnecessary `+1 - 1` rounding error).
    pub fn from_shifted(shifted_weight: Tensor, eps: f64) -> Result<Self> {
        let original_weight = (&shifted_weight - 1.0)?;
        Ok(Self {
            eps,
            original_weight,
            weight: shifted_weight,
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn original_weight(&self) -> &Tensor {
        &self.original_weight
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        rms_norm_with_weight(xs, &self.weight, self.eps)
    }
}

/// Core RMSNorm computation in F32 (or using candle_nn's CUDA `rms_norm` when contiguous).
pub fn rms_norm_with_weight(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let orig_dtype = xs.dtype();
    if xs.device().is_cuda()
        && xs.is_contiguous()
        && weight.is_contiguous()
        && weight.dtype() == orig_dtype
        && weight.rank() == 1
        && xs.dim(D::Minus1)? == weight.dim(0)?
    {
        return candle_nn::ops::rms_norm(xs, weight, eps as f32);
    }
    let x_f32 = xs.to_dtype(DType::F32)?;
    let hidden = x_f32.dim(D::Minus1)? as f64;
    let var = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
    let normed = x_f32.broadcast_div(&(var + eps)?.sqrt()?)?;
    let w = if weight.dtype() != DType::F32 {
        weight.to_dtype(DType::F32)?
    } else {
        weight.clone()
    };
    normed.broadcast_mul(&w)?.to_dtype(orig_dtype)
}

/// Grouped RMSNorm over the last dimension split into `groups` equal chunks (used by hyper-connections).
pub fn grouped_rms_norm(xs: &Tensor, weight: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
    let orig_dims = xs.dims().to_vec();
    let d = *orig_dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("grouped_rms_norm: empty tensor".into()))?;
    if !d.is_multiple_of(groups) {
        candle_core::bail!("grouped_rms_norm: dim {d} not divisible by {groups}");
    }
    let group_dim = d / groups;
    let mut reshaped = orig_dims[..orig_dims.len() - 1].to_vec();
    reshaped.push(groups);
    reshaped.push(group_dim);

    let x_f32 = xs.reshape(reshaped)?.to_dtype(DType::F32)?;
    let var = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / (group_dim as f64))?;
    let normed = x_f32.broadcast_div(&(var + eps)?.sqrt()?)?.reshape(orig_dims)?;
    let w = if weight.dtype() != DType::F32 {
        weight.to_dtype(DType::F32)?
    } else {
        weight.clone()
    };
    normed.broadcast_mul(&w)?.to_dtype(xs.dtype())
}
