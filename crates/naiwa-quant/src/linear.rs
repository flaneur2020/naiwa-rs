//! Single-device linear layer supporting both GGUF quantized weights (`QTensor`) and dense `Tensor`s.

use std::sync::Arc;

use candle_core::{quantized::QTensor, Device, Result, Tensor};

use crate::gguf::qtensor_matmul;

/// A single-device linear projection `y = x @ W^T (+ b)`.
#[derive(Clone)]
pub enum QLinear {
    Quantized {
        weight: Arc<QTensor>,
        bias: Option<Tensor>,
    },
    Dense {
        weight: Tensor,
        bias: Option<Tensor>,
    },
}

impl QLinear {
    pub fn from_qtensor(weight: Arc<QTensor>, bias: Option<Tensor>) -> Self {
        Self::Quantized { weight, bias }
    }

    pub fn from_dense(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self::Dense { weight, bias }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Quantized { weight, .. } => weight.device(),
            Self::Dense { weight, .. } => weight.device().clone(),
        }
    }

    pub fn qtensor(&self) -> Option<&Arc<QTensor>> {
        match self {
            Self::Quantized { weight, .. } => Some(weight),
            Self::Dense { .. } => None,
        }
    }

    /// Forward pass: `xs` has shape `[.., in_features]`, output has shape `[.., out_features]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out = match self {
            Self::Quantized { weight, .. } => qtensor_matmul(weight, xs)?,
            Self::Dense { weight, .. } => {
                let w = if weight.dtype() != xs.dtype() {
                    weight.to_dtype(xs.dtype())?
                } else {
                    weight.clone()
                };
                xs.broadcast_matmul(&w.t()?)?
            }
        };
        if let Some(bias) = match self {
            Self::Quantized { bias, .. } | Self::Dense { bias, .. } => bias.as_ref(),
        } {
            let b = if bias.dtype() != out.dtype() {
                bias.to_dtype(out.dtype())?
            } else {
                bias.clone()
            };
            out = out.broadcast_add(&b)?;
        }
        Ok(out)
    }
}
