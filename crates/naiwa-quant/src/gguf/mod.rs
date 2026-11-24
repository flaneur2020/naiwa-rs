//! GGUF archive loading, CPU/CUDA IQ & K-quant kernels, MMQ/MMVQ matmul, and MoE kernels.

pub mod archive;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dequant;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(feature = "cuda")]
mod ffi;
pub mod fused_cpu;
pub mod iq2s;
pub mod iq3s;
pub mod iq3xxs;
pub mod iq4nl;
pub mod iq4xs;
pub mod q2_0;

pub use archive::{GgufArchive, GgufDType, GgufTensorInfo};
#[cfg(feature = "cuda")]
pub use cuda::{
    indexed_moe_fused_decode, moe_workspace_bytes, qtensor_indexed_moe_forward,
    ACT_GELU_PYTORCH_TANH, ACT_SILU,
};
pub use fused_cpu::{fused_moe_decode, install_task_dispatcher};

use std::sync::Arc;

use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Result, Tensor,
};
use candle_nn::Module;

/// Compatibility helper namespace for chunked MMVQ calls (used by IQ/Q2_0 unit tests).
pub struct GgufMatMul;

impl GgufMatMul {
    #[cfg(feature = "cuda")]
    pub fn mmvq_chunked(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        let k = *dims.last().expect("input has at least one dimension");
        let rows = dims[..dims.len() - 1].iter().product::<usize>();
        let nrows = w.shape().dims2()?.0;
        let flat = xs.reshape((rows, k))?;
        let mut chunks = Vec::new();
        for start in (0..rows).step_by(fast_mmvq::MMVQ_MAX_BATCH) {
            let len = fast_mmvq::MMVQ_MAX_BATCH.min(rows - start);
            chunks.push(fast_mmvq::plain(w, &flat.narrow(0, start, len)?)?);
        }
        let out = Tensor::cat(&chunks, 0)?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(nrows);
        out.reshape(out_dims)
    }
}

/// Multiply `xs` (`[.., K]`) by quantized weight `weight` (`[N, K]`).
///
/// On CUDA: uses `fast_mmvq` for `rows <= 8` (including M38 q8_1 activation cache) and `fast_mmq`
/// for `rows > 8` whenever the dtype is supported, falling back to `QMatMul::from_arc` otherwise.
/// On CPU: uses candle's `QMatMul` (or dequantizes IQ/Q2_0 formats via `dequant::dequantize`).
pub fn qtensor_matmul(weight: &Arc<QTensor>, xs: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if xs.device().is_cuda() && weight.device().is_cuda() {
        let dtype = weight.dtype();
        let rank = xs.rank();
        if rank >= 2 {
            let k = xs.dim(rank - 1)?;
            let batch_rows = xs.elem_count() / k.max(1);
            if batch_rows <= fast_mmvq::MMVQ_MAX_BATCH && fast_mmvq::supports(dtype) {
                return fast_mmvq::plain(weight, xs);
            }
            if batch_rows > fast_mmvq::MMVQ_MAX_BATCH && fast_mmq::supports(dtype) {
                return fast_mmq::plain(weight, xs);
            }
        }
    }

    // Fallback for F32/F16/BF16/CPU or unsupported dtypes
    let orig_dtype = xs.dtype();
    if xs.device().is_cpu() && is_custom_iq_dtype(weight.dtype()) {
        let w = dequant_qtensor_cpu(weight)?;
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let out = xs_f32.broadcast_matmul(&w.t()?)?;
        return out.to_dtype(orig_dtype);
    }
    let qmm = QMatMul::from_arc(Arc::clone(weight))?;
    let xs_f32 = if orig_dtype != DType::F32 {
        xs.to_dtype(DType::F32)?
    } else {
        xs.clone()
    };
    let out = qmm.forward(&xs_f32)?;
    if out.dtype() != orig_dtype {
        out.to_dtype(orig_dtype)
    } else {
        Ok(out)
    }
}

/// Returns `true` for the IQ/Q2_0 quantized formats implemented in this crate rather than upstream candle CPU.
pub fn is_custom_iq_dtype(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::IQ4XS
            | GgmlDType::IQ4NL
            | GgmlDType::IQ3_XXS
            | GgmlDType::IQ2_S
            | GgmlDType::IQ3_S
            | GgmlDType::Q2_0
    )
}

/// Dequantize a custom IQ/Q2_0 CPU `QTensor` into an `F32` `Tensor`.
pub fn dequant_qtensor_cpu(qt: &QTensor) -> Result<Tensor> {
    let raw = qt.data()?;
    let n_elems = qt.shape().elem_count();
    let mut out = vec![0.0f32; n_elems];
    dequant::dequantize(&raw, qt.dtype(), &mut out)?;
    Tensor::from_vec(out, qt.shape().clone(), &candle_core::Device::Cpu)
}
