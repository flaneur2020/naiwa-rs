//! Experts: CPU fallback execution and byte-level slicing of stacked QTensor weights.
//!
//! `CpuExperts` owns the model's host weights; its decode-sized path calls the fused decode in
//! `stages/cpu_miss.rs`. The cache provider uses it for the MTP head, for layers with no cache
//! slots, and when the cache budget is zero. The byte helpers never dequantize: they slice one
//! expert out of the stacked `[E, N, K]` QTensor.

use std::sync::Arc;

use candle_core::quantized::{ggml_file::qtensor_from_ggml, QTensor};
use candle_core::{DType, Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::RoutedExperts;
use mistralrs_core::MoEExperts;

use super::fused_decode;
use crate::host_memory::PinnedStack;
use crate::prefill_stream::PrefillStream;

/// Calls of at most this many tokens are decode-sized (one token, or a speculative verify step)
/// and take the fused CPU path; longer ones are prefill chunks, where the generic path's routed
/// gemvs already amortize their per-call cost.
pub(crate) const FUSED_MAX_TOKENS: usize = 16;

/// Routed experts holding the model's own weights in host memory, run by mistral.rs's own
/// kernels. MoE steps move activations in and the result back; long prefill chunks run on the
/// GPU instead when a prefill stream is attached.
pub(crate) struct CpuExperts {
    /// Page-locked registrations of `experts`' stacks (prefill streaming only).
    _pinned: Vec<PinnedStack>,
    experts: Arc<MoEExperts>,
    /// The experts' activation, for the fused decode path.
    act: Activation,
    layer_device: Device,
    layer: usize,
    prefill: Option<Arc<PrefillStream>>,
}

impl CpuExperts {
    pub(crate) fn new(
        experts: MoEExperts,
        act: Activation,
        layer_device: Device,
        layer: usize,
        prefill: Option<Arc<PrefillStream>>,
        pinned: Vec<PinnedStack>,
    ) -> Self {
        Self {
            _pinned: pinned,
            experts: Arc::new(experts),
            act,
            layer_device,
            layer,
            prefill,
        }
    }
}

impl RoutedExperts for CpuExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        if let Some(ps) = &self.prefill {
            if let Some(ys) = ps.forward(self.layer, xs, &topk_weights, topk_ids)? {
                return Ok(ys);
            }
        }
        let xs_cpu = xs.to_device(&Device::Cpu)?;
        let weights_cpu = topk_weights.to_device(&Device::Cpu)?;
        let ids_cpu = topk_ids.to_device(&Device::Cpu)?;
        let (batch, seq, hidden) = xs.dims3()?;
        let tokens = batch * seq;
        if tokens > FUSED_MAX_TOKENS {
            // A prefill chunk.
            let ys = self.experts.forward(&xs_cpu, weights_cpu, &ids_cpu)?;
            return ys.to_device(&self.layer_device);
        }
        // One token, or a speculative verify step's rows.
        let (experts, act) = (Arc::clone(&self.experts), self.act);
        let run = move || -> candle_core::Result<Vec<f32>> {
            let x = xs_cpu
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let ids = ids_cpu
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let ws = weights_cpu
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if let Some(ys) = fused_decode(&experts, act, &x, &ids, &ws, ids.len() / tokens)? {
                return Ok(ys);
            }
            experts
                .forward(&xs_cpu, weights_cpu, &ids_cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        if seq == 1 && self.layer_device.is_cuda() {
            // Decode: return before the CPU experts finish; the next layer's launches overlap
            // them (`deferred`).
            let ys = crate::executor::cpu()
                .submit_result(&self.layer_device, batch, hidden, run)?
                .upload()?;
            return ys.reshape((batch, seq, hidden))?.to_dtype(xs.dtype());
        }
        Tensor::from_vec(run()?, (batch, seq, hidden), &Device::Cpu)?
            .to_dtype(xs.dtype())?
            .to_device(&self.layer_device)
    }
}

/// The byte range of expert `e` in a stacked `[E, N, K]` `QTensor`.
///
/// Row-block formats never cross expert boundaries, so `[e·stride, (e+1)·stride)` of the
/// stack's buffer is exactly expert `e`.
pub(crate) fn expert_range(qt: &QTensor, e: usize) -> candle_core::Result<std::ops::Range<usize>> {
    let dims = qt.shape().dims();
    if dims.len() < 2 {
        candle_core::bail!("expert QTensor must be at least 2-D, got {:?}", dims);
    }
    let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let block_size = qt.dtype().block_size();
    if k % block_size != 0 {
        candle_core::bail!(
            "expert K={k} is not divisible by block_size={block_size} for dtype {:?}",
            qt.dtype()
        );
    }
    let stride = n * (k / block_size) * qt.dtype().type_size();
    let range = e * stride..(e + 1) * stride;
    if range.end > qt.storage_size_in_bytes() {
        candle_core::bail!(
            "expert {e} byte range {range:?} exceeds buffer length {}",
            qt.storage_size_in_bytes()
        );
    }
    Ok(range)
}

/// Build a single-expert `[N, K]` `QTensor` on `device` from a stacked `[E, N, K]` `QTensor`.
///
/// Copies straight out of the stack, with no intermediate host buffer. When the stack is pinned
/// (`host_memory`), that copy is an asynchronous DMA that may still be reading after this
/// returns; it is safe because the stack outlives every copy (the layer owns it).
pub(crate) fn expert_qtensor_on_device(
    stack: &QTensor,
    e: usize,
    device: &Device,
) -> candle_core::Result<Arc<QTensor>> {
    let dims = stack.shape().dims();
    let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let range = expert_range(stack, e)?;
    let data = stack.data()?;
    let qt = qtensor_from_ggml(stack.dtype(), &data[range], vec![n, k], device)?;
    Ok(Arc::new(qt))
}
