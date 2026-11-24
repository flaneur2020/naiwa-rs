//! Per-expert weight slicing and CPU fused MoE decode helper.

use std::sync::Arc;

use candle_core::quantized::{ggml_file::qtensor_from_ggml, QTensor};
use candle_core::Device;
use naiwa_quant::{Activation, MoEExperts};

/// A decode step's routed experts on the CPU in two parallel passes
/// (`naiwa_quant::fused_moe_decode`): gate+up+act per distinct (expert, row), then down with
/// the router-weighted sum. Returns `None` when the generic fallback must run instead.
pub(crate) fn fused_decode(
    experts: &MoEExperts,
    act: Activation,
    xs: &[f32],
    ids: &[u32],
    weights: &[f32],
    topk: usize,
) -> candle_core::Result<Option<Vec<f32>>> {
    if !matches!(act, Activation::Silu | Activation::Swish) {
        return Ok(None);
    }
    let Some((gate, up, down)) = experts.expert_qtensors() else {
        return Ok(None);
    };
    naiwa_quant::fused_moe_decode(&gate, &up, &down, xs, ids, weights, topk)
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
/// (`host_weights`), that copy is an asynchronous DMA that may still be reading after this
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
