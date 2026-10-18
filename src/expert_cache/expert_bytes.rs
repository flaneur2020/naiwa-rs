//! Byte-level extraction of one expert from a stacked `[E, N, K]` QTensor
//! (never dequantize). Module docs: `super`.

use std::sync::Arc;

use candle_core::quantized::{ggml_file::qtensor_from_ggml, QTensor};
use candle_core::Device;

// ---------------------------------------------------------------------------
// Byte-level expert extraction (never dequantize)
// ---------------------------------------------------------------------------

/// The byte range of expert `e` in a stacked `[E, N, K]` `QTensor`.
///
/// Row-block formats never cross expert boundaries, so `[e·stride, (e+1)·stride)` of the
/// stack's buffer is exactly expert `e`.
pub(super) fn expert_range(qt: &QTensor, e: usize) -> candle_core::Result<std::ops::Range<usize>> {
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
pub(super) fn expert_qtensor_on_device(
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

/// Slot-bytes estimate for a single expert from a `QTensor` stack.
/// Returns `None` when the stack is not accessible (dense/float).
pub(super) fn slot_bytes_from_qtensor(qt: &QTensor) -> Option<u64> {
    let dims = qt.shape().dims();
    if dims.len() < 2 {
        return None;
    }
    let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let block_size = qt.dtype().block_size();
    if k % block_size != 0 {
        return None;
    }
    let type_size = qt.dtype().type_size();
    // gate + up + down (gate/up are [inter, hidden]; down is [hidden, inter] — same bytes)
    Some((3 * n * (k / block_size) * type_size) as u64)
}
