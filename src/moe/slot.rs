//! Slot storage and per-layer state for the MoE cache.

use std::sync::Arc;

use candle_core::quantized::QTensor;

use crate::copy_stream::{ReadFence, SlotMemory};

/// One cache slot: reservation/published state, permanent storage, and LRU age.
/// See the state list in `m9a.md`; `storage` is cloned out under the state lock so uploads and
/// lending never hold the state lock while waiting for readers.
pub(crate) struct Slot {
    /// Reserved (weights copying) or published expert; `None` = empty.
    pub(crate) expert: Option<usize>,
    /// Published contents. Readers clone this `Arc` under the state lock, so a later fill never
    /// changes what an in-flight token computes with.
    pub(crate) buf: Option<Arc<SlotBuf>>,
    /// Copy-stream mode: the slot's permanent buffer; `None` while lent.
    pub(crate) storage: Arc<parking_lot::Mutex<Option<Arc<SlotBuf>>>>,
    /// `tick` of the last decode hit or fill; drives LRU replacement.
    pub(crate) last_use: u64,
}

impl Slot {
    pub(crate) fn empty() -> Self {
        Self {
            expert: None,
            buf: None,
            storage: Arc::new(parking_lot::Mutex::new(None)),
            last_use: 0,
        }
    }
}

/// One expert's three projections.
#[derive(Clone)]
pub(crate) struct Projections {
    pub(crate) gate: Arc<QTensor>,
    pub(crate) up: Arc<QTensor>,
    pub(crate) down: Arc<QTensor>,
}

/// Slots `[first, first + len)` of a layer in copy-stream mode: one `[len, N, K]` device
/// allocation per projection, which the fused decode kernel indexes by `slot - first`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct Bank {
    pub(crate) first: usize,
    pub(crate) proj: Projections,
}

/// The two legal ways a slot can hold an expert.
pub(crate) enum SlotData {
    /// Non-copy-stream mode: this slot owns a fresh per-expert QTensor set.
    Owned(Projections),
    /// Copy-stream mode: this slot is a view into a shared bank, overwritten in place.
    Banked {
        bank: Arc<Bank>,
        memory: [SlotMemory; 3],
    },
}

/// One slot's contents plus the read fence readers record after launching its kernels.
pub(crate) struct SlotBuf {
    pub(crate) data: SlotData,
    pub(crate) fence: ReadFence,
}

/// Slot-bytes estimate for a single expert from a `QTensor` stack.
/// Returns `None` when the stack is not accessible (dense/float).
pub(crate) fn slot_bytes_from_qtensor(qt: &QTensor) -> Option<u64> {
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
