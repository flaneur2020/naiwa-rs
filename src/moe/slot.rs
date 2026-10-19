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

pub(crate) struct LayerState {
    /// Step 5 doorbell: publishes this layer's top-k into mapped memory so the host read needs
    /// no stream sync or D2H. Acquired lazily at first decode.
    pub(crate) doorbell: Option<Arc<crate::doorbell::TopkDoorbell>>,
    /// seq the last ring of this layer advanced past (starts 0; first ring -> 1).
    pub(crate) doorbell_seq: u32,
    pub(crate) slots: Vec<Slot>,
    pub(crate) hit_counts: Vec<u64>,
    /// Decode steps seen by this layer.
    pub(crate) tick: u64,
    /// Slots `[0, usable)` are in service; the rest are lent to the prefill stream.
    pub(crate) usable: usize,
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
