//! `--pcie-frac`: the model-wide staging ring and the decode path's DMA of misses.
//! Module docs: `super`.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use candle_core::quantized::QTensor;
use candle_core::Device;
use tracing::info;

use crate::copy_stream::{CopyStream, ReadFence};

use super::{
    expert_range, scratch_ring_slots, Bank, LayerRegistry,
    Projections, ScratchKey, ScratchRing, SlotBuf, SlotData, SCRATCH_RING_FIRST,
};

impl LayerRegistry {
    /// The staging ring for this shape of expert stacks on `device`, creating it at the first
    /// decode step that needs it. `None` only on allocation failure — a layer on another
    /// device simply never finds `cs` for it (copy streams are per-device).
    pub(crate) fn scratch_ring(
        &self,
        cs: &CopyStream,
        stacks: &(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>),
        device: &Device,
    ) -> candle_core::Result<Option<Arc<Mutex<ScratchRing>>>> {
        // The per-projection expert byte counts are the ground truth a ring must match (they
        // fold in dtype and dims, including per-projection differences inside one layer).
        let bytes: [usize; 3] = std::array::from_fn(|i| {
            expert_range(&[&stacks.0, &stacks.1, &stacks.2][i], 0)
                .map_or(0, |r| r.len())
        });
        let key: ScratchKey = (bytes, stacks.0.dtype() as u8);
        let mut rings = self.scratch.lock().unwrap();
        if let Some(ring) = rings.get(&key) {
            return Ok(Some(Arc::clone(ring)));
        }
        let len = scratch_ring_slots();
        let slots = alloc_bank(cs, stacks, SCRATCH_RING_FIRST, len, device)?;
        let per_slot: usize = bytes.iter().sum();
        let ring = Arc::new(Mutex::new(ScratchRing { slots, cursor: 0 }));
        info!(
            "--pcie-frac: {len} staging-ring slots for dtype {:?} shared by its layers \
             (~{} MiB total, beyond the cache), DMA'd misses run on the GPU",
            stacks.0.dtype(),
            (per_slot * len) >> 20
        );
        rings.insert(key, Arc::clone(&ring));
        self.ring_creations.fetch_add(1, Ordering::Relaxed);
        Ok(Some(ring))
    }
}
impl ScratchRing {
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Advance the cursor by `n` and return where this step's slots start. `n <= len` always
    /// (the callers clamp to the ring size).
    pub(crate) fn acquire(&mut self, n: usize) -> usize {
        let len = self.slots.len();
        debug_assert!(n <= len);
        let start = self.cursor;
        self.cursor = (self.cursor + n) % len;
        start
    }

    /// The `n` slots starting at `from`, wrapping past the end, in that order.
    pub(crate) fn take<'a>(&'a self, from: usize, n: usize) -> Vec<&'a Arc<SlotBuf>> {
        let len = self.slots.len();
        (0..n).map(|i| &self.slots[(from + i) % len]).collect()
    }
}

/// Allocate the device memory of slots `[first, first + len)` as one bank on the copy stream's
/// device. The caller orders the copy stream after the allocation (`after_device_work`).
pub(super) fn alloc_bank(
    cs: &CopyStream,
    stacks: &(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>),
    first: usize,
    len: usize,
    device: &Device,
) -> candle_core::Result<Vec<Arc<SlotBuf>>> {
    let (gate, gate_views) = cs.alloc_bank(&stacks.0, len)?;
    let (up, up_views) = cs.alloc_bank(&stacks.1, len)?;
    let (down, down_views) = cs.alloc_bank(&stacks.2, len)?;
    let bank = Arc::new(Bank {
        first,
        proj: Projections { gate, up, down },
    });
    gate_views
        .into_iter()
        .zip(up_views)
        .zip(down_views)
        .map(|((g, u), d)| {
            Ok(Arc::new(SlotBuf {
                data: SlotData::Banked {
                    bank: Arc::clone(&bank),
                    memory: [g, u, d],
                },
                fence: ReadFence::new(device)?,
            }))
        })
        .collect()
}
