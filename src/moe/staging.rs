//! `--pcie-frac`: the model-wide staging ring and the decode path's DMA of misses.
//! Module docs: `super`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::quantized::QTensor;
use candle_core::Device;
use tracing::info;

use crate::copy_stream::{CopyStream, ReadFence};

use super::{expert_range, Bank, Projections, SlotBuf, SlotData};

/// With `--pcie-frac`, a decode step DMAs its highest-weight misses into a staging ring shared
/// by every layer of the model (Strata's `kStagingBlobs` shape), instead of a per-layer staging
/// bank: per-layer banks could only hold `ceil(frac*top_k)` slots each, which at `frac` 0.55
/// meant 6 slots x 48 layers of VRAM and OOM'd, while cross-layer reuse needs one fixed pool of
/// slots that every layer's step borrows in turn. The fence model already serializes reusers —
/// a reader holds the `SlotBuf` `Arc` until it has recorded its fence, and the next layer's DMA
/// waits on that fence — so no cross-layer accounting is needed.
pub(crate) const STAGING_RING_FIRST: usize = 1 << 30;

/// Ring capacity: enough that consecutive layers' steps rarely collide on one slot, small
/// enough to be a rounding error next to the cache. Strata uses 16 (`kStagingBlobs`).
/// Fixed: 16 measured the same as 64, and a bigger ring only costs VRAM.
/// Staging-ring slots per quant shape. Measured: 16 and 64 perform identically (the ring has no id -> slot
/// lookup, so a staged expert is copied again either way), so this is a constant, not a knob.
pub(crate) const fn staging_ring_slots() -> usize {
    16
}

/// A staging ring's shape key: every projection's per-expert byte count (folds in dtype and
/// dims, and separates a layer whose gate/up/down are heterogeneously quantized), plus the
/// stacks' dtype.
type StagingKey = ([usize; 3], u8);

/// The model-wide `--pcie-frac` staging ring (see `STAGING_RING_FIRST`): one bank of slots
/// past every resident slot, shared by every layer, borrowed round-robin per decode step.
///
/// `acquire(n)` runs under the `CachedMoe` ring lock, so consecutive layers get disjoint slots even
/// though all steps are on the token thread (the lock also covers a future multi-threaded
/// caller). Round-robin keeps the reuse distance at one full revolution of the ring: by the
/// time the cursor returns to a slot, several layers' readers have long recorded their fences,
/// so `wait_for_readers` never spins in practice.
/// Ring slots' `Bank.first` is `STAGING_RING_FIRST`, past every resident slot.
pub(crate) struct StagingRing {
    slots: Vec<Arc<SlotBuf>>,
    cursor: usize,
}

#[derive(Default)]
pub(crate) struct StagingRings {
    rings: Mutex<HashMap<StagingKey, Arc<Mutex<StagingRing>>>>,
    /// How many rings have actually been created (not looked up). After warmup this must stay
    /// constant: a later increment means a ring materialized on the decode path.
    creations: AtomicUsize,
}

impl StagingRings {
    pub(crate) fn creations(&self) -> usize {
        self.creations.load(Ordering::Relaxed)
    }

    /// The staging ring for this shape of expert stacks on `device`, creating it at the first
    /// decode step that needs it. `None` only on allocation failure — a layer on another
    /// device simply never finds `cs` for it (copy streams are per-device).
    pub(crate) fn ring(
        &self,
        cs: &CopyStream,
        stacks: &(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>),
        device: &Device,
    ) -> candle_core::Result<Option<Arc<Mutex<StagingRing>>>> {
        // The per-projection expert byte counts are the ground truth a ring must match (they
        // fold in dtype and dims, including per-projection differences inside one layer).
        let bytes: [usize; 3] = std::array::from_fn(|i| {
            expert_range(&[&stacks.0, &stacks.1, &stacks.2][i], 0).map_or(0, |r| r.len())
        });
        let key: StagingKey = (bytes, stacks.0.dtype() as u8);
        let mut rings = self.rings.lock().unwrap();
        if let Some(ring) = rings.get(&key) {
            return Ok(Some(Arc::clone(ring)));
        }
        let len = staging_ring_slots();
        let slots = alloc_bank(cs, stacks, STAGING_RING_FIRST, len, device)?;
        let per_slot: usize = bytes.iter().sum();
        let ring = Arc::new(Mutex::new(StagingRing { slots, cursor: 0 }));
        info!(
            "--pcie-frac: {len} staging-ring slots for dtype {:?} shared by its layers \
             (~{} MiB total, beyond the cache), DMA'd misses run on the GPU",
            stacks.0.dtype(),
            (per_slot * len) >> 20
        );
        rings.insert(key, Arc::clone(&ring));
        self.creations.fetch_add(1, Ordering::Relaxed);
        Ok(Some(ring))
    }
}
impl StagingRing {
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
