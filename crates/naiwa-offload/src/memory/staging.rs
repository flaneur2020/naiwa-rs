//! `--pcie-frac`: the model-wide staging ring and the decode path's DMA of misses.
//! Module docs: `super`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::quantized::QTensor;
use candle_core::Device;
use tracing::{debug, info};

use crate::transport::copy_stream::{CopyStream, ReadFence};

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
pub(crate) type StagingKey = ([usize; 3], u8);

/// Compute the shape key for a layer's three expert stacks.
pub(crate) fn staging_key(stacks: &(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>)) -> StagingKey {
    let bytes: [usize; 3] = std::array::from_fn(|i| {
        expert_range(&[&stacks.0, &stacks.1, &stacks.2][i], 0).map_or(0, |r| r.len())
    });
    (bytes, stacks.0.dtype() as u8)
}

/// One shared VRAM allocation aliased by every quant shape's staging ring.
///
/// Heterogeneously quantized models (e.g. Qwen3.8-Flash-Next IQ1_M with 7 distinct quant shapes)
/// previously allocated 7 separate 16-slot rings (~215 MiB). Because only one layer stages misses
/// at a time, all shapes can alias a single 16-slot pool (~40 MiB) at 256-byte aligned offsets,
/// with cross-shape fence waits ensuring prior readers finish before another shape overwrites.
struct SharedStagingPool {
    slab: super::slab::Slab,
    offsets: [usize; 3],
}

/// The model-wide `--pcie-frac` staging ring (see `STAGING_RING_FIRST`): one bank of slots
/// past every resident slot, shared by every layer, borrowed round-robin per decode step.
///
/// `acquire(n)` runs under the provider's staging-ring lock, so consecutive layers get disjoint slots even
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
    pool: Mutex<Option<SharedStagingPool>>,
    /// `(key, start, count)` of the last staged batch so a shape switch waits on its readers.
    last_used: Mutex<Option<(StagingKey, usize, usize)>>,
    /// How many rings have actually been created (not looked up). After warmup this must stay
    /// constant: a later increment means a ring materialized on the decode path.
    creations: AtomicUsize,
}

impl StagingRings {
    pub(crate) fn creations(&self) -> usize {
        self.creations.load(Ordering::Relaxed)
    }

    /// Pre-allocate one shared VRAM backing pool large enough for `staging_ring_slots()` slots of
    /// the largest `[gate, up, down]` expert projections across all layers.
    pub(crate) fn warmup_pool(
        &self,
        max_proj_bytes: [usize; 3],
        device: &Device,
    ) -> candle_core::Result<()> {
        let mut pool_guard = self.pool.lock().unwrap();
        if pool_guard.is_some() || max_proj_bytes.iter().all(|&b| b == 0) {
            return Ok(());
        }
        let len = staging_ring_slots();
        // Add generous padding per projection for `QCudaStorage::borrow_from` row padding
        // (`MATRIX_ROW_PADDING`) and keep every offset 4096-byte aligned (satisfies 256B IQ loads).
        let align4k = |n: usize| (n + 4095) & !4095;
        let p0 = align4k(len * max_proj_bytes[0] + 65_536);
        let p1 = align4k(len * max_proj_bytes[1] + 65_536);
        let p2 = align4k(len * max_proj_bytes[2] + 65_536);
        let total = p0 + p1 + p2;
        let slab = super::slab::Slab::new(device, total)?;
        info!(
            "--pcie-frac: unified staging pool {:.1} MiB ({len} slots shared across all quant shapes)",
            total as f64 / (1024.0 * 1024.0)
        );
        *pool_guard = Some(SharedStagingPool {
            slab,
            offsets: [0, p0, p0 + p1],
        });
        Ok(())
    }

    /// When switching to a different quant shape on the unified pool, wait for the previous
    /// shape's staged slots to be released by host readers and finished on the compute stream.
    pub(crate) fn wait_cross_shape(
        &self,
        key: StagingKey,
        cs: &CopyStream,
    ) -> candle_core::Result<()> {
        if self.pool.lock().unwrap().is_none() {
            return Ok(());
        }
        let prev = *self.last_used.lock().unwrap();
        let Some((prev_key, start, count)) = prev else {
            return Ok(());
        };
        if prev_key == key || count == 0 {
            return Ok(());
        }
        let prev_ring = self.rings.lock().unwrap().get(&prev_key).cloned();
        if let Some(prev_ring) = prev_ring {
            let guard = prev_ring.lock().unwrap();
            for slot in guard.take(start, count) {
                crate::transport::copy_stream::wait_for_readers(slot);
                cs.wait_fence(&slot.fence)?;
            }
        }
        Ok(())
    }

    /// Record the most recently staged `(key, start, count)` batch for cross-shape synchronization.
    pub(crate) fn record_used(&self, key: StagingKey, start: usize, count: usize) {
        if count > 0 {
            *self.last_used.lock().unwrap() = Some((key, start, count));
        }
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
        let key = staging_key(stacks);
        let bytes = key.0;
        let mut rings = self.rings.lock().unwrap();
        if let Some(ring) = rings.get(&key) {
            return Ok(Some(Arc::clone(ring)));
        }
        let len = staging_ring_slots();
        let slots = {
            let pool_guard = self.pool.lock().unwrap();
            if let Some(pool) = pool_guard.as_ref() {
                alloc_bank_pooled(cs, pool, stacks, STAGING_RING_FIRST, len, device)?
            } else {
                alloc_bank(cs, stacks, STAGING_RING_FIRST, len, device)?
            }
        };
        // Order the copy stream after `alloc_bank`'s `QCudaStorage::zeros` on candle's stream,
        // otherwise the first lazy ring's H2D copy can race with its zero-initialization.
        cs.after_device_work()?;
        let per_slot: usize = bytes.iter().sum();
        let pooled = self.pool.lock().unwrap().is_some();
        let ring = Arc::new(Mutex::new(StagingRing { slots, cursor: 0 }));
        if pooled {
            debug!(
                "--pcie-frac: {len} pooled staging-ring views for dtype {:?} (~{} MiB aliased)",
                stacks.0.dtype(),
                (per_slot * len) >> 20
            );
        } else {
            info!(
                "--pcie-frac: {len} staging-ring slots for dtype {:?} shared by its layers \
                 (~{} MiB total, beyond the cache), DMA'd misses run on the GPU",
                stacks.0.dtype(),
                (per_slot * len) >> 20
            );
        }
        rings.insert(key, Arc::clone(&ring));
        self.creations.fetch_add(1, Ordering::Relaxed);
        Ok(Some(ring))
    }
}

fn alloc_bank_pooled(
    cs: &CopyStream,
    pool: &SharedStagingPool,
    stacks: &(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>),
    first: usize,
    len: usize,
    device: &Device,
) -> candle_core::Result<Vec<Arc<SlotBuf>>> {
    let borrow = |stack: &QTensor, offset: usize| {
        let dims = stack.shape().dims();
        let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
        pool.slab.borrow_at(offset, len * n * k, stack.dtype(), device)
    };
    let (gate, gate_views) = cs.bank_from_storage(&stacks.0, len, borrow(&stacks.0, pool.offsets[0])?)?;
    let (up, up_views) = cs.bank_from_storage(&stacks.1, len, borrow(&stacks.1, pool.offsets[1])?)?;
    let (down, down_views) = cs.bank_from_storage(&stacks.2, len, borrow(&stacks.2, pool.offsets[2])?)?;
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
    alloc_bank_in(cs, None, stacks, first, len, device)
}

/// The same bank, carved from the cache's slab when one is given: one range per projection, so the layer's bank is
/// still contiguous and every projection still one `[slots, n, k]` tensor, but the bytes belong to the slab and
/// can be lent out later. Falls back to its own allocation when the slab cannot take the range.
pub(crate) fn alloc_bank_in(
    cs: &CopyStream,
    slab: Option<&super::slab::Slab>,
    stacks: &(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>),
    first: usize,
    len: usize,
    device: &Device,
) -> candle_core::Result<Vec<Arc<SlotBuf>>> {
    let storage = |stack: &QTensor| -> candle_core::Result<candle_core::quantized::cuda::QCudaStorage> {
        if let Some(slab) = slab {
            let dims = stack.shape().dims();
            let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
            if let Some(range) = slab.alloc(len * n * k, stack.dtype(), device)? {
                return Ok(range);
            }
            // The window is full: this bank takes VRAM of its own. Expected - the window holds two layer buffers,
            // not the whole cache - and free, because those bytes were never lendable.
            debug!(
                "expert cache: the lending window is full ({} of {} bytes); this bank allocates its own memory",
                slab.used(),
                slab.bytes()
            );
        }
        cs.bank_storage(stack, len)
    };
    let (gate, gate_views) = cs.bank_from_storage(&stacks.0, len, storage(&stacks.0)?)?;
    let (up, up_views) = cs.bank_from_storage(&stacks.1, len, storage(&stacks.1)?)?;
    let (down, down_views) = cs.bank_from_storage(&stacks.2, len, storage(&stacks.2)?)?;
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
