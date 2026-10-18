//! `--pcie-frac`: the model-wide staging ring and the decode path's DMA of misses.
//! Module docs: `super`.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use candle_core::quantized::QTensor;
use candle_core::Device;
use tracing::info;

use crate::copy_stream::{self, CopyStream, ReadFence};
use crate::stats;

use super::decode::Hit;
use super::expert_bytes::expert_range;
use super::layer::streamed_misses;
use super::{
    schedule, scratch_ring_slots, Bank, CachedLayer, LayerRegistry, ScratchKey, ScratchRing,
    ScratchSlot, SlotBuf, SCRATCH_RING_FIRST,
};

impl LayerRegistry {
    /// The staging ring for this shape of expert stacks on `device`, creating it at the first
    /// decode step that needs it. `None` only on allocation failure — a layer on another
    /// device simply never finds `cs` for it (copy streams are per-device).
    pub(super) fn scratch_ring(
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
        let bufs = alloc_bank(cs, stacks, SCRATCH_RING_FIRST, len, device)?;
        let per_slot: usize = bytes.iter().sum();
        let mut slots = Vec::with_capacity(len);
        for buf in bufs {
            slots.push(ScratchSlot { buf });
        }
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
    /// Advance the cursor by `n` and return where this step's slots start. `n <= len` always
    /// (the callers clamp to the ring size).
    pub(super) fn acquire(&mut self, n: usize) -> usize {
        let len = self.slots.len();
        debug_assert!(n <= len);
        let start = self.cursor;
        self.cursor = (self.cursor + n) % len;
        start
    }

    /// The `n` slots starting at `from`, wrapping past the end, in that order.
    pub(super) fn take<'a>(&'a self, from: usize, n: usize) -> Vec<&'a ScratchSlot> {
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
        gate,
        up,
        down,
    });
    gate_views
        .into_iter()
        .zip(up_views)
        .zip(down_views)
        .map(|((g, u), d)| {
            Ok(Arc::new(SlotBuf {
                weights: None,
                memory: Some([g, u, d]),
                bank: Some(Arc::clone(&bank)),
                fence: ReadFence::new(device)?,
            }))
        })
        .collect()
}
impl CachedLayer {
    pub(super) fn stream_pcie(
        self: &Arc<Self>,
        misses: &mut [Vec<(u32, f32)>],
        hits: &mut [Vec<Hit>],
        pass: u32,
    ) -> candle_core::Result<()> {
        let Some(cs) = &self.copy_stream else {
            return Ok(());
        };
        let staged = self.stage_misses(cs, misses, hits);
        // M4: release this layer's DMA gate on *every* path — nothing staged, staged, or failed.
        // The gate is replayed for every layer of every pass, so a point left unpublished would
        // spin forever. This is a stream memory op on the copy stream, ordered after the copies
        // above it, so a node spinning on the word cannot see it before the data has landed.
        #[cfg(feature = "cuda")]
        if let Some(sched) = schedule() {
            let id = sched.sync_id(self.layer as u32, crate::doorbell::SYNC_STAGING);
            sched.gate(id).publish_stream(cs.cu_stream(), pass)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pass;
        staged
    }

    /// The DMA half of [`Self::stream_pcie`]: pick this step's misses, queue their copies on the
    /// copy stream, and turn them into hits.
    ///
    /// The copies record no completion event: the caller publishes this layer's `SYNC_STAGING`
    /// point on this same stream right after them, and the layer's gate graph spins on that word.
    fn stage_misses(
        &self,
        cs: &Arc<CopyStream>,
        misses: &mut [Vec<(u32, f32)>],
        hits: &mut [Vec<Hit>],
    ) -> candle_core::Result<()> {
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(());
        };
        // Lazily create the ring at the first decode step that has misses to stream (model
        // load is over by then, so the allocation is not fighting load-time fragmentation).
        let Some(ring) = self.registry.scratch_ring(
            cs,
            &(
                Arc::clone(&gate),
                Arc::clone(&up),
                Arc::clone(&down),
            ),
            &self.slot_device,
        )?
        else {
            return Ok(());
        };
        // Everything below runs under the ring mutex: the whole pipeline is on the token
        // thread, so this lock only excludes the (one-time) creation race in `scratch_ring`.
        // Working on `&ScratchSlot` directly matters: `wait_for_readers` waits for the refcount
        // to fall back to the ring's own `Arc` — cloning into a borrow list first would count
        // that clone and spin forever (W105: three 900s-timeout hangs, exactly here).
        let slot_of = {
            let mut ring = ring.lock().unwrap();
            let chosen = streamed_misses(misses, self.pcie_frac, ring.slots.len());
            if chosen.is_empty() {
                HashMap::new()
            } else {
                let len = ring.slots.len();
                let start = ring.acquire(chosen.len());
                let taken = ring.take(start, chosen.len());
                let (g, u, d) = (gate.data()?, up.data()?, down.data()?);
                let mut slot_of: HashMap<u32, (Arc<SlotBuf>, usize)> =
                    HashMap::with_capacity(chosen.len());
                for (i, scratch) in taken.into_iter().enumerate() {
                    let expert = chosen[i];
                    let pos = (start + i) % len;
                    // Same guard as `upload_in_place`: wait for the readers that captured this
                    // slot last step of the ring (they record their fence before dropping it),
                    // then let the copy stream wait on that fence before overwriting. This step
                    // is itself the next reader through the `Arc` injected into `slot_of`.
                    copy_stream::wait_for_readers(&scratch.buf);
                    let buf = &scratch.buf;
                    let memory = buf.memory.as_ref().expect("scratch slot has memory");
                    let (er, e) = (expert_range(&gate, expert as usize)?, expert as usize);
                    let bytes = er.len()
                        + expert_range(&up, e)?.len()
                        + expert_range(&down, e)?.len();
                    // SAFETY: the stacks are page-locked, and `self.cpu` keeps them alive and
                    // unmodified for the model's lifetime, well past `done`.
                    unsafe {
                        stats::record_round_staged_bytes(bytes as u64);
                        cs.queue_async(
                            &buf.fence,
                            &[
                                (&memory[0], &g[er]),
                                (&memory[1], &u[expert_range(&up, e)?]),
                                (&memory[2], &d[expert_range(&down, e)?]),
                            ],
                        )?;
                    }
                    // M4: the completion is no longer a `cudaStreamWaitEvent` on candle's stream;
                    // the copy stream publishes the layer's `SYNC_STAGING` point itself (below),
                    // and this layer's gate graph spins on it.
                    stats::record_cache_pcie(bytes as u64);
                    slot_of.insert(expert, (Arc::clone(buf), pos));
                }
                slot_of
            }
        };
        let bank_first = SCRATCH_RING_FIRST;
        if slot_of.is_empty() {
            return Ok(());
        }
        // Move every (row, weight) routed to a streamed expert from the CPU list to the hits.
        for (row_misses, row_hits) in misses.iter_mut().zip(hits.iter_mut()) {
            row_misses.retain(|(id, w)| match slot_of.get(id) {
                Some((buf, slot)) => {
                    row_hits.push((Arc::clone(buf), bank_first + slot, *w));
                    false
                }
                None => true,
            });
        }
        Ok(())
    }

}
