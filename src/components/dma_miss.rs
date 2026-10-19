//! DMA miss: stage misses into the PCIe scratch ring and publish SYNC_STAGING.

use std::collections::HashMap;
use std::sync::Arc;

use crate::copy_stream::{self, CopyStream};
use crate::expert_cache::{
    expert_range, schedule, streamed_misses, CachedLayer, SlotBuf, SlotData, SCRATCH_RING_FIRST,
};
use crate::stats;

use super::Hit;

impl CachedLayer {
    pub(crate) fn stream_pcie(
        &self,
        misses: &mut [Vec<(u32, f32)>],
        hits: &mut [Vec<Hit>],
        pass: u32,
    ) -> candle_core::Result<()> {
        let Some(cs) = &self.copy_stream() else {
            return Ok(());
        };
        let staged = self.stage_misses(cs, misses, hits);
        // M4: release this layer's DMA gate on *every* path — nothing staged, staged, or failed.
        // The gate is replayed for every layer of every pass, so a point left unpublished would
        // spin forever. This is a stream memory op on the copy stream, ordered after the copies
        // above it, so a node spinning on the word cannot see it before the data has landed.
        #[cfg(feature = "cuda")]
        if let Some(sched) = schedule() {
            let id = sched.sync_id(self.layer() as u32, crate::doorbell::SYNC_STAGING);
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
        let Some((gate, up, down)) = self.cpu().expert_qtensors() else {
            return Ok(());
        };
        // Lazily create the ring at the first decode step that has misses to stream (model
        // load is over by then, so the allocation is not fighting load-time fragmentation).
        let Some(ring) = self.registry().scratch_ring(
            cs,
            &(
                Arc::clone(&gate),
                Arc::clone(&up),
                Arc::clone(&down),
            ),
            &self.slot_device(),
        )?
        else {
            return Ok(());
        };
        // Everything below runs under the ring mutex: the whole pipeline is on the token
        // thread, so this lock only excludes the (one-time) creation race in `scratch_ring`.
        // Working on `&Arc<SlotBuf>` directly matters: `wait_for_readers` waits for the refcount
        // to fall back to the ring's own `Arc` — cloning into a borrow list first would count
        // that clone and spin forever (W105: three 900s-timeout hangs, exactly here).
        let slot_of = {
            let mut ring = ring.lock().unwrap();
            let chosen = streamed_misses(misses, self.pcie_frac(), ring.len());
            if chosen.is_empty() {
                HashMap::new()
            } else {
                let len = ring.len();
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
                    copy_stream::wait_for_readers(scratch);
                    let buf = scratch;
                    let SlotData::Banked { memory, .. } = &buf.data else {
                        unreachable!("staging ring slots are banked");
                    };
                    let (er, e) = (expert_range(&gate, expert as usize)?, expert as usize);
                    let bytes = er.len()
                        + expert_range(&up, e)?.len()
                        + expert_range(&down, e)?.len();
                    // SAFETY: the stacks are page-locked, and `self.cpu()` keeps them alive and
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

pub(crate) fn stage(
    layer: &Arc<CachedLayer>,
    plan: &mut super::RoutePlan,
    pass: u32,
) -> candle_core::Result<()> {
    layer.stream_pcie(&mut plan.misses, &mut plan.hits, pass)
}

