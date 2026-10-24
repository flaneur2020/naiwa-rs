//! DMA miss: stage misses into the PCIe staging ring and record its completion event.

use std::collections::HashMap;
use std::sync::Arc;

use crate::copy_stream::{self, CopyStream};
use crate::moe::{
    expert_range, streamed_misses, CachedLayer, SlotBuf, SlotData, STAGING_RING_FIRST,
};
use crate::stats;

use super::Hit;

fn stream_pcie(
    layer: &Arc<CachedLayer>,
    misses: &mut [Vec<(u32, f32)>],
    hits: &mut [Vec<Hit>],
) -> candle_core::Result<()> {
    let (Some(cs), Some(done)) = (layer.copy_stream(), layer.staging_done()) else {
        return Ok(());
    };
    // Promotion first: those copies then fall under the same completion event as this step's staging, and the
    // pending `Arc`s are released before the ring may re-use their slots.
    // This step's staging replaces last step's candidate list, and the clearing is **unconditional**: a clone left
    // in there keeps a ring slot's refcount above one, and `stage_misses` waits on that refcount
    // (`wait_for_readers`). Gating this on `promoting()` was the bug that hung warmup at cap 0.
    layer.set_staged_now(Vec::new());
    // Every path records the layer's completion event: `upload_async` after each expert's copies,
    // and one bare record when there was nothing to stage, so the consumer's wait is satisfied
    // either way (a wait on an event that was never recorded would hang the stream).
    let staged = stage_misses(layer, cs, misses, hits);
    done.record(cs)?;
    staged
}

/// The DMA half of [`Self::stream_pcie`]: pick this step's misses, queue their copies on the
/// copy stream, and turn them into hits.
///
/// The copies queue without a completion event: the caller records the layer's event once, after
/// the last of them, and `candle`'s stream waits on it before it reads a staged slot.
fn stage_misses(
    layer: &Arc<CachedLayer>,
    cs: &Arc<CopyStream>,
    misses: &mut [Vec<(u32, f32)>],
    hits: &mut [Vec<Hit>],
) -> candle_core::Result<()> {
    let Some((gate, up, down)) = layer.cpu().expert_qtensors() else {
        return Ok(());
    };
    // Lazily create the ring at the first decode step that has misses to stream (model
    // load is over by then, so the allocation is not fighting load-time fragmentation).
    let Some(ring) = layer.staging().ring(
        cs,
        &(Arc::clone(&gate), Arc::clone(&up), Arc::clone(&down)),
        layer.slot_device(),
    )?
    else {
        return Ok(());
    };
    // Everything below runs under the ring mutex: the whole pipeline is on the token
    // thread, so this lock only excludes the (one-time) creation race in `staging_ring`.
    // Working on `&Arc<SlotBuf>` directly matters: `wait_for_readers` waits for the refcount
    // to fall back to the ring's own `Arc` — cloning into a borrow list first would count
    // that clone and spin forever (W105: three 900s-timeout hangs, exactly here).
    let mut slot_of_pending: Vec<(u32, Arc<SlotBuf>)> = Vec::new();
    let slot_of = {
        let mut ring = ring.lock().unwrap();
        let chosen = streamed_misses(misses, layer.pcie_frac(), ring.len());
        if chosen.is_empty() {
            HashMap::new()
        } else {
            let len = ring.len();
            let start = ring.acquire(chosen.len());
            let taken = ring.take(start, chosen.len());
            let (g, u, d) = (gate.data()?, up.data()?, down.data()?);
            let mut slot_of: HashMap<u32, (Arc<SlotBuf>, usize)> =
                HashMap::with_capacity(chosen.len());
            for (i, staging) in taken.into_iter().enumerate() {
                let expert = chosen[i];
                let pos = (start + i) % len;
                // Same guard as `upload_in_place`: wait for the readers that captured this
                // slot last step of the ring (they record their fence before dropping it),
                // then let the copy stream wait on that fence before overwriting. This step
                // is itself the next reader through the `Arc` injected into `slot_of`.
                copy_stream::wait_for_readers(staging);
                let buf = staging;
                let SlotData::Banked { memory, .. } = &buf.data else {
                    unreachable!("staging ring slots are banked");
                };
                let (er, e) = (expert_range(&gate, expert as usize)?, expert as usize);
                let bytes = er.len() + expert_range(&up, e)?.len() + expert_range(&down, e)?.len();
                // SAFETY: the stacks are page-locked, and `layer.cpu()` keeps them alive and
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
                stats::record_cache_pcie(bytes as u64);
                slot_of.insert(expert, (Arc::clone(buf), pos));
            }
            let pending: Vec<(u32, Arc<SlotBuf>)> = slot_of
                .iter()
                .map(|(id, (buf, _))| (*id, Arc::clone(buf)))
                .collect();
            slot_of_pending = pending;
            slot_of
        }
    };
    // Only worth remembering when something will drain it (same step, on this thread).
    if layer.promoting() {
        layer.set_staged_now(slot_of_pending);
    }
    let bank_first = STAGING_RING_FIRST;
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
pub(crate) struct DmaMiss;

impl DmaMiss {
    pub(crate) fn submit(
        layer: &Arc<CachedLayer>,
        plan: &mut super::RoutePlan,
    ) -> candle_core::Result<()> {
        stream_pcie(layer, &mut plan.misses, &mut plan.hits)
    }
}
