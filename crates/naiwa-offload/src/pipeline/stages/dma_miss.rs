//! DMA miss: stage misses into the PCIe staging ring and record its completion event.

use std::collections::HashMap;
use std::sync::Arc;

use crate::transport::copy_stream::{self, CopyStream};
use crate::memory::{
    expert_range, staging_key, streamed_misses, CachedLayer, SlotBuf, SlotData, STAGING_RING_FIRST,
};
use crate::stats;

use super::Hit;

fn stream_pcie(
    layer: &Arc<CachedLayer>,
    misses: &mut [Vec<(u32, f32)>],
    hits: &mut [Vec<Hit>],
    before_dma: &mut dyn FnMut(&[Vec<(u32, f32)>], &[u32]) -> candle_core::Result<()>,
) -> candle_core::Result<Vec<(u32, Arc<SlotBuf>)>> {
    let (Some(cs), Some(done)) = (layer.copy_stream(), layer.staging_done()) else {
        return Ok(Vec::new());
    };
    // Every path records the layer's completion event: even when nothing was staged, so the
    // consumer's wait is satisfied either way (a wait on an unrecorded event would hang).
    let staged = stage_misses(layer, cs, misses, hits, before_dma);
    done.record(cs)?;
    staged
}

/// The DMA half of [`stream_pcie`]: pick this step's misses, queue their copies on the copy
/// stream, move them into `hits`, and return the promotion candidates on the stack.
///
/// Returning candidates by value (instead of stashing them on `CachedLayer`) guarantees every
/// staging-ring `Arc<SlotBuf>` is released before the step finishes, even on error.
fn stage_misses(
    layer: &Arc<CachedLayer>,
    cs: &Arc<CopyStream>,
    misses: &mut [Vec<(u32, f32)>],
    hits: &mut [Vec<Hit>],
    before_dma: &mut dyn FnMut(&[Vec<(u32, f32)>], &[u32]) -> candle_core::Result<()>,
) -> candle_core::Result<Vec<(u32, Arc<SlotBuf>)>> {
    let Some((gate, up, down)) = layer.cpu().expert_qtensors() else {
        return Ok(Vec::new());
    };
    let stacks = (Arc::clone(&gate), Arc::clone(&up), Arc::clone(&down));
    let key = staging_key(&stacks);
    // Lazily create the ring at the first decode step that has misses to stream (model
    // load is over by then, so the allocation is not fighting load-time fragmentation).
    let Some(ring) = layer.staging().ring(cs, &stacks, layer.slot_device())? else {
        return Ok(Vec::new());
    };
    let ring_len = crate::memory::staging_ring_slots();
    let chosen = streamed_misses(misses, layer.pcie_frac(), ring_len);
    // Selection is unchanged; the hook may submit only the complementary CPU set.
    before_dma(misses, &chosen)?;
    // NAIWA_STAGE_TRACE: the staged set is what decides whether a routed expert is computed by
    // the GPU (streamed into the ring) or by the CPU, so repeated runs can be compared step by
    // step. Limited to the first ticks, since only the earliest difference is diagnostic.
    if crate::config::diag().stage_trace {
        let tick = layer.state().lock().unwrap().tick;
        if tick <= 12 {
            let mut ids: Vec<u32> = chosen.clone();
            ids.sort_unstable();
            eprintln!(
                "STAGE tick={tick} layer={} rows={} n={} chosen={ids:?}",
                layer.layer(),
                misses.len(),
                ids.len()
            );
        }
    }
    if crate::config::diag().admission_trace {
        let mut weights: HashMap<u32, f32> = HashMap::new();
        for &(id, w) in misses.iter().flatten() {
            *weights.entry(id).or_default() += w;
        }
        let mut ranked: Vec<_> = weights.into_iter().collect();
        ranked.sort_by(|(a, wa), (b, wb)| wb.total_cmp(wa).then(a.cmp(b)));
        eprintln!(
            "ADMISSION layer={} rows={} ranked={:?} chosen={:?}",
            layer.layer(),
            misses.len(),
            ranked,
            chosen
        );
    }
    // If another quant shape previously used the unified staging pool, wait for its readers
    // BEFORE locking `ring` (zero nested ring locks, zero extra Arc strong_count).
    if !chosen.is_empty() {
        layer.staging().wait_cross_shape(key, cs)?;
    }
    // Everything below runs under the ring mutex: the whole pipeline is on the token
    // thread, so this lock only excludes the (one-time) creation race in `staging_ring`.
    // Working on `&Arc<SlotBuf>` directly matters: `wait_for_readers` waits for the refcount
    // to fall back to the ring's own `Arc` — cloning into a borrow list first would count
    // that clone and spin forever (W105: three 900s-timeout hangs, exactly here).
    let promote = layer.promoting();
    let mut promote_candidates: Vec<(u32, Arc<SlotBuf>)> = Vec::new();
    let (slot_of, used_range) = if chosen.is_empty() {
        (HashMap::new(), None)
    } else {
        let mut ring = ring.lock().unwrap();
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
            if stats::schedule::enabled() {
                let tick = layer.state().lock().unwrap().tick;
                stats::schedule::bytes(layer.layer(), tick, "staged_h2d", bytes as u64);
            }
            slot_of.insert(expert, (Arc::clone(buf), pos));
        }
        if promote {
            promote_candidates = slot_of
                .iter()
                .map(|(id, (buf, _))| (*id, Arc::clone(buf)))
                .collect();
        }
        (slot_of, Some((start, chosen.len())))
    };
    if let Some((start, count)) = used_range {
        layer.staging().record_used(key, start, count);
    }
    let bank_first = STAGING_RING_FIRST;
    if slot_of.is_empty() {
        return Ok(promote_candidates);
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
    Ok(promote_candidates)
}
pub(crate) struct DmaMiss;

impl DmaMiss {
    pub(crate) fn submit(
        layer: &Arc<CachedLayer>,
        plan: &mut super::RoutePlan,
    ) -> candle_core::Result<Vec<(u32, Arc<SlotBuf>)>> {
        stream_pcie(layer, &mut plan.misses, &mut plan.staged, &mut |_, _| Ok(()))
    }

    pub(crate) fn submit_cpu_first(
        layer: &Arc<CachedLayer>,
        plan: &mut super::RoutePlan,
        before_dma: &mut dyn FnMut(&[Vec<(u32, f32)>], &[u32]) -> candle_core::Result<()>,
    ) -> candle_core::Result<Vec<(u32, Arc<SlotBuf>)>> {
        stream_pcie(layer, &mut plan.misses, &mut plan.staged, before_dma)
    }
}
