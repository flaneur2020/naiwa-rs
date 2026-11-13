//! DMA miss: stage misses into the PCIe staging ring and record its completion event.

use std::collections::HashMap;
use std::sync::Arc;

use crate::copy_stream::{self, CopyStream};
use crate::moe::{
    expert_range, streamed_misses, CachedLayer, SlotBuf, SlotData, STAGING_RING_FIRST,
};
use crate::stats;

use super::Hit;

/// How many waves a layer's staging splits into (M26). 1 = the whole step's misses in one batch, which is
/// what every earlier milestone did; 2 is the shipping value: the second half's copies hide behind the
/// first half's kernels.
fn staging_waves() -> usize {
    // M26 measured waves = 2 against waves = 1 and found no difference (TPOT 38.1 against 38.5 ms, inside
    // the +-8% noise) while `staging_wait` grew by 1.6 ms, so the default stays at one batch. M36 §13
    // re-measured: 4 waves win on the decode-only arm (42.8 / 44.2 / 42.3 against 44.3 / 45.5 / 44.2 ms per
    // token) but lose in the chat harness (TPOT median 41.6 against 40.2 over 154 turns) - the waves
    // interleave with prefill and with reuse, so the single-batch default is the one that ships.
    std::env::var("NAIWA_STAGING_WAVES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}


/// The three projection sub-ranges of one staged expert, as (slot, host bytes) pairs.
fn staged_parts<'a>(
    buf: &'a SlotBuf,
    src: (&'a [u8], &'a [u8], &'a [u8]),
) -> Vec<(&'a crate::copy_stream::SlotMemory, &'a [u8])> {
    let SlotData::Banked { memory, .. } = &buf.data else {
        unreachable!("staging ring slots are banked");
    };
    vec![
        (&memory[0], src.0),
        (&memory[1], src.1),
        (&memory[2], src.2),
    ]
}

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
        // NAIWA_STAGE_TRACE: the staged set is what decides whether a routed expert is computed by
        // the GPU (streamed into the ring) or by the CPU, so repeated runs can be compared step by
        // step. Limited to the first ticks, since only the earliest difference is diagnostic.
        static STAGE_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *STAGE_TRACE.get_or_init(|| std::env::var_os("NAIWA_STAGE_TRACE").is_some()) {
            let tick = layer.state().lock().unwrap().tick;
            if tick <= 12 {
                let mut ids: Vec<u32> = chosen.clone();
                ids.sort_unstable();
                eprintln!("STAGE tick={tick} layer={} rows={} n={} chosen={ids:?}",
                    layer.layer(), misses.len(), ids.len());
            }
        }
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *TRACE.get_or_init(|| std::env::var_os("NAIWA_ADMISSION_TRACE").is_some()) {
            let mut weights: HashMap<u32, f32> = HashMap::new();
            for &(id, w) in misses.iter().flatten() {
                *weights.entry(id).or_default() += w;
            }
            let mut ranked: Vec<_> = weights.into_iter().collect();
            ranked.sort_by(|(a, wa), (b, wb)| wb.total_cmp(wa).then(a.cmp(b)));
            eprintln!("ADMISSION layer={} rows={} ranked={:?} chosen={:?}",
                layer.layer(), misses.len(), ranked, chosen);
        }
        if chosen.is_empty() {
            HashMap::new()
        } else {
            let len = ring.len();
            let start = ring.acquire(chosen.len());
            let taken = ring.take(start, chosen.len());
            let (g, u, d) = (gate.data()?, up.data()?, down.data()?);
            // M26: stage in waves so the compute of wave `w` runs while wave `w+1`'s copies are still
            // in flight. One event per wave boundary, recorded after that wave's copies are queued; the
            // consumer (`decode`) waits for a wave's event, submits that wave, and only then waits for
            // the next. `STAGING_WAVES` of 1 restores the single-batch behaviour exactly.
            let waves = staging_waves().min(chosen.len()).max(1);
            let per_wave = chosen.len().div_ceil(waves);
            let wave_events = if waves > 1 {
                layer.ensure_wave_events(cs, waves - 1)?
            } else {
                Vec::new()
            };
            let mut wave_of: Vec<(u32, usize)> = Vec::with_capacity(chosen.len());
            let mut slot_of: HashMap<u32, (Arc<SlotBuf>, usize)> =
                HashMap::with_capacity(chosen.len());
            static BATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let batch = *BATCH.get_or_init(|| std::env::var_os("NAIWA_STAGE_GPU_FETCH").is_some());
            for (i, staging) in taken.into_iter().enumerate() {
                let expert = chosen[i];
                let pos = (start + i) % len;
                // Same guard as `upload_in_place`: wait for the readers that captured this
                // slot last step of the ring (they record their fence before dropping it),
                // then let the copy stream wait on that fence before overwriting. This step
                // is itself the next reader through the `Arc` injected into `slot_of`.
                copy_stream::wait_for_readers(staging);
                let buf = staging;
                let (er, e) = (expert_range(&gate, expert as usize)?, expert as usize);
                let up_er = expert_range(&up, e)?;
                let down_er = expert_range(&down, e)?;
                let bytes = er.len() + up_er.len() + down_er.len();
                let wave = i / per_wave;
                wave_of.push((pos as u32, wave));
                slot_of.insert(expert, (Arc::clone(buf), pos));
                if waves > 1 && wave + 1 < waves && (i + 1) % per_wave == 0 {
                    // Waves keep the per-expert copies: their point is to start wave `w` before
                    // `w+1` is queued, which one batch for the whole layer would defeat.
                    let batch_parts = staged_parts(&buf, (&g[er], &u[up_er], &d[down_er]));
                    unsafe { cs.queue_async(&buf.fence, &batch_parts)? };
                    wave_events[wave].record(cs)?;
                } else if !batch {
                    let parts = staged_parts(&buf, (&g[er], &u[up_er], &d[down_er]));
                    // SAFETY: the stacks are page-locked, and `layer.cpu()` keeps them alive and
                    // unmodified for the model's lifetime, well past `done`.
                    unsafe { cs.queue_async(&buf.fence, &parts)? };
                }
                stats::record_round_staged_bytes(bytes as u64);
                stats::record_cache_pcie(bytes as u64);
            }
            // One launch for the whole layer's staging: the per-expert memcpys cost an API call
            // each, which is the same order as the copy itself at these sizes.
            if batch && waves == 1 {
                let mut fences: Vec<&crate::copy_stream::ReadFence> = Vec::with_capacity(chosen.len());
                let mut parts: Vec<(&crate::copy_stream::SlotMemory, &[u8])> = Vec::with_capacity(chosen.len() * 3);
                let mut owners: Vec<Arc<SlotBuf>> = Vec::with_capacity(chosen.len());
                for (expert, (buf, _)) in slot_of.iter() {
                    let SlotData::Banked { memory, .. } = &buf.data else {
                        unreachable!("staging ring slots are banked");
                    };
                    let (er, e) = (expert_range(&gate, *expert as usize)?, *expert as usize);
                    fences.push(&buf.fence);
                    parts.extend(staged_parts(buf, (&g[er], &u[expert_range(&up, e)?], &d[expert_range(&down, e)?])));
                    owners.push(Arc::clone(buf));
                }
                // SAFETY: the registered stacks outlive the completion event; `owners` and `parts`
                // keep every slot and source borrow alive until the launch is queued.
                unsafe { cs.queue_fetch_batch(&fences, &parts)? };
                drop((owners, parts, fences));
            }
            layer.set_wave_of(wave_of);
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
