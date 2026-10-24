//! The decode path: split top-k into GPU hits and CPU misses, run both, sum.

use std::sync::Arc;

use candle_core::Tensor;
use mistralrs_core::layers::Activation;

use super::stages::{CpuMiss, DmaMiss, Gpu, Hit, Preload, Route};
use crate::stats;

use super::CachedLayer;

/// Decode path (one token, or the few rows of a speculative verify step): split each row's
/// top-k into GPU hits and CPU misses, run both concurrently, sum. Misses are never loaded on
/// the token path — they run on the CPU.
///
/// Every CUDA call stays on the calling thread; the misses run on the `deferred` worker over
/// host copies. Launching the hits from a second thread while this one copied the activation
/// to the host produced NaN logits on CUDA once the lookahead was swapping slots.
///
/// Returns without waiting for the misses: their sum is copied in on candle's stream when
/// the worker is done (`deferred`), so the next layer's launches overlap the CPU experts.
/// `--pcie-frac`: DMA the highest-weight misses of this step into the staging slots and turn
/// them into hits, so `run_hits` runs them on the GPU with the residents and the rest stay on
/// the CPU. The chosen ids are removed from `misses`, leaving the caller's CPU padding and
/// `deferred::start` untouched. The injected hits carry the staging `SlotBuf`, so the caller's
/// fence loop covers them and the next step's DMA waits on this step's readers.
///
/// `upload_async` returns as soon as the copy is queued; the layer's staging completion event is
/// waited on candle's stream, so candle's stream never blocks the host and the host enqueues one
/// event wait per layer instead of one stream wait per streamed expert.

pub(crate) fn forward(
    layer: &Arc<CachedLayer>,
    xs: &Tensor,
    topk_weights: &Tensor,
    topk_ids: &Tensor,
    act: Activation,
) -> candle_core::Result<Tensor> {
    let (batch, seq, hidden) = xs.dims3()?;
    let rows = batch * seq;
    let x2d = xs.reshape((rows, hidden))?;
    stats::record_step_layer(layer.slot_device(), layer.layer());
    Preload::submit(layer, &x2d, rows, hidden)?;

    let mut plan = Route::submit(layer, topk_ids, topk_weights, rows)?;
    let n_hits: usize = plan.hits.iter().map(Vec::len).sum();
    stats::record_cache_hits(n_hits);

    // `--pcie-frac`: DMA the highest-weight misses into the staging ring and run them on the
    // GPU with the hits, leaving the rest on the CPU. `stage_misses` only *appends* to
    // `plan.hits`, so the resident prefix is exactly what was there before the call.
    #[cfg(feature = "cuda")]
    let resident_prefix: Vec<usize> = plan.hits.iter().map(Vec::len).collect();
    if layer.staging_enabled() {
        DmaMiss::submit(layer, &mut plan)?;
    }

    let n_cpu: usize = plan.misses.iter().map(Vec::len).sum();
    stats::record_cache_misses(n_cpu);
    // Experts that went to the staging ring: counted apart from both hits and CPU misses.
    #[cfg(feature = "cuda")]
    {
        let staged_n: usize = plan.hits.iter().map(Vec::len).sum::<usize>() - n_hits;
        stats::record_cache_staged(staged_n);
        stats::record_round_plan(n_hits, staged_n, n_cpu);
    }

    let cpu = CpuMiss::submit(layer, &plan.misses, &x2d, rows)?;

    // Wait for the latest preload claim on candle's stream. The claim carries its own completion
    // event, so this also drains a copy the lookahead queued late; a layer that never gets
    // predicted has no claim and costs no CUDA call at all.
    #[cfg(feature = "cuda")]
    {
        layer.wait_preload()?;
        if layer.promoting() {
            layer.wait_promote()?;
        }
    }
    // S2: the resident experts run *before* the staging wait, so the DMA lands under their compute
    // instead of in front of it (measured 78.9 -> 62.8 ms per pass, -20%, and the p50 wait drops
    // from 858 us to 11 us). The staged experts still need the ring, so they keep the wait. This is
    // the one fixed behaviour - there is no switch.
    #[cfg(feature = "cuda")]
    let split = layer.staging_enabled();
    #[cfg(not(feature = "cuda"))]
    let split = false;

    #[cfg(feature = "cuda")]
    let resident_acc = if split {
        let residents: Vec<Vec<Hit>> = plan
            .hits
            .iter()
            .zip(&resident_prefix)
            .map(|(row, &n)| row[..n].to_vec())
            .collect();
        let acc = Gpu::submit(layer, &residents, &x2d, act);
        let mut fenced = Ok(());
        for (buf, _, _) in residents.iter().flatten() {
            if let Err(e) = buf.fence.record_reads(layer.slot_device()) {
                fenced = fenced.and(Err(e));
            }
        }
        if fenced.is_err() {
            let _ = layer.slot_device().synchronize();
        }
        fenced?;
        Some(acc?)
    } else {
        None
    };

    // ... and the staging DMA's completion event, on candle's stream.
    #[cfg(feature = "cuda")]
    layer.wait_staging()?;

    #[cfg(feature = "cuda")]
    let hits_to_run: Vec<Vec<Hit>> = if split {
        plan.hits
            .iter()
            .zip(&resident_prefix)
            .map(|(row, &n)| row[n..].to_vec())
            .collect()
    } else {
        Vec::new()
    };
    #[cfg(feature = "cuda")]
    let acc = if let Some((resident_out, resident_bank)) = resident_acc {
        if hits_to_run.iter().all(|row| row.is_empty()) {
            Ok((resident_out, resident_bank))
        } else {
            let (staged_out, staged_bank) = Gpu::submit(layer, &hits_to_run, &x2d, act)?;
            Ok(((resident_out + staged_out)?, resident_bank.or(staged_bank)))
        }
    } else {
        Gpu::submit(layer, &plan.hits, &x2d, act)
    };
    #[cfg(not(feature = "cuda"))]
    let acc = Gpu::submit(layer, &plan.hits, &x2d, act);
    // Kernels reading these slots may be queued even if a later step failed: fence every
    // slot before letting go of it, so an in-place fill cannot overtake them. If a fence
    // cannot be recorded, wait for the device instead.
    let mut fenced = Ok(());
    for (buf, _, _) in plan.hits.iter().flatten() {
        if let Err(e) = buf.fence.record_reads(layer.slot_device()) {
            fenced = fenced.and(Err(e));
        }
    }
    if fenced.is_err() {
        let _ = layer.slot_device().synchronize();
    }
    // Promote this step's staged experts into the cache (d2d, capped): the bytes already crossed the link, so
    // admitting one is a copy rather than the next step's CPU-path miss. Queued after the fence loop, so the copies
    // sit behind every kernel that reads the staging slots.
    #[cfg(feature = "cuda")]
    if layer.staging_enabled() && layer.promoting() {
        if let Some(cs) = layer.copy_stream() {
            if let Err(e) = layer.promote_staged(cs) {
                tracing::warn!("promote_on_stage: {e}");
            }
        }
    }
    drop(plan.hits);
    #[allow(unused_variables)]
    let (gpu_out, capture_bank) = acc?;
    fenced?;

    // Queue the misses' upload even if the hits failed, so the worker's buffer is released.
    let cpu_out = cpu.map(|p| p.upload()).transpose();
    let sum = match cpu_out? {
        None => gpu_out,
        Some(c) => (gpu_out + c)?,
    };
    sum.to_dtype(xs.dtype())?.reshape((batch, seq, hidden))
}
