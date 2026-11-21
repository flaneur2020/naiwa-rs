//! Linear decode pipeline for one MoE layer:
//!
//! 1. Publish activation for next-layer lookahead prediction (`Preload`).
//! 2. Read router top-k and classify resident hits vs misses under one cache lock (`Route`).
//! 3. Stage top-weight misses via PCIe DMA and dispatch complementary misses to CPU (`DmaMiss` + `CpuMiss`).
//! 4. Wait preload completion, run resident GPU experts, and record their read fences.
//! 5. Wait staging completion, run staged GPU experts, record their read fences, and promote hot staged slots (D2D).
//! 6. Drop all `SlotBuf` handles (after fencing) and join CPU results via `ResultDoorbell`.

use std::sync::Arc;

use candle_core::Tensor;
use mistralrs_core::layers::Activation;

use super::observer::DecodeObserver;
use super::stages::{CpuMiss, CpuPending, DmaMiss, Gpu, Hit, Preload, Route};
use super::CachedLayer;

/// Tracks whether the complementary CPU-miss job was already submitted inside `DmaMiss::submit_cpu_first`.
enum CpuWork {
    Deferred,
    Submitted(Option<CpuPending>),
}

/// Submit GPU expert compute for `hits`, record a read fence on every `SlotBuf`, and drop `hits`
/// only after fencing. Consuming `hits` by value makes it impossible to drop a `SlotBuf` unfenced.
#[inline]
fn run_and_fence_gpu(
    layer: &CachedLayer,
    hits: Vec<Vec<Hit>>,
    x2d: &Tensor,
    act: Activation,
) -> candle_core::Result<Tensor> {
    let out = Gpu::submit(layer, &hits, x2d, act);
    let mut fenced = Ok(());
    for (buf, _, _) in hits.iter().flatten() {
        if let Err(e) = buf.fence.record_reads(layer.slot_device()) {
            fenced = fenced.and(Err(e));
        }
    }
    if fenced.is_err() {
        let _ = layer.slot_device().synchronize();
    }
    drop(hits);
    let out = out?;
    fenced?;
    Ok(out)
}

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
    let mut obs = DecodeObserver::begin(layer, &x2d, rows);

    // 1. Publish activation for next-layer lookahead.
    Preload::submit(layer, &x2d, rows, hidden)?;

    // 2. Read router top-k and classify resident hits vs misses under one cache lock.
    let mut plan = Route::submit(layer, topk_ids, topk_weights, rows)?;
    obs.on_route(layer, &plan)?;

    // 3. Stage top-weight misses via DMA and submit complementary misses to the CPU worker.
    let route_ready = obs.route_ready();
    let mut cpu_work = CpuWork::Deferred;
    let staged_candidates = if layer.staging_enabled() {
        if crate::config::runtime().cpu_before_dma {
            DmaMiss::submit_cpu_first(layer, &mut plan, &mut |misses, chosen| {
                let cpu_misses: Vec<Vec<(u32, f32)>> = misses
                    .iter()
                    .map(|row| {
                        row.iter()
                            .copied()
                            .filter(|(id, _)| !chosen.contains(id))
                            .collect()
                    })
                    .collect();
                let pending = CpuMiss::submit(layer, &cpu_misses, &x2d, rows, route_ready)?;
                cpu_work = CpuWork::Submitted(pending);
                Ok(())
            })?
        } else {
            DmaMiss::submit(layer, &mut plan)?
        }
    } else {
        Vec::new()
    };
    obs.on_dma_enqueued(&mut plan)?;

    let cpu = match cpu_work {
        CpuWork::Submitted(pending) => pending,
        CpuWork::Deferred => CpuMiss::submit(layer, &plan.misses, &x2d, rows, route_ready)?,
    };
    obs.on_cpu_submitted();

    // 4. Wait for any in-flight lookahead preload before reading resident slots.
    #[cfg(feature = "cuda")]
    layer.wait_preload()?;
    obs.gpu_mark("preload_ready");

    #[cfg(feature = "cuda")]
    let split = layer.staging_enabled();
    #[cfg(not(feature = "cuda"))]
    let split = false;

    // 5. Run GPU experts: resident hits first (overlapping staging DMA), then staged hits,
    //    recording read fences before dropping each `SlotBuf` batch, and promote hot staged slots.
    let gpu_out = if split {
        let resident_out = run_and_fence_gpu(layer, plan.resident, &x2d, act)?;
        obs.gpu_mark("resident_end");

        #[cfg(feature = "cuda")]
        layer.wait_staging()?;
        obs.gpu_mark("staging_ready");

        let staged_out = if plan.staged.iter().all(Vec::is_empty) {
            drop(plan.staged);
            None
        } else {
            Some(run_and_fence_gpu(layer, plan.staged, &x2d, act)?)
        };
        obs.gpu_mark("staged_end");

        // Promote staged experts into the cache (D2D on candle's stream, ordered after staged
        // kernels). Consuming `staged_candidates` drops all ring `Arc<SlotBuf>`s before returning.
        #[cfg(feature = "cuda")]
        if !staged_candidates.is_empty() {
            if let candle_core::Device::Cuda(dev) = layer.slot_device() {
                let stream = dev.cuda_stream();
                if let Err(e) = layer.promote_staged(&stream, staged_candidates) {
                    tracing::warn!("promote_on_stage: {e}");
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        drop(staged_candidates);

        match staged_out {
            Some(staged) => (resident_out + staged)?,
            None => resident_out,
        }
    } else {
        drop(staged_candidates);
        drop(plan.staged);
        obs.gpu_mark("resident_end");
        #[cfg(feature = "cuda")]
        layer.wait_staging()?;
        obs.gpu_mark("staging_ready");
        let out = run_and_fence_gpu(layer, plan.resident, &x2d, act)?;
        obs.gpu_mark("staged_end");
        out
    };

    // 6. Queue CPU-result upload via `ResultDoorbell` and combine with GPU output.
    obs.gpu_mark("promotion_end");
    let cpu_out = cpu.map(|p| p.upload()).transpose();
    obs.gpu_mark("cpu_ready");
    let sum = match cpu_out? {
        None => gpu_out,
        Some(c) => (gpu_out + c)?,
    };
    let out = sum.to_dtype(xs.dtype())?.reshape((batch, seq, hidden))?;
    obs.gpu_mark("combine_end");
    Ok(out)
}
