//! The decode path: split top-k into GPU hits and CPU misses, run both, sum.

use std::sync::Arc;

use candle_core::Tensor;
use mistralrs_core::layers::Activation;

use super::stages::{CpuMiss, CpuPending, DmaMiss, Gpu, Hit, Preload, Route};
use crate::stats;

use super::CachedLayer;

/// Tracks whether the complementary CPU-miss job was already submitted inside `DmaMiss::submit_cpu_first`.
enum CpuWork {
    Deferred,
    Submitted(Option<CpuPending>),
}

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
    // A step's first layer marks the boundary: print the previous step's budget (see `stats::budget`).
    // M28: allocation ledger. Every `CudaDevice::alloc`/`alloc_zeros` between two layers belongs to the previous
    // layer, so the window is taken at the layer boundary; at a step boundary the previous step's totals print.
    if candle_core::cuda_backend::alloc_trace::enabled() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static MAXB: AtomicU64 = AtomicU64::new(0);
        static SUMB: AtomicU64 = AtomicU64::new(0);
        static SUMC: AtomicU64 = AtomicU64::new(0);
        let (b, c) = candle_core::cuda_backend::alloc_trace::take();
        SUMB.fetch_add(b, Ordering::Relaxed);
        SUMC.fetch_add(c, Ordering::Relaxed);
        MAXB.fetch_max(b, Ordering::Relaxed);
        if layer.layer() == 0 && SUMB.load(Ordering::Relaxed) > 0 {
            eprintln!(
                "ALLOC_TRACE step: {} allocs, {:.1} MB total, {:.1} MB peak per layer",
                SUMC.swap(0, Ordering::Relaxed),
                SUMB.swap(0, Ordering::Relaxed) as f64 / 1e6,
                MAXB.swap(0, Ordering::Relaxed) as f64 / 1e6
            );
        }
    }
    if layer.layer() == 0 {
        stats::record_step_boundary(layer.layers().len());
    }
    let budget = stats::budget_enabled();
    let mut t = budget.then(std::time::Instant::now);
    macro_rules! mark {
        ($phase:expr) => {
            if let Some(prev) = t.as_mut() {
                stats::record_budget($phase, prev.elapsed().as_nanos() as u64);
                *prev = std::time::Instant::now();
            }
        };
    }
    Preload::submit(layer, &x2d, rows, hidden)?;
    // M42: hand the same activation to the CPU experts through mapped memory, queued here (before
    // the routing read) so the token thread never has to sync for it.
    let cpu_act = match layer.cpu_chan() {
        Some(chan) if rows * hidden <= chan.max_n() => chan.publish(&x2d, rows * hidden)?,
        _ => None,
    };

    // NAIWA_XSUM: checksum of this layer's MoE input, for the first few ticks only. Repeated runs
    // of one greedy request that end up with different tokens diverge somewhere; the first layer
    // whose input checksum differs is that point, which no earlier probe could localize. Costs a
    // D2H per layer, so it is diagnostic only.
    if crate::config::diag().xsum {
        let tick = layer.state().lock().unwrap().tick + 1;
        if tick <= 3 {
            if let Ok(v) = x2d.to_dtype(candle_core::DType::F32).and_then(|t| t.flatten_all()?.to_vec1::<f32>()) {
                let sum: f64 = v.iter().map(|&x| x as f64).sum();
                let l1: f64 = v.iter().map(|&x| (x as f64).abs()).sum();
                eprintln!("XSUM tick={tick} layer={} rows={rows} sum={sum:.9} l1={l1:.9} first={:.9}", layer.layer(), v[0]);
            }
        }
    }

    // M27: the raw routing demand, before any cache decision, for the allocation study.
    // Do not synchronize/copy routing ids for a disabled histogram. The debug
    // override preserves the previous behaviour for an A/B measurement.
    if crate::config::diag().expert_hist.is_some()
        || crate::config::diag().force_hist_read
    {
        if let Ok(ids) = topk_ids.flatten_all() {
            if let Ok(ids) = ids.to_dtype(candle_core::DType::U32).and_then(|t| t.to_vec1::<u32>()) {
                stats::record_expert_demand(layer.layer(), &ids);
            }
        }
    }
    let mut plan = Route::submit(layer, topk_ids, topk_weights, rows)?;
    let schedule_tick = if stats::schedule::enabled() {
        layer.state().lock().unwrap().tick
    } else { 0 };
    #[cfg(feature = "cuda")]
    let partition_stream = if stats::gpu_partition::enabled() && layer.slot_device().is_cuda() {
        Some(layer.slot_device().as_cuda_device()?.cuda_stream())
    } else { None };
    macro_rules! gpu_mark { ($phase:expr) => { #[cfg(feature = "cuda")] if let Some(stream) = &partition_stream { stats::gpu_partition::mark(stream, layer.layer(), rows, $phase); } }; }
    gpu_mark!("route");
    stats::schedule::event(layer.layer(), schedule_tick, "route_ready");
    // M41: this boundary follows host routing/classification, not GPU top-k completion.
    let route_ready = crate::config::diag()
        .cpu_timeline
        .then(std::time::Instant::now);
    let n_hits: usize = plan.resident.iter().map(Vec::len).sum();
    stats::record_cache_hits(n_hits);

    // `--pcie-frac`: DMA the highest-weight misses into the staging ring (`plan.staged`), leaving
    // the complementary misses on the CPU (`plan.misses`).
    mark!(stats::BudgetPhase::Plan);
    let mut cpu_work = CpuWork::Deferred;
    if layer.staging_enabled() {
        if crate::config::runtime().cpu_before_dma {
            DmaMiss::submit_cpu_first(layer, &mut plan, &mut |misses, chosen| {
                let cpu_misses: Vec<Vec<(u32, f32)>> = misses.iter().map(|row| {
                    row.iter().copied().filter(|(id, _)| !chosen.contains(id)).collect()
                }).collect();
                let pending = CpuMiss::submit(layer, &cpu_misses, &x2d, rows,
                    route_ready, cpu_act.clone())?;
                cpu_work = CpuWork::Submitted(pending);
                Ok(())
            })?;
        } else {
            DmaMiss::submit(layer, &mut plan)?;
        }
    }

    stats::schedule::event(layer.layer(), schedule_tick, "dma_enqueued");
    if let Some(mut trace) = plan.work_trace.take() {
        trace["gpu_slots_weight_bits"] = serde_json::json!(plan.resident.iter()
            .zip(&plan.staged)
            .map(|(res, stg)| res.iter().chain(stg.iter()).map(|(_, slot, w)| (*slot, w.to_bits())).collect::<Vec<_>>())
            .collect::<Vec<_>>());
        trace["cpu_ids_weight_bits"] = serde_json::json!(plan.misses.iter()
            .map(|row| row.iter().map(|(id, w)| (*id, w.to_bits())).collect::<Vec<_>>())
            .collect::<Vec<_>>());
        // Separate sink prevents tracing's stdout logger from interleaving JSON bytes.
        static WORK_FILE: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
        let file = WORK_FILE.get_or_init(|| crate::config::diag().work_trace_file.as_ref()
            .map(|path| std::sync::Mutex::new(std::fs::File::create(path)
                .expect("create NAIWA_WORK_TRACE_FILE"))));
        if let Some(file) = file {
            use std::io::Write;
            writeln!(file.lock().unwrap(), "MOE_WORK {trace}")
                .map_err(|e| candle_core::Error::Msg(format!("write work trace: {e}")))?;
        } else {
            eprintln!("MOE_WORK {trace}");
        }
    }
    mark!(stats::BudgetPhase::Stage);
    let n_cpu: usize = plan.misses.iter().map(Vec::len).sum();
    stats::record_cache_misses(n_cpu);
    // M22.3: how much of the step's expert demand is distinct experts, i.e. the headroom a per-expert row batch
    // (llama.cpp MMVQ's `ncols`) would have. Resident and staged hits are counted by slot, misses by expert id.
    #[cfg(feature = "cuda")]
    if budget {
        // Every (row, expert) pair this layer will run, resident + staged + CPU.
        let staged_n: usize = plan.staged.iter().map(Vec::len).sum();
        let pairs = n_hits + staged_n + n_cpu;
        let mut ids: Vec<usize> = plan
            .resident
            .iter()
            .chain(&plan.staged)
            .flatten()
            .map(|h| h.1)
            .collect();
        ids.extend(plan.misses.iter().flatten().map(|&(id, _)| id as usize));
        ids.sort_unstable();
        ids.dedup();
        stats::record_budget_pairs(ids.len(), pairs);
    }
    // Experts that went to the staging ring: counted apart from both hits and CPU misses.
    #[cfg(feature = "cuda")]
    {
        let staged_n: usize = plan.staged.iter().map(Vec::len).sum();
        stats::record_cache_staged(staged_n);
        stats::record_round_plan(n_hits, staged_n, n_cpu);
    }

    // M28: this layer's allocation bytes (no reset here; the step boundary reads the window).
    #[cfg(feature = "cuda")]
    if candle_core::cuda_backend::alloc_trace::enabled() {
        // Handled at the step boundary: the window accumulates across the step.
    }
    let cpu = match cpu_work {
        CpuWork::Submitted(pending) => pending,
        CpuWork::Deferred => CpuMiss::submit(layer, &plan.misses, &x2d, rows, route_ready, cpu_act)?,
    };
    stats::schedule::event(layer.layer(), schedule_tick, "cpu_submit_return");
    mark!(stats::BudgetPhase::Spawn);

    gpu_mark!("plan_submit");
    // Wait for the latest preload claim on candle's stream. The claim carries its own completion
    // event, so this also drains a copy the lookahead queued late; a layer that never gets
    // predicted has no claim and costs no CUDA call at all.
    #[cfg(feature = "cuda")]
    {
        layer.wait_preload()?;
    }
    gpu_mark!("preload_ready");
    mark!(stats::BudgetPhase::PreloadWait);
    // S2: the resident experts run *before* the staging wait, so the DMA lands under their compute
    // instead of in front of it (measured 78.9 -> 62.8 ms per pass, -20%, and the p50 wait drops
    // from 858 us to 11 us). The staged experts still need the ring, so they keep the wait. This is
    // the one fixed behaviour - there is no switch.
    #[cfg(feature = "cuda")]
    let split = layer.staging_enabled();
    #[cfg(not(feature = "cuda"))]
    let split = false;

    #[cfg(feature = "cuda")]
    let cuda_stream = if stats::budget_enabled() {
        Some(layer.slot_device().as_cuda_device()?.cuda_stream())
    } else {
        None
    };
    let expert_ev = cuda_stream.as_ref().and_then(|s| stats::budget_dev::start(s));
    #[cfg(feature = "cuda")]
    let resident_acc = if split {
        let acc = Gpu::submit(layer, &plan.resident, &x2d, act);
        let mut fenced = Ok(());
        for (buf, _, _) in plan.resident.iter().flatten() {
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

    gpu_mark!("resident_end");
    // ... and the staging DMA's completion event, on candle's stream.
    #[cfg(feature = "cuda")]
    layer.wait_staging()?;
    gpu_mark!("staging_ready");
    mark!(stats::BudgetPhase::Gpu);

    #[cfg(feature = "cuda")]
    let acc = if let Some(resident_out) = resident_acc {
        if plan.staged.iter().all(Vec::is_empty) {
            Ok(resident_out)
        } else {
            // M26: submit the staged experts wave by wave, waiting only for the wave about to run, so
            // the later waves' DMA lands under the earlier waves' kernels.
            let waves = layer.wave_count();
            let mut acc: Option<Tensor> = None;
            if waves > 1 {
                let rows = plan.staged.len();
                let mut groups: Vec<Vec<Vec<Hit>>> =
                    (0..waves).map(|_| vec![Vec::new(); rows]).collect();
                for (t, row) in plan.staged.iter().enumerate() {
                    for hit in row {
                        let pos = hit
                            .1
                            .saturating_sub(crate::moe::STAGING_RING_FIRST) as u32;
                        let w = layer.wave_of_slot(pos).min(waves - 1);
                        groups[w][t].push(hit.clone());
                    }
                }
                for (w, group) in groups.iter().enumerate() {
                    if group.iter().all(|row| row.is_empty()) {
                        continue;
                    }
                    // Wave `w`'s own event, or the layer's final event for the last wave.
                    if w + 1 < waves {
                        layer.wait_staging_wave(w)?;
                    } else {
                        layer.wait_staging()?;
                    }
                    let out = Gpu::submit(layer, group, &x2d, act)?;
                    acc = Some(match acc {
                        Some(a) => (a + out)?,
                        None => out,
                    });
                }
            }
            let staged_out = match acc {
                Some(v) => v,
                None => Gpu::submit(layer, &plan.staged, &x2d, act)?,
            };
            resident_out + staged_out
        }
    } else {
        Gpu::submit(layer, &plan.resident, &x2d, act)
    };
    #[cfg(not(feature = "cuda"))]
    let acc = Gpu::submit(layer, &plan.resident, &x2d, act);
    #[cfg(feature = "cuda")]
    if let Some(stream) = &cuda_stream {
        stats::budget_dev::end(stream, expert_ev);
    }
    gpu_mark!("staged_end");
    // Kernels reading these slots may be queued even if a later step failed: fence every
    // slot before letting go of it, so an in-place fill cannot overtake them. When `split`
    // is active, resident slots were already fenced right after their compute; fence the
    // staged slots here (or all resident slots when `!split`).
    let mut fenced = Ok(());
    let slots_to_fence = if split { &plan.staged } else { &plan.resident };
    for (buf, _, _) in slots_to_fence.iter().flatten() {
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
        // On candle's stream: the d2d copy needs ordering against the kernels that read the staging slot, and that
        // is the stream they run on. The PCIe copy stream is for the host-side staging and is already saturated.
        if let candle_core::Device::Cuda(dev) = layer.slot_device() {
            let stream = dev.cuda_stream();
            if let Err(e) = layer.promote_staged(&stream) {
                tracing::warn!("promote_on_stage: {e}");
            }
        }
    }
    mark!(stats::BudgetPhase::StagingWait);
    drop(plan.resident);
    drop(plan.staged);
    let gpu_out = acc?;
    fenced?;
    mark!(stats::BudgetPhase::Fence);

    // Queue the misses' upload even if the hits failed, so the worker's buffer is released.
    gpu_mark!("promotion_end");
    let cpu_out = cpu.map(|p| p.upload()).transpose();
    gpu_mark!("cpu_ready");
    mark!(stats::BudgetPhase::CpuJoin);
    let sum = match cpu_out? {
        None => gpu_out,
        Some(c) => (gpu_out + c)?,
    };
    let out = sum.to_dtype(xs.dtype())?.reshape((batch, seq, hidden))?;
    gpu_mark!("combine_end");
    mark!(stats::BudgetPhase::Other);
    Ok(out)
}
