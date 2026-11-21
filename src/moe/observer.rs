//! Zero-cost (`#[inline]`) per-step decode observer.
//!
//! Keeps diagnostic probes (`ALLOC_TRACE`, `NAIWA_XSUM`, `NAIWA_WORK_TRACE`, `NAIWA_SCHEDULE_FILE`,
//! `NAIWA_GPU_PARTITION_FILE`, `NAIWA_CPU_TIMELINE`, and cache counters) out of the linear control
//! flow in [`super::decode`]. When probes are disabled, every hook reduces to a cached boolean check.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use candle_core::Tensor;

use crate::stats;

use super::stages::RoutePlan;
use super::CachedLayer;

pub(super) struct DecodeObserver {
    layer: usize,
    rows: usize,
    schedule_tick: u64,
    route_ready: Option<Instant>,
    #[cfg(feature = "cuda")]
    partition_stream: Option<std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>>,
}

impl DecodeObserver {
    /// Step entry hook: allocation ledger and optional `NAIWA_XSUM` input checksum.
    #[inline]
    pub(super) fn begin(layer: &CachedLayer, x2d: &Tensor, rows: usize) -> Self {
        if candle_core::cuda_backend::alloc_trace::enabled() {
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
        if crate::config::diag().xsum {
            let tick = layer.state().lock().unwrap().tick + 1;
            if tick <= 3 {
                if let Ok(v) = x2d
                    .to_dtype(candle_core::DType::F32)
                    .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
                {
                    let sum: f64 = v.iter().map(|&x| x as f64).sum();
                    let l1: f64 = v.iter().map(|&x| (x as f64).abs()).sum();
                    eprintln!(
                        "XSUM tick={tick} layer={} rows={rows} sum={sum:.9} l1={l1:.9} first={:.9}",
                        layer.layer(),
                        v[0]
                    );
                }
            }
        }
        Self {
            layer: layer.layer(),
            rows,
            schedule_tick: 0,
            route_ready: None,
            #[cfg(feature = "cuda")]
            partition_stream: None,
        }
    }

    /// Invoked right after `Route::submit` classifies resident hits and misses.
    #[inline]
    pub(super) fn on_route(
        &mut self,
        layer: &CachedLayer,
        plan: &RoutePlan,
    ) -> candle_core::Result<()> {
        if stats::schedule::enabled() {
            self.schedule_tick = layer.state().lock().unwrap().tick;
        }
        #[cfg(feature = "cuda")]
        if stats::gpu_partition::enabled() && layer.slot_device().is_cuda() {
            self.partition_stream = Some(layer.slot_device().as_cuda_device()?.cuda_stream());
        }
        self.gpu_mark("route");
        stats::schedule::event(self.layer, self.schedule_tick, "route_ready");
        self.route_ready = crate::config::diag().cpu_timeline.then(Instant::now);
        let n_hits: usize = plan.resident.iter().map(Vec::len).sum();
        stats::record_cache_hits(n_hits);
        Ok(())
    }

    #[inline]
    pub(super) fn route_ready(&self) -> Option<Instant> {
        self.route_ready
    }

    /// Invoked after `DmaMiss` finishes staging and mutating `plan.{staged, misses}`.
    #[inline]
    pub(super) fn on_dma_enqueued(&self, plan: &mut RoutePlan) -> candle_core::Result<()> {
        stats::schedule::event(self.layer, self.schedule_tick, "dma_enqueued");
        if let Some(mut trace) = plan.work_trace.take() {
            emit_work_trace(&mut trace, plan)?;
        }
        let n_cpu: usize = plan.misses.iter().map(Vec::len).sum();
        stats::record_cache_misses(n_cpu);
        #[cfg(feature = "cuda")]
        {
            let staged_n: usize = plan.staged.iter().map(Vec::len).sum();
            stats::record_cache_staged(staged_n);
        }
        Ok(())
    }

    #[inline]
    pub(super) fn on_cpu_submitted(&self) {
        stats::schedule::event(self.layer, self.schedule_tick, "cpu_submit_return");
        self.gpu_mark("plan_submit");
    }

    #[inline]
    pub(super) fn gpu_mark(&self, phase: &'static str) {
        #[cfg(feature = "cuda")]
        if let Some(stream) = &self.partition_stream {
            stats::gpu_partition::mark(stream, self.layer, self.rows, phase);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = phase;
    }
}

#[cold]
fn emit_work_trace(trace: &mut serde_json::Value, plan: &RoutePlan) -> candle_core::Result<()> {
    trace["gpu_slots_weight_bits"] = serde_json::json!(plan
        .resident
        .iter()
        .zip(&plan.staged)
        .map(|(res, stg)| res
            .iter()
            .chain(stg.iter())
            .map(|(_, slot, w)| (*slot, w.to_bits()))
            .collect::<Vec<_>>())
        .collect::<Vec<_>>());
    trace["cpu_ids_weight_bits"] = serde_json::json!(plan
        .misses
        .iter()
        .map(|row| row.iter().map(|(id, w)| (*id, w.to_bits())).collect::<Vec<_>>())
        .collect::<Vec<_>>());
    // Separate sink prevents tracing's stdout logger from interleaving JSON bytes.
    static WORK_FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let file = WORK_FILE.get_or_init(|| {
        crate::config::diag().work_trace_file.as_ref().map(|path| {
            Mutex::new(std::fs::File::create(path).expect("create NAIWA_WORK_TRACE_FILE"))
        })
    });
    if let Some(file) = file {
        use std::io::Write;
        writeln!(file.lock().unwrap(), "MOE_WORK {trace}")
            .map_err(|e| candle_core::Error::Msg(format!("write work trace: {e}")))?;
    } else {
        eprintln!("MOE_WORK {trace}");
    }
    Ok(())
}
