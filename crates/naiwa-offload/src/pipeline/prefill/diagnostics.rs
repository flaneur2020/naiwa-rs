//! Opt-in prefill diagnostics (`NAIWA_PREFILL_PEAK`, `NAIWA_REQ_TIME`, `NAIWA_STREAM_DBG`),
//! kept out of `forward()`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tracing::info;

use super::buffers::LayerBuf;
use super::planner::{activation_bytes, MAX_HEADROOM};
use super::PrefillStream;

/// One chunk's VRAM footprint, sampled at every layer (`NAIWA_PREFILL_PEAK=1`).
#[derive(Default, Clone)]
pub(super) struct Peak {
    /// Free VRAM just before the chunk allocated its layer buffer; 0 until a chunk starts.
    free_at_start: usize,
    /// Lowest free VRAM seen inside the chunk.
    min_free: usize,
    /// The layer the low water was first seen at.
    low_layer: usize,
    trace: Vec<(usize, usize)>,
    layers: usize,
    /// The rows this chunk actually carried (not the plan's ceiling).
    rows: usize,
}

impl PrefillStream {
    pub(super) fn sample_peak(&self, layer: usize, rows: usize) {
        if !crate::config::diag().prefill_peak {
            return;
        }
        let Some(free) = crate::transport::copy_stream::free_bytes(&self.device) else {
            return;
        };
        let total = self.layers.read().len();
        let mut peak = self.peak.lock().unwrap();
        if layer == 0 || peak.free_at_start == 0 {
            *peak = Peak {
                free_at_start: free,
                min_free: free,
                low_layer: layer,
                trace: Vec::with_capacity(total),
                layers: 0,
                rows,
            };
            if layer != 0 {
                return;
            }
        }
        if peak.min_free > free {
            peak.low_layer = layer;
        }
        peak.min_free = peak.min_free.min(free);
        peak.trace.push((layer, free));
        peak.layers += 1;
        if total == 0 || peak.layers < total {
            return;
        }
        let mib = |b: usize| (b as f64) / (1u64 << 20) as f64;
        let used = peak.free_at_start.saturating_sub(peak.min_free);
        let plan = self.current();
        info!(
            "prefill peak (low water at layer {}): chunk {} rows, {} layers, {:.0} MiB used \
             ({:.0} MiB free at start, {:.0} MiB low water); layer buffer {:.0} MiB, planner estimate {:.0} MiB \
             (act {:.0})",
            peak.low_layer,
            peak.rows,
            peak.layers,
            mib(used),
            mib(peak.free_at_start),
            mib(peak.min_free),
            mib(self.layer_bytes() * plan.stride),
            mib(activation_bytes(plan.chunk) + MAX_HEADROOM),
            mib(activation_bytes(plan.chunk)),
        );
        let base = peak.free_at_start;
        let trace: Vec<String> = peak
            .trace
            .iter()
            .map(|(l, f)| format!("{l}:{}", (base.saturating_sub(*f)) >> 20))
            .collect();
        info!("prefill peak trace (layer: MiB used): {}", trace.join(" "));
        let (live, retired) = naiwa_quant::moe_workspace_bytes();
        info!(
            "prefill peak: MoE workspaces {:.0} MiB live + {:.0} MiB retired (freed only when a graph may hold \
             them)",
            mib(live),
            mib(retired)
        );
        peak.free_at_start = 0;
    }

    /// `NAIWA_REQ_TIME=1`: absolute wall-clock stamps at the first and last layer of a chunk.
    pub(super) fn note_req_time(&self, layer: usize) {
        if !crate::config::diag().req_time {
            return;
        }
        static FIRST: AtomicU64 = AtomicU64::new(0);
        static LAST: AtomicU64 = AtomicU64::new(0);
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        if layer == 0 {
            FIRST.store(now_us, Ordering::Relaxed);
        }
        if self
            .layers
            .read()
            .iter()
            .rposition(|l| l.is_some())
            .is_some_and(|last| last == layer)
        {
            LAST.store(now_us, Ordering::Relaxed);
            let first = FIRST.load(Ordering::Relaxed);
            crate::dbg_note(
                "req_time",
                &format!(
                    "submit_us={first} first_layer_us={first} last_layer_us={now_us} layers_us={}",
                    now_us - first
                ),
            );
        }
    }

    pub(super) fn log_copy_and_layer_timing(
        &self,
        layer: usize,
        rows: usize,
        stride: usize,
        gap: Option<Duration>,
        prefetch: Duration,
        waited: Duration,
        compute_start: Instant,
        buf: &LayerBuf,
    ) {
        let transport = buf.queued.map(|queued| {
            let copy = queued.elapsed();
            let gbs = buf.bytes as f64 / copy.as_secs_f64().max(1e-9) / 1e9;
            format!(
                " copy={copy:?} {gbs:.1} GB/s {} MiB x {}",
                buf.bytes >> 20,
                buf.copies
            )
        });
        crate::dbg_note(
            "prefill_stream",
            &format!(
                "layer {layer} rows {rows} stride {stride} gap={gap:?} pf={prefetch:?} dma_wait={waited:?} compute={:?}{}",
                compute_start.elapsed(),
                transport.unwrap_or_default()
            ),
        );
    }
}
