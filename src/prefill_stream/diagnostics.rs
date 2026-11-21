//! Opt-in prefill diagnostics (`NAIWA_PREFILL_PEAK`, `NAIWA_REQ_TIME`, `NAIWA_BORROW_CHECK`,
//! `NAIWA_BORROW_SNAP`, `NAIWA_BORROW_CANARY`, `NAIWA_COPY_TIME`), kept out of `forward()`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use candle_core::{DType, Device, Tensor};
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
        let Some(free) = crate::copy_stream::free_bytes(&self.device) else {
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
        let (live, retired) = mistralrs_quant::gguf::cuda::moe_workspace_bytes();
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

    /// `NAIWA_BORROW_CHECK=1`: check whether the chunk's row inputs are finite.
    pub(super) fn check_inputs_finite(&self, layer: usize, xs: &Tensor, topk_weights: &Tensor) {
        if !crate::config::diag().borrow_check {
            return;
        }
        let finite = |t: &Tensor| -> bool {
            t.flatten_all()
                .and_then(|f| f.to_dtype(DType::F32))
                .and_then(|f| f.to_device(&Device::Cpu))
                .and_then(|f| f.to_vec1::<f32>())
                .map(|v| v.iter().all(|x| x.is_finite()))
                .unwrap_or(true)
        };
        static N: AtomicUsize = AtomicUsize::new(0);
        if (!finite(xs) || !finite(topk_weights)) && N.fetch_add(1, Ordering::Relaxed) < 8 {
            eprintln!("PREFILL_NAN layer {layer}: the chunk's row inputs are not finite");
        }
    }

    /// `NAIWA_BORROW_SNAP=1` pre-compute snapshot queued on the copy stream + `NAIWA_BORROW_CANARY=1`.
    pub(super) fn probe_before_compute(
        &self,
        layer: usize,
        buf: &LayerBuf,
    ) -> candle_core::Result<bool> {
        let take_snap = self.borrows() && crate::config::diag().borrow_snap;
        if take_snap {
            let shadow = {
                let mut slot = self.snap.lock().unwrap();
                if slot.is_none() {
                    let dev = self.device.as_cuda_device().expect("cuda");
                    let st = candle_core::quantized::cuda::QCudaStorage::zeros(
                        dev,
                        2 * 3 * 4096 / 4,
                        candle_core::quantized::GgmlDType::F32,
                    )?;
                    slot.replace(Arc::new(candle_core::quantized::QTensor::new(
                        candle_core::quantized::QStorage::Cuda(st),
                        (2 * 3 * 4096 / 4,),
                    )?));
                }
                Arc::clone(slot.as_ref().unwrap())
            };
            if let Some(m) = buf.memory.as_ref() {
                for (i, mem) in m.iter().enumerate() {
                    mem.tensor().copy_range_to(
                        &shadow,
                        0,
                        i * 4096,
                        4096,
                        self.copy.cuda_stream(),
                    )?;
                }
            }
        }
        if self.borrows() && crate::config::diag().borrow_canary {
            if let Some(m) = buf.memory.as_ref() {
                let before: Vec<Vec<u8>> = m
                    .iter()
                    .map(|mem| mem.tensor().read_bytes_at(0, 256).unwrap_or_default())
                    .collect();
                for (i, mem) in m.iter().enumerate() {
                    if let Ok(now) = mem.tensor().read_bytes_at(0, 256) {
                        if now != before[i] {
                            let at = now
                                .iter()
                                .zip(&before[i])
                                .position(|(a, b)| a != b)
                                .unwrap_or(0);
                            eprintln!(
                                "CANARY_MISMATCH layer {layer} proj {i} first byte {at}: was {:02x} now {:02x}",
                                before[i][at], now[at]
                            );
                        }
                    }
                }
            }
        }
        Ok(take_snap)
    }

    /// `NAIWA_BORROW_SNAP=1` post-compute snapshot queued on candle's stream.
    pub(super) fn snap_after_compute(&self, buf: &LayerBuf) -> candle_core::Result<()> {
        let shadow = self.snap.lock().unwrap().as_ref().map(Arc::clone);
        if let Some(shadow) = shadow {
            if let Some(m) = buf.memory.as_ref() {
                let dev = self
                    .device
                    .as_cuda_device()
                    .map_err(|e| candle_core::Error::Msg(format!("cuda: {e}")))?;
                for (i, mem) in m.iter().enumerate() {
                    mem.tensor().copy_range_to(
                        &shadow,
                        0,
                        3 * 4096 + i * 4096,
                        4096,
                        &dev.cuda_stream(),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Compare pre- and post-compute snapshots against host stacks after the copy stream syncs.
    pub(super) fn verify_snap_after_sync(&self, layer: usize) -> candle_core::Result<()> {
        let shadow = self.snap.lock().unwrap().as_ref().map(Arc::clone);
        if let Some(shadow) = shadow {
            let host_stacks = self
                .stacks(layer)
                .unwrap_or_else(|| unreachable!("loaded above"));
            for (i, stack) in host_stacks.iter().enumerate() {
                let host = stack.data()?;
                let want = &host[..4096.min(host.len())];
                let a = shadow.read_bytes_at(i * 4096, 4096).unwrap_or_default();
                let b = shadow
                    .read_bytes_at(3 * 4096 + i * 4096, 4096)
                    .unwrap_or_default();
                if a.len() == want.len() && a != want {
                    let at = a.iter().zip(want).position(|(x, y)| x != y).unwrap_or(0);
                    eprintln!(
                        "SNAP_A_MISMATCH layer {layer} proj {i} byte {at}: upload wrote {:02x}, host has {:02x}",
                        a[at], want[at]
                    );
                }
                if a != b {
                    let at = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(0);
                    eprintln!(
                        "SNAP_B_MISMATCH layer {layer} proj {i} byte {at}: after upload {:02x}, after compute {:02x}",
                        a[at], b[at]
                    );
                }
            }
        }
        Ok(())
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
