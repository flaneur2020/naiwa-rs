//! Measurement probes (`NAIWA_LOOKAHEAD_DBG`, `NAIWA_ROUND_STATS`, `NAIWA_STEP_DBG`).

/// `NAIWA_LOOKAHEAD_DBG=1`: per decode step, the synchronous D2H of the activation that the lookahead
/// needs, in ms. `--no-lookahead` skips the prediction (and its D2H) entirely, so the cache only
/// ever holds what prefill left, and the A/B is the price of the prediction against its benefit.
pub(super) mod lookahead_probe {
    use std::sync::Mutex;

    const WINDOW: u64 = 384; // eight layer passes

    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    /// Token-thread cost of queueing the publish (the `publish` call, host side).
    pub(crate) fn record_publish(ms: f64) {
        if !on() {
            return;
        }
        static ACC: Mutex<(u64, f64)> = Mutex::new((0, 0.0));
        let mut a = ACC.lock().unwrap();
        a.0 += 1;
        a.1 += ms;
        if a.0 % WINDOW == 0 {
            eprintln!(
                "LOOKAHEAD_DBG publishes {} publish_ms/step {:.3}",
                a.0,
                a.1 / WINDOW as f64
            );
            *a = (0, 0.0);
        }
    }

    /// Worker-side job duration: `wait_publish` latency plus the CPU matmul.
    pub(crate) fn record_job(ms: f64) {
        if !on() {
            return;
        }
        static ACC: Mutex<(u64, f64, f64)> = Mutex::new((0, 0.0, 0.0));
        let mut a = ACC.lock().unwrap();
        a.0 += 1;
        a.1 += ms;
        if ms > a.2 {
            a.2 = ms;
        }
        if a.0 % WINDOW == 0 {
            eprintln!(
                "LOOKAHEAD_DBG jobs {} job_ms/step {:.3} worst {} ms",
                a.0,
                a.1 / WINDOW as f64,
                a.2
            );
            *a = (0, 0.0, 0.0);
        }
    }

    pub(crate) fn record(d2h_ms: f64, queued: bool) {
        let on = *ON.get_or_init(|| {
            std::env::var_os("NAIWA_LOOKAHEAD_DBG").is_some() || !queued
        });
        if !on {
            return;
        }
        static ACC: Mutex<(u64, f64)> = Mutex::new((0, 0.0));
        let mut a = ACC.lock().unwrap();
        a.0 += 1;
        a.1 += d2h_ms;
        if a.0 % WINDOW == 0 {
            eprintln!(
                "LOOKAHEAD_DBG steps {} d2h_ms/step {:.3}",
                a.0,
                a.1 / WINDOW as f64
            );
            *a = (0, 0.0);
        }
    }

    fn on() -> bool {
        *ON.get_or_init(|| std::env::var_os("NAIWA_LOOKAHEAD_DBG").is_some())
    }
}

/// The CPU job's gemv time, reported per pass by `--round-stats` (`NAIWA_ROUND_STATS`).
pub(crate) fn record_cpu_compute(ns: u64) {
    round_stats::record_cpu_compute(ns);
}

/// Per-pass accounting of the three producers, so the bench can print where a round's MoE time went:
/// resident hits, staged experts (PCIe) and CPU-computed misses, plus the CPU compute time that the
/// deferred probe measured for this pass.
pub(super) mod round_stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    static HITS: AtomicU64 = AtomicU64::new(0);
    static STAGED: AtomicU64 = AtomicU64::new(0);
    static CPU_MISSES: AtomicU64 = AtomicU64::new(0);
    static LAYERS: AtomicU64 = AtomicU64::new(0);
    static STAGED_BYTES: AtomicU64 = AtomicU64::new(0);
    static CPU_NS: AtomicU64 = AtomicU64::new(0);
    static HEADER: Mutex<bool> = Mutex::new(false);

    pub(crate) fn record_plan(hits: usize, staged: usize, cpu: usize) {
        HITS.fetch_add(hits as u64, Ordering::Relaxed);
        STAGED.fetch_add(staged as u64, Ordering::Relaxed);
        CPU_MISSES.fetch_add(cpu as u64, Ordering::Relaxed);
    }

    /// Called by the staging path: how many bytes this layer copied over PCIe.
    pub(crate) fn record_staged_bytes(bytes: u64) {
        STAGED_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Called by the CPU job when it finishes: the gemv time of one job.
    pub(crate) fn record_cpu_compute(ns: u64) {
        CPU_NS.fetch_add(ns, Ordering::Relaxed);
    }

    /// Called once per pass (layer 0).
    pub(crate) fn finish_pass(on: bool) {
        let layers = LAYERS.fetch_add(1, Ordering::Relaxed) + 1;
        if !on || layers % 4 != 0 {
            return;
        }
        let h = HITS.swap(0, Ordering::Relaxed);
        let st = STAGED.swap(0, Ordering::Relaxed);
        let cpu = CPU_MISSES.swap(0, Ordering::Relaxed);
        let bytes = STAGED_BYTES.swap(0, Ordering::Relaxed);
        let cpu_ns = CPU_NS.swap(0, Ordering::Relaxed);
        let sel = h + st + cpu;
        if sel == 0 {
            return;
        }
        let mut hdr = HEADER.lock().unwrap();
        if !*hdr {
            *hdr = true;
            eprintln!(
                "ROUND_STATS rounds hit% staged% cpu% staged_MiB cpu_ms/stage"
            );
        }
        eprintln!(
            "ROUND_STATS {layers} {:.1} {:.1} {:.1} {:.1} {:.2}",
            100.0 * h as f64 / sel as f64,
            100.0 * st as f64 / sel as f64,
            100.0 * cpu as f64 / sel as f64,
            bytes as f64 / (1024.0 * 1024.0),
            cpu_ns as f64 / 1e6,
        );
    }
}

pub(super) mod step_probe {
    use std::sync::Mutex;
    use std::time::Instant;

    use candle_core::Device;

    const WINDOW: usize = 48; // one token's worth of layers

    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    #[cfg(feature = "cuda")]
    mod imp {
        use candle_core::cuda::cudarc::driver::sys;
        use candle_core::Device;

        pub(super) struct Event(candle_core::cuda::cudarc::driver::CudaEvent);

        impl Event {
            pub(crate) fn new(device: &Device) -> candle_core::Result<Self> {
                let Device::Cuda(dev) = device else {
                    unreachable!("caller checks is_cuda");
                };
                let event = dev
                    .cuda_stream()
                    .context()
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                    .map_err(candle_core::Error::wrap)?;
                Ok(Self(event))
            }

            pub(crate) fn record(&self, device: &Device) -> candle_core::Result<()> {
                let Device::Cuda(dev) = device else {
                    unreachable!("caller checks is_cuda");
                };
                self.0.record(&dev.cuda_stream()).map_err(candle_core::Error::wrap)
            }

            pub(crate) fn is_complete(&self) -> bool {
                // cudarc's `is_complete` returns `bool` (false on query errors too).
                self.0.is_complete()
            }
        }
    }

    #[cfg(feature = "cuda")]
    use imp::Event;

    struct Acc {
        last: Option<(Instant, Option<Event>)>,
        busy_ns: u128,
        dry_ns: u128,
        wait_ns: u128,
        n: usize,
    }

    impl Acc {
        const EMPTY: Acc = Acc { last: None, busy_ns: 0, dry_ns: 0, wait_ns: 0, n: 0 };
    }

    static ACC: Mutex<Acc> = Mutex::new(Acc::EMPTY);

    // Per-round wall time (round = one pass, marked at layer 0), because with speculative decoding a round
    // yields more than one token and TPOT stops describing it.
    static ROUND: Mutex<(Option<Instant>, Vec<f64>)> = Mutex::new((None, Vec::new()));

    pub(crate) fn mark_round(layer: usize) {
        if layer != 0 {
            return;
        }
        super::round_stats::finish_pass(
            std::env::var_os("NAIWA_ROUND_STATS").is_some(),
        );
        let mut g = ROUND.lock().unwrap();
        let now = Instant::now();
        if let Some(prev) = g.0.replace(now) {
            let ms = now.saturating_duration_since(prev).as_secs_f64() * 1e3;
            // A round is tens of ms here (speculation makes it longer), so only filter real gaps:
            // between requests the sampler and the bench take hundreds of ms, and a round is never < 5 ms.
            if (5.0..500.0).contains(&ms) {
                g.1.push(ms);
                if g.1.len() == 8 {
                    let mut v = g.1.clone();
                    v.sort_by(|a, b| a.total_cmp(b));
                    eprintln!(
                        "ROUND_DBG n={} min {:.2} p50 {:.2} max {:.2} mean {:.2}",
                        v.len(),
                        v[0],
                        v[v.len() / 2],
                        v[v.len() - 1],
                        v.iter().sum::<f64>() / v.len() as f64
                    );
                    g.1.clear();
                }
            }
        }
    }

    pub(crate) fn mark(device: &Device, layer: usize) {
        mark_round(layer);
        let on = *ON.get_or_init(|| std::env::var_os("NAIWA_STEP_DBG").is_some());
        if !on || !device.is_cuda() {
            return;
        }
        let mut acc = ACC.lock().unwrap();
        if let Some((t_last, ev_last)) = acc.last.take() {
            let now = Instant::now();
            // The wall time this layer entry arrived after the previous one; the GPU was
            // BUSY for the part before the previous entry's work completed, DRY after.
            let wall = now.duration_since(t_last).as_nanos();
            match ev_last {
                Some(ev) if ev.is_complete() => acc.dry_ns += wall,
                Some(_) => acc.busy_ns += wall,
                None => acc.wait_ns += wall,
            }
            acc.n += 1;
        }
        let recorded = Event::new(device)
            .and_then(|ev| ev.record(device).map(|_| ev))
            .ok();
        acc.last = Some((Instant::now(), recorded));
        if acc.n >= WINDOW {
            let a = &mut *acc;
            let ms = |ns: u128| ns as f64 / 1e6;
            eprintln!(
                "STEP_DBG layers {} busy_ms/layer {:.3} dry_ms/layer {:.3} (no-event {:.3})",
                a.n,
                ms(a.busy_ns) / a.n as f64,
                ms(a.dry_ns) / a.n as f64,
                ms(a.wait_ns) / a.n as f64,
            );
            *a = Acc::EMPTY;
        }
    }
}
