//! `NAIWA_RET_DBG`: how long a miss expert's return takes, split into "waiting for the worker"
//! (queue) and "the gemv" (compute). The GPU blocks on the publish point for exactly this long,
//! so it is the number to shrink.

use std::sync::Mutex;
use std::time::Instant;

use super::{record_round_cpu_compute, Probe, Window};

#[derive(Default)]
pub(crate) struct RetiredStats {
    pub jobs: u64,
    pub queue_ns: u64,
    pub compute_ns: u64,
}

static RETIRED_STATS: Mutex<RetiredStats> = Mutex::new(RetiredStats {
    jobs: 0,
    queue_ns: 0,
    compute_ns: 0,
});
static PROBE: Probe = Probe::new("NAIWA_RET_DBG");
static WINDOW: Window = Window::new(48);

pub(crate) fn record_retired(t_submit: Instant, t_start: Instant) {
    let done = Instant::now();
    // Always account the gemv time, so `--round-stats` sees every job whatever path it took.
    // This has to come before the probe's own early return.
    record_round_cpu_compute(done.saturating_duration_since(t_start).as_nanos() as u64);
    if !PROBE.enabled(false) {
        return;
    }
    let due = {
        let mut s = RETIRED_STATS.lock().unwrap();
        s.jobs += 1;
        s.queue_ns += t_start.saturating_duration_since(t_submit).as_nanos() as u64;
        s.compute_ns += done.saturating_duration_since(t_start).as_nanos() as u64;
        WINDOW.tick() == WINDOW.size()
    };
    if due {
        let _ = super::dump_retired_stats();
    }
}

/// Take and reset the window, print the `RET_DBG` line, and return the snapshot.
pub(crate) fn dump_retired_stats() -> RetiredStats {
    let out = std::mem::take(&mut *RETIRED_STATS.lock().unwrap());
    let n = out.jobs.max(1) as f64;
    let q = out.queue_ns as f64 / n / 1e3;
    let c = out.compute_ns as f64 / n / 1e3;
    eprintln!(
        "RET_DBG queue_us {q:.1} compute_us {c:.1} total_us {:.1}",
        q + c
    );
    out
}
