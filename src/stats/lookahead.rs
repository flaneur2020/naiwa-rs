//! `NAIWA_LOOKAHEAD_DBG=1`: the lookahead pipeline's cost per decode step.
//!
//! `--no-lookahead` skips the prediction (and its D2H) entirely, so the cache only ever holds
//! what prefill left; the D2H line still reports in that mode, which is the A/B baseline.

use std::sync::Mutex;

use super::{Probe, Window};

#[derive(Default)]
pub(crate) struct LookaheadStats {
    pub publishes: u64,
    pub publish_ms: f64,
    pub jobs: u64,
    pub job_ms: f64,
    pub worst_job_ms: f64,
    pub steps: u64,
    pub d2h_ms: f64,
}

static LOOKAHEAD_STATS: Mutex<LookaheadStats> = Mutex::new(LookaheadStats {
    publishes: 0,
    publish_ms: 0.0,
    jobs: 0,
    job_ms: 0.0,
    worst_job_ms: 0.0,
    steps: 0,
    d2h_ms: 0.0,
});
static PROBE: Probe = Probe::new("NAIWA_LOOKAHEAD_DBG");
static WINDOW: Window = Window::new(384); // eight layer passes

/// Token-thread cost of queueing the publish (the `publish` call, host side).
pub(crate) fn record_lookahead_publish(ms: f64) {
    if !PROBE.enabled(false) {
        return;
    }
    let mut s = LOOKAHEAD_STATS.lock().unwrap();
    s.publishes += 1;
    s.publish_ms += ms;
}

/// Worker-side job duration: `wait_publish` latency plus the CPU matmul.
pub(crate) fn record_lookahead_job(ms: f64) {
    if !PROBE.enabled(false) {
        return;
    }
    let mut s = LOOKAHEAD_STATS.lock().unwrap();
    s.jobs += 1;
    s.job_ms += ms;
    if ms > s.worst_job_ms {
        s.worst_job_ms = ms;
    }
}

/// Per-step D2H of the activation the lookahead needs, in ms. `queued` is false when the
/// prediction is disabled, which keeps this line on even without the environment variable.
/// This is also the window's clock: publishes and jobs are reported with the step they belong to.
pub(crate) fn record_lookahead_step(d2h_ms: f64, queued: bool) {
    if !PROBE.enabled(!queued) {
        return;
    }
    let due = {
        let mut s = LOOKAHEAD_STATS.lock().unwrap();
        s.steps += 1;
        s.d2h_ms += d2h_ms;
        WINDOW.tick() == WINDOW.size()
    };
    if due {
        let _ = super::dump_lookahead_stats();
    }
}

/// Take and reset the window, print one combined line, and return it.
pub(crate) fn dump_lookahead_stats() -> LookaheadStats {
    let out = std::mem::take(&mut *LOOKAHEAD_STATS.lock().unwrap());
    let publishes = out.publishes.max(1) as f64;
    let jobs = out.jobs.max(1) as f64;
    let steps = out.steps.max(1) as f64;
    eprintln!(
        "LOOKAHEAD_DBG publishes {} publish_ms/step {:.3} jobs {} job_ms/step {:.3} worst {} ms steps {} d2h_ms/step {:.3}",
        out.publishes,
        out.publish_ms / publishes,
        out.jobs,
        out.job_ms / jobs,
        out.worst_job_ms,
        out.steps,
        out.d2h_ms / steps,
    );
    out
}
