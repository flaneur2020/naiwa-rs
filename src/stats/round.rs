//! `NAIWA_ROUND_STATS`: per-pass accounting of the three MoE producers - resident hits, staged
//! experts (PCIe) and CPU-computed misses - plus the CPU gemv time the deferred worker measured.
//! Also owns the always-on `ROUND_DBG` wall-time window (one round = one pass at layer 0).

use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::time::Instant;

use super::{bytes_to_mib, header_once, ns_to_ms, Probe, Window};

#[derive(Default)]
pub(crate) struct RoundStats {
    /// Cumulative passes seen; the `rounds` column of the line.
    pub layers: u64,
    pub hits: u64,
    pub staged: u64,
    pub cpu_misses: u64,
    pub staged_bytes: u64,
    pub cpu_ns: u64,
    /// Last layer-0 entry and the current eight-round window, for `ROUND_DBG`.
    pub last_round: Option<Instant>,
    pub round_times: Vec<f64>,
}

static ROUND_STATS: Mutex<RoundStats> = Mutex::new(RoundStats {
    layers: 0,
    hits: 0,
    staged: 0,
    cpu_misses: 0,
    staged_bytes: 0,
    cpu_ns: 0,
    last_round: None,
    round_times: Vec::new(),
});
static PROBE: Probe = Probe::new("NAIWA_ROUND_STATS");
static WINDOW: Window = Window::new(4); // passes
static HEADER: AtomicBool = AtomicBool::new(false);

pub(crate) fn record_round_plan(hits: usize, staged: usize, cpu: usize) {
    let mut s = ROUND_STATS.lock().unwrap();
    s.hits += hits as u64;
    s.staged += staged as u64;
    s.cpu_misses += cpu as u64;
}

/// Called by the staging path: how many bytes this layer copied over PCIe.
pub(crate) fn record_round_staged_bytes(bytes: u64) {
    ROUND_STATS.lock().unwrap().staged_bytes += bytes;
}

/// Called by the CPU job when it finishes: the gemv time of one job.
pub(crate) fn record_round_cpu_compute(ns: u64) {
    ROUND_STATS.lock().unwrap().cpu_ns += ns;
}

/// Called once per pass (layer 0): advances the round clock and, every four passes, dumps the
/// producer accounting when `NAIWA_ROUND_STATS` is set.
pub(crate) fn record_round_pass(layer: usize) {
    if layer != 0 {
        return;
    }
    let due = {
        let mut s = ROUND_STATS.lock().unwrap();
        s.layers += 1;
        let now = Instant::now();
        if let Some(prev) = s.last_round.replace(now) {
            let ms = now.saturating_duration_since(prev).as_secs_f64() * 1e3;
            // A round is tens of ms here (speculation makes it longer), so only filter real gaps:
            // between requests the sampler and the bench take hundreds of ms, and a round is never < 5 ms.
            if (5.0..500.0).contains(&ms) {
                s.round_times.push(ms);
                if s.round_times.len() == 8 {
                    let mut v = s.round_times.clone();
                    v.sort_by(|a, b| a.total_cmp(b));
                    eprintln!(
                        "ROUND_DBG n={} min {:.2} p50 {:.2} max {:.2} mean {:.2}",
                        v.len(),
                        v[0],
                        v[v.len() / 2],
                        v[v.len() - 1],
                        v.iter().sum::<f64>() / v.len() as f64
                    );
                    s.round_times.clear();
                }
            }
        }
        PROBE.enabled(false) && WINDOW.tick() == WINDOW.size()
    };
    if due {
        let _ = super::dump_round_stats();
    }
}

/// Take and reset the producer counters, print the `ROUND_STATS` line, and return the snapshot.
/// The cumulative layer count and the round-time window survive the reset.
pub(crate) fn dump_round_stats() -> RoundStats {
    let mut s = ROUND_STATS.lock().unwrap();
    let layers = s.layers;
    let last_round = s.last_round;
    let round_times = std::mem::take(&mut s.round_times);
    let mut out = std::mem::take(&mut *s);
    s.layers = layers;
    s.last_round = last_round;
    s.round_times = round_times.clone();
    out.layers = layers;
    out.round_times = round_times;
    let sel = out.hits + out.staged + out.cpu_misses;
    if sel == 0 {
        return out;
    }
    header_once(
        &HEADER,
        "ROUND_STATS rounds hit% staged% cpu% staged_MiB cpu_ms/stage",
    );
    eprintln!(
        "ROUND_STATS {layers} {:.1} {:.1} {:.1} {:.1} {:.2}",
        100.0 * out.hits as f64 / sel as f64,
        100.0 * out.staged as f64 / sel as f64,
        100.0 * out.cpu_misses as f64 / sel as f64,
        bytes_to_mib(out.staged_bytes),
        ns_to_ms(out.cpu_ns),
    );
    out
}
