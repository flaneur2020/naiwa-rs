//! `NAIWA_FUSED_DBG=1`: wall time of the fused CPU decode per call, per (expert, row) pair and
//! per distinct expert, plus the idle gap between calls.

use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::time::Instant;

use super::{header_once, ns_to_us, Probe, Window};

#[derive(Default)]
pub(crate) struct FusedStats {
    pub calls: u64,
    pub busy_ns: u64,
    pub gap_ns: u64,
    pub pairs: u64,
    pub experts: u64,
    pub rows: u64,
    /// Kept across dumps so the inter-call gap is continuous.
    pub last_end: Option<Instant>,
}

static FUSED_STATS: Mutex<FusedStats> = Mutex::new(FusedStats {
    calls: 0,
    busy_ns: 0,
    gap_ns: 0,
    pairs: 0,
    experts: 0,
    rows: 0,
    last_end: None,
});
static PROBE: Probe = Probe::new("NAIWA_FUSED_DBG");
static WINDOW: Window = Window::new(480);
static HEADER: AtomicBool = AtomicBool::new(false);

pub(crate) fn record_fused_decode(t0: Instant, ids: &[u32], weights: &[f32], topk: usize) {
    if !PROBE.enabled(false) {
        return;
    }
    let end = Instant::now();
    let mut pairs: Vec<(u32, usize)> = ids
        .iter()
        .zip(weights)
        .enumerate()
        .filter(|(_, (_, &w))| w != 0.0)
        .map(|(i, (&e, _))| (e, i / topk.max(1)))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    let mut experts: Vec<u32> = pairs.iter().map(|p| p.0).collect();
    experts.dedup();
    let due = {
        let mut s = FUSED_STATS.lock().unwrap();
        if let Some(last) = s.last_end {
            // Gaps over 50 ms are between steps (sampling, the bench's own work), not layers.
            let gap = t0.saturating_duration_since(last).as_nanos() as u64;
            if gap < 50_000_000 {
                s.gap_ns += gap;
            }
        }
        s.last_end = Some(end);
        s.calls += 1;
        s.busy_ns += (end - t0).as_nanos() as u64;
        s.pairs += pairs.len() as u64;
        s.experts += experts.len() as u64;
        s.rows += (ids.len() / topk.max(1)) as u64;
        WINDOW.tick() == WINDOW.size()
    };
    if due {
        let _ = super::dump_fused_stats();
    }
}

/// Take and reset the window, print the `FUSED_DBG` line, and return the snapshot. `last_end`
/// survives the reset so the gap measurement continues across windows.
pub(crate) fn dump_fused_stats() -> FusedStats {
    let mut s = FUSED_STATS.lock().unwrap();
    let last_end = s.last_end;
    let mut out = std::mem::take(&mut *s);
    s.last_end = last_end;
    out.last_end = last_end;
    header_once(
        &HEADER,
        "FUSED_DBG calls rows/call pairs/call experts/call us/call us/pair us/expert gap_us/call",
    );
    let c = out.calls.max(1) as f64;
    let us = ns_to_us(out.busy_ns);
    eprintln!(
        "FUSED_DBG {} {:.2} {:.2} {:.2} {:.1} {:.1} {:.1} {:.1}",
        out.calls,
        out.rows as f64 / c,
        out.pairs as f64 / c,
        out.experts as f64 / c,
        us / c,
        us / out.pairs.max(1) as f64,
        us / out.experts.max(1) as f64,
        ns_to_us(out.gap_ns) / c,
    );
    out
}
