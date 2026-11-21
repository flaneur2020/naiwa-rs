//! Process-wide instrumentation: stable counters and the `NAIWA_*` measurement probes.
//!
//! Each submodule is one concern and owns exactly one `*Stats` struct plus a private static:
//!
//! - `cache` - the expert-cache counters the benchmark prints.
//! - `lookahead` / `round` / `step` - the expert cache's per-step, per-round and per-layer probes.
//! - `fused` / `retired` - the CPU MoE path: fused-decode cost and result-handoff latency.
//! - `predict` - how well the lookahead's one-layer-ahead prediction matches the real routing.
//!
//! The modules are private; their `record_*` and `dump_*` functions are re-exported here so call
//! sites use `crate::stats::record_*` / `crate::stats::dump_*`. Probes keep their accumulator and
//! output format in their own file; only the repeated plumbing is shared below: the one-shot
//! environment gate, the reporting window, the once-per-process header, and the unit conversions.

mod budget;
mod cache;
mod fused;
mod lookahead;
mod predict;
mod retired;
mod round;
mod step;
pub(crate) mod schedule;

pub(crate) mod expert_hist;
pub(crate) use expert_hist::record_demand as record_expert_demand;
pub(crate) use budget::{add as record_budget, record_pairs as record_budget_pairs, enabled as budget_enabled, step_boundary as record_step_boundary, Phase as BudgetPhase};
#[cfg(feature = "cuda")]
pub(crate) use budget::dev as budget_dev;
pub(crate) use cache::{
    dump_cache_stats, record_cache_hits, record_promotions, record_cache_installed, record_cache_lookahead_dropped,
    record_cache_lookahead_load, record_cache_misses, record_cache_pcie, record_cache_result_stall,
    record_cache_staged, record_cache_swap, record_cache_sync_wait,
};
pub(crate) use fused::{dump_fused_stats, record_fused_decode};
pub(crate) use lookahead::{
    dump_lookahead_stats, record_lookahead_job, record_lookahead_publish, record_lookahead_step,
};
pub(crate) use predict::{
    dump_predict_stats, enabled as predict_enabled, record_actual, record_predicted,
};
pub(crate) use retired::{dump_retired_stats, record_retired};
pub(crate) use round::{
    dump_round_stats, record_round_cpu_compute, record_round_pass, record_round_plan,
    record_round_staged_bytes,
};
pub(crate) use step::{dump_step_stats, record_step_layer};

// The `*Stats` types are part of the API: every `dump_*` returns one. Most callers use them by
// inference, so the names are not always referenced by path; keep the re-export anyway.
#[allow(unused_imports)]
pub(crate) use self::{
    cache::CacheStats, fused::FusedStats, lookahead::LookaheadStats, predict::PredStats,
    retired::RetiredStats, round::RoundStats, step::StepStats,
};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

/// One probe's environment gate, backed by [`crate::config::diag`].
struct Probe {
    flag: fn(&crate::config::DiagConfig) -> bool,
    on: OnceLock<bool>,
}

impl Probe {
    const fn new(flag: fn(&crate::config::DiagConfig) -> bool) -> Self {
        Self {
            flag,
            on: OnceLock::new(),
        }
    }

    /// Is the probe on? `extra` forces it on for a case that must always be reported even
    /// without the variable (the lookahead probe reports the no-lookahead path this way).
    #[inline]
    fn enabled(&self, extra: bool) -> bool {
        *self
            .on
            .get_or_init(|| extra || (self.flag)(crate::config::diag()))
    }
}

/// A fixed reporting cadence. `tick` returns the event's position in the current window
/// (`1..=size`); `pos == size` closes it. Unlike a reset-on-flush counter this stays correct
/// when several threads report into the same window.
struct Window {
    size: u64,
    n: AtomicU64,
}

impl Window {
    const fn new(size: u64) -> Self {
        Self {
            size,
            n: AtomicU64::new(0),
        }
    }

    #[inline]
    fn tick(&self) -> u64 {
        let pos = (self.n.fetch_add(1, Ordering::Relaxed) + 1) % self.size;
        if pos == 0 {
            self.size
        } else {
            pos
        }
    }

    const fn size(&self) -> u64 {
        self.size
    }
}

/// Print a probe's column header once per process.
fn header_once(flag: &AtomicBool, text: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        eprintln!("{text}");
    }
}

#[inline]
fn ns_to_us(ns: u64) -> f64 {
    ns as f64 / 1e3
}

#[inline]
fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

#[inline]
fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(feature = "cuda")]
pub(crate) mod gpu_partition;
#[cfg(feature = "cuda")]
pub(crate) mod model_partition;
#[cfg(feature = "cuda")]
pub(crate) mod head_partition;
