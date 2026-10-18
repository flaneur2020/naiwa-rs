//! Stable expert-cache counters, reported by `bench` through `dump_cache_stats`.

use std::sync::Mutex;

/// Expert-cache counters since the last dump.
#[derive(Default)]
pub(crate) struct CacheStats {
    /// One expert cache was installed in this process.
    pub installed: bool,
    /// Routed experts of decode steps served from a slot.
    pub hits: u64,
    /// Routed experts of decode steps run on the CPU.
    pub misses: u64,
    /// Experts served from the staging ring over PCIe: neither a cache hit nor a CPU miss.
    pub staged: u64,
    /// Experts the lookahead loaded into empty slots.
    pub lookahead_loads: u64,
    /// Experts the lookahead loaded over a least-recently-used resident.
    pub swaps: u64,
    /// Lookahead requests dropped because the worker was busy.
    pub lookahead_dropped: u64,
    /// v2 M4: layer DMA gates replayed (one per layer per pass, when PCIe staging is on).
    pub gate_replays: u64,
    /// v2 M4: DMA-gate waits launched warm because the gate graph could not be captured. Must stay
    /// 0 — a non-zero value means the graph route failed and only the fallback ran.
    pub staging_eager_waits: u64,
    /// Experts DMA'd into a scratch slot and run on the GPU instead of the CPU (`--pcie-frac`).
    pub pcie_misses: u64,
    /// Bytes those DMAs moved host-to-device.
    pub pcie_bytes: u64,
}

static CACHE_STATS: Mutex<CacheStats> = Mutex::new(CacheStats {
    installed: false,
    hits: 0,
    misses: 0,
    staged: 0,
    lookahead_loads: 0,
    swaps: 0,
    lookahead_dropped: 0,
    gate_replays: 0,
    staging_eager_waits: 0,
    pcie_misses: 0,
    pcie_bytes: 0,
});

/// One expert cache was installed in this process.
pub(crate) fn record_cache_installed() {
    CACHE_STATS.lock().unwrap().installed = true;
}

/// Routed experts of decode steps served from a slot.
pub(crate) fn record_cache_hits(n: usize) {
    CACHE_STATS.lock().unwrap().hits += n as u64;
}

/// Routed experts of decode steps run on the CPU.
pub(crate) fn record_cache_misses(n: usize) {
    CACHE_STATS.lock().unwrap().misses += n as u64;
}

/// Experts served from the staging ring over PCIe: neither a cache hit nor a CPU miss.
pub(crate) fn record_cache_staged(n: usize) {
    CACHE_STATS.lock().unwrap().staged += n as u64;
}

/// An expert the lookahead loaded into an empty slot.
pub(crate) fn record_cache_lookahead_load() {
    CACHE_STATS.lock().unwrap().lookahead_loads += 1;
}

/// An expert the lookahead loaded over a least-recently-used resident.
pub(crate) fn record_cache_swap() {
    CACHE_STATS.lock().unwrap().swaps += 1;
}

/// `--pcie-frac`: one expert DMA'd into a scratch slot and run on the GPU instead of the CPU.
pub(crate) fn record_cache_pcie(bytes: u64) {
    let mut s = CACHE_STATS.lock().unwrap();
    s.pcie_misses += 1;
    s.pcie_bytes += bytes;
}

/// v2 M4: one layer DMA gate replayed.
pub(crate) fn record_cache_gate_replay() {
    CACHE_STATS.lock().unwrap().gate_replays += 1;
}

pub(crate) fn record_cache_lookahead_dropped() {
    CACHE_STATS.lock().unwrap().lookahead_dropped += 1;
}

/// Take and reset the counters. `installed` is process-level and survives the reset, so a dump
/// after the bench's warmup reset still reports the run's counters.
pub(crate) fn dump_cache_stats() -> CacheStats {
    let mut s = CACHE_STATS.lock().unwrap();
    let installed = s.installed;
    let mut out = std::mem::take(&mut *s);
    out.installed = installed;
    s.installed = installed;
    out
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        self.hits as f64 / (self.hits + self.misses).max(1) as f64
    }
}
