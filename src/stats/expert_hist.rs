//! `NAIWA_EXPERT_HIST=<path>`: the routing demand table, for M27's allocation study.
//!
//! Counts every `(layer, expert)` routing decision of a decode step and rewrites `<path>` every 32 layers'
//! worth of steps. It exists to answer one question with a single run: is 26 slots per layer - an assumption
//! nobody ever measured - costing misses, compared with the same 1 248 slots handed out where the demand is?
//!
//! The file is one header line and then `<layer> <expert> <count>` lines, so the analysis is a `sort`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

const LAYERS: usize = 48;
const EXPERTS: usize = 256;
const DUMP_EVERY: u64 = 2048;

static HIST: OnceLock<Vec<AtomicU64>> = OnceLock::new();
static CALLS: AtomicU64 = AtomicU64::new(0);

fn path() -> Option<String> {
    std::env::var("NAIWA_EXPERT_HIST").ok()
}

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| path().is_some())
}

fn table() -> &'static Vec<AtomicU64> {
    HIST.get_or_init(|| (0..LAYERS * EXPERTS).map(|_| AtomicU64::new(0)).collect())
}

/// One layer's routing decisions: `ids` is the step's `[rows, topk]` expert indices, flattened.
pub(crate) fn record_demand(layer: usize, ids: &[u32]) {
    if !on() || layer >= LAYERS {
        return;
    }
    let table = table();
    for &id in ids {
        let id = id as usize;
        if id < EXPERTS {
            table[layer * EXPERTS + id].fetch_add(1, Ordering::Relaxed);
        }
    }
    // Dump on a step boundary, from the token thread: a few ms of file I/O every DUMP_EVERY steps, and the
    // numbers are there even if the harness kills the server rather than letting it exit.
    if CALLS.fetch_add(1, Ordering::Relaxed) % DUMP_EVERY == 0 {
        dump();
    }
}

/// Rewrite the table. Called on the dump cadence and again on a clean shutdown.
pub(crate) fn dump() {
    let Some(path) = path() else {
        return;
    };
    let table = table();
    let mut out = String::with_capacity(LAYERS * EXPERTS * 12);
    out.push_str("# layer expert count\n");
    for l in 0..LAYERS {
        for e in 0..EXPERTS {
            let n = table[l * EXPERTS + e].load(Ordering::Relaxed);
            if n > 0 {
                out.push_str(&format!("{l} {e} {n}\n"));
            }
        }
    }
    let _ = std::fs::write(&path, out);
}
