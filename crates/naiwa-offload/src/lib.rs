//! `naiwa-offload`: edge heterogeneous MoE offloading engine.
//!
//! Strict 3-layer unidirectional DAG (+ cross-cutting `config` / `stats`):
//! - `transport` (L1): hardware async primitives (copy stream, mapped doorbells, CPU executor, spin pool).
//! - `memory`    (L2): host pinned stacks, VRAM slab/borrow gate, per-layer cache directory, staging rings.
//! - `pipeline`  (L3): mutually-exclusive Decode & Prefill execution pipelines + MoE provider factory.

pub mod config;
pub mod memory;
pub mod pipeline;
pub mod stats;
pub mod transport;

/// `NAIWA_STREAM_DBG=1`: the prefill stream's per-layer DMA wait against its compute, so a slow
/// prefill chunk can be attributed instead of guessed at.
pub(crate) fn dbg_note(site: &str, msg: &str) {
    if config::diag().stream_dbg {
        eprintln!("STREAM_DBG {site}: {msg}");
    }
}
