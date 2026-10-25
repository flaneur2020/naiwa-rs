//! A decode step's time budget, by phase (`NAIWA_STEP_BUDGET=1`).
//!
//! TPOT has been measured as one number for five milestones, and every candidate mechanism was argued from that
//! total. This splits one step into the phases a mechanism could change, accumulates them over the step's layers,
//! and prints one line per step:
//!
//! ```text
//! STEP_BUDGET layers 48 total 39.8 ms | plan 1.2 spawn 0.3 preload_wait 0.1 gpu 18.4 staging_wait 1.1 cpu_join 18.2
//!                       fence 0.2 | waits>200us: preload 0/48 staging 6/48 | misses 512 hits 380 staged 96
//! ```
//!
//! `total` is the wall time of the step's layer calls; `cpu_join` is where the token thread waits for the CPU experts'
//! result ring, which is the phase a miss shows up in. Attribution is by construction: the phases are the arguments
//! of one `CachedLayer::forward`, so their sum is the total minus the caller's own work.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum Phase {
    /// Routing the step's experts: the doorbell read and the plan (hits / staged / CPU).
    Plan = 0,
    /// Building and submitting the CPU expert job.
    Spawn = 1,
    /// Waiting for the lookahead's preload claim (a prediction's copy).
    PreloadWait = 2,
    /// The resident (and then the staged) GPU expert kernels, queued.
    Gpu = 3,
    /// Waiting for the staging DMA's completion event.
    StagingWait = 4,
    /// Waiting for the CPU experts' result ring.
    CpuJoin = 5,
    /// Recording read fences for the slots the step used, and the promotion copies.
    Fence = 6,
    /// Everything else in the layer call (reshape, combine, dtype).
    Other = 7,
}

const PHASES: usize = 8;

static NS: [AtomicU64; PHASES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static SLOW_PRELOAD: AtomicU64 = AtomicU64::new(0);
static SLOW_STAGING: AtomicU64 = AtomicU64::new(0);
static SLOW_CPU: AtomicU64 = AtomicU64::new(0);

/// The threshold above which a layer's wait counts as a stub rather than scheduling noise.
const SLOW_NS: u64 = 200_000;

fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NAIWA_STEP_BUDGET").is_some())
}

pub(crate) fn enabled() -> bool {
    on()
}

pub(crate) fn add(phase: Phase, ns: u64) {
    if !on() {
        return;
    }
    NS[phase as usize].fetch_add(ns, Ordering::Relaxed);
    match phase {
        Phase::PreloadWait if ns > SLOW_NS => {
            SLOW_PRELOAD.fetch_add(1, Ordering::Relaxed);
        }
        Phase::StagingWait if ns > SLOW_NS => {
            SLOW_STAGING.fetch_add(1, Ordering::Relaxed);
        }
        Phase::CpuJoin if ns > SLOW_NS => {
            SLOW_CPU.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

/// Called at the start of a step (layer 0's call): print the previous step's totals and reset.
pub(crate) fn step_boundary(layers: usize) {
    if !on() {
        return;
    }
    // Read, then clear: swapping the accumulator to zero and reading it again (the first version did) prints zeros.
    let ns: [u64; PHASES] = std::array::from_fn(|i| NS[i].load(Ordering::Relaxed));
    let total: u64 = ns.iter().sum();
    for v in NS.iter() {
        v.store(0, Ordering::Relaxed);
    }
    let slow = (
        SLOW_PRELOAD.swap(0, Ordering::Relaxed),
        SLOW_STAGING.swap(0, Ordering::Relaxed),
        SLOW_CPU.swap(0, Ordering::Relaxed),
    );
    if total == 0 {
        return;
    }
    // The previous step's counters are read before this step's begin: `swap` above cleared them, so the print is
    // one step behind, which is fine - the phases were accumulated over exactly those layers.
    let ms = |i: usize| ns[i] as f64 / 1e6;
    eprintln!(
        "STEP_BUDGET layers {layers} total {:.1} ms | plan {:.1} spawn {:.1} preload_wait {:.1} gpu {:.1} \
         staging_wait {:.1} cpu_join {:.1} fence {:.1} other {:.1} | waits>200us preload {}/{} staging {}/{} \
         cpu {}/{}",
        total as f64 / 1e6,
        ms(Phase::Plan as usize),
        ms(Phase::Spawn as usize),
        ms(Phase::PreloadWait as usize),
        ms(Phase::Gpu as usize),
        ms(Phase::StagingWait as usize),
        ms(Phase::CpuJoin as usize),
        ms(Phase::Fence as usize),
        ms(Phase::Other as usize),
        slow.0,
        layers,
        slow.1,
        layers,
        slow.2,
        layers,
    );
}
