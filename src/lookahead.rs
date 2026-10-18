//! The lookahead/prefetch scheduler: **one** dedicated thread, woken by events.
//!
//! A prediction is advisory — decode never waits on one — so the only thing that matters is that it
//! *starts* before the layer it predicts runs. What it does is small and mostly asynchronous: wait
//! for the activation the token thread published on the GPU stream, multiply by the router, then
//! issue the prefetch copies on the copy stream. None of that needs concurrency, so there is no
//! team: a team of threads is just cores the expert batch (rayon) does not get.
//!
//! Two events matter, and both wake the thread:
//!
//! - **A prediction was submitted** (once per layer, from the token thread).
//! - **A CPU-expert batch finished.** A prediction's router matmul and slot uploads must run in the
//!   *gaps* between batches, not beside them (W106: beside them, 10 ms/step slower — the matmul is
//!   2–3 cores and the batch wants every core). Polling for that gap from inside the job burned
//!   a core; the batch's completion *is* the signal, so the batch reports it
//!   ([`crate::deferred::BatchBusy`] → [`wake`]).
//!
//! So the thread parks, and on every wake it drains what it can — but only while no batch is
//! running. A job that arrives during a batch waits for that batch's completion wake instead of
//! spinning; nothing is ever lost, because the queue is the state and the wake is just an edge.
//!
//! The queue is bounded and **drops on overflow**: a missed prediction only costs a CPU expert, so
//! dropping is always better than blocking the token thread. Job bodies also drop stale entries (a
//! layer that has since queued a newer prediction), which is why a deep queue is free.
//!
//! Configuration is fixed here, by commit, not by an environment variable (`architecture-v2` §6.4).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// A queued prediction.
pub type Thunk = Box<dyn FnOnce() + Send>;

/// Safety net only: every real wake comes from a submit or a batch completion, so a lost unpark
/// costs at most this much.
const PARK: Duration = Duration::from_millis(2);

/// Queue depth: one step's worth of layer jobs (48) plus headroom. Overflow drops.
const DEPTH: usize = 64;

/// Pending predictions. `len` mirrors `jobs.len()` so the drain loop can skip the mutex.
static JOBS: Mutex<VecDeque<Thunk>> = Mutex::new(VecDeque::new());
static LEN: AtomicUsize = AtomicUsize::new(0);

/// The one thread, once it exists.
static THREAD: OnceLock<std::thread::Thread> = OnceLock::new();

/// Wake the lookahead thread: there may be work (a submit) or permission to run it (a batch ended).
pub fn wake() {
    if let Some(t) = THREAD.get() {
        t.unpark();
    }
}

/// Queue a prediction and wake the thread. Never blocks the token thread: on overflow the job is
/// dropped.
pub fn submit(job: Thunk) {
    THREAD.get_or_init(|| {
        std::thread::Builder::new()
            .name("lookahead".into())
            .spawn(worker)
            .map(|h| h.thread().clone())
            .unwrap_or_else(|e| {
                tracing::warn!("no lookahead thread ({e}); predictions will not run");
                std::thread::current()
            })
    });
    {
        let mut jobs = JOBS.lock().unwrap();
        if jobs.len() >= DEPTH {
            crate::stats::record_cache_lookahead_dropped();
            return;
        }
        jobs.push_back(job);
        LEN.store(jobs.len(), Ordering::Release);
    }
    wake();
}

fn take() -> Option<Thunk> {
    if LEN.load(Ordering::Acquire) == 0 {
        return None;
    }
    let mut jobs = JOBS.lock().unwrap();
    let job = jobs.pop_front();
    LEN.store(jobs.len(), Ordering::Release);
    job
}

/// Drain the queue while no expert batch is running, then park until the next event.
fn worker() {
    loop {
        // The gap check is re-tested before every job: a batch can start mid-drain, and its
        // completion will wake us again.
        while crate::deferred::expert_batch_idle() {
            let Some(job) = take() else { break };
            job();
        }
        std::thread::park_timeout(PARK);
    }
}
