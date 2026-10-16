//! Pinned spin pool for CPU expert computation and decode lookahead (Steps 3–4 of
//! `.plans/spin-pool-doorbell.md`).
//!
//! # Design
//!
//! Each decode layer has two serialised phases:
//!
//! 1. **Expert batch** — fused CPU MoE decode over the miss experts.  The host
//!    publishes a closure; `N` workers claim tasks; the host also participates
//!    (`host_works`); everyone drains to `done == n`.
//! 2. **Lookahead** — one worker runs `prefetch_predicted` for the next layer
//!    while the others spin-wait for the next layer's expert batch.
//!
//! Between phases (and between layers) workers execute `std::hint::spin_loop()`
//! for up to `SPIN_MS` milliseconds before parking, matching strata's
//! `kSpinBeforeSleep{20}`.  A decode layer gap is ~560 µs, so on a normal
//! decode stream workers never reach the park threshold.
//!
//! Workers are pinned to physical cores 1–N (core 0 is left to the host thread
//! that drives the GPU stream and will later spin on the doorbell).  Pinning
//! uses `pthread_setaffinity_np` on Linux.
//!
//! # Relation to `deferred.rs`
//!
//! The staging-buffer mechanism in `deferred.rs` is unchanged: the host still
//! queues a CUDA host function that blocks the stream until the worker signals,
//! then copies from the pinned staging buffer.  The only difference is that the
//! worker signal comes from this pool rather than from the single channel-based
//! `cpu-experts` thread.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Workers spin for this long before parking.  Strata uses 20 ms; a decode layer
/// gap is ~560 µs, so workers never actually park during continuous decode.
const SPIN_MS: u64 = 20;

/// Maximum number of pool workers.
const MAX_WORKERS: usize = 16;

// ---------------------------------------------------------------------------
// Atomic epoch | count | index packing (mirrors strata's head word)
// ---------------------------------------------------------------------------

/// Pack (epoch: u32, njobs: u16, start: u16) into one u64.
/// Workers CAS on this; a late worker whose epoch is stale simply fails.
#[inline]
fn pack(epoch: u32, njobs: u16, start: u16) -> u64 {
    ((epoch as u64) << 32) | ((njobs as u64) << 16) | (start as u64)
}

#[inline]
fn unpack(v: u64) -> (u32, u16, u16) {
    ((v >> 32) as u32, ((v >> 16) & 0xffff) as u16, (v & 0xffff) as u16)
}

// ---------------------------------------------------------------------------
// Job slot
// ---------------------------------------------------------------------------

type Thunk = Box<dyn FnOnce() + Send>;

struct JobSlot {
    /// The closure to run.  Written by the host before publishing; read by the
    /// one worker that claims this slot.  Protected by the epoch protocol.
    inner: UnsafeCell<Option<Thunk>>,
}

// SAFETY: only the host writes and exactly one worker reads, sequenced by the
// epoch CAS.
unsafe impl Sync for JobSlot {}

impl JobSlot {
    fn new() -> Self {
        Self { inner: UnsafeCell::new(None) }
    }

    /// Called by the host before bumping the epoch.
    ///
    /// SAFETY: caller must own the slot (no worker has seen this epoch yet).
    unsafe fn set(&self, f: Thunk) {
        *self.inner.get() = Some(f);
    }

    /// Called by the one worker that wins the CAS for this slot.
    ///
    /// SAFETY: exactly one worker calls this after winning the claim CAS.
    unsafe fn take_and_run(&self) {
        if let Some(f) = (*self.inner.get()).take() {
            f();
        }
    }
}

// ---------------------------------------------------------------------------
// Pool inner state (heap-allocated so it has a stable address)
// ---------------------------------------------------------------------------

struct PoolInner {
    /// epoch | njobs | start packed into one word; workers CAS on it.
    head: AtomicU64,
    /// Incremented by each worker when it finishes its job; host waits for this
    /// to equal the published job count.
    done: AtomicU32,
    /// Set to stop workers.
    stop: AtomicBool,
    /// Per-job slots; indexed 0..njobs in the current batch.
    slots: Box<[JobSlot]>,
    /// Dispatch mode (the fused decode's parallel-for): the current call's
    /// task function as a lifetime-erased fat pointer.  The reference is valid
    /// for the whole `dispatch` call: it is published before the head store,
    /// and `dispatch` waits for every task's `done` before returning, so a
    /// claim under the same epoch can safely rebuild it.
    dispatch_task: UnsafeCell<Option<*const (dyn Fn(usize) + Sync + Send)>>,
    /// Dispatch segment stride: a claimed task `si` covers indices
    /// `si, si+parts, si+2*parts, ...` (round-robin), bounded below by
    /// `dispatch_len`.  Claiming one strided segment per CAS (instead of one
    /// index) takes a 160-240-task fused call down to ~6 claims, one per
    /// worker plus the host (W93: the CAS loop cost more than the cold-wake
    /// it saved).  Round-robin rather than contiguous: fused-decode task ids
    /// are ordered [pair0 chunks | pair1 chunks | ...], so a contiguous range
    /// degenerates into whole pairs — and pairs are the unit of load
    /// imbalance (heterogeneous expert dtypes).  Striding by `parts` spreads
    /// each claim across all pairs, restoring balance (W94 probe: contiguous
    /// segments cost the call +16% and the pair +26% vs per-index draining).
    dispatch_parts: UnsafeCell<usize>,
    /// Dispatch total length `n` of the calling parallel-for.
    dispatch_len: UnsafeCell<usize>,
    /// Whether the current epoch is a dispatch (parallel-for over indices)
    /// rather than a batch of closures.  Published with the task pointer,
    /// before the head store.
    dispatch_on: AtomicBool,
    /// Serialises `run`/`dispatch` calls.  A decode step's CPU expert work is
    /// sequential by construction (one closure at a time on the deferred
    /// worker); this only guards against a second concurrent caller.
    publish_lock: Mutex<()>,
}

// SAFETY: `JobSlot` is `Sync` under the epoch protocol.
unsafe impl Sync for PoolInner {}
unsafe impl Send for PoolInner {}

impl PoolInner {
    fn new(capacity: usize) -> Self {
        Self {
            head: AtomicU64::new(0),
            done: AtomicU32::new(0),
            stop: AtomicBool::new(false),
            slots: (0..capacity).map(|_| JobSlot::new()).collect::<Vec<_>>().into_boxed_slice(),
            dispatch_task: UnsafeCell::new(None),
            dispatch_parts: UnsafeCell::new(1),
            dispatch_len: UnsafeCell::new(0),
            dispatch_on: AtomicBool::new(false),
            publish_lock: Mutex::new(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Expert-batch busy signal (lookahead gating)
// ---------------------------------------------------------------------------

/// In-flight expert batches (`run`/`dispatch`) on the pool.  The lookahead
/// team polls this to keep its predictions truly OFF the critical path.
static EXPERT_BATCH_DEPTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// True while no expert batch is executing — the CPU/PCIe gaps the lookahead
/// predictions may run in.  Always true when the pool is disabled
/// (`NAIWA_SPIN_POOL=0`): no pool batches exist, so nothing is gated.
pub fn expert_batch_idle() -> bool {
    EXPERT_BATCH_DEPTH.load(Ordering::Relaxed) == 0
}

/// RAII marker of one executing expert batch.
struct BatchDepthGuard;

impl BatchDepthGuard {
    fn new() -> Self {
        EXPERT_BATCH_DEPTH.fetch_add(1, Ordering::Release);
        BatchDepthGuard
    }
}

impl Drop for BatchDepthGuard {
    fn drop(&mut self) {
        EXPERT_BATCH_DEPTH.fetch_sub(1, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

fn worker_loop(inner: &PoolInner, spin_ms: Duration) {
    let mut seen_epoch: u32 = 0;
    loop {
        // Phase: spin-wait for a new epoch.
        //
        // W106: workers must NOT touch the lookahead queue. A lookahead job now
        // blocks (it waits for the layer's `publish_f32` kernel, which can sit
        // behind a result-wait kernel on the stream); claimed by a worker, it
        // stalls `drain_epoch` for the whole wait — every expert batch behind it
        // pays 50ms-class stalls per layer, which measured as the 0.2-0.6 T/s
        // runaway. The queue is drained by the dedicated `cpu-lookahead` thread
        // only (`submit_lookahead_job`).
        let start = Instant::now();
        let (epoch, _njobs, _start_idx) = loop {
            let v = inner.head.load(Ordering::Acquire);
            let (epoch, njobs, si) = unpack(v);
            if epoch != seen_epoch {
                break (epoch, njobs, si);
            }
            if inner.stop.load(Ordering::Relaxed) {
                return;
            }
            std::hint::spin_loop();
            if start.elapsed() >= spin_ms {
                // Park briefly, then wake and keep trying.  The park is timed,
                // so we re-read `head` without any unpark (W95).
                std::thread::park_timeout(Duration::from_micros(50));
            }
        };
        seen_epoch = epoch;
        if inner.stop.load(Ordering::Relaxed) {
            return;
        }

        // Phase: drain — claim and run jobs until none remain.
        drain_epoch(inner, epoch);
    }
}

/// Claim and run the remaining tasks/closures of `epoch`.  Used by both the
/// workers (after seeing a new epoch) and a publishing host that participates
/// in the drain; the host's own `done` increments are counted here too.
fn drain_epoch(inner: &PoolInner, epoch: u32) {
    // The mode is published before the head store; read once like the workers.
    let is_dispatch = inner.dispatch_on.load(Ordering::Acquire);
    loop {
        let old = inner.head.load(Ordering::Acquire);
        let (e, nj, si) = unpack(old);
        if e != epoch || si >= nj {
            return;
        }
        let new = pack(e, nj, si + 1);
        if inner.head
            .compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            if is_dispatch {
                // SAFETY: the task function was published before the head
                // store; the current `dispatch` call keeps it alive until
                // every task's `done`.
                // SAFETY: reading the published task function is ordered by
                // the head-load Acquire above; the Option part is always set
                // before the head store of a dispatch epoch.
                let fptr = unsafe { *inner.dispatch_task.get() }
                    .expect("dispatch epoch without a task function");
                // SAFETY: the current `dispatch` call keeps the function alive
                // until every task's `done`.
                let f: &(dyn Fn(usize) + Sync + Send) = unsafe { &*fptr };
                // One round-robin stride per claim: si, si+parts, ...
                // (see the field doc: contiguous ranges degenerate into
                // whole pairs, which are the unit of load imbalance).
                let (parts, len) =
                    unsafe { (*inner.dispatch_parts.get(), *inner.dispatch_len.get()) };
                let mut i = si as usize;
                while i < len {
                    f(i);
                    i += parts;
                }
            } else {
                // SAFETY: the unique winner of this CAS for this slot.
                unsafe { inner.slots[si as usize].take_and_run() };
            }
            inner.done.fetch_add(1, Ordering::Release);
        }
    }
}

/// `&dyn` -> lifetime-erased fat pointer.  The caller guarantees the referent
/// outlives whatever reads it back; the protocol in `ExpertPool::dispatch` is
/// that guarantee.
/// `NAIWA_HOST_WORKS=0`: the publishing thread skips its own claims in
/// `ExpertPool::dispatch` (publish + wait_done only). A/B knob for the W97a
/// host_works-skew lever; default is on (the host participates).
fn host_works_on() -> bool {
    thread_local! {
        static ON: bool = std::env::var_os("NAIWA_HOST_WORKS")
            .map(|v| v != "0")
            .unwrap_or(true);
    }
    ON.with(std::clone::Clone::clone)
}

fn erase_lifetime(f: &(dyn Fn(usize) + Sync + Send)) -> *const (dyn Fn(usize) + Sync + Send) {
    // SAFETY: casting to a raw pointer discards the lifetime; the caller's
    // protocol (see `ExpertPool::dispatch`) keeps the referent alive for the
    // whole epoch, so reintroducing a reference in `drain_epoch` is sound.
    unsafe { std::mem::transmute(f) }
}

/// The function installed into `mistralrs-quant`: the fused decode's two
/// parallel passes call this instead of rayon while the spin pool owns them
/// (Step 3 of `.plans/spin-pool-doorbell.md`).
fn moe_task_dispatch(n: usize, f: &(dyn Fn(usize) + Sync + Send)) {
    let Some(pool) = ExpertPool::global() else {
        // The installer only runs with the pool enabled; this is a broken invariants.
        panic!("moe_task_dispatch called with the spin pool disabled");
    };
    pool.dispatch(n, f);
}

/// Install the fused decode's task dispatch onto the singleton pool.  A no-op
/// when the pool is disabled (`NAIWA_SPIN_POOL=0`): the fork keeps its rayon
/// pool, so an A/B run can move the experts on and off the pool with one env
/// var per arm, from one binary.
pub fn install() {
    if ExpertPool::global().is_some() {
        mistralrs_quant::install_moe_task_dispatcher(Some(moe_task_dispatch));
    }
}

/// Submit a lookahead job to the dedicated thread team (drop-if-full, never
/// blocks decode).
///
/// W106: the job now BLOCKS — it waits for the layer's `publish_f32` kernel on
/// the GPU stream — so it must never run on a spin-pool worker. The pool drains
/// a worker per lookahead job, and a worker stuck in that wait starves the CPU
/// experts batch that the stream's result-wait kernel is spinning for: a
/// structural deadlock that measured as 0.2-0.6 T/s whenever their scheduling
/// interleaved (np2). The dedicated team makes the wait harmless — threads
/// park on a 100µs poll, steal nothing from the experts.
///
/// W106 postscript: the job BODY also gates on `expert_batch_idle` before its
/// matmul/loader work — running the predictions concurrently with the expert
/// batches (2-3 cores of router matmuls plus copy-stream uploads in the same
/// window) measured 10ms/step slower than the W105 pool's accidental
/// run-in-the-gaps behavior. The gate keeps the team's concurrency (hit rate)
/// while restoring the ordering: predictions run in the batches' gaps.
pub fn submit_lookahead_job(job: Thunk) {
    fallback_lookahead().store(job);
    // Wake the parked team members, if any.
    if let Some(ts) = fallback_thread() {
        for t in ts {
            t.unpark();
        }
    }
}

// ---------------------------------------------------------------------------
// Public pool
// ---------------------------------------------------------------------------

/// A pinned spin pool for CPU expert jobs and lookahead prefetch.
///
/// Obtained via [`ExpertPool::global()`]; one pool is shared across all cached
/// layers.  The pool is safe to call from multiple threads; the host is
/// expected to call [`ExpertPool::run`] from exactly one thread per layer
/// (the decode thread).
pub struct ExpertPool {
    inner: Box<PoolInner>,
    n_workers: usize,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl ExpertPool {
    /// Build a pool with `n_workers` workers, pinned one per core with
    /// `first_core` (1) as the first worker core.  SMT siblings come after the
    /// physical cores: on this 6-core/12-thread host worker 0-4 sit on cores
    /// 1-5 and workers 5+ on 7-11, leaving core 0 *and its sibling 6* to the
    /// host, as in strata's layout (core 0 drives the GPU stream and will later
    /// poll the doorbell).
    ///
    /// `n_workers = 0` uses `available_parallelism - 2`, capped at
    /// `MAX_WORKERS`.
    pub fn new(n_workers: usize, first_core: usize) -> Self {
        let n = if n_workers == 0 {
            std::thread::available_parallelism()
                .map_or(4, |n| n.get().saturating_sub(2))
                .min(MAX_WORKERS)
        } else {
            n_workers.min(MAX_WORKERS)
        };

        let capacity = n.max(1);
        let inner = Box::new(PoolInner::new(capacity));
        let inner_ptr = &*inner as *const PoolInner as usize;
        let spin_ms = Duration::from_millis(SPIN_MS);

        let threads: Vec<_> = (0..n)
            .map(|i| {
                let core = worker_core(first_core, i);
                std::thread::Builder::new()
                    .name(format!("cpu-expert-{i}"))
                    .spawn(move || {
                        pin_to_core(core);
                        // SAFETY: `inner` lives for the lifetime of `ExpertPool`
                        // which outlives all threads.
                        let inner = unsafe { &*(inner_ptr as *const PoolInner) };
                        worker_loop(inner, spin_ms);
                    })
                    .expect("spawn expert pool worker")
            })
            .collect();

        Self { inner, n_workers: n, threads }
    }

    /// Number of workers.
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }

    /// Run a batch of closures in parallel, waiting for all to finish before
    /// returning.  The calling thread also participates (host_works), so `n+1`
    /// threads execute the batch.
    ///
    /// `jobs` must have at most `MAX_WORKERS` entries.
    ///
    /// Panics if `jobs.len() > MAX_WORKERS` or the pool is stopping.
    pub fn run(&self, jobs: Vec<Thunk>) {
        let n = jobs.len();
        assert!(n <= MAX_WORKERS, "too many jobs for expert pool");
        if n == 0 {
            return;
        }
        let _guard = self.inner.publish_lock.lock().unwrap();
        let _busy = BatchDepthGuard::new();

        let new_epoch = self.next_epoch();
        // Write jobs into slots before publishing.
        for (i, job) in jobs.into_iter().enumerate() {
            // SAFETY: new_epoch not yet published, no worker has seen it.
            unsafe { self.inner.slots[i].set(job) };
        }
        self.publish(new_epoch, n, false);
        // The calling thread participates in the drain.
        drain_epoch(&self.inner, new_epoch);
        self.wait_done(n);
    }

    /// Run the fused decode-style parallel-for `f(0..n)` on the pool, waiting
    /// for all tasks before returning (the caller — the deferred CPU worker —
    /// also participates).  This is the shape `mistralrs-quant` calls through
    /// [`moe_task_dispatch`].
    pub fn dispatch(&self, n: usize, f: &(dyn Fn(usize) + Sync + Send)) {
        if n == 0 {
            return;
        }
        let _guard = self.inner.publish_lock.lock().unwrap();
        let _busy = BatchDepthGuard::new();

        let new_epoch = self.next_epoch();
        // Publish the task function and the mode before the head store.
        // SAFETY: erasing the lifetime is sound because the protocol keeps
        // the task function alive for exactly as long as the epoch it was
        // published for: `dispatch` waits for every claim's `done` before
        // returning, and the next publish happens only after that.
        // SAFETY: see the comment above the struct field; the protocol
        // keeps the function alive for the whole epoch.
        let fptr = erase_lifetime(f);
        // Stride: workers + the participating host, claiming contiguous
        // One claim per worker plus the host; claim si takes indices
        // si, si+parts, si+2*parts, ... (round-robin, see the field doc).
        // `NAIWA_HOST_WORKS=0` skips the host's own claims (W97a lever 1):
        // the host publishes and waits, the workers take the freed claims —
        // isolates what the host's participation costs the critical path.
        let host_works = host_works_on();
        let parts = (self.n_workers + usize::from(host_works)).min(n);
        unsafe {
            *self.inner.dispatch_task.get() = Some(fptr);
            *self.inner.dispatch_parts.get() = parts;
            *self.inner.dispatch_len.get() = n;
        }
        self.publish(new_epoch, parts, true);
        if host_works {
            drain_epoch(&self.inner, new_epoch);
        }
        // One done per claim; claims = parts.
        self.wait_done(parts);
    }

    /// The next epoch counter, one past the currently published one.
    fn next_epoch(&self) -> u32 {
        let old = self.inner.head.load(Ordering::Acquire);
        let (epoch, _, _) = unpack(old);
        epoch.wrapping_add(1)
    }

    /// Publish an epoch: reset `done`, expose the mode and jobs with the head
    /// store, and wake any parked workers.
    fn publish(&self, epoch: u32, n: usize, is_dispatch: bool) {
        // Reset done counter.
        self.inner.done.store(0, Ordering::Release);
        // Publish the mode together with the jobs, before the head store.
        self.inner.dispatch_on.store(is_dispatch, Ordering::Release);
        // Publish: atomically set epoch, njobs, start=0.  No unpark loop:
        // during decode workers never park (layer gaps ~560µs << SPIN_MS 20ms),
        // and if one does park (long host stall), its park is timed (50µs),
        // so it self-wakes and re-reads `head` on time anyway (W95).
        self.inner.head.store(pack(epoch, n as u16, 0), Ordering::Release);
    }

    /// Wait until the published `n` tasks have all incremented `done`.
    fn wait_done(&self, n: usize) {
        while self.inner.done.load(Ordering::Acquire) < n as u32 {
            std::hint::spin_loop();
        }
    }

    /// The process-wide singleton pool.  Defaults to `available_parallelism - 2`
    /// workers (auto-size).  `NAIWA_SPIN_POOL=0` disables the pool entirely:
    /// lookahead goes back to a dedicated thread (the old `Lookahead`
    /// semantics) and the fused decode keeps the fork's rayon pool — kept for
    /// A/B runs.
    pub fn global() -> &'static Option<Self> {
        static POOL: OnceLock<Option<ExpertPool>> = OnceLock::new();
        POOL.get_or_init(|| {
            if std::env::var_os("NAIWA_SPIN_POOL").as_deref() == Some(std::ffi::OsStr::new("0")) {
                return None;
            }
            let n = std::env::var("NAIWA_POOL_WORKERS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            Some(ExpertPool::new(n, 1))
        })
    }
}

impl Drop for ExpertPool {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::Release);
        // Bump epoch so workers notice the stop flag rather than sleeping.
        let old = self.inner.head.load(Ordering::Acquire);
        let (e, _, _) = unpack(old);
        self.inner.head.store(pack(e.wrapping_add(1), 0, 0), Ordering::Release);
        for t in &self.threads {
            t.thread().unpark();
        }
        // threads are joined on drop via JoinHandle
    }
}

// ---------------------------------------------------------------------------
// Lookahead queue (separate from the main batch so Phase 2 never races with
// the start of the next expert batch)
// ---------------------------------------------------------------------------

/// Bounded FIFO lookahead queue, drained by the dedicated lookahead threads
/// ([`fallback_lazy`]).  A submission when the queue is full drops the job,
/// exactly the old `Lookahead`'s `TrySendError::Full` semantics: the prediction
/// is advisory, decode never waits on it.
///
/// W106 postscript: this used to be one-deep with a single consumer thread.
/// A step queues ~one job per layer (48), each a `wait_publish` plus a CPU
/// router matmul — a serial consumer needs >100ms per step, so every
/// prediction landed after the step had already missed and loaded the experts
/// itself ("loaded 0 into empty slots" with thousands of predictions
/// "over residents"). The queue is now deep enough for a whole step and the
/// consumer is a small thread team.
struct LookaheadQueue {
    jobs: Mutex<std::collections::VecDeque<Thunk>>,
}

/// Queue depth: one step's worth of layer jobs (48 layers × one publish each).
/// Stale-entry eviction is the job body's business (see `submit_lookahead_job`):
/// a job whose layer has since queued a newer prediction is dropped at consume
/// time, so a deep queue costs nothing but memory.
const LOOKAHEAD_QUEUE_DEPTH: usize = 8;

impl LookaheadQueue {
    fn store(&self, job: Thunk) {
        let mut q = self.jobs.lock().unwrap();
        if q.len() < LOOKAHEAD_QUEUE_DEPTH {
            q.push_back(job);
        } // else: drop — never block the token thread
    }

    fn try_claim(&self) -> Option<Thunk> {
        self.jobs.lock().unwrap().pop_front()
    }
}

/// The queue shared by the pool's workers and `submit_lookahead_job`.
fn lookahead_queue() -> &'static LookaheadQueue {
    static Q: OnceLock<LookaheadQueue> = OnceLock::new();
    Q.get_or_init(|| LookaheadQueue { jobs: Mutex::new(Default::default()) })
}

/// `NAIWA_SPIN_POOL=0`: a small dedicated lookahead thread team instead of
/// pool workers.  They poll the shared queue and park between jobs; a submit
/// wakes them (thread unpark), like the old `Lookahead`'s channel.
fn fallback_lookahead() -> &'static LookaheadQueue {
    fallback_lazy();
    lookahead_queue()
}

/// The team's handles, when it was spawned.
static FALLBACK_THREAD: OnceLock<Option<Vec<std::thread::Thread>>> = OnceLock::new();

/// Team size: enough to drain a step's layer jobs concurrently (the wait +
/// matmul per job is milliseconds; one thread per layer job is neither needed
/// nor affordable on a 12-thread host), small enough to never compete with
/// the CPU expert workers for cores.  Jobs block in `wait_publish` for a large
/// fraction of their runtime, which is exactly when a teammate picks up the
/// next job — serial W106 measured hit 87-89% against ~90-93% concurrent.
const LOOKAHEAD_TEAM: usize = 4;

fn fallback_lazy() {
    FALLBACK_THREAD.get_or_init(|| {
        let mut handles = Vec::new();
        for i in 0..LOOKAHEAD_TEAM {
            let h = std::thread::Builder::new()
                .name(format!("cpu-lookahead-{i}"))
                .spawn(|| {
                    // Reuse the shared queue so submission has one implementation
                    // in both modes.
                    let q = lookahead_queue();
                    // Wake once at start so an early submission is not missed.
                    loop {
                        if let Some(job) = q.try_claim() {
                            job();
                        } else {
                            std::thread::park();
                        }
                    }
                })
                .ok();
            match h {
                Some(h) => handles.push(h.thread().clone()),
                None => break,
            }
        }
        (!handles.is_empty()).then_some(handles)
    });
}

/// The team's thread handles, when it was spawned.
fn fallback_thread() -> Option<&'static [std::thread::Thread]> {
    // `get()` returns `Option<&Option<Vec<Thread>>>`: flatten the inner Option
    // (spawn failure) and borrow the Vec as a slice.
    FALLBACK_THREAD.get().and_then(|ts| ts.as_deref())
}

// ---------------------------------------------------------------------------
// CPU core pinning (Linux only)
// ---------------------------------------------------------------------------

/// The core worker `i` is pinned to: the physical cores `first_core..half`
/// first, then their SMT siblings.  On this 6-core/12-thread host with
/// `first_core = 1`: workers 0-4 on cores 1-5 (core 0 is held by the host),
/// then workers 5-9 on 7-11 (siblings of 1-5; core 6 is core 0's sibling and
/// stays with the host).
fn worker_core(first_core: usize, i: usize) -> usize {
    let half = cpu_count() / 2; // physical cores live in 0..half
    let physical = half.saturating_sub(first_core); // e.g. 5 for first_core 1
    if i < physical {
        first_core + i
    } else {
        // The sibling half `half..cpus`; skip the sibling of core 0.
        half + 1 + (i - physical)
    }
}

/// Number of logical CPUs; a fixed 12 fallback keeps the layout valid even
/// before `/sys` is consulted.
fn cpu_count() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(12, |n| n.get()))
}

#[cfg(target_os = "linux")]
fn pin_to_core(core: usize) {
    use std::mem::zeroed;
    unsafe {
        let mut set: libc::cpu_set_t = zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core, &mut set);
        let _ = libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_to_core(_core: usize) {}
