//! Results the CPU is still computing, already queued on the GPU.
//!
//! A decode step's CPU experts used to finish before the MoE block returned, so the host only
//! started launching the next layer's attention, norms and router (~0.5 ms of launches per
//! layer) once they were done. Now [`start`] hands the CPU work to a worker thread, the caller
//! queues its own GPU work (the cache hits), and [`Pending::upload`] gives candle's stream, in
//! order,
//!
//! 1. a wait for the worker's result published on a mapped flag channel (R2b),
//! 2. a stream-ordered kernel that waits for the publish then copies the payload slot
//!    into the returned tensor.
//!
//! There is no host function and no H2D memcpy launch: the worker's stores become visible to
//! the copy kernel through the mapped seq counter, so the stream never hosts a host callback
//! and the pattern stays graph-capturable (the R2b prerequisite for R5). Steps are sequential,
//! so one channel serves the whole engine: the worker publishes seq `k`, whose slot is
//! `k % DEPTH`, and a slow GPU reading slot `k` never races the worker writing slot `k + 1`.
//!
//! A failed computation leaves the returned tensor unwritten; the error is returned by the next
//! [`start`] (the next layer of the same step, or the next step's first).
//!
//! The module also owns the single lookahead prediction worker. Predictions only run in the gaps
//! between CPU-expert batches, so the batch completion is the wake-up; see [`submit_lookahead`].

use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use candle_core::{Device, Result, Tensor};

/// How many CPU-expert batches are running right now.
///
/// The lookahead scheduler reads this to run its router matmuls and slot uploads in the **gaps**
/// between batches rather than beside them (W106: beside them measured 10 ms/step slower — the
/// matmul is 2–3 cores and the batch, on rayon, wants every core). It is set by the thread that
/// runs the batches, whichever pool the parallel passes inside them use.
static BATCH_DEPTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// No CPU-expert batch is running.
pub fn expert_batch_idle() -> bool {
    BATCH_DEPTH.load(std::sync::atomic::Ordering::Acquire) == 0
}

/// Marks a CPU-expert batch as running for as long as it is alive.
struct BatchBusy;

impl BatchBusy {
    fn enter() -> Self {
        BATCH_DEPTH.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self
    }
}

impl Drop for BatchBusy {
    fn drop(&mut self) {
        if BATCH_DEPTH.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
            // The gap opened: predictions may run now. This is the wake-up that replaces polling
            // for `expert_batch_idle` from inside a prediction.
            wake_lookahead();
        }
    }
}

/// A queued prediction.
type Thunk = Box<dyn FnOnce() + Send>;

/// Safety net only: every real wake comes from a submit or a batch completion, so a lost unpark
/// costs at most this much.
const LOOKAHEAD_PARK: Duration = Duration::from_millis(2);

/// Queue depth: one step's worth of layer jobs (48) plus headroom. Overflow drops.
const LOOKAHEAD_DEPTH: usize = 64;

/// Pending predictions. `LEN` mirrors `JOBS.len()` so the drain loop can skip the mutex.
static LOOKAHEAD_JOBS: Mutex<VecDeque<Thunk>> = Mutex::new(VecDeque::new());
static LOOKAHEAD_LEN: AtomicUsize = AtomicUsize::new(0);

/// The one prediction thread, once it exists.
static LOOKAHEAD_THREAD: OnceLock<std::thread::Thread> = OnceLock::new();

/// Wake the prediction thread: there may be work (a submit) or permission to run it (a batch
/// ended).
pub fn wake_lookahead() {
    if let Some(t) = LOOKAHEAD_THREAD.get() {
        t.unpark();
    }
}

/// Queue a prediction and wake its thread. Never blocks the token thread: on overflow the job is
/// dropped.
pub fn submit_lookahead(job: Thunk) {
    LOOKAHEAD_THREAD.get_or_init(|| {
        std::thread::Builder::new()
            .name("lookahead".into())
            .spawn(lookahead_worker)
            .map(|h| h.thread().clone())
            .unwrap_or_else(|e| {
                tracing::warn!("no lookahead thread ({e}); predictions will not run");
                std::thread::current()
            })
    });
    {
        let mut jobs = LOOKAHEAD_JOBS.lock().unwrap();
        if jobs.len() >= LOOKAHEAD_DEPTH {
            crate::stats::record_cache_lookahead_dropped();
            return;
        }
        jobs.push_back(job);
        LOOKAHEAD_LEN.store(jobs.len(), std::sync::atomic::Ordering::Release);
    }
    wake_lookahead();
}

fn take_lookahead() -> Option<Thunk> {
    if LOOKAHEAD_LEN.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return None;
    }
    let mut jobs = LOOKAHEAD_JOBS.lock().unwrap();
    let job = jobs.pop_front();
    LOOKAHEAD_LEN.store(jobs.len(), std::sync::atomic::Ordering::Release);
    job
}

/// Drain predictions while no expert batch is running, then park until the next event.
fn lookahead_worker() {
    loop {
        // The gap check is re-tested before every job: a batch can start mid-drain, and its
        // completion will wake us again.
        while expert_batch_idle() {
            let Some(job) = take_lookahead() else { break };
            job();
        }
        std::thread::park_timeout(LOOKAHEAD_PARK);
    }
}

/// A CPU computation started by [`start`].
pub struct Pending {
    device: Device,
    rows: usize,
    cols: usize,
    inner: Inner,
}

enum Inner {
    /// Off CUDA: computed inline.
    Done(Vec<f32>),
    #[cfg(feature = "cuda")]
    Running(cuda::Running),
}

/// Start `compute()`, whose result is `rows * cols` F32 values for `device`. On CUDA it runs on
/// the worker thread and publishes into the global result doorbell; otherwise it runs now.
pub fn start<F>(
    device: &Device,
    rows: usize,
    cols: usize,
    compute: F,
) -> Result<Pending>
where
    F: FnOnce() -> Result<Vec<f32>> + Send + 'static,
{
    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = device {
        let chan = cuda::global_channel(dev, rows * cols)?;
        let running = cuda::start(dev, &chan, rows * cols, Box::new(compute))?;
        return Ok(Pending {
            device: device.clone(),
            rows,
            cols,
            inner: Inner::Running(running),
        });
    }
    Ok(Pending {
        device: device.clone(),
        rows,
        cols,
        inner: Inner::Done(compute()?),
    })
}

impl Pending {
    /// The result as a `[rows, cols]` F32 tensor on the device, queued behind everything the
    /// caller queued so far; the host does not wait for it on CUDA.
    pub fn upload(self) -> Result<Tensor> {
        match self.inner {
            Inner::Done(data) => Tensor::from_vec(data, (self.rows, self.cols), &self.device),
            #[cfg(feature = "cuda")]
            Inner::Running(running) => {
                let Device::Cuda(dev) = &self.device else {
                    unreachable!("started on CUDA")
                };
                cuda::upload(dev, running, self.rows, self.cols)
            }
        }
    }
}
#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::mpsc::{channel, Sender};
    use std::sync::{Arc, Mutex};

    use super::BatchBusy;

    use candle_core::cuda_backend::CudaDevice;
    use candle_core::{DType, Device, Error, Result, Tensor};
    use crate::doorbell::ResultDoorbell;

    type Job = Box<dyn FnOnce() + Send>;

    /// The single worker thread: steps are sequential, so jobs never overlap one another.
    fn worker() -> &'static Mutex<Sender<Job>> {
        static WORKER: std::sync::OnceLock<Mutex<Sender<Job>>> = std::sync::OnceLock::new();
        WORKER.get_or_init(|| {
            let (tx, rx) = channel::<Job>();
            std::thread::Builder::new()
                .name("cpu-experts".into())
                .spawn(move || {
                    for job in rx {
                        let _busy = BatchBusy::enter();
                        job();
                    }
                })
                .expect("spawn the CPU experts thread");
            Mutex::new(tx)
        })
    }

    static FAILED: Mutex<Option<String>> = Mutex::new(None);

    pub fn take_error() -> Result<()> {
        match FAILED.lock().unwrap().take() {
            Some(e) => Err(Error::Msg(format!("CPU experts failed: {e}"))),
            None => Ok(()),
        }
    }

    /// The engine-wide CPU-return channel: one result doorbell plus its seq allocator.
    ///
    /// Steps are sequential and the worker's FIFO order is the stream's consumption order, so one
    /// channel is enough. `seq` is allocated here; the doorbell derives the ring slot as
    /// `seq % DEPTH`.
    pub struct Channel {
        doorbell: Arc<ResultDoorbell>,
        next: Mutex<u32>,
    }

    impl Channel {
        pub fn doorbell(&self) -> &Arc<ResultDoorbell> {
            &self.doorbell
        }

        pub fn max_n(&self) -> usize {
            self.doorbell.max_n()
        }

        fn next_job(&self) -> u32 {
            let mut seq = self.next.lock().unwrap();
            *seq = seq.wrapping_add(1);
            *seq
        }
    }

    /// The **global** channel. It grows and retires the mapped allocation when a larger result
    /// window arrives; retired channels are kept alive because a queued wait kernel may still hold
    /// their raw mapped pointers.
    pub fn global_channel(dev: &CudaDevice, min_n: usize) -> Result<Arc<Channel>> {
        static CHANNEL: Mutex<(Option<Arc<Channel>>, Vec<Arc<Channel>>)> =
            Mutex::new((None, Vec::new()));
        let mut guard = CHANNEL.lock().unwrap();
        let (chan, retired) = &mut *guard;
        if chan.as_ref().is_none_or(|c| c.max_n() < min_n) {
            // Grow with headroom so verify windows a few rows larger do not realloc again.
            let want = min_n.max(chan.as_ref().map_or(0, |c| 2 * c.max_n()));
            let next = ResultDoorbell::acquire(&Device::Cuda(dev.clone()), want)
                .ok()
                .flatten()
                .map(|doorbell| {
                    Arc::new(Channel {
                        doorbell: Arc::new(doorbell),
                        next: Mutex::new(0),
                    })
                });
            if let (Some(old), Some(_next)) = (chan.take(), &next) {
                retired.push(old);
            }
            *chan = next;
        }
        Ok(Arc::clone(chan.as_ref().ok_or_else(|| {
            Error::Msg("the mapped result doorbell is unavailable".into())
        })?))
    }

    /// A job handed to the worker: run `compute`, publish the result on the channel.
    pub struct Running {
        chan: Arc<Channel>,
        seq: u32,
        len: usize,
    }

    pub fn start(
        _dev: &CudaDevice,
        chan: &Arc<Channel>,
        len: usize,
        compute: Box<dyn FnOnce() -> Result<Vec<f32>> + Send>,
    ) -> Result<Running> {
        take_error()?;
        let seq = chan.next_job();
        let done = SetFlag {
            bell: Arc::clone(chan.doorbell()),
            seq,
            published: false,
        };
        let len_ok = len;
        // RET_DBG (naiwa): the handoff latency the GPU pays per miss - submit -> the worker starting the
        // compute (queue) and the compute + publish itself.
        let t_submit = std::time::Instant::now();
        let job: Job = Box::new(move || {
            let t_start = std::time::Instant::now();
            let mut done = done;
            match compute() {
                Ok(v) if v.len() == len_ok => {
                    // Payload then release-fence then seq: the copy kernel reads behind seq.
                    done.publish(&v);
                }
                Ok(v) => {
                    *FAILED.lock().unwrap() = Some(format!(
                        "returned {} values, expected {len_ok}",
                        v.len()
                    ));
                    // Still bump seq so the stream's wait kernel does not spin forever.
                    done.publish_empty();
                }
                Err(e) => {
                    *FAILED.lock().unwrap() = Some(e.to_string());
                    done.publish_empty();
                }
            }
            crate::stats::record_retired(t_submit, t_start);
        });
        worker()
            .lock()
            .unwrap()
            .send(job)
            .map_err(|_| Error::Msg("the CPU experts thread is gone".into()))?;
        Ok(Running { chan: Arc::clone(chan), seq, len })
    }

    /// Publishes a payload slot (or just the seq bump on failure). Bumps seq exactly once per
    /// job — also on drop during unwind — because the stream-side wait kernel spins until seq
    /// reaches its target and a panicking job must not leave the GPU stuck (the failure is
    /// reported by the next `start` instead).
    struct SetFlag {
        bell: Arc<ResultDoorbell>,
        seq: u32,
        published: bool,
    }

    impl SetFlag {
        fn publish_raw(&mut self, values: Option<&[f32]>) {
            if self.published {
                return;
            }
            self.bell.publish(values, self.seq);
            self.published = true;
        }

        fn publish(&mut self, values: &[f32]) {
            self.publish_raw(Some(values));
        }

        fn publish_empty(&mut self) {
            self.publish_raw(None);
        }
    }

    impl Drop for SetFlag {
        fn drop(&mut self) {
            self.publish_raw(None);
        }
    }

    pub fn upload(dev: &CudaDevice, running: Running, rows: usize, cols: usize) -> Result<Tensor> {
        let Running { chan, seq, len } = running;
        debug_assert_eq!(len, rows * cols);
        // SAFETY: fully overwritten by the copy kernel before anything reads it.
        let out = unsafe { Tensor::empty((rows, cols), DType::F32, &Device::Cuda(dev.clone()))? };
        chan.doorbell().wait_copy(seq, len, &out)?;
        Ok(out)
    }
}
