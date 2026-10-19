//! A single-thread task executor for the CPU side of decode.
//!
//! [`cpu`] is the engine-wide executor. A caller submits a [`Task`]; the executor's one worker
//! thread runs it FIFO. Result-bearing tasks use [`Executor::submit_result`], which additionally
//! publishes the result through a `ResultDoorbell` and returns a [`Pending`] that the caller can
//! later queue as a stream-ordered wait+copy.
//!
//! The executor also exposes whether it is idle ([`Executor::is_idle`]) and a one-shot idle
//! notification ([`Executor::on_idle`]). The lookahead scheduler in `components/preload` uses
//! those to run its prediction jobs in the gaps between CPU tasks. The executor does not know
//! what a prediction is; it only knows tasks, busy/idle and result slots.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::Thread;
use std::time::{Duration, Instant};

use candle_core::{Device, Result, Tensor};

/// A unit of work submitted to an [`Executor`].
pub(crate) type Task = Box<dyn FnOnce() + Send + 'static>;

/// Safety net only: every real wake comes from a submit or a batch completion, so a lost unpark
/// costs at most this much.
const IDLE_PARK: Duration = Duration::from_millis(2);

/// The engine-wide CPU executor. One worker, FIFO, one active task at a time.
pub(crate) struct Executor {
    name: &'static str,
    jobs: Mutex<VecDeque<Task>>,
    thread: OnceLock<Thread>,
    /// True while a task is running or the queue is non-empty.
    busy: AtomicBool,
    /// Threads to unpark on the busy -> idle transition.
    idle_wakers: Mutex<Vec<Thread>>,
}

static CPU: Executor = Executor::new("cpu-experts");

/// The engine-wide CPU executor.
pub(crate) fn cpu() -> &'static Executor {
    &CPU
}

impl Executor {
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            jobs: Mutex::new(VecDeque::new()),
            thread: OnceLock::new(),
            busy: AtomicBool::new(false),
            idle_wakers: Mutex::new(Vec::new()),
        }
    }

    /// Queue a task. Never blocks: the token thread only takes the queue lock and wakes the worker.
    pub(crate) fn submit(&'static self, task: Task) {
        self.busy.store(true, Ordering::Release);
        self.jobs.lock().unwrap().push_back(task);
        self.ensure_thread();
        self.wake();
    }

    /// Is no task running and none queued?
    pub(crate) fn is_idle(&self) -> bool {
        !self.busy.load(Ordering::Acquire)
    }

    /// Unpark `thread` whenever the executor becomes idle. Used by the prediction scheduler.
    pub(crate) fn on_idle(&'static self, thread: Thread) {
        let mut wakers = self.idle_wakers.lock().unwrap();
        wakers.push(thread.clone());
        if self.is_idle() {
            thread.unpark();
        }
    }

    /// Wake the worker (there may be a queued task, or an idle gate opened).
    pub(crate) fn wake(&self) {
        if let Some(t) = self.thread.get() {
            t.unpark();
        }
    }

    /// Submit a result-bearing task. On CUDA the task runs on the worker and publishes into the
    /// global result doorbell; otherwise it runs inline.
    pub(crate) fn submit_result<F>(
        &'static self,
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
            cuda::take_error()?;
            let chan = cuda::global_channel(dev, rows * cols)?;
            let seq = chan.next_job();
            let len = rows * cols;
            let done = cuda::SetFlag::new(Arc::clone(chan.doorbell()), seq);
            let t_submit = Instant::now();
            self.submit(Box::new(move || {
                let t_start = Instant::now();
                let mut done = done;
                match compute() {
                    Ok(v) if v.len() == len => done.publish(&v),
                    Ok(v) => {
                        cuda::record_error(format!("returned {} values, expected {len}", v.len()));
                        done.publish_empty();
                    }
                    Err(e) => {
                        cuda::record_error(e.to_string());
                        done.publish_empty();
                    }
                }
                crate::stats::record_retired(t_submit, t_start);
            }));
            return Ok(Pending {
                device: device.clone(),
                rows,
                cols,
                inner: Inner::Running(cuda::Running { chan, seq, len }),
            });
        }
        Ok(Pending {
            device: device.clone(),
            rows,
            cols,
            inner: Inner::Done(compute()?),
        })
    }

    fn ensure_thread(&'static self) {
        self.thread.get_or_init(|| {
            std::thread::Builder::new()
                .name(self.name.into())
                .spawn(move || self.run())
                .map(|h| h.thread().clone())
                .unwrap_or_else(|e| {
                    tracing::warn!("no {} thread ({e}); tasks will not run", self.name);
                    std::thread::current()
                })
        });
    }

    fn run(&'static self) {
        loop {
            loop {
                let task = self.jobs.lock().unwrap().pop_front();
                let Some(task) = task else { break };
                self.busy.store(true, Ordering::Release);
                task();
            }
            self.busy.store(false, Ordering::Release);
            for t in self.idle_wakers.lock().unwrap().iter() {
                t.unpark();
            }
            // A submit may have raced the empty check; it set `busy` and unparked us.
            if !self.jobs.lock().unwrap().is_empty() {
                continue;
            }
            std::thread::park_timeout(IDLE_PARK);
        }
    }
}

/// A CPU computation started by [`Executor::submit_result`].
pub(crate) struct Pending {
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

impl Pending {
    /// The result as a `[rows, cols]` F32 tensor on the device, queued behind everything the
    /// caller queued so far; the host does not wait for it on CUDA.
    pub(crate) fn upload(self) -> Result<Tensor> {
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
    use std::sync::{Arc, Mutex};

    use crate::doorbell::ResultDoorbell;
    use candle_core::cuda_backend::CudaDevice;
    use candle_core::{DType, Device, Error, Result, Tensor};

    static FAILED: Mutex<Option<String>> = Mutex::new(None);

    pub(super) fn record_error(message: String) {
        *FAILED.lock().unwrap() = Some(message);
    }

    pub(super) fn take_error() -> Result<()> {
        match FAILED.lock().unwrap().take() {
            Some(e) => Err(Error::Msg(format!("CPU experts failed: {e}"))),
            None => Ok(()),
        }
    }

    /// The engine-wide CPU-return channel: one result doorbell plus its seq allocator.
    pub(super) struct Channel {
        doorbell: Arc<ResultDoorbell>,
        next: Mutex<u32>,
    }

    impl Channel {
        pub(super) fn doorbell(&self) -> &Arc<ResultDoorbell> {
            &self.doorbell
        }

        pub(super) fn max_n(&self) -> usize {
            self.doorbell.max_n()
        }

        pub(super) fn next_job(&self) -> u32 {
            let mut seq = self.next.lock().unwrap();
            *seq = seq.wrapping_add(1);
            *seq
        }
    }

    /// The global channel. It grows and retires the mapped allocation when a larger result window
    /// arrives; retired channels are kept alive because a queued wait kernel may still hold their
    /// raw mapped pointers.
    pub(super) fn global_channel(dev: &CudaDevice, min_n: usize) -> Result<Arc<Channel>> {
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

    pub(super) struct Running {
        pub(super) chan: Arc<Channel>,
        pub(super) seq: u32,
        pub(super) len: usize,
    }

    pub(super) struct SetFlag {
        bell: Arc<ResultDoorbell>,
        seq: u32,
        published: bool,
    }

    impl SetFlag {
        pub(super) fn new(bell: Arc<ResultDoorbell>, seq: u32) -> Self {
            Self {
                bell,
                seq,
                published: false,
            }
        }

        pub(super) fn publish(&mut self, values: &[f32]) {
            self.publish_raw(Some(values));
        }

        pub(super) fn publish_empty(&mut self) {
            self.publish_raw(None);
        }

        fn publish_raw(&mut self, values: Option<&[f32]>) {
            if self.published {
                return;
            }
            self.bell.publish(values, self.seq);
            self.published = true;
        }
    }

    impl Drop for SetFlag {
        fn drop(&mut self) {
            self.publish_raw(None);
        }
    }

    pub(super) fn upload(
        dev: &CudaDevice,
        running: Running,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        let Running { chan, seq, len } = running;
        debug_assert_eq!(len, rows * cols);
        // SAFETY: fully overwritten by the copy kernel before anything reads it.
        let out = unsafe { Tensor::empty((rows, cols), DType::F32, &Device::Cuda(dev.clone()))? };
        chan.doorbell().wait_copy(seq, len, &out)?;
        Ok(out)
    }
}
