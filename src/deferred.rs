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
//! so one channel serves the whole engine: the worker publishes slot `k % RING_DEPTH`, and a
//! slow GPU reading slot `k` never races the worker writing slot `k + 1`.
//!
//! A failed computation leaves the returned tensor unwritten; the error is returned by the next
//! [`start`] (the next layer of the same step, or the next step's first).

use std::sync::Arc;

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
            crate::lookahead::wake();
        }
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

/// A layer's CPU-return channel on CUDA; a unit placeholder elsewhere, so the call sites stay
/// feature-agnostic.
#[cfg(feature = "cuda")]
pub type CpuChannel = cuda::Channel;
#[cfg(not(feature = "cuda"))]
pub type CpuChannel = ();

/// M3: submit a job that publishes into the shared [`crate::doorbell::ScheduleBuf`] at
/// `(layer, pass)` — the graph path. Publication order is the worker's FIFO order, which the
/// single-buffered payload and the monotonic `ready` word require.
#[cfg(feature = "cuda")]
pub fn start_schedule(
    schedule: std::sync::Arc<crate::doorbell::ScheduleBuf>,
    layer: u32,
    pass: u32,
    len: usize,
    compute: Box<dyn FnOnce() -> Result<Vec<f32>> + Send>,
) -> Result<()> {
    cuda::start_schedule(schedule, layer, pass, len, compute)
}

/// Start `compute()`, whose result is `rows * cols` F32 values for `device`. On CUDA with a
/// per-layer `chan` it runs on the worker thread and publishes into that channel; otherwise it
/// runs now.
pub fn start<F>(
    device: &Device,
    chan: Option<&std::sync::Arc<CpuChannel>>,
    rows: usize,
    cols: usize,
    compute: F,
) -> Result<Pending>
where
    F: FnOnce() -> Result<Vec<f32>> + Send + 'static,
{
    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = device {
        // Per-layer channel when the caller has one (the decode path), else the legacy global.
        let chan = match chan {
            Some(c) => Arc::clone(c),
            None => cuda::global_channel(dev, rows * cols)?,
        };
        let running = cuda::start(dev, &chan, rows * cols, Box::new(compute))?;
        return Ok(Pending {
            device: device.clone(),
            rows,
            cols,
            inner: Inner::Running(running),
        });
    }
    let _ = chan;
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
    use crate::doorbell::FlagBuf;

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

    /// A layer's CPU-return channel: its own mapped flag buffer and its own `(slot, seq)`
    /// numbering.
    ///
    /// **Per-layer, not engine-wide.** A shared channel has to share one `(slot, seq)`
    /// counter, so any job that does not consume the channel (a layer with two banks, or a
    /// step wider than the fast path allows) desynchronizes every later job from a device-side
    /// consumer. Per-layer keeps the numbering 1:1 with the layer's own jobs *by construction*,
    /// and it means a captured wait node is never stranded on a channel that grew and retired
    /// its allocation. This is `architecture-v2` §3.4's `flag_host_to_gpu: per-layer`.
    pub struct Channel {
        chan: Arc<FlagBuf>,
        next: Mutex<(usize, u32)>,
    }

    impl Channel {
        /// Allocate a channel whose payload slot holds at least `min_n` f32. `Ok(None)` when the
        /// device is not CUDA or the mapped allocation fails.
        pub fn acquire(device: &Device, min_n: usize) -> Result<Option<Arc<Channel>>> {
            if !device.is_cuda() {
                return Ok(None);
            }
            // Headroom: a verify window a few rows larger must not need another allocation (a
            // capture would otherwise be stranded on the retired one).
            let want = min_n.max(1) * 2;
            Ok(FlagBuf::acquire(device, want)?.map(|chan| {
                Arc::new(Channel {
                    chan: Arc::new(chan),
                    next: Mutex::new((0, 0)),
                })
            }))
        }

        /// The flag buffer itself (mapped cells + payload ring).
        pub fn flag(&self) -> &Arc<FlagBuf> {
            &self.chan
        }

        pub fn max_n(&self) -> usize {
            self.chan.max_n()
        }

        fn next_job(&self) -> (usize, u32) {
            let mut p = self.next.lock().unwrap();
            p.0 = p.0.wrapping_add(1);
            p.1 = p.1.wrapping_add(1);
            (p.0, p.1)
        }
    }

    /// The legacy **global** channel, for call sites without a per-layer one (`CpuExperts`, the
    /// `--cpu-moe` path with no expert cache). It keeps the old grow-and-retire behavior. Those
    /// call sites never capture a wait node, so neither the 1:1 numbering requirement nor the
    /// retirement hazard applies — and the decode path (the one that captures) always passes its
    /// own per-layer channel.
    pub fn global_channel(dev: &CudaDevice, min_n: usize) -> Result<Arc<Channel>> {
        static CHANNEL: Mutex<(Option<Arc<Channel>>, Vec<Arc<Channel>>)> =
            Mutex::new((None, Vec::new()));
        let mut guard = CHANNEL.lock().unwrap();
        let (chan, retired) = &mut *guard;
        if chan.as_ref().is_none_or(|c| c.max_n() < min_n) {
            // Grow with headroom so verify windows a few rows larger do not realloc again.
            let want = min_n.max(chan.as_ref().map_or(0, |c| 2 * c.max_n()));
            let next = FlagBuf::acquire(&Device::Cuda(dev.clone()), want)
                .ok()
                .flatten()
                .map(|f| {
                    Arc::new(Channel {
                        chan: Arc::new(f),
                        next: Mutex::new((0, 0)),
                    })
                });
            if let (Some(old), Some(_next)) = (chan.take(), &next) {
                // A queued wait kernel may still hold raw mapped pointers; keep retired channels
                // alive forever (a few hundred KB each, growth bounded by the doubling).
                retired.push(old);
            }
            *chan = next;
        }
        Ok(Arc::clone(chan.as_ref().ok_or_else(|| {
            Error::Msg("the mapped flag channel is unavailable".into())
        })?))
    }

    /// A job handed to the worker: run `compute`, publish the result on the channel.
    pub struct Running {
        chan: Arc<Channel>,
        slot: usize,
        publish_seq: u32,
        len: usize,
    }

    pub fn start(
        _dev: &CudaDevice,
        chan: &Arc<Channel>,
        len: usize,
        compute: Box<dyn FnOnce() -> Result<Vec<f32>> + Send>,
    ) -> Result<Running> {
        take_error()?;
        let (slot, publish_seq) = chan.next_job();
        let done = SetFlag {
            chan: Arc::clone(chan.flag()),
            slot,
            publish_seq,
            published: false,
        };
        let len_ok = len;
        let job: Job = Box::new(move || {
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
        });
        worker()
            .lock()
            .unwrap()
            .send(job)
            .map_err(|_| Error::Msg("the CPU experts thread is gone".into()))?;
        Ok(Running { chan: Arc::clone(chan), slot, publish_seq, len })
    }

    /// Publishes a payload slot (or just the seq bump on failure). Bumps seq exactly once per
    /// job — also on drop during unwind — because the stream-side wait kernel spins until seq
    /// reaches its target and a panicking job must not leave the GPU stuck (the failure is
    /// reported by the next `start` instead).
    struct SetFlag {
        chan: Arc<FlagBuf>,
        slot: usize,
        publish_seq: u32,
        published: bool,
    }

    impl SetFlag {
        fn publish_raw(&mut self, values: Option<&[f32]>) {
            if self.published {
                return;
            }
            self.chan.write_slot(values, self.slot, self.publish_seq);
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

    pub fn start_schedule(
        schedule: Arc<crate::doorbell::ScheduleBuf>,
        layer: u32,
        pass: u32,
        len: usize,
        compute: Box<dyn FnOnce() -> Result<Vec<f32>> + Send>,
    ) -> Result<()> {
        take_error()?;
        let id = schedule.sync_id(layer, crate::doorbell::SYNC_CPU_EXPERTS);
        let job: Job = Box::new(move || match compute() {
            Ok(v) if v.len() >= len => schedule.publish_cpu(layer, pass, &v[..len]),
            Ok(v) => {
                *FAILED.lock().unwrap() =
                    Some(format!("returned {} values, expected at least {len}", v.len()));
                schedule.publish_point(id, pass);
            }
            Err(e) => {
                *FAILED.lock().unwrap() = Some(e.to_string());
                schedule.publish_point(id, pass);
            }
        });
        worker()
            .lock()
            .unwrap()
            .send(job)
            .map_err(|e| Error::Msg(format!("the CPU experts thread is gone: {e}")))?;
        Ok(())
    }

    pub fn upload(dev: &CudaDevice, running: Running, rows: usize, cols: usize) -> Result<Tensor> {
        let Running { chan, slot, publish_seq, len } = running;
        debug_assert_eq!(len, rows * cols);
        // SAFETY: fully overwritten by the copy kernel before anything reads it.
        let out = unsafe { Tensor::empty((rows, cols), DType::F32, &Device::Cuda(dev.clone()))? };
        chan.flag().wait_copy_slot(slot, publish_seq, len, &out)?;
        Ok(out)
    }
}
