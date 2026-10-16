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

use candle_core::{Device, Result, Tensor};

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
/// the worker thread; elsewhere it runs now.
pub fn start<F>(device: &Device, rows: usize, cols: usize, compute: F) -> Result<Pending>
where
    F: FnOnce() -> Result<Vec<f32>> + Send + 'static,
{
    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = device {
        let running = cuda::start(dev, rows * cols, Box::new(compute))?;
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

    use candle_core::cuda_backend::CudaDevice;
    use candle_core::{DType, Device, Error, Result, Tensor};
    use mistralrs_core::doorbell::FlagBuf;

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

    /// The engine-wide mapped channel: the worker publishes each result into a payload slot
    /// and bumps seq; the stream-ordered wait+copy kernel (queued by [`upload`]) waits on the
    /// same counter. One channel serves the whole engine because steps are sequential — at
    /// most one publish is unread per slot, and the ring has two slots.
    ///
    /// The slot size is not known up front (it is `rows x hidden`, and rows vary with the
    /// speculative depth), so the channel grows on demand: a `start` larger than the current
    /// slot replaces the channel. A swapped-out channel is safe — each `Running` and its job
    /// hold their own `Arc`, and the worker publishes the global monotonic `publish_seq`
    /// (absolute, not incremental), so a fresh channel's counter simply jumps to it.
    fn flag_channel(dev: &CudaDevice, min_n: usize) -> Result<Arc<FlagBuf>> {
        static CHANNEL: Mutex<(Option<Arc<FlagBuf>>, Vec<Arc<FlagBuf>>)> = Mutex::new((None, Vec::new()));
        let mut guard = CHANNEL.lock().unwrap();
        let (chan, retired) = &mut *guard;
        if chan.as_ref().is_none_or(|c| c.max_n() < min_n) {
            // Grow with headroom so verify windows a few rows larger do not realloc again.
            let want = min_n.max(chan.as_ref().map_or(0, |c| 2 * c.max_n()));
            let next = FlagBuf::acquire(&Device::Cuda(dev.clone()), want)
                .ok()
                .flatten()
                .map(Arc::new);
            if let (Some(old), Some(_next)) = (chan.take(), &next) {
                // The old channel's wait kernel may still be queued with raw mapped
                // pointers; retired channels are kept alive forever (a few hundred KB
                // each, growth bounded by the doubling) instead of freed under it.
                retired.push(old);
            }
            *chan = next;
        }
        Ok(Arc::clone(chan.as_ref().ok_or_else(|| {
            Error::Msg("the mapped flag channel is unavailable".into())
        })?))
    }

    /// Which slot and seq the next publish uses (monotonic; slot cycles mod RING_DEPTH).
    static PUBLISH: Mutex<(usize, u32)> = Mutex::new((0, 0));

    /// A job handed to the worker: run `compute`, publish the result on the channel.
    pub struct Running {
        chan: Arc<FlagBuf>,
        slot: usize,
        publish_seq: u32,
        len: usize,
    }

    pub fn start(
        dev: &CudaDevice,
        len: usize,
        compute: Box<dyn FnOnce() -> Result<Vec<f32>> + Send>,
    ) -> Result<Running> {
        take_error()?;
        let chan = flag_channel(dev, len)?;
        let (slot, publish_seq) = {
            let mut p = PUBLISH.lock().unwrap();
            p.0 = p.0.wrapping_add(1);
            p.1 = p.1.wrapping_add(1);
            (p.0, p.1)
        };
        let done = SetFlag {
            chan: Arc::clone(&chan),
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
        Ok(Running { chan, slot, publish_seq, len })
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

    pub fn upload(dev: &CudaDevice, running: Running, rows: usize, cols: usize) -> Result<Tensor> {
        let Running { chan, slot, publish_seq, len } = running;
        debug_assert_eq!(len, rows * cols);
        // SAFETY: fully overwritten by the copy kernel before anything reads it.
        let out = unsafe { Tensor::empty((rows, cols), DType::F32, &Device::Cuda(dev.clone()))? };
        chan.wait_copy_slot(slot, publish_seq, len, &out)?;
        Ok(out)
    }
}
