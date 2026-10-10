//! Results the CPU is still computing, already queued on the GPU.
//!
//! A decode step's CPU experts used to finish before the MoE block returned, so the host only
//! started launching the next layer's attention, norms and router (~0.5 ms of launches per
//! layer) once they were done. Now [`start`] hands the CPU work to a worker thread, the caller
//! queues its own GPU work (the cache hits), and [`Pending::upload`] gives candle's stream, in
//! order,
//!
//! 1. a host function that blocks the stream until the worker is done,
//! 2. a copy of the worker's result from a pinned staging buffer into the returned tensor.
//!
//! Everything the caller queues afterwards runs after that copy, so the next layer's launches
//! overlap the CPU experts, and the host only waits at the next read back (the next MoE block's
//! top-k). The worker makes no CUDA calls and the host function only waits on a condition
//! variable: CUDA calls from a second thread are what broke decode in G8, and a host function
//! must not call CUDA at all.
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
    use std::ffi::c_void;
    use std::sync::mpsc::{channel, Sender};
    use std::sync::{Arc, Condvar, Mutex};

    use candle_core::cuda::cudarc::driver::{result, sys, CudaContext, CudaEvent, DevicePtr};
    use candle_core::cuda_backend::CudaDevice;
    use candle_core::{DType, Device, Error, Result, Storage, Tensor};

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

    /// Set once the worker has written the staging buffer (or given up).
    #[derive(Default)]
    struct Signal {
        done: Mutex<bool>,
        cv: Condvar,
    }

    impl Signal {
        fn set(&self) {
            *self.done.lock().unwrap() = true;
            self.cv.notify_all();
        }

        fn wait(&self) {
            let mut done = self.done.lock().unwrap();
            while !*done {
                done = self.cv.wait(done).unwrap();
            }
        }
    }

    /// Sets the signal when dropped, so a panicking job cannot leave the stream blocked.
    struct SetOnDrop(Arc<Signal>);

    impl Drop for SetOnDrop {
        fn drop(&mut self) {
            self.0.set();
        }
    }

    /// The host function queued on candle's stream: wait for the worker. No CUDA calls.
    unsafe extern "C" fn wait_for_worker(arg: *mut c_void) {
        // SAFETY: `arg` is the `Box<Arc<Signal>>` leaked by `upload_when_ready`, passed once.
        let signal = unsafe { Box::from_raw(arg.cast::<Arc<Signal>>()) };
        signal.wait();
    }

    /// Page-locked host memory the copy reads when the stream reaches it, and an event recorded
    /// after that copy: the buffer is free again once the event has completed.
    struct Staging {
        ptr: *mut f32,
        len: usize,
        copied: CudaEvent,
        ctx: Arc<CudaContext>,
    }

    // SAFETY: the buffer is plain host memory; it is written by one worker job at a time, and
    // only reused after the copy that reads it has completed.
    unsafe impl Send for Staging {}

    impl Drop for Staging {
        fn drop(&mut self) {
            let _ = self.ctx.bind_to_thread();
            let _ = self.copied.synchronize();
            // SAFETY: allocated by `malloc_host` in `Staging::new` and no copy reads it now.
            unsafe {
                let _ = result::free_host(self.ptr.cast());
            }
        }
    }

    impl Staging {
        fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
            // SAFETY: a fresh allocation of `len` f32s.
            let ptr = unsafe { result::malloc_host(len * std::mem::size_of::<f32>(), 0) }
                .map_err(Error::wrap)?
                .cast::<f32>();
            let copied = ctx
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .map_err(Error::wrap)?;
            Ok(Self {
                ptr,
                len,
                copied,
                ctx: Arc::clone(ctx),
            })
        }
    }

    /// Staging buffers whose copy may still be in flight. A step uses one per layer and the
    /// copies finish within a layer or two, so a handful stay in circulation.
    static POOL: Mutex<Vec<Staging>> = Mutex::new(Vec::new());

    fn acquire(ctx: &Arc<CudaContext>, len: usize) -> Result<Staging> {
        let mut pool = POOL.lock().unwrap();
        let free = pool
            .iter()
            .position(|s| Arc::ptr_eq(&s.ctx, ctx) && s.len >= len && s.copied.is_complete());
        match free {
            Some(i) => Ok(pool.swap_remove(i)),
            None => Staging::new(ctx, len),
        }
    }

    /// The raw staging pointer, sent to the worker.
    struct SendPtr(*mut f32);
    // SAFETY: see `Staging`.
    unsafe impl Send for SendPtr {}

    /// A job handed to the worker, writing into `staging`.
    pub struct Running {
        staging: Staging,
        signal: Arc<Signal>,
    }

    pub fn start(
        dev: &CudaDevice,
        len: usize,
        compute: Box<dyn FnOnce() -> Result<Vec<f32>> + Send>,
    ) -> Result<Running> {
        take_error()?;
        let staging = acquire(dev.cuda_stream().context(), len)?;
        let signal = Arc::new(Signal::default());
        let dst_ptr = SendPtr(staging.ptr);
        let done = SetOnDrop(Arc::clone(&signal));
        let job: Job = Box::new(move || {
            let _done = done;
            let dst_ptr = dst_ptr;
            match compute() {
                Ok(v) if v.len() == len => {
                    // SAFETY: the staging buffer holds at least `len` f32s and nothing else
                    // touches it until the copy queued behind this job's signal.
                    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), dst_ptr.0, len) };
                }
                Ok(v) => {
                    *FAILED.lock().unwrap() =
                        Some(format!("returned {} values, expected {len}", v.len()));
                }
                Err(e) => *FAILED.lock().unwrap() = Some(e.to_string()),
            }
        });
        worker()
            .lock()
            .unwrap()
            .send(job)
            .map_err(|_| Error::Msg("the CPU experts thread is gone".into()))?;
        Ok(Running { staging, signal })
    }

    pub fn upload(dev: &CudaDevice, running: Running, rows: usize, cols: usize) -> Result<Tensor> {
        let Running { staging, signal } = running;
        let len = rows * cols;
        let stream = dev.cuda_stream();
        // SAFETY: fully overwritten by the copy below before anything reads it.
        let out = unsafe { Tensor::empty((rows, cols), DType::F32, &Device::Cuda(dev.clone()))? };
        let queued = (|| -> Result<()> {
            let (storage, layout) = out.storage_and_layout();
            let Storage::Cuda(cuda) = &*storage else {
                candle_core::bail!("deferred upload target is not on CUDA");
            };
            let slice = cuda.as_cuda_slice::<f32>()?;
            let (dst, _guard) = slice.device_ptr(&stream);
            let dst = dst + (layout.start_offset() * std::mem::size_of::<f32>()) as u64;
            let arg = Box::into_raw(Box::new(Arc::clone(&signal))).cast::<c_void>();
            // SAFETY: `wait_for_worker` takes back `arg` exactly once and makes no CUDA calls.
            // The copy reads pinned memory, so it reads it when the stream reaches it, i.e.
            // after the worker has written it.
            unsafe {
                if let Err(e) =
                    result::stream::launch_host_function(stream.cu_stream(), wait_for_worker, arg)
                {
                    drop(Box::from_raw(arg.cast::<Arc<Signal>>()));
                    return Err(Error::wrap(e));
                }
                let src = std::slice::from_raw_parts(staging.ptr, len);
                result::memcpy_htod_async(dst, src, stream.cu_stream()).map_err(Error::wrap)?;
            }
            staging.copied.record(&stream).map_err(Error::wrap)
        })();
        if queued.is_err() {
            // The staging buffer may still be written: wait for the worker before freeing it.
            signal.wait();
            let _ = stream.synchronize();
        }
        POOL.lock().unwrap().push(staging);
        queued?;
        Ok(out)
    }
}
