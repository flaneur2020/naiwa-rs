//! Slot uploads on a dedicated CUDA copy stream, so they overlap the model's compute.
//!
//! candle runs all work for a device on one stream. Uploading a slot through candle
//! (`qtensor_from_ggml`) allocates and copies on that stream, behind every kernel already queued.
//! Instead, each cached layer allocates its slot memory once at load and overwrites it from a
//! second stream, which runs on the copy engine alongside compute.
//!
//! Overwriting memory that queued kernels may still read needs two guards:
//! - Each slot has a [`ReadFence`]. A reader records it on candle's stream after enqueueing the
//!   kernels that read the slot, and the copy stream waits on it before overwriting.
//! - No upload may start while a reader holds the slot between capturing it and recording its
//!   fence; see [`wait_for_readers`].
//!
//! [`CopyStream::upload`] returns only after its copies have landed. A slot is therefore
//! published fully written, and candle's stream never waits on the copy stream.

use std::sync::Arc;
use std::time::Duration;

/// Block until `owner` is the only reference to its value, i.e. every reader that captured a
/// clone has finished with it (and, by the protocol above, recorded its fence).
pub fn wait_for_readers<T>(owner: &Arc<T>) {
    while Arc::strong_count(owner) > 1 {
        std::thread::sleep(Duration::from_micros(50));
    }
}

/// One copy stream per CUDA device, created on first use and shared by its layers.
#[derive(Default)]
pub struct CopyStreams(std::sync::Mutex<Vec<(candle_core::Device, Arc<CopyStream>)>>);

impl CopyStreams {
    /// `None` off CUDA, or when the stream cannot be created (logged once per device).
    pub fn get(&self, device: &candle_core::Device) -> Option<Arc<CopyStream>> {
        let mut streams = self.0.lock().unwrap();
        if let Some((_, cs)) = streams.iter().find(|(d, _)| d.same_device(device)) {
            return Some(Arc::clone(cs));
        }
        match CopyStream::new(device) {
            Ok(Some(cs)) => {
                let cs = Arc::new(cs);
                streams.push((device.clone(), Arc::clone(&cs)));
                Some(cs)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!(
                    "no CUDA copy stream ({e}); expert uploads will not overlap compute"
                );
                None
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::{Arc, Mutex};

    use candle_core::cuda::cudarc::driver::{result, sys, CudaEvent, CudaStream};
    use candle_core::quantized::cuda::QCudaStorage;
    use candle_core::quantized::{QStorage, QTensor};
    use candle_core::{Device, Error, Result};

    /// Marks the point on candle's stream after which a slot is no longer read.
    pub struct ReadFence {
        /// `None` on non-CUDA devices and for slots that are never overwritten.
        event: Mutex<Option<CudaEvent>>,
    }

    impl ReadFence {
        pub fn new(device: &Device) -> Result<Self> {
            let event = match device {
                Device::Cuda(dev) => Some(
                    dev.cuda_stream()
                        .context()
                        .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                        .map_err(Error::wrap)?,
                ),
                _ => None,
            };
            Ok(Self {
                event: Mutex::new(event),
            })
        }

        pub fn none() -> Self {
            Self {
                event: Mutex::new(None),
            }
        }

        /// Call after enqueueing every kernel that reads the slot. Recording under the lock keeps
        /// the stored event the latest one, whatever order concurrent readers arrive in.
        pub fn record_reads(&self, device: &Device) -> Result<()> {
            let event = self.event.lock().unwrap();
            if let (Some(event), Device::Cuda(dev)) = (event.as_ref(), device) {
                event.record(&dev.cuda_stream()).map_err(Error::wrap)?;
            }
            Ok(())
        }
    }

    /// One expert projection's device memory, overwritten in place.
    pub struct SlotMemory {
        tensor: Arc<QTensor>,
        ptr: sys::CUdeviceptr,
        len: usize,
    }

    impl SlotMemory {
        pub fn tensor(&self) -> &Arc<QTensor> {
            &self.tensor
        }
    }

    pub struct CopyStream {
        stream: Arc<CudaStream>,
        device: Device,
    }

    impl CopyStream {
        /// `None` when `device` is not CUDA.
        pub fn new(device: &Device) -> Result<Option<Self>> {
            let Device::Cuda(dev) = device else {
                return Ok(None);
            };
            let stream = dev
                .cuda_stream()
                .context()
                .new_stream()
                .map_err(Error::wrap)?;
            Ok(Some(Self {
                stream,
                device: device.clone(),
            }))
        }

        /// Device memory for `slots` experts of `stack` (`[E, N, K]` -> `[slots, N, K]`) and one
        /// view per expert, each holding the allocation. Allocated and zeroed on candle's stream;
        /// call [`Self::after_device_work`] before the first upload.
        pub fn alloc_bank(
            &self,
            stack: &QTensor,
            slots: usize,
        ) -> Result<(Arc<QTensor>, Vec<SlotMemory>)> {
            let Device::Cuda(dev) = &self.device else {
                candle_core::bail!("copy stream on a non-CUDA device");
            };
            let dims = stack.shape().dims();
            let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
            let dtype = stack.dtype();
            let stride = n * (k / dtype.block_size()) * dtype.type_size();
            let storage = QCudaStorage::zeros(dev, slots * n * k, dtype)?;
            if storage.storage_size_in_bytes() < slots * stride {
                candle_core::bail!(
                    "bank of {slots} experts got {} bytes, needs {}",
                    storage.storage_size_in_bytes(),
                    slots * stride
                );
            }
            let base = storage.device_ptr()? as sys::CUdeviceptr;
            let tensor = Arc::new(QTensor::new(QStorage::Cuda(storage), (slots, n, k))?);
            let views = (0..slots)
                .map(|i| SlotMemory {
                    tensor: Arc::clone(&tensor),
                    ptr: base + (i * stride) as sys::CUdeviceptr,
                    len: stride,
                })
                .collect();
            Ok((tensor, views))
        }

        /// Device memory shaped like the whole of `stack`.
        pub fn alloc_like(&self, stack: &QTensor) -> Result<SlotMemory> {
            let Device::Cuda(dev) = &self.device else {
                candle_core::bail!("copy stream on a non-CUDA device");
            };
            let storage = QCudaStorage::zeros(dev, stack.shape().elem_count(), stack.dtype())?;
            let len = storage.storage_size_in_bytes();
            let ptr = storage.device_ptr()? as sys::CUdeviceptr;
            let tensor = Arc::new(QTensor::new(
                QStorage::Cuda(storage),
                stack.shape().clone(),
            )?);
            Ok(SlotMemory { tensor, ptr, len })
        }

        /// Wait for everything queued on this stream so far.
        pub fn synchronize(&self) -> Result<()> {
            self.stream.synchronize().map_err(Error::wrap)
        }

        /// Order this stream after everything already queued on candle's stream, i.e. the slot
        /// allocations and their zero-fill.
        pub fn after_device_work(&self) -> Result<()> {
            let Device::Cuda(dev) = &self.device else {
                return Ok(());
            };
            let done = dev.cuda_stream().record_event(None).map_err(Error::wrap)?;
            self.stream.wait(&done).map_err(Error::wrap)
        }

        /// Queue copies of each `src` into its `dst` once the readers are done (`fence`), then
        /// record `done`; returns at once.
        ///
        /// # Safety
        /// Every `src` must be page-locked (or the call blocks until it is staged) and must stay
        /// alive and unmodified until `done` completes.
        pub unsafe fn upload_async(
            &self,
            fence: &ReadFence,
            parts: &[(&SlotMemory, &[u8])],
            done: &CopyDone,
        ) -> Result<()> {
            self.queue_copies(fence, parts)?;
            done.event.record(&self.stream).map_err(Error::wrap)
        }

        /// Queue the copies and return at once, **without** recording a completion event — for a
        /// producer that records one event for a whole batch of copies after queueing them all
        /// (the doorbell preload claims its pass only once the event exists).
        ///
        /// # Safety
        /// As [`Self::upload_async`], minus the event: the caller must record one after the last
        /// copy and order every consumer after it.
        pub unsafe fn queue_async(
            &self,
            fence: &ReadFence,
            parts: &[(&SlotMemory, &[u8])],
        ) -> Result<()> {
            self.queue_copies(fence, parts)
        }

        /// Copy each `src` into its `dst` once the slot's readers are done (`fence`), and return
        /// after the copies have landed.
        pub fn upload(&self, fence: &ReadFence, parts: &[(&SlotMemory, &[u8])]) -> Result<()> {
            self.queue_copies(fence, parts)?;
            self.stream.synchronize().map_err(Error::wrap)
        }

        fn queue_copies(&self, fence: &ReadFence, parts: &[(&SlotMemory, &[u8])]) -> Result<()> {
            self.stream
                .context()
                .bind_to_thread()
                .map_err(Error::wrap)?;
            if let Some(event) = fence.event.lock().unwrap().as_ref() {
                self.stream.wait(event).map_err(Error::wrap)?;
            }
            for (dst, src) in parts {
                if src.len() != dst.len {
                    candle_core::bail!(
                        "slot upload of {} bytes into a {}-byte slot",
                        src.len(),
                        dst.len
                    );
                }
                // SAFETY: `dst.ptr` is a live allocation of `dst.len` bytes owned by the slot;
                // the callers keep `src` alive until the copy completes.
                unsafe { result::memcpy_htod_async(dst.ptr, src, self.stream.cu_stream()) }
                    .map_err(Error::wrap)?;
            }
            Ok(())
        }
    }

    /// Return this device's unused pool blocks to the driver.
    ///
    /// candle allocates through CUDA's stream-ordered pool, which keeps a freed block for reuse
    /// *at the same size*. A mixed-quant GGUF has one expert shape per layer group, so a caller
    /// that frees a whole-layer buffer and then allocates the next shape's (the prefill stream)
    /// accumulates one ~500 MB block per shape and OOMs.
    ///
    /// Synchronises the **context** first, not just candle's stream: `cuMemFreeAsync` runs on
    /// cudarc's dealloc stream, so a free is still pending after `Device::synchronize()` and
    /// `cuMemPoolTrimTo` would not release it.
    pub fn trim_device_pool(device: &Device) -> Result<()> {
        let Device::Cuda(dev) = device else {
            return Ok(());
        };
        let stream = dev.cuda_stream();
        let context = stream.context();
        context.synchronize().map_err(Error::wrap)?;
        // Both pools: the device's *default* pool and the one it is currently set to. A stream's
        // `cuMemAllocAsync` takes blocks from the device's current pool, which is the default
        // unless something set another; trimming only the default released 288 MiB of 1863 in the
        // prefill stream's hand-over, so trim both and never guess.
        let dev = context.ordinal() as i32;
        let mut pools: [sys::CUmemoryPool; 2] = [std::ptr::null_mut(); 2];
        // SAFETY: `context` is live and bound to this device; both handles are owned by CUDA.
        let rc = unsafe { sys::cuDeviceGetDefaultMemPool(&mut pools[0], dev) };
        if rc != sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuDeviceGetDefaultMemPool failed: {rc:?}");
        }
        let rc = unsafe { sys::cuDeviceGetMemPool(&mut pools[1], dev) };
        if rc != sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuDeviceGetMemPool failed: {rc:?}");
        }
        for pool in pools {
            if pool.is_null() {
                continue;
            }
            // SAFETY: live pool of this context; the caller synchronised, so no block is in use.
            let rc = unsafe { sys::cuMemPoolTrimTo(pool, 0) };
            if rc != sys::CUresult::CUDA_SUCCESS {
                candle_core::bail!("cuMemPoolTrimTo failed: {rc:?}");
            }
        }
        Ok(())
    }

    /// Completion of an asynchronous upload, recorded on the copy stream.
    pub struct CopyDone {
        event: CudaEvent,
    }

    impl CopyDone {
        pub fn new(cs: &CopyStream) -> Result<Self> {
            let event = cs
                .stream
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .map_err(Error::wrap)?;
            Ok(Self { event })
        }

        /// Record this completion on the copy stream, after everything queued on it so far.
        /// Used by a producer that queues its copies first and records one event for the batch
        /// (the doorbell preload), rather than one event per batch member.
        ///
        /// Binds the context: a producer that queued nothing has not bound it on this thread yet
        /// (`queue_copies` is the other caller), and `cuEventRecord` needs it.
        pub fn record(&self, cs: &CopyStream) -> Result<()> {
            cs.stream
                .context()
                .bind_to_thread()
                .map_err(Error::wrap)?;
            self.event.record(&cs.stream).map_err(Error::wrap)
        }

        /// Make candle's stream wait for the upload (a GPU-side wait; the host does not block).
        pub fn wait_on(&self, device: &Device) -> Result<()> {
            let Device::Cuda(dev) = device else {
                return Ok(());
            };
            dev.cuda_stream().wait(&self.event).map_err(Error::wrap)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{trim_device_pool, CopyDone, CopyStream, ReadFence, SlotMemory};

#[cfg(not(feature = "cuda"))]
mod stub {
    use std::sync::Arc;

    use candle_core::quantized::QTensor;
    use candle_core::{Device, Result};

    /// Without CUDA there is nothing to fence.
    pub struct ReadFence;

    impl ReadFence {
        pub fn new(_device: &Device) -> Result<Self> {
            Ok(Self)
        }

        pub fn none() -> Self {
            Self
        }

        pub fn record_reads(&self, _device: &Device) -> Result<()> {
            Ok(())
        }
    }

    /// Never constructed without CUDA.
    pub enum SlotMemory {}

    impl SlotMemory {
        pub fn tensor(&self) -> &Arc<QTensor> {
            match *self {}
        }
    }

    /// Never constructed without CUDA.
    pub enum CopyDone {}

    impl CopyDone {
        pub fn new(cs: &CopyStream) -> Result<Self> {
            match *cs {}
        }

        pub fn record(&self, _cs: &CopyStream) -> Result<()> {
            match *self {}
        }

        pub fn wait_on(&self, _device: &Device) -> Result<()> {
            match *self {}
        }
    }

    /// Nothing to trim without CUDA.
    pub fn trim_device_pool(_device: &Device) -> Result<()> {
        Ok(())
    }

    /// Never constructed without CUDA.
    pub enum CopyStream {}

    impl CopyStream {
        pub fn alloc_like(&self, _stack: &QTensor) -> Result<SlotMemory> {
            match *self {}
        }

        pub fn synchronize(&self) -> Result<()> {
            match *self {}
        }

        /// # Safety
        /// See the CUDA version.
        pub unsafe fn upload_async(
            &self,
            _fence: &ReadFence,
            _parts: &[(&SlotMemory, &[u8])],
            _done: &CopyDone,
        ) -> Result<()> {
            match *self {}
        }

        pub fn new(_device: &Device) -> Result<Option<Self>> {
            Ok(None)
        }

        pub fn alloc_bank(
            &self,
            _stack: &QTensor,
            _slots: usize,
        ) -> Result<(Arc<QTensor>, Vec<SlotMemory>)> {
            match *self {}
        }

        pub fn after_device_work(&self) -> Result<()> {
            match *self {}
        }

        pub fn upload(&self, _fence: &ReadFence, _parts: &[(&SlotMemory, &[u8])]) -> Result<()> {
            match *self {}
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use stub::{trim_device_pool, CopyDone, CopyStream, ReadFence, SlotMemory};

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::wait_for_readers;

    /// Does a candle free land back in the driver after `trim_device_pool`, and is the pool
    /// retention real in the first place?
    #[test]
    fn pool_trim_returns_freed_tensors() {
        use candle_core::quantized::QTensor;
        use candle_core::quantized::cuda::QCudaStorage;
        use candle_core::quantized::QStorage;
        let Ok(dev) = candle_core::Device::new_cuda(0) else {
            return;
        };
        let candle_core::Device::Cuda(cdev) = &dev else {
            return;
        };
        let free = |d: &candle_core::Device| -> usize {
            let c = d.as_cuda_device().unwrap().cuda_stream().context().clone();
            c.mem_get_info().unwrap().0
        };
        let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
        let bytes = 400 << 20; // ~400 MiB of f16
        let before = free(&dev);
        let t = {
            let s =
                QCudaStorage::zeros(cdev, bytes / 2, candle_core::quantized::GgmlDType::F16).unwrap();
            QTensor::new(QStorage::Cuda(s), (bytes / 2,)).unwrap()
        };
        let held = free(&dev);
        drop(t);
        super::trim_device_pool(&dev).unwrap();
        let after = free(&dev);
        println!(
            "before {:.0} MiB, with tensor {:.0} MiB, after drop+trim {:.0} MiB",
            mib(before),
            mib(held),
            mib(after)
        );
        // Compare against the *held* measurement, not `before`: other tests in this binary
        // allocate on the same card and run in parallel.
        assert!(
            after.saturating_sub(held) >= bytes * 3 / 4,
            "trim returned {:.0} MiB of the {:.0} MiB tensor",
            mib(after.saturating_sub(held)),
            mib(bytes)
        );
    }

    #[test]
    fn wait_for_readers_returns_once_clones_are_dropped() {
        let owner = Arc::new(());
        let reader = Arc::clone(&owner);
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            drop(reader);
        });
        let start = Instant::now();
        wait_for_readers(&owner);
        assert!(start.elapsed() >= Duration::from_millis(15));
        assert_eq!(Arc::strong_count(&owner), 1);
        t.join().unwrap();
        // already sole owner: returns at once
        wait_for_readers(&owner);
    }
}
