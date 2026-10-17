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

        /// The raw handle, so a producer here can order a stream memory op on it — M4 publishes a
        /// sync point with `cuStreamWriteValue32` right after a batch of copies.
        pub fn cu_stream(&self) -> sys::CUstream {
            self.stream.cu_stream()
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
        /// producer whose completion signal is a doorbell point (`cuStreamWriteValue32` on this
        /// stream, M4) rather than a CUDA event.
        ///
        /// # Safety
        /// As [`Self::upload_async`], minus the event: the caller must order every consumer after
        /// the copies some other way (the doorbell point, or a stream wait on the *reader* side).
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
pub use cuda::{CopyDone, CopyStream, ReadFence, SlotMemory};

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

        pub fn wait_on(&self, _device: &Device) -> Result<()> {
            match *self {}
        }
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
pub use stub::{CopyDone, CopyStream, ReadFence, SlotMemory};

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::wait_for_readers;

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
