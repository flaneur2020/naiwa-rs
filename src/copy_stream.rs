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

        /// Device memory shaped like one expert of `stack` (`[E, N, K]` -> `[N, K]`). Allocated
        /// and zeroed on candle's stream; call [`Self::after_device_work`] before the first upload.
        pub fn alloc_expert(&self, stack: &QTensor) -> Result<SlotMemory> {
            let Device::Cuda(dev) = &self.device else {
                candle_core::bail!("copy stream on a non-CUDA device");
            };
            let dims = stack.shape().dims();
            let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
            let storage = QCudaStorage::zeros(dev, n * k, stack.dtype())?;
            let len = storage.storage_size_in_bytes();
            let ptr = storage.device_ptr()? as sys::CUdeviceptr;
            let tensor = Arc::new(QTensor::new(QStorage::Cuda(storage), (n, k))?);
            Ok(SlotMemory { tensor, ptr, len })
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

        /// Copy each `src` into its `dst` once the slot's readers are done (`fence`), and return
        /// after the copies have landed.
        pub fn upload(&self, fence: &ReadFence, parts: &[(&SlotMemory, &[u8])]) -> Result<()> {
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
                // SAFETY: `dst.ptr` is a live allocation of `dst.len` bytes owned by the slot,
                // and `src` outlives the copy: this function waits for it below.
                unsafe { result::memcpy_htod_async(dst.ptr, src, self.stream.cu_stream()) }
                    .map_err(Error::wrap)?;
            }
            self.stream.synchronize().map_err(Error::wrap)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{CopyStream, ReadFence, SlotMemory};

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
    pub enum CopyStream {}

    impl CopyStream {
        pub fn new(_device: &Device) -> Result<Option<Self>> {
            Ok(None)
        }

        pub fn alloc_expert(&self, _stack: &QTensor) -> Result<SlotMemory> {
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
pub use stub::{CopyStream, ReadFence, SlotMemory};

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
