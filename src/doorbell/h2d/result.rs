//! Host-to-device CPU-result doorbell: the worker writes a result slot, the stream waits and copies.

use std::ffi::c_void;

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{Device, Result, Tensor};

use crate::doorbell::ffi;
use crate::doorbell::mapped::store_release_u32;
use crate::doorbell::payload_ring::PayloadRing;

/// `[seq u32][pad][ring: DEPTH * max_n f32]`. The host worker is the only writer; the device
/// wait+copy kernel is the only reader.
///
/// The producer must not run more than the doorbell ring depth jobs ahead of the stream. In the shipped path
/// the worker's CPU expert gemv is slower than the GPU's copy, so it stays well inside the ring;
/// this is the one invariant the primitive cannot check without a device-written consumption
/// counter.
pub struct ResultDoorbell {
    ring: PayloadRing,
}

impl ResultDoorbell {
    pub fn acquire(device: &Device, max_n: usize) -> Result<Option<Self>> {
        Ok(PayloadRing::acquire(device, max_n)?.map(|ring| Self { ring }))
    }

    pub fn max_n(&self) -> usize {
        self.ring.max_n()
    }

    /// Host side: write `values` into slot `seq % DEPTH`, then publish seq with a release fence.
    /// `None` publishes no payload, just the seq bump, so a failed computation's stream-side wait
    /// still unblocks.
    pub fn publish(&self, values: Option<&[f32]>, seq: u32) {
        let p = self.ring.slot_host(seq);
        // SAFETY: the slot covers `max_n` f32s; the caller checks the length.
        unsafe {
            if let Some(v) = values {
                assert!(
                    v.len() <= self.ring.max_n(),
                    "result doorbell publish of {} f32 exceeds the slot capacity ({})",
                    v.len(),
                    self.ring.max_n()
                );
                std::ptr::copy_nonoverlapping(v.as_ptr(), p, v.len());
            }
        }
        store_release_u32(self.ring.seq_host(), seq);
    }

    /// Stream side: queue a wait+copy kernel that spins until seq reaches `seq`, then copies slot
    /// `seq % DEPTH` into the freshly allocated CUDA tensor `dst`.
    pub fn wait_copy(&self, seq: u32, len: usize, dst: &Tensor) -> Result<()> {
        if len > self.ring.max_n() {
            candle_core::bail!(
                "result doorbell copy of {len} f32 exceeds the slot capacity ({})",
                self.ring.max_n()
            );
        }
        let dev = dst.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let (storage, layout) = dst.storage_and_layout();
        let candle_core::Storage::Cuda(s) = &*storage else {
            candle_core::bail!("result doorbell: destination is not on CUDA");
        };
        let slice = s.as_cuda_slice::<f32>()?;
        let view = slice.slice(layout.start_offset()..);
        let (ptr, _guard) = view.device_ptr(&stream);
        // SAFETY: the destination comes from the tensor; `slot_dev` is the mapped slot; the launch
        // is stream-ordered.
        unsafe {
            ffi::doorbell_wait_copy_f32(
                self.ring.seq_dev(),
                seq,
                self.ring.slot_dev(seq),
                0,
                ptr as *mut c_void,
                len as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }
}
