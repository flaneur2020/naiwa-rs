//! Device-to-host top-k doorbell: the router's ids/weights, so the host read needs no D2H launch.

use std::ffi::c_void;
use std::time::Duration;

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Tensor};

use crate::doorbell::ffi;
use crate::doorbell::mapped::{load_f32, load_u32, wait_host_ge, MappedBuf};

const IDS_OFF: usize = 8;

/// `[seq u32][pad][ids max_n u32][weights max_n f32]`.
pub struct TopkDoorbell {
    buf: MappedBuf,
    max_n: usize,
}

impl TopkDoorbell {
    pub fn acquire(device: &Device, max_n: usize) -> Result<Option<Self>> {
        if max_n == 0 {
            return Ok(None);
        }
        let Some(bytes) = max_n.checked_mul(8).and_then(|n| n.checked_add(IDS_OFF)) else {
            return Ok(None);
        };
        Ok(MappedBuf::alloc_zeroed(device, bytes)?.map(|buf| Self { buf, max_n }))
    }

    pub fn max_n(&self) -> usize {
        self.max_n
    }

    /// Queue the ring on the device's current stream: copy `n` ids/weights into the mapped buffer,
    /// then bump seq. Stream-ordered after whatever produced the tensors.
    pub fn publish(&self, ids: &Tensor, weights: &Tensor, n: usize) -> Result<()> {
        if n > self.max_n {
            candle_core::bail!(
                "top-k doorbell publish of {n} exceeds the slot capacity ({})",
                self.max_n
            );
        }
        let dev = ids.device().as_cuda_device()?;
        let stream = dev.cuda_stream();

        let ids32 = ids.to_dtype(DType::U32)?.contiguous()?;
        let w32 = weights.to_dtype(DType::F32)?.contiguous()?;

        let (ids_storage, ids_layout) = ids32.storage_and_layout();
        let candle_core::Storage::Cuda(ids_s) = &*ids_storage else {
            candle_core::bail!("top-k doorbell: ids not on CUDA");
        };
        let ids_slice = ids_s.as_cuda_slice::<u32>()?;
        let ids_view = ids_slice.slice(ids_layout.start_offset()..);
        let (ids_ptr, _ids_guard) = ids_view.device_ptr(&stream);

        let (w_storage, w_layout) = w32.storage_and_layout();
        let candle_core::Storage::Cuda(w_s) = &*w_storage else {
            candle_core::bail!("top-k doorbell: weights not on CUDA");
        };
        let w_slice = w_s.as_cuda_slice::<f32>()?;
        let w_view = w_slice.slice(w_layout.start_offset()..);
        let (w_ptr, _w_guard) = w_view.device_ptr(&stream);

        // SAFETY: the device pointers come from the tensors; the mapped destination lies inside
        // this allocation; the launch is ordered on the current stream.
        unsafe {
            ffi::doorbell_ring(
                self.buf.dev(0),
                self.buf.dev(IDS_OFF),
                self.buf.dev(IDS_OFF + self.max_n * 4),
                ids_ptr as *const c_void,
                w_ptr as *const c_void,
                n as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// Wait until seq advances past `last_seq`, then copy out the published ids and weights.
    /// No CUDA calls: safe from the token-driving thread. The wait never times out — stale ids
    /// would silently corrupt routing.
    pub fn wait_and_read(&self, last_seq: u32, n: usize) -> (u32, Vec<u32>, Vec<f32>) {
        assert!(n <= self.max_n, "top-k doorbell read exceeds capacity");
        let want = last_seq.wrapping_add(1);
        // `>=` not `==`: a lagging consumer may see the counter already past `want`.
        let _ = wait_host_ge(
            self.buf.host().cast(),
            want,
            None,
            Duration::from_micros(50),
        );
        // SAFETY: seq read orders these payload reads (the device fenced before the bump).
        let ids_ptr = unsafe { self.buf.host().add(IDS_OFF).cast::<u32>() };
        let w_ptr = unsafe { self.buf.host().add(IDS_OFF + self.max_n * 4).cast::<f32>() };
        let ids = (0..n)
            .map(|i| load_u32(unsafe { ids_ptr.add(i) }))
            .collect();
        let ws = (0..n).map(|i| load_f32(unsafe { w_ptr.add(i) })).collect();
        (want, ids, ws)
    }
}
