//! Device-to-host top-k doorbell: the router's ids/weights, so the host read needs no D2H launch.

use std::ffi::c_void;
use std::time::Duration;

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Tensor};

use crate::doorbell::ffi;
use crate::doorbell::mapped::{load_f32, load_u32, wait_host_ge, MappedBuf, HEADER_BYTES};

const IDS_OFF: usize = HEADER_BYTES;

/// `[seq u32][pad][ids max_n u32][weights max_n f32]`.
///
/// The device counter is the only sequence: the publish kernel bumps it, and this host reads it
/// before queueing the next publish. Depth 1 means there is never more than one publish
/// outstanding, so "the value before I published, plus one" is exactly the publish to wait for,
/// with no host-side mirror to keep in sync.
pub struct TopkDoorbell {
    buf: MappedBuf,
    max_n: usize,
}

impl TopkDoorbell {
    /// `max_n` f32 of ids plus `max_n` f32 of weights. Sized once at warmup for the largest window
    /// the layer will ever publish (`rows * top_k`).
    pub fn acquire(device: &Device, max_n: usize) -> Result<Option<Self>> {
        if max_n == 0 {
            return Ok(None);
        }
        let Some(bytes) = max_n.checked_mul(8).and_then(|n| n.checked_add(IDS_OFF)) else {
            return Ok(None);
        };
        Ok(MappedBuf::alloc_zeroed(device, bytes)?.map(|buf| Self { buf, max_n }))
    }

    /// Queue a publish, wait for its seq, and return the ids/weights. This is the whole public
    /// round trip: the doorbell owns the sequence, so callers do not carry a parallel `seq`.
    pub fn publish_and_read(
        &self,
        ids: &Tensor,
        weights: &Tensor,
        n: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let seq = self.seq();
        self.publish(ids, weights, n)?;
        Ok(self.wait_and_read(seq, n))
    }

    /// The device counter's current value: the seq of the last completed publish.
    fn seq(&self) -> u32 {
        load_u32(self.buf.host_u32(0))
    }

    /// Queue the ring on the device's current stream: copy `n` ids/weights into the mapped buffer,
    /// then bump the device seq. Stream-ordered after whatever produced the tensors.
    fn publish(&self, ids: &Tensor, weights: &Tensor, n: usize) -> Result<()> {
        if n > self.max_n {
            candle_core::bail!(
                "top-k doorbell publish of {n} f32 exceeds the slot capacity ({})",
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

    /// Wait until the device counter advances past `last_seq`, then copy out the published ids and
    /// weights. No CUDA calls: safe from the token-driving thread. The wait never times out —
    /// stale ids would silently corrupt routing.
    fn wait_and_read(&self, last_seq: u32, n: usize) -> (Vec<u32>, Vec<f32>) {
        assert!(n <= self.max_n, "top-k doorbell read exceeds capacity");
        let want = last_seq.wrapping_add(1);
        // `>=` not `==`: a lagging consumer may see the counter already past `want`.
        let _ = wait_host_ge(
            self.buf.host_u32(0),
            want,
            None,
            Duration::from_micros(50),
        );
        // SAFETY: seq read orders these payload reads (the device fenced before the bump).
        let ids_ptr = self.buf.host_u32(IDS_OFF);
        let w_ptr = unsafe { self.buf.host().add(IDS_OFF + self.max_n * 4).cast::<f32>() };
        let ids = (0..n)
            .map(|i| load_u32(unsafe { ids_ptr.add(i) }))
            .collect();
        let ws = (0..n).map(|i| load_f32(unsafe { w_ptr.add(i) })).collect();
        (ids, ws)
    }
}
