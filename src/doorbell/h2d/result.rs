//! Host-to-device CPU-result doorbell: the worker writes a result slot, the stream waits and copies.

use std::ffi::c_void;
use std::time::Duration;

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{Device, Result, Tensor};

use crate::doorbell::ffi;
#[cfg(all(test, feature = "cuda"))]
use crate::doorbell::mapped::load_u32;
use crate::doorbell::mapped::{store_release_u32, wait_host_ge};
use crate::doorbell::payload_ring::{PayloadRing, DEPTH};

/// The longest a publish waits for the device to free its slot. Copying one result slot is tens of
/// microseconds of PCIe; 2 ms is two orders of margin, and short enough that a stalled stream is
/// reported instead of hanging the worker.
const SLOT_WAIT: Duration = Duration::from_millis(2);

/// `[seq u32][consumed u32][ring: DEPTH * max_n f32]`. The host worker is the only writer; the
/// device wait+copy kernel is the only reader, and writes `consumed = seq` after it has copied a
/// slot out.
///
/// Backpressure is checked, not assumed: `publish` waits (bounded) for the slot's previous lap to
/// be consumed before overwriting it, so a CPU that outruns the GPU cannot silently hand the model
/// the previous job's expert output.
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

    pub fn depth(&self) -> usize {
        DEPTH
    }

    /// Host side: write `values` into slot `seq % DEPTH`, then publish seq with a release fence.
    /// `None` publishes no payload, just the seq bump, so a failed computation's stream-side wait
    /// still unblocks. Returns `false` when the payload was dropped because the ring was still
    /// full; the seq bump is published either way.
    pub fn publish(&self, values: Option<&[f32]>, seq: u32) -> bool {
        let mut written = true;
        if let Some(v) = values {
            assert!(
                v.len() <= self.ring.max_n(),
                "result doorbell publish of {} f32 exceeds the slot capacity ({})",
                v.len(),
                self.ring.max_n()
            );
            if self.reserve(seq) {
                let p = self.ring.slot_host(seq);
                // SAFETY: the slot covers `max_n` f32s; the length is checked above.
                unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, v.len()) };
            } else {
                written = false;
            }
        }
        store_release_u32(self.ring.seq_host(), seq);
        written
    }

    /// Is slot `seq % DEPTH` free to overwrite? Its previous occupant is `seq - DEPTH`, and the
    /// device publishes `consumed` once that slot's copy out has read it.
    fn reserve(&self, seq: u32) -> bool {
        let depth = DEPTH as u32;
        if seq <= depth {
            return true;
        }
        wait_host_ge(
            self.ring.consumed_host(),
            seq - depth,
            Some(SLOT_WAIT),
            Duration::from_micros(50),
        )
    }

    /// Stream side: queue a wait+copy kernel that spins until seq reaches `seq`, then copies slot
    /// `seq % DEPTH` into the freshly allocated CUDA tensor `dst` and publishes `consumed`.
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
                self.ring.consumed_dev(),
                ptr as *mut c_void,
                len as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    #[cfg(all(test, feature = "cuda"))]
    pub fn consumed(&self) -> u32 {
        load_u32(self.ring.consumed_host())
    }
}
