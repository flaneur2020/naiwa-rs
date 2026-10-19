//! Device-to-host activation doorbell for the lookahead: a ring of f32 slots.

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Duration;

use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::cuda_backend::cudarc::driver::{CudaEvent, CudaStream, DevicePtr};
use candle_core::{DType, Device, Result, Tensor};

use crate::doorbell::ffi;
use crate::doorbell::mapped::{load_f32, wait_host_ge};
use crate::doorbell::payload_ring::{PayloadRing, DEPTH};

/// `[seq u32][pad][ring: DEPTH * max_n f32]`. The device publish kernel is the only writer; the
/// host only reads after seq advances.
pub struct ActivationDoorbell {
    ring: PayloadRing,
    /// Publish side stream: publish kernels are ordered after `src`'s producers by an event, so
    /// they never sit behind the main stream's result-wait kernels.
    side: Arc<CudaStream>,
    /// One persistent ordering event, created once instead of per publish (a per-publish event
    /// was `cuEventCreate` + `cuEventDestroy` with timing enabled). Reuse is safe because
    /// `publish` records it and enqueues the side stream's wait on the same thread, so every wait
    /// is enqueued after its own record.
    ready: CudaEvent,
}

impl ActivationDoorbell {
    /// Both the mapped ring and the side stream/event are created here, at warmup: a failure is a
    /// missing channel (`Ok(None)` → the caller keeps its blocking fallback), never a panic on the
    /// decode path.
    pub fn acquire(device: &Device, max_n: usize) -> Result<Option<Self>> {
        let Some(ring) = PayloadRing::acquire(device, max_n)? else {
            return Ok(None);
        };
        let dev = device.as_cuda_device()?;
        let main = dev.cuda_stream();
        let context = main.context();
        let (Ok(side), Ok(ready)) = (
            context.new_stream(),
            context.new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)),
        ) else {
            return Ok(None);
        };
        Ok(Some(Self { ring, side, ready }))
    }

    pub fn max_n(&self) -> usize {
        self.ring.max_n()
    }

    pub fn depth(&self) -> usize {
        DEPTH
    }

    /// Queue a publish of `[n]` f32 from `src` into slot `seq % DEPTH`; the kernel bumps seq.
    /// `seq` is the caller's job id, so the host knows which slot and which seq to wait for.
    pub fn publish(&self, src: &Tensor, n: usize, seq: u32) -> Result<()> {
        if n > self.ring.max_n() {
            candle_core::bail!(
                "activation doorbell publish of {n} f32 exceeds the slot capacity ({})",
                self.ring.max_n()
            );
        }
        let src = src.to_dtype(DType::F32)?.contiguous()?;
        let dev = src.device().as_cuda_device()?;
        let main = dev.cuda_stream();
        // Order the publish after `src`'s producers only.
        self.ready.record(&main).map_err(candle_core::Error::wrap)?;
        self.side
            .wait(&self.ready)
            .map_err(candle_core::Error::wrap)?;

        let (storage, layout) = src.storage_and_layout();
        let candle_core::Storage::Cuda(s) = &*storage else {
            candle_core::bail!("activation doorbell: src not on CUDA");
        };
        let slice = s.as_cuda_slice::<f32>()?;
        let view = slice.slice(layout.start_offset()..);
        let (ptr, _guard) = view.device_ptr(&main);
        // SAFETY: the source pointer comes from the tensor and stays alive while the side stream's
        // kernel runs behind the `ready` event recorded after its producers; the mapped
        // destination is inside this allocation.
        unsafe {
            ffi::doorbell_copy_f32(
                self.ring.seq_dev(),
                self.ring.slot_dev(seq),
                ptr as *const c_void,
                n as i32,
                self.side.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// Wait for the publish whose job id is `seq` (the device counter must reach `seq + 1`), then
    /// read slot `seq % DEPTH`. `None` after 250 ms: the prediction is advisory, so the caller
    /// treats the payload as lost instead of burning a core.
    pub fn wait_publish(&self, seq: u32, n: usize) -> Option<Vec<f32>> {
        if n > self.ring.max_n() {
            return None;
        }
        let want = seq.wrapping_add(1);
        if !wait_host_ge(
            self.ring.seq_host(),
            want,
            Some(Duration::from_millis(250)),
            Duration::from_micros(100),
        ) {
            return None;
        }
        let p = self.ring.slot_host(seq);
        Some((0..n).map(|i| load_f32(unsafe { p.add(i) })).collect())
    }
}
