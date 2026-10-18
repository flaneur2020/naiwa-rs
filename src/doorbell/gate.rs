//! Device-side pass gates: an epoch (device-written once per pass) and one ready word per gate.
//!
//! A gate is ready when `ready[id] >= epoch`. Producers publish a pass on the host or on a stream;
//! the device waits on the gate inside its graph. There is no payload here: CPU results use
//! [`crate::doorbell::ResultDoorbell`].

use std::ffi::c_void;

use candle_core::cuda::cudarc::driver::sys;
use candle_core::{Device, Result};

#[cfg(all(test, feature = "cuda"))]
use crate::doorbell::mapped::{load_u32, store_release_u32};

use crate::doorbell::ffi;
use crate::doorbell::mapped::MappedBuf;

/// Layer-internal gate ids, in the order the GPU hits them.
pub const SYNC_PRELOAD: u32 = 0;
pub const SYNC_STAGING: u32 = 1;

const EPOCH_OFF: usize = 0;
const READY_OFF: usize = 8;

/// `[epoch u32][pad][ready: slots * waits_per_layer u32]`, one mapping for every layer.
pub struct GateSet {
    buf: MappedBuf,
    waits_per_layer: u32,
    k: usize,
}

impl GateSet {
    /// `slots * waits_per_layer` ready words. `Ok(None)` off CUDA or on allocation failure.
    pub fn acquire(device: &Device, waits_per_layer: u32, slots: usize) -> Result<Option<Self>> {
        if waits_per_layer == 0 || slots == 0 {
            return Ok(None);
        }
        let Some(k) = slots.checked_mul(waits_per_layer as usize) else {
            return Ok(None);
        };
        let Some(bytes) = k.checked_mul(4).and_then(|n| n.checked_add(READY_OFF)) else {
            return Ok(None);
        };
        Ok(MappedBuf::alloc_zeroed(device, bytes)?.map(|buf| Self {
            buf,
            waits_per_layer,
            k,
        }))
    }

    /// The static id of `(layer, kind)` — the word this gate owns.
    pub fn sync_id(&self, layer: u32, kind: u32) -> u32 {
        let id = layer * self.waits_per_layer + kind;
        assert!((id as usize) < self.k, "sync id {id} out of range");
        id
    }

    /// Queue the prologue: bump the epoch once. Belongs in layer 0's graph, once per pass.
    pub fn bump_epoch(&self, device: &Device) -> Result<()> {
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        // SAFETY: the cell is inside the mapped allocation; stream-ordered launch. The kernel
        // writes the device view, not the host pointer.
        unsafe {
            ffi::doorbell_epoch_bump(self.buf.dev(EPOCH_OFF), stream.cu_stream() as i64);
        }
        Ok(())
    }

    pub fn gate(&self, id: u32) -> Gate<'_> {
        assert!((id as usize) < self.k, "sync id {id} out of range");
        Gate { set: self, id }
    }

    #[cfg(all(test, feature = "cuda"))]
    pub fn epoch(&self) -> u32 {
        load_u32(self.buf.host().cast())
    }

    fn epoch_dev(&self) -> *mut c_void {
        self.buf.dev(EPOCH_OFF)
    }

    #[cfg(all(test, feature = "cuda"))]
    fn ready_host(&self, id: u32) -> *mut u32 {
        // SAFETY: `id < k` and the ready words follow the epoch word.
        unsafe { self.buf.host().add(READY_OFF + id as usize * 4).cast() }
    }

    fn ready_dev(&self, id: u32) -> *mut c_void {
        self.buf.dev(READY_OFF + id as usize * 4)
    }

    fn ready_base_dev(&self) -> *mut c_void {
        self.buf.dev(READY_OFF)
    }
}

#[derive(Clone, Copy)]
pub struct Gate<'a> {
    set: &'a GateSet,
    id: u32,
}

impl Gate<'_> {
    /// Host producer: publish `pass` with a release fence.
    #[cfg(all(test, feature = "cuda"))]
    pub fn publish_host(&self, pass: u32) {
        store_release_u32(self.set.ready_host(self.id), pass);
    }

    /// Stream producer: `cuStreamWriteValue32`, ordered after whatever was enqueued before it.
    pub fn publish_stream(&self, stream: sys::CUstream, pass: u32) -> Result<()> {
        let addr = self.set.ready_dev(self.id) as sys::CUdeviceptr;
        // SAFETY: the word is inside the mapped allocation; the write is ordered on the stream.
        let rc = unsafe {
            sys::cuStreamWriteValue32_v2(
                stream,
                addr,
                pass,
                sys::CUstreamWriteValue_flags_enum::CU_STREAM_WRITE_VALUE_DEFAULT as u32,
            )
        };
        if rc != sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuStreamWriteValue32 failed: {rc:?}");
        }
        Ok(())
    }

    /// Monotonic stream publish for a producer that may be a pass behind the token thread.
    /// `atomicMax(ready[id], pass)` keeps a stale publish from moving the level backwards.
    pub fn publish_stream_max(&self, device: &Device, pass: u32) -> Result<()> {
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        // SAFETY: the word is inside the mapped allocation; the launch is stream-ordered.
        unsafe {
            ffi::doorbell_publish_max(
                self.set.ready_base_dev(),
                self.id as i32,
                pass,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// Device consumer: queue a wait node that spins until `ready[id] >= *epoch`.
    pub fn wait(&self, device: &Device) -> Result<()> {
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        // SAFETY: both pointers are inside the mapped allocation; the launch is stream-ordered.
        unsafe {
            ffi::doorbell_wait_ready(
                self.set.epoch_dev(),
                self.set.ready_base_dev(),
                self.id as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    #[cfg(all(test, feature = "cuda"))]
    pub fn ready(&self) -> u32 {
        load_u32(self.set.ready_host(self.id))
    }
}
