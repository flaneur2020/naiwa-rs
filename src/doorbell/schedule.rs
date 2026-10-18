//! v2 per-pass sync schedule (module docs: `super`).

use std::ffi::c_void;

use candle_core::cuda::cudarc::driver::sys;
use candle_core::Result;

use super::ffi;

/// Layer-internal CPU -> GPU sync point ids, in the order the GPU hits them.
pub const SYNC_PRELOAD: u32 = 0;
pub const SYNC_STAGING: u32 = 1;
pub const SYNC_CPU_EXPERTS: u32 = 2;

/// v2 §3.2 final form: one mapped allocation shared by every layer.
///
/// ```text
/// [epoch u32 @0][pad][ready u32 @8: K words][payload @8 + K*4: slots * max_n f32]
/// ```
///
/// Each sync point is **one word**: `ready[id]` holds the pass whose data is ready. The device
/// bumps `epoch` once per pass (layer 0's prologue); a node reads its own word, compares it
/// against the epoch and, for `SYNC_CPU_EXPERTS`, accumulates its payload slot.
///
/// No ring, no double buffer, no backpressure. Correctness rests on one invariant: the host
/// cannot enter pass `k+1` without consuming a GPU-produced signal (today the per-layer router
/// doorbell, which fires after all of pass `k`'s layers), so pass `k`'s payload cannot be
/// overwritten before its consumers ran. See `architecture-v2` §3.2.
pub struct ScheduleBuf {
    host: *mut u8,
    dev_view: *mut c_void,
    slots: usize,
    k: usize,
    max_n: usize,
    waits_per_layer: u32,
}

// SAFETY: mapped memory handed to CUDA; access is ordered by the epoch/ready protocol.
unsafe impl Send for ScheduleBuf {}
unsafe impl Sync for ScheduleBuf {}

impl ScheduleBuf {
    /// `slots` payload slots (one per layer), `waits_per_layer` sync points per layer, `max_n`
    /// floats each. `Ok(None)` off CUDA or on allocation failure.
    pub fn acquire(
        device: &candle_core::Device,
        waits_per_layer: u32,
        slots: usize,
        max_n: usize,
    ) -> Result<Option<ScheduleBuf>> {
        if !device.is_cuda() || max_n == 0 || slots == 0 || waits_per_layer == 0 {
            return Ok(None);
        }
        let k = slots * waits_per_layer as usize;
        let bytes = 8 + k * 4 + slots * max_n * 4;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: fresh allocation of `bytes`; on failure the caller falls back.
        let rc = unsafe { ffi::doorbell_alloc_mapped(&mut host, bytes) };
        if rc != 0 || host.is_null() {
            return Ok(None);
        }
        let mut dev_view: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host` holds a live cudaHostAllocMapped allocation.
        unsafe { ffi::doorbell_get_device_ptr(host, &mut dev_view) };
        if dev_view.is_null() {
            // SAFETY: allocated above, not freed elsewhere yet.
            unsafe { ffi::doorbell_free_host(host) };
            return Ok(None);
        }
        // SAFETY: zero the whole mapping - epoch, every ready word (so the first bump is 0 -> 1 and no
        // point looks ready before its producer ran) *and every payload slot*.
        //
        // The payload area matters: `cudaHostAlloc` does not hand out zeroed memory, and a consumer that
        // passes its gate while the slot still holds garbage adds it to the layer's accumulator. Garbage
        // bits read as f32 are frequently NaN/Inf, which is exactly how a rare `NaN logits` failure
        // looked before this (non-deterministic, ~1 run in 3 at one point, never reproducible on demand).
        unsafe {
            std::ptr::write_bytes(host, 0, bytes);
        }
        Ok(Some(ScheduleBuf {
            host: host.cast(),
            dev_view,
            slots,
            k,
            max_n,
            waits_per_layer,
        }))
    }

    #[cfg(all(test, feature = "cuda"))]
    fn epoch_ptr(&self) -> *mut u32 {
        self.host.cast()
    }

    fn ready_ptr(&self) -> *mut u32 {
        // SAFETY: offset 8 inside the allocation, `k` u32s.
        unsafe { self.host.add(8).cast::<u32>() }
    }

    fn payload_ptr(&self) -> *mut f32 {
        // SAFETY: offset 8 + k*4 inside the allocation, `slots * max_n` f32s.
        unsafe { self.host.add(8 + self.k * 4).cast::<f32>() }
    }

    fn dev_ready_ptr(&self) -> *mut c_void {
        // SAFETY: device view of the same allocation.
        unsafe { self.dev_view.cast::<u8>().add(8) as *mut c_void }
    }

    #[cfg(all(test, feature = "cuda"))]
    fn dev_payload_ptr(&self) -> *mut c_void {
        // SAFETY: as above.
        unsafe {
            self.dev_view
                .cast::<u8>()
                .add(8 + self.k * 4) as *mut c_void
        }
    }

    /// The static id of `(layer, kind)` — the word this sync point owns.
    pub fn sync_id(&self, layer: u32, kind: u32) -> u32 {
        layer * self.waits_per_layer + kind
    }

    /// The last pass the **device** announced. The host normally does *not* read this: it keeps its
    /// own pass counter, which equals the device's for the same pass.
    #[cfg(all(test, feature = "cuda"))]
    pub fn epoch(&self) -> u32 {
        // SAFETY: mapped word, written by the device behind a system fence.
        unsafe { std::ptr::read_volatile(self.epoch_ptr()) }
    }

    /// Queue the prologue: bump the epoch once. Belongs in layer 0's graph, once per pass.
    pub fn bump_epoch(&self, device: &candle_core::Device) -> Result<()> {
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        // SAFETY: the cell is inside the mapped allocation; stream-ordered launch. The kernel
        // writes the **device view**, not the host pointer.
        unsafe {
            ffi::doorbell_epoch_bump(self.dev_view, stream.cu_stream() as i64);
        }
        Ok(())
    }

    /// Publish layer `layer`'s CPU result for pass `pass`: payload first, then the point's word.
    ///
    /// `pass` is the host's own pass counter. It equals the epoch the node will read for this
    /// pass, and the payload slot is single-buffered — safe because the host cannot advance a pass
    /// without a GPU-produced signal (see the type docs).
    pub fn publish_cpu(&self, layer: u32, pass: u32, values: &[f32]) {
        assert!(
            values.len() <= self.max_n,
            "publish of {} f32 exceeds the payload capacity ({})",
            values.len(),
            self.max_n
        );
        let slot = layer as usize;
        assert!(slot < self.slots, "publish for layer {layer} has no slot");
        let p = self.payload_ptr();
        // SAFETY: the slot region is `max_n` f32s inside the allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), p.add(slot * self.max_n), values.len());
        }
        self.publish_point(self.sync_id(layer, SYNC_CPU_EXPERTS), pass);
    }

    /// Mark a point ready for `pass` with no payload — the failure path, and (M4) the DMA-done
    /// points. A waiting node is never left spinning forever.
    pub fn publish_point(&self, id: u32, pass: u32) {
        assert!((id as usize) < self.k, "sync id {id} out of range");
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        // SAFETY: the word is inside the mapped allocation and is read by the device.
        unsafe {
            std::ptr::write_volatile(self.ready_ptr().add(id as usize), pass);
        }
    }

    /// Queue a node: read `*epoch`, spin until `ready[id] >= epoch`, then accumulate `slot`'s
    /// payload into `acc` (which already holds the GPU hits' sum).
    ///
    /// `acc` is a raw device pointer because a *captured* call must resolve it outside the capture
    /// window (a guarded `device_ptr` records an event, which capture forbids).
    #[cfg(all(test, feature = "cuda"))]
    pub fn wait_add(&self, device: &candle_core::Device, n: usize, id: u32, slot: u32, acc: u64) -> Result<()> {
        if n > self.max_n {
            candle_core::bail!(
                "schedule add of {n} f32 exceeds the payload capacity ({})",
                self.max_n
            );
        }
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        // SAFETY: every pointer is inside this allocation or the caller's device buffer; the
        // launch is stream-ordered.
        unsafe {
            ffi::doorbell_wait_add_ready(
                self.host.cast(),
                self.dev_ready_ptr(),
                self.dev_payload_ptr(),
                acc as *mut c_void,
                n as i32,
                id as i32,
                slot as i32,
                self.max_n as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// Read a point's ready word (diagnostics / tests).
    #[cfg(all(test, feature = "cuda"))]
    pub fn ready(&self, id: u32) -> u32 {
        assert!((id as usize) < self.k, "sync id {id} out of range");
        // SAFETY: mapped word.
        unsafe { std::ptr::read_volatile(self.ready_ptr().add(id as usize)) }
    }

    /// M4: mark point `id` ready for `pass` **on `device`'s current stream**, ordered after
    /// whatever was enqueued before it (a DMA, typically). No payload — the data is already in VRAM.
    ///
    /// This is what lets a DMA completion be a graph-visible doorbell: the copy stream publishes the
    /// point itself (`cuStreamWriteValue32`, a stream memory op: no kernel, no host involvement),
    /// and the node in the main-stream graph spins on it.
    #[cfg(all(test, feature = "cuda"))]
    pub fn publish_point_on_stream(
        &self,
        device: &candle_core::Device,
        id: u32,
        pass: u32,
    ) -> Result<()> {
        let dev = device.as_cuda_device()?;
        self.publish_point_on(dev.cuda_stream().cu_stream(), id, pass)
    }

    /// [`Self::publish_point_on_stream`] on a raw stream handle — for a producer that is not on
    /// candle's stream (the copy stream's DMA path, M4).
    pub fn publish_point_on(&self, stream: sys::CUstream, id: u32, pass: u32) -> Result<()> {
        assert!((id as usize) < self.k, "sync id {id} out of range");
        let addr = self.dev_ready_ptr() as sys::CUdeviceptr + (id as usize * 4) as sys::CUdeviceptr;
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

    /// M5b: [`Self::publish_point_on`], but **monotonic** — a one-thread kernel doing
    /// `atomicMax(ready[id], pass)` on the stream, ordered after whatever was queued before it.
    ///
    /// `ready[]` is a level, not an edge, so an unconditional stream write lets a stale producer move
    /// the word *backwards* and strand every later gate on that point. With `atomicMax` a stale,
    /// duplicated or racing publish is harmless — which is what allows the producer to be the
    /// lookahead thread, a pass behind the token thread's counter.
    pub fn publish_point_on_max(
        &self,
        device: &candle_core::Device,
        id: u32,
        pass: u32,
    ) -> Result<()> {
        assert!((id as usize) < self.k, "sync id {id} out of range");
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        let ready = self.dev_ready_ptr();
        // SAFETY: the word is inside the mapped allocation; the launch is stream-ordered.
        unsafe {
            ffi::doorbell_publish_max(ready, id as i32, pass, stream.cu_stream() as i64);
        }
        Ok(())
    }

    /// M4: queue the wait-only node for point `id` (waits for `ready[id] >= *epoch`, nothing else).
    pub fn wait_ready(&self, device: &candle_core::Device, id: u32) -> Result<()> {
        assert!((id as usize) < self.k, "sync id {id} out of range");
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        // SAFETY: both pointers are inside the mapped allocation; the launch is stream-ordered.
        unsafe {
            ffi::doorbell_wait_ready(
                self.dev_view,
                self.dev_ready_ptr(),
                id as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// The common case: the CPU-experts point of `layer`.
    #[cfg(all(test, feature = "cuda"))]
    pub fn wait_add_cpu(&self, device: &candle_core::Device, n: usize, layer: u32, acc: u64) -> Result<()> {
        self.wait_add(device, n, self.sync_id(layer, SYNC_CPU_EXPERTS), layer, acc)
    }
}

impl Drop for ScheduleBuf {
    fn drop(&mut self) {
        if !self.host.is_null() {
            // SAFETY: allocated by doorbell_alloc_mapped, freed nowhere else.
            unsafe { ffi::doorbell_free_host(self.host as *mut c_void) };
        }
    }
}
