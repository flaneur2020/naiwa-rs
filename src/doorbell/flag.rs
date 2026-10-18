//! Mapped flag channel: GPU -> host activations and host -> GPU CPU-expert results
//! (module docs: `super`).

use std::ffi::c_void;

use candle_core::{Result, Tensor};

use super::ffi;

/// A generic mapped flag channel (R2): a seq counter at offset 0 plus `max_n`
/// f32 payload slots, host and device views of the same bytes.
///
/// GPU -> host: [`FlagBuf::publish_f32`] queues a stream-ordered kernel that
/// copies a device tensor into the payload and bumps seq; the host spins in
/// [`FlagBuf::wait_publish`] — no stream sync, no D2H launch.
///
/// host -> GPU: [`FlagBuf::write_slot`] writes the payload (plain stores, in
/// publish order) and bumps seq with a release store; [`FlagBuf::wait_copy_slot`]
/// queues the stream-ordered wait+copy kernel. The host never blocks the stream
/// with a host function (graph-capturable, unlike `cudaLaunchHostFunc`).
pub struct FlagBuf {
    host: *mut u8,
    dev_view: *mut c_void,
    // layout: [seq: u32 (offset 0)] [pad (4)] [payload ring at offset 8:
    //          RING_DEPTH slots of max_n f32 each]
    //         [target: u32] [pad (4)] [slot_off: i64]  <- F3/v2 control cells, appended so
    //          every pre-existing offset above is untouched
    max_n: usize,
    /// The publish side stream (GPU -> host leg): publish kernels queue here,
    /// ordered behind `src`'s producers by a recorded event, so they never sit
    /// behind the main stream's result-wait kernels. `FlagBuf::acquire` only
    /// takes `&self`, so the stream is lazily created on first publish.
    side_stream: std::sync::OnceLock<std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>>,
}

impl FlagBuf {
    /// Payload ring depth: one slot being consumed, the rest absorbing publishes
    /// the worker has not picked up yet (the GPU->host leg's worker lags while
    /// queued behind other jobs).
    pub const RING_DEPTH: usize = 8;
}

// SAFETY: mapped memory handed to CUDA; access is ordered by the seq protocol.
unsafe impl Send for FlagBuf {}
unsafe impl Sync for FlagBuf {}

impl FlagBuf {
    /// Allocate a channel whose payload slot holds up to `max_n` f32 values
    /// (times `RING_DEPTH` slots for the host -> GPU leg). `Ok(None)` when the
    /// device is not CUDA or the mapped allocation fails.
    pub fn acquire(device: &candle_core::Device, max_n: usize) -> Result<Option<FlagBuf>> {
        if !device.is_cuda() {
            return Ok(None);
        }
        let bytes = 8 + max_n * Self::RING_DEPTH * 4 + 16;
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
        // SAFETY: zero the seq counter so the first publish advances 0 -> 1.
        unsafe { std::ptr::write_volatile(host.cast::<u32>(), 0) };
        Ok(Some(FlagBuf {
            host: host.cast(),
            dev_view,
            max_n,
            side_stream: std::sync::OnceLock::new(),
        }))
    }

    fn seq_ptr(&self) -> *mut u32 {
        self.host.cast()
    }

    fn payload_ptr(&self) -> *mut f32 {
        // SAFETY: offset 8 inside the allocation, `max_n * RING_DEPTH` f32s.
        unsafe { self.host.add(8).cast::<f32>() }
    }

    fn dev_ptr(&self, off: usize) -> *mut c_void {
        // SAFETY: the device view of the same allocation, offset by `off`.
        unsafe { self.dev_view.cast::<u8>().add(off) as *mut c_void }
    }

    /// F3/v2: offset of the mapped `target` cell (a u32), past the payload ring.
    #[cfg(all(test, feature = "cuda"))]
    fn target_off(&self) -> usize {
        8 + self.max_n * Self::RING_DEPTH * 4
    }

    /// F3/v2: offset of the mapped `slot_off` cell (an i64), after `target` + padding.
    #[cfg(all(test, feature = "cuda"))]
    fn slot_off_off(&self) -> usize {
        self.target_off() + 8
    }

    /// F3/v2: record the seq value this channel's stream-side wait targets. The wait kernels read
    /// it from mapped memory rather than taking it as a kernel argument, so a captured graph can
    /// be replayed with a fresh target each step (a kernel argument is frozen at capture).
    #[cfg(all(test, feature = "cuda"))]
    pub fn set_target(&self, target: u32) {
        // SAFETY: the cell lies inside the mapped allocation; a plain volatile store suffices for
        // the device to read it (the stream-ordered kernel is launched after this call).
        unsafe {
            std::ptr::write_volatile(self.host.add(self.target_off()).cast::<u32>(), target)
        }
    }

    /// F3/v2: record which payload slot the stream-side copy reads, as a float offset into the
    /// ring. Written before queueing [`Self::wait_copy_slot_ptr`].
    #[cfg(all(test, feature = "cuda"))]
    pub fn set_slot_off(&self, slot: usize) {
        let off = (slot % Self::RING_DEPTH) as i64 * self.max_n as i64;
        // SAFETY: as `set_target`.
        unsafe {
            std::ptr::write_volatile(self.host.add(self.slot_off_off()).cast::<i64>(), off)
        }
    }

    /// F3/v2: zero-argument wait: the target comes from the mapped cell ([`Self::set_target`])
    /// instead of a kernel argument — capture-safe.
    #[cfg(all(test, feature = "cuda"))]
    pub fn wait_seq_ge_ptr(&self, device: &candle_core::Device) -> Result<()> {
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        let target = self.dev_ptr(self.target_off());
        // SAFETY: both pointers are inside the mapped allocation; stream-ordered launch.
        unsafe {
            ffi::doorbell_wait_seq_ge_ptr(self.dev_ptr(0), target, stream.cu_stream() as i64);
        }
        Ok(())
    }

    /// F3/v2: like [`Self::wait_copy_slot`], but the target and slot offset come from the mapped
    /// cells ([`Self::set_target`] / [`Self::set_slot_off`]) instead of kernel arguments.
    #[cfg(all(test, feature = "cuda"))]
    pub fn wait_copy_slot_ptr(&self, len: usize, dst: &Tensor) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        if len > self.max_n {
            candle_core::bail!(
                "flag copy of {len} f32 exceeds the slot capacity ({})",
                self.max_n
            );
        }
        let dev = dst.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let (storage, layout) = dst.storage_and_layout();
        let candle_core::Storage::Cuda(s) = &*storage else {
            candle_core::bail!("flag result destination is not on CUDA");
        };
        let slice = s.as_cuda_slice::<f32>()?;
        let view = slice.slice(layout.start_offset()..);
        let (ptr, _guard) = view.device_ptr(&stream);
        let target = self.dev_ptr(self.target_off());
        let slot_off = self.dev_ptr(self.slot_off_off());
        // SAFETY: destination comes from the tensor; the launch is stream-ordered.
        unsafe {
            ffi::doorbell_wait_copy_f32_ptr(
                self.dev_ptr(0),
                target,
                self.dev_ptr(8),
                slot_off,
                ptr as *mut c_void,
                len as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// M3: like [`Self::wait_copy_slot_ptr`], but **accumulates** the published payload into
    /// `acc` (read-modify-write) instead of overwriting. `acc` is typically the MoE graph's
    /// output buffer, which already holds the GPU hits' sum, so one node folds the CPU experts'
    /// contribution in and the graph's output is the completed MoE result.
    #[cfg(all(test, feature = "cuda"))]
    pub fn wait_add_slot_ptr(&self, len: usize, acc: &Tensor) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        if len > self.max_n {
            candle_core::bail!(
                "flag add of {len} f32 exceeds the slot capacity ({})",
                self.max_n
            );
        }
        let dev = acc.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let (storage, layout) = acc.storage_and_layout();
        let candle_core::Storage::Cuda(s) = &*storage else {
            candle_core::bail!("flag accumulator is not on CUDA");
        };
        let slice = s.as_cuda_slice::<f32>()?;
        let view = slice.slice(layout.start_offset()..);
        let (ptr, _guard) = view.device_ptr(&stream);
        let target = self.dev_ptr(self.target_off());
        let slot_off = self.dev_ptr(self.slot_off_off());
        // SAFETY: accumulator comes from the tensor; the launch is stream-ordered.
        unsafe {
            ffi::doorbell_wait_add_f32_ptr(
                self.dev_ptr(0),
                target,
                self.dev_ptr(8),
                slot_off,
                ptr as *mut c_void,
                len as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// M3: accumulate `src`'s published payload slot into `acc`, with the **job number
    /// claimed on the device** from
    /// `counter` (`atomicAdd`, 1-based so it matches the publisher's `publish_seq`) and both the
    /// target seq and the payload slot are derived from it: `target = n`,
    /// `slot = n % RING_DEPTH`.
    ///
    /// Nothing per-step is read from the host, so this is safe inside a captured graph even
    /// though the token thread runs arbitrarily far ahead of the GPU — the exact failure mode a
    /// host-written cell has (measured: it made `--pcie-frac 0` nondeterministic). `counter` is
    /// shared by every node that consumes this channel; the nodes are stream-ordered, so the k-th
    /// node claims job k.
    ///
    /// The publisher's obligation is unchanged from `wait_copy_slot`: it must not publish more
    /// than `RING_DEPTH` jobs ahead of the device, or a slot is overwritten before it is read.
    #[cfg(all(test, feature = "cuda"))]
    pub fn wait_add_indexed_from(
        &self,
        device: &candle_core::Device,
        counter: u64,
        len: usize,
        acc: u64,
    ) -> Result<()> {
        if len > self.max_n {
            candle_core::bail!(
                "flag add of {len} f32 exceeds the channel capacity ({})",
                self.max_n
            );
        }
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        // SAFETY: seq/payload are inside this channel's mapped allocation; `counter` and `acc`
        // are device pointers supplied by the caller; the launch is stream-ordered.
        unsafe {
            ffi::doorbell_wait_add_indexed_f32(
                self.dev_ptr(0),
                counter as *mut c_void,
                self.dev_ptr(8),
                acc as *mut c_void,
                len as i32,
                Self::RING_DEPTH as i32,
                self.max_n as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    pub fn max_n(&self) -> usize {
        self.max_n
    }

    /// Queue the publish on a SIDE stream: the main stream records an event after
    /// whatever produced `src`, the side stream waits for it, then copies `n` f32
    /// values from the CUDA tensor into payload slot `slot` of the mapped ring and
    /// bumps seq.
    ///
    /// Not the main stream (W106): the publish used to queue behind everything
    /// else there — including the result-wait kernel that spins for the CPU
    /// experts — so a busy step delayed every lookahead publish by the miss
    /// latency, and the host-side `wait_publish` timed out and dropped the
    /// prediction (hit 92 -> 86%). The side stream runs the publish as soon as
    /// `src` itself is ready, independent of the main stream's tail.
    pub fn publish_f32(&self, src: &Tensor, n: usize, slot: usize) -> Result<()> {

        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let src = src.to_dtype(candle_core::DType::F32)?.contiguous()?;
        let dev = src.device().as_cuda_device()?;
        // The side stream is created once per FlagBuf: publish kernels are
        // serialized on it (slot ordering is per-channel), so one stream is enough.
        let side = self.side_stream.get_or_init(|| {
            dev.cuda_stream()
                .context()
                .new_stream()
                .expect("create the flag publish side stream")
        });
        let main = dev.cuda_stream();
        // Order the publish after `src`'s producers only.
        let ready = main
            .record_event(None)
            .map_err(candle_core::Error::wrap)?;
        side.wait(&ready).map_err(candle_core::Error::wrap)?;
        // Real check, not a debug_assert: an oversized publish overflows the slot,
        // and on the last ring slot it writes past the whole allocation into the
        // neighboring mapped buffer (W106: the doorbell's ids read back as the
        // activation's f32 bits).
        if n > self.max_n {
            candle_core::bail!(
                "flag publish of {n} f32 exceeds the slot capacity ({})",
                self.max_n
            );
        }
        let slot_off = (slot % Self::RING_DEPTH) * self.max_n;
        let (storage, layout) = src.storage_and_layout();
        let candle_core::Storage::Cuda(s) = &*storage else {
            candle_core::bail!("flag publish: src not on CUDA");
        };
        let slice = s.as_cuda_slice::<f32>()?;
        let view = slice.slice(layout.start_offset()..);
        let (ptr, _guard) = view.device_ptr(&main);
        // SAFETY: the source pointer comes from the tensor and stays alive while
        // the side stream's kernel runs behind the `ready` event recorded after
        // its producers on the tensor's stream; the mapped destination is inside
        // this allocation.
        unsafe {
            ffi::doorbell_copy_f32(
                self.dev_ptr(0),
                self.dev_ptr(8 + slot_off * 4),
                ptr as *const c_void,
                n as i32,
                side.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// Wait until seq advances past `last_seq`, then copy `n` values out of payload
    /// slot `slot`. No CUDA calls — safe from any host thread.
    ///
    /// Returns `None` if the publish has not landed within `TIMEOUT_MS`: the queued
    /// publish kernel may sit behind a long-running kernel on the stream (e.g. a
    /// result-copy wait), and burning CPU on a spin would steal cores from the CPU
    /// experts — the caller treats the payload as lost (the lookahead prediction is
    /// advisory). Spin briefly, then sleep-poll (same hybrid as `wait_and_read`).
    ///
    /// W106: raised 50 -> 250ms — even with the publish on its own side stream, a
    /// large verify batch's miss computation can delay the next step's publishers
    /// for tens of ms; 50ms dropped enough predictions to cost 3+ points of hit.
    pub fn wait_publish(&self, last_seq: u32, n: usize, slot: usize) -> Option<Vec<f32>> {
        const TIMEOUT_MS: u64 = 250;
        let want = last_seq.wrapping_add(1);
        let seq = self.seq_ptr();
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(TIMEOUT_MS);
        // ~50µs of spin first: the publish may already be in flight.
        let mut spins: u32 = 0;
        const SPIN_BUDGET: u32 = 2000;
        loop {
            // SAFETY: mapped memory the device writes; read_volatile each poll.
            // `>=` not `==`: a lagging consumer may see the counter already past
            // `want` (later publish); exact match would never match again.
            let v = unsafe { std::ptr::read_volatile(seq) };
            if v >= want {
                break;
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            if spins < SPIN_BUDGET {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        let p = self.payload_ptr();
        let slot_off = (slot % Self::RING_DEPTH) * self.max_n;
        // SAFETY: the seq read orders these payload reads (device fenced first).
        Some((0..n)
            .map(|i| unsafe { std::ptr::read_volatile(p.add(slot_off + i)) })
            .collect())
    }

    /// Host -> GPU result leg (R2b), publish side (worker thread): write
    /// `values` into payload slot `slot` and bump seq to `publish_seq`.
    /// `None` publishes no payload — just the seq bump, for a failed
    /// computation whose stream-side wait must still unblock.
    pub fn write_slot(&self, values: Option<&[f32]>, slot: usize, publish_seq: u32) {
        let slot_off = slot % Self::RING_DEPTH;
        let p = self.payload_ptr();
        // SAFETY: the slot region is `max_n` f32s inside the allocation; payload
        // stores first, then the release fence + volatile seq store publishes.
        unsafe {
            if let Some(v) = values {
                // Real check: a host overflow writes past the whole mapped
                // allocation into its neighbor (same failure mode as the GPU leg).
                assert!(
                    v.len() <= self.max_n,
                    "flag publish of {} f32 exceeds the slot capacity ({})",
                    v.len(),
                    self.max_n
                );
                std::ptr::copy_nonoverlapping(
                    v.as_ptr(),
                    p.add(slot_off * self.max_n),
                    v.len(),
                );
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(self.seq_ptr(), publish_seq);
        }
    }

    /// Host -> GPU result leg (R2b), stream side (caller's thread): queue the
    /// stream-ordered wait+copy kernel that waits for `publish_seq` then copies
    /// `len` f32 from payload slot `slot` into the CUDA tensor `dst` (freshly
    /// allocated, fully overwritten by the copy). Replaces the wait-for-worker
    /// host function + H2D memcpy; graph-capturable. The ring depth >= 2 keeps
    /// a slow GPU reading slot `slot` from racing the next publish.
    pub fn wait_copy_slot(
        &self,
        slot: usize,
        publish_seq: u32,
        len: usize,
        dst: &Tensor,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        if len > self.max_n {
            candle_core::bail!(
                "flag copy of {len} f32 exceeds the slot capacity ({})",
                self.max_n
            );
        }
        let slot_off = (slot % Self::RING_DEPTH) as i64;
        let dev = dst.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let (storage, layout) = dst.storage_and_layout();
        let candle_core::Storage::Cuda(s) = &*storage else {
            candle_core::bail!("flag result destination is not on CUDA");
        };
        let slice = s.as_cuda_slice::<f32>()?;
        let view = slice.slice(layout.start_offset()..);
        let (ptr, _guard) = view.device_ptr(&stream);
        // SAFETY: destination comes from the tensor; the launch is stream-ordered.
        unsafe {
            ffi::doorbell_wait_copy_f32(
                self.dev_ptr(0),
                publish_seq,
                self.dev_ptr(8),
                slot_off * self.max_n as i64,
                ptr as *mut c_void,
                len as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }
}

impl Drop for FlagBuf {
    fn drop(&mut self) {
        if !self.host.is_null() {
            // SAFETY: allocated by doorbell_alloc_mapped, freed nowhere else.
            unsafe { ffi::doorbell_free_host(self.host as *mut c_void) };
        }
    }
}
