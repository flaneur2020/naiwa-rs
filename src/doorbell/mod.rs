//! Doorbell buffers and the v2 sync protocol: mapped pinned memory the device and the host use to
//! hand each other data and progress, with no stream syncs and no host callbacks.
//!
//! Three shapes live here, all built on the same rule — *payload first, then a volatile bump of a
//! counter, with a fence between*:
//!
//! - [`DoorbellBuf`] — device → host: the router's top-k, so the host read needs no D2H launch.
//! - [`FlagBuf`] — both directions: the lookahead activation (device → host) and the CPU experts'
//!   result over a ring (host → device).
//! - [`ScheduleBuf`] — the v2 per-pass schedule: `epoch` (device-written) + one `ready` word per
//!   sync point (written by its producer, host or copy stream) + a payload slot per layer.
//!
//! See `architecture-v2` §3.1/§3.2 for the protocol and its invariants.

use std::ffi::c_void;

use candle_core::cuda::cudarc::driver::sys;
use candle_core::{Result, Tensor};

/// The doorbell kernels' `extern "C"` surface. Module-private: everything above it is safe.
#[allow(dead_code)]
mod ffi {
    use std::ffi::c_void;

    extern "C" {
    pub(crate) fn doorbell_alloc_mapped(out: *mut *mut c_void, bytes: usize) -> i32;
    pub(crate) fn doorbell_free_host(p: *mut c_void);
    pub(crate) fn doorbell_get_device_ptr(host: *mut c_void, dev: *mut *mut c_void);
    pub(crate) fn doorbell_ring(
        d_seq: *mut c_void,       // mapped seq counter
        d_ids: *mut c_void,       // mapped buffer, [rows, topk] u32
        d_weights: *mut c_void,   // mapped buffer, [rows, topk] f32
        src_ids: *const c_void,   // device tensor of ids (stream-ordered before the ring)
        src_weights: *const c_void, // device tensor of weights
        n: i32,                   // rows * topk
        stream: i64,
    );
    pub(crate) fn doorbell_copy_f32(
        d_seq: *mut c_void,   // mapped seq counter
        d_dst: *mut c_void,   // mapped payload buffer, device view
        src: *const c_void,   // device f32 tensor, stream-ordered before this launch
        n: i32,               // float count
        stream: i64,
    );
    pub(crate) fn doorbell_wait_seq_ge(
        d_seq: *mut c_void,   // mapped seq counter the host bumps
        target: u32,          // the seq value that means "payload complete"
        stream: i64,
    );
    pub(crate) fn doorbell_wait_copy_f32(
        d_seq: *mut c_void,     // mapped seq counter the host bumps
        target: u32,            // seq value that means "payload slot complete"
        d_payload: *mut c_void, // mapped payload ring, device view
        slot_off: i64,          // float offset of the slot to read
        dst: *mut c_void,       // device f32 destination
        n: i32,                 // float count
        stream: i64,
    );
    // F3 / v2 M0: zero-argument variants. `target` and `slot_off` are read from mapped memory by
    // the kernel, so one captured graph can be replayed with fresh values each step.
    pub(crate) fn doorbell_wait_seq_ge_ptr(
        d_seq: *mut c_void,    // mapped seq counter the host bumps
        d_target: *mut c_void, // mapped u32 cell: the seq value that means "complete"
        stream: i64,
    );
    pub(crate) fn doorbell_publish_max(
        d_ready: *mut c_void,
        id: i32,
        pass: u32,
        stream: i64,
    );
    pub(crate) fn doorbell_epoch_bump(d_epoch: *mut c_void, stream: i64);
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn doorbell_dispatch_topk(
        ids: *const c_void,
        weights: *const c_void,
        n: i32,
        bitmap: *const c_void,
        frac_num: i32,
        frac_den: i32,
        hits_slot: *mut c_void,
        hits_w: *mut c_void,
        hits_n: *mut c_void,
        stream_ids: *mut c_void,
        stream_n: *mut c_void,
        cpu_ids: *mut c_void,
        cpu_n: *mut c_void,
        max_hits: i32,
        stream: i64,
    );
    pub(crate) fn doorbell_wait_ready(
        d_epoch: *mut c_void,
        d_ready: *mut c_void,
        id: i32,
        stream: i64,
    );
    pub(crate) fn doorbell_wait_add_ready(
        d_epoch: *mut c_void,   // device u32, bumped once per pass
        d_ready: *mut c_void,   // mapped [K] u32, one word per sync point
        d_payload: *mut c_void, // mapped [slots][max_n] f32, device view
        acc: *mut c_void,       // device f32 accumulator
        n: i32,
        id: i32,
        slot: i32,
        max_n: i32,
        stream: i64,
    );
    pub(crate) fn doorbell_wait_add_indexed_f32(
        d_seq: *mut c_void,     // mapped seq counter the host bumps
        d_counter: *mut c_void, // device u32 job counter (atomicAdd per node)
        d_payload: *mut c_void, // mapped payload ring, device view
        acc: *mut c_void,       // device f32 accumulator (read-modify-write)
        n: i32,                 // float count
        ring_depth: i32,        // ring slots
        max_n: i32,             // floats per slot
        stream: i64,
    );
    pub(crate) fn doorbell_wait_add_f32_ptr(
        d_seq: *mut c_void,      // mapped seq counter the host bumps
        d_target: *mut c_void,   // mapped u32 cell: the seq value to wait for
        d_payload: *mut c_void,  // mapped payload ring, device view
        d_slot_off: *mut c_void, // mapped i64 cell: the slot's float offset
        acc: *mut c_void,        // device f32 accumulator (read-modify-write)
        n: i32,                  // float count
        stream: i64,
    );
    pub(crate) fn doorbell_wait_copy_f32_ptr(
        d_seq: *mut c_void,      // mapped seq counter the host bumps
        d_target: *mut c_void,   // mapped u32 cell: the seq value to wait for
        d_payload: *mut c_void,  // mapped payload ring, device view
        d_slot_off: *mut c_void, // mapped i64 cell: the slot's float offset
        dst: *mut c_void,        // device f32 destination
        n: i32,                  // float count
        stream: i64,
    );
    }
}

/// A doorbell buffer: ids/weights payload plus the seq counter, host and device views
/// of the same mapped bytes.
pub struct DoorbellBuf {
    host: *mut u8,
    dev_view: *mut c_void,
    // layout: [seq: u32 (offset 0)] [ids: max_n u32 (offset 8)] [weights: max_n f32]
    max_n: usize,
}

// SAFETY: the buffer is written by the device behind the stream order and the seq
// protocol; no Rust aliasing (mapped memory handed to CUDA).
unsafe impl Send for DoorbellBuf {}
unsafe impl Sync for DoorbellBuf {}

const IDS_OFF: usize = 8; // seq (4) + pad (4)

impl DoorbellBuf {
    /// Allocate a buffer for up to `max_n` (rows*topk) ids and weights. `Ok(None)` when
    /// the device is not CUDA or the mapped allocation fails; the caller then keeps the
    /// path it uses today.
    pub fn acquire(device: &candle_core::Device, max_n: usize) -> Result<Option<DoorbellBuf>> {
        if !device.is_cuda() {
            return Ok(None);
        }
        let bytes = IDS_OFF + max_n * 4 + max_n * 4;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: fresh allocation of `bytes`; on non-zero rc the caller falls back.
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
        // SAFETY: zero the seq counter so the first ring advances 0 -> 1.
        unsafe { std::ptr::write_volatile(host.cast::<u32>(), 0) };
        Ok(Some(DoorbellBuf { host: host.cast(), dev_view, max_n }))
    }

    fn seq_ptr(&self) -> *mut u32 {
        self.host.cast()
    }

    fn ids_ptr(&self) -> *mut u32 {
        // SAFETY: offset 8 inside the allocation.
        unsafe { self.host.add(IDS_OFF).cast::<u32>() }
    }

    fn weights_ptr(&self) -> *mut f32 {
        // SAFETY: past the ids block, within the allocation.
        unsafe { self.host.add(IDS_OFF + self.max_n * 4).cast::<f32>() }
    }

    fn dev_ptr(&self, off: usize) -> *mut c_void {
        // SAFETY: the device view of the same allocation, offset by `off`.
        unsafe { self.dev_view.cast::<u8>().add(off) as *mut c_void }
    }

    /// Queue the ring on the device's current stream: copy `n` ids/weights from the
    /// device tensors into the mapped buffer, then bump seq. Stream-ordered after
    /// whatever produced the tensors — that ordering is the whole point.
    pub fn ring(&self, ids: &Tensor, weights: &Tensor, n: usize) -> Result<()> {

        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let dev = ids.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let stream_raw = stream.cu_stream() as i64;

        let ids32 = ids.to_dtype(candle_core::DType::U32)?.contiguous()?;
        let w32 = weights.to_dtype(candle_core::DType::F32)?.contiguous()?;

        // graph.rs's copy_dtype pattern: storage_and_layout + as_cuda_slice + device_ptr
        let (ids_storage, ids_layout) = ids32.storage_and_layout();
        let candle_core::Storage::Cuda(ids_s) = &*ids_storage else {
            candle_core::bail!("doorbell ring: ids not on CUDA");
        };
        let ids_slice = ids_s.as_cuda_slice::<u32>()?;
        let ids_view = ids_slice.slice(ids_layout.start_offset()..);
        let (ids_ptr, _ids_guard) = ids_view.device_ptr(&stream);

        let (w_storage, w_layout) = w32.storage_and_layout();
        let candle_core::Storage::Cuda(w_s) = &*w_storage else {
            candle_core::bail!("doorbell ring: weights not on CUDA");
        };
        let w_slice = w_s.as_cuda_slice::<f32>()?;
        let w_view = w_slice.slice(w_layout.start_offset()..);
        let (w_ptr, _w_guard) = w_view.device_ptr(&stream);

        // SAFETY: the device pointers come from the tensors; off-pointer math is inside
        // the mapped allocation; the launch is ordered on the current stream.
        unsafe {
            ffi::doorbell_ring(
                self.dev_ptr(0) as *mut c_void,               // d_seq (mapped)
                self.dev_ptr(IDS_OFF) as *mut c_void,         // d_ids (mapped)
                self.dev_ptr(IDS_OFF + self.max_n * 4) as *mut c_void, // d_weights
                ids_ptr as *const c_void,
                w_ptr as *const c_void,
                n as i32,
                stream_raw,
            );
        }
        Ok(())
    }

    /// Wait until the seq advances past `last_seq`, then copy out the published ids and
    /// weights. No CUDA calls — safe from the token-driving thread.
    ///
    /// Adaptive hybrid wait (W106 postscript): busy-spin for a short budget first —
    /// the result typically lands within microseconds of the wait starting (the
    /// `wait_copy` kernel already sat queued on the stream), and a 50µs sleep poll
    /// costs on average a quarter of its interval, ~25µs per wake, which across
    /// 48 layers measured as a 10ms/step TPOT regression versus the blocking-copy
    /// baseline. Then fall back to sleep-polling (50µs): the wait legitimately
    /// spans milliseconds while the CPU experts compute (the ring kernel can sit
    /// in the stream behind a result-wait kernel), and a pure spin would burn a
    /// whole core on the token thread (W106) and, worse, a give-up bound crossed
    /// returned STALE top-k ids — wrong routing, hit collapse, the 0.2-0.6 T/s
    /// runaway. The wait cannot time out: stale ids are silent corruption. There
    /// is no deadlock — the result-wait is released by the CPU experts' threads.
    pub fn wait_and_read(&self, last_seq: u32, n: usize) -> (u32, Vec<u32>, Vec<f32>) {
        let want = last_seq.wrapping_add(1);
        let seq = self.seq_ptr();
        // ~50µs of spin, ~2000 iterations at ~25ns per volatile read + pause.
        let mut spins: u32 = 0;
        const SPIN_BUDGET: u32 = 2000;
        loop {
            // SAFETY: mapped memory the device writes; read_volatile each poll.
            let v = unsafe { std::ptr::read_volatile(seq) };
            if v >= want {
                break;
            }
            if spins < SPIN_BUDGET {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        let ids_ptr = self.ids_ptr();
        let w_ptr = self.weights_ptr();
        // SAFETY: the seq read orders these payload reads (device fenced before bump).
        let ids = (0..n).map(|i| unsafe { std::ptr::read_volatile(ids_ptr.add(i)) }).collect();
        let ws = (0..n).map(|i| unsafe { std::ptr::read_volatile(w_ptr.add(i)) }).collect();
        (want, ids, ws)
    }

    /// Read seq without spinning (the value `wait_and_read` last returned).
    pub fn seq(&self) -> u32 {
        // SAFETY: first 4 bytes of the allocation.
        unsafe { std::ptr::read_volatile(self.seq_ptr()) }
    }

    pub fn max_n(&self) -> usize {
        self.max_n
    }
}

impl Drop for DoorbellBuf {
    fn drop(&mut self) {
        if !self.host.is_null() {
            // SAFETY: allocated by doorbell_alloc_mapped, freed nowhere else.
            unsafe { ffi::doorbell_free_host(self.host as *mut c_void) };
        }
    }
}

/// A generic mapped flag channel (R2): a seq counter at offset 0 plus `max_n`
/// f32 payload slots, host and device views of the same bytes.
///
/// GPU -> host: [`FlagBuf::publish_f32`] queues a stream-ordered kernel that
/// copies a device tensor into the payload and bumps seq; the host spins in
/// [`FlagBuf::wait_publish`] — no stream sync, no D2H launch.
///
/// host -> GPU: the host writes the payload (plain stores, in publish order),
/// bumps seq with a release store, and queues [`FlagBuf::wait_seq_ge`] on the
/// stream: a 1-thread kernel spinning on the mapped seq. Whatever the stream
/// runs afterwards sees the payload; the host never blocks the stream with a
/// host function (graph-capturable, unlike `cudaLaunchHostFunc`).
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
    fn target_off(&self) -> usize {
        8 + self.max_n * Self::RING_DEPTH * 4
    }

    /// F3/v2: offset of the mapped `slot_off` cell (an i64), after `target` + padding.
    fn slot_off_off(&self) -> usize {
        self.target_off() + 8
    }

    /// F3/v2: record the seq value this channel's stream-side wait targets. The wait kernels read
    /// it from mapped memory rather than taking it as a kernel argument, so a captured graph can
    /// be replayed with a fresh target each step (a kernel argument is frozen at capture).
    pub fn set_target(&self, target: u32) {
        // SAFETY: the cell lies inside the mapped allocation; a plain volatile store suffices for
        // the device to read it (the stream-ordered kernel is launched after this call).
        unsafe {
            std::ptr::write_volatile(self.host.add(self.target_off()).cast::<u32>(), target)
        }
    }

    /// F3/v2: record which payload slot the stream-side copy reads, as a float offset into the
    /// ring. Written before queueing [`Self::wait_copy_slot_ptr`].
    pub fn set_slot_off(&self, slot: usize) {
        let off = (slot % Self::RING_DEPTH) as i64 * self.max_n as i64;
        // SAFETY: as `set_target`.
        unsafe {
            std::ptr::write_volatile(self.host.add(self.slot_off_off()).cast::<i64>(), off)
        }
    }

    /// F3/v2: like [`Self::wait_seq_ge`], but the target comes from the mapped cell
    /// ([`Self::set_target`]) instead of a kernel argument — capture-safe.
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

    /// M3: wait for `src`'s published seq and accumulate `src`'s payload slot into the device
    /// buffer at `acc`.
    ///
    /// `self` supplies the mapped `target` / `slot_off` **cells**, `src` supplies the seq and
    /// payload. The cells must be per-bank: the CPU-return channel is global, so two layers
    /// sharing one set of cells would overwrite them before the earlier layer's wait node runs
    /// (the graph node reads the cells at *execution* time, while the token thread runs ahead).
    ///
    /// `acc` is a raw device pointer because a captured call must resolve it *outside* the
    /// capture window (a guarded `device_ptr` records an event, which capture forbids).
    pub fn wait_add_from(
        &self,
        device: &candle_core::Device,
        src: &FlagBuf,
        len: usize,
        acc: u64,
    ) -> Result<()> {
        if len > src.max_n {
            candle_core::bail!(
                "flag add of {len} f32 exceeds the channel capacity ({})",
                src.max_n
            );
        }
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        let target = self.dev_ptr(self.target_off());
        let slot_off = self.dev_ptr(self.slot_off_off());
        // SAFETY: seq/payload come from `src`; cells from `self`; all inside mapped allocations.
        unsafe {
            ffi::doorbell_wait_add_f32_ptr(
                src.dev_ptr(0),
                target,
                src.dev_ptr(8),
                slot_off,
                acc as *mut c_void,
                len as i32,
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// M3: like [`Self::wait_add_from`], but the **job number is claimed on the device** from
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

    /// The seq the host should wait for after `publish_f32` (call before it).
    pub fn next_seq(&self) -> u32 {
        // SAFETY: mapped memory the device bumps; a single u32 read.
        unsafe { std::ptr::read_volatile(self.seq_ptr()) }.wrapping_add(1)
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

    /// Host -> GPU: write `values` into the payload and publish, returning the
    /// seq value the GPU must see. The caller must have waited for any earlier
    /// GPU publishes first (one channel, one direction at a time).
    pub fn write_payload(&self, values: &[f32], publish_seq: u32) -> u32 {
        let p = self.payload_ptr();
        // SAFETY: `values` fits (caller checks `max_n`); plain stores, then the
        // release seq store publishes them.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), p, values.len());
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(self.seq_ptr(), publish_seq);
        }
        publish_seq
    }

    /// Queue the GPU-side wait on `device`'s current stream: a 1-thread kernel
    /// spinning until the mapped seq reaches `target`. Everything queued after
    /// this on the stream runs behind the host's payload.
    pub fn wait_seq_ge(&self, device: &candle_core::Device, target: u32) -> Result<()> {
        let dev = device.as_cuda_device()?;
        let stream = dev.cuda_stream();
        // SAFETY: the launch is ordered on the current stream.
        unsafe {
            ffi::doorbell_wait_seq_ge(self.dev_ptr(0), target, stream.cu_stream() as i64);
        }
        Ok(())
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

    fn dev_payload_ptr(&self) -> *mut c_void {
        // SAFETY: as above.
        unsafe {
            self.dev_view
                .cast::<u8>()
                .add(8 + self.k * 4) as *mut c_void
        }
    }

    pub fn max_n(&self) -> usize {
        self.max_n
    }

    /// The static id of `(layer, kind)` — the word this sync point owns.
    pub fn sync_id(&self, layer: u32, kind: u32) -> u32 {
        layer * self.waits_per_layer + kind
    }

    /// The last pass the **device** announced. The host normally does *not* read this: it keeps its
    /// own pass counter, which equals the device's for the same pass.
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

#[cfg(all(test, feature = "cuda"))]
mod ptr_wait_tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// F3 / v2 M0: the zero-argument wait+copy kernel must deliver exactly the payload whose seq
    /// it waited for, reading `target` and `slot_off` from mapped memory rather than from kernel
    /// arguments — so the *same* object serves successive values without recompiling a launch.
    #[test]
    fn wait_copy_slot_ptr_reads_target_and_slot_from_mapped_memory() {
        let dev = Device::new_cuda(0).unwrap();
        let max_n = 8usize;
        let flag = FlagBuf::acquire(&dev, max_n).unwrap().unwrap();
        let dst = Tensor::zeros(max_n, DType::F32, &dev).unwrap();

        let first: Vec<f32> = (0..max_n).map(|i| i as f32 + 1.5).collect();
        flag.write_slot(Some(&first), 2, 7);
        flag.set_target(7);
        flag.set_slot_off(2);
        flag.wait_copy_slot_ptr(max_n, &dst).unwrap();
        assert_eq!(dst.to_vec1::<f32>().unwrap(), first);

        // Same channel, same kernel parameters: only the mapped cells moved.
        let second: Vec<f32> = (0..max_n).map(|i| 100.0 - i as f32).collect();
        flag.write_slot(Some(&second), 5, 8);
        flag.set_target(8);
        flag.set_slot_off(5);
        flag.wait_copy_slot_ptr(max_n, &dst).unwrap();
        assert_eq!(dst.to_vec1::<f32>().unwrap(), second);

        // A failed computation still unblocks the wait (seq bumps, no payload).
        flag.set_target(9);
        flag.write_slot(None, 5, 9);
        flag.wait_seq_ge_ptr(&dev).unwrap();
    }

    /// v2 §3.2: the property every earlier variant lacked — a node whose `(layer, kind)` are
    /// **constants** advances to the next pass because the *device* moved the epoch, not the host.
    #[test]
    fn epoch_advances_a_replayed_node_to_the_next_pass() {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let dev = Device::new_cuda(0).unwrap();
        let (max_n, slots, layer) = (8usize, 4usize, 2u32);
        let sched = ScheduleBuf::acquire(&dev, 3, slots, max_n).unwrap().unwrap();
        let acc = Tensor::zeros(max_n, DType::F32, &dev).unwrap();
        let acc_ptr = {
            let (st, _) = acc.storage_and_layout();
            let candle_core::Storage::Cuda(s) = &*st else {
                panic!("acc must be on CUDA");
            };
            let slice = s.as_cuda_slice::<f32>().unwrap();
            let (p, _guard) = slice.device_ptr(slice.stream());
            p
        };

        // Pass 1: the prologue bumps the epoch; the host publishes for its own pass counter.
        sched.bump_epoch(&dev).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(sched.epoch(), 1, "one pass -> epoch 1");
        let first = vec![1.0f32; max_n];
        sched.publish_cpu(layer, 1, &first);
        sched.wait_add_cpu(&dev, max_n, layer, acc_ptr).unwrap();
        assert_eq!(acc.to_vec1::<f32>().unwrap(), first);

        // Pass 2, SAME node constants: it must wait for pass 2's word, not be satisfied by pass
        // 1's. The payload slot is single-buffered — safe because the host cannot reach pass 2
        // without a GPU-produced signal from pass 2.
        sched.bump_epoch(&dev).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(sched.epoch(), 2, "two passes -> epoch 2");
        let second = vec![10.0f32; max_n];
        sched.publish_cpu(layer, 2, &second);
        sched.wait_add_cpu(&dev, max_n, layer, acc_ptr).unwrap();
        let want: Vec<f32> = first.iter().zip(&second).map(|(a, b)| a + b).collect();
        assert_eq!(acc.to_vec1::<f32>().unwrap(), want);
    }

    /// M4: a DMA-done point published **by the stream itself** satisfies a wait-only node, and the
    /// host sees the same value — the shape both `PRELOAD` and `STAGING` will use.
    #[test]
    fn stream_published_point_satisfies_a_wait_only_node() {
        let dev = Device::new_cuda(0).unwrap();
        let sched = ScheduleBuf::acquire(&dev, 3, 2, 8).unwrap().unwrap();
        let id = sched.sync_id(1, super::SYNC_STAGING);

        sched.bump_epoch(&dev).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(sched.epoch(), 1);
        // Published by the stream (as the copy stream does after a DMA), then waited on.
        sched.publish_point_on_stream(&dev, id, 1).unwrap();
        sched.wait_ready(&dev, id).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(sched.ready(id), 1, "the host sees the stream-published pass");

        // Next pass: the point must be re-armed by the stream again.
        sched.bump_epoch(&dev).unwrap();
        dev.synchronize().unwrap();
        sched.publish_point_on_stream(&dev, id, 2).unwrap();
        sched.wait_ready(&dev, id).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(sched.ready(id), 2);
    }

    /// M3: the indexed node claims its job **on the device**, so two nodes queued together
    /// consume slots 1 and 2 with no host-written per-step cell — the property that makes it
    /// safe inside a captured graph while the token thread runs ahead.
    #[test]
    fn wait_add_indexed_claims_jobs_on_the_device() {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let dev = Device::new_cuda(0).unwrap();
        let cuda = dev.as_cuda_device().unwrap();
        let max_n = 8usize;
        let flag = FlagBuf::acquire(&dev, max_n).unwrap().unwrap();
        let acc = Tensor::zeros(max_n, DType::F32, &dev).unwrap();
        let counter = cuda.alloc_zeros::<u32>(1).unwrap();

        let first: Vec<f32> = (0..max_n).map(|i| i as f32 + 1.0).collect();
        let second: Vec<f32> = (0..max_n).map(|i| 10.0 + i as f32).collect();
        // Jobs 1 and 2 live on ring slots 1 and 2 (the publisher's numbering).
        flag.write_slot(Some(&first), 1, 1);
        flag.write_slot(Some(&second), 2, 2);

        let counter_ptr = counter.device_ptr(counter.stream()).0;
        let acc_ptr = {
            let (st, _) = acc.storage_and_layout();
            let candle_core::Storage::Cuda(s) = &*st else {
                panic!("acc must be on CUDA");
            };
            let slice = s.as_cuda_slice::<f32>().unwrap();
            let (p, _guard) = slice.device_ptr(slice.stream());
            p
        };
        flag.wait_add_indexed_from(&dev, counter_ptr, max_n, acc_ptr)
            .unwrap();
        flag.wait_add_indexed_from(&dev, counter_ptr, max_n, acc_ptr)
            .unwrap();

        let want: Vec<f32> = first.iter().zip(&second).map(|(a, b)| a + b).collect();
        assert_eq!(acc.to_vec1::<f32>().unwrap(), want);
    }

    /// M3: the wait-then-accumulate node folds the payload into the buffer that already holds
    /// the GPU hits' sum, so the graph's output is the completed MoE result.
    #[test]
    fn wait_add_slot_ptr_accumulates_into_the_output_buffer() {
        let dev = Device::new_cuda(0).unwrap();
        let max_n = 8usize;
        let flag = FlagBuf::acquire(&dev, max_n).unwrap().unwrap();
        let acc = Tensor::full(1.0f32, max_n, &dev).unwrap();

        let cpu: Vec<f32> = (0..max_n).map(|i| 10.0 + i as f32).collect();
        flag.write_slot(Some(&cpu), 1, 3);
        flag.set_target(3);
        flag.set_slot_off(1);
        flag.wait_add_slot_ptr(max_n, &acc).unwrap();

        let want: Vec<f32> = (0..max_n).map(|i| 11.0 + i as f32).collect();
        assert_eq!(acc.to_vec1::<f32>().unwrap(), want);
    }
}
