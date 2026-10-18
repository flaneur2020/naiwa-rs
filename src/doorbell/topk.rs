//! Device -> host top-k doorbell ring (module docs: `super`).

use std::ffi::c_void;

use candle_core::{Result, Tensor};

use super::ffi;

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
