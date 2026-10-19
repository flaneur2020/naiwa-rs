//! The mapped-allocation owner shared by every doorbell.
//!
//! `cudaHostAllocMapped` memory has a host address and a device address for the same bytes. All
//! doorbells are views over one of these allocations; the protocol in each doorbell decides who
//! reads/writes which word and how the two sides order those accesses.

use std::ffi::c_void;
use std::time::{Duration, Instant};

use candle_core::{Device, Result};

use super::ffi;

/// Bytes reserved at the front of every doorbell layout for its counters (`seq`, `consumed`), so
/// every payload starts 8-byte aligned and the offsets stay one number instead of one per file.
pub(super) const HEADER_BYTES: usize = 8;

pub(super) struct MappedBuf {
    host: *mut u8,
    dev: *mut c_void,
    len: usize,
}

// SAFETY: this is mapped pinned memory handed to CUDA. The raw pointers are only dereferenced
// through the doorbell protocols (volatile reads/writes and stream-ordered kernels), which declare
// a single writer per word and order payload before its publication counter.
unsafe impl Send for MappedBuf {}
unsafe impl Sync for MappedBuf {}

impl MappedBuf {
    /// Allocate `len` bytes of pinned mapped memory and zero all of it. `Ok(None)` off CUDA or on
    /// allocation failure; the caller then keeps its fallback path.
    pub(super) fn alloc_zeroed(device: &Device, len: usize) -> Result<Option<Self>> {
        if !device.is_cuda() || len == 0 {
            return Ok(None);
        }
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: fresh allocation of `len` bytes; on failure the caller falls back.
        let rc = unsafe { ffi::doorbell_alloc_mapped(&mut host, len) };
        if rc != 0 || host.is_null() {
            return Ok(None);
        }
        let mut dev: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host` holds a live cudaHostAllocMapped allocation.
        unsafe { ffi::doorbell_get_device_ptr(host, &mut dev) };
        if dev.is_null() {
            // SAFETY: allocated above, not freed elsewhere yet.
            unsafe { ffi::doorbell_free_host(host) };
            return Ok(None);
        }
        // SAFETY: inside the allocation. Zeroing the whole mapping means a consumer that runs
        // before its producer sees zeroes, never uninitialised memory (the old schedule payload
        // NaN bug class).
        unsafe { std::ptr::write_bytes(host, 0, len) };
        Ok(Some(Self {
            host: host.cast(),
            dev,
            len,
        }))
    }

    pub(super) fn host(&self) -> *mut u8 {
        self.host
    }

    pub(super) fn dev(&self, offset: usize) -> *mut c_void {
        debug_assert!(offset <= self.len);
        // SAFETY: the device view covers the same `len` bytes.
        unsafe { self.dev.cast::<u8>().add(offset) as *mut c_void }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Host view of a `u32` word at `offset`.
    pub(super) fn host_u32(&self, offset: usize) -> *mut u32 {
        debug_assert!(offset + 4 <= self.len);
        // SAFETY: `offset` is inside the allocation.
        unsafe { self.host.add(offset).cast() }
    }
}

impl Drop for MappedBuf {
    fn drop(&mut self) {
        if !self.host.is_null() {
            // SAFETY: allocated by doorbell_alloc_mapped, freed nowhere else.
            let rc = unsafe { ffi::doorbell_free_host(self.host.cast()) };
            if rc != 0 {
                // Benign at process exit: the doorbell statics outlive the CUDA context, so the
                // driver is already deinitialized (error 4) when they drop.
                tracing::debug!("cudaFreeHost failed on a doorbell mapping: {rc}");
            }
        }
    }
}

pub(super) fn load_u32(word: *const u32) -> u32 {
    // SAFETY: mapped memory; the caller's protocol orders it against the writer.
    unsafe { std::ptr::read_volatile(word) }
}

pub(super) fn load_f32(word: *const f32) -> f32 {
    // SAFETY: as `load_u32`.
    unsafe { std::ptr::read_volatile(word) }
}

/// Host publication: payload stores first, then the release fence, then the volatile counter.
pub(super) fn store_release_u32(word: *mut u32, value: u32) {
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    // SAFETY: mapped memory; the release fence orders the payload stores before this write.
    unsafe { std::ptr::write_volatile(word, value) };
}

/// Spin briefly, then sleep-poll until `*word >= target`. `timeout == None` never gives up.
pub(super) fn wait_host_ge(
    word: *const u32,
    target: u32,
    timeout: Option<Duration>,
    poll: Duration,
) -> bool {
    const SPIN_BUDGET: u32 = 2000;
    let deadline = timeout.map(|t| Instant::now() + t);
    let mut spins: u32 = 0;
    loop {
        if load_u32(word) >= target {
            return true;
        }
        if let Some(deadline) = deadline {
            if Instant::now() >= deadline {
                return false;
            }
        }
        if spins < SPIN_BUDGET {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::sleep(poll);
        }
    }
}
