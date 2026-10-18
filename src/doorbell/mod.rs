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

mod flag;
mod schedule;
mod topk;

#[cfg(all(test, feature = "cuda"))]
mod tests;

pub use flag::FlagBuf;
pub use schedule::{ScheduleBuf, SYNC_CPU_EXPERTS, SYNC_PRELOAD, SYNC_STAGING};
pub use topk::DoorbellBuf;
