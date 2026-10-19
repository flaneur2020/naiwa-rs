//! Doorbell primitives: mapped pinned memory the device and the host use to hand each other data
//! and progress, with no stream syncs and no host callbacks.
//!
//! Every primitive is one direction. The shared rule is *payload first, then a volatile bump of a
//! counter, with a fence between*:
//!
//! - [`TopkDoorbell`] — device -> host: the router's top-k. It owns its host sequence;
//!   call `publish_and_read` and do not track a second seq.
//! - [`ActivationDoorbell`] — device -> host: the lookahead activation over a slot ring.
//!   `LookaheadChannel` owns its publish/read seq and backpressure.
//! - [`ResultDoorbell`] — host -> device: the CPU experts' result over a slot ring.
//!   `executor::cuda::Channel` owns its seq allocation.
//! - [`GateSet`] / `Gate` — device-side pass gates: an epoch plus one ready word per gate.
//!   `GateSet::publish_stream` / `wait` hide the raw `sync_id`. There is no payload here;
//!   CPU results use [`ResultDoorbell`].
//!
//! The slot index is always derived from the sequence number (`seq % DEPTH`); callers own the
//! sequence/backpressure policy. [`MappedBuf`](mapped::MappedBuf) is the single allocation owner
//! and is private, so the unsafe mapped-memory surface stays inside this module.

/// The doorbell kernels' `extern "C"` surface. Module-private: everything above it is safe.
#[allow(dead_code)]
mod ffi {
    use std::ffi::c_void;

    extern "C" {
        pub(crate) fn doorbell_alloc_mapped(out: *mut *mut c_void, bytes: usize) -> i32;
        pub(crate) fn doorbell_free_host(p: *mut c_void);
        pub(crate) fn doorbell_get_device_ptr(host: *mut c_void, dev: *mut *mut c_void);
        pub(crate) fn doorbell_ring(
            d_seq: *mut c_void,         // mapped seq counter
            d_ids: *mut c_void,         // mapped buffer, [rows, topk] u32
            d_weights: *mut c_void,     // mapped buffer, [rows, topk] f32
            src_ids: *const c_void,     // device tensor of ids (stream-ordered before the ring)
            src_weights: *const c_void, // device tensor of weights
            n: i32,                     // rows * topk
            stream: i64,
        );
        pub(crate) fn doorbell_copy_f32(
            d_seq: *mut c_void, // mapped seq counter
            d_dst: *mut c_void, // mapped payload buffer, device view
            src: *const c_void, // device f32 tensor, stream-ordered before this launch
            n: i32,             // float count
            stream: i64,
        );
        pub(crate) fn doorbell_wait_copy_f32(
            d_seq: *mut c_void,     // mapped seq counter the host bumps
            target: u32,            // seq value that means "payload slot complete"
            d_payload: *mut c_void, // mapped payload slot, device view
            slot_off: i64,          // float offset from d_payload (0 when it is the slot itself)
            dst: *mut c_void,       // device f32 destination
            n: i32,                 // float count
            stream: i64,
        );
        pub(crate) fn doorbell_publish_max(d_ready: *mut c_void, id: i32, pass: u32, stream: i64);
        pub(crate) fn doorbell_epoch_bump(d_epoch: *mut c_void, stream: i64);
        pub(crate) fn doorbell_wait_ready(
            d_epoch: *mut c_void,
            d_ready: *mut c_void,
            id: i32,
            stream: i64,
        );
    }
}

mod d2h;
mod gate;
mod h2d;
mod mapped;
mod payload_ring;

#[cfg(all(test, feature = "cuda"))]
mod tests;

pub use d2h::{ActivationDoorbell, TopkDoorbell};
pub use gate::{GateSet, SYNC_PRELOAD, SYNC_STAGING};
pub use h2d::ResultDoorbell;
