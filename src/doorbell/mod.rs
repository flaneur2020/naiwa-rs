//! Doorbell primitives: mapped pinned memory the device and the host use to hand each other data
//! and progress, with no stream syncs and no host callbacks.
//!
//! Every primitive is one direction, and there is exactly one rule:
//!
//! > **Payload first, then a system fence, then a `volatile` bump of a mapped counter.**
//!
//! The consumer sees the counter and may read the payload behind it. Device-side words are
//! written `volatile` and fenced with `__threadfence_system()`; host-side words use a release
//! fence and a volatile store (which also relies on x86 TSO for the device to observe host stores
//! in order).
//!
//! - [`TopkDoorbell`] — device -> host: the router's top-k, depth 1. The device counter *is* the
//!   sequence; the host reads it before publishing and waits for `seq + 1`.
//! - [`ActivationDoorbell`] — device -> host: the lookahead activation over a slot ring.
//!   `LookaheadChannel` owns its publish/read seq and backpressure.
//! - [`ResultDoorbell`] — host -> device: the CPU experts' result over a slot ring. The device
//!   writes a `consumed` word after each copy, so the host can check its ring depth instead of
//!   assuming it.
//! - [`PassCell`] — a cross-thread *completion* handoff with no payload: the producer (lookahead
//!   thread) records a [`CopyDone`](crate::copy_stream::CopyDone) and claims a pass, the consumer
//!   (token thread) waits for the claim it sees. Both barriers in the decode path — the staging
//!   DMA and the preload — are this one mechanism; there are no gate words and no epoch.
//!
//! The slot index is always derived from the sequence number (`seq % DEPTH`).
//! [`MappedBuf`](mapped::MappedBuf) is the single allocation owner and is private, so the unsafe
//! mapped-memory surface stays inside this module.

/// The doorbell kernels' `extern "C"` surface. Module-private: everything above it is safe.
#[allow(dead_code)]
mod ffi {
    use std::ffi::c_void;

    extern "C" {
        pub(crate) fn doorbell_alloc_mapped(out: *mut *mut c_void, bytes: usize) -> i32;
        pub(crate) fn doorbell_free_host(p: *mut c_void) -> i32;
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
            d_seq: *mut c_void,      // mapped seq counter the host bumps
            target: u32,             // seq value that means "payload slot complete"
            d_payload: *mut c_void,  // mapped payload slot, device view
            slot_off: i64,           // float offset from d_payload (0 when it is the slot itself)
            d_consumed: *mut c_void, // mapped word the kernel sets to `target` after copying
            dst: *mut c_void,        // device f32 destination
            n: i32,                  // float count
            stream: i64,
        );
    }
}

mod d2h;
mod handoff;
mod h2d;
mod mapped;
mod payload_ring;

#[cfg(all(test, feature = "cuda"))]
mod tests;

pub use d2h::{ActivationDoorbell, TopkDoorbell};
pub use handoff::PassCell;
pub use h2d::ResultDoorbell;
