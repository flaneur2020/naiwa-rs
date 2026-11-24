//! The shared `seq + consumed + ring of f32 slots` layout used by the two payload doorbells.
//!
//! This is layout/ownership only; the direction-specific protocol lives in
//! `d2h::ActivationDoorbell` and `h2d::ResultDoorbell`.

use std::ffi::c_void;

use candle_core::{Device, Result};

use super::mapped::{MappedBuf, HEADER_BYTES};

/// Slots in the ring. One slot is being consumed while the others absorb publishes the consumer
/// has not picked up yet.
pub(super) const DEPTH: usize = 8;

const SEQ_OFF: usize = 0;

/// Device-written "the consumer has copied slot `consumed % DEPTH` out". Only the host-to-device
/// ring uses it (there the *device* is the consumer); the device-to-host ring tracks consumption
/// on the host side instead.
const CONSUMED_OFF: usize = 4;

const SLOTS_OFF: usize = HEADER_BYTES;

pub(super) struct PayloadRing {
    buf: MappedBuf,
    max_n: usize,
}

impl PayloadRing {
    pub(super) fn acquire(device: &Device, max_n: usize) -> Result<Option<Self>> {
        if max_n == 0 {
            return Ok(None);
        }
        let Some(bytes) = max_n
            .checked_mul(DEPTH)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(SLOTS_OFF))
        else {
            return Ok(None);
        };
        Ok(MappedBuf::alloc_zeroed(device, bytes)?.map(|buf| Self { buf, max_n }))
    }

    pub(super) fn max_n(&self) -> usize {
        self.max_n
    }

    pub(super) fn seq_host(&self) -> *mut u32 {
        self.buf.host_u32(SEQ_OFF)
    }

    pub(super) fn seq_dev(&self) -> *mut c_void {
        self.buf.dev(SEQ_OFF)
    }

    pub(super) fn consumed_host(&self) -> *mut u32 {
        self.buf.host_u32(CONSUMED_OFF)
    }

    pub(super) fn consumed_dev(&self) -> *mut c_void {
        self.buf.dev(CONSUMED_OFF)
    }

    fn slot_offset(&self, seq: u32) -> usize {
        (seq as usize % DEPTH) * self.max_n * 4
    }

    pub(super) fn slot_host(&self, seq: u32) -> *mut f32 {
        let offset = SLOTS_OFF + self.slot_offset(seq);
        debug_assert!(offset + self.max_n * 4 <= self.buf.len());
        // SAFETY: offset is inside the allocation and covers `max_n` f32s.
        unsafe { self.buf.host().add(offset).cast() }
    }

    pub(super) fn slot_dev(&self, seq: u32) -> *mut c_void {
        let offset = SLOTS_OFF + self.slot_offset(seq);
        debug_assert!(offset + self.max_n * 4 <= self.buf.len());
        self.buf.dev(offset)
    }
}
