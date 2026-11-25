//! Advisory GPU -> host activation channel over mapped memory (`ActivationDoorbell`).
//!
//! Predictions may be skipped under backpressure or dropped when superseded by a newer publish
//! (`next_seq > last_seq + 1`).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use candle_core::{Device, Tensor};

use crate::transport::doorbell::ActivationDoorbell;

/// Advisory GPU -> host activation channel for the lookahead thread.
pub(crate) struct LookaheadChannel {
    chan: Arc<ActivationDoorbell>,
    /// The seq the last queued GPU publish will bump the counter to (0 before any).
    /// Touched only by the token thread.
    next_seq: AtomicU32,
    /// How many queued publishes the worker has consumed (`wait_publish` returned).
    consumed: AtomicU32,
}

impl LookaheadChannel {
    /// Allocate a mapped activation channel covering at least `max_rows.max(16)` rows.
    pub(crate) fn new(
        device: &Device,
        max_rows: usize,
        hidden: usize,
    ) -> candle_core::Result<Option<Arc<Self>>> {
        let max_rows = max_rows.max(16);
        Ok(ActivationDoorbell::acquire(device, max_rows * hidden)
            .ok()
            .flatten()
            .map(|chan| {
                Arc::new(Self {
                    chan: Arc::new(chan),
                    next_seq: AtomicU32::new(0),
                    consumed: AtomicU32::new(0),
                })
            }))
    }

    /// The widest publish this channel's mapped slot can hold, in `f32` elements.
    #[inline]
    pub(crate) fn max_n(&self) -> usize {
        self.chan.max_n()
    }

    /// Queue the publish on the stream (token thread). Returns `None` under backpressure.
    #[inline]
    pub(crate) fn publish(
        self: &Arc<Self>,
        x: &Tensor,
        n: usize,
    ) -> candle_core::Result<Option<(Arc<Self>, u32)>> {
        if n > self.chan.max_n() {
            candle_core::bail!(
                "lookahead publish of {n} f32 exceeds the channel slot ({})",
                self.chan.max_n()
            );
        }
        let next = self.next_seq.load(Ordering::Relaxed);
        let consumed = self.consumed.load(Ordering::Relaxed);
        if next - consumed >= (self.chan.depth() as u32).saturating_sub(1) {
            return Ok(None);
        }
        self.chan.publish(x, n, next)?;
        self.next_seq.store(next + 1, Ordering::Relaxed);
        Ok(Some((Arc::clone(self), next)))
    }

    /// Wait for the publish `last_seq`, dropping it when a newer publish has already superseded it.
    #[inline]
    pub(crate) fn read(&self, last_seq: u32, n: usize) -> Option<Vec<f32>> {
        let next = last_seq.wrapping_add(1);
        if self.next_seq.load(Ordering::Relaxed) > next {
            self.consumed.fetch_max(next, Ordering::Relaxed);
            return None;
        }
        let out = self.chan.wait_publish(last_seq, n);
        self.consumed.fetch_max(next, Ordering::Relaxed);
        out
    }
}
