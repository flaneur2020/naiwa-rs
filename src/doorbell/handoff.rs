//! The cross-thread completion handoff: the latest claim a producer has published.
//!
//! The preload is produced by the lookahead thread and consumed by the token thread, so a
//! completion signal has to be published *with* the claim: a bare pass number cannot express "wait
//! for the copies of the claim I just read", and a stale claim would be skipped instead of drained.
//! This cell can, and the consumer waits for whatever is in it — including a claim from an older
//! pass, which is exactly what drains an in-flight copy before the current pass reads the slots.
//!
//! The cell owns one persistent event and its own claim counter, like every other doorbell:
//! nothing outside carries a pass number and nobody bumps a device-side epoch. Re-recording one
//! event is safe because a *new* claim for a layer is only produced by a job the token thread
//! scheduled one layer earlier, i.e. after the consumer's wait for the previous claim was already
//! enqueued (`cuStreamWaitEvent` binds to the record that is most recent at call time). Recording a
//! completion event per claim instead costs ~1 ms per job: destroying an event whose record is still
//! queued on the copy stream makes the destroy block.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use candle_core::{Device, Result};

use crate::copy_stream::{CopyDone, CopyStream};

pub struct PassCell {
    done: Arc<CopyDone>,
    /// Claims published so far.
    claims: AtomicU32,
}

impl PassCell {
    /// `done` must be created at warmup, on the producer's copy stream.
    pub fn new(done: Arc<CopyDone>) -> Self {
        Self {
            done,
            claims: AtomicU32::new(0),
        }
    }

    /// Producer: record the completion of every copy queued so far, then publish the claim.
    pub fn claim(&self, cs: &CopyStream) -> Result<u32> {
        self.done.record(cs)?;
        Ok(self.claims.fetch_add(1, Ordering::AcqRel).wrapping_add(1))
    }

    /// Consumer: enqueue a wait on the calling stream for the latest claim. No CUDA work when
    /// nothing was ever claimed.
    pub fn wait_latest(&self, device: &Device) -> Result<Option<u32>> {
        let claims = self.claims.load(Ordering::Acquire);
        if claims == 0 {
            return Ok(None);
        }
        self.done.wait_on(device)?;
        Ok(Some(claims))
    }

    /// Number of claims so far, for tests and logging.
    #[cfg(all(test, feature = "cuda"))]
    pub fn claimed(&self) -> u32 {
        self.claims.load(Ordering::Acquire)
    }
}
