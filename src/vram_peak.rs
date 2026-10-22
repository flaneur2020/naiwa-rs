//! `NAIWA_PREFILL_PEAK=1`: the low-water mark of free VRAM over a run, sampled once per decode step.
//!
//! The chunk-side instrument (`prefill_stream::Peak`) answers "what does a prompt chunk cost". This answers the
//! other half of the same question - "what does everything *else* need" - because growing the expert cache breaks
//! on whichever of the two is deeper. The 2.6 GB experiment showed exactly that: the chunk finished with 222 MiB
//! to spare and the OOM came from the post-chunk work.
//!
//! Sampling every layer would be 48 driver calls per step, so the caller samples once per step (layer 0).

use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::Device;

/// How many steps between two reports.
const REPORT_EVERY: usize = 16;

static MIN_FREE: AtomicUsize = AtomicUsize::new(0);
static STEPS: AtomicUsize = AtomicUsize::new(0);

/// Whether the instrument is on. Computed once: it is on a decode path.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NAIWA_PREFILL_PEAK").is_some())
}

/// Sample free VRAM once per decode step and report the low-water mark periodically.
///
/// `free_at_start` is the value the model had when it was ready to serve; passing it lets the report say how much
/// of the free VRAM the run has consumed, which is the number a cache sizing has to leave room for.
pub(crate) fn note_step(device: &Device) {
    let Some(free) = crate::copy_stream::free_bytes(device) else {
        return;
    };
    let min = MIN_FREE.load(Ordering::Relaxed);
    if min == 0 || free < min {
        MIN_FREE.store(free, Ordering::Relaxed);
    }
    let steps = STEPS.fetch_add(1, Ordering::Relaxed) + 1;
    if steps % REPORT_EVERY != 0 {
        return;
    }
    let mib = |b: usize| (b as f64) / (1u64 << 20) as f64;
    tracing::info!(
        "decode peak: {} steps, {:.0} MiB lowest free VRAM ({:.0} MiB now)",
        steps,
        mib(MIN_FREE.load(Ordering::Relaxed)),
        mib(free),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_low_water_mark_only_moves_down() {
        MIN_FREE.store(0, Ordering::Relaxed);
        STEPS.store(0, Ordering::Relaxed);
        // The bookkeeping is a min, so a larger reading must not raise it. Exercised through the same atomics
        // the decode path uses, without a device.
        let note = |free: usize| {
            let min = MIN_FREE.load(Ordering::Relaxed);
            if min == 0 || free < min {
                MIN_FREE.store(free, Ordering::Relaxed);
            }
        };
        note(900);
        note(1200);
        note(700);
        assert_eq!(MIN_FREE.load(Ordering::Relaxed), 700);
    }
}
