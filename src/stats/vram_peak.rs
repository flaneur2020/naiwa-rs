//! `NAIWA_PREFILL_PEAK=1`: the low-water mark of free VRAM over a run, sampled once per decode step.
//!
//! Sampling every layer would be 48 driver calls per step, so the caller samples once per step (layer 0).

use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::Device;

/// How many steps between two reports.
const REPORT_EVERY: usize = 16;

static MIN_FREE: AtomicUsize = AtomicUsize::new(0);
static STEPS: AtomicUsize = AtomicUsize::new(0);

/// Whether the instrument is on.
#[inline]
pub(crate) fn enabled() -> bool {
    crate::config::diag().prefill_peak
}

/// Sample free VRAM once per decode step and report the low-water mark periodically.
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
