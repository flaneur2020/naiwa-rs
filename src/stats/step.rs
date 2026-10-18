//! `NAIWA_STEP_DBG=1`: at each layer's decode entry, is the GPU still busy with the previous
//! layers of this step (host-bound) or already done (GPU-bound)? Reports the busy/dry fractions
//! plus the host wall time between entries.

use std::sync::Mutex;
use std::time::Instant;

use candle_core::Device;

use super::{record_round_pass, Probe, Window};

#[derive(Default)]
pub(crate) struct StepStats {
    /// Layer entries with a previous entry to compare against.
    pub layers: u64,
    pub busy_ns: u128,
    pub dry_ns: u128,
    pub wait_ns: u128,
    /// Previous entry and the event recorded on the stream there.
    last: Option<(Instant, Option<Event>)>,
}

static STEP_STATS: Mutex<StepStats> = Mutex::new(StepStats {
    layers: 0,
    busy_ns: 0,
    dry_ns: 0,
    wait_ns: 0,
    last: None,
});
static PROBE: Probe = Probe::new("NAIWA_STEP_DBG");
static WINDOW: Window = Window::new(48); // one token's worth of layers

#[cfg(feature = "cuda")]
mod imp {
    use candle_core::cuda::cudarc::driver::sys;
    use candle_core::Device;

    pub(super) struct Event(candle_core::cuda::cudarc::driver::CudaEvent);

    impl Event {
        pub(crate) fn new(device: &Device) -> candle_core::Result<Self> {
            let Device::Cuda(dev) = device else {
                unreachable!("caller checks is_cuda");
            };
            let event = dev
                .cuda_stream()
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .map_err(candle_core::Error::wrap)?;
            Ok(Self(event))
        }

        pub(crate) fn record(&self, device: &Device) -> candle_core::Result<()> {
            let Device::Cuda(dev) = device else {
                unreachable!("caller checks is_cuda");
            };
            self.0
                .record(&dev.cuda_stream())
                .map_err(candle_core::Error::wrap)
        }

        pub(crate) fn is_complete(&self) -> bool {
            // cudarc's `is_complete` returns `bool` (false on query errors too).
            self.0.is_complete()
        }
    }
}

#[cfg(feature = "cuda")]
use imp::Event;

/// Non-CUDA stub: `record_step_layer` returns before touching it, but the accumulator type must
/// still exist.
#[cfg(not(feature = "cuda"))]
struct Event;

#[cfg(not(feature = "cuda"))]
impl Event {
    fn new(_device: &Device) -> candle_core::Result<Self> {
        Ok(Self)
    }

    fn record(&self, _device: &Device) -> candle_core::Result<()> {
        Ok(())
    }

    fn is_complete(&self) -> bool {
        false
    }
}

/// Mark one layer entry. Always advances the round clock; with `NAIWA_STEP_DBG` set, also
/// accounts the busy/dry split and dumps it every 48 transitions.
pub(crate) fn record_step_layer(device: &Device, layer: usize) {
    record_round_pass(layer);
    if !PROBE.enabled(false) || !device.is_cuda() {
        return;
    }
    let due = {
        let mut s = STEP_STATS.lock().unwrap();
        let mut ticked = false;
        if let Some((t_last, ev_last)) = s.last.take() {
            let now = Instant::now();
            // The wall time this layer entry arrived after the previous one; the GPU was
            // BUSY for the part before the previous entry's work completed, DRY after.
            let wall = now.duration_since(t_last).as_nanos();
            match ev_last {
                Some(ev) if ev.is_complete() => s.dry_ns += wall,
                Some(_) => s.busy_ns += wall,
                None => s.wait_ns += wall,
            }
            s.layers += 1;
            ticked = true;
        }
        let recorded = Event::new(device)
            .and_then(|ev| ev.record(device).map(|_| ev))
            .ok();
        s.last = Some((Instant::now(), recorded));
        ticked && WINDOW.tick() == WINDOW.size()
    };
    if due {
        let _ = super::dump_step_stats();
    }
}

/// Take and reset the window, print the `STEP_DBG` line, and return the snapshot.
pub(crate) fn dump_step_stats() -> StepStats {
    let out = std::mem::take(&mut *STEP_STATS.lock().unwrap());
    if out.layers == 0 {
        return out;
    }
    let n = out.layers as f64;
    eprintln!(
        "STEP_DBG layers {} busy_ms/layer {:.3} dry_ms/layer {:.3} (no-event {:.3})",
        out.layers,
        out.busy_ns as f64 / 1e6 / n,
        out.dry_ns as f64 / 1e6 / n,
        out.wait_ns as f64 / 1e6 / n,
    );
    out
}
