//! Diagnostic buffered host-clock events. No formatting or I/O on the token/CPU path.

use std::sync::OnceLock;
use std::time::Instant;

use super::sink::AsyncJsonSink;

struct Event {
    ns: u64,
    layer: usize,
    tick: u64,
    phase: &'static str,
    bytes: u64,
    expert: Option<usize>,
    slot: Option<usize>,
}

struct Recorder {
    origin: Instant,
    sink: AsyncJsonSink<Event>,
}

static RECORDER: OnceLock<Option<Recorder>> = OnceLock::new();

#[inline]
pub(crate) fn enabled() -> bool {
    crate::config::diag().schedule_file.is_some()
}

#[inline]
pub(crate) fn event(layer: usize, tick: u64, phase: &'static str) {
    bytes(layer, tick, phase, 0);
}

#[inline]
pub(crate) fn bytes(layer: usize, tick: u64, phase: &'static str, bytes: u64) {
    detail(layer, tick, phase, bytes, None, None);
}

pub(crate) fn detail(
    layer: usize,
    tick: u64,
    phase: &'static str,
    bytes: u64,
    expert: Option<usize>,
    slot: Option<usize>,
) {
    if !enabled() {
        return;
    }
    let recorder = RECORDER.get_or_init(|| {
        let path = crate::config::diag().schedule_file.as_ref()?;
        let sink = AsyncJsonSink::spawn("schedule-dump", path, 1_048_576, |e: Event| {
            serde_json::json!({
                "ns": e.ns,
                "layer": e.layer,
                "tick": e.tick,
                "phase": e.phase,
                "bytes": e.bytes,
                "expert": e.expert,
                "slot": e.slot,
            })
        });
        Some(Recorder {
            origin: Instant::now(),
            sink,
        })
    });
    if let Some(r) = recorder {
        let ns = Instant::now().duration_since(r.origin).as_nanos() as u64;
        r.sink.push(Event {
            ns,
            layer,
            tick,
            phase,
            bytes,
            expert,
            slot,
        });
    }
}
