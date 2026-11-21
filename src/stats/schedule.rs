//! Diagnostic buffered host-clock events. No formatting or I/O on the token/CPU path.
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

struct Event {
    ns: u64,
    layer: usize,
    tick: u64,
    phase: &'static str,
    bytes: u64,
    expert: Option<usize>,
    slot: Option<usize>,
}
struct Buffer {
    events: Vec<Event>,
    last: Instant,
    dropped: u64,
}
struct Recorder {
    origin: Instant,
    buffer: Mutex<Buffer>,
}
static RECORDER: OnceLock<Option<std::sync::Arc<Recorder>>> = OnceLock::new();

#[inline]
pub(crate) fn enabled() -> bool {
    crate::config::diag().schedule_file.is_some()
}

pub(crate) fn event(layer: usize, tick: u64, phase: &'static str) {
    bytes(layer, tick, phase, 0);
}

pub(crate) fn bytes(layer: usize, tick: u64, phase: &'static str, bytes: u64) {
    detail(layer, tick, phase, bytes, None, None);
}

pub(crate) fn detail(layer: usize, tick: u64, phase: &'static str, bytes: u64, expert: Option<usize>, slot: Option<usize>) {
    if !enabled() { return; }
    let recorder = RECORDER.get_or_init(|| {
        let path = crate::config::diag().schedule_file.as_ref()?;
        let file = std::fs::File::create(path).expect("create schedule event file");
        let r = std::sync::Arc::new(Recorder {
            origin: Instant::now(),
            buffer: Mutex::new(Buffer { events: Vec::with_capacity(1048576), last: Instant::now(), dropped: 0 }),
        });
        let writer = r.clone();
        std::thread::Builder::new().name("schedule-dump".into()).spawn(move || {
            use std::io::Write;
            let mut file = std::io::BufWriter::new(file);
            loop {
                std::thread::sleep(Duration::from_millis(100));
                let batch = {
                    let mut b = writer.buffer.lock().unwrap();
                    if b.events.is_empty() || b.last.elapsed() < Duration::from_millis(300) { continue; }
                    let dropped = b.dropped;
                    b.dropped = 0;
                    (std::mem::replace(&mut b.events, Vec::with_capacity(1048576)), dropped)
                };
                for e in batch.0 {
                    writeln!(file, "{}", serde_json::json!({"ns": e.ns, "layer": e.layer, "tick": e.tick, "phase": e.phase, "bytes": e.bytes, "expert": e.expert, "slot": e.slot})).unwrap();
                }
                writeln!(file, "{}", serde_json::json!({"flush": true, "dropped": batch.1})).unwrap();
                file.flush().unwrap();
            }
        }).expect("schedule dump thread");
        Some(r)
    });
    if let Some(r) = recorder {
        let now = Instant::now();
        let mut b = r.buffer.lock().unwrap();
        if b.events.len() < 1048576 {
            b.events.push(Event { ns: now.duration_since(r.origin).as_nanos() as u64, layer, tick, phase, bytes, expert, slot });
        } else { b.dropped += 1; }
        b.last = now;
    }
}
