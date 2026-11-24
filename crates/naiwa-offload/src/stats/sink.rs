//! Shared bounded asynchronous JSONL writer for diagnostic traces.
//!
//! Producers only push small numeric structs into a mutex-protected `Vec` — zero JSON formatting
//! or file I/O on the token or worker thread. A background thread drains the buffer once it has
//! been idle for 300 ms, formats each record via the caller's serializer, and writes a trailing
//! `{"flush": true, "dropped": N}` sentinel matching `harness/attribution/` expectations.

use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct Buffer<T> {
    items: Vec<T>,
    last: Instant,
    dropped: u64,
}

pub(crate) struct AsyncJsonSink<T> {
    capacity: usize,
    buffer: Arc<Mutex<Buffer<T>>>,
}

impl<T: Send + 'static> AsyncJsonSink<T> {
    pub(crate) fn spawn<F>(
        thread_name: &str,
        path: &OsStr,
        capacity: usize,
        mut format_item: F,
    ) -> Self
    where
        F: FnMut(T) -> serde_json::Value + Send + 'static,
    {
        let file = File::create(path).expect("create diagnostic trace file");
        let buffer = Arc::new(Mutex::new(Buffer {
            items: Vec::with_capacity(capacity),
            last: Instant::now(),
            dropped: 0,
        }));
        let writer = Arc::clone(&buffer);
        std::thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || {
                let mut out = BufWriter::new(file);
                loop {
                    std::thread::sleep(Duration::from_millis(100));
                    let batch = {
                        let mut guard = writer.lock().unwrap();
                        if guard.items.is_empty()
                            || guard.last.elapsed() < Duration::from_millis(300)
                        {
                            continue;
                        }
                        let items = std::mem::replace(&mut guard.items, Vec::with_capacity(capacity));
                        let dropped = guard.dropped;
                        guard.dropped = 0;
                        (items, dropped)
                    };
                    for item in batch.0 {
                        writeln!(out, "{}", format_item(item)).unwrap();
                    }
                    writeln!(
                        out,
                        "{}",
                        serde_json::json!({"flush": true, "dropped": batch.1})
                    )
                    .unwrap();
                    out.flush().unwrap();
                }
            })
            .expect("spawn diagnostic sink thread");
        Self { capacity, buffer }
    }

    pub(crate) fn push(&self, item: T) {
        let now = Instant::now();
        let mut guard = self.buffer.lock().unwrap();
        if guard.items.len() < self.capacity {
            guard.items.push(item);
        } else {
            guard.dropped += 1;
        }
        guard.last = now;
    }
}
