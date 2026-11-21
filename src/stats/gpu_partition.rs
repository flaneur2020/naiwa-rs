//! Opt-in CUDA-event diagnostic with reusable events and idle background formatting.
//! Events span host submission gaps as well as kernels. No added stream synchronize.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use candle_core::cuda_backend::cudarc::driver::{sys, CudaEvent, CudaStream};

use super::sink::AsyncJsonSink;

const PHASES: [&str; 9] = [
    "route",
    "plan_submit",
    "preload_ready",
    "resident_end",
    "staging_ready",
    "staged_end",
    "promotion_end",
    "cpu_ready",
    "combine_end",
];

struct Layer {
    layer: usize,
    rows: usize,
    count: usize,
    events: Vec<CudaEvent>,
}

struct Row {
    layer: usize,
    rows: usize,
    ms: [f32; 10],
}

static SINK: OnceLock<AsyncJsonSink<Row>> = OnceLock::new();

thread_local! {
    static PENDING: RefCell<Vec<Layer>> = const { RefCell::new(Vec::new()) };
    static FREE: RefCell<Vec<Layer>> = const { RefCell::new(Vec::new()) };
}

#[inline]
pub(crate) fn enabled() -> bool {
    crate::config::diag().gpu_partition_file.is_some()
}

fn sink() -> &'static AsyncJsonSink<Row> {
    SINK.get_or_init(|| {
        let path = crate::config::diag()
            .gpu_partition_file
            .as_ref()
            .expect("NAIWA_GPU_PARTITION_FILE");
        AsyncJsonSink::spawn("gpu-partition", path, 65_536, |r: Row| {
            let mut endpoints: Vec<_> = PHASES
                .iter()
                .zip(r.ms.iter())
                .map(|(p, t)| serde_json::json!([p, t]))
                .collect();
            endpoints.push(serde_json::json!(["next_route", r.ms[9]]));
            serde_json::json!({
                "layer": r.layer,
                "rows": r.rows,
                "endpoints_ms": endpoints,
            })
        })
    })
}

pub(crate) fn mark(stream: &Arc<CudaStream>, layer: usize, rows: usize, phase: &'static str) {
    if !enabled() {
        return;
    }
    let sink = sink();
    PENDING.with(|p| {
        let mut p = p.borrow_mut();
        if phase == "route" {
            let mut entry = FREE.with(|f| f.borrow_mut().pop()).unwrap_or_else(|| Layer {
                layer,
                rows,
                count: 0,
                events: (0..9)
                    .map(|_| {
                        stream
                            .context()
                            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .expect("partition event")
                    })
                    .collect(),
            });
            entry.layer = layer;
            entry.rows = rows;
            entry.count = 0;
            p.push(entry);
        }
        let a = p.last_mut().expect("partition layer");
        assert_eq!(PHASES[a.count], phase);
        a.events[a.count].record(stream).expect("partition record");
        a.count += 1;
        // Two subsequent host routing reads ensure the first two route events completed.
        if phase == "route" && p.len() > 2 {
            let a = &p[0];
            assert_eq!(a.count, 9);
            let start = &a.events[0];
            let mut ms = [0.0f32; 10];
            for (i, e) in a.events.iter().enumerate() {
                ms[i] = start.elapsed_ms(e).expect("partition elapsed");
            }
            ms[9] = start.elapsed_ms(&p[1].events[0]).expect("partition span");
            sink.push(Row {
                layer: a.layer,
                rows: a.rows,
                ms,
            });
            let old = p.remove(0);
            FREE.with(|f| f.borrow_mut().push(old));
        }
    });
}
