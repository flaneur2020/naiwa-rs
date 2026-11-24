//! Tail-only diagnostic: GPU event intervals and sampler host counters, no stream synchronization.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{mpsc, OnceLock};
use std::time::Instant;

use candle_core::cuda_backend::cudarc::driver::{sys, CudaEvent};
use candle_core::Tensor;

const EXPECTED: [&str; 4] = ["tail_start", "head_start", "head_end", "next_model"];

struct Tail {
    events: Vec<CudaEvent>,
    count: usize,
    rows: usize,
    next_rows: usize,
    host: Instant,
}

#[derive(Default)]
struct Sample {
    begin: Option<Instant>,
    sum_ns: u64,
    calls: u64,
    device_greedy: u64,
    blocks: [u64; 6],
    pipeline: BTreeMap<&'static str, u64>,
    cpu_begin: Option<Instant>,
    cpu_ns: u64,
}

thread_local! {
    static SAMPLE: RefCell<Sample> = RefCell::new(Sample::default());
    static PENDING: RefCell<Vec<Tail>> = const { RefCell::new(Vec::new()) };
    static FREE: RefCell<Vec<Tail>> = const { RefCell::new(Vec::new()) };
}

static TX: OnceLock<mpsc::SyncSender<serde_json::Value>> = OnceLock::new();

#[inline]
pub(crate) fn enabled() -> bool {
    crate::config::diag().head_partition_file.is_some()
}

pub(crate) fn host_mark(phase: &'static str) {
    if !enabled() {
        return;
    }
    SAMPLE.with(|s| {
        let mut s = s.borrow_mut();
        match phase {
            "pipeline_cpu_begin" => s.cpu_begin = Some(Instant::now()),
            "pipeline_cpu_end" => {
                if let Some(t) = s.cpu_begin.take() {
                    s.cpu_ns += t.elapsed().as_nanos() as u64;
                }
            }
            p if p.starts_with("pipeline_") => {
                *s.pipeline.entry(p).or_default() += 1;
            }
            "sample_begin" => {
                assert!(s.begin.is_none());
                s.begin = Some(Instant::now());
            }
            "sample_device_greedy" => s.device_greedy += 1,
            "sample_end" => {
                let start = s.begin.take().expect("sample begin");
                s.sum_ns += start.elapsed().as_nanos() as u64;
                s.calls += 1;
            }
            "sample_block_cpu_input" => s.blocks[5] += 1,
            "sample_block_logprobs" => s.blocks[0] += 1,
            "sample_block_speculative" => s.blocks[1] += 1,
            "sample_block_multi" => s.blocks[2] += 1,
            "sample_block_temperature" => s.blocks[3] += 1,
            "sample_block_processors" => s.blocks[4] += 1,
            _ => unreachable!(),
        }
    });
}

pub(crate) fn mark(xs: &Tensor, phase: &str) {
    if !enabled() || !xs.device().is_cuda() {
        return;
    }
    let tx = TX.get_or_init(|| {
        let (tx, rx) = mpsc::sync_channel::<serde_json::Value>(4096);
        let path = crate::config::diag()
            .head_partition_file
            .as_ref()
            .expect("NAIWA_HEAD_PARTITION_FILE");
        let file = std::fs::File::create(path).expect("head file");
        std::thread::Builder::new()
            .name("head-partition".into())
            .spawn(move || {
                let mut f = std::io::BufWriter::new(file);
                while let Ok(v) = rx.recv() {
                    writeln!(f, "{v}").unwrap();
                    f.flush().unwrap();
                }
            })
            .expect("spawn head-partition writer");
        tx
    });
    let stream = xs.device().as_cuda_device().unwrap().cuda_stream();
    PENDING.with(|p| {
        let mut p = p.borrow_mut();
        if phase == "tail_start" {
            let mut a = FREE.with(|f| f.borrow_mut().pop()).unwrap_or_else(|| Tail {
                events: (0..4)
                    .map(|_| {
                        stream
                            .context()
                            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .unwrap()
                    })
                    .collect(),
                count: 0,
                rows: 0,
                next_rows: 0,
                host: Instant::now(),
            });
            a.count = 0;
            a.rows = xs.elem_count() / xs.dims().last().copied().unwrap_or(1);
            a.host = Instant::now();
            p.push(a);
        }
        if phase == "next_model" && p.last().is_none_or(|a| a.count != 3) {
            return;
        }
        let a = p.last_mut().unwrap();
        if phase == "next_model" {
            a.next_rows = xs.elem_count() / xs.dims().last().copied().unwrap_or(1);
        }
        assert_eq!(EXPECTED[a.count], phase);
        a.events[a.count].record(&stream).unwrap();
        a.count += 1;
        if phase == "next_model" {
            SAMPLE.with(|s| {
                let mut s = s.borrow_mut();
                tx.try_send(serde_json::json!({
                    "kind": "sampler_host",
                    "calls": s.calls,
                    "device_greedy": s.device_greedy,
                    "blocks": s.blocks,
                    "pipeline": s.pipeline,
                    "pipeline_cpu_ms": s.cpu_ns as f64 / 1e6,
                    "total_ms": s.sum_ns as f64 / 1e6,
                    "next_rows": a.next_rows,
                }))
                .expect("sample trace overflow");
                s.calls = 0;
                s.device_greedy = 0;
                s.sum_ns = 0;
                s.blocks = [0; 6];
                s.pipeline.clear();
                s.cpu_ns = 0;
            });
        }
        if phase == "tail_start" && p.len() > 2 {
            let a = &p[0];
            assert_eq!(a.count, 4);
            let ms: Vec<_> = a
                .events
                .iter()
                .map(|e| a.events[0].elapsed_ms(e).unwrap())
                .collect();
            tx.try_send(serde_json::json!({
                "rows": a.rows,
                "next_rows": a.next_rows,
                "endpoints_ms": ms,
                "host_tail_start_to_next_tail_ms": p[1].host.duration_since(a.host).as_secs_f64() * 1000.0,
            }))
            .expect("head trace overflow");
            let a = p.remove(0);
            FREE.with(|f| f.borrow_mut().push(a));
        }
    });
}
