//! Model-level diagnostic stream partition; reusable events, background numeric output.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

use candle_core::cuda_backend::cudarc::driver::{sys, CudaEvent};
use candle_core::Tensor;

use super::sink::AsyncJsonSink;

const PHASES: [&str; 18] = [
    "layer_start",
    "attn_norm_end",
    "mix_down",
    "mix_act",
    "mix_up",
    "attn_mix_end",
    "mixer_start",
    "mixer_end",
    "mlp_norm_end",
    "mix_down",
    "mix_act",
    "mix_up",
    "mlp_mix_end",
    "moe_start",
    "routed_end",
    "shared_end",
    "moe_end",
    "layer_end",
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
    ms: [f32; 19],
}

static SINK: OnceLock<AsyncJsonSink<Row>> = OnceLock::new();

thread_local! {
    static PENDING: RefCell<Vec<Layer>> = const { RefCell::new(Vec::new()) };
    static FREE: RefCell<Vec<Layer>> = const { RefCell::new(Vec::new()) };
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

pub fn install() {
    if crate::config::diag().model_partition_file.is_some() || super::head_partition::enabled() {
        assert!(mistralrs_core::qwen4_split::set_boundary_probe(dispatch));
    }
}

fn dispatch(xs: &Tensor, layer: usize, phase: &'static str) {
    if phase.starts_with("sample_") || phase.starts_with("pipeline_") {
        super::head_partition::host_mark(phase);
        return;
    }
    if matches!(phase, "tail_start" | "head_start" | "head_end" | "next_model") {
        super::head_partition::mark(xs, phase);
        return;
    }
    if crate::config::diag().model_partition_file.is_some() {
        mark(xs, layer, phase);
    }
}

fn sink() -> &'static AsyncJsonSink<Row> {
    SINK.get_or_init(|| {
        let path = crate::config::diag()
            .model_partition_file
            .as_ref()
            .expect("NAIWA_MODEL_PARTITION_FILE");
        AsyncJsonSink::spawn("model-partition", path, 65_536, |r: Row| {
            let mut ep: Vec<_> = PHASES
                .iter()
                .zip(r.ms)
                .enumerate()
                .map(|(i, (p, t))| {
                    let name = match i {
                        2 => "attn_down",
                        3 => "attn_act",
                        4 => "attn_up",
                        9 => "mlp_down",
                        10 => "mlp_act",
                        11 => "mlp_up",
                        _ => p,
                    };
                    serde_json::json!([name, t])
                })
                .collect();
            ep.push(serde_json::json!(["next_layer", r.ms[18]]));
            serde_json::json!({
                "layer": r.layer,
                "rows": r.rows,
                "endpoints_ms": ep,
            })
        })
    })
}

fn mark(xs: &Tensor, layer: usize, phase: &'static str) {
    let stride = crate::config::diag().model_partition_stride;
    if phase == "layer_start" {
        ACTIVE.with(|a| a.set(layer % stride == 0));
    }
    if !ACTIVE.with(|a| a.get()) || !xs.device().is_cuda() {
        return;
    }
    let stream = xs.device().as_cuda_device().unwrap().cuda_stream();
    let sink = sink();
    PENDING.with(|p| {
        let mut p = p.borrow_mut();
        if phase == "layer_start" {
            let mut a = FREE.with(|f| f.borrow_mut().pop()).unwrap_or_else(|| Layer {
                layer,
                rows: 0,
                count: 0,
                events: (0..18)
                    .map(|_| {
                        stream
                            .context()
                            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .unwrap()
                    })
                    .collect(),
            });
            a.layer = layer;
            a.rows = 0; // Set from actual mixer input shape below.
            a.count = 0;
            p.push(a);
        }
        let a = p.last_mut().expect("model layer");
        // Final head's HC mix is outside an active decoder layer.
        if a.count == 18 && phase.starts_with("mix_") {
            return;
        }
        if phase == "mixer_start" {
            a.rows = xs.elem_count() / xs.dims().last().copied().unwrap_or(1);
        }
        assert_eq!(PHASES[a.count], phase);
        a.events[a.count].record(&stream).unwrap();
        a.count += 1;
        // Routed host reads in previous two layers completed before reaching layer_start here.
        if phase == "layer_start" && p.len() > 2 {
            let a = &p[0];
            assert_eq!(a.count, 18);
            let mut ms = [0.0f32; 19];
            for (i, e) in a.events.iter().enumerate() {
                ms[i] = a.events[0].elapsed_ms(e).unwrap();
            }
            ms[18] = a.events[0].elapsed_ms(&p[1].events[0]).unwrap();
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
