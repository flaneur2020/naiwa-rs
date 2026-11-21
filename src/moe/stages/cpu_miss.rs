//! CPU miss experts: the fused decode, submit the CPU job, upload its result through the result
//! doorbell.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::MoEExperts;

use crate::executor;
use crate::moe::CachedLayer;

/// A decode step's routed experts on the CPU in two parallel passes
/// (`mistralrs_quant::fused_moe_decode`): gate+up+act per distinct (expert, row), then down with
/// the router-weighted sum. At decode sizes the generic `MoEExperts::forward` - three routed
/// gemvs with candle ops between them - spends more on its per-call structure than on the dots.
///
/// `xs` is `rows x hidden` F32; `ids` and `weights` are row-major `[rows, topk]`, weight-0 slots
/// skipped. `None` when the generic path must run instead: an activation other than SiLU, experts
/// that are not GGUF stacks, a dtype or shape the fused path does not serve, or
pub(crate) fn fused_decode(
    experts: &MoEExperts,
    act: Activation,
    xs: &[f32],
    ids: &[u32],
    weights: &[f32],
    topk: usize,
) -> candle_core::Result<Option<Vec<f32>>> {
    // M9: one fixed behaviour - the fused path for the activations it supports, the generic one otherwise.
    if !matches!(act, Activation::Silu | Activation::Swish) {
        return Ok(None);
    }
    let Some((gate, up, down)) = experts.expert_qtensors() else {
        return Ok(None);
    };
    mistralrs_quant::fused_moe_decode(&gate, &up, &down, xs, ids, weights, topk)
}

pub(crate) struct CpuPending {
    inner: executor::Pending,
    timeline: Option<(std::time::Instant, usize, u64)>,
}

impl CpuPending {
    pub(crate) fn upload(self) -> Result<Tensor> {
        if let Some((origin, layer, tick)) = self.timeline {
            eprintln!("CPU_UPLOAD layer={layer} tick={tick} host_enqueue_us={}", origin.elapsed().as_micros());
        }
        self.inner.upload()
    }
}

pub(crate) struct CpuMiss;

impl CpuMiss {
    pub(crate) fn submit(
        layer: &Arc<CachedLayer>,
        misses: &[Vec<(u32, f32)>],
        x2d: &Tensor,
        rows: usize,
        route_ready: Option<std::time::Instant>,
    ) -> Result<Option<CpuPending>> {
        if misses.iter().all(Vec::is_empty) {
            return Ok(None);
        }
        let schedule_tick = if crate::stats::schedule::enabled() {
            layer.state().lock().unwrap().tick
        } else { 0 };
        let schedule_layer = layer.layer();
        let timeline = route_ready.map(|origin| (origin, layer.layer(), layer.state().lock().unwrap().tick));
        let enter_us = route_ready.map(|t| t.elapsed().as_micros()).unwrap_or(0);
        let width = misses.iter().map(Vec::len).max().unwrap_or(0);
        let pad = misses.iter().flatten().next().map_or(0, |&(id, _)| id);
        let mut ids = Vec::with_capacity(rows * width);
        let mut ws = Vec::with_capacity(rows * width);
        for row in misses {
            for i in 0..width {
                let (id, w) = row.get(i).copied().unwrap_or((pad, 0.0));
                ids.push(id);
                ws.push(w);
            }
        }
        let copy_begin_us = route_ready.map(|t| t.elapsed().as_micros()).unwrap_or(0);
        let xs_cpu = x2d.unsqueeze(0)?.to_device(&Device::Cpu)?;
        crate::stats::schedule::event(schedule_layer, schedule_tick, "cpu_submit");
        let submit_us = route_ready.map(|t| t.elapsed().as_micros()).unwrap_or(0);
        let n = rows * layer.hidden();
        let this = Arc::clone(layer);
        let job = move || {
            crate::stats::schedule::event(schedule_layer, schedule_tick, "cpu_worker");
            let worker_us = route_ready.map(|t| t.elapsed().as_micros()).unwrap_or(0);
            let x = xs_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let trace = crate::config::diag().cpu_layer_trace;
            let started = trace.then(std::time::Instant::now);
            let result = fused_decode(&this.cpu(), this.act(), &x, &ids, &ws, width)?;
            if let Some(started) = started {
                // Time computation only; formatting and stderr are outside the measurement.
                let us = started.elapsed().as_micros();
                let active: Vec<_> = ids.iter().zip(&ws).filter(|(_, w)| **w != 0.0).map(|(id, _)| *id).collect();
                let mut distinct = active.clone();
                distinct.sort_unstable();
                distinct.dedup();
                let dtype = this.cpu().expert_qtensors().map(|(g, _, _)| g.dtype());
                eprintln!("CPU_IDS layer={} rows={} ids={:?}", this.layer(), rows, active);
                eprintln!("CPU_LAYER layer={} rows={} pairs={} experts={} us={} dtype={:?} fused={}",
                    this.layer(), rows, active.len(), distinct.len(), us, dtype, result.is_some());
            }
            if let Some((origin, layer, tick)) = timeline {
                // Absolute offsets from Route::submit return. This is compute return, not publication
                // or GPU consumption. Logging occurs after the timestamp and perturbs publication.
                let done_us = origin.elapsed().as_micros();
                eprintln!("CPU_TASK layer={layer} tick={tick} rows={rows} enter_us={enter_us} copy_begin_us={copy_begin_us} submit_us={submit_us} worker_us={worker_us} compute_return_us={done_us} fused={} ids={ids:?} weights={:?}", result.is_some(), ws.iter().map(|w| w.to_bits()).collect::<Vec<_>>());
            }
            if let Some(ys) = result {
                crate::stats::schedule::event(schedule_layer, schedule_tick, "cpu_compute_return");
                return Ok(ys);
            }
            let xs_cpu = Tensor::from_vec(x, (1, rows, n / rows.max(1)), &Device::Cpu)?;
            let miss_w_t = Tensor::from_vec(ws, (rows, width), &Device::Cpu)?;
            let miss_ids_t = Tensor::from_vec(ids, (rows, width), &Device::Cpu)?;
            this.cpu()
                .forward(&xs_cpu, miss_w_t, &miss_ids_t)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        let device = layer.slot_device().clone();
        let hidden = layer.hidden();
        let tag = crate::stats::schedule::enabled().then_some((schedule_layer, schedule_tick));
        let pending = executor::cpu().submit_result_tagged(&device, rows, hidden, tag, job)?;
        Ok(Some(CpuPending { inner: pending, timeline }))
    }
}
