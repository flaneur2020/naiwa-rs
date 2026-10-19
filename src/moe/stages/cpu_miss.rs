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
    static GENERIC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // M9: one fixed behaviour - the fused path for the activations it supports, the generic one otherwise.
    let _ = &GENERIC;
    if !matches!(act, Activation::Silu | Activation::Swish) {
        return Ok(None);
    }
    let Some((gate, up, down)) = experts.expert_qtensors() else {
        return Ok(None);
    };
    let t0 = std::time::Instant::now();
    let ys = mistralrs_quant::fused_moe_decode(&gate, &up, &down, xs, ids, weights, topk);
    crate::stats::record_fused_decode(t0, ids, weights, topk);
    ys
}

pub(crate) struct CpuPending {
    inner: executor::Pending,
}

impl CpuPending {
    pub(crate) fn upload(self) -> Result<Tensor> {
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
    ) -> Result<Option<CpuPending>> {
        if misses.iter().all(Vec::is_empty) {
            return Ok(None);
        }
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
        let xs_cpu = x2d.unsqueeze(0)?.to_device(&Device::Cpu)?;
        let this = Arc::clone(layer);
        let job = move || {
            let x = xs_cpu
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if let Some(ys) = fused_decode(&this.cpu(), this.act(), &x, &ids, &ws, width)? {
                return Ok(ys);
            }
            let miss_w_t = Tensor::from_vec(ws, (rows, width), &Device::Cpu)?;
            let miss_ids_t = Tensor::from_vec(ids, (rows, width), &Device::Cpu)?;
            this.cpu()
                .forward(&xs_cpu, miss_w_t, &miss_ids_t)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        let pending =
            executor::cpu().submit_result(layer.slot_device(), rows, layer.hidden(), job)?;
        Ok(Some(CpuPending { inner: pending }))
    }
}
