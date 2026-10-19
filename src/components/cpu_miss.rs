//! CPU miss experts: submit the CPU job, upload its result through the result doorbell.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};

use crate::executor;
use crate::expert_cache::CachedLayer;
use crate::cpu_experts::fused_decode as offload_fused_decode;

pub(crate) struct CpuPending {
    inner: executor::Pending,
}

impl CpuPending {
    pub(crate) fn upload(self) -> Result<Tensor> {
        self.inner.upload()
    }
}

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
        if let Some(ys) = offload_fused_decode(&this.cpu(), this.act(), &x, &ids, &ws, width)? {
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
    let pending = executor::cpu().submit_result(layer.slot_device(), rows, layer.hidden(), job)?;
    Ok(Some(CpuPending { inner: pending }))
}
