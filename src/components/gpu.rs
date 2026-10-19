//! GPU expert compute: fused decode over resident/staged slots.

use std::sync::Arc;

use candle_core::quantized::QMatMul;
use candle_core::{DType, Module, Tensor};
use mistralrs_core::layers::Activation;

use crate::expert_cache::{Bank, CachedLayer, Projections, SlotData};

use super::Hit;

/// The fused MoE decode kernel's activation id, if it has one for `act`.
#[cfg(feature = "cuda")]
fn fused_act(act: Activation) -> Option<i32> {
    match act {
        Activation::Silu | Activation::Swish => Some(mistralrs_quant::ACT_SILU),
        Activation::GeluPytorchTanh => Some(mistralrs_quant::ACT_GELU_PYTORCH_TANH),
        _ => None,
    }
}

/// `sum_j weights[t, j] * expert(bank, ids[t, j])(x2d[t])` for each of `rows` tokens,
/// `[rows, hidden]` F32, by mistral.rs's fused MoE decode (quantize, gate+up+act, quantize,
/// down+weighted sum). `ids` and `weights` are row-major `[rows, topk]`.
#[cfg(feature = "cuda")]
fn fused_decode(
    bank: &Bank,
    x2d: &Tensor,
    ids: Vec<u32>,
    weights: Vec<f32>,
    rows: usize,
    act_type: i32,
) -> candle_core::Result<Tensor> {
    use candle_core::cuda::cudarc::driver::DevicePtr;
    let dev = x2d.device().as_cuda_device()?;
    let topk = ids.len() / rows;
    let ids = Tensor::new(ids, x2d.device())?;
    let weights = Tensor::new(weights, x2d.device())?;
    let (ids_storage, _) = ids.storage_and_layout();
    let candle_core::Storage::Cuda(ids_cuda) = &*ids_storage else {
        candle_core::bail!("expert cache: fused decode ids are not on CUDA");
    };
    let (w_storage, _) = weights.storage_and_layout();
    let candle_core::Storage::Cuda(w_cuda) = &*w_storage else {
        candle_core::bail!("expert cache: fused decode weights are not on CUDA");
    };
    let w_slice = w_cuda.as_cuda_slice::<f32>()?;
    let w_ptr = w_slice.device_ptr(w_slice.stream()).0 as *const f32;
    // SAFETY: `w_ptr` points at `topk` f32s of `weights`, whose free is ordered on candle's
    // stream after the kernels queued here.
    unsafe {
        mistralrs_quant::indexed_moe_fused_decode(
            &bank.proj.gate,
            &bank.proj.up,
            &bank.proj.down,
            x2d,
            ids_cuda.as_cuda_slice::<u32>()?,
            w_ptr,
            rows,
            topk,
            act_type,
            dev,
        )
    }
}

/// A resident expert a token is routed to: its slot buffer, slot index and routing weight.
/// The weighted sum of the hits' outputs, `[rows, hidden]` F32 on the slot device, for
/// `hits[t]` the hits of row `t` of `x2d: [rows, hidden]`. In copy-stream mode each bank's hits
/// run as one fused MoE decode over every row (four kernels); otherwise every expert runs on
/// its own.
pub(crate) fn run(
    layer: &CachedLayer,
    hits: &[Vec<Hit>],
    x2d: &Tensor,
    act: Activation,
) -> candle_core::Result<(Tensor, Option<Arc<Bank>>)> {
    let rows = hits.len();
    // M9: no captures, so nothing to remember for the CPU-return node.
    let capture_bank: Option<Arc<Bank>> = None;
    let mut acc: Option<Tensor> = None;
    let mut add = |y: Tensor| -> candle_core::Result<()> {
        acc = Some(match acc.take() {
            Some(a) => (a + y)?,
            None => y,
        });
        Ok(())
    };
    #[cfg(feature = "cuda")]
    if let (Some(act_type), true) = (
        fused_act(act),
        hits.iter()
            .flatten()
            .all(|(buf, _, _)| matches!(buf.data, SlotData::Banked { .. })),
    ) {
        // Per bank, each row's (local slot, weight) pairs. The kernel takes the same number
        // per row, so short rows are padded with a weight-0 entry on a slot of that bank that
        // some row hits (published and fenced, so it holds finite values).
        let mut groups: Vec<(&Arc<Bank>, Vec<Vec<(u32, f32)>>)> = Vec::new();
        for (t, row) in hits.iter().enumerate() {
            for (buf, slot, w) in row {
                let SlotData::Banked { bank, .. } = &buf.data else {
                    unreachable!("checked above");
                };
                let local = (slot - bank.first) as u32;
                let i = match groups.iter().position(|(b, _)| Arc::ptr_eq(b, bank)) {
                    Some(i) => i,
                    None => {
                        groups.push((bank, vec![Vec::new(); rows]));
                        groups.len() - 1
                    }
                };
                groups[i].1[t].push((local, *w));
            }
        }
        for (bank, per_row) in groups {
            let width = per_row.iter().map(Vec::len).max().unwrap_or(0);
            let pad = per_row.iter().flatten().next().map_or(0, |&(l, _)| l);
            let mut ids = Vec::with_capacity(rows * width);
            let mut ws = Vec::with_capacity(rows * width);
            for row in &per_row {
                for i in 0..width {
                    let (l, w) = row.get(i).copied().unwrap_or((pad, 0.0));
                    ids.push(l);
                    ws.push(w);
                }
            }
            add(fused_decode(bank, x2d, ids, ws, rows, act_type)?)?;
        }
        return match acc {
            Some(acc) => Ok((acc, capture_bank)),
            None => Ok((
                Tensor::zeros((rows, layer.hidden()), DType::F32, &layer.slot_device())?,
                None,
            )),
        };
    }
    // One expert at a time, row by row.
    let mut out = Vec::with_capacity(rows);
    for (t, row) in hits.iter().enumerate() {
        let x = x2d.narrow(0, t, 1)?;
        let mut sum = Tensor::zeros((1, layer.hidden()), DType::F32, &layer.slot_device())?;
        for (buf, _, w) in row {
            let SlotData::Owned(proj) = &buf.data else {
                candle_core::bail!("expert cache: slot has no weights to run");
            };
            let y = run_expert(proj, &x, act)?;
            sum = (sum + (y.to_dtype(DType::F32)? * *w as f64)?)?;
        }
        out.push(sum);
    }
    add(Tensor::cat(&out, 0)?)?;
    Ok((acc.expect("added above"), capture_bank))
}

/// Run one token through a cached expert. `x2d: [1, hidden]` on the slot device.
fn run_expert(w: &Projections, x2d: &Tensor, act: Activation) -> candle_core::Result<Tensor> {
    let g = QMatMul::QTensor(w.gate.clone()).forward(x2d)?;
    let u = QMatMul::QTensor(w.up.clone()).forward(x2d)?;
    let h = (act.forward(&g)? * u)?;
    QMatMul::QTensor(w.down.clone()).forward(&h)
}
