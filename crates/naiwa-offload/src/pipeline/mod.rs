//! Layer 3 (`pipeline`): mutually-exclusive Decode and Prefill execution regimes.
//!
//! Orchestrates Layer 2 (`memory`) and Layer 1 (`transport`) into two clear paths:
//! - [`decode`]: per-token/verify-step resident + staged + CPU-miss linear pipeline.
//! - [`prefill`]: whole-chunk streaming GPU prefill over pinned host stacks / loaned slab spans.
//! - [`provider`]: [`GpuCacheMoeProvider`] wiring layers, staging rings, and prefill streams.

pub(crate) mod decode;
mod observer;
pub mod prefill;
pub(crate) mod provider;
pub(crate) mod stages;

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use naiwa_quant::{Activation, MoEExperts, RoutedExperts};

use crate::memory::expert::fused_decode;
use crate::memory::host_weights::PinnedStack;
use crate::memory::slab::{BorrowGate, Slab};
use crate::memory::{CachedLayer, Layers};
use crate::transport::CopyStream;

/// Calls of at most this many tokens are decode-sized and take the fused CPU path in [`CpuExperts`].
const FUSED_MAX_TOKENS: usize = 16;

pub use provider::GpuCacheMoeProvider;
pub(crate) use stages::quiesce;

#[inline]
fn prefill_threshold() -> usize {
    crate::config::runtime().prefill_threshold
}

/// Transition from Decode into a Prefill chunk that borrows `[0, loaned)` of `slab`:
/// quiesce lookahead, raise `borrow_gate`, order `copy` after prior GPU work, zero the span,
/// empty overlapping cache slots, and synchronize prior fills.
pub(crate) fn enter_prefill_loan(
    layers: &Layers,
    borrow_gate: &BorrowGate,
    copy: &CopyStream,
    slab: &Slab,
    stride: usize,
    loaned: usize,
) -> candle_core::Result<usize> {
    quiesce(true);
    {
        let _guard = borrow_gate.lock_access();
        borrow_gate.set_borrowed(true);
    }
    let base = slab.base()?;
    copy.after_device_work()?;
    let stream = copy.cuda_stream();
    let zeroed = loaned.max(650usize << 20).min(slab.bytes());
    // SAFETY: the span is inside the live slab; memset is queued before any upload.
    unsafe {
        let rc = candle_core::cuda::cudarc::driver::sys::cuMemsetD8Async(
            base,
            0,
            zeroed,
            stream.cu_stream(),
        );
        if rc != candle_core::cuda::cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("memset of the borrowed span failed: {rc:?}");
        }
    }
    let emptied = layers.empty_range(base, base + loaned as u64);
    copy.synchronize()?;
    crate::dbg_note(
        "prefill_borrow",
        &format!(
            "chunk: borrowed {:.0} MiB of the pool in {stride} range(s), {emptied} slots emptied",
            (loaned as f64) / (1u64 << 20) as f64
        ),
    );
    Ok(emptied)
}

/// Transition back into Decode mode: release `borrow_gate`, resume lookahead at layer 0, and
/// refill any displaced loan slots.
#[inline]
pub(crate) fn enter_decode(layers: &Layers, borrow_gate: &BorrowGate, is_layer_0: bool) {
    borrow_gate.set_borrowed(false);
    if is_layer_0 {
        quiesce(false);
        if layers.refill_lent() > 0 {
            crate::dbg_note("prefill_borrow", "the loan was given back");
        }
    }
}

/// `RoutedExperts` entry point dispatching each layer call into either [`prefill`] or [`decode`].
pub(crate) struct CachedExperts {
    pub(crate) layer: Arc<CachedLayer>,
    pub(crate) prefill: Option<Arc<prefill::PrefillStream>>,
}

impl RoutedExperts for CachedExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let layer = &self.layer;
        let (batch, seq, hidden_dim) = xs.dims3()?;
        crate::dbg_note(
            "moe_forward",
            &format!("layer {}: batch {batch} seq {seq}", layer.weights.layer),
        );
        debug_assert_eq!(hidden_dim, layer.weights.hidden);
        let num_tokens = batch * seq;

        if layer.weights.layer == 0 && crate::config::diag().layer_dbg {
            eprintln!(
                "LAYER_DBG rows={num_tokens} prefill_threshold={} slab_borrowed={} prefill_engine={}",
                prefill_threshold(),
                layer.cache.borrow_gate.borrowed(),
                self.prefill.is_some(),
            );
        }
        if num_tokens > prefill_threshold() {
            let ids_vec = topk_ids.flatten_all()?.to_vec1::<u32>()?;
            {
                let mut st = layer.cache.state.lock().unwrap();
                for &id in &ids_vec {
                    st.hit_counts[id as usize] += 1;
                }
            }
            if !crate::config::runtime().no_prefill_fill && !Slab::borrowing() {
                if let Some(next) = layer.cache.layers.get(layer.weights.layer + 1) {
                    next.spawn_fill();
                }
                layer.spawn_fill();
            }
            if let Some(ps) = &self.prefill {
                if let Some(out) = ps.forward(layer.weights.layer, xs, &topk_weights, topk_ids)? {
                    return Ok(out);
                }
            }
            let out = layer.weights.cpu.forward(
                &xs.to_device(&Device::Cpu)?,
                topk_weights.to_device(&Device::Cpu)?,
                &topk_ids.to_device(&Device::Cpu)?,
            )?;
            return out.to_device(&layer.weights.slot_device);
        }

        // Decode regime: ensure slab loan is released and any displaced slots are refilled at layer 0.
        enter_decode(
            &layer.cache.layers,
            &layer.cache.borrow_gate,
            layer.weights.layer == 0,
        );
        decode::forward(layer, xs, &topk_weights, topk_ids, layer.weights.act)
    }
}

/// Routed experts holding weights in host memory (no GPU cache slots), delegating long prefill
/// chunks to [`prefill::PrefillStream`] when attached.
pub(crate) struct CpuExperts {
    _pinned: Vec<PinnedStack>,
    experts: Arc<MoEExperts>,
    act: Activation,
    layer_device: Device,
    layer: usize,
    prefill: Option<Arc<prefill::PrefillStream>>,
}

impl CpuExperts {
    pub(crate) fn new(
        experts: MoEExperts,
        act: Activation,
        layer_device: Device,
        layer: usize,
        prefill: Option<Arc<prefill::PrefillStream>>,
        pinned: Vec<PinnedStack>,
    ) -> Self {
        Self {
            _pinned: pinned,
            experts: Arc::new(experts),
            act,
            layer_device,
            layer,
            prefill,
        }
    }
}

impl RoutedExperts for CpuExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        if let Some(ps) = &self.prefill {
            if let Some(ys) = ps.forward(self.layer, xs, &topk_weights, topk_ids)? {
                return Ok(ys);
            }
        }
        let xs_cpu = xs.to_device(&Device::Cpu)?;
        let weights_cpu = topk_weights.to_device(&Device::Cpu)?;
        let ids_cpu = topk_ids.to_device(&Device::Cpu)?;
        let (batch, seq, hidden) = xs.dims3()?;
        let tokens = batch * seq;
        if tokens > FUSED_MAX_TOKENS {
            let ys = self.experts.forward(&xs_cpu, weights_cpu, &ids_cpu)?;
            return ys.to_device(&self.layer_device);
        }
        let (experts, act) = (Arc::clone(&self.experts), self.act);
        let run = move || -> candle_core::Result<Vec<f32>> {
            let x = xs_cpu
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let ids = ids_cpu
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let ws = weights_cpu
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if let Some(ys) = fused_decode(&experts, act, &x, &ids, &ws, ids.len() / tokens)? {
                return Ok(ys);
            }
            experts
                .forward(&xs_cpu, weights_cpu, &ids_cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        if seq == 1 && self.layer_device.is_cuda() {
            let ys = crate::transport::executor::cpu()
                .submit_result(&self.layer_device, batch, hidden, run)?
                .upload()?;
            return ys.reshape((batch, seq, hidden))?.to_dtype(xs.dtype());
        }
        Tensor::from_vec(run()?, (batch, seq, hidden), &Device::Cpu)?
            .to_dtype(xs.dtype())?
            .to_device(&self.layer_device)
    }
}
