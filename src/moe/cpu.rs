//! CPU routed experts.
//!
//! `CpuExperts` owns the model's host weights and the optional prefill-stream attachment; its
//! decode-sized path calls the fused decode in `stages/cpu_miss.rs`. The cache provider uses it
//! for the MTP head, for layers with no cache slots, and when the cache budget is zero.

use std::sync::Arc;

use super::fused_decode;
use crate::copy_stream::CopyStreams;
use crate::host_memory::{pin_expert_stacks, PinnedStack};
use crate::prefill_stream::{PrefillStream, PrefillStreams};
use candle_core::{DType, Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::{ExpertLoadContext, RoutedExperts};
use mistralrs_core::MoEExperts;

/// With `--prefill-stream`, pin `experts`' stacks and attach the layer to its device's prefill
/// stream. The pinned registrations must live as long as the layer.
pub(crate) fn attach_prefill(
    prefill: &Option<PrefillStreams>,
    ctx: &ExpertLoadContext<'_>,
    experts: &MoEExperts,
    copy_streams: &CopyStreams,
) -> candle_core::Result<(Vec<PinnedStack>, Option<Arc<PrefillStream>>)> {
    let Some(streams) = prefill else {
        return Ok((Vec::new(), None));
    };
    // Prefill uploads read straight from the stacks, asynchronously: they must be pinned.
    let (pinned, failed) = pin_expert_stacks(experts, ctx.layer_device);
    let stream = streams.attach(ctx, experts, failed == 0, copy_streams)?;
    Ok((pinned, stream))
}

/// Calls of at most this many tokens are decode-sized (one token, or a speculative verify step)
/// and take the fused CPU path; longer ones are prefill chunks, where the generic path's routed
/// gemvs already amortize their per-call cost.
pub(crate) const FUSED_MAX_TOKENS: usize = 16;

/// Routed experts holding the model's own weights in host memory, run by mistral.rs's own
/// kernels. MoE steps move activations in and the result back; long prefill chunks run on the
/// GPU instead when a prefill stream is attached.
pub(crate) struct CpuExperts {
    /// Page-locked registrations of `experts`' stacks (prefill streaming only).
    _pinned: Vec<PinnedStack>,
    experts: Arc<MoEExperts>,
    /// The experts' activation, for the fused decode path.
    act: Activation,
    layer_device: Device,
    layer: usize,
    prefill: Option<Arc<PrefillStream>>,
}

impl CpuExperts {
    pub(crate) fn new(
        experts: MoEExperts,
        act: Activation,
        layer_device: Device,
        layer: usize,
        prefill: Option<Arc<PrefillStream>>,
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
            // A prefill chunk.
            let ys = self.experts.forward(&xs_cpu, weights_cpu, &ids_cpu)?;
            return ys.to_device(&self.layer_device);
        }
        // One token, or a speculative verify step's rows.
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
            // Decode: return before the CPU experts finish; the next layer's launches overlap
            // them (`deferred`).
            let ys = crate::executor::cpu()
                .submit_result(&self.layer_device, batch, hidden, run)?
                .upload()?;
            return ys.reshape((batch, seq, hidden))?.to_dtype(xs.dtype());
        }
        Tensor::from_vec(run()?, (batch, seq, hidden), &Device::Cpu)?
            .to_dtype(xs.dtype())?
            .to_device(&self.layer_device)
    }
}
