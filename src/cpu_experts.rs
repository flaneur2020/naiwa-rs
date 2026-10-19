//! CPU-side routed experts: the `CpuExperts` fallback, its fused decode, and the
//! prefill-stream attachment. This is execution code, not model wiring; `load.rs` installs it.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::{ExpertLoadContext, RoutedExperts, RoutedExpertsProvider};
use mistralrs_core::MoEExperts;
use tracing::debug;

use crate::copy_stream::CopyStreams;
use crate::host_memory::{advise_expert_stacks, pin_expert_stacks, PinnedStack};
use crate::prefill_stream::{PrefillStream, PrefillStreams};

pub(crate) struct CpuMoeProvider {
    prefill: Option<PrefillStreams>,
    copy_streams: CopyStreams,
}

impl CpuMoeProvider {
    pub(crate) fn new(prefill: Option<PrefillStreams>) -> Self {
        Self {
            prefill,
            copy_streams: CopyStreams::default(),
        }
    }
}

impl RoutedExpertsProvider for CpuMoeProvider {
    fn build(
        &self,
        ctx: ExpertLoadContext<'_>,
    ) -> candle_core::Result<Option<Arc<dyn RoutedExperts>>> {
        if ctx.layer_device.is_cpu() {
            return Ok(None);
        }
        debug!(
            "Routed experts of {} layer {} load on the CPU",
            if ctx.site.is_mtp { "MTP" } else { "decoder" },
            ctx.site.layer
        );
        let experts = ctx.load_builtin(Device::Cpu)?;
        let huge_bytes = advise_expert_stacks(&experts)?;
        debug!("  {} MiB of its stacks on huge pages", huge_bytes >> 20);
        let (pinned, prefill) = attach_prefill(&self.prefill, &ctx, &experts, &self.copy_streams)?;
        Ok(Some(Arc::new(CpuExperts::new(
            experts,
            ctx.act,
            ctx.layer_device.clone(),
            ctx.site.layer,
            prefill,
            pinned,
        ))))
    }
}

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
