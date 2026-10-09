//! `--cpu-moe`: routed experts live in host memory and run on the CPU, while everything else
//! (attention, DeltaNet, routers, shared experts, the MTP head, the KV cache) stays on the GPU.
//!
//! With `--expert-cache-gb <N>`, the most-used experts are additionally cached in VRAM for decode
//! (`expert_cache`). With `--prefill-stream`, long prefill chunks run the experts on the GPU,
//! streaming each layer's experts in while the previous layer computes (`prefill_stream`).

use std::sync::Arc;

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_core::moe_offload::{
    install_routed_experts_provider, ExpertLoadContext, RoutedExperts, RoutedExpertsProvider,
};
use mistralrs_core::MoEExperts;
use tracing::{debug, info};

use crate::copy_stream::CopyStreams;
use crate::expert_cache::GpuCacheMoeProvider;
use crate::host_memory::{advise_expert_stacks, pin_expert_stacks, PinnedStack};
use crate::prefill_stream::{PrefillStream, PrefillStreams};

/// Device-layer map that keeps every decoder layer on the main device; mistral.rs clamps the
/// count to the model's layer count. Taken-over experts are not counted, so auto mapping would
/// spill whole layers to the CPU.
pub const ALL_LAYERS_ON_DEVICE: usize = usize::MAX;

/// `prefill_stream`: `Some(min_tokens)` with `--prefill-stream`.
pub fn install_cpu_moe(expert_cache_gb: Option<f64>, prefill_stream: Option<usize>) -> Result<()> {
    let prefill = prefill_stream.map(PrefillStreams::new);
    if let Some(min_tokens) = prefill_stream {
        info!(
            "--prefill-stream: prefill chunks of {min_tokens} tokens or more run their experts on the GPU"
        );
    }
    if let Some(gb) = expert_cache_gb {
        install_routed_experts_provider(Arc::new(GpuCacheMoeProvider::new(gb, prefill)))?;
        info!(
            "--cpu-moe + expert cache: routed experts in host memory; {:.1} GB GPU cache",
            gb
        );
    } else {
        install_routed_experts_provider(Arc::new(CpuMoeProvider {
            prefill,
            copy_streams: CopyStreams::default(),
        }))?;
        info!("--cpu-moe: routed experts load into host memory and run on the CPU");
    }
    Ok(())
}

struct CpuMoeProvider {
    prefill: Option<PrefillStreams>,
    copy_streams: CopyStreams,
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

/// Routed experts holding the model's own weights in host memory, run by mistral.rs's own
/// kernels. MoE steps move activations in and the result back; long prefill chunks run on the
/// GPU instead when a prefill stream is attached.
pub(crate) struct CpuExperts {
    /// Page-locked registrations of `experts`' stacks (prefill streaming only).
    _pinned: Vec<PinnedStack>,
    experts: MoEExperts,
    layer_device: Device,
    layer: usize,
    prefill: Option<Arc<PrefillStream>>,
}

impl CpuExperts {
    pub(crate) fn new(
        experts: MoEExperts,
        layer_device: Device,
        layer: usize,
        prefill: Option<Arc<PrefillStream>>,
        pinned: Vec<PinnedStack>,
    ) -> Self {
        Self {
            _pinned: pinned,
            experts,
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
        let ys = self.experts.forward(
            &xs.to_device(&Device::Cpu)?,
            topk_weights.to_device(&Device::Cpu)?,
            &topk_ids.to_device(&Device::Cpu)?,
        )?;
        ys.to_device(&self.layer_device)
    }
}
