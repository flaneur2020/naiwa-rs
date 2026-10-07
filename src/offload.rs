//! `--cpu-moe`: routed experts live in host memory and run on the CPU, while everything else
//! (attention, DeltaNet, routers, shared experts, the MTP head, the KV cache) stays on the GPU.
//!
//! With `--expert-cache-gb <N>`, the most-used experts are additionally cached in VRAM in their
//! original quantized format. During decode (1 token) GPU cache hits and CPU misses run
//! sequentially; prefill routes everything through the CPU so every expert gets hit and admission
//! counts are primed.

use std::sync::Arc;

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_core::moe_offload::{
    install_routed_experts_provider, ExpertLoadContext, RoutedExperts, RoutedExpertsProvider,
};
use mistralrs_core::MoEExperts;
use tracing::{debug, info};

use crate::expert_cache::GpuCacheMoeProvider;
use crate::host_memory::advise_expert_stacks;

/// Device-layer map that keeps every decoder layer on the main device; mistral.rs clamps the
/// count to the model's layer count. Taken-over experts are not counted, so auto mapping would
/// spill whole layers to the CPU.
pub const ALL_LAYERS_ON_DEVICE: usize = usize::MAX;

pub fn install_cpu_moe(expert_cache_gb: Option<f64>) -> Result<()> {
    if let Some(gb) = expert_cache_gb {
        install_routed_experts_provider(Arc::new(GpuCacheMoeProvider::new(gb)))?;
        info!(
            "--cpu-moe + expert cache: routed experts in host memory; {:.1} GB GPU cache",
            gb
        );
    } else {
        install_routed_experts_provider(Arc::new(CpuMoeProvider))?;
        info!("--cpu-moe: routed experts load into host memory and run on the CPU");
    }
    Ok(())
}

struct CpuMoeProvider;

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
        Ok(Some(Arc::new(CpuExperts {
            experts,
            layer_device: ctx.layer_device.clone(),
        })))
    }
}

/// Routed experts holding the model's own weights in host memory, run by mistral.rs's own
/// kernels. MoE steps move activations in and the result back.
struct CpuExperts {
    experts: MoEExperts,
    layer_device: Device,
}

impl RoutedExperts for CpuExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let ys = self.experts.forward(
            &xs.to_device(&Device::Cpu)?,
            topk_weights.to_device(&Device::Cpu)?,
            &topk_ids.to_device(&Device::Cpu)?,
        )?;
        ys.to_device(&self.layer_device)
    }
}
