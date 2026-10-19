//! The `RoutedExpertsProvider` used by `--cpu-moe` without `--expert-cache-gb`: every routed
//! expert stays in host memory and runs on the CPU.

use std::sync::Arc;

use candle_core::Device;
use mistralrs_core::moe_offload::{ExpertLoadContext, RoutedExperts, RoutedExpertsProvider};
use tracing::debug;

use crate::copy_stream::CopyStreams;
use crate::cpu_experts::{attach_prefill, CpuExperts};
use crate::host_memory::advise_expert_stacks;
use crate::prefill_stream::PrefillStreams;

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
