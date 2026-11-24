//! Transitional bridge adapting `naiwa_offload::pipeline::GpuCacheMoeProvider`
//! (`naiwa_quant::RoutedExpertsProvider`) to `mistralrs_core::moe_offload::RoutedExpertsProvider`.
//!
//! Deleted in Phase 5 once `naiwa-models` + `naiwa-runtime` replace `mistralrs-core`.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_core::layers::Activation as MistralAct;
use mistralrs_core::moe_offload::{
    ExpertLoadContext, RoutedExperts as MistralRoutedExperts,
    RoutedExpertsProvider as MistralRoutedExpertsProvider,
};
use naiwa_quant::{
    Activation as NaiwaAct, ExpertLoadRequest, ExpertSite, MoEExpertsConfig,
    RoutedExperts as NaiwaRoutedExperts, RoutedExpertsProvider as NaiwaRoutedExpertsProvider,
};

fn map_activation(act: MistralAct) -> Result<NaiwaAct> {
    match act {
        MistralAct::Silu => Ok(NaiwaAct::Silu),
        MistralAct::Swish => Ok(NaiwaAct::Swish),
        MistralAct::Gelu => Ok(NaiwaAct::Gelu),
        MistralAct::GeluPytorchTanh | MistralAct::NewGelu => Ok(NaiwaAct::GeluPytorchTanh),
        MistralAct::Relu => Ok(NaiwaAct::Relu),
        other => candle_core::bail!("unsupported MoE activation for naiwa-offload: {other:?}"),
    }
}

struct BridgedRoutedExperts(Arc<dyn NaiwaRoutedExperts>);

impl MistralRoutedExperts for BridgedRoutedExperts {
    fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor> {
        self.0.forward(xs, topk_weights, topk_ids)
    }
}

pub struct MistralRsMoeBridge {
    inner: Arc<dyn NaiwaRoutedExpertsProvider>,
}

impl MistralRsMoeBridge {
    pub fn new(inner: Arc<dyn NaiwaRoutedExpertsProvider>) -> Self {
        Self { inner }
    }
}

impl MistralRoutedExpertsProvider for MistralRsMoeBridge {
    fn build(&self, ctx: ExpertLoadContext<'_>) -> Result<Option<Arc<dyn MistralRoutedExperts>>> {
        let act = map_activation(ctx.act)?;
        let config = MoEExpertsConfig {
            num_experts: ctx.config.num_experts,
            num_experts_per_tok: ctx.config.num_experts_per_tok,
            hidden_size: ctx.config.hidden_size,
            moe_intermediate_size: ctx.config.moe_intermediate_size,
        };
        let router_weight = ctx
            .vb
            .pp("gate")
            .get((config.num_experts, config.hidden_size), "weight")
            .and_then(|t| t.to_device(&Device::Cpu))
            .and_then(|t| t.to_dtype(DType::F32))
            .ok();
        let load_stacks = |dev: &Device| {
            let builtin = ctx.load_builtin(dev.clone())?;
            builtin.expert_qtensors().ok_or_else(|| {
                candle_core::Error::Msg(
                    "expected GGUF quantized expert stacks (QTensor-backed MoE)".into(),
                )
            })
        };
        let req = ExpertLoadRequest {
            site: ExpertSite {
                layer: ctx.site.layer,
                is_mtp: ctx.site.is_mtp,
            },
            config: &config,
            layer_device: ctx.layer_device,
            act,
            router_weight,
            load_stacks: &load_stacks,
        };
        Ok(self
            .inner
            .build(req)?
            .map(|e| Arc::new(BridgedRoutedExperts(e)) as Arc<dyn MistralRoutedExperts>))
    }
}
