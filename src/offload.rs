//! `--cpu-moe`: routed experts live in host memory and run on the CPU, while everything else
//! (attention, DeltaNet, routers, shared experts, the MTP head, the KV cache) stays on the GPU.
//!
//! Built on mistral.rs's routed-experts hook (`mistralrs_core::moe_offload`): each MoE block's
//! experts are loaded with mistral.rs's own `MoEExperts` on `Device::Cpu`, and every MoE step
//! moves the activations and router top-k to the CPU and the result back.

use std::sync::Arc;

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_core::moe_offload::{
    install_routed_experts_provider, ExpertLoadContext, RoutedExperts, RoutedExpertsProvider,
};
use mistralrs_core::MoEExperts;
use tracing::{debug, info};

/// Device-layer map that keeps every decoder layer on the main device; mistral.rs clamps the
/// count to the model's layer count. Taken-over experts are not counted, so auto mapping would
/// spill whole layers to the CPU.
pub const ALL_LAYERS_ON_DEVICE: usize = usize::MAX;

pub fn install_cpu_moe() -> Result<()> {
    install_routed_experts_provider(Arc::new(CpuMoeProvider))?;
    info!("--cpu-moe: routed experts load into host memory and run on the CPU");
    Ok(())
}

struct CpuMoeProvider;

impl RoutedExpertsProvider for CpuMoeProvider {
    fn build(
        &self,
        ctx: ExpertLoadContext<'_>,
    ) -> candle_core::Result<Option<Arc<dyn RoutedExperts>>> {
        // Blocks already on the CPU gain nothing from a detour; load them as usual.
        if ctx.layer_device.is_cpu() {
            return Ok(None);
        }
        debug!(
            "Routed experts of {} layer {} load on the CPU",
            if ctx.site.is_mtp { "MTP" } else { "decoder" },
            ctx.site.layer
        );
        let experts = ctx.load_builtin(Device::Cpu)?;
        Ok(Some(Arc::new(CpuExperts {
            experts,
            layer_device: ctx.layer_device.clone(),
        })))
    }
}

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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, IndexOp};
    use mistralrs_core::layers::Activation;
    use mistralrs_core::moe_offload::ExpertSite;
    use mistralrs_core::{ExpertProjNames, MoEExpertsConfig};
    use mistralrs_quant::{Comm, Id, ShardedSafeTensors};

    use super::*;

    const EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const HIDDEN: usize = 8;
    const INTER: usize = 16;
    const TOKENS: usize = 3;

    struct Block {
        gate: Vec<Tensor>,
        up: Vec<Tensor>,
        down: Vec<Tensor>,
        tensors: HashMap<String, Tensor>,
    }

    fn block() -> Block {
        let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
        let mut tensors = HashMap::new();
        for i in 0..EXPERTS {
            let g = Tensor::randn(0f32, 0.5, (INTER, HIDDEN), &Device::Cpu).unwrap();
            let u = Tensor::randn(0f32, 0.5, (INTER, HIDDEN), &Device::Cpu).unwrap();
            let d = Tensor::randn(0f32, 0.5, (HIDDEN, INTER), &Device::Cpu).unwrap();
            tensors.insert(format!("experts.{i}.gate_proj.weight"), g.clone());
            tensors.insert(format!("experts.{i}.up_proj.weight"), u.clone());
            tensors.insert(format!("experts.{i}.down_proj.weight"), d.clone());
            gate.push(g);
            up.push(u);
            down.push(d);
        }
        Block {
            gate,
            up,
            down,
            tensors,
        }
    }

    fn with_ctx<T>(
        tensors: HashMap<String, Tensor>,
        f: impl FnOnce(ExpertLoadContext<'_>) -> T,
    ) -> T {
        let config = MoEExpertsConfig {
            num_experts: EXPERTS,
            num_experts_per_tok: TOP_K,
            hidden_size: HIDDEN,
            moe_intermediate_size: INTER,
            expert_proj_names: ExpertProjNames::DEFAULT,
        };
        let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1).unwrap());
        f(ExpertLoadContext {
            site: ExpertSite {
                layer: 0,
                is_mtp: false,
            },
            config: &config,
            vb: ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu),
            layer_device: &Device::Cpu,
            comm: &comm,
            loading_isq: false,
            quantization_config: &None,
            act: Activation::Silu,
        })
    }

    #[test]
    fn cpu_resident_blocks_load_as_usual() {
        let taken = with_ctx(block().tensors, |ctx| CpuMoeProvider.build(ctx).unwrap());
        assert!(taken.is_none());
    }

    #[test]
    fn cpu_experts_match_the_reference_moe() {
        let block = block();
        let experts = with_ctx(block.tensors.clone(), |ctx| {
            ctx.load_builtin(Device::Cpu).unwrap()
        });
        let cpu_experts = CpuExperts {
            experts,
            layer_device: Device::Cpu,
        };

        let xs = Tensor::randn(0f32, 1.0, (1, TOKENS, HIDDEN), &Device::Cpu).unwrap();
        let ids = Tensor::new(&[[0u32, 2], [1, 3], [3, 0]], &Device::Cpu).unwrap();
        let weights = Tensor::new(&[[0.7f32, 0.3], [0.5, 0.5], [0.9, 0.1]], &Device::Cpu).unwrap();
        let ys = cpu_experts
            .forward(&xs, weights.clone(), &ids)
            .unwrap()
            .reshape((TOKENS, HIDDEN))
            .unwrap();

        let ids = ids.to_vec2::<u32>().unwrap();
        let weights = weights.to_vec2::<f32>().unwrap();
        for t in 0..TOKENS {
            let x = xs.i((0, t)).unwrap().unsqueeze(1).unwrap(); // [hidden, 1]
            let mut expected = Tensor::zeros((HIDDEN, 1), DType::F32, &Device::Cpu).unwrap();
            for k in 0..TOP_K {
                let e = ids[t][k] as usize;
                let g = block.gate[e].matmul(&x).unwrap();
                let u = block.up[e].matmul(&x).unwrap();
                let silu = (&g / (g.neg().unwrap().exp().unwrap() + 1.0).unwrap()).unwrap();
                let h = (silu * u).unwrap();
                let y = block.down[e].matmul(&h).unwrap();
                expected = (expected + (y * weights[t][k] as f64).unwrap()).unwrap();
            }
            let expected = expected.squeeze(1).unwrap().to_vec1::<f32>().unwrap();
            let actual = ys.i(t).unwrap().to_vec1::<f32>().unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "token {t}: {actual:?} vs {expected:?}"
                );
            }
        }
    }
}
