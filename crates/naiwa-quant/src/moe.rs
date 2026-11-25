//! Single-device MoE expert stacks and `RoutedExperts` offloading interface.
//!
//! Placing these thin types in `naiwa-quant` decouples `naiwa-offload` from `naiwa-models`:
//! neither crate depends on the other; both depend only on `naiwa-quant`.

use std::sync::Arc;

use candle_core::{quantized::QTensor, DType, Device, Result, Tensor, D};

use crate::gguf::cpu::qtensor_indexed_moe_forward;
#[cfg(feature = "cuda")]
use crate::utils::GluActivationType;

/// Minimum token count to switch from per-token fused decode to grouped MMQ/GEMM prefill on CUDA.
#[cfg(feature = "cuda")]
const GROUPED_PREFILL_MIN_TOKENS: usize = 9;

/// Activation function applied to the MoE gate projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Silu,
    Swish,
    Gelu,
    GeluPytorchTanh,
    Relu,
}

impl Activation {
    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Silu | Self::Swish => candle_nn::ops::silu(xs),
            Self::Gelu => xs.gelu_erf(),
            Self::GeluPytorchTanh => xs.gelu(),
            Self::Relu => xs.relu(),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_code(&self) -> Option<i32> {
        match self {
            Self::Silu | Self::Swish => Some(crate::gguf::ACT_SILU),
            Self::GeluPytorchTanh => Some(crate::gguf::ACT_GELU_PYTORCH_TANH),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn glu_activation(&self) -> Option<GluActivationType> {
        match self {
            Self::Silu | Self::Swish => Some(GluActivationType::Silu),
            Self::GeluPytorchTanh => Some(GluActivationType::Gelu),
            Self::Gelu => Some(GluActivationType::GeluErf),
            Self::Relu => Some(GluActivationType::Relu),
        }
    }
}

impl candle_core::Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.apply(xs)
    }
}

/// Configuration for one routed MoE expert block.
#[derive(Clone, Debug)]
pub struct MoEExpertsConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
}

/// Identifies which MoE block is being constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertSite {
    /// Decoder layer index (or the MTP head's assigned layer index).
    pub layer: usize,
    /// Whether the block belongs to the MTP draft head.
    pub is_mtp: bool,
}

/// Routed experts of one MoE block.
pub trait RoutedExperts: Send + Sync {
    /// `xs: [batch, seq, hidden]` on the layer device; `topk_weights` / `topk_ids`:
    /// `[tokens, top_k]` from the router. Returns `[batch, seq, hidden]` (or `[tokens, hidden]`)
    /// matching `xs`'s shape and dtype on the layer device.
    fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor>;
}

/// Loaded 3-projection stacked quantized weights for one MoE layer:
/// - `gate`: `[E, inter, hidden]`
/// - `up`:   `[E, inter, hidden]`
/// - `down`: `[E, hidden, inter]`
pub struct MoEExperts {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
    num_experts: usize,
    num_experts_per_tok: usize,
    act: Activation,
}

impl MoEExperts {
    pub fn from_qtensors(
        gate: Arc<QTensor>,
        up: Arc<QTensor>,
        down: Arc<QTensor>,
        num_experts: usize,
        num_experts_per_tok: usize,
        act: Activation,
    ) -> Self {
        Self {
            gate,
            up,
            down,
            num_experts,
            num_experts_per_tok,
            act,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }

    pub fn activation(&self) -> Activation {
        self.act
    }

    /// Returns `(gate, up, down)` stacked `QTensor` handles.
    pub fn expert_qtensors(&self) -> Option<(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>)> {
        Some((
            Arc::clone(&self.gate),
            Arc::clone(&self.up),
            Arc::clone(&self.down),
        ))
    }

    /// Standard routed MoE forward over stacked quantized experts (GPU CUDA or CPU).
    pub fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let orig_dtype = xs.dtype();
        let hidden = *orig_shape
            .last()
            .ok_or_else(|| candle_core::Error::Msg("MoEExperts::forward empty input".into()))?;
        let tokens = xs.elem_count() / hidden;
        let x2d = xs.reshape((tokens, hidden))?;
        let topk = topk_ids.dim(D::Minus1)?;
        let ids2d = topk_ids.reshape((tokens, topk))?;
        let w2d = topk_weights.reshape((tokens, topk))?;

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = x2d.device() {
            if self.gate.device().is_cuda() {
                if tokens < GROUPED_PREFILL_MIN_TOKENS {
                    if let Some(y) =
                        self.forward_cuda_fused_decode(&x2d, &ids2d, &w2d, tokens, topk, dev)?
                    {
                        return y.to_dtype(orig_dtype)?.reshape(orig_shape);
                    }
                } else if let Some(y) =
                    self.forward_cuda_grouped(&x2d, &ids2d, &w2d, tokens, topk, orig_dtype, dev)?
                {
                    return y.to_dtype(orig_dtype)?.reshape(orig_shape);
                }

                // Generic CUDA indexed MoE fallback
                let x3 = x2d.unsqueeze(1)?;
                let gate_out = crate::gguf::qtensor_indexed_moe_forward(&self.gate, &x3, &ids2d)?;
                let up_out = crate::gguf::qtensor_indexed_moe_forward(&self.up, &x3, &ids2d)?;
                let mid = self.act.apply(&gate_out)?.mul(&up_out)?;
                let down_out = crate::gguf::qtensor_indexed_moe_forward(&self.down, &mid, &ids2d)?;
                let w3 = w2d.to_dtype(down_out.dtype())?.unsqueeze(2)?;
                let sum = down_out.broadcast_mul(&w3)?.sum(1)?;
                return sum.to_dtype(orig_dtype)?.reshape(orig_shape);
            }
        }

        // CPU path: try fused_moe_decode for SiLU decode-sized calls, else indexed_moe per projection.
        if x2d.device().is_cpu() && self.gate.device().is_cpu() && tokens <= 16 {
            if matches!(self.act, Activation::Silu | Activation::Swish) {
                let x_vec = x2d
                    .to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let id_vec = ids2d
                    .to_dtype(DType::U32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<u32>()?;
                let w_vec = w2d
                    .to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                if let Some(y_vec) = crate::gguf::fused_moe_decode(
                    &self.gate, &self.up, &self.down, &x_vec, &id_vec, &w_vec, topk,
                )? {
                    let y = Tensor::from_vec(y_vec, (tokens, hidden), x2d.device())?;
                    return y.to_dtype(orig_dtype)?.reshape(orig_shape);
                }
            }
        }

        let x3 = x2d.unsqueeze(1)?;
        let gate_out = qtensor_indexed_moe_forward(&self.gate, &x3, &ids2d)?;
        let up_out = qtensor_indexed_moe_forward(&self.up, &x3, &ids2d)?;
        let mid = self.act.apply(&gate_out)?.mul(&up_out)?;
        let down_out = qtensor_indexed_moe_forward(&self.down, &mid, &ids2d)?;
        let w3 = w2d.to_dtype(DType::F32)?.unsqueeze(2)?;
        let sum = down_out.to_dtype(DType::F32)?.broadcast_mul(&w3)?.sum(1)?;
        sum.to_dtype(orig_dtype)?.reshape(orig_shape)
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda_fused_decode(
        &self,
        x2d: &Tensor,
        ids2d: &Tensor,
        w2d: &Tensor,
        tokens: usize,
        topk: usize,
        dev: &candle_core::CudaDevice,
    ) -> Result<Option<Tensor>> {
        use candle_core::cuda::cudarc::driver::DevicePtr;
        use candle_core::Storage;

        let Some(act_code) = self.act.cuda_code() else {
            return Ok(None);
        };
        let ids_u32 = ids2d.flatten_all()?.to_dtype(DType::U32)?.contiguous()?;
        let w_f32 = w2d.flatten_all()?.to_dtype(DType::F32)?.contiguous()?;
        let (ids_s, ids_l) = ids_u32.storage_and_layout();
        let (w_s, w_l) = w_f32.storage_and_layout();
        let (Storage::Cuda(ic), Storage::Cuda(wc)) = (&*ids_s, &*w_s) else {
            return Ok(None);
        };
        let ids_slice = ic.as_cuda_slice::<u32>()?;
        let ids_owned;
        let ids_ref = if ids_l.start_offset() == 0 {
            ids_slice
        } else {
            let ids_view = ids_slice.slice(ids_l.start_offset()..);
            ids_owned = dev
                .cuda_stream()
                .clone_dtod(&ids_view)
                .map_err(candle_core::Error::wrap)?;
            &ids_owned
        };
        let w_slice = wc.as_cuda_slice::<f32>()?;
        let stream = dev.cuda_stream();
        let (w_ptr, _guard) = w_slice.device_ptr(&stream);
        let w_ptr = (w_ptr as *const f32).wrapping_add(w_l.start_offset());
        let y = unsafe {
            crate::gguf::indexed_moe_fused_decode(
                &self.gate, &self.up, &self.down, x2d, ids_ref, w_ptr, tokens, topk, act_code, dev,
            )?
        };
        Ok(Some(y))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn forward_cuda_grouped(
        &self,
        x2d: &Tensor,
        ids2d: &Tensor,
        w2d: &Tensor,
        tokens: usize,
        topk: usize,
        orig_dtype: DType,
        dev: &candle_core::CudaDevice,
    ) -> Result<Option<Tensor>> {
        use crate::gguf::{cuda, fast_mmq};
        use candle_core::cuda::cudarc::driver::DevicePtr;
        use candle_core::Storage;

        let total_assignments = tokens * topk;
        let num_experts = self.num_experts;

        let ids_u32 = ids2d.flatten_all()?.to_dtype(DType::U32)?.contiguous()?;
        let (ids_s, ids_l) = ids_u32.storage_and_layout();
        let Storage::Cuda(ic) = &*ids_s else {
            return Ok(None);
        };
        let ids_slice = ic.as_cuda_slice::<u32>()?;
        let ids_owned;
        let ids_ref = if ids_l.start_offset() == 0 {
            ids_slice
        } else {
            let ids_view = ids_slice.slice(ids_l.start_offset()..);
            ids_owned = dev
                .cuda_stream()
                .clone_dtod(&ids_view)
                .map_err(candle_core::Error::wrap)?;
            &ids_owned
        };

        let (expert_bounds, sorted_token_ids, sorted_source_ids) =
            cuda::moe_dispatch_build(ids_ref, total_assignments, num_experts, topk, dev)?;

        enum GroupedGateUp {
            Packed(Tensor),
            SortedPair { gate: Tensor, up: Tensor },
        }

        let use_mmq_gate_up =
            self.gate.dtype() == self.up.dtype() && fast_mmq::supports(self.gate.dtype());
        let gate_up = if use_mmq_gate_up {
            GroupedGateUp::Packed(fast_mmq::grouped_pair_packed(
                &self.gate,
                &self.up,
                x2d,
                &sorted_source_ids,
                &sorted_token_ids,
                &expert_bounds,
                total_assignments,
                topk,
                num_experts,
                dev,
            )?)
        } else {
            if !cuda::supports_moe_grouped(self.gate.dtype())
                || !cuda::supports_moe_grouped(self.up.dtype())
            {
                return Ok(None);
            }
            let (input_q8, k, k_padded) = cuda::quantize_input_q8_1(x2d, dev)?;
            let gate = cuda::grouped_moe_gemm_prequantized(
                &self.gate,
                &input_q8,
                k,
                k_padded,
                &expert_bounds,
                &sorted_token_ids,
                None,
                total_assignments,
                topk,
                num_experts,
                1,
                dev,
            )?;
            let up = cuda::grouped_moe_gemm_prequantized(
                &self.up,
                &input_q8,
                k,
                k_padded,
                &expert_bounds,
                &sorted_token_ids,
                None,
                total_assignments,
                topk,
                num_experts,
                1,
                dev,
            )?;
            drop(input_q8);
            GroupedGateUp::SortedPair { gate, up }
        };

        let tw_f32 = w2d.flatten_all()?.to_dtype(DType::F32)?.contiguous()?;
        let (tw_s, tw_l) = tw_f32.storage_and_layout();
        let Storage::Cuda(wc) = &*tw_s else {
            return Ok(None);
        };
        let tw_slice = wc.as_cuda_slice::<f32>()?;
        let stream = dev.cuda_stream();
        let (tw_ptr, _tw_guard) = tw_slice.device_ptr(&stream);
        let tw_ptr = (tw_ptr as *const f32).wrapping_add(tw_l.start_offset());

        let down = if let (true, Some(glu_act)) = (
            fast_mmq::supports(self.down.dtype()),
            self.act.glu_activation(),
        ) {
            let down_assignments = match &gate_up {
                GroupedGateUp::Packed(gu) => fast_mmq::grouped_from_glu_packed(
                    &self.down,
                    gu,
                    &sorted_token_ids,
                    &sorted_token_ids,
                    &expert_bounds,
                    total_assignments,
                    tokens,
                    num_experts,
                    glu_act as i32,
                    dev,
                )?,
                GroupedGateUp::SortedPair { gate, up } => fast_mmq::grouped_from_glu_sorted_pair(
                    &self.down,
                    gate,
                    up,
                    &sorted_token_ids,
                    &expert_bounds,
                    total_assignments,
                    tokens,
                    num_experts,
                    glu_act as i32,
                    dev,
                )?,
            };
            if orig_dtype == DType::BF16 {
                unsafe {
                    cuda::moe_weighted_reduce_flat_bf16(
                        &down_assignments,
                        tw_ptr,
                        tokens,
                        topk,
                        dev,
                    )?
                }
            } else {
                unsafe {
                    cuda::moe_weighted_reduce_flat(&down_assignments, tw_ptr, tokens, topk, dev)?
                }
            }
        } else {
            if !cuda::supports_moe_grouped(self.down.dtype()) {
                return Ok(None);
            }
            let (activated, down_input_dim1) = match &gate_up {
                GroupedGateUp::Packed(gu) => {
                    let inter = gu.dim(D::Minus1)? / 2;
                    let gate = gu.narrow(D::Minus1, 0, inter)?;
                    let up = gu.narrow(D::Minus1, inter, inter)?;
                    (self.act.apply(&gate)?.mul(&up)?, 2)
                }
                GroupedGateUp::SortedPair { gate, up } => (self.act.apply(gate)?.mul(up)?, 0),
            };
            let (down_q8, down_k, down_k_padded) = cuda::quantize_input_q8_1(&activated, dev)?;
            cuda::grouped_moe_gemm_prequantized(
                &self.down,
                &down_q8,
                down_k,
                down_k_padded,
                &expert_bounds,
                &sorted_token_ids,
                Some((tw_ptr, 0)),
                total_assignments,
                topk,
                num_experts,
                down_input_dim1,
                dev,
            )?
        };
        Ok(Some(down))
    }
}

impl RoutedExperts for MoEExperts {
    fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor> {
        MoEExperts::forward(self, xs, topk_weights, topk_ids)
    }
}

/// Callback / loader context passed to [`RoutedExpertsProvider`] when a model loads an MoE block.
pub struct ExpertLoadRequest<'a> {
    pub site: ExpertSite,
    pub config: &'a MoEExpertsConfig,
    pub layer_device: &'a Device,
    pub act: Activation,
    /// Optional router weight `[num_experts, hidden]` on CPU/device (for lookahead prediction).
    pub router_weight: Option<Tensor>,
    /// Closure that materializes the `(gate, up, down)` stacked `QTensor`s on a requested `Device`.
    pub load_stacks: &'a dyn Fn(&Device) -> Result<(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>)>,
}

/// Supplies routed experts for MoE blocks at model load time (explicit dependency injection).
pub trait RoutedExpertsProvider: Send + Sync {
    /// Return `Some` to take over this block's routed experts, or `None` to load builtin on `layer_device`.
    fn build(&self, req: ExpertLoadRequest<'_>) -> Result<Option<Arc<dyn RoutedExperts>>>;
}
