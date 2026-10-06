//! GPU expert cache for `--cpu-moe --expert-cache-gb N`.
//!
//! Each decoder layer gets its own per-layer slot range from a total VRAM budget. Slots hold the
//! expert's *quantized* bytes (GGML block format) — never a dequantized copy — so both the GPU
//! and CPU paths run with the same quantization and the cache costs the same VRAM as the CPU tier
//! costs in RAM.
//!
//! # Format support
//!
//! - **GGUF** (Q4_K, Q6_K, …): `get_qtensor()` returns the stacked `[E, N, K]` `QTensor`; bytes
//!   are sliced per expert and re-assembled as a `[N, K]` `QTensor` on the slot device.
//! - **ISQ-quantized** (`--isq`): same path, because ISQ leaves a `QTensor` behind.
//! - **BF16 safetensors**: `get_qtensor()` returns `None`; the cache falls back to a dense
//!   `Tensor` copy instead. Dense slots cost ~4× more VRAM per expert than Q4K.
//!
//! # Hit/miss dispatch (decode, single token)
//!
//! Router assigns each token up to `top_k` experts. For each decode step:
//! - Experts whose slot is resident → computed via `QMatMul` (or dense `matmul`) on the slot device.
//! - Remaining experts → fall through to `MoEExperts::forward` on the CPU.
//!
//! The CPU path calls `MoEExperts::forward` with only the *miss* indices (using the fork's
//! `forward_gather` fix that derives `top_k` from `topk_ids.dim(-1)` rather than the fixed config
//! value). The per-expert results are summed with their router weights applied **once** at the end.
//!
//! Prefill (more than one token): routes everything through the CPU and primes admission counts.
//!
//! # Admission
//!
//! Per-layer slots, filled in arrival order (first-seen / compulsory-miss policy). No eviction in
//! this stage — a fixed admission policy that also evicts has two effects moving at once and the
//! hit rate cannot be attributed to either.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::quantized::{ggml_file::qtensor_from_ggml, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::MoEExperts;
use tracing::{debug, info};

// Number of prefill tokens above which we skip the cache entirely.
// A decode step is always 1 token, so PREFILL_THRESHOLD = 1 is exact.
const PREFILL_THRESHOLD: usize = 1;

// ---------------------------------------------------------------------------
// Provider state
// ---------------------------------------------------------------------------

struct ProviderInner {
    /// Total budget remaining in bytes.
    remaining_bytes: u64,
    /// Per-layer slot quota (bytes), determined on first build() call.
    bytes_per_layer: Option<u64>,
}

/// GPU expert cache provider.
///
/// Installed once before the model loads; `build()` is called once per MoE block during
/// weight loading. Decoder layers fill their quota from the total budget; the MTP head is
/// passed through to the CPU-only path.
pub struct GpuCacheMoeProvider {
    cache_gb: f64,
    inner: Mutex<ProviderInner>,
}

impl GpuCacheMoeProvider {
    pub fn new(cache_gb: f64) -> Self {
        let total_bytes = (cache_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        Self {
            cache_gb,
            inner: Mutex::new(ProviderInner {
                remaining_bytes: total_bytes,
                bytes_per_layer: None,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Byte-level expert extraction (never dequantize)
// ---------------------------------------------------------------------------

/// Bytes for expert `e` from a stacked `[E, N, K]` `QTensor`.
///
/// Row-block formats never cross expert boundaries, so slicing
/// `[e·stride, (e+1)·stride)` from the host buffer is always exact.
fn expert_bytes(qt: &QTensor, e: usize) -> candle_core::Result<Vec<u8>> {
    let dims = qt.shape().dims();
    // Expect [E, N, K] or [N, K]
    if dims.len() < 2 {
        candle_core::bail!("expert QTensor must be at least 2-D, got {:?}", dims);
    }
    let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let block_size = qt.dtype().block_size();
    if k % block_size != 0 {
        candle_core::bail!(
            "expert K={k} is not divisible by block_size={block_size} for dtype {:?}",
            qt.dtype()
        );
    }
    let type_size = qt.dtype().type_size();
    let stride = n * (k / block_size) * type_size;
    let all_bytes = qt.data()?;
    let start = e * stride;
    let end = start + stride;
    if end > all_bytes.len() {
        candle_core::bail!(
            "expert {e} byte range [{start}..{end}) exceeds buffer length {}",
            all_bytes.len()
        );
    }
    Ok(all_bytes[start..end].to_vec())
}

/// Build a single-expert `[N, K]` `QTensor` on `device` from a stacked `[E, N, K]` `QTensor`.
fn expert_qtensor_on_device(
    stack: &QTensor,
    e: usize,
    device: &Device,
) -> candle_core::Result<Arc<QTensor>> {
    let dims = stack.shape().dims();
    let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let bytes = expert_bytes(stack, e)?;
    let qt = qtensor_from_ggml(stack.dtype(), &bytes, vec![n, k], device)?;
    Ok(Arc::new(qt))
}

/// Slot-bytes estimate for a single expert from a `QTensor` stack.
/// Returns `None` when the stack is not accessible (dense/float).
fn slot_bytes_from_qtensor(qt: &QTensor) -> Option<u64> {
    let dims = qt.shape().dims();
    if dims.len() < 2 {
        return None;
    }
    let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let block_size = qt.dtype().block_size();
    if k % block_size != 0 {
        return None;
    }
    let type_size = qt.dtype().type_size();
    // gate + up + down (gate/up are [inter, hidden]; down is [hidden, inter] — same bytes)
    Some((3 * n * (k / block_size) * type_size) as u64)
}

// ---------------------------------------------------------------------------
// RoutedExpertsProvider impl
// ---------------------------------------------------------------------------

use mistralrs_core::moe_offload::{ExpertLoadContext, RoutedExperts, RoutedExpertsProvider};

impl RoutedExpertsProvider for GpuCacheMoeProvider {
    fn build(
        &self,
        ctx: ExpertLoadContext<'_>,
    ) -> candle_core::Result<Option<Arc<dyn RoutedExperts>>> {
        // Never cache MTP-head experts or CPU-device layers.
        if ctx.site.is_mtp || ctx.layer_device.is_cpu() {
            return Ok(None);
        }

        let cpu_experts = ctx.load_builtin(Device::Cpu)?;
        let num_experts = ctx.config.num_experts;

        // Try to get the quantized stacked weight for the gate projection — this tells us
        // the dtype and per-expert slot bytes, and whether we're in quantized or dense mode.
        let (slot_bytes, is_quantized) = {
            let qt_opt = cpu_experts.expert_qtensors();
            match qt_opt
                .as_ref()
                .and_then(|(gate, _, _)| slot_bytes_from_qtensor(gate))
            {
                Some(sb) => (sb, true),
                None => {
                    // Dense fallback: estimate from hidden/inter in BF16
                    let sb = 3u64
                        * ctx.config.hidden_size as u64
                        * ctx.config.moe_intermediate_size as u64
                        * 2; // BF16 = 2 bytes
                    (sb, false)
                }
            }
        };

        let mut inner = self.inner.lock().unwrap();

        // On first call, compute per-layer quota.  We assume all layers use the same weight
        // shape (true for Qwen3.8-Flash-Next).  Since we don't know n_layers up front, we
        // set bytes_per_layer lazily and refine downward as more layers arrive.
        if inner.bytes_per_layer.is_none() {
            // Estimate: 48 decoder layers is the Qwen3.8-Flash-Next default; the real value is
            // refined as layers arrive.  The quota can only stay the same or shrink, never grow,
            // because earlier layers have already claimed their slots.
            let estimated_layers = 48u64;
            let per_layer = inner.remaining_bytes / estimated_layers;
            inner.bytes_per_layer = Some(per_layer);
            info!(
                "expert cache: {:.1} GB total, ~{} slots/layer (estimated {} layers, {} bytes/slot)",
                self.cache_gb,
                per_layer / slot_bytes.max(1),
                estimated_layers,
                slot_bytes,
            );
        }

        let per_layer_budget = inner.bytes_per_layer.unwrap();
        let num_slots = ((per_layer_budget / slot_bytes.max(1)) as usize).min(num_experts);

        if num_slots == 0 {
            debug!(
                "expert cache: layer {} gets 0 slots ({} bytes/slot > {}/layer budget); skipping",
                ctx.site.layer, slot_bytes, per_layer_budget
            );
            // Still install a CPU-only provider so the layer is consistent.
            let layer_device = ctx.layer_device.clone();
            return Ok(Some(Arc::new(CpuOnlyExperts {
                cpu: cpu_experts,
                layer_device,
            })));
        }

        let layer_idx = ctx.site.layer;
        debug!(
            "expert cache: layer {} → {} GPU slots / {} experts ({}, {} B/slot)",
            layer_idx,
            num_slots,
            num_experts,
            if is_quantized { "quantized" } else { "dense" },
            slot_bytes,
        );

        // Charge the budget.
        let charged = slot_bytes * num_slots as u64;
        inner.remaining_bytes = inner.remaining_bytes.saturating_sub(charged);

        let hidden = ctx.config.hidden_size;
        let inter = ctx.config.moe_intermediate_size;
        let slot_device = ctx.layer_device.clone();

        drop(inner);

        Ok(Some(Arc::new(CachedLayer {
            cpu: cpu_experts,
            slot_device,
            hidden,
            inter,
            num_experts,
            state: Mutex::new(CachedLayerState {
                slots: HashMap::new(),
                resident: vec![None; num_slots],
                hit_counts: vec![0u64; num_experts],
            }),
            slot_gate: (0..num_slots).map(|_| None).collect(),
            slot_up: (0..num_slots).map(|_| None).collect(),
            slot_down: (0..num_slots).map(|_| None).collect(),
        })))
    }
}

// ---------------------------------------------------------------------------
// CPU-only fallback (zero slots)
// ---------------------------------------------------------------------------

struct CpuOnlyExperts {
    cpu: MoEExperts,
    layer_device: Device,
}

impl RoutedExperts for CpuOnlyExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let ys = self.cpu.forward(
            &xs.to_device(&Device::Cpu)?,
            topk_weights.to_device(&Device::Cpu)?,
            &topk_ids.to_device(&Device::Cpu)?,
        )?;
        ys.to_device(&self.layer_device)
    }
}

// ---------------------------------------------------------------------------
// CachedLayer — the real workhorse
// ---------------------------------------------------------------------------

struct CachedLayerState {
    slots: HashMap<usize, usize>,
    resident: Vec<Option<usize>>,
    hit_counts: Vec<u64>,
}

struct CachedLayer {
    cpu: MoEExperts,
    slot_device: Device,
    hidden: usize,
    #[allow(dead_code)] // kept for VRAM accounting / future dense-fallback path
    inter: usize,
    num_experts: usize,
    state: Mutex<CachedLayerState>,
    // Parallel Vecs: slot_gate[i] / slot_up[i] / slot_down[i] form a complete slot.
    // None = empty slot.
    slot_gate: Vec<Option<Arc<QTensor>>>,
    slot_up: Vec<Option<Arc<QTensor>>>,
    slot_down: Vec<Option<Arc<QTensor>>>,
}

impl CachedLayer {
    #[allow(dead_code)]
    fn num_slots(&self) -> usize {
        self.slot_gate.len()
    }

    /// Load expert `expert_id` into slot `slot` from the CPU `MoEExperts` quantized stack.
    ///
    /// Uses raw byte slicing — never dequantizes.  Returns `true` if the slot was populated,
    /// `false` if the backend has no `QTensor` (rare; caller should use a dense copy instead).
    fn load_slot(&self, expert_id: usize, slot: usize) -> candle_core::Result<bool> {
        let qt_opt = self.cpu.expert_qtensors();
        let Some((gate_stack, up_stack, down_stack)) = qt_opt.as_ref() else {
            return Ok(false);
        };

        let gate = expert_qtensor_on_device(gate_stack, expert_id, &self.slot_device)?;
        let up = expert_qtensor_on_device(up_stack, expert_id, &self.slot_device)?;
        let down = expert_qtensor_on_device(down_stack, expert_id, &self.slot_device)?;

        // SAFETY: slot indices are bounded by `num_slots()`; each slot is only written here,
        // called from behind the state mutex.  We use raw pointer writes because the Vec is
        // pre-allocated with `None`s and we need interior mutability without wrapping each
        // element in an extra lock.  The subsequent GPU read is on the same thread/stream as
        // the write in this function.
        unsafe {
            let gp = self.slot_gate.as_ptr().add(slot) as *mut Option<Arc<QTensor>>;
            let up_p = self.slot_up.as_ptr().add(slot) as *mut Option<Arc<QTensor>>;
            let dp = self.slot_down.as_ptr().add(slot) as *mut Option<Arc<QTensor>>;
            *gp = Some(gate);
            *up_p = Some(up);
            *dp = Some(down);
        }
        Ok(true)
    }

    /// Run one token through a cached expert at `slot`.
    ///
    /// `x2d: [1, hidden]` on `slot_device` → `[1, hidden]`.
    fn run_slot(&self, slot: usize, x2d: &Tensor, act: Activation) -> candle_core::Result<Tensor> {
        let gate_qt = self.slot_gate[slot]
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("slot gate is empty"))?;
        let up_qt = self.slot_up[slot]
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("slot up is empty"))?;
        let down_qt = self.slot_down[slot]
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("slot down is empty"))?;

        let gate_mm = QMatMul::QTensor(gate_qt.clone());
        let up_mm = QMatMul::QTensor(up_qt.clone());
        let down_mm = QMatMul::QTensor(down_qt.clone());

        // x2d: [1, hidden]; gate/up weights [inter, hidden] → output [1, inter]
        let g = gate_mm.forward(x2d)?;
        let u = up_mm.forward(x2d)?;
        let h = (act.forward(&g)? * u)?;
        down_mm.forward(&h) // [1, hidden]
    }

    /// Decode path: split top-k into GPU hits and CPU misses, run both, sum.
    fn decode_forward(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        act: Activation,
    ) -> candle_core::Result<Tensor> {
        let ids_vec = topk_ids.i(0)?.to_vec1::<u32>()?;
        let weights_vec = topk_weights.i(0)?.to_vec1::<f32>()?;

        let mut hit_ids: Vec<u32> = Vec::new();
        let mut hit_weights: Vec<f32> = Vec::new();
        let mut miss_ids: Vec<u32> = Vec::new();
        let mut miss_weights: Vec<f32> = Vec::new();

        {
            let mut state = self.state.lock().unwrap();
            for (&id, &w) in ids_vec.iter().zip(&weights_vec) {
                state.hit_counts[id as usize] += 1;
                if let Some(&slot) = state.slots.get(&(id as usize)) {
                    hit_ids.push(id);
                    hit_weights.push(w);
                    let _ = slot;
                } else {
                    miss_ids.push(id);
                    miss_weights.push(w);

                    // Opportunistic admission: if there is an empty slot, fill it now.
                    if let Some(empty_slot) = state.resident.iter().position(|s| s.is_none()) {
                        // Release the lock before the (potentially slow) byte copy.
                        drop(state);
                        match self.load_slot(id as usize, empty_slot) {
                            Ok(true) => {
                                let mut st2 = self.state.lock().unwrap();
                                st2.slots.insert(id as usize, empty_slot);
                                st2.resident[empty_slot] = Some(id as usize);
                                // Move from miss to hit.
                                miss_ids.pop();
                                miss_weights.pop();
                                hit_ids.push(id);
                                hit_weights.push(w);
                            }
                            Ok(false) | Err(_) => {}
                        }
                        state = self.state.lock().unwrap();
                    }
                }
            }
        }

        let x_gpu = xs.i((0, 0))?.unsqueeze(0)?; // [1, hidden] on slot_device

        // GPU hits.
        let mut out = Tensor::zeros((1, self.hidden), DType::F32, &self.slot_device)?;
        {
            let state = self.state.lock().unwrap();
            for (&id, &w) in hit_ids.iter().zip(&hit_weights) {
                if let Some(&slot) = state.slots.get(&(id as usize)) {
                    let y = self.run_slot(slot, &x_gpu, act)?;
                    let y_f32 = y.to_dtype(DType::F32)?;
                    out = (out + (y_f32 * w as f64)?)?;
                }
            }
        }

        // CPU misses.
        if !miss_ids.is_empty() {
            let miss_ids_t = Tensor::new(
                miss_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                &Device::Cpu,
            )?
            .unsqueeze(0)?; // [1, n_miss]
            let miss_w_t = Tensor::new(miss_weights.clone(), &Device::Cpu)?.unsqueeze(0)?;
            let xs_cpu = xs.to_device(&Device::Cpu)?;
            let cpu_out = self.cpu.forward(&xs_cpu, miss_w_t, &miss_ids_t)?; // [1, 1, hidden]
            let cpu_f32 = cpu_out
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_device(&self.slot_device)?;
            out = (out + cpu_f32)?;
        }

        out.to_dtype(xs.dtype())?.unsqueeze(0) // [1, 1, hidden]
    }
}

impl RoutedExperts for CachedLayer {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq, hidden_dim) = xs.dims3()?;
        debug_assert_eq!(hidden_dim, self.hidden);
        let num_tokens = batch * seq;

        // Prefill: route everything through CPU and update counts for admission.
        if num_tokens > PREFILL_THRESHOLD {
            let flat_ids = topk_ids.reshape((num_tokens * topk_ids.dim(1)?,))?;
            let ids_vec = flat_ids.to_vec1::<u32>()?;
            {
                let mut state = self.state.lock().unwrap();
                for &id in &ids_vec {
                    state.hit_counts[id as usize] += 1;
                }
                // Fill any empty slot with the most-counted uncached expert.
                while let Some(empty_slot) = state.resident.iter().position(|s| s.is_none()) {
                    let best = (0..self.num_experts)
                        .filter(|e| !state.slots.contains_key(e))
                        .max_by_key(|e| state.hit_counts[*e]);
                    let Some(expert_id) = best else { break };
                    if state.hit_counts[expert_id] == 0 {
                        break;
                    }
                    // Release lock before slow I/O.
                    drop(state);
                    match self.load_slot(expert_id, empty_slot) {
                        Ok(true) => {
                            let mut st = self.state.lock().unwrap();
                            st.slots.insert(expert_id, empty_slot);
                            st.resident[empty_slot] = Some(expert_id);
                            state = st;
                        }
                        _ => {
                            // No slot filled — drop and re-acquire so the loop variable
                            // stays consistent, then break.
                            let _guard = self.state.lock().unwrap();
                            break;
                        }
                    }
                }
            }
            let out = self.cpu.forward(
                &xs.to_device(&Device::Cpu)?,
                topk_weights.to_device(&Device::Cpu)?,
                &topk_ids.to_device(&Device::Cpu)?,
            )?;
            return out.to_device(&self.slot_device);
        }

        // Decode: single token, split GPU hits / CPU misses.
        let act = Activation::Silu; // Qwen4-Exp always uses SiLU
        self.decode_forward(xs, &topk_weights, topk_ids, act)
    }
}
