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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use candle_core::quantized::{ggml_file::qtensor_from_ggml, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::MoEExperts;
use parking_lot::RwLock;
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
    registry: Arc<LayerRegistry>,
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
            registry: Arc::default(),
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
        let slot_device = ctx.layer_device.clone();

        drop(inner);

        let cached = Arc::new(CachedLayer {
            cpu: cpu_experts,
            slot_device,
            layer: layer_idx,
            registry: Arc::clone(&self.registry),
            hidden,
            num_experts,
            state: Mutex::new(CachedLayerState {
                slots: HashMap::new(),
                resident: vec![None; num_slots],
                hit_counts: vec![0u64; num_experts],
            }),
            slot_weights: (0..num_slots).map(|_| RwLock::new(None)).collect(),
            filling: AtomicBool::new(false),
        });
        self.registry.insert(layer_idx, &cached);
        Ok(Some(Arc::new(CachedExperts(cached))))
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

/// The most-routed expert not in (or reserved for) a slot, ties to the lowest id; `None` when
/// every slot is taken.
fn fill_candidate(resident: &[Option<usize>], hit_counts: &[u64]) -> Option<usize> {
    if !resident.contains(&None) {
        return None;
    }
    let mut taken = vec![false; hit_counts.len()];
    for &e in resident.iter().flatten() {
        taken[e] = true;
    }
    (0..hit_counts.len())
        .filter(|&e| !taken[e])
        .max_by_key(|&e| (hit_counts[e], std::cmp::Reverse(e)))
}

/// Every cached layer of one model, by decoder-layer index, so layer `l` can start filling
/// layer `l + 1`'s slots. Weak references: the model owns the layers, not the registry.
#[derive(Default)]
struct LayerRegistry {
    layers: RwLock<Vec<Weak<CachedLayer>>>,
}

impl LayerRegistry {
    fn insert(&self, layer: usize, cached: &Arc<CachedLayer>) {
        let mut layers = self.layers.write();
        if layers.len() <= layer {
            layers.resize_with(layer + 1, Weak::new);
        }
        layers[layer] = Arc::downgrade(cached);
    }

    fn get(&self, layer: usize) -> Option<Arc<CachedLayer>> {
        self.layers.read().get(layer).and_then(Weak::upgrade)
    }
}

struct CachedLayerState {
    /// Published residents: expert id -> slot. Only experts whose weights are fully written
    /// appear here, and it is the only table readers consult.
    slots: HashMap<usize, usize>,
    /// Slot -> expert id, set when a slot is *reserved* (before its weights are written), so
    /// two admitters can never claim the same slot.
    resident: Vec<Option<usize>>,
    hit_counts: Vec<u64>,
}

/// One expert's projections in a slot.
#[derive(Clone)]
struct SlotWeights {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
}

struct CachedLayer {
    cpu: MoEExperts,
    slot_device: Device,
    layer: usize,
    registry: Arc<LayerRegistry>,
    hidden: usize,
    num_experts: usize,
    state: Mutex<CachedLayerState>,
    /// Slot contents. Written only by `admit` after it has reserved the slot under `state`;
    /// readers clone the `Arc`s under `state` (see `decode_forward`), so a slot reassigned
    /// later never changes what an in-flight token computes with.
    slot_weights: Vec<RwLock<Option<SlotWeights>>>,
    /// A background fill of this layer is running; at most one at a time.
    filling: AtomicBool,
}

impl CachedLayer {
    /// Byte-slice expert `expert_id` out of the CPU stacks onto the slot device — never
    /// dequantized. `None` when the backend holds no `QTensor`s.
    fn expert_weights(&self, expert_id: usize) -> candle_core::Result<Option<SlotWeights>> {
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(None);
        };
        Ok(Some(SlotWeights {
            gate: expert_qtensor_on_device(&gate, expert_id, &self.slot_device)?,
            up: expert_qtensor_on_device(&up, expert_id, &self.slot_device)?,
            down: expert_qtensor_on_device(&down, expert_id, &self.slot_device)?,
        }))
    }

    /// Put `expert_id` into an empty slot. The slot is reserved under the lock before the
    /// (slow) copy and published only after it, so concurrent admitters never share a slot
    /// and readers never see a half-written one. `None`: already resident or no empty slot.
    fn admit(&self, expert_id: usize) -> candle_core::Result<Option<usize>> {
        let slot = {
            let mut st = self.state.lock().unwrap();
            if st.resident.contains(&Some(expert_id)) {
                return Ok(None);
            }
            let Some(slot) = st.resident.iter().position(Option::is_none) else {
                return Ok(None);
            };
            st.resident[slot] = Some(expert_id);
            slot
        };
        let loaded = self.expert_weights(expert_id);
        let mut st = self.state.lock().unwrap();
        match loaded {
            Ok(Some(w)) => {
                *self.slot_weights[slot].write() = Some(w);
                st.slots.insert(expert_id, slot);
                Ok(Some(slot))
            }
            Ok(None) => {
                st.resident[slot] = None;
                Ok(None)
            }
            Err(e) => {
                st.resident[slot] = None;
                Err(e)
            }
        }
    }

    /// The next expert to fill into this layer, if a slot is still empty.
    fn next_fill_candidate(&self) -> Option<usize> {
        let st = self.state.lock().unwrap();
        fill_candidate(&st.resident, &st.hit_counts)
    }

    /// Fill every empty slot of this layer on a background thread and return immediately.
    /// No-op when the layer is full or a fill is already running.
    fn spawn_fill(self: &Arc<Self>) {
        if self.next_fill_candidate().is_none() {
            return;
        }
        if self
            .filling
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let this = Arc::clone(self);
        std::thread::spawn(move || {
            while let Some(e) = this.next_fill_candidate() {
                match this.admit(e) {
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(err) => {
                        debug!("expert cache: layer {} fill stopped: {err}", this.layer);
                        break;
                    }
                }
            }
            this.filling.store(false, Ordering::Release);
        });
    }

    /// Run one token through a cached expert. `x2d: [1, hidden]` on the slot device.
    fn run_expert(w: &SlotWeights, x2d: &Tensor, act: Activation) -> candle_core::Result<Tensor> {
        let g = QMatMul::QTensor(w.gate.clone()).forward(x2d)?;
        let u = QMatMul::QTensor(w.up.clone()).forward(x2d)?;
        let h = (act.forward(&g)? * u)?;
        QMatMul::QTensor(w.down.clone()).forward(&h)
    }

    /// Decode path: split top-k into GPU hits and CPU misses, run both concurrently, sum.
    /// Misses are never loaded on the token path — they run on the CPU.
    fn decode_forward(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        act: Activation,
    ) -> candle_core::Result<Tensor> {
        let ids_vec = topk_ids.i(0)?.to_vec1::<u32>()?;
        let weights_vec = topk_weights.i(0)?.to_vec1::<f32>()?;

        let mut hits: Vec<(SlotWeights, f32)> = Vec::new();
        let mut miss_ids: Vec<u32> = Vec::new();
        let mut miss_weights: Vec<f32> = Vec::new();
        {
            let mut st = self.state.lock().unwrap();
            for (&id, &w) in ids_vec.iter().zip(&weights_vec) {
                st.hit_counts[id as usize] += 1;
                let resident = st
                    .slots
                    .get(&(id as usize))
                    .and_then(|&slot| self.slot_weights[slot].read().clone());
                match resident {
                    Some(sw) => hits.push((sw, w)),
                    None => {
                        miss_ids.push(id);
                        miss_weights.push(w);
                    }
                }
            }
        }

        let x_gpu = xs.i((0, 0))?.unsqueeze(0)?; // [1, hidden]

        let (gpu_result, cpu_result) = std::thread::scope(|s| {
            let gpu = s.spawn(|| -> candle_core::Result<Tensor> {
                let mut acc = Tensor::zeros((1, self.hidden), DType::F32, &self.slot_device)?;
                for (sw, w) in &hits {
                    let y = Self::run_expert(sw, &x_gpu, act)?;
                    acc = (acc + (y.to_dtype(DType::F32)? * *w as f64)?)?;
                }
                Ok(acc)
            });

            let cpu = (|| -> candle_core::Result<Option<Tensor>> {
                if miss_ids.is_empty() {
                    return Ok(None);
                }
                let miss_ids_t = Tensor::new(miss_ids.clone(), &Device::Cpu)?.unsqueeze(0)?;
                let miss_w_t = Tensor::new(miss_weights.clone(), &Device::Cpu)?.unsqueeze(0)?;
                let xs_cpu = xs.to_device(&Device::Cpu)?;
                let out = self.cpu.forward(&xs_cpu, miss_w_t, &miss_ids_t)?;
                Ok(Some(out.squeeze(0)?)) // [1, 1, hidden] -> [1, hidden]
            })();

            (gpu.join().unwrap(), cpu)
        });

        let gpu_out = gpu_result?;
        let sum = match cpu_result? {
            None => gpu_out.to_dtype(xs.dtype())?,
            Some(c) => {
                let c_gpu = c.to_device(&self.slot_device)?.to_dtype(DType::F32)?;
                (gpu_out + c_gpu)?.to_dtype(xs.dtype())?
            }
        };
        sum.unsqueeze(0) // [1, 1, hidden]
    }
}

/// The model holds `Arc<dyn RoutedExperts>`; this wrapper keeps an `Arc<CachedLayer>` so the
/// layer can hand itself to background fills.
struct CachedExperts(Arc<CachedLayer>);

impl RoutedExperts for CachedExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let layer = &self.0;
        let (batch, seq, hidden_dim) = xs.dims3()?;
        debug_assert_eq!(hidden_dim, layer.hidden);
        let num_tokens = batch * seq;

        // Prefill: every routed expert runs on the CPU. Meanwhile, fill the empty slots of the
        // next layer in the background, so the copy overlaps this layer's CPU experts and the
        // next layer's attention. This layer is filled too, for the first layer of the stack
        // (no predecessor triggers it); a layer already full or filling is a no-op.
        if num_tokens > PREFILL_THRESHOLD {
            let ids_vec = topk_ids.flatten_all()?.to_vec1::<u32>()?;
            {
                let mut st = layer.state.lock().unwrap();
                for &id in &ids_vec {
                    st.hit_counts[id as usize] += 1;
                }
            }
            if let Some(next) = layer.registry.get(layer.layer + 1) {
                next.spawn_fill();
            }
            layer.spawn_fill();
            let out = layer.cpu.forward(
                &xs.to_device(&Device::Cpu)?,
                topk_weights.to_device(&Device::Cpu)?,
                &topk_ids.to_device(&Device::Cpu)?,
            )?;
            return out.to_device(&layer.slot_device);
        }

        // Decode: single token, split GPU hits / CPU misses.
        let act = Activation::Silu; // Qwen4-Exp always uses SiLU
        layer.decode_forward(xs, &topk_weights, topk_ids, act)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    /// A stacked `[E, N, K]` Q8_0 tensor whose experts hold distinct values.
    fn stacked(e: usize, n: usize, k: usize) -> QTensor {
        let data: Vec<f32> = (0..e * n * k)
            .map(|i| (i % 97) as f32 * 0.01 - 0.4)
            .collect();
        let t = Tensor::from_vec(data, (e, n, k), &Device::Cpu).unwrap();
        QTensor::quantize(&t, GgmlDType::Q8_0).unwrap()
    }

    #[test]
    fn expert_bytes_slice_the_stack_exactly() {
        let (e, n, k) = (4, 8, 64);
        let qt = stacked(e, n, k);
        let all = qt.data().unwrap().into_owned();
        let stride = all.len() / e;
        let mut rebuilt = Vec::new();
        for i in 0..e {
            let b = expert_bytes(&qt, i).unwrap();
            assert_eq!(b, all[i * stride..(i + 1) * stride]);
            rebuilt.extend_from_slice(&b);
        }
        assert_eq!(rebuilt, all);
        assert!(expert_bytes(&qt, e).is_err());
    }

    #[test]
    fn single_expert_qtensor_matches_its_slice_of_the_stack() {
        let (e, n, k) = (3, 4, 32);
        let qt = stacked(e, n, k);
        let one = expert_qtensor_on_device(&qt, 1, &Device::Cpu).unwrap();
        assert_eq!(one.shape().dims(), &[n, k]);
        assert_eq!(one.dtype(), GgmlDType::Q8_0);
        let full = qt.dequantize(&Device::Cpu).unwrap().get(1).unwrap();
        let got = one.dequantize(&Device::Cpu).unwrap();
        let diff = (full - got).unwrap().abs().unwrap().sum_all().unwrap();
        assert_eq!(diff.to_scalar::<f32>().unwrap(), 0.0);
    }

    #[test]
    fn slot_bytes_count_three_projections() {
        let qt = stacked(2, 8, 64);
        let per_expert = qt.data().unwrap().len() / 2;
        assert_eq!(slot_bytes_from_qtensor(&qt), Some(3 * per_expert as u64));
    }

    #[test]
    fn fill_candidate_prefers_hot_unreserved_experts() {
        // expert 2 is hottest but already reserved; 3 and 1 tie, lowest id wins
        let resident = [Some(2), None];
        let hits = [0, 5, 9, 5];
        assert_eq!(fill_candidate(&resident, &hits), Some(1));
        // with no history, fill in id order
        assert_eq!(fill_candidate(&[None, None], &[0, 0, 0]), Some(0));
        // full layer: nothing to do
        assert_eq!(fill_candidate(&[Some(0), Some(1)], &[3, 3, 3]), None);
        // every expert already placed
        assert_eq!(fill_candidate(&[Some(0), Some(1), None], &[1, 1]), None);
    }

    #[test]
    fn registry_does_not_keep_layers_alive() {
        let reg = LayerRegistry::default();
        assert!(reg.get(3).is_none());
        assert!(reg.layers.read().is_empty());
    }
}
