//! `--cpu-moe`: routed experts live in host memory and run on the CPU, while everything else
//! (attention, DeltaNet, routers, shared experts, the MTP head, the KV cache) stays on the GPU.
//!
//! With `--expert-cache-gb <N>`, the most-used experts are additionally cached in BF16 on the
//! GPU. During decode (1 token), GPU cache hits and CPU misses run concurrently; prefill routes
//! everything through the CPU so every expert gets hit anyway and the cache is primed.
//!
//! Expert selection uses a simple count-based promotion: each call records which experts were
//! used; the top-N by count fill the GPU slots. Eviction is LRU within those slots.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use mistralrs_core::moe_offload::{
    install_routed_experts_provider, ExpertLoadContext, RoutedExperts, RoutedExpertsProvider,
};
use mistralrs_core::MoEExperts;
use tracing::{debug, info};

/// Device-layer map that keeps every decoder layer on the main device; mistral.rs clamps the
/// count to the model's layer count. Taken-over experts are not counted, so auto mapping would
/// spill whole layers to the CPU.
pub const ALL_LAYERS_ON_DEVICE: usize = usize::MAX;

pub fn install_cpu_moe(expert_cache_gb: Option<f64>) -> Result<()> {
    install_routed_experts_provider(Arc::new(CpuMoeProvider { expert_cache_gb }))?;
    if let Some(gb) = expert_cache_gb {
        info!(
            "--cpu-moe + expert cache: routed experts in host memory; {:.1} GB GPU cache",
            gb
        );
    } else {
        info!("--cpu-moe: routed experts load into host memory and run on the CPU");
    }
    Ok(())
}

struct CpuMoeProvider {
    expert_cache_gb: Option<f64>,
}

use std::sync::Arc;

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
        let cpu_experts = ctx.load_builtin(Device::Cpu)?;

        if let Some(cache_gb) = self.expert_cache_gb {
            let cache = ExpertCache::new(
                cpu_experts,
                ctx.config.num_experts,
                ctx.config.hidden_size,
                ctx.config.moe_intermediate_size,
                ctx.layer_device.clone(),
                cache_gb,
            )?;
            Ok(Some(Arc::new(cache)))
        } else {
            Ok(Some(Arc::new(CpuExperts {
                experts: cpu_experts,
                layer_device: ctx.layer_device.clone(),
            })))
        }
    }
}

// ---------------------------------------------------------------------------
// Plain CPU-only experts (no cache)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Per-expert BF16 slot on the GPU
// ---------------------------------------------------------------------------

/// BF16 weights for one expert: gate_proj `[inter, hidden]`, up_proj `[inter, hidden]`,
/// down_proj `[hidden, inter]`, all resident on the GPU.
struct GpuExpert {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

impl GpuExpert {
    /// Run one expert for `x: [hidden]` and return `[hidden]`.
    fn forward(&self, x: &Tensor, act: &candle_core::Activation) -> candle_core::Result<Tensor> {
        let g = self.gate.forward(x)?;
        let u = self.up.forward(x)?;
        let h = (act.forward(&g)? * u)?;
        self.down.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// LRU expert cache backed by GPU BF16 slots
// ---------------------------------------------------------------------------

struct CacheState {
    /// `expert_id -> (slot_index, last_use_tick)`.
    slots: HashMap<usize, (usize, u64)>,
    /// `slot_index -> expert_id` for reverse lookup during eviction.
    resident: Vec<Option<usize>>,
    /// Monotonically increasing access counter.
    tick: u64,
    /// Use-count since last cache reset — drives promotion.
    hit_counts: Vec<u64>,
}

/// BF16 GPU expert cache with LRU eviction.
///
/// Each slot holds gate/up/down projections dequantized from the CPU `MoEExperts`. At decode
/// time (1 token) GPU hits and CPU misses run concurrently in a `std::thread::scope`.
struct ExpertCache {
    /// CPU-resident quantized experts — ground truth for every expert.
    cpu: MoEExperts,
    /// GPU slots: `Vec<[inter, hidden]>` for gate/up, `[hidden, inter]` for down.
    gpu_gate: Vec<Tensor>,
    gpu_up: Vec<Tensor>,
    gpu_down: Vec<Tensor>,
    /// Stacked gate/up/down projections from the CPU model, dequantized once at construction.
    /// Shape `[num_experts, inter, hidden]` for gate/up and `[num_experts, hidden, inter]` for down.
    all_gate: Tensor,
    all_up: Tensor,
    all_down: Tensor,
    state: Mutex<CacheState>,
    num_experts: usize,
    hidden: usize,
    inter: usize,
    gpu: Device,
    activation: candle_core::Activation,
}

const PREFILL_THRESHOLD: usize = 1; // token counts above this skip the cache

impl ExpertCache {
    fn new(
        cpu: MoEExperts,
        num_experts: usize,
        hidden: usize,
        inter: usize,
        gpu: Device,
        cache_gb: f64,
    ) -> candle_core::Result<Self> {
        // Dequantize the stacked weights once. Shape: [E, inter, hidden] for gate/up,
        // [E, hidden, inter] for down.
        let (all_gate, all_up, all_down) = cpu.dequantize_experts()?;

        let slot_bytes = (inter * hidden * 2 + inter * hidden * 2 + hidden * inter * 2) as f64;
        let num_slots = ((cache_gb * 1024.0 * 1024.0 * 1024.0) / slot_bytes) as usize;
        let num_slots = num_slots.min(num_experts);

        info!(
            "Expert cache: {num_slots} GPU slots out of {num_experts} total ({:.1}%)",
            100.0 * num_slots as f64 / num_experts as f64
        );

        // Pre-allocate empty GPU slot tensors; we'll fill them on first use.
        // Using `zeros` so they're valid tensors even if never populated.
        let mut gpu_gate = Vec::with_capacity(num_slots);
        let mut gpu_up = Vec::with_capacity(num_slots);
        let mut gpu_down = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            gpu_gate.push(all_gate.i(0)?.to_device(&gpu)?.to_dtype(DType::BF16)?);
            gpu_up.push(all_up.i(0)?.to_device(&gpu)?.to_dtype(DType::BF16)?);
            gpu_down.push(all_down.i(0)?.to_device(&gpu)?.to_dtype(DType::BF16)?);
            let _ = i;
        }

        Ok(Self {
            cpu,
            gpu_gate,
            gpu_up,
            gpu_down,
            all_gate,
            all_up,
            all_down,
            state: Mutex::new(CacheState {
                slots: HashMap::new(),
                resident: vec![None; num_slots],
                tick: 0,
                hit_counts: vec![0u64; num_experts],
            }),
            num_experts,
            hidden,
            inter,
            gpu,
            activation: candle_core::Activation::Silu,
        })
    }

    fn num_slots(&self) -> usize {
        self.gpu_gate.len()
    }

    /// Record usage counts and promote the most-used experts that aren't cached yet.
    fn record_and_maybe_promote(&self, ids: &[u32]) -> candle_core::Result<()> {
        if self.num_slots() == 0 {
            return Ok(());
        }
        let mut state = self.state.lock().unwrap();
        state.tick += 1;
        for &id in ids {
            state.hit_counts[id as usize] += 1;
        }
        // Try to fill empty slots with the globally most-used uncached expert.
        let empty_slot = state.resident.iter().position(|s| s.is_none());
        if let Some(slot) = empty_slot {
            // Find the highest-count uncached expert.
            let best = (0..self.num_experts)
                .filter(|e| !state.slots.contains_key(e))
                .max_by_key(|e| state.hit_counts[*e]);
            if let Some(expert_id) = best {
                if state.hit_counts[expert_id] > 0 {
                    drop(state); // release lock before I/O
                    self.load_expert_into_slot(expert_id, slot)?;
                    let mut state = self.state.lock().unwrap();
                    state.slots.insert(expert_id, (slot, state.tick));
                    state.resident[slot] = Some(expert_id);
                }
            }
        }
        Ok(())
    }

    /// Copy expert `expert_id`'s BF16 weights from the dequantized CPU tensors into GPU slot.
    fn load_expert_into_slot(&self, expert_id: usize, slot: usize) -> candle_core::Result<()> {
        let g = self
            .all_gate
            .i(expert_id)?
            .to_dtype(DType::BF16)?
            .to_device(&self.gpu)?;
        let u = self
            .all_up
            .i(expert_id)?
            .to_dtype(DType::BF16)?
            .to_device(&self.gpu)?;
        let d = self
            .all_down
            .i(expert_id)?
            .to_dtype(DType::BF16)?
            .to_device(&self.gpu)?;
        // SAFETY: slot index was checked by the caller; no other thread writes the same slot.
        unsafe {
            let g_ptr = &self.gpu_gate[slot] as *const Tensor as *mut Tensor;
            let u_ptr = &self.gpu_up[slot] as *const Tensor as *mut Tensor;
            let d_ptr = &self.gpu_down[slot] as *const Tensor as *mut Tensor;
            *g_ptr = g;
            *u_ptr = u;
            *d_ptr = d;
        }
        Ok(())
    }

    /// Evict the LRU slot and load `expert_id` in its place.
    fn evict_and_load(&self, expert_id: usize) -> candle_core::Result<usize> {
        let slot = {
            let state = self.state.lock().unwrap();
            // Find the slot with the oldest last-use tick.
            state
                .slots
                .values()
                .min_by_key(|(_, tick)| *tick)
                .map(|(slot, _)| *slot)
                .unwrap_or(0)
        };
        self.load_expert_into_slot(expert_id, slot)?;
        let mut state = self.state.lock().unwrap();
        if let Some(evicted) = state.resident[slot] {
            state.slots.remove(&evicted);
        }
        state.slots.insert(expert_id, (slot, state.tick));
        state.resident[slot] = Some(expert_id);
        Ok(slot)
    }

    /// Run one token through a GPU-cached expert. `x: [hidden]` → `[hidden]`.
    fn gpu_expert_forward(&self, slot: usize, x: &Tensor) -> candle_core::Result<Tensor> {
        let g = self.gpu_gate[slot].forward(x)?;
        let u = self.gpu_up[slot].forward(x)?;
        let h = (self.activation.forward(&g)? * u)?;
        self.gpu_down[slot].forward(&h)
    }

    /// Forward for a single token (decode). Splits top-k into GPU hits and CPU misses,
    /// runs them concurrently in a thread scope, then sums.
    fn decode_forward(
        &self,
        x: &Tensor,
        topk_weights: &[f32],
        topk_ids: &[u32],
    ) -> candle_core::Result<Tensor> {
        debug_assert_eq!(topk_weights.len(), topk_ids.len());

        let mut gpu_pairs: Vec<(usize, f32, usize)> = Vec::new(); // (expert_id, weight, slot)
        let mut cpu_ids: Vec<(usize, f32)> = Vec::new();

        {
            let mut state = self.state.lock().unwrap();
            state.tick += 1;
            let tick = state.tick;
            for (&id, &w) in topk_ids.iter().zip(topk_weights) {
                state.hit_counts[id as usize] += 1;
                if let Some((slot, last_use)) = state.slots.get_mut(&(id as usize)) {
                    *last_use = tick;
                    gpu_pairs.push((id as usize, w, *slot));
                } else {
                    cpu_ids.push((id as usize, w));
                }
            }
        }

        // Opportunistically load one CPU miss into an empty slot (non-blocking).
        if let Some(&(miss_id, _)) = cpu_ids.first() {
            let empty = {
                let state = self.state.lock().unwrap();
                state.resident.iter().position(|s| s.is_none())
            };
            if let Some(slot) = empty {
                // Best-effort: ignore errors here; the miss will still run on the CPU.
                let _ = self.load_expert_into_slot(miss_id, slot);
                let mut state = self.state.lock().unwrap();
                state.slots.insert(miss_id, (slot, state.tick));
                state.resident[slot] = Some(miss_id);
                // Move from cpu_ids to gpu_pairs.
                if let Some(pos) = cpu_ids.iter().position(|(id, _)| *id == miss_id) {
                    let (id, w) = cpu_ids.remove(pos);
                    gpu_pairs.push((id, w, slot));
                }
            }
        }

        let xs_gpu = x.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, hidden]

        // Run GPU hits and CPU misses concurrently.
        let (gpu_result, cpu_result) = std::thread::scope(|s| {
            let gpu_handle = s.spawn(|| -> candle_core::Result<Tensor> {
                if gpu_pairs.is_empty() {
                    return Tensor::zeros(self.hidden, DType::BF16, &self.gpu);
                }
                let mut acc = Tensor::zeros(self.hidden, DType::BF16, &self.gpu)?;
                let x_bf16 = x.to_dtype(DType::BF16)?;
                for (_, weight, slot) in &gpu_pairs {
                    let y = self.gpu_expert_forward(*slot, &x_bf16)?;
                    acc = (acc + (y * *weight as f64)?)?;
                }
                Ok(acc)
            });

            let cpu_result: candle_core::Result<Option<Tensor>> = if cpu_ids.is_empty() {
                Ok(None)
            } else {
                let cpu_ids_tensor = Tensor::new(
                    cpu_ids.iter().map(|(id, _)| *id as u32).collect::<Vec<_>>(),
                    &Device::Cpu,
                )?
                .unsqueeze(0)?; // [1, n_cpu]
                let cpu_weights_tensor = Tensor::new(
                    cpu_ids.iter().map(|(_, w)| *w).collect::<Vec<_>>(),
                    &Device::Cpu,
                )?
                .unsqueeze(0)?;
                let xs_cpu = xs_gpu.to_device(&Device::Cpu)?;
                let out = self.cpu.forward(&xs_cpu, cpu_weights_tensor, &cpu_ids_tensor)?;
                Ok(Some(out.squeeze(0)?.squeeze(0)?)) // [hidden]
            };

            (gpu_handle.join().unwrap(), cpu_result)
        });

        let gpu_out = gpu_result?;
        let cpu_out = cpu_result?;

        let sum = match cpu_out {
            None => gpu_out.to_dtype(x.dtype())?,
            Some(c) => {
                let c_on_gpu = c.to_device(&self.gpu)?.to_dtype(DType::BF16)?;
                (gpu_out + c_on_gpu)?.to_dtype(x.dtype())?
            }
        };
        Ok(sum)
    }
}

impl RoutedExperts for ExpertCache {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq, hidden) = xs.dims3()?;
        debug_assert_eq!(hidden, self.hidden);
        let num_tokens = batch * seq;

        // Prefill: all tokens, all experts needed. Run on CPU and prime usage counts.
        if num_tokens > PREFILL_THRESHOLD {
            let flat_ids = topk_ids.reshape((num_tokens * topk_ids.dim(2).unwrap_or(1),))?;
            let ids_vec = flat_ids.to_vec1::<u32>()?;
            let out = self.cpu.forward(
                &xs.to_device(&Device::Cpu)?,
                topk_weights.to_device(&Device::Cpu)?,
                &topk_ids.to_device(&Device::Cpu)?,
            )?;
            // Update counts and maybe promote.
            self.record_and_maybe_promote(&ids_vec)?;
            return out.to_device(&self.gpu);
        }

        // Decode: single token.
        let x = xs.i((0, 0))?; // [hidden], on GPU
        let x_cpu = x.to_device(&Device::Cpu)?;

        let weights_vec = topk_weights.i(0)?.to_vec1::<f32>()?;
        let ids_vec = topk_ids.i(0)?.to_vec1::<u32>()?;

        let out = self.decode_forward(&x_cpu, &weights_vec, &ids_vec)?;
        out.unsqueeze(0)?.unsqueeze(0)?.to_device(&self.gpu) // [1, 1, hidden]
    }
}
