//! MoE execution: the GPU expert cache, CPU-side routed experts, and both providers.
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
//! # Prefill
//!
//! Every routed expert runs on the CPU (all of them are needed anyway). Each layer first starts
//! a background fill of the *next* layer's empty slots. On CUDA a fill is a copy on a dedicated
//! copy stream into slot memory allocated once at load, so it overlaps this layer's CPU experts
//! and the next layer's attention (`copy_stream`). Elsewhere it uploads through candle's stream.
//!
//! # Decode (single token)
//!
//! - Top-k is split into slot hits and misses. Hits run via `QMatMul` on the slot device, misses
//!   on the CPU with only the miss ids, through the fused two-pass decode (`stages::cpu_miss::fused_decode`)
//!   or, for experts it does not serve, `MoEExperts::forward`; both sides at once, router weights
//!   applied once per side. The misses run on the `deferred` worker over host copies; every CUDA
//!   call stays on the calling thread.
//! - Misses are never loaded on the token path. Instead, each layer submits its activation to a
//!   lookahead thread that runs the next layer's router on it and loads the predicted experts:
//!   into empty slots freely, over least-recently-used residents at most `--lookahead-swaps`
//!   times per row. Whatever is not resident when the next layer runs goes to the CPU.
//! - A token captures its hits' weights (`Arc`s) under the layer lock, so a slot replaced later

mod arena;
mod cached_layer;
mod decode;
mod expert;
mod provider;
mod rings;
pub(crate) mod rings_exec;
mod resident;
mod slot;
mod stages;
mod staging;
#[cfg(test)]
mod tests;

pub use provider::GpuCacheMoeProvider;
pub(crate) use arena::SlotArena;
pub(crate) use rings::{plan as plan_rings, RingPlan, Round, RoundSlots};
pub(crate) use rings_exec as prefill_rings;
pub(crate) use stages::fused_decode;

// Cross-module items are re-exported here; siblings import them from `super`, never from
// another child module directly. The submodules themselves stay private.
#[cfg(test)]
pub(in crate::moe) use cached_layer::{fill_candidate, lru_victim, top_k_ids, union_by_votes};
pub(crate) use cached_layer::{
    streamed_misses, CachedExperts, CachedLayer, Layers, LayerState, LookaheadChannel,
    MAX_DECODE_ROWS,
};
pub(crate) use expert::{expert_qtensor_on_device, expert_range};
pub(crate) use slot::{slot_bytes_from_qtensor, Bank, Projections, Slot, SlotBuf, SlotData};
pub(in crate::moe) use staging::alloc_bank;
#[cfg(test)]
pub(in crate::moe) use staging::staging_ring_slots;
pub(crate) use staging::{StagingRings, STAGING_RING_FIRST};
