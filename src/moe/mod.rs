//! MoE execution: the GPU expert cache, CPU-side routed experts, and providers.
//!
//! # Lock Ordering & Synchronization Invariants
//!
//! 1. **Lock hierarchy (strict DAG):**
//!    `BorrowGate::access` → `Layers::lent` → `CachedLayer::state` → `Slot::storage`.
//!    - `StagingRing` is independent and must never be acquired while holding `CachedLayer::state`.
//!    - `copy_stream::wait_for_readers` must **never** be called while holding `CachedLayer::state`
//!      (a background fill holding an `Arc<SlotBuf>` needs `state` to finish).
//!    - Token-thread callers (`promote_staged`) must use `BorrowGate::try_lock_access()` and stand
//!      down if contended, never blocking on a lookahead fill spinning in `wait_for_readers`.
//! 2. **Stream ordering:**
//!    - Preload completion (`PassCell::wait_latest`) is waited before resident compute.
//!    - Resident experts run before `wait_staging`; staged experts run after `wait_staging`.
//!    - Read fences (`ReadFence::record_reads`) are recorded after kernels reading a slot are
//!      enqueued and before any `Arc<SlotBuf>` is dropped.

mod activation_channel;
mod cache_policy;
mod cache_updates;
mod cached_layer;
mod decode;
mod expert;
mod layer_registry;
mod profile;
mod provider;
mod resident;
mod rings;
pub(crate) mod rings_exec;
pub(crate) mod slab;
mod slot;
mod stages;
mod staging;
#[cfg(test)]
mod tests;

pub use profile::Profile;
pub use provider::GpuCacheMoeProvider;
pub(crate) use stages::{fused_decode, quiesce};

// Cross-module items are re-exported here; siblings import them from `super`.
pub(crate) use activation_channel::{CpuActivationChannel, LookaheadChannel};
#[cfg(test)]
pub(in crate::moe) use cache_policy::{fill_candidate, lru_victim, top_k_ids, union_by_votes};
pub(crate) use cache_policy::streamed_misses;
pub(crate) use cached_layer::{CachedExperts, CachedLayer, LayerState, MAX_DECODE_ROWS};
pub(crate) use expert::{expert_qtensor_on_device, expert_range};
pub(crate) use layer_registry::Layers;
#[cfg(test)]
pub(in crate::moe) use slot::slot_bytes_from_qtensor;
pub(crate) use slot::{slot_bytes_from_stack, Bank, Projections, Slot, SlotBuf, SlotData};
#[cfg(test)]
pub(in crate::moe) use staging::staging_ring_slots;
pub(crate) use staging::{StagingRings, STAGING_RING_FIRST};
