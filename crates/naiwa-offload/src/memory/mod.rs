//! Layer 2 (`memory`): cross-domain weight & VRAM ownership.
//!
//! Owns host pinned weight stacks, per-layer GPU expert cache directories, VRAM slab allocation,
//! `BorrowGate`, cross-layer staging rings, and Q2_0 resident drafter pools.
//! Strictly depends only on Layer 1 (`transport`) and cross-cutting `config` / `stats`.
//!
//! # Lock Ordering & Synchronization Invariants
//!
//! 1. **Lock hierarchy (strict DAG):**
//!    `BorrowGate::access` → `Layers::lent` → `LayerCache::state` → `Slot::storage`.
//!    - `StagingRing` is independent and must never be acquired while holding `LayerCache::state`.
//!    - `copy_stream::wait_for_readers` must **never** be called while holding `LayerCache::state`
//!      (a background fill holding an `Arc<SlotBuf>` needs `state` to finish).
//!    - Token-thread callers (`promote_staged`) must use `BorrowGate::try_lock_access()` and stand
//!      down if contended, never blocking on a lookahead fill spinning in `wait_for_readers`.
//! 2. **Stream ordering:**
//!    - Preload completion (`PassCell::wait_latest`) is waited before resident compute.
//!    - Resident experts run before `wait_staging`; staged experts run after `wait_staging`.
//!    - Read fences (`ReadFence::record_reads`) are recorded after kernels reading a slot are
//!      enqueued and before any `Arc<SlotBuf>` is dropped.

mod cache_policy;
mod cache_updates;
pub(crate) mod expert;
pub(crate) mod host_weights;
mod layer;
mod layer_registry;
pub(crate) mod profile;
pub(crate) mod resident;
pub(crate) mod slab;
mod slot;
mod staging;
#[cfg(test)]
mod tests;

pub(crate) use cache_policy::streamed_misses;
#[cfg(test)]
pub(in crate::memory) use cache_policy::{fill_candidate, lru_victim, top_k_ids, union_by_votes};
pub(crate) use expert::{expert_qtensor_on_device, expert_range};
pub(crate) use layer::{
    CachedLayer, LayerCache, LayerState, LayerTransport, LayerWeights, MAX_DECODE_ROWS,
};
pub(crate) use layer_registry::Layers;
pub use profile::Profile;
#[cfg(test)]
pub(in crate::memory) use slot::slot_bytes_from_qtensor;
pub(crate) use slot::{slot_bytes_from_stack, Bank, Projections, Slot, SlotBuf, SlotData};
pub(crate) use staging::{
    alloc_bank_in, staging_key, staging_ring_slots, StagingRings, STAGING_RING_FIRST,
};
