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
//!   on the CPU with only the miss ids, through the fused two-pass decode (`cpu_experts::fused_decode`)
//!   or, for experts it does not serve, `MoEExperts::forward`; both sides at once, router weights
//!   applied once per side. The misses run on the `deferred` worker over host copies; every CUDA
//!   call stays on the calling thread.
//! - Misses are never loaded on the token path. Instead, each layer submits its activation to a
//!   lookahead thread that runs the next layer's router on it and loads the predicted experts:
//!   into empty slots freely, over least-recently-used residents at most `--lookahead-swaps`
//!   times per row. Whatever is not resident when the next layer runs goes to the CPU.
//! - A token captures its hits' weights (`Arc`s) under the layer lock, so a slot replaced later

mod cpu_experts;
mod expert_bytes;
mod layer;
mod provider_cpu;
mod provider_gpu;
mod registry;
mod staging;
#[cfg(test)]
mod tests;

pub(crate) use cpu_experts::fused_decode;
pub(crate) use provider_cpu::CpuMoeProvider;
pub use provider_gpu::GpuCacheMoeProvider;

// Cross-module items are re-exported here; siblings import them from `super`, never from
// another child module directly. The submodules themselves stay private.
pub(crate) use expert_bytes::{expert_qtensor_on_device, expert_range, slot_bytes_from_qtensor};
#[cfg(test)]
pub(in crate::moe) use layer::{fill_candidate, lru_victim, top_k_ids, union_by_votes};
pub(crate) use layer::{streamed_misses, LookaheadChannel};
pub(in crate::moe) use staging::alloc_bank;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize};
#[cfg(feature = "cuda")]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex, Weak};

use candle_core::quantized::QTensor;
use candle_core::{Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::RoutedExperts;
use mistralrs_core::MoEExperts;
use parking_lot::RwLock;

use crate::copy_stream::{CopyStream, ReadFence, SlotMemory};
use crate::host_memory::PinnedStack;
use crate::prefill_stream::PrefillStream;

// Number of prefill tokens above which we skip the cache entirely.
// Calls with more tokens than this are prefill chunks: every expert runs on the CPU (or the
// prefill stream). Up to it they take the decode split, which is what a speculative verify step
// needs (the target runs the n drafts plus one token in one forward).
const PREFILL_THRESHOLD: usize = 16;

/// With `--prefill-stream`, each layer keeps its last slots in a separate bank of about its
/// share of the loan: two layers of experts, spread over the estimated 48 layers, plus a
/// quarter for layers with smaller slots than the largest one. The loan lends whole banks.
fn loan_bank_slots(num_experts: usize) -> usize {
    (2 * num_experts * 5).div_ceil(4 * 48)
}

/// With `--pcie-frac`, a decode step DMAs its highest-weight misses into a staging ring shared
/// by every layer of the model (Strata's `kStagingBlobs` shape), instead of a per-layer scratch
/// bank: per-layer banks could only hold `ceil(frac*top_k)` slots each, which at `frac` 0.55
/// meant 6 slots x 48 layers of VRAM and OOM'd, while cross-layer reuse needs one fixed pool of
/// slots that every layer's step borrows in turn. The fence model already serializes reusers —
/// a reader holds the `SlotBuf` `Arc` until it has recorded its fence, and the next layer's DMA
/// waits on that fence — so no cross-layer accounting is needed.
pub(crate) const SCRATCH_RING_FIRST: usize = 1 << 30;

/// Ring capacity: enough that consecutive layers' steps rarely collide on one slot, small
/// enough to be a rounding error next to the cache. Strata uses 16 (`kStagingBlobs`).
/// Fixed: 16 measured the same as 64, and a bigger ring only costs VRAM.
/// Staging-ring slots per quant shape. Measured: 16 and 64 perform identically (the ring has no id -> slot
/// lookup, so a staged expert is copied again either way), so this is a constant, not a knob.
const fn scratch_ring_slots() -> usize {
    16
}

#[cfg(feature = "cuda")]
static SCHEDULE: OnceLock<Arc<crate::doorbell::GateSet>> = OnceLock::new();

#[cfg(feature = "cuda")]
pub(crate) fn schedule() -> Option<&'static Arc<crate::doorbell::GateSet>> {
    SCHEDULE.get()
}

/// Every cached layer of one model, by decoder-layer index, so layer `l` can start filling
/// layer `l + 1`'s slots. Weak references: the model owns the layers, not the registry.
#[derive(Default)]
pub(crate) struct LayerRegistry {
    layers: RwLock<Vec<Weak<CachedLayer>>>,
    /// The model-wide `--pcie-frac` staging rings, keyed by the shape of the expert stacks
    /// (dtype and per-projection dims): mixed-quant models give different layers different
    /// GGML dtypes, and one bank cannot serve two shapes (W105: 524800-byte experts against a
    /// 627200-byte slot). Each shape gets its own ring of `scratch_ring_slots()` slots.
    /// Behind a `Mutex` because `ScratchRing::acquire` mutates each ring's cursor.
    scratch: Mutex<HashMap<ScratchKey, Arc<Mutex<ScratchRing>>>>,
    /// v2 M1: how many rings have actually been *created* (not looked up). `warmup_rings` runs
    /// after load and before the first decode; any further increment means a ring materialized
    /// on the decode path, i.e. an address that was not fixed up front.
    ring_creations: AtomicUsize,
}

/// A staging ring's shape key: every projection's per-expert byte count (folds in dtype and
/// dims, and separates a layer whose gate/up/down are heterogeneously quantized), plus the
/// stacks' dtype.
type ScratchKey = ([usize; 3], u8);

/// One cache slot: reservation/published state, permanent storage, and LRU age.
/// See the state list in `m9a.md`; `storage` is cloned out under the state lock so uploads and
/// lending never hold the state lock while waiting for readers.
pub(crate) struct Slot {
    /// Reserved (weights copying) or published expert; `None` = empty.
    pub(crate) expert: Option<usize>,
    /// Published contents. Readers clone this `Arc` under the state lock, so a later fill never
    /// changes what an in-flight token computes with.
    pub(crate) buf: Option<Arc<SlotBuf>>,
    /// Copy-stream mode: the slot's permanent buffer; `None` while lent.
    pub(crate) storage: Arc<parking_lot::Mutex<Option<Arc<SlotBuf>>>>,
    /// `tick` of the last decode hit or fill; drives LRU replacement.
    pub(crate) last_use: u64,
}

impl Slot {
    pub(crate) fn empty() -> Self {
        Self {
            expert: None,
            buf: None,
            storage: Arc::new(parking_lot::Mutex::new(None)),
            last_use: 0,
        }
    }
}

pub(crate) struct LayerState {
    /// Step 5 doorbell: publishes this layer's top-k into mapped memory so the host read needs
    /// no stream sync or D2H. Acquired lazily at first decode.
    pub(crate) doorbell: Option<Arc<crate::doorbell::TopkDoorbell>>,
    /// seq the last ring of this layer advanced past (starts 0; first ring -> 1).
    pub(crate) doorbell_seq: u32,
    pub(crate) slots: Vec<Slot>,
    pub(crate) hit_counts: Vec<u64>,
    /// Decode steps seen by this layer.
    pub(crate) tick: u64,
    /// Slots `[0, usable)` are in service; the rest are lent to the prefill stream.
    pub(crate) usable: usize,
}

/// One expert's three projections.
#[derive(Clone)]
pub(crate) struct Projections {
    pub(crate) gate: Arc<QTensor>,
    pub(crate) up: Arc<QTensor>,
    pub(crate) down: Arc<QTensor>,
}

/// Slots `[first, first + len)` of a layer in copy-stream mode: one `[len, N, K]` device
/// allocation per projection, which the fused decode kernel indexes by `slot - first`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct Bank {
    pub(crate) first: usize,
    pub(crate) proj: Projections,
}

/// The two legal ways a slot can hold an expert.
pub(crate) enum SlotData {
    /// Non-copy-stream mode: this slot owns a fresh per-expert QTensor set.
    Owned(Projections),
    /// Copy-stream mode: this slot is a view into a shared bank, overwritten in place.
    Banked {
        bank: Arc<Bank>,
        memory: [SlotMemory; 3],
    },
}

/// One slot's contents plus the read fence readers record after launching its kernels.
pub(crate) struct SlotBuf {
    pub(crate) data: SlotData,
    pub(crate) fence: ReadFence,
}

/// The model-wide `--pcie-frac` staging ring (see `SCRATCH_RING_FIRST`): one bank of slots
/// past every resident slot, shared by every layer, borrowed round-robin per decode step.
///
/// `acquire(n)` runs under the registry lock, so consecutive layers get disjoint slots even
/// though all steps are on the token thread (the lock also covers a future multi-threaded
/// caller). Round-robin keeps the reuse distance at one full revolution of the ring: by the
/// time the cursor returns to a slot, several layers' readers have long recorded their fences,
/// so `wait_for_readers` never spins in practice.
/// Ring slots' `Bank.first` is `SCRATCH_RING_FIRST`, past every resident slot.
pub(crate) struct ScratchRing {
    slots: Vec<Arc<SlotBuf>>,
    cursor: usize,
}

#[cfg(feature = "cuda")]
const WAITS_PER_LAYER: u32 = 2;

pub(crate) struct CachedLayer {
    /// Page-locked registrations of `cpu`'s stacks; each holds its stack, so it stays valid
    /// for as long as it is registered.
    _pinned: Vec<PinnedStack>,
    cpu: MoEExperts,
    slot_device: Device,
    layer: usize,
    registry: Arc<LayerRegistry>,
    hidden: usize,
    num_experts: usize,
    top_k: usize,
    act: Activation,
    /// `[num_experts, hidden]` F32 on the CPU; `None` disables the lookahead for this layer.
    router: Option<Tensor>,
    /// R2a: the layer's GPU->host activation channel. `Some` only when the router exists
    /// and the slot device is CUDA.
    lookahead_chan: Option<Arc<LookaheadChannel>>,
    state: Mutex<LayerState>,
    /// Uploads fills alongside compute; `None` off CUDA, where fills go through candle.
    copy_stream: Option<Arc<CopyStream>>,
    /// Copy-stream mode: the banks as `(first slot, len)`, in slot order. Lent whole.
    banks: Vec<(usize, usize)>,
    /// `--lookahead-swaps`: how many predicted experts may displace residents per row and pass
    /// (`0` means a prediction can only fill an empty slot).
    lookahead_swaps: usize,
    /// `--no-lookahead`: this layer does not predict for the next one.
    lookahead: bool,
    /// `--pcie-frac`: share of a step's misses to stream over PCIe into the model-wide staging
    /// ring (held by the registry, created lazily at the first decode — `stream_pcie`; 0.0
    /// disables the feature).
    pcie_frac: f32,
    /// Runs long prefill chunks' experts on the GPU (`--prefill-stream`).
    prefill: Option<Arc<PrefillStream>>,
    /// A background fill of this layer is running; at most one at a time.
    filling: AtomicBool,
    /// v2 M5b: the pass whose preload copies the prediction thread has claimed (0 = none). Claimed
    /// before the copies are queued and before the point is published, so the token thread replays
    /// the preload gate exactly when a publish is guaranteed to arrive for its pass.
    preload_claimed: AtomicU32,
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
        // next layer in the background, off the token path. This layer is filled too, for the
        // first layer of the stack
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
            if let Some(ps) = &layer.prefill {
                if let Some(out) = ps.forward(layer.layer, xs, &topk_weights, topk_ids)? {
                    return Ok(out);
                }
            }
            let out = layer.cpu.forward(
                &xs.to_device(&Device::Cpu)?,
                topk_weights.to_device(&Device::Cpu)?,
                &topk_ids.to_device(&Device::Cpu)?,
            )?;
            return out.to_device(&layer.slot_device);
        }

        // Decode: single token, split GPU hits / CPU misses.
        if let Some(ps) = &layer.prefill {
            ps.note_decode(layer.layer)?;
        }
        layer.decode_forward(xs, &topk_weights, topk_ids, layer.act)
    }
}
