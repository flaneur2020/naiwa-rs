//! Cached layers and the model-wide `Layers` index.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use candle_core::{DType, Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::RoutedExperts;
use mistralrs_core::MoEExperts;
use parking_lot::RwLock;
use tracing::debug;

use crate::copy_stream::{self, CopyDone, CopyStream, ReadFence};
use crate::doorbell::{PassCell, TopkDoorbell};
use crate::host_memory::PinnedStack;
use crate::prefill_stream::PrefillStream;
use crate::stats;

use super::{
    expert_qtensor_on_device, expert_range, Projections, Slot, SlotBuf,
    SlotData, StagingRings,
};

/// Every cached layer of one model, by decoder-layer index.
/// Weak references: the model owns the layers, not this index.
pub(crate) struct Layers {
    layers: RwLock<Vec<Weak<CachedLayer>>>,
}

impl Default for Layers {
    fn default() -> Self {
        Self {
            layers: RwLock::new(Vec::new()),
        }
    }
}

impl Layers {
    pub(crate) fn insert(&self, layer: usize, cached: &Arc<CachedLayer>) {
        let mut layers = self.layers.write();
        if layers.len() <= layer {
            layers.resize_with(layer + 1, Weak::new);
        }
        layers[layer] = Arc::downgrade(cached);
    }

    pub(crate) fn get(&self, layer: usize) -> Option<Arc<CachedLayer>> {
        self.layers.read().get(layer).and_then(Weak::upgrade)
    }

    /// Empty every slot whose bank overlaps `[base, end)`, and report how many were emptied.
    ///
    /// This is the borrow's half of the deal: the prefill stream is about to write those bytes, so whatever experts
    /// lived there are gone. The slots keep their storage (the banks are not rebuilt), they simply come back empty
    /// and refill as after any eviction. The buffer is taken out of the slot before waiting for its readers - a
    /// reader still holding a clone would make the wait spin forever (the trap `wait_for_readers` sets).
    pub(crate) fn empty_range(&self, base: u64, end: u64) -> usize {
        let layers = self.all();
        // Phase 1: take every overlapping slot's buffer out, holding only the locks that are needed for that. The
        // wait comes after every lock is released: a background fill holds a clone of the buffer it is filling and
        // needs `layer.state` to finish, so waiting under that lock deadlocks the two against each other (the
        // W105 family of hang, and the first version of this function did exactly that).
        let mut held: Vec<(usize, usize, Arc<SlotBuf>)> = Vec::new();
        for (li, layer) in layers.iter().enumerate() {
            let mut state = layer.state.lock().unwrap();
            for (si, slot) in state.slots.iter_mut().enumerate() {
                let overlaps = {
                    let storage = slot.storage.lock();
                    let Some(buf) = storage.as_ref() else { continue };
                    let SlotData::Banked { memory, .. } = &buf.data else {
                        continue;
                    };
                    memory.iter().any(|part| {
                        let start = part.ptr() as u64;
                        let stop = start + part.len() as u64;
                        start < end && stop > base
                    })
                };
                if !overlaps {
                    continue;
                }
                let Some(buf) = slot.storage.lock().take() else {
                    continue;
                };
                slot.expert = None;
                slot.buf = None;
                held.push((li, si, buf));
            }
        }
        // Phase 2: the fills and reads in flight finish, with no lock held.
        for (_, _, buf) in &held {
            crate::copy_stream::wait_for_readers(buf);
        }
        let emptied = held.len();
        // After emptying, no slot in the range may still claim an expert: if one does, some path re-published it
        // (or never cleared it) and the chunk is about to overwrite bytes a decode step would trust.
        let mut still = 0usize;
        for layer in self.all() {
            let state = layer.state.lock().unwrap();
            for slot in state.slots.iter() {
                let overlaps = {
                    let storage = slot.storage.lock();
                    let Some(buf) = storage.as_ref() else { continue };
                    let SlotData::Banked { memory, .. } = &buf.data else { continue };
                    memory.iter().any(|part| {
                        let start = part.ptr() as u64;
                        let stop = start + part.len() as u64;
                        start < end && stop > base
                    })
                };
                if overlaps && slot.expert.is_some() {
                    still += 1;
                }
            }
        }
        if still > 0 {
            eprintln!("BORROW_STILL {still} slots still resident in the borrow range");
        }
        // Phase 3: the storages go back where they came from, empty - and cleared *again*, because a fill that was
        // already in flight reserved its slot (expert = Some, buf = None) before phase 1 and publishes after
        // phase 2. Without the second clear the slot would claim an expert whose bytes the chunk has overwritten,
        // which is exactly the garbage a decode step then computes on.
        for (li, si, buf) in held {
            let mut state = layers[li].state.lock().unwrap();
            let slot = &mut state.slots[si];
            slot.expert = None;
            slot.buf = None;
            slot.last_use = 0;
            *slot.storage.lock() = Some(buf);
        }
        emptied
    }

    pub(crate) fn all(&self) -> Vec<Arc<CachedLayer>> {
        self.layers
            .read()
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }
}

/// R2a: the lookahead's per-layer publish channel (GPU -> host). At each MoE entry the
/// token thread queues ONE kernel that copies this step's activation into mapped memory
/// and bumps seq; the lookahead worker spins for the seq and runs the router matmul on
/// the CPU (the router weights moved back to the host, `[128, 2048]` F32 ~ 1 MiB/layer).
/// The token thread never waits — no `to_vec1` D2H (W105b: 0.30-0.34 ms/layer of blocking
/// wait) and no way for the prediction to queue late on the stream (the W99 failure that
/// made an earlier mode switch regress hit 74% -> 38-45%).
pub(crate) struct LookaheadChannel {
    chan: Arc<crate::doorbell::ActivationDoorbell>,
    /// The seq the last queued GPU publish will bump the counter to (0 before any):
    /// `last_seq + 1` is the value `wait_publish` waits for. Touched only by the token thread.
    next_seq: std::sync::atomic::AtomicU32,
    /// How many queued publishes the worker has consumed (`wait_publish` returned).
    /// The difference to `next_seq` is the queue depth; publish backpressure keeps
    /// it below the doorbell's ring depth so a lagging job's slot is never overwritten.
    consumed: std::sync::atomic::AtomicU32,
}

impl LookaheadChannel {
    /// The channel plus the CPU matmul weights `router` must already provide.
    ///
    /// `max_rows` must cover the LARGEST decode batch this layer will see — not
    /// just np+1. The target's verify batch is `staged + 1` rows and the MTP
    /// drafter runs its own windows; sizing for one shape and receiving a
    /// bigger one overflowed the slot and (on the last ring slot) wrote past
    /// the whole allocation into the neighboring doorbell buffer: garbage
    /// router ids shaped like the activation's f32 bits (W106). 16 rows is
    /// headroom over any current window (np4 verify = 5, observed verify = 7).
    pub(super) fn new(
        device: &Device,
        max_rows: usize,
        hidden: usize,
    ) -> candle_core::Result<Option<Arc<Self>>> {
        let max_rows = max_rows.max(16);
        Ok(
            crate::doorbell::ActivationDoorbell::acquire(device, max_rows * hidden)
                .ok()
                .flatten()
                .map(|chan| {
                    Arc::new(Self {
                        chan: Arc::new(chan),
                        next_seq: std::sync::atomic::AtomicU32::new(0),
                        consumed: std::sync::atomic::AtomicU32::new(0),
                    })
                }),
        )
    }

    /// Queue the publish on the stream (token thread): one kernel writes the `[rows,
    /// hidden]` activation into a slot and bumps seq. Returns this channel plus the
    /// `last_seq` the worker job needs to wait and read, or `None` when the worker is more
    /// than `depth - 1` publishes behind — the prediction is advisory, so skipping this
    /// publish is cheaper than overwriting an unread slot.
    /// The widest publish this channel's mapped slot can hold, in `f32` elements.
    pub(crate) fn max_n(&self) -> usize {
        self.chan.max_n()
    }

    pub(crate) fn publish(
        self: &Arc<Self>,
        x: &Tensor,
        n: usize,
    ) -> candle_core::Result<Option<(Arc<Self>, u32)>> {
        // Hard error, not a skip: an oversized publish would overflow the slot
        // (the doorbell checks too, but fail before queueing anything).
        if n > self.chan.max_n() {
            candle_core::bail!(
                "lookahead publish of {n} f32 exceeds the channel slot ({})",
                self.chan.max_n()
            );
        }
        let next = self.next_seq.load(Ordering::Relaxed);
        let consumed = self.consumed.load(Ordering::Relaxed);
        if next - consumed >= (self.chan.depth() as u32).saturating_sub(1) {
            return Ok(None); // backpressure: an unread slot would be overwritten
        }
        self.chan.publish(x, n, next)?;
        self.next_seq.store(next + 1, Ordering::Relaxed);
        Ok(Some((Arc::clone(self), next)))
    }

    /// Wait for the publish `last_seq`, dropping it when a newer publish has already superseded
    /// it. Always advances `consumed`, including timeout/stale, so a lost prediction does not
    /// keep consuming the backpressure budget.
    pub(crate) fn read(&self, last_seq: u32, n: usize) -> Option<Vec<f32>> {
        let next = last_seq.wrapping_add(1);
        if self.next_seq.load(Ordering::Relaxed) > next {
            self.consumed.fetch_max(next, Ordering::Relaxed);
            return None;
        }
        let out = self.chan.wait_publish(last_seq, n);
        self.consumed.fetch_max(next, Ordering::Relaxed);
        out
    }
}

// ---------------------------------------------------------------------------
// CachedLayer — the real workhorse
// ---------------------------------------------------------------------------

/// The most-routed expert not in (or reserved for) a slot, ties to the lowest id; `None` when
/// every slot is taken.
pub(super) fn fill_candidate(slots: &[Slot], hit_counts: &[u64]) -> Option<usize> {
    if slots.iter().all(|s| s.expert.is_some()) {
        return None;
    }
    let mut taken = vec![false; hit_counts.len()];
    for e in slots.iter().filter_map(|s| s.expert) {
        taken[e] = true;
    }
    (0..hit_counts.len())
        .filter(|&e| !taken[e])
        .max_by_key(|&e| (hit_counts[e], std::cmp::Reverse(e)))
}

/// Indices of the `k` largest `logits`, largest first (ties: lowest index).
pub(super) fn top_k_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    ids.truncate(k);
    ids
}

/// Every expert in `rankings` once, ordered by how many rankings list it, then by its best rank,
/// then by id.
pub(super) fn union_by_votes(rankings: impl IntoIterator<Item = Vec<usize>>) -> Vec<usize> {
    let mut seen: HashMap<usize, (usize, usize)> = HashMap::new(); // expert -> (votes, best rank)
    for ranking in rankings {
        for (rank, e) in ranking.into_iter().enumerate() {
            let entry = seen.entry(e).or_insert((0, rank));
            entry.0 += 1;
            entry.1 = entry.1.min(rank);
        }
    }
    let mut union: Vec<(usize, (usize, usize))> = seen.into_iter().collect();
    union.sort_by(|(a, (va, ra)), (b, (vb, rb))| vb.cmp(va).then(ra.cmp(rb)).then(a.cmp(b)));
    union.into_iter().map(|(e, _)| e).collect()
}

/// The misses to stream over PCIe (`--pcie-frac`): the distinct experts a step missed, ranked by
/// the total router weight routed to them, truncated to `cap`. An expert missed by several rows
/// is DMA'd once and serves them all, so the distinct count is what `frac` applies to; when the
/// fraction does not cover every miss, the ones that contribute most to the output go first.
pub(crate) fn streamed_misses(misses: &[Vec<(u32, f32)>], frac: f32, cap: usize) -> Vec<u32> {
    if frac <= 0.0 || cap == 0 {
        return Vec::new();
    }
    let mut weight: HashMap<u32, f32> = HashMap::new();
    for (id, w) in misses.iter().flatten() {
        *weight.entry(*id).or_insert(0.0) += *w;
    }
    let take = ((frac * weight.len() as f32).ceil() as usize).min(cap);
    let mut ranked: Vec<(u32, f32)> = weight.into_iter().collect();
    ranked.sort_by(|(a, wa), (b, wb)| wb.total_cmp(wa).then(a.cmp(b)));
    ranked.truncate(take);
    ranked.into_iter().map(|(id, _)| id).collect()
}

/// The least-recently-used published resident that is not in `protect`, as `(expert, slot)`.
pub(super) fn lru_victim(slots: &[Slot], protect: &[usize]) -> Option<(usize, usize)> {
    slots
        .iter()
        .enumerate()
        .filter_map(|(slot, s)| {
            let expert = s.expert?;
            (s.buf.is_some() && !protect.contains(&expert)).then_some((expert, slot))
        })
        .min_by_key(|(expert, slot)| (slots[*slot].last_use, *expert))
}

// Number of rows above which we skip the cache entirely.
// Calls with more rows than this are prefill chunks: every expert runs on the CPU (or the prefill
// stream). Up to it they take the decode split, which is what a speculative verify step needs (the
// target runs the n drafts plus one token in one forward).
//
// 32 because of the *tail* of a prompt chunk. The recurrent checkpoint boundary is
// `align_down(prompt, 32)`, so every prompt is prefilled as `[0, boundary)` followed by a remainder
// of at most 31 rows — and at 16 those remainders took the whole-chunk CPU path, where a few rows
// still touch nearly every expert of every layer (measured 2.3 s per `doc`/`long` turn). At 32 they
// take the decode split instead, whose expert cache the preceding chunk has just warmed.
const PREFILL_THRESHOLD: usize = 32;

/// The largest decode window the cache serves, i.e. the row count every warmup-sized mapped buffer
/// and the CPU-result ring must cover. `PREFILL_THRESHOLD` is the same number by construction.
pub(crate) const MAX_DECODE_ROWS: usize = PREFILL_THRESHOLD;

/// Mutable per-layer decode state: doorbell, slots and LRU/hit counters.
pub(crate) struct LayerState {
    /// Step 5 doorbell: publishes this layer's top-k into mapped memory so the host read needs
    /// no stream sync or D2H. Acquired lazily at first decode.
    pub(crate) doorbell: Option<Arc<crate::doorbell::TopkDoorbell>>,
    pub(crate) slots: Vec<Slot>,
    pub(crate) hit_counts: Vec<u64>,
    /// Decode steps seen by this layer.
    pub(crate) tick: u64,
    /// Slots `[0, usable)` are in service; the rest are lent to the prefill stream.
    pub(crate) usable: usize,
}

pub(crate) struct CachedLayer {
    /// Page-locked registrations of `cpu`'s stacks; each holds its stack, so it stays valid
    /// for as long as it is registered.
    pub(crate) _pinned: Vec<PinnedStack>,
    pub(crate) cpu: MoEExperts,
    pub(crate) slot_device: Device,
    pub(crate) layer: usize,
    pub(crate) layers: Arc<Layers>,
    pub(crate) staging: Arc<StagingRings>,
    /// This step's staged experts, with the ring buffer each was staged into. Written and consumed **within one
    /// step on the token thread** - deliberately not across a step boundary: a surviving `Arc` of a ring slot keeps
    /// its refcount above one, and `stage_misses` waits on that refcount (`wait_for_readers`).
    pub(crate) staged_now: Mutex<Vec<(u32, Arc<SlotBuf>)>>,
    pub(crate) hidden: usize,
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) act: Activation,
    /// `[num_experts, hidden]` F32 on the CPU; `None` disables the lookahead for this layer.
    pub(crate) router: Option<Tensor>,
    /// R2a: the layer's GPU->host activation channel. `Some` only when the router exists
    /// and the slot device is CUDA.
    pub(crate) lookahead_chan: Option<Arc<LookaheadChannel>>,
    pub(crate) state: Mutex<LayerState>,
    /// Uploads fills alongside compute; `None` off CUDA, where fills go through candle.
    pub(crate) copy_stream: Option<Arc<CopyStream>>,
    /// `--lookahead-swaps`: how many predicted experts may displace residents per row and pass
    /// (`0` means a prediction can only fill an empty slot).
    pub(crate) lookahead_swaps: usize,
    /// `--no-lookahead`: this layer does not predict for the next one.
    pub(crate) lookahead: bool,
    /// `--pcie-frac`: share of a step's misses to stream over PCIe into the model-wide staging
    /// ring (held by the provider, created lazily at the first decode — `stream_pcie`; 0.0
    /// disables the feature).
    pub(crate) pcie_frac: f32,
    /// Runs long prefill chunks' experts on the GPU (`--prefill-stream`).
    pub(crate) prefill: Option<Arc<PrefillStream>>,
    /// A background fill of this layer is running; at most one at a time.
    pub(crate) filling: AtomicBool,
    /// The preload handoff: the lookahead thread records its batch's completion and claims it, the
    /// token thread waits for the claim it sees (`wait_preload`). `None` when the layer has no copy
    /// stream, i.e. when its fills go through candle's stream and need no ordering.
    pub(crate) preload: Option<PassCell>,
    /// Completion of this layer's staging DMA on the copy stream (`--pcie-frac`). `None` disables
    /// the staging path for this layer: without a completion signal the staged slots could be read
    /// while the copies are still in flight, so the path fails closed instead.
    pub(crate) staging_done: Option<Arc<CopyDone>>,
}

/// The model holds `Arc<dyn RoutedExperts>`; this wrapper keeps an `Arc<CachedLayer>` so the
/// layer can hand itself to background fills.
pub(crate) struct CachedExperts(pub(crate) Arc<CachedLayer>);

impl RoutedExperts for CachedExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let layer = &self.0;
        let (batch, seq, hidden_dim) = xs.dims3()?;
        // Which shape the MoE forward is handed is the difference between one batched expert pass and one pass per
        // row, so it belongs in the log rather than in an inference from the timing (`NAIWA_STREAM_DBG=1`).
        crate::dbg_note("moe_forward", &format!("layer {}: batch {batch} seq {seq}", layer.layer));
        debug_assert_eq!(hidden_dim, layer.hidden);
        let num_tokens = batch * seq;

        // Prefill: every routed expert runs on the CPU. Meanwhile, fill the empty slots of the
        // next layer in the background, off the token path. This layer is filled too, for the
        // first layer of the stack
        // (no predecessor triggers it); a layer already full or filling is a no-op.
        // The decode side of the VRAM low-water instrument (`NAIWA_PREFILL_PEAK=1`), once per step: sampling
        // every layer would be 48 driver calls per token.
        if layer.layer == 0 && num_tokens <= PREFILL_THRESHOLD && crate::vram_peak::enabled() {
            crate::vram_peak::note_step(&layer.slot_device);
        }
        if num_tokens > PREFILL_THRESHOLD {
            let ids_vec = topk_ids.flatten_all()?.to_vec1::<u32>()?;
            {
                let mut st = layer.state.lock().unwrap();
                for &id in &ids_vec {
                    st.hit_counts[id as usize] += 1;
                }
            }
            // `NAIWA_NO_PREFILL_FILL=1` (probe): the fills below issue PCIe copies on the *same* copy stream the
            // prefill stream's layer buffers use, so a chunk's DMA competes with them for the link. When the
            // stream is serving this layer, its experts are already on the device and the fill is redundant.
            // A borrowing stream writes into the cache's own bytes, so nothing may be filling those slots at the
            // same time: a background fill would hold a slot buffer (and queue a copy into it) across the chunk.
            // The fills are worthless during a chunk anyway - measured, `NAIWA_NO_PREFILL_FILL` changes nothing.
            if std::env::var_os("NAIWA_NO_PREFILL_FILL").is_none()
                && !crate::moe::slab::Slab::borrowing()
            {
                if let Some(next) = layer.layers.get(layer.layer + 1) {
                    next.spawn_fill();
                }
                layer.spawn_fill();
            }
            // Ring batching (`NAIWA_PREFILL_RINGS=1`): the chunk's experts in rounds over the layer's own
            // slots, which is the memory-free version of the prefill stream.
            if let Some(out) = crate::moe::rings_exec::forward(layer, xs, &topk_weights, topk_ids)? {
                return Ok(out);
            }
            if let Some(ps) = &layer.prefill {
                if crate::vram_peak::checking() {
                    let finite_in = xs
                        .to_dtype(DType::F32)?
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|v| v.is_finite());
                    if !finite_in {
                        eprintln!("BORROW_IN layer {}: chunk INPUT not finite", layer.layer);
                    }
                }
                if let Some(out) = ps.forward(layer.layer, xs, &topk_weights, topk_ids)? {
                    // `NAIWA_BORROW_CHECK=1`: a chunk whose output is not finite is the borrow corrupting its own
                    // buffer; one whose output is finite points the failure at the decode side instead.
                    if std::env::var_os("NAIWA_BORROW_CHECK").is_some() {
                        // `flatten_all`, not `to_vec2`: `out` is [batch, seq, hidden] and `to_vec2` requires
                        // rank 2 - the check itself raised "unexpected rank" until this was fixed, which made a
                        // diagnostic look like the defect.
                        let finite = out
                            .to_dtype(candle_core::DType::F32)?
                            .to_device(&candle_core::Device::Cpu)?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                            .iter()
                            .all(|v| v.is_finite());
                        if !finite {
                            crate::dbg_note(
                                "prefill_borrow",
                                &format!("layer {}: chunk output is NOT finite", layer.layer),
                            );
                            eprintln!("BORROW_CHECK layer {}: chunk output not finite", layer.layer);
                        }
                    }
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

        // Decode: single token, split GPU hits / CPU misses. A decode step means the chunk that borrowed the
        // cache's bytes is over, so the lookahead may fill again.
        super::slab::set_borrowed(false);
        if layer.layer == 0 {
            super::quiesce(false);
        }
        let out = super::decode::forward(layer, xs, &topk_weights, topk_ids, layer.act)?;
        // `NAIWA_BORROW_CHECK=1`: the first layer whose decode output is not finite is where garbage entered; a
        // borrowed chunk empties the whole cache, so a decode step right after it reads slots that must be refilled.
        if crate::vram_peak::checking() {
            let finite = out
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite());
            if !finite {
                static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 8 {
                    eprintln!("DECODE_NAN layer {} (rows {num_tokens})", layer.layer);
                }
            }
        }
        Ok(out)
    }
}

/// Promotions spent in the current step, across layers (see `CachedLayer::promote_staged`).
static PROMOTED_THIS_STEP: AtomicUsize = AtomicUsize::new(0);

impl CachedLayer {
    pub(crate) fn state(&self) -> &Mutex<LayerState> {
        &self.state
    }

    pub(crate) fn cpu(&self) -> &MoEExperts {
        &self.cpu
    }

    pub(crate) fn slot_device(&self) -> &Device {
        &self.slot_device
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    pub(crate) fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }

    pub(crate) fn act(&self) -> Activation {
        self.act
    }

    pub(crate) fn router(&self) -> Option<&Tensor> {
        self.router.as_ref()
    }

    pub(crate) fn lookahead(&self) -> bool {
        self.lookahead
    }

    pub(crate) fn lookahead_chan(&self) -> Option<&Arc<LookaheadChannel>> {
        self.lookahead_chan.as_ref()
    }

    pub(crate) fn copy_stream(&self) -> Option<&Arc<CopyStream>> {
        self.copy_stream.as_ref()
    }

    pub(crate) fn pcie_frac(&self) -> f32 {
        self.pcie_frac
    }

    /// Remember this step's staged experts, replacing anything left over (which drops those `Arc`s).
    pub(crate) fn set_staged_now(&self, v: Vec<(u32, Arc<SlotBuf>)>) {
        *self.staged_now.lock().unwrap() = v;
    }

    pub(crate) fn staging_done(&self) -> Option<&Arc<CopyDone>> {
        self.staging_done.as_ref()
    }

    pub(crate) fn layers(&self) -> &Arc<Layers> {
        &self.layers
    }

    pub(crate) fn staging(&self) -> &StagingRings {
        &self.staging
    }

    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    /// Is the PCIe staging path in service for this layer? True only when the copies have a
    /// completion event: `--pcie-frac 0`, or a layer whose event could not be created, must not
    /// DMA into slots that nobody waits for.
    pub(crate) fn staging_enabled(&self) -> bool {
        self.staging_done.is_some()
    }

    /// Wait on the calling stream for this layer's staging DMA (a GPU-side event wait; the host
    /// does not block). A no-op when the layer does not stage.
    pub(crate) fn wait_staging(&self) -> candle_core::Result<()> {
        let Some(done) = &self.staging_done else {
            return Ok(());
        };
        stats::record_cache_sync_wait();
        done.wait_on(&self.slot_device)
    }

    /// Wait on the calling stream for the latest preload claim, if any. Waiting for the claim this
    /// call *sees* — rather than for a pass number of its own — is what drains a copy the lookahead
    /// queued late: a stale claim is still waited for, so the slots cannot be read mid-copy.
    pub(crate) fn wait_preload(&self) -> candle_core::Result<()> {
        let Some(cell) = &self.preload else {
            return Ok(());
        };
        if cell.wait_latest(&self.slot_device)?.is_some() {
            stats::record_cache_sync_wait();
        }
        Ok(())
    }

    /// Create this layer's top-k doorbell at warmup, sized for the largest decode window. Without
    /// it the router read falls back to a blocking D2H, which is correct but slow.
    pub(crate) fn warmup_doorbell(&self) -> candle_core::Result<()> {
        let cap = MAX_DECODE_ROWS * self.top_k;
        if let Some(bell) = TopkDoorbell::acquire(&self.slot_device, cap)? {
            self.state.lock().unwrap().doorbell = Some(Arc::new(bell));
        }
        Ok(())
    }

    /// v2 M1: create this layer's staging ring now (see `GpuCacheMoeProvider::warmup_rings`).
    /// A no-op when the layer has no copy stream or no CPU `QTensor`s. Called once after load.
    pub(super) fn warmup_staging_ring(&self) -> candle_core::Result<()> {
        let Some(cs) = &self.copy_stream else {
            return Ok(());
        };
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(());
        };
        self.staging.ring(
            cs,
            &(Arc::clone(&gate), Arc::clone(&up), Arc::clone(&down)),
            &self.slot_device,
        )?;
        Ok(())
    }

    /// Byte-slice expert `expert_id` out of the CPU stacks onto the slot device — never
    /// dequantized. `None` when the backend holds no `QTensor`s.
    fn expert_weights(&self, expert_id: usize) -> candle_core::Result<Option<Projections>> {
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(None);
        };
        Ok(Some(Projections {
            gate: expert_qtensor_on_device(&gate, expert_id, &self.slot_device)?,
            up: expert_qtensor_on_device(&up, expert_id, &self.slot_device)?,
            down: expert_qtensor_on_device(&down, expert_id, &self.slot_device)?,
        }))
    }

    /// Promote this step's staged experts into the cache, up to `NAIWA_PROMOTE_MAX` per layer per step.
    ///
    /// A staged expert has **already crossed the link**, so admitting it is a device-to-device copy rather than a
    /// second H2D: it turns the next step's most expensive miss (the CPU path, ~250 us) into a hit. Polluting the
    /// cache is the risk, hence the cap and the order (empty slots first, LRU victims after).
    ///
    /// Called at the **end** of a layer's step, after the fence loop has recorded the staged slots' readers - so the
    /// copies queue behind the kernels that read them, on the copy stream. The *next* step waits on
    /// [`Self::promote_done`] before it submits the resident experts, which is what orders the readers of the
    /// published slots.
    ///
    /// Nothing here outlives the step: the `Arc`s come from the step's own hit list, so the staging ring's refcounts
    /// fall as they always did (`wait_for_readers` depends on exactly that).
    /// Promote up to this many staged experts **per layer per step** (default 1; `0` switches promotion off
    /// entirely, and is the only value that makes the decode path pay nothing for it).
    fn promote_cap() -> usize {
        static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CAP.get_or_init(|| {
            std::env::var("NAIWA_PROMOTE_MAX")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1)
        })
    }

    /// Whether promotion is switched on at all. Read once: the decode path calls this every layer, so it must not
    /// touch the environment.
    pub(crate) fn promoting(&self) -> bool {
        Self::promote_cap() > 0
    }

    pub(crate) fn promote_staged(&self, candle_stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>) -> candle_core::Result<usize> {
        // The cap is **per step across all layers**, not per layer: at a per-layer cap of 8 this queued
        // 48 x 8 = 384 d2d copies of 1.88 MB (720 MB) per token, which ran prefill at 5.8 T/s.
        let cap = Self::promote_cap();
        if self.layer == 0 {
            PROMOTED_THIS_STEP.store(0, Ordering::Relaxed);
        }
        let _ = &PROMOTED_THIS_STEP;
        // Always drain, even with the cap at zero: these `Arc`s must not survive the step.
        let staged = std::mem::take(&mut *self.staged_now.lock().unwrap());
        if std::env::var_os("NAIWA_PROMOTE_TRACE").is_some() {
            eprintln!(
                "PROMOTE_ENTER layer {} cap {} staged {} promoting {}",
                self.layer,
                cap,
                staged.len(),
                self.promoting()
            );
        }
        if cap == 0 || staged.is_empty() {
            return Ok(0);
        }
        // Never block the token thread here: the lookahead holds this lock while it waits for *our* readers to
        // drop (see `slab::try_lock_access`), so blocking would deadlock. Standing down costs one step of
        // promotions, which is the right failure for an optimisation.
        let Some(_access) = super::slab::try_lock_access() else {
            return Ok(0);
        };
        if super::slab::borrowed() {
            return Ok(0);
        }
        // **Second sighting, at least**: an expert that just missed is by definition a rare one, and evicting a hot
        // resident to hold it is how this mechanism lost 2-4 ms (§23/§24). `hit_counts` counts *demand* (every routed
        // expert, resident or not), so this admits only what has actually been asked for repeatedly.
        let min_sightings = std::env::var("NAIWA_PROMOTE_MIN_HITS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(4);
        let mut promoted = 0usize;
        let mut skipped = 0usize;
        let mut cold = 0usize;
        let mut seen: Vec<u32> = Vec::with_capacity(cap);
        for (expert, src_buf) in staged.iter() {
            // The cap is **per layer per step**: a per-step budget spent itself in the first few layers, which
            // made the choice of what to promote arbitrary (measured: layers 0-3 consumed all 8, layers 45-47
            // promoted nothing). At `cap = 1` that is 48 copies = 90 MB per step, half of what staging moves.
            if promoted >= cap {
                break;
            }
            if seen.contains(expert) {
                skipped += 1;
                continue;
            }
            seen.push(*expert);
            let expert = *expert as usize;
            let slot = {
                let mut st = self.state.lock().unwrap();
                if st.slots.iter().any(|s| s.expert == Some(expert)) {
                    skipped += 1;
                    continue;
                }
                if st.hit_counts[expert] < min_sightings {
                    cold += 1;
                    continue;
                }
                // An empty slot first, then the least-recently-used one - but never a slot this step touched:
                // `Route::submit` stamps `last_use = tick` on every hit, so `last_use < tick` is exactly "no kernel
                // of this step reads it", which is what makes overwriting it safe (the plan was built before
                // promotion, so its read fence does not cover this copy).
                let usable = st.usable;
                let tick = st.tick;
                let empty = st.slots[..usable].iter().position(|s| s.expert.is_none());
                let victim = st.slots[..usable]
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.expert.is_some() && s.buf.is_some() && s.last_use < tick)
                    .min_by_key(|(slot, s)| (s.last_use, *slot))
                    .map(|(slot, _)| slot);
                let Some(slot) = empty.or(victim) else {
                    skipped += 1;
                    continue;
                };
                st.slots[slot].expert = Some(expert);
                st.slots[slot].buf = None;
                slot
            };
            let storage = {
                let st = self.state.lock().unwrap();
                Arc::clone(&st.slots[slot].storage)
            };
            let copied: candle_core::Result<Arc<SlotBuf>> = 'copy: {
                let entry = storage.lock();
                let Some(dst) = entry.as_ref() else {
                    break 'copy Ok(Arc::clone(src_buf));
                };
                match (&dst.data, &src_buf.data) {
                    (SlotData::Banked { memory: dmem, .. }, SlotData::Banked { memory: smem, .. }) => {
                        // **d2d on candle's own stream**, not on the PCIe copy stream: the kernels reading the
                        // staging slot are queued there, so the copy is ordered after them by the stream itself -
                        // no event to wait on, and nothing added to the queue that carries this step's 180 MB of
                        // staging. The slot's fence is then recorded *after* the copy, so a later upload from the
                        // lookahead (which waits on that fence) cannot overwrite it mid-flight.
                        for (d, s2) in [(&dmem[0], &smem[0]), (&dmem[1], &smem[1]), (&dmem[2], &smem[2])] {
                            if s2.len() < d.len() {
                                candle_core::bail!("promotion of {} bytes into a {}-byte slot", s2.len(), d.len());
                            }
                            // SAFETY: both are live device allocations, `d.len` is a prefix of the source, and the
                            // copy is queued on the stream that reads the source and will read the destination.
                            unsafe {
                                candle_core::cuda_backend::cudarc::driver::result::memcpy_dtod_async(
                                    d.ptr(), s2.ptr(), d.len(), candle_stream.cu_stream(),
                                )
                            }
                            .map_err(candle_core::Error::wrap)?;
                        }
                        dst.fence.record_reads(&self.slot_device)?;
                        Ok(Arc::clone(dst))
                    }
                    _ => Ok(Arc::clone(src_buf)),
                }
            };
            let mut st = self.state.lock().unwrap();
            match copied {
                Ok(buf) => {
                    let tick = st.tick;
                    let s = &mut st.slots[slot];
                    s.buf = Some(buf);
                    s.last_use = tick;
                    PROMOTED_THIS_STEP.fetch_add(1, Ordering::Relaxed);
                    promoted += 1;
                }
                Err(e) => {
                    st.slots[slot].expert = None;
                    return Err(e);
                }
            }
        }
        if promoted > 0 {
            stats::record_promotions(promoted);
        }
        if std::env::var_os("NAIWA_PROMOTE_TRACE").is_some() {
            // Off by default; `NAIWA_PROMOTE_TRACE=1` prints one line per layer per step.
            eprintln!(
                "PROMOTE layer {} staged {} promoted {} skipped {} cold {} filled {}",
                self.layer,
                staged.len(),
                promoted,
                skipped,
                cold,
                self.state.lock().unwrap().slots.iter().filter(|s| s.expert.is_some()).count()
            );
        }
        Ok(promoted)
    }

    /// Put `expert_id` into an empty slot. The slot is reserved under the lock before the
    /// (slow) copy and published only after it, so concurrent admitters never share a slot
    /// and readers never see a half-written one. `None`: already resident or no empty slot.
    pub(crate) fn admit(&self, expert_id: usize, sync: bool) -> candle_core::Result<Option<usize>> {
        if super::slab::borrowed() {
            // A borrowed chunk owns these bytes right now (see `slab::borrowed`).
            return Ok(None);
        }
        let slot = {
            let mut st = self.state.lock().unwrap();
            if st.slots.iter().any(|s| s.expert == Some(expert_id)) {
                return Ok(None);
            }
            let usable = st.usable;
            let Some(slot) = st.slots[..usable].iter().position(|s| s.expert.is_none()) else {
                return Ok(None);
            };
            st.slots[slot].expert = Some(expert_id);
            slot
        };
        self.load_reserved(expert_id, slot, sync)
    }

    /// Replace the least-recently-used resident outside `protect` with `expert_id`. The victim
    /// is unpublished and the slot reserved in one critical section. Tokens already holding the
    /// victim keep computing with it: a fresh upload leaves the old buffer alone, and an
    /// in-place one waits for them (`upload_in_place`).
    pub(crate) fn replace(
        &self,
        expert_id: usize,
        protect: &[usize],
        sync: bool,
    ) -> candle_core::Result<Option<usize>> {
        if super::slab::borrowed() {
            return Ok(None);
        }
        let slot = {
            let mut st = self.state.lock().unwrap();
            if st.slots.iter().any(|s| s.expert == Some(expert_id)) {
                return Ok(None);
            }
            let Some((_victim, slot)) = lru_victim(&st.slots, protect) else {
                return Ok(None);
            };
            st.slots[slot].expert = Some(expert_id);
            st.slots[slot].buf = None;
            slot
        };
        self.load_reserved(expert_id, slot, sync)
    }

    /// Copy `expert_id` into `slot`, which the caller has reserved, then publish it. A failed
    /// copy frees the slot.
    fn load_reserved(
        &self,
        expert_id: usize,
        slot: usize,
        sync: bool,
    ) -> candle_core::Result<Option<usize>> {
        // One critical section against the borrow: the gate check, the copy and the publish all happen under
        // `lock_access`. Without this the *publish* was the hole - a fill that queued its copy before the borrow
        // raised the flag could publish `slot.buf` afterwards, re-marking a slot whose bytes the chunk owns (which
        // is why the defect came back after a rebuild). Lock order is `access -> state`, the same as the borrow's.
        let _access = super::slab::lock_access();
        if super::slab::borrowed() {
            self.state.lock().unwrap().slots[slot].expert = None;
            return Ok(None);
        }
        let loaded = match &self.copy_stream {
            Some(cs) => self.upload_in_place(cs, expert_id, slot, sync),
            None => self.expert_weights(expert_id).map(|w| {
                w.map(|proj| {
                    Arc::new(SlotBuf {
                        data: SlotData::Owned(proj),
                        fence: ReadFence::none(),
                    })
                })
            }),
        };
        let mut st = self.state.lock().unwrap();
        // The borrow could have raised the flag while the copy was in flight; `_access` has been held throughout,
        // so it cannot have - but check anyway, because publishing into a slot the chunk owns is the failure this
        // whole gate exists for.
        if super::slab::borrowed() {
            st.slots[slot].expert = None;
            return Ok(None);
        }
        match loaded {
            // Lent while this upload ran: drop it rather than publish into a lent slot.
            Ok(Some(_)) if slot >= st.usable => {
                st.slots[slot].expert = None;
                Ok(None)
            }
            Ok(Some(buf)) => {
                let tick = st.tick;
                let s = &mut st.slots[slot];
                s.buf = Some(buf);
                s.last_use = tick;
                Ok(Some(slot))
            }
            Ok(None) => {
                st.slots[slot].expert = None;
                Ok(None)
            }
            Err(e) => {
                st.slots[slot].expert = None;
                Err(e)
            }
        }
    }

    /// Overwrite `slot`'s device memory with `expert_id` on the copy stream. The slot is
    /// reserved and unpublished, so no new reader can reach it; this waits for the readers
    /// that captured it earlier to record their fence, then the copy waits on that fence.
    fn upload_in_place(
        &self,
        cs: &CopyStream,
        expert_id: usize,
        slot: usize,
        sync: bool,
    ) -> candle_core::Result<Option<Arc<SlotBuf>>> {
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(None);
        };
        // Clone the per-slot storage under the state lock, then take its own lock for the whole
        // upload. Lending this slot waits for the upload to finish.
        let storage = {
            let st = self.state.lock().unwrap();
            Arc::clone(&st.slots[slot].storage)
        };
        let entry = storage.lock();
        let Some(buf) = entry.as_ref() else {
            return Ok(None); // lent
        };
        let SlotData::Banked { memory, .. } = &buf.data else {
            return Ok(None);
        };
        copy_stream::wait_for_readers(buf);
        let (g, u, d) = (gate.data()?, up.data()?, down.data()?);
        let parts = [
            (&memory[0], &g[expert_range(&gate, expert_id)?]),
            (&memory[1], &u[expert_range(&up, expert_id)?]),
            (&memory[2], &d[expert_range(&down, expert_id)?]),
        ];
        // The caller (`load_reserved`) holds `lock_access` across this check and the copies below, so the borrow
        // cannot take the range in between. No lock is taken here: it is not reentrant.
        if super::slab::borrowed() {
            // A borrowed chunk owns the slab's low end right now: this slot may *be* part of the chunk's buffer, so
            // the upload must not happen at all. The slot keeps its reservation and has no published buffer, so the
            // hit path treats it as a miss.
            if std::env::var_os("NAIWA_BORROW_CHECK").is_some() {
                eprintln!("UPLOAD_REFUSED layer {} expert {expert_id}", self.layer);
            }
            return Ok(None);
        }
        if sync {
            cs.upload(&buf.fence, &parts)?;
        } else {
            // The caller records one completion event after the whole batch and claims it, so the
            // event — not a host wait — orders the reader. Blocking here was 81% of a prediction
            // job.
            // SAFETY: the sources are page-locked stacks that outlive the copies, and the caller
            // records the batch's completion on the same stream.
            unsafe { cs.queue_async(&buf.fence, &parts)? };
        }
        Ok(Some(Arc::clone(buf)))
    }

    /// This layer's predicted experts for `rows` activations `x` (`[rows, hidden]`), by its own
    /// router: the union of each row's top-k, the experts most rows want first.
    ///
    /// R2a: the router lives on the CPU and `x` arrives as a CPU tensor (read from the layer's
    /// mapped channel in the lookahead job), so the matmul never touches the GPU stream — no
    /// `to_device`, no `dev_synchronize`. The W92-W99 stream-position hazards (prediction
    /// queuing behind the calling layer's tail kernels, late arrivals halving the hit rate)
    /// cannot occur: the activation is published by a stream-ordered kernel the moment the
    /// layer's compute is done.
    fn predict(&self, x: &Tensor) -> candle_core::Result<Vec<usize>> {
        let Some(router) = &self.router else {
            return Ok(Vec::new());
        };
        let rows = x.dims().first().copied().unwrap_or(0);
        let x = x.to_dtype(DType::F32)?.reshape((rows, self.hidden))?;
        let logits = x.matmul(&router.t()?)?;
        Ok(union_by_votes(
            logits
                .to_vec2::<f32>()?
                .iter()
                .map(|row| top_k_ids(row, self.top_k)),
        ))
    }

    /// Turn the prediction for `x` into **asynchronous** copies on the copy stream, record one
    /// completion event after them, and claim the pass with that event — all from the prediction
    /// thread, in the gap after the CPU-expert batch, so the DMA hides in the host's work for the
    /// next layer.
    ///
    /// The event is recorded *before* the claim is visible, so a consumer that sees the claim
    /// always has a completion signal to wait on. A consumer that skips a pass (no new claim)
    /// waits for the previous claim, which is exactly what keeps an in-flight copy from being read.
    pub(crate) fn prefetch_predicted(&self, x: &Tensor) -> candle_core::Result<()> {
        let rows = x.dims().first().copied().unwrap_or(0);
        let predicted = self.predict(x)?;
        let Some(cs) = &self.copy_stream else {
            return Ok(());
        };
        let max_swaps = self.lookahead_swaps * rows;
        let mut swaps = 0;
        for &e in &predicted {
            if e >= self.num_experts {
                continue;
            }
            if self.admit(e, false)?.is_some() {
                stats::record_cache_lookahead_load();
                continue;
            }
            if swaps < max_swaps && self.replace(e, &predicted, false)?.is_some() {
                swaps += 1;
                stats::record_cache_swap();
            }
        }
        // Record the batch's completion, then claim: a reader that sees the claim has an event.
        if let Some(cell) = &self.preload {
            cell.claim(cs)?;
        }
        Ok(())
    }

    /// The next expert to fill into this layer, if a slot is still empty.
    pub(super) fn next_fill_candidate(&self) -> Option<usize> {
        let st = self.state.lock().unwrap();
        fill_candidate(&st.slots[..st.usable], &st.hit_counts)
    }

    /// Fill every empty slot of this layer on a background thread and return immediately.
    /// No-op when the layer is full or a fill is already running.
    pub(super) fn spawn_fill(self: &Arc<Self>) {
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
                // Background fill: synchronous copies, no per-pass gate protects these slots.
                match this.admit(e, true) {
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
}
