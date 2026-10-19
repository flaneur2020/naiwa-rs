//! `CachedLayer` lifecycle: slots, fills, LRU replacement, lookahead prediction.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::MoEExperts;
use tracing::debug;

use crate::copy_stream::{self, CopyStream, ReadFence};
use crate::stats;

use super::{
    alloc_bank, expert_qtensor_on_device, expert_range, schedule, CachedLayer, LayerRegistry,
    LayerState, Projections, Slot, SlotBuf, SlotData,
};

/// R2a: the lookahead's per-layer publish channel (GPU -> host). At each MoE entry the
/// token thread queues ONE kernel that copies this step's activation into mapped memory
/// and bumps seq; the lookahead worker spins for the seq and runs the router matmul on
/// the CPU (the router weights moved back to the host, `[128, 2048]` F32 ~ 1 MiB/layer).
/// The token thread never waits — no `to_vec1` D2H (W105b: 0.30-0.34 ms/layer of blocking
/// wait) and no way for the prediction to queue late on the stream (the W99 failure that
/// made an earlier mode switch regress hit 74% -> 38-45%).
pub(crate) struct LookaheadChannel {
    pub(crate) chan: Arc<crate::doorbell::ActivationDoorbell>,
    /// The seq the last queued GPU publish will bump the counter to (0 before any):
    /// `last_seq + 1` is the value `wait_publish` waits for. Touched only by the token thread.
    pub(crate) next_seq: std::sync::atomic::AtomicU32,
    /// How many queued publishes the worker has consumed (`wait_publish` returned).
    /// The difference to `next_seq` is the queue depth; publish backpressure keeps
    /// it below the doorbell's ring depth so a lagging job's slot is never overwritten.
    pub(crate) consumed: std::sync::atomic::AtomicU32,
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

    pub(crate) fn registry(&self) -> &Arc<LayerRegistry> {
        &self.registry
    }

    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    pub(crate) fn preload_claimed(&self) -> u32 {
        self.preload_claimed.load(Ordering::Acquire)
    }
    /// v2 M4: is the PCIe staging path in service for this layer? One predicate for all three
    /// sites — capture the DMA gate, publish the point, replay the gate — because they must agree:
    /// a gate whose point is never published spins forever, and at `--pcie-frac 0` the copy stream
    /// still exists (the cache banks use it) while nothing is ever staged.
    #[cfg(feature = "cuda")]
    pub(crate) fn staging_enabled(&self) -> bool {
        self.pcie_frac > 0.0 && self.copy_stream.is_some()
    }

    /// M4/M5b/M6: gate this layer's segment on `kind` by launching the wait node on the stream.
    ///
    /// M6 folds the gate graphs away: the node was already launched warm whenever capture failed,
    /// and a plain launch is *cheaper* than replaying a one-node graph (measured 5.9 us vs 15.4 us
    /// per launch), so a captured gate buys nothing. Stream ordering is identical.
    #[cfg(feature = "cuda")]
    pub(crate) fn replay_gate(&self, kind: u32) -> candle_core::Result<()> {
        let Some(sched) = schedule() else {
            return Ok(());
        };
        stats::record_cache_gate_replay();
        let id = sched.sync_id(self.layer as u32, kind);
        let gate: crate::doorbell::Gate<'_> = sched.gate(id);
        gate.wait(&self.slot_device)
    }

    /// v2 M1: create this layer's staging ring now (see `GpuCacheMoeProvider::warmup_rings`).
    /// A no-op when the layer has no copy stream or no CPU `QTensor`s. Called once after load.
    pub(super) fn warmup_scratch_ring(&self) -> candle_core::Result<()> {
        let Some(cs) = &self.copy_stream else {
            return Ok(());
        };
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(());
        };
        self.registry.scratch_ring(
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

    /// Put `expert_id` into an empty slot. The slot is reserved under the lock before the
    /// (slow) copy and published only after it, so concurrent admitters never share a slot
    /// and readers never see a half-written one. `None`: already resident or no empty slot.
    pub(crate) fn admit(&self, expert_id: usize, sync: bool) -> candle_core::Result<Option<usize>> {
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
        if sync {
            cs.upload(&buf.fence, &parts)?;
        } else {
            // M5b: the caller claims this pass and publishes `SYNC_PRELOAD` on this same stream right
            // after the copies, so the layer's gate — not a host wait — orders the reader. Blocking
            // here was 81% of a prediction job.
            // SAFETY: the sources are page-locked stacks that outlive the copies, and the caller
            // publishes the completion point on the same stream.
            unsafe { cs.queue_async(&buf.fence, &parts)? };
        }
        Ok(Some(Arc::clone(buf)))
    }

    /// Bytes of one slot: one expert's gate, up and down.
    pub(super) fn slot_bytes(&self) -> usize {
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return 0;
        };
        [&gate, &up, &down]
            .iter()
            .map(|s| expert_range(s, 0).map_or(0, |r| r.len()))
            .sum()
    }

    /// Take at least the top `k` usable slots out of service for the prefill stream (whole banks
    /// in copy-stream mode): unpublish and empty them, then release their device memory. A free
    /// is ordered on candle's stream after any kernel already reading the slot. Returns the
    /// number of slots lent.
    pub(super) fn lend(&self, k: usize) -> usize {
        let (from, to, storages) = {
            let mut st = self.state.lock().unwrap();
            let to = st.usable;
            let want = to.saturating_sub(k);
            // A bank's memory is one allocation: lend from the start of the bank holding `want`.
            let from = self
                .banks
                .iter()
                .map(|&(first, _)| first)
                .filter(|&first| first <= want)
                .max()
                .unwrap_or(want);
            for slot in from..to {
                let s = &mut st.slots[slot];
                s.expert = None;
                s.buf = None;
            }
            st.usable = from;
            let storages = st.slots[from..to]
                .iter()
                .map(|s| Arc::clone(&s.storage))
                .collect::<Vec<_>>();
            (from, to, storages)
        };
        for storage in storages {
            // Waits for an upload into the slot; its result is then discarded.
            if let Some(buf) = storage.lock().take() {
                copy_stream::wait_for_readers(&buf);
            }
        }
        to - from
    }

    /// Put every lent slot back in service, empty (with fresh device memory in copy-stream
    /// mode), and refill the layer in the background by routing count.
    pub(super) fn give_back(self: &Arc<Self>) -> candle_core::Result<()> {
        let (from, to, storages) = {
            let st = self.state.lock().unwrap();
            let from = st.usable;
            let to = st.slots.len();
            let storages = st.slots[from..to]
                .iter()
                .map(|s| Arc::clone(&s.storage))
                .collect::<Vec<_>>();
            (from, to, storages)
        };
        if from == to {
            return Ok(());
        }
        if let (Some(cs), Some(stacks)) = (&self.copy_stream, self.cpu.expert_qtensors()) {
            for &(first, len) in self.banks.iter().filter(|&&(first, _)| first >= from) {
                let bufs = alloc_bank(cs, &stacks, first, len, &self.slot_device)?;
                for (i, buf) in bufs.into_iter().enumerate() {
                    *storages[first - from + i].lock() = Some(buf);
                }
            }
            cs.after_device_work()?;
        }
        self.state.lock().unwrap().usable = to;
        self.spawn_fill();
        Ok(())
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

    /// M5b: turn the prediction for `x` into **asynchronous** copies on the copy stream, then claim
    /// this pass and publish `SYNC_PRELOAD` for it — all from the prediction thread, in the gap after
    /// the CPU-expert batch, so the DMA hides in the host's work for the next layer.
    ///
    /// The copies are queued *before* the claim is published, and the publish is monotonic
    /// (`Gate::publish_stream_max`), so the token thread either sees no claim (nothing to wait for) or
    /// sees one whose point is already on its way and can only move the word forward.
    pub(crate) fn prefetch_predicted(&self, x: &Tensor, pass: u32) -> candle_core::Result<()> {
        let rows = x.dims().first().copied().unwrap_or(0);
        let predicted = self.predict(x)?;
        if self.copy_stream.is_none() {
            return Ok(());
        }
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
        // Claim then publish: a reader that sees `pass` knows the point is coming.
        if let Some(sched) = schedule() {
            self.preload_claimed.store(pass, Ordering::Release);
            let id = sched.sync_id(self.layer as u32, crate::doorbell::SYNC_PRELOAD);
            sched.gate(id).publish_stream_max(&self.slot_device, pass)?;
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
