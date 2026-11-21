//! Cache mutation, promotion, lookahead prediction, and background fill paths on [`CachedLayer`].

use std::sync::atomic::Ordering;
use std::sync::Arc;

use candle_core::{DType, Tensor};
use tracing::debug;

use crate::copy_stream::{self, CopyStream, ReadFence};
use crate::stats;

use super::cache_policy::{fill_candidate, lru_victim, top_k_ids, union_by_votes};
use super::{
    expert_qtensor_on_device, expert_range, CachedLayer, Projections, SlotBuf, SlotData,
};

impl CachedLayer {
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

    #[inline]
    fn promote_cap() -> usize {
        crate::config::runtime().promote_max
    }

    /// Whether promotion is switched on at all.
    #[inline]
    pub(crate) fn promoting(&self) -> bool {
        Self::promote_cap() > 0
    }

    /// Promote this step's staged experts into the cache, up to `NAIWA_PROMOTE_MAX` per layer per step.
    pub(crate) fn promote_staged(
        &self,
        candle_stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    ) -> candle_core::Result<usize> {
        let cap = Self::promote_cap();
        // Always drain, even with the cap at zero: these `Arc`s must not survive the step.
        // Sort by expert id so promotion is a deterministic function of demand.
        let mut staged = std::mem::take(&mut *self.staged_now.lock().unwrap());
        staged.sort_unstable_by_key(|(id, _)| *id);
        if crate::config::diag().promote_trace {
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
        // Never block the token thread here: the lookahead holds `access` while waiting for our readers.
        let Some(_access) = self.borrow_gate.try_lock_access() else {
            return Ok(0);
        };
        if self.borrow_gate.borrowed() {
            return Ok(0);
        }
        let min_sightings = crate::config::runtime().promote_min_hits;
        let mut promoted = 0usize;
        let mut skipped = 0usize;
        let mut cold = 0usize;
        let mut seen: Vec<u32> = Vec::with_capacity(cap);
        for (expert, src_buf) in staged.iter() {
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
                // Empty slot first, then LRU with `last_use < tick` (never a slot this step touched).
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
                    (
                        SlotData::Banked { memory: dmem, .. },
                        SlotData::Banked { memory: smem, .. },
                    ) => {
                        for (d, s2) in [
                            (&dmem[0], &smem[0]),
                            (&dmem[1], &smem[1]),
                            (&dmem[2], &smem[2]),
                        ] {
                            if s2.len() < d.len() {
                                candle_core::bail!(
                                    "promotion of {} bytes into a {}-byte slot",
                                    s2.len(),
                                    d.len()
                                );
                            }
                            // SAFETY: both are live device allocations; copy is stream-ordered on candle's stream.
                            unsafe {
                                candle_core::cuda_backend::cudarc::driver::result::memcpy_dtod_async(
                                    d.ptr(),
                                    s2.ptr(),
                                    d.len(),
                                    candle_stream.cu_stream(),
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
                    crate::stats::schedule::detail(
                        self.layer,
                        tick,
                        "promotion_publish",
                        0,
                        Some(expert),
                        Some(slot),
                    );
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
        if crate::config::diag().promote_trace {
            eprintln!(
                "PROMOTE layer {} staged {} promoted {} skipped {} cold {} filled {}",
                self.layer,
                staged.len(),
                promoted,
                skipped,
                cold,
                self.state
                    .lock()
                    .unwrap()
                    .slots
                    .iter()
                    .filter(|s| s.expert.is_some())
                    .count()
            );
        }
        Ok(promoted)
    }

    /// Put `expert_id` into an empty slot. The slot is reserved under the lock before the copy
    /// and published only after it. `None`: already resident or no empty slot.
    pub(crate) fn admit(&self, expert_id: usize, sync: bool) -> candle_core::Result<Option<usize>> {
        if self.borrow_gate.borrowed() {
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

    /// Replace the least-recently-used resident outside `protect` with `expert_id`.
    pub(crate) fn replace(
        &self,
        expert_id: usize,
        protect: &[usize],
        sync: bool,
    ) -> candle_core::Result<Option<usize>> {
        if self.borrow_gate.borrowed() {
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

    /// Copy `expert_id` into reserved `slot` under `borrow_gate.lock_access()`, then publish.
    fn load_reserved(
        &self,
        expert_id: usize,
        slot: usize,
        sync: bool,
    ) -> candle_core::Result<Option<usize>> {
        // Lock order: `access -> state -> storage`.
        let _access = self.borrow_gate.lock_access();
        if self.borrow_gate.borrowed() {
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
        if self.borrow_gate.borrowed() {
            st.slots[slot].expert = None;
            return Ok(None);
        }
        match loaded {
            Ok(Some(_)) if slot >= st.usable => {
                st.slots[slot].expert = None;
                Ok(None)
            }
            Ok(Some(buf)) => {
                let tick = st.tick;
                let s = &mut st.slots[slot];
                s.buf = Some(buf);
                s.last_use = tick;
                crate::stats::schedule::detail(
                    self.layer,
                    tick,
                    "fill_publish",
                    0,
                    Some(expert_id),
                    Some(slot),
                );
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

    /// Overwrite `slot`'s device memory with `expert_id` on the copy stream.
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
        if self.borrow_gate.borrowed() {
            return Ok(None);
        }
        if sync {
            cs.upload(&buf.fence, &parts)?;
        } else {
            // SAFETY: sources are page-locked stacks outliving the copies; caller records batch completion.
            unsafe { cs.queue_async(&buf.fence, &parts)? };
        }
        if crate::stats::schedule::enabled() {
            let bytes = parts.iter().map(|(_, src)| src.len() as u64).sum();
            let tick = self.state.lock().unwrap().tick;
            crate::stats::schedule::detail(
                self.layer,
                tick,
                "cache_fill_h2d",
                bytes,
                Some(expert_id),
                Some(slot),
            );
        }
        Ok(Some(Arc::clone(buf)))
    }

    /// Predict this layer's routed experts for CPU activation `x` (`[rows, hidden]`).
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

    /// Queue asynchronous copies for predicted experts, record one completion event, and claim it.
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
