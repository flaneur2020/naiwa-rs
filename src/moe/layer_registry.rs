//! Model-wide `Layers` registry and prefill-borrow slot emptying / refill coordination.

use std::sync::{Arc, Mutex, Weak};

use parking_lot::RwLock;
use tracing::{debug, warn};

use super::slab::{BorrowGate, Slab};
use super::{CachedLayer, SlotBuf, SlotData};
use crate::copy_stream::CopyStream;

/// Every cached layer of one model, by decoder-layer index.
/// Weak references: the model owns the layers, not this index.
pub(crate) struct Layers {
    layers: RwLock<Vec<Weak<CachedLayer>>>,
    /// The loan's ledger: every `(layer, expert)` the prompt path displaced, in the order it took them.
    ///
    /// A loan is not a loss: the borrow takes the bytes of whichever slots overlap it, and this
    /// remembers what they held so the first decode step can put it back.
    lent: Mutex<Vec<(usize, usize)>>,
}

impl Default for Layers {
    fn default() -> Self {
        Self {
            layers: RwLock::new(Vec::new()),
            lent: Mutex::new(Vec::new()),
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

    #[inline]
    pub(crate) fn get(&self, layer: usize) -> Option<Arc<CachedLayer>> {
        self.layers.read().get(layer).and_then(Weak::upgrade)
    }

    /// Transition from Decode into a Prefill chunk that borrows `[0, loaned)` of `slab`:
    /// quiesce lookahead, raise `borrow_gate`, order `copy` after prior GPU work, zero the span,
    /// empty overlapping cache slots, and synchronize prior fills.
    pub(crate) fn enter_prefill_loan(
        &self,
        borrow_gate: &BorrowGate,
        copy: &CopyStream,
        slab: &Slab,
        stride: usize,
        loaned: usize,
    ) -> candle_core::Result<usize> {
        super::quiesce(true);
        {
            let _guard = borrow_gate.lock_access();
            borrow_gate.set_borrowed(true);
        }
        let base = slab.base()?;
        copy.after_device_work()?;
        let stream = copy.cuda_stream();
        let zeroed = loaned.max(650usize << 20).min(slab.bytes());
        // SAFETY: the span is inside the live slab; memset is queued before any upload.
        unsafe {
            let rc = candle_core::cuda::cudarc::driver::sys::cuMemsetD8Async(
                base,
                0,
                zeroed,
                stream.cu_stream(),
            );
            if rc != candle_core::cuda::cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                candle_core::bail!("memset of the borrowed span failed: {rc:?}");
            }
        }
        let emptied = self.empty_range(base, base + loaned as u64);
        copy.synchronize()?;
        crate::dbg_note(
            "prefill_borrow",
            &format!(
                "chunk: borrowed {:.0} MiB of the pool in {stride} range(s), {emptied} slots emptied",
                (loaned as f64) / (1u64 << 20) as f64
            ),
        );
        Ok(emptied)
    }

    /// Transition back into Decode mode: release `borrow_gate`, resume lookahead at layer 0, and
    /// refill any displaced loan slots.
    #[inline]
    pub(crate) fn enter_decode(&self, borrow_gate: &BorrowGate, is_layer_0: bool) {
        borrow_gate.set_borrowed(false);
        if is_layer_0 {
            super::quiesce(false);
            if self.refill_lent() > 0 {
                crate::dbg_note("prefill_borrow", "the loan was given back");
            }
        }
    }

    /// Empty every slot whose bank overlaps `[base, end)`, and report how many were emptied.
    ///
    /// Three-phase protocol to respect lock ordering (`access -> lent -> state -> storage`, and
    /// `wait_for_readers` never under `state`):
    /// 1. Under `state` + `storage`, take overlapping slot buffers out and clear metadata.
    /// 2. With all locks released, `wait_for_readers` on every taken buffer.
    /// 3. Return the buffers to their slots and clear metadata a second time so an in-flight fill
    ///    that raced phase 1 cannot leave a stale reservation pointing at overwritten bytes.
    pub(crate) fn empty_range(&self, base: u64, end: u64) -> usize {
        let layers = self.all();
        let mut ledger = self.lent.lock().unwrap();
        ledger.clear();
        let mut held: Vec<(usize, usize, Arc<SlotBuf>)> = Vec::new();
        let mut skipped = 0usize;
        for (li, layer) in layers.iter().enumerate() {
            let mut state = layer.cache.state.lock().unwrap();
            for (si, slot) in state.slots.iter_mut().enumerate() {
                let overlaps = {
                    let storage = slot.storage.lock();
                    let Some(buf) = storage.as_ref() else {
                        if slot.expert.is_some() {
                            skipped += 1;
                        }
                        continue;
                    };
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
                if let Some(expert) = slot.expert {
                    ledger.push((li, expert));
                }
                slot.expert = None;
                slot.buf = None;
                held.push((li, si, buf));
            }
        }
        // Phase 2: wait for readers/fills with no lock held.
        for (_, _, buf) in &held {
            crate::copy_stream::wait_for_readers(buf);
        }
        let emptied = held.len();
        if skipped > 0 {
            eprintln!("BORROW_SKIPPED {skipped} slots with no storage while resident");
        }
        let mut still = 0usize;
        for layer in self.all() {
            let state = layer.cache.state.lock().unwrap();
            for slot in state.slots.iter() {
                let overlaps = {
                    let storage = slot.storage.lock();
                    let Some(buf) = storage.as_ref() else {
                        continue;
                    };
                    let SlotData::Banked { memory, .. } = &buf.data else {
                        continue;
                    };
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
        // Phase 3: restore storage buffers and re-clear metadata.
        for (li, si, buf) in held {
            let mut state = layers[li].cache.state.lock().unwrap();
            let slot = &mut state.slots[si];
            slot.expert = None;
            slot.buf = None;
            slot.last_use = 0;
            *slot.storage.lock() = Some(buf);
        }
        emptied
    }

    /// Put back what the loan displaced: one queued copy per `(layer, expert)`, followed by one
    /// copy-stream synchronize before decode reads the refilled slots.
    pub(crate) fn refill_lent(&self) -> usize {
        let lent: Vec<(usize, usize)> = {
            let mut ledger = self.lent.lock().unwrap();
            std::mem::take(&mut *ledger)
        };
        if !crate::config::runtime().refill_lent {
            if !lent.is_empty() {
                tracing::info!(
                    "expert cache: the loan is given back, 0 of {} displaced experts refilled (NAIWA_REFILL=0)",
                    lent.len()
                );
            }
            return 0;
        }
        let total = lent.len();
        let mut asked = 0;
        for (li, expert) in lent {
            let Some(layer) = self.get(li) else { continue };
            match layer.admit(expert, false) {
                Ok(Some(_)) => asked += 1,
                Ok(None) => {}
                Err(e) => debug!("expert cache: refilling layer {li} expert {expert} failed: {e}"),
            }
        }
        if total > 0 {
            if asked > 0 {
                if let Some(cs) = self
                    .all()
                    .iter()
                    .find_map(|layer| layer.copy_stream().map(Arc::clone))
                {
                    if let Err(e) = cs.synchronize() {
                        warn!("expert cache: waiting for the refill's copies failed: {e}");
                    }
                }
            }
            tracing::info!(
                "expert cache: the loan is given back, {asked} of {total} displaced experts queued for refill"
            );
        }
        asked
    }

    pub(crate) fn all(&self) -> Vec<Arc<CachedLayer>> {
        self.layers
            .read()
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }
}
