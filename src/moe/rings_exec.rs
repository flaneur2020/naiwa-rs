//! Ring batching for prompt chunks: run a layer's routed experts in rounds over the slots the layer owns
//! (and, once a lender is wired, over slots lent by same-shape layers).
//!
//! The round plan comes from [`super::rings::plan`]. This module executes it: for each round the experts are
//! DMA'd from the pinned host stacks into the layer's slots, the round's `(row, expert)` pairs are computed with
//! the same fused kernel the decode hits use, and the rounds' outputs are summed. Every pair is computed exactly
//! once, so the result is the same sum as one big pass, up to floating-point order.
//!
//! `NAIWA_PREFILL_RINGS=1` turns it on; without the variable a chunk takes the prefill stream or the CPU as
//! before.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use mistralrs_core::layers::Activation;

use super::cached_layer::CachedLayer;
use super::expert::expert_range;
use super::rings::{self, RoundSlots};
use super::slot::{SlotBuf, SlotData};
use super::stages::{Gpu, Hit};
use crate::copy_stream::{CopyDone, CopyStream};

/// Whether a chunk should take the ring path. Off by default.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NAIWA_PREFILL_RINGS").is_some())
}

/// One rented slot: the buffer to DMA into and the slot index the kernel addresses.
struct Rented {
    buf: Arc<SlotBuf>,
    slot: usize,
}

/// `xs: [batch, seq, hidden]` on the slot device. `Ok(None)` when this layer cannot take the ring path, in which
/// case the caller falls back to the prefill stream and then the CPU.
pub(crate) fn forward(
    layer: &Arc<CachedLayer>,
    xs: &Tensor,
    topk_weights: &Tensor,
    topk_ids: &Tensor,
) -> candle_core::Result<Option<Tensor>> {
    if !enabled() || !layer.slot_device.is_cuda() {
        return Ok(None);
    }
    let Some(copy) = layer.copy_stream.as_ref() else {
        return Ok(None);
    };
    let Some((gate, up, down)) = layer.cpu.expert_qtensors() else {
        return Ok(None);
    };
    let (batch, seq, hidden) = xs.dims3()?;
    let rows = batch * seq;
    if rows <= 1 {
        return Ok(None); // decode: the ordinary split is cheaper
    }

    // Rent the layer's own slots. `usable` is what the layer is allowed to use; the rest stays for decode.
    //
    // The `Arc` is **taken out of the slot** rather than cloned: `wait_for_readers` spins until the refcount is
    // 1, so a clone kept alive by the slot would spin forever (the W105 hang this codebase already paid for
    // once). Taking it also makes the cache unable to serve the slot while the round owns it, which is what we
    // want.
    let rented: Vec<Rented> = {
        let mut state = layer.state.lock().unwrap();
        let usable = state.usable.min(state.slots.len());
        let mut out = Vec::with_capacity(usable);
        for index in 0..usable {
            let slot = &mut state.slots[index];
            let taken = slot.storage.lock().take();
            let Some(buf) = taken else {
                continue;
            };
            // Everything that could keep a reference alive must be dropped **before** the wait: the storage
            // (taken above), the published contents (`buf`) and the expert label. Only then is the refcount
            // this function's own, which is what `wait_for_readers` waits for.
            slot.expert = None;
            slot.buf = None;
            crate::copy_stream::wait_for_readers(&buf);
            out.push(Rented { buf, slot: index });
        }
        out
    };
    if rented.is_empty() {
        return Ok(None);
    }

    // Routing to the host: `rows x top_k` u32 ids and f32 weights.
    let ids = topk_ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
    let weights = topk_weights.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
    let mut entries: Vec<(usize, u32, f32)> = Vec::with_capacity(rows * layer.top_k);
    for (row, (row_ids, row_weights)) in ids.iter().zip(&weights).enumerate() {
        for (expert, weight) in row_ids.iter().zip(row_weights) {
            entries.push((row, *expert, *weight));
        }
    }

    // Plan with the layer's own slots as the ring. A lender would add its slots here.
    let plan = rings::plan(&entries, rented.len(), 0, None);
    let plan_rounds = plan.rounds.len();

    let g = gate.data()?;
    let u = up.data()?;
    let d = down.data()?;
    let x2d = xs.reshape((rows, hidden))?;
    let act: Activation = layer.act;
    let mut sum: Option<Tensor> = None;
    let mut computed = 0usize;
    for round in &plan.rounds {
        debug_assert!(matches!(round.slots, RoundSlots::Own), "no lender is wired yet");
        if round.experts.is_empty() {
            continue;
        }
        // DMA this round's experts into the rented slots, in order.
        let mut slot_of: HashMap<u32, usize> = HashMap::with_capacity(round.experts.len());
        let done = CopyDone::new(copy)?;
        for (i, expert) in round.experts.iter().enumerate() {
            let held = &rented[i % rented.len()];
            let SlotData::Banked { memory, .. } = &held.buf.data else {
                unreachable!("cache slots are banked")
            };
            let e = *expert as usize;
            let er = expert_range(&gate, e)?;
            let upper = expert_range(&up, e)?;
            let lower = expert_range(&down, e)?;
            // SAFETY: the stacks are page-locked and outlive the copies (`layer.cpu()` holds them).
            unsafe {
                copy.queue_async(
                    &held.buf.fence,
                    &[
                        (&memory[0], &g[er]),
                        (&memory[1], &u[upper]),
                        (&memory[2], &d[lower]),
                    ],
                )?;
            }
            slot_of.insert(*expert, held.slot);
            computed += 1;
        }
        done.record(copy)?;
        done.wait_on(&layer.slot_device)?;

        // Per row, the round's hits with their weights, addressed inside the slot buffers' banks.
        let mut hits: Vec<Vec<Hit>> = vec![Vec::with_capacity(layer.top_k); rows];
        for (row, experts) in &round.rows {
            for (expert, weight) in experts {
                let slot = slot_of[expert];
                let held = &rented[slot_index_of(&rented, slot)];
                hits[*row].push((Arc::clone(&held.buf), slot, *weight));
            }
        }
        let (out, _) = Gpu::submit(layer, &hits, &x2d, act)?;
        sum = Some(match sum {
            Some(acc) => acc.add(&out)?,
            None => out,
        });
    }

    crate::dbg_note(
        "prefill_rings",
        &format!(
            "layer {} {rows} rows: {plan_rounds} rounds over {} slots, {computed} expert DMA",
            layer.layer,
            rented.len()
        ),
    );

    // Give the slots back: their contents were overwritten by the rounds, so they come back empty and the
    // cache refills them as it would after any eviction.
    {
        let mut state = layer.state.lock().unwrap();
        for rent in &rented {
            let slot = &mut state.slots[rent.slot];
            *slot.storage.lock() = Some(Arc::clone(&rent.buf));
            slot.expert = None;
            slot.buf = None;
        }
    }

    match sum {
        Some(sum) => Ok(Some(sum.reshape((batch, seq, hidden))?.to_dtype(xs.dtype())?)),
        None => Ok(None),
    }
}

/// The index of the rented slot with plain slot number `slot`.
fn slot_index_of(rented: &[Rented], slot: usize) -> usize {
    rented
        .iter()
        .position(|r| r.slot == slot)
        .expect("a hit was built for a slot that was not rented")
}
