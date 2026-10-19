//! Unit tests for the expert cache (see `super`).

use super::*;
use super::{fill_candidate, lru_victim, top_k_ids, union_by_votes};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::MoEExperts;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crate::copy_stream::ReadFence;

fn slot(expert: Option<usize>, last_use: u64) -> Slot {
    let mut s = Slot::empty();
    s.expert = expert;
    s.last_use = last_use;
    s
}

fn slots(experts: &[Option<usize>]) -> Vec<Slot> {
    experts.iter().map(|&e| slot(e, 0)).collect()
}

fn published_slot(expert: usize, last_use: u64) -> Slot {
    let qt = Arc::new(stacked(1, 1, 32));
    let proj = Projections {
        gate: Arc::clone(&qt),
        up: Arc::clone(&qt),
        down: qt,
    };
    let mut s = slot(Some(expert), last_use);
    s.buf = Some(Arc::new(SlotBuf {
        data: SlotData::Owned(proj),
        fence: ReadFence::none(),
    }));
    s
}

/// A stacked `[E, N, K]` Q8_0 tensor whose experts hold distinct values.
fn stacked(e: usize, n: usize, k: usize) -> QTensor {
    let data: Vec<f32> = (0..e * n * k)
        .map(|i| (i % 97) as f32 * 0.01 - 0.4)
        .collect();
    let t = Tensor::from_vec(data, (e, n, k), &Device::Cpu).unwrap();
    QTensor::quantize(&t, GgmlDType::Q8_0).unwrap()
}

#[test]
fn expert_ranges_tile_the_stack_exactly() {
    let (e, n, k) = (4, 8, 64);
    let qt = stacked(e, n, k);
    let all = qt.data().unwrap().into_owned();
    let stride = all.len() / e;
    let mut rebuilt = Vec::new();
    for i in 0..e {
        let r = expert_range(&qt, i).unwrap();
        assert_eq!(r, i * stride..(i + 1) * stride);
        rebuilt.extend_from_slice(&all[r]);
    }
    assert_eq!(rebuilt, all);
    assert!(expert_range(&qt, e).is_err());
}

#[test]
fn single_expert_qtensor_matches_its_slice_of_the_stack() {
    let (e, n, k) = (3, 4, 32);
    let qt = stacked(e, n, k);
    let one = expert_qtensor_on_device(&qt, 1, &Device::Cpu).unwrap();
    assert_eq!(one.shape().dims(), &[n, k]);
    assert_eq!(one.dtype(), GgmlDType::Q8_0);
    let full = qt.dequantize(&Device::Cpu).unwrap().get(1).unwrap();
    let got = one.dequantize(&Device::Cpu).unwrap();
    let diff = (full - got).unwrap().abs().unwrap().sum_all().unwrap();
    assert_eq!(diff.to_scalar::<f32>().unwrap(), 0.0);
}

#[test]
fn slot_bytes_count_three_projections() {
    let qt = stacked(2, 8, 64);
    let per_expert = qt.data().unwrap().len() / 2;
    assert_eq!(slot_bytes_from_qtensor(&qt), Some(3 * per_expert as u64));
}

#[test]
fn fill_candidate_prefers_hot_unreserved_experts() {
    // expert 2 is hottest but already reserved; 3 and 1 tie, lowest id wins
    let resident = [Some(2), None];
    let hits = [0, 5, 9, 5];
    assert_eq!(fill_candidate(&slots(&resident), &hits), Some(1));
    // with no history, fill in id order
    assert_eq!(fill_candidate(&slots(&[None, None]), &[0, 0, 0]), Some(0));
    // full layer: nothing to do
    assert_eq!(
        fill_candidate(&slots(&[Some(0), Some(1)]), &[3, 3, 3]),
        None
    );
    // every expert already placed
    assert_eq!(
        fill_candidate(&slots(&[Some(0), Some(1), None]), &[1, 1]),
        None
    );
}

#[test]
fn cached_moe_does_not_keep_layers_alive() {
    let cached_moe = CachedMoe::default();
    assert!(cached_moe.get(3).is_none());
    assert!(cached_moe.all().is_empty());
}

#[test]
fn union_by_votes_orders_by_votes_then_rank() {
    // 7 is in all three rankings; 3 in two; 1 and 9 once, 1 at a better rank.
    let got = union_by_votes([vec![3, 7, 9], vec![7, 3], vec![1, 7]]);
    assert_eq!(got, vec![7, 3, 1, 9]);
    assert_eq!(union_by_votes([vec![4, 2]]), vec![4, 2]);
}

#[test]
fn streamed_misses_ranks_by_total_weight_and_dedups() {
    // 3 is missed by both rows (0.4 + 0.1 = 0.5); 1 by one (0.9); 2 by one (0.2).
    let misses = vec![vec![(1u32, 0.9f32), (3, 0.4)], vec![(3, 0.1), (2, 0.2)]];
    // frac 1.0, cap 3: all three distinct experts, highest summed weight first.
    assert_eq!(streamed_misses(&misses, 1.0, 3), vec![1, 3, 2]);
    // frac 0.5 of 3 distinct -> 2 experts.
    assert_eq!(streamed_misses(&misses, 0.5, 3), vec![1, 3]);
    // cap 1 clamps below the fraction.
    assert_eq!(streamed_misses(&misses, 1.0, 1), vec![1]);
    // off, or no staging to hold anything.
    assert!(streamed_misses(&misses, 0.0, 3).is_empty());
    assert!(streamed_misses(&misses, 1.0, 0).is_empty());
}

#[test]
fn staging_ring_slots_is_fixed() {
    // Measured: 16 and 64 slots perform identically, so the ring size is a constant (M9).
    assert_eq!(staging_ring_slots(), 16);
}

#[test]
fn top_k_ids_orders_by_logit_then_index() {
    assert_eq!(top_k_ids(&[0.1, 0.9, 0.5, 0.9], 3), vec![1, 3, 2]);
    assert_eq!(top_k_ids(&[1.0, 2.0], 5), vec![1, 0]);
}

#[test]
fn lru_victim_skips_protected_and_prefers_oldest() {
    // slot 1 is oldest but holds a protected (predicted) expert
    let slots = vec![
        published_slot(10, 5),
        published_slot(11, 1),
        published_slot(12, 3),
    ];
    assert_eq!(lru_victim(&slots, &[11]), Some((12, 2)));
    assert_eq!(lru_victim(&slots, &[]), Some((11, 1)));
    assert_eq!(lru_victim(&slots, &[10, 11, 12]), None);
}

/// A cached layer on the CPU (fresh-upload mode) over small Q8_0 stacks: `e` experts,
/// `slots` slots, top-2.
fn cpu_layer(e: usize, slots: usize) -> Arc<CachedLayer> {
    let (hidden, inter) = (64, 32);
    let q = |n: usize, k: usize, seed: f32| {
        let data: Vec<f32> = (0..e * n * k)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect();
        let t = Tensor::from_vec(data, (e, n, k), &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap())
    };
    let comm = Arc::new(
        mistralrs_quant::Comm::from_device(mistralrs_quant::Id::new(), &Device::Cpu, 0, 1).unwrap(),
    );
    let cpu = MoEExperts::from_qtensors(
        q(inter, hidden, 0.1),
        q(inter, hidden, 0.7),
        q(hidden, inter, 1.3),
        e,
        2,
        &comm,
        Activation::Silu,
    )
    .unwrap();
    Arc::new(CachedLayer {
        _pinned: Vec::new(),
        cpu,
        slot_device: Device::Cpu,
        layer: 0,
        cached_moe: Arc::default(),
        hidden,
        num_experts: e,
        top_k: 2,
        act: Activation::Silu,
        router: None,
        lookahead_chan: None,
        state: Mutex::new(LayerState {
            doorbell: None,
            doorbell_seq: 0,
            slots: (0..slots).map(|_| Slot::empty()).collect(),
            hit_counts: vec![0; e],
            tick: 0,
            usable: slots,
        }),
        copy_stream: None,
        banks: Vec::new(),
        pcie_frac: 0.0,
        lookahead_swaps: 2,
        lookahead: true,
        prefill: None,
        filling: AtomicBool::new(false),
        preload_claimed: AtomicU32::new(0),
    })
}

fn wait_for_fill(layer: &CachedLayer) {
    while layer.filling.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
}

#[test]
fn lent_slots_leave_service_and_come_back_refilled() {
    let layer = cpu_layer(8, 4);
    for e in [5, 6, 7, 4] {
        assert!(layer.admit(e, true).unwrap().is_some());
    }
    assert_eq!(layer.lend(2), 2);
    {
        let st = layer.state.lock().unwrap();
        assert_eq!(st.usable, 2);
        assert_eq!(
            st.slots.iter().map(|s| s.expert).collect::<Vec<_>>(),
            vec![Some(5), Some(6), None, None]
        );
        assert!(st.slots[2].buf.is_none() && st.slots[3].buf.is_none());
        assert_eq!(st.slots.iter().filter(|s| s.buf.is_some()).count(), 2);
    }
    // no room while lent: lent slots are not admitted into, nor filled
    assert!(layer.admit(1, true).unwrap().is_none());
    assert!(layer.next_fill_candidate().is_none());

    // give back: refilled by routing count (2 and 3 are the hottest non-residents)
    layer.state.lock().unwrap().hit_counts = vec![0, 1, 9, 8, 0, 0, 0, 0];
    layer.give_back().unwrap();
    wait_for_fill(&layer);
    let st = layer.state.lock().unwrap();
    assert_eq!(st.usable, 4);
    assert_eq!(
        st.slots.iter().map(|s| s.expert).collect::<Vec<_>>(),
        vec![Some(5), Some(6), Some(2), Some(3)]
    );
    assert!(st.slots[..4].iter().all(|s| s.buf.is_some()));
}

#[test]
fn decode_split_matches_running_every_expert_on_the_cpu() {
    let layer = cpu_layer(8, 4);
    for e in [0, 2] {
        assert!(layer.admit(e, true).unwrap().is_some());
    }
    let xs: Vec<f32> = (0..64).map(|i| (i as f32 * 0.11).cos()).collect();
    let xs = Tensor::from_vec(xs, (1, 1, 64), &Device::Cpu).unwrap();
    // expert 0 is a hit, expert 3 a miss
    let ids = Tensor::new(&[[0u32, 3]], &Device::Cpu).unwrap();
    let ws = Tensor::new(&[[0.6f32, 0.4]], &Device::Cpu).unwrap();
    let got = super::decode::forward(&layer, &xs, &ws, &ids, Activation::Silu).unwrap();
    let want = layer.cpu.forward(&xs, ws, &ids).unwrap();
    assert_eq!(got.dims(), want.dims());
    let diff = (got - want)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    // Hits run per-expert `QMatMul`s, the reference the fused kernel: equal on aarch64, but
    // on x86 only to Q8_0 activation rounding (~0.005). A swapped weight moves it by ~0.09.
    assert!(diff < 2e-2, "hit/miss split diverges by {diff}");
}

#[test]
fn verify_rows_split_matches_running_every_expert_on_the_cpu() {
    // A verify step: three rows, with hits and misses mixed, rows with no hit (row 2) and no
    // miss (row 1), and expert 3 missed by two rows.
    let layer = cpu_layer(8, 4);
    for e in [0, 2] {
        assert!(layer.admit(e, true).unwrap().is_some());
    }
    let xs: Vec<f32> = (0..3 * 64).map(|i| (i as f32 * 0.11).cos()).collect();
    let xs = Tensor::from_vec(xs, (1, 3, 64), &Device::Cpu).unwrap();
    let ids = Tensor::new(&[[0u32, 3], [2, 0], [3, 5]], &Device::Cpu).unwrap();
    let ws = Tensor::new(&[[0.6f32, 0.4], [0.7, 0.3], [0.5, 0.5]], &Device::Cpu).unwrap();
    let got = super::decode::forward(&layer, &xs, &ws, &ids, Activation::Silu).unwrap();
    let want = layer.cpu.forward(&xs, ws, &ids).unwrap();
    assert_eq!(got.dims(), want.dims());
    let diff = (got - want)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 2e-2, "verify-row split diverges by {diff}");
}
