//! Doorbell tests: the D2H/H2D round trips and the v2 gate protocol.

use super::*;
use candle_core::{DType, Device, Tensor};

#[test]
fn topk_doorbell_round_trips_ids_and_weights() {
    let Ok(dev) = Device::new_cuda(0) else {
        return;
    };
    let Ok(Some(bell)) = TopkDoorbell::acquire(&dev, 8) else {
        return;
    };
    let ids = Tensor::new(&[3u32, 7, 1, 5], &dev).unwrap();
    let ws = Tensor::new(&[0.5f32, 0.25, 0.125, 0.0625], &dev).unwrap();
    bell.publish(&ids, &ws, 4).unwrap();
    let (seq, got_ids, got_ws) = bell.wait_and_read(0, 4);
    assert_eq!(seq, 1);
    assert_eq!(got_ids, vec![3, 7, 1, 5]);
    assert_eq!(got_ws, vec![0.5, 0.25, 0.125, 0.0625]);
}

#[test]
fn activation_doorbell_uses_seq_to_pick_the_ring_slot() {
    let Ok(dev) = Device::new_cuda(0) else {
        return;
    };
    let Ok(Some(bell)) = ActivationDoorbell::acquire(&dev, 4) else {
        return;
    };
    for seq in 0..3 {
        let values: Vec<f32> = (0..4).map(|i| seq as f32 * 10.0 + i as f32).collect();
        let src = Tensor::from_vec(values.clone(), 4, &dev).unwrap();
        bell.publish(&src, 4, seq).unwrap();
        assert_eq!(bell.wait_publish(seq, 4).unwrap(), values);
    }
}

#[test]
fn result_doorbell_copies_host_payload_to_the_device() {
    let Ok(dev) = Device::new_cuda(0) else {
        return;
    };
    let Ok(Some(bell)) = ResultDoorbell::acquire(&dev, 4) else {
        return;
    };
    let dst = Tensor::zeros(4, DType::F32, &dev).unwrap();
    let values = [1.0f32, 2.0, 3.0, 4.0];
    bell.publish(Some(&values), 1);
    bell.wait_copy(1, 4, &dst).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(dst.to_vec1::<f32>().unwrap(), values);
}

#[test]
fn epoch_advances_a_replayed_gate_to_the_next_pass() {
    let Ok(dev) = Device::new_cuda(0) else {
        return;
    };
    let Ok(Some(set)) = GateSet::acquire(&dev, 1, 2) else {
        return;
    };
    let gate = set.gate(set.sync_id(0, SYNC_STAGING));

    set.bump_epoch(&dev).unwrap();
    gate.publish_host(1);
    gate.wait(&dev).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(set.epoch(), 1, "one pass -> epoch 1");

    set.bump_epoch(&dev).unwrap();
    gate.publish_stream(dev.as_cuda_device().unwrap().cuda_stream().cu_stream(), 2)
        .unwrap();
    gate.wait(&dev).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(set.epoch(), 2, "two passes -> epoch 2");
    assert_eq!(gate.ready(), 2);
}

#[test]
fn stream_published_gate_satisfies_a_wait_only_node() {
    let Ok(dev) = Device::new_cuda(0) else {
        return;
    };
    let Ok(Some(set)) = GateSet::acquire(&dev, 1, 2) else {
        return;
    };
    let gate = set.gate(set.sync_id(1, SYNC_PRELOAD));
    set.bump_epoch(&dev).unwrap();
    gate.publish_stream(dev.as_cuda_device().unwrap().cuda_stream().cu_stream(), 1)
        .unwrap();
    gate.wait(&dev).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(gate.ready(), 1, "the host sees the stream-published pass");

    // A stale producer must not move the level backwards.
    gate.publish_host(1);
    gate.publish_stream_max(&dev, 0).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(gate.ready(), 1);
    gate.publish_stream_max(&dev, 3).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(gate.ready(), 3);
}
