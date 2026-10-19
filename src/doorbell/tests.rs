//! Doorbell tests: the D2H/H2D round trips, the result ring's backpressure and the preload
//! handoff.

use super::*;
use crate::copy_stream::{CopyDone, CopyStream};
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
    let (got_ids, got_ws) = bell.publish_and_read(&ids, &ws, 4).unwrap();
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
    assert!(bell.publish(Some(&values), 1), "an empty ring always takes the payload");
    bell.wait_copy(1, 4, &dst).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(dst.to_vec1::<f32>().unwrap(), values);
    // The copy kernel publishes `consumed` once it has read the slot out.
    assert!(bell.consumed() >= 1, "consumed = {}", bell.consumed());
}

#[test]
fn result_doorbell_backpressure_waits_for_consumed_and_then_drops() {
    let Ok(dev) = Device::new_cuda(0) else {
        return;
    };
    let Ok(Some(bell)) = ResultDoorbell::acquire(&dev, 4) else {
        return;
    };
    let depth = bell.depth() as u32;
    let values = [1.0f32, 2.0, 3.0, 4.0];
    // Fill the whole ring without ever queueing a copy: the slots are unpublished but their
    // previous lap does not exist, so they are all free.
    for seq in 1..=depth {
        assert!(bell.publish(Some(&values), seq), "seq {seq} is inside the first lap");
    }
    // The next lap's first slot belongs to seq 1, which the device never consumed.
    assert!(
        !bell.publish(Some(&values), depth + 1),
        "a full ring must refuse the payload instead of overwriting an unread slot"
    );
    // ... and the seq bump still happened, so the stream cannot hang waiting for it.
    let dst = Tensor::zeros(4, DType::F32, &dev).unwrap();
    bell.wait_copy(depth + 1, 4, &dst).unwrap();
    dev.synchronize().unwrap();
    assert!(bell.consumed() >= depth + 1);
    // Slot 1 is consumed now, so the next lap is accepted again.
    assert!(bell.publish(Some(&values), depth + 2));
}

#[test]
fn pass_cell_hands_the_latest_claim_to_the_consumer() {
    let Ok(dev) = Device::new_cuda(0) else {
        return;
    };
    let Ok(Some(cs)) = CopyStream::new(&dev) else {
        return;
    };
    let cell = PassCell::new(std::sync::Arc::new(CopyDone::new(&cs).unwrap()));
    assert_eq!(cell.wait_latest(&dev).unwrap(), None, "no claim, no CUDA work");

    assert_eq!(cell.claim(&cs).unwrap(), 1);
    dev.synchronize().unwrap();
    assert_eq!(cell.wait_latest(&dev).unwrap(), Some(1));

    assert_eq!(cell.claim(&cs).unwrap(), 2);
    assert_eq!(cell.claimed(), 2);
    dev.synchronize().unwrap();
    assert_eq!(cell.wait_latest(&dev).unwrap(), Some(2));
}
