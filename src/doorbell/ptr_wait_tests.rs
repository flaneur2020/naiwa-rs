//! Mapped-memory wait/copy kernel tests (see `super`).

use super::*;
use candle_core::{DType, Device, Tensor};

/// F3 / v2 M0: the zero-argument wait+copy kernel must deliver exactly the payload whose seq
/// it waited for, reading `target` and `slot_off` from mapped memory rather than from kernel
/// arguments — so the *same* object serves successive values without recompiling a launch.
#[test]
fn wait_copy_slot_ptr_reads_target_and_slot_from_mapped_memory() {
    let dev = Device::new_cuda(0).unwrap();
    let max_n = 8usize;
    let flag = FlagBuf::acquire(&dev, max_n).unwrap().unwrap();
    let dst = Tensor::zeros(max_n, DType::F32, &dev).unwrap();

    let first: Vec<f32> = (0..max_n).map(|i| i as f32 + 1.5).collect();
    flag.write_slot(Some(&first), 2, 7);
    flag.set_target(7);
    flag.set_slot_off(2);
    flag.wait_copy_slot_ptr(max_n, &dst).unwrap();
    assert_eq!(dst.to_vec1::<f32>().unwrap(), first);

    // Same channel, same kernel parameters: only the mapped cells moved.
    let second: Vec<f32> = (0..max_n).map(|i| 100.0 - i as f32).collect();
    flag.write_slot(Some(&second), 5, 8);
    flag.set_target(8);
    flag.set_slot_off(5);
    flag.wait_copy_slot_ptr(max_n, &dst).unwrap();
    assert_eq!(dst.to_vec1::<f32>().unwrap(), second);

    // A failed computation still unblocks the wait (seq bumps, no payload).
    flag.set_target(9);
    flag.write_slot(None, 5, 9);
    flag.wait_seq_ge_ptr(&dev).unwrap();
}

/// v2 §3.2: the property every earlier variant lacked — a node whose `(layer, kind)` are
/// **constants** advances to the next pass because the *device* moved the epoch, not the host.
#[test]
fn epoch_advances_a_replayed_node_to_the_next_pass() {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let dev = Device::new_cuda(0).unwrap();
    let (max_n, slots, layer) = (8usize, 4usize, 2u32);
    let sched = ScheduleBuf::acquire(&dev, 3, slots, max_n).unwrap().unwrap();
    let acc = Tensor::zeros(max_n, DType::F32, &dev).unwrap();
    let acc_ptr = {
        let (st, _) = acc.storage_and_layout();
        let candle_core::Storage::Cuda(s) = &*st else {
            panic!("acc must be on CUDA");
        };
        let slice = s.as_cuda_slice::<f32>().unwrap();
        let (p, _guard) = slice.device_ptr(slice.stream());
        p
    };

    // Pass 1: the prologue bumps the epoch; the host publishes for its own pass counter.
    sched.bump_epoch(&dev).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(sched.epoch(), 1, "one pass -> epoch 1");
    let first = vec![1.0f32; max_n];
    sched.publish_cpu(layer, 1, &first);
    sched.wait_add_cpu(&dev, max_n, layer, acc_ptr).unwrap();
    assert_eq!(acc.to_vec1::<f32>().unwrap(), first);

    // Pass 2, SAME node constants: it must wait for pass 2's word, not be satisfied by pass
    // 1's. The payload slot is single-buffered — safe because the host cannot reach pass 2
    // without a GPU-produced signal from pass 2.
    sched.bump_epoch(&dev).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(sched.epoch(), 2, "two passes -> epoch 2");
    let second = vec![10.0f32; max_n];
    sched.publish_cpu(layer, 2, &second);
    sched.wait_add_cpu(&dev, max_n, layer, acc_ptr).unwrap();
    let want: Vec<f32> = first.iter().zip(&second).map(|(a, b)| a + b).collect();
    assert_eq!(acc.to_vec1::<f32>().unwrap(), want);
}

/// M4: a DMA-done point published **by the stream itself** satisfies a wait-only node, and the
/// host sees the same value — the shape both `PRELOAD` and `STAGING` will use.
#[test]
fn stream_published_point_satisfies_a_wait_only_node() {
    let dev = Device::new_cuda(0).unwrap();
    let sched = ScheduleBuf::acquire(&dev, 3, 2, 8).unwrap().unwrap();
    let id = sched.sync_id(1, super::SYNC_STAGING);

    sched.bump_epoch(&dev).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(sched.epoch(), 1);
    // Published by the stream (as the copy stream does after a DMA), then waited on.
    sched.publish_point_on_stream(&dev, id, 1).unwrap();
    sched.wait_ready(&dev, id).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(sched.ready(id), 1, "the host sees the stream-published pass");

    // Next pass: the point must be re-armed by the stream again.
    sched.bump_epoch(&dev).unwrap();
    dev.synchronize().unwrap();
    sched.publish_point_on_stream(&dev, id, 2).unwrap();
    sched.wait_ready(&dev, id).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(sched.ready(id), 2);
}

/// M3: the indexed node claims its job **on the device**, so two nodes queued together
/// consume slots 1 and 2 with no host-written per-step cell — the property that makes it
/// safe inside a captured graph while the token thread runs ahead.
#[test]
fn wait_add_indexed_claims_jobs_on_the_device() {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let dev = Device::new_cuda(0).unwrap();
    let cuda = dev.as_cuda_device().unwrap();
    let max_n = 8usize;
    let flag = FlagBuf::acquire(&dev, max_n).unwrap().unwrap();
    let acc = Tensor::zeros(max_n, DType::F32, &dev).unwrap();
    let counter = cuda.alloc_zeros::<u32>(1).unwrap();

    let first: Vec<f32> = (0..max_n).map(|i| i as f32 + 1.0).collect();
    let second: Vec<f32> = (0..max_n).map(|i| 10.0 + i as f32).collect();
    // Jobs 1 and 2 live on ring slots 1 and 2 (the publisher's numbering).
    flag.write_slot(Some(&first), 1, 1);
    flag.write_slot(Some(&second), 2, 2);

    let counter_ptr = counter.device_ptr(counter.stream()).0;
    let acc_ptr = {
        let (st, _) = acc.storage_and_layout();
        let candle_core::Storage::Cuda(s) = &*st else {
            panic!("acc must be on CUDA");
        };
        let slice = s.as_cuda_slice::<f32>().unwrap();
        let (p, _guard) = slice.device_ptr(slice.stream());
        p
    };
    flag.wait_add_indexed_from(&dev, counter_ptr, max_n, acc_ptr)
        .unwrap();
    flag.wait_add_indexed_from(&dev, counter_ptr, max_n, acc_ptr)
        .unwrap();

    let want: Vec<f32> = first.iter().zip(&second).map(|(a, b)| a + b).collect();
    assert_eq!(acc.to_vec1::<f32>().unwrap(), want);
}

/// M3: the wait-then-accumulate node folds the payload into the buffer that already holds
/// the GPU hits' sum, so the graph's output is the completed MoE result.
#[test]
fn wait_add_slot_ptr_accumulates_into_the_output_buffer() {
    let dev = Device::new_cuda(0).unwrap();
    let max_n = 8usize;
    let flag = FlagBuf::acquire(&dev, max_n).unwrap().unwrap();
    let acc = Tensor::full(1.0f32, max_n, &dev).unwrap();

    let cpu: Vec<f32> = (0..max_n).map(|i| 10.0 + i as f32).collect();
    flag.write_slot(Some(&cpu), 1, 3);
    flag.set_target(3);
    flag.set_slot_off(1);
    flag.wait_add_slot_ptr(max_n, &acc).unwrap();

    let want: Vec<f32> = (0..max_n).map(|i| 11.0 + i as f32).collect();
    assert_eq!(acc.to_vec1::<f32>().unwrap(), want);
}
