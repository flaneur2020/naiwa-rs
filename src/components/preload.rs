//! Preload: publish this layer's activation for the next layer's prediction.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use tracing::debug;

use crate::expert_cache::CachedLayer;
use crate::stats;

pub(crate) fn submit(
    layer: &Arc<CachedLayer>,
    x2d: &Tensor,
    rows: usize,
    hidden: usize,
    pass: u32,
) -> Result<()> {
    // Start predicting the next layer's experts now, off the token path; whatever is not
    // resident by the time that layer runs goes to the CPU.
    //
    // R2a: the token thread queues ONE publish kernel on the stream (copy this
    // step's activation into the layer's mapped slot, bump seq) and hands the
    // (last_seq, slot) to the lookahead job. The worker spins for the seq, rebuilds
    // the activation on the CPU and matmuls against the CPU router — the prediction
    // never touches the GPU stream. This removes the W105b 0.30-0.34 ms/layer
    // blocking `to_vec1` D2H from the token thread AND the W99 late-prediction
    // hazard (the publish is stream-ordered right after this layer's last kernel,
    // so the worker can never run before the data exists); the old mode switch is
    // obsolete with the channel path.
    if let Some(next) = layer.registry().get(layer.layer() + 1) {
        if next.router().is_some() {
            let queued = layer.lookahead();
            if !layer.slot_device().is_cuda() {
                // Non-CUDA slot device: cheap host copy, same job shape.
                let x = x2d.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                let x = x.flatten_all()?.to_vec1::<f32>()?;
                if queued {
                    let next = Arc::clone(&next);
                    crate::lookahead::submit(Box::new(move || {
                        let r = x.len() / next.hidden().max(1);
                        let x = Tensor::from_vec(x, (r, next.hidden()), &Device::Cpu);
                        if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                            debug!("expert cache: layer {} lookahead failed: {e}", next.layer());
                        }
                    }));
                }
                stats::record_lookahead_step(0.0, queued);
            } else if let (true, Some(chan)) = (queued, layer.lookahead_chan()) {
                let n = rows * hidden;
                let next = Arc::clone(&next);
                let t0 = std::time::Instant::now();
                let pub_res = chan.publish(&x2d, n);
                stats::record_lookahead_publish(t0.elapsed().as_secs_f64() * 1e3);
                if let Some((sh, last_seq)) = pub_res? {
                    crate::lookahead::submit(Box::new(move || {
                        let t0 = std::time::Instant::now();
                        // The GAP is the scheduler's business, not this
                        // closure's: the lookahead thread only drains its queue
                        // while no expert batch is running, and a batch's
                        // completion wakes it (W106: the router matmul is 2-3
                        // cores and running it beside a batch measured 10
                        // ms/step slower).
                        // Stale check: this layer queued a NEWER prediction
                        // while this job waited — the newer publish carries a
                        // fresher activation (its `wait_publish` job will do
                        // the prefetch), so drop this one. Every layer
                        // resubmits every step; dropping old ones is free.
                        if sh.next_seq.load(Ordering::Relaxed) > last_seq.wrapping_add(1) {
                            stats::record_lookahead_job(t0.elapsed().as_secs_f64() * 1e3);
                            sh.consumed.fetch_max(last_seq + 1, Ordering::Relaxed);
                            return;
                        }
                        if let Some(v) = sh.chan.wait_publish(last_seq, n) {
                            let x = Tensor::from_vec(v, (rows, next.hidden()), &Device::Cpu);
                            if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                                debug!(
                                    "expert cache: layer {} lookahead failed: {e}",
                                    next.layer()
                                );
                            }
                        }
                        stats::record_lookahead_job(t0.elapsed().as_secs_f64() * 1e3);
                        // Paid back either way: a timed-out publish's slot must
                        // stop counting against the backpressure budget.
                        // `fetch_max`: team threads finish out of order — a
                        // plain store from an older job would regress the
                        // counter and wedge the backpressure shut.
                        sh.consumed.fetch_max(last_seq + 1, Ordering::Relaxed);
                    }));
                }
                stats::record_lookahead_step(0.0, true);
            } else if queued {
                // CUDA without the channel (the mapped allocation failed): fall
                // back to the W105 blocking `to_vec1` on the token thread rather
                // than silently losing the lookahead.
                let t0 = std::time::Instant::now();
                let x_clone = x2d.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let d2h = t0.elapsed().as_secs_f64() * 1e3;
                let next = Arc::clone(&next);
                crate::lookahead::submit(Box::new(move || {
                    let x = Tensor::from_vec(x_clone, (rows, next.hidden()), &Device::Cpu);
                    if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                        debug!("expert cache: layer {} lookahead failed: {e}", next.layer());
                    }
                }));
                stats::record_lookahead_step(d2h, true);
            } else {
                stats::record_lookahead_step(0.0, false);
            }
        }
    }

    Ok(())
}
