//! Preload: publish this layer's activation for the next layer's prediction.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use candle_core::{DType, Device, Result, Tensor};
use tracing::debug;

use crate::executor::{self, Task};
use crate::moe::CachedLayer;
use crate::stats;

pub(crate) struct Preload;

impl Preload {
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
        if let Some(next) = layer.layers().get(layer.layer() + 1) {
            if next.router().is_some() {
                let queued = layer.lookahead();
                if !layer.slot_device().is_cuda() {
                    // Non-CUDA slot device: cheap host copy, same job shape.
                    let x = x2d.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    let x = x.flatten_all()?.to_vec1::<f32>()?;
                    if queued {
                        let next = Arc::clone(&next);
                        schedule(Box::new(move || {
                            let r = x.len() / next.hidden().max(1);
                            let x = Tensor::from_vec(x, (r, next.hidden()), &Device::Cpu);
                            if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                                debug!(
                                    "expert cache: layer {} lookahead failed: {e}",
                                    next.layer()
                                );
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
                        schedule(Box::new(move || {
                            let t0 = std::time::Instant::now();
                            // The GAP is the scheduler's business, not this
                            // closure's: the lookahead thread only drains its queue
                            // while no expert batch is running, and a batch's
                            // completion wakes it (W106: the router matmul is 2-3
                            // cores and running it beside a batch measured 10
                            // ms/step slower).
                            if let Some(v) = sh.read(last_seq, n) {
                                let x = Tensor::from_vec(v, (rows, next.hidden()), &Device::Cpu);
                                if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                                    debug!(
                                        "expert cache: layer {} lookahead failed: {e}",
                                        next.layer()
                                    );
                                }
                            }
                            stats::record_lookahead_job(t0.elapsed().as_secs_f64() * 1e3);
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
                    schedule(Box::new(move || {
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
}

const LOOKAHEAD_PARK: Duration = Duration::from_millis(2);
const LOOKAHEAD_DEPTH: usize = 64;

static LOOKAHEAD_JOBS: Mutex<VecDeque<Task>> = Mutex::new(VecDeque::new());
static LOOKAHEAD_LEN: AtomicUsize = AtomicUsize::new(0);
static LOOKAHEAD_THREAD: OnceLock<std::thread::Thread> = OnceLock::new();

fn ensure_thread() -> std::thread::Thread {
    LOOKAHEAD_THREAD
        .get_or_init(|| {
            let t = std::thread::Builder::new()
                .name("lookahead".into())
                .spawn(lookahead_worker)
                .map(|h| h.thread().clone())
                .unwrap_or_else(|e| {
                    tracing::warn!("no lookahead thread ({e}); predictions will not run");
                    std::thread::current()
                });
            executor::cpu().on_idle(t.clone());
            t
        })
        .clone()
}

fn schedule(job: Task) {
    let thread = ensure_thread();
    {
        let mut jobs = LOOKAHEAD_JOBS.lock().unwrap();
        if jobs.len() >= LOOKAHEAD_DEPTH {
            stats::record_cache_lookahead_dropped();
            return;
        }
        jobs.push_back(job);
        LOOKAHEAD_LEN.store(jobs.len(), Ordering::Release);
    }
    thread.unpark();
}

fn take() -> Option<Task> {
    if LOOKAHEAD_LEN.load(Ordering::Acquire) == 0 {
        return None;
    }
    let mut jobs = LOOKAHEAD_JOBS.lock().unwrap();
    let job = jobs.pop_front();
    LOOKAHEAD_LEN.store(jobs.len(), Ordering::Release);
    job
}

fn lookahead_worker() {
    loop {
        while executor::cpu().is_idle() {
            let Some(job) = take() else { break };
            job();
        }
        std::thread::park_timeout(LOOKAHEAD_PARK);
    }
}
