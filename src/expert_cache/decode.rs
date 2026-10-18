//! The decode path: split top-k into GPU hits and CPU misses, run both, sum.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor};
use mistralrs_core::layers::Activation;
use tracing::{debug, warn};

use crate::deferred;
use crate::offload::fused_decode as offload_fused_decode;

use super::probe::{lookahead_probe, round_stats, step_probe};
use super::{schedule, Bank, CachedLayer, SlotBuf, SlotWeights, COUNTERS};

/// The fused MoE decode kernel's activation id, if it has one for `act`.
#[cfg(feature = "cuda")]
fn fused_act(act: Activation) -> Option<i32> {
    match act {
        Activation::Silu | Activation::Swish => Some(mistralrs_quant::ACT_SILU),
        Activation::GeluPytorchTanh => Some(mistralrs_quant::ACT_GELU_PYTORCH_TANH),
        _ => None,
    }
}

/// `sum_j weights[t, j] * expert(bank, ids[t, j])(x2d[t])` for each of `rows` tokens,
/// `[rows, hidden]` F32, by mistral.rs's fused MoE decode (quantize, gate+up+act, quantize,
/// down+weighted sum). `ids` and `weights` are row-major `[rows, topk]`.
#[cfg(feature = "cuda")]
fn fused_decode(
    bank: &Bank,
    x2d: &Tensor,
    ids: Vec<u32>,
    weights: Vec<f32>,
    rows: usize,
    act_type: i32,
) -> candle_core::Result<Tensor> {
    use candle_core::cuda::cudarc::driver::DevicePtr;
    let dev = x2d.device().as_cuda_device()?;
    let topk = ids.len() / rows;
    let ids = Tensor::new(ids, x2d.device())?;
    let weights = Tensor::new(weights, x2d.device())?;
    let (ids_storage, _) = ids.storage_and_layout();
    let candle_core::Storage::Cuda(ids_cuda) = &*ids_storage else {
        candle_core::bail!("expert cache: fused decode ids are not on CUDA");
    };
    let (w_storage, _) = weights.storage_and_layout();
    let candle_core::Storage::Cuda(w_cuda) = &*w_storage else {
        candle_core::bail!("expert cache: fused decode weights are not on CUDA");
    };
    let w_slice = w_cuda.as_cuda_slice::<f32>()?;
    let w_ptr = w_slice.device_ptr(w_slice.stream()).0 as *const f32;
    // SAFETY: `w_ptr` points at `topk` f32s of `weights`, whose free is ordered on candle's
    // stream after the kernels queued here.
    unsafe {
        mistralrs_quant::indexed_moe_fused_decode(
            &bank.gate,
            &bank.up,
            &bank.down,
            x2d,
            ids_cuda.as_cuda_slice::<u32>()?,
            w_ptr,
            rows,
            topk,
            act_type,
            dev,
        )
    }
}

/// A resident expert a token is routed to: its slot buffer, slot index and routing weight.
pub(super) type Hit = (Arc<SlotBuf>, usize, f32);
impl CachedLayer {
    /// The weighted sum of the hits' outputs, `[rows, hidden]` F32 on the slot device, for
    /// `hits[t]` the hits of row `t` of `x2d: [rows, hidden]`. In copy-stream mode each bank's hits
    /// run as one fused MoE decode over every row (four kernels); otherwise every expert runs on
    /// its own.
    fn run_hits(
        &self,
        hits: &[Vec<Hit>],
        x2d: &Tensor,
        act: Activation,
    ) -> candle_core::Result<(Tensor, Option<Arc<Bank>>)> {
        let rows = hits.len();
        // M9: no captures, so nothing to remember for the CPU-return node.
        let capture_bank: Option<Arc<Bank>> = None;
        let mut acc: Option<Tensor> = None;
        let mut add = |y: Tensor| -> candle_core::Result<()> {
            acc = Some(match acc.take() {
                Some(a) => (a + y)?,
                None => y,
            });
            Ok(())
        };
        #[cfg(feature = "cuda")]
        if let (Some(act_type), true) = (
            fused_act(act),
            hits.iter().flatten().all(|(buf, _, _)| buf.bank.is_some()),
        ) {
            // Per bank, each row's (local slot, weight) pairs. The kernel takes the same number
            // per row, so short rows are padded with a weight-0 entry on a slot of that bank that
            // some row hits (published and fenced, so it holds finite values).
            let mut groups: Vec<(&Arc<Bank>, Vec<Vec<(u32, f32)>>)> = Vec::new();
            for (t, row) in hits.iter().enumerate() {
                for (buf, slot, w) in row {
                    let bank = buf.bank.as_ref().expect("checked above");
                    let local = (slot - bank.first) as u32;
                    let i = match groups.iter().position(|(b, _)| Arc::ptr_eq(b, bank)) {
                        Some(i) => i,
                        None => {
                            groups.push((bank, vec![Vec::new(); rows]));
                            groups.len() - 1
                        }
                    };
                    groups[i].1[t].push((local, *w));
                }
            }
            for (bank, per_row) in groups {
                let width = per_row.iter().map(Vec::len).max().unwrap_or(0);
                let pad = per_row.iter().flatten().next().map_or(0, |&(l, _)| l);
                let mut ids = Vec::with_capacity(rows * width);
                let mut ws = Vec::with_capacity(rows * width);
                for row in &per_row {
                    for i in 0..width {
                        let (l, w) = row.get(i).copied().unwrap_or((pad, 0.0));
                        ids.push(l);
                        ws.push(w);
                    }
                }
                add(fused_decode(bank, x2d, ids, ws, rows, act_type)?)?;
            }
            return match acc {
                Some(acc) => Ok((acc, capture_bank)),
                None => Ok((
                    Tensor::zeros((rows, self.hidden), DType::F32, &self.slot_device)?,
                    None,
                )),
            };
        }
        // One expert at a time, row by row.
        let mut out = Vec::with_capacity(rows);
        for (t, row) in hits.iter().enumerate() {
            let x = x2d.narrow(0, t, 1)?;
            let mut sum = Tensor::zeros((1, self.hidden), DType::F32, &self.slot_device)?;
            for (buf, _, w) in row {
                let Some(weights) = &buf.weights else {
                    candle_core::bail!("expert cache: slot has no weights to run");
                };
                let y = Self::run_expert(weights, &x, act)?;
                sum = (sum + (y.to_dtype(DType::F32)? * *w as f64)?)?;
            }
            out.push(sum);
        }
        add(Tensor::cat(&out, 0)?)?;
        Ok((acc.expect("added above"), capture_bank))
    }

    /// Run one token through a cached expert. `x2d: [1, hidden]` on the slot device.
    fn run_expert(w: &SlotWeights, x2d: &Tensor, act: Activation) -> candle_core::Result<Tensor> {
        let g = QMatMul::QTensor(w.gate.clone()).forward(x2d)?;
        let u = QMatMul::QTensor(w.up.clone()).forward(x2d)?;
        let h = (act.forward(&g)? * u)?;
        QMatMul::QTensor(w.down.clone()).forward(&h)
    }

    /// Decode path (one token, or the few rows of a speculative verify step): split each row's
    /// top-k into GPU hits and CPU misses, run both concurrently, sum. Misses are never loaded on
    /// the token path — they run on the CPU.
    ///
    /// Every CUDA call stays on the calling thread; the misses run on the `deferred` worker over
    /// host copies. Launching the hits from a second thread while this one copied the activation
    /// to the host produced NaN logits on CUDA once the lookahead was swapping slots.
    ///
    /// Returns without waiting for the misses: their sum is copied in on candle's stream when
    /// the worker is done (`deferred`), so the next layer's launches overlap the CPU experts.
    /// `--pcie-frac`: DMA the highest-weight misses of this step into the scratch slots and turn
    /// them into hits, so `run_hits` runs them on the GPU with the residents and the rest stay on
    /// the CPU. The chosen ids are removed from `misses`, leaving the caller's CPU padding and
    /// `deferred::start` untouched. The injected hits carry the scratch `SlotBuf`, so the caller's
    /// fence loop covers them and the next step's DMA waits on this step's readers.
    ///
    /// `upload_async` returns as soon as the copy is queued; the layer's DMA gate (a captured wait
    /// node, M4) spins until the copy stream has published `SYNC_STAGING` for this pass, so candle's
    /// stream never waits on the copy stream and the host enqueues nothing per streamed expert.

    pub(super) fn decode_forward(
        self: &Arc<Self>,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        act: Activation,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq, hidden) = xs.dims3()?;
        let rows = batch * seq;
        let x2d = xs.reshape((rows, hidden))?;
        step_probe::mark(&self.slot_device, self.layer);
        // M3/M4: this layer's pass index. Bumped once per pass at layer 0, together with the device
        // epoch, and only once the schedule exists — so the two counters cannot drift. It has to be
        // known before the DMA, because that is what publishes the staging point for this pass.
        #[cfg(feature = "cuda")]
        let pass = {
            static PASS: AtomicU32 = AtomicU32::new(0);
            if self.layer == 0 {
                match schedule() {
                    Some(sched) => {
                        let p = PASS.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Err(e) = sched.bump_epoch(&self.slot_device) {
                            warn!("expert cache: epoch bump failed: {e}");
                        }
                        p
                    }
                    None => 0,
                }
            } else {
                PASS.load(Ordering::Relaxed)
            }
        };
        #[cfg(not(feature = "cuda"))]
        let pass: u32 = 0;
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
        if let Some(next) = self.registry.get(self.layer + 1) {
            if next.router.is_some() {
                let queued = self.lookahead;
                if !self.slot_device.is_cuda() {
                    // Non-CUDA slot device: cheap host copy, same job shape.
                    let x = x2d.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    let x = x.flatten_all()?.to_vec1::<f32>()?;
                    if queued {
                        let next = Arc::clone(&next);
                        crate::lookahead::submit(Box::new(move || {
                            let r = x.len() / next.hidden.max(1);
                            let x = Tensor::from_vec(x, (r, next.hidden), &Device::Cpu);
                            if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                                debug!("expert cache: layer {} lookahead failed: {e}", next.layer);
                            }
                        }));
                    }
                    lookahead_probe::record(0.0, queued);
                } else if let (true, Some(chan)) = (queued, &self.lookahead_chan) {
                    let n = rows * hidden;
                    let next = Arc::clone(&next);
                    let t0 = std::time::Instant::now();
                    let pub_res = chan.publish(&x2d, n);
                    lookahead_probe::record_publish(t0.elapsed().as_secs_f64() * 1e3);
                    if let Some((sh, last_seq, slot)) = pub_res? {
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
                                lookahead_probe::record_job(t0.elapsed().as_secs_f64() * 1e3);
                                sh.consumed.fetch_max(last_seq + 1, Ordering::Relaxed);
                                return;
                            }
                            if let Some(v) = sh.chan.wait_publish(last_seq, n, slot) {
                                let x = Tensor::from_vec(v, (rows, next.hidden), &Device::Cpu);
                                if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                                    debug!(
                                        "expert cache: layer {} lookahead failed: {e}",
                                        next.layer
                                    );
                                }
                            }
                            lookahead_probe::record_job(t0.elapsed().as_secs_f64() * 1e3);
                            // Paid back either way: a timed-out publish's slot must
                            // stop counting against the backpressure budget.
                            // `fetch_max`: team threads finish out of order — a
                            // plain store from an older job would regress the
                            // counter and wedge the backpressure shut.
                            sh.consumed.fetch_max(last_seq + 1, Ordering::Relaxed);
                        }));
                    }
                    lookahead_probe::record(0.0, true);
                } else if queued {
                    // CUDA without the channel (the mapped allocation failed): fall
                    // back to the W105 blocking `to_vec1` on the token thread rather
                    // than silently losing the lookahead.
                    let t0 = std::time::Instant::now();
                    let x_clone = x2d
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let d2h = t0.elapsed().as_secs_f64() * 1e3;
                    let next = Arc::clone(&next);
                    crate::lookahead::submit(Box::new(move || {
                        let x = Tensor::from_vec(x_clone, (rows, next.hidden), &Device::Cpu);
                        if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                            debug!("expert cache: layer {} lookahead failed: {e}", next.layer);
                        }
                    }));
                    lookahead_probe::record(d2h, true);
                } else {
                    lookahead_probe::record(0.0, false);
                }
            }
        }

        let n = rows * self.top_k;
        // The dispatch decision is made **on the GPU**: the kernel emits the resident / staged / CPU lists
        // with rows and weights, so the host never hashes an id. The only host-side guard is the shape check
        // (the startup dummy run reaches here with a selection that does not match `xs`).
        #[cfg(feature = "cuda")]
        let device_flip = self.top_k > 0
            && topk_ids.elem_count() % self.top_k == 0
            && topk_ids.elem_count() / self.top_k == rows;
        #[cfg(not(feature = "cuda"))]
        let device_flip = false;

        // Step 5 doorbell: lazily acquire (first decode), queue the ring on the same
        // stream the top-k kernels ran on, then spin host-side for the payload — no
        // stream sync, no D2H launch. Falls back to `to_vec2` when unavailable.
        let ring = if device_flip {
            None
        } else {
            let mut st = self.state.lock().unwrap();
            if st.doorbell.is_none() {
                let cap = n.max(8 * self.top_k).max(64);
                st.doorbell = crate::doorbell::DoorbellBuf::acquire(&self.slot_device, cap)
                    .ok()
                    .flatten()
                    .map(Arc::new);
            }
            let db = st.doorbell.clone().filter(|db| db.max_n() >= n);
            match db {
                Some(db) => match db.ring(topk_ids, topk_weights, n) {
                    Ok(()) => {
                        let seq = st.doorbell_seq;
                        st.doorbell_seq = seq.wrapping_add(1);
                        Some((db, seq))
                    }
                    Err(_) => None,
                },
                None => None,
            }
        };
        let (ids_rows, weight_rows): (Vec<Vec<u32>>, Vec<Vec<f32>>);
        if let Some((db, seq)) = ring {
            let (_, ids, ws) = db.wait_and_read(seq, n);
            // W106 sanitation: a garbage id here means the doorbell payload was read
            // stale or corrupted; log everything needed to diagnose and fall back to
            // the blocking D2H for this step rather than indexing out of bounds.
            if ids.iter().any(|&id| id as usize >= self.num_experts) {
                let bad: Vec<u32> = ids
                    .iter()
                    .copied()
                    .filter(|id| *id as usize >= self.num_experts)
                    .take(8)
                    .collect();
                candle_core::bail!(
                    "doorbell garbage: layer {} seq {} n {} cap {} rows {} top_k {} ids_shape {:?} bad_ids {:x?}",
                    self.layer,
                    seq,
                    n,
                    db.max_n(),
                    rows,
                    self.top_k,
                    topk_ids.shape(),
                    bad
                );
            }
            ids_rows = ids.chunks(self.top_k).map(<[u32]>::to_vec).collect();
            weight_rows = ws.chunks(self.top_k).map(<[f32]>::to_vec).collect();
        } else {
            ids_rows = topk_ids.reshape((rows, ()))?.to_vec2::<u32>()?;
            weight_rows = topk_weights
                .reshape((rows, ()))?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
        }

        let mut hits: Vec<Vec<Hit>> = vec![Vec::new(); rows];
        let mut misses: Vec<Vec<(u32, f32)>> = vec![Vec::new(); rows];
        // (removed) notes on the old host-side flip:
        //   1. the lists are shared across layers in one mapping, and the counts are read after a launch
        //      the host does not wait on, so a layer can read the previous layer's larger list (the OOB
        //      that killed the dummy run: len 1, index 210). The list area must be per layer and the count
        //      published with a seq the host waits on - the same protocol the top-k ring uses today.
        //   2. the row count must come from the selection, not from `xs` (already handled here, but it is
        //      only safe once 1 is fixed).
        {
            let mut st = self.state.lock().unwrap();
            st.tick += 1;
            let tick = st.tick;
            for (t, (ids, ws)) in ids_rows.iter().zip(&weight_rows).enumerate() {
                for (&id, &w) in ids.iter().zip(ws) {
                    st.hit_counts[id as usize] += 1;
                    let resident = st
                        .slots
                        .get(&(id as usize))
                        .copied()
                        .and_then(|slot| st.published[slot].clone().map(|buf| (slot, buf)));
                    match resident {
                        Some((slot, buf)) => {
                            st.last_use[slot] = tick;
                            hits[t].push((buf, slot, w));
                        }
                        None => misses[t].push((id, w)),
                    }
                }
            }
        }

        // Hit rate counts only experts actually resident in the cache. Staged ones are injected into
        // `hits` by `stream_pcie` below, so the count has to happen *before* it - otherwise a layer that
        // streams everything reads as a 100% cache hit.
        // Probe `NAIWA_GPU_ONLY_EXPERTS`: once the cache is full, drop every expert that is not resident and
        // never stage or compute one. This is *lossy* (the layer loses those experts' contribution, so the
        // text changes) - it exists to measure what a round costs with no CPU leg and no DMA at all, i.e. the
        // upper bound a perfect cache would give.
        #[cfg(feature = "cuda")]
        if std::env::var_os("NAIWA_GPU_ONLY_EXPERTS").is_some() {
            let full = {
                let st = self.state.lock().unwrap();
                st.resident[..st.usable].iter().all(|s| s.is_some())
            };
            if full {
                for row in misses.iter_mut() {
                    row.clear();
                }
            }
        }
        let n_hits: usize = hits.iter().map(Vec::len).sum();
        COUNTERS.hits.fetch_add(n_hits as u64, Ordering::Relaxed);



        // `--pcie-frac`: DMA the highest-weight misses into the staging ring and run them on the
        // GPU with the hits, leaving the rest on the CPU. Every CUDA call stays on this thread.
        // `stage_misses` only *appends* to `hits`, so the resident prefix is exactly what was there
        // before the call - which is what lets the split below run the residents without the ring.
        #[cfg(feature = "cuda")]
        let resident_prefix: Vec<usize> = hits.iter().map(Vec::len).collect();
        if self.pcie_frac > 0.0 {
            self.stream_pcie(&mut misses, &mut hits, pass)?;
        }

        let n_cpu: usize = misses.iter().map(Vec::len).sum();
        COUNTERS.misses.fetch_add(n_cpu as u64, Ordering::Relaxed);
        // Experts that went to the staging ring: counted apart from both hits and CPU misses.
        #[cfg(feature = "cuda")]
        {
            let staged_n: usize = hits.iter().map(Vec::len).sum::<usize>() - n_hits;
            COUNTERS.staged.fetch_add(staged_n as u64, Ordering::Relaxed);
            round_stats::record_plan(n_hits, staged_n, n_cpu);
        }

        let compute: Option<Box<dyn FnOnce() -> candle_core::Result<Vec<f32>> + Send>> =
            if n_cpu == 0 {
                None
            } else {
                let width = misses.iter().map(Vec::len).max().unwrap_or(0);
                let pad = misses.iter().flatten().next().map_or(0, |&(id, _)| id);
                let mut ids = Vec::with_capacity(rows * width);
                let mut ws = Vec::with_capacity(rows * width);
                for row in &misses {
                    for i in 0..width {
                        let (id, w) = row.get(i).copied().unwrap_or((pad, 0.0));
                        ids.push(id);
                        ws.push(w);
                    }
                }
                let xs_cpu = x2d.unsqueeze(0)?.to_device(&Device::Cpu)?;
                let this = Arc::clone(self);
                Some(Box::new(move || {
                    let x = xs_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    if let Some(ys) =
                        offload_fused_decode(&this.cpu, this.act, &x, &ids, &ws, width)?
                    {
                        return Ok(ys);
                    }
                    let miss_w_t = Tensor::from_vec(ws, (rows, width), &Device::Cpu)?;
                    let miss_ids_t = Tensor::from_vec(ids, (rows, width), &Device::Cpu)?;
                    this.cpu
                        .forward(&xs_cpu, miss_w_t, &miss_ids_t)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()
                }))
            };

        // M9: there are no segment captures, so no captured CPU-return node can consume this layer's
        // result - the deferred channel always carries it.
        let fast = false;

        // Publication must stay strictly FIFO per point: the payload slot is single-buffered and
        // `ready` must not go backwards. So *every* point for this layer goes through the worker,
        // including the "no misses, publish zeros" case — a synchronous publish from the token
        // thread could overtake a slower job for the previous pass and invert them.
        #[cfg(feature = "cuda")]
        let cpu = if fast {
            let sched = schedule().expect("checked above");
            let len = rows * self.hidden;
            let job: Box<dyn FnOnce() -> candle_core::Result<Vec<f32>> + Send> = match compute {
                Some(f) => f,
                None => Box::new(move || Ok(vec![0f32; len])),
            };
            deferred::start_schedule(
                Arc::clone(sched),
                self.layer as u32,
                pass,
                len,
                job,
            )?;
            None
        } else {
            let chan = self.cpu_chan.get();
            compute
                .map(|f| deferred::start(&self.slot_device, chan, rows, self.hidden, f))
                .transpose()?
        };
        #[cfg(not(feature = "cuda"))]
        #[allow(unused_variables)]
        let cpu = compute
            .map(|f| deferred::start(&self.slot_device, None, rows, self.hidden, f))
            .transpose()?;

        // M4: the copy stream published this layer's `SYNC_STAGING` point on its own stream, after
        // this step's copies. Replaying the layer's gate spins here, inside a captured node, so
        // nothing below can read a staging slot that is still in flight — and the host enqueues no
        // per-expert stream wait to get that guarantee.
        //
        // When warmup could not capture the gate, the *same* node is launched warm: one host call
        // per layer, like the `cudaStreamWaitEvent` it replaces, so the protocol stays one protocol.
        // M5b: same for the preload point, but only when this pass was actually claimed — a layer
        // with no preload this pass has nothing to wait for, and replaying its gate would wait for a
        // publish that nobody owes (`--no-lookahead` makes that the normal case).
        #[cfg(feature = "cuda")]
        if self.preload_claimed.load(Ordering::Acquire) == pass {
            self.replay_gate(crate::doorbell::SYNC_PRELOAD)?;
        }
        // S2: the resident experts run *before* the staging wait, so the DMA lands under their compute
        // instead of in front of it (measured 78.9 -> 62.8 ms per pass, -20%, and the gate's p50 wait drops
        // from 858 us to 11 us). The staged experts still need the ring, so they keep the gate. This is the
        // one fixed behaviour - there is no switch.
        #[cfg(feature = "cuda")]
        let split = self.staging_enabled();
        #[cfg(not(feature = "cuda"))]
        let split = false;

        #[cfg(feature = "cuda")]
        let resident_acc = if split {
            let residents: Vec<Vec<Hit>> = hits
                .iter()
                .zip(&resident_prefix)
                .map(|(row, &n)| row[..n].to_vec())
                .collect();
            let acc = self.run_hits(&residents, &x2d, act);
            let mut fenced = Ok(());
            for (buf, _, _) in residents.iter().flatten() {
                if let Err(e) = buf.fence.record_reads(&self.slot_device) {
                    fenced = fenced.and(Err(e));
                }
            }
            if fenced.is_err() {
                let _ = self.slot_device.synchronize();
            }
            fenced?;
            Some(acc?)
        } else {
            None
        };

        // M4: ... and the staging point, which every pass publishes.
        #[cfg(feature = "cuda")]
        if self.staging_enabled() {
            self.replay_gate(crate::doorbell::SYNC_STAGING)?;
        }

        #[cfg(feature = "cuda")]
        let hits_to_run: Vec<Vec<Hit>> = if split {
            hits.iter()
                .zip(&resident_prefix)
                .map(|(row, &n)| row[n..].to_vec())
                .collect()
        } else {
            Vec::new()
        };
        #[cfg(feature = "cuda")]
        let acc = if let Some((resident_out, resident_bank)) = resident_acc {
            if hits_to_run.iter().all(|row| row.is_empty()) {
                Ok((resident_out, resident_bank))
            } else {
                let (staged_out, staged_bank) = self.run_hits(&hits_to_run, &x2d, act)?;
                Ok(((resident_out + staged_out)?, resident_bank.or(staged_bank)))
            }
        } else {
            self.run_hits(&hits, &x2d, act)
        };
        #[cfg(not(feature = "cuda"))]
        let acc = self.run_hits(&hits, &x2d, act);
        // Kernels reading these slots may be queued even if a later step failed: fence every
        // slot before letting go of it, so an in-place fill cannot overtake them. If a fence
        // cannot be recorded, wait for the device instead.
        let mut fenced = Ok(());
        for (buf, _, _) in hits.iter().flatten() {
            if let Err(e) = buf.fence.record_reads(&self.slot_device) {
                fenced = fenced.and(Err(e));
            }
        }
        if fenced.is_err() {
            let _ = self.slot_device.synchronize();
        }
        drop(hits);
        #[allow(unused_variables)]
        let (gpu_out, capture_bank) = acc?;
        fenced?;


        // Queue the misses' upload even if the hits failed, so the worker's buffer is released.
        let cpu_out = cpu.map(deferred::Pending::upload).transpose();
        let sum = match cpu_out? {
            None => gpu_out,
            Some(c) => (gpu_out + c)?,
        };
        sum.to_dtype(xs.dtype())?.reshape((batch, seq, hidden))
    }
}
