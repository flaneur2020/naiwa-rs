//! The decode path: split top-k into GPU hits and CPU misses, run both, sum.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use candle_core::Tensor;
use mistralrs_core::layers::Activation;
use tracing::warn;

use crate::components::{run_gpu, Hit};
use crate::stats;

use crate::moe::{schedule, CachedLayer};

impl CachedLayer {
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

    pub(crate) fn decode_forward(
        self: &Arc<Self>,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        act: Activation,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq, hidden) = xs.dims3()?;
        let rows = batch * seq;
        let x2d = xs.reshape((rows, hidden))?;
        stats::record_step_layer(self.slot_device(), self.layer());
        // M3/M4: this layer's pass index. Bumped once per pass at layer 0, together with the device
        // epoch, and only once the schedule exists — so the two counters cannot drift. It has to be
        // known before the DMA, because that is what publishes the staging point for this pass.
        #[cfg(feature = "cuda")]
        let pass = {
            static PASS: AtomicU32 = AtomicU32::new(0);
            if self.layer() == 0 {
                match schedule() {
                    Some(sched) => {
                        let p = PASS.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Err(e) = sched.bump_epoch(self.slot_device()) {
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
        crate::components::submit_preload(self, &x2d, rows, hidden, pass)?;

        let mut plan = crate::components::split_route(self, topk_ids, topk_weights, rows)?;
        let n_hits: usize = plan.hits.iter().map(Vec::len).sum();
        stats::record_cache_hits(n_hits);

        // `--pcie-frac`: DMA the highest-weight misses into the staging ring and run them on the
        // GPU with the hits, leaving the rest on the CPU. `stage_misses` only *appends* to
        // `plan.hits`, so the resident prefix is exactly what was there before the call.
        #[cfg(feature = "cuda")]
        let resident_prefix: Vec<usize> = plan.hits.iter().map(Vec::len).collect();
        if self.pcie_frac() > 0.0 {
            crate::components::stage_dma_miss(self, &mut plan, pass)?;
        }

        let n_cpu: usize = plan.misses.iter().map(Vec::len).sum();
        stats::record_cache_misses(n_cpu);
        // Experts that went to the staging ring: counted apart from both hits and CPU misses.
        #[cfg(feature = "cuda")]
        {
            let staged_n: usize = plan.hits.iter().map(Vec::len).sum::<usize>() - n_hits;
            stats::record_cache_staged(staged_n);
            stats::record_round_plan(n_hits, staged_n, n_cpu);
        }

        let cpu = crate::components::submit_cpu_miss(self, &plan.misses, &x2d, rows)?;

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
        if self.preload_claimed() == pass {
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
            let residents: Vec<Vec<Hit>> = plan
                .hits
                .iter()
                .zip(&resident_prefix)
                .map(|(row, &n)| row[..n].to_vec())
                .collect();
            let acc = run_gpu(self, &residents, &x2d, act);
            let mut fenced = Ok(());
            for (buf, _, _) in residents.iter().flatten() {
                if let Err(e) = buf.fence.record_reads(self.slot_device()) {
                    fenced = fenced.and(Err(e));
                }
            }
            if fenced.is_err() {
                let _ = self.slot_device().synchronize();
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
            plan.hits
                .iter()
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
                let (staged_out, staged_bank) = run_gpu(self, &hits_to_run, &x2d, act)?;
                Ok(((resident_out + staged_out)?, resident_bank.or(staged_bank)))
            }
        } else {
            run_gpu(self, &plan.hits, &x2d, act)
        };
        #[cfg(not(feature = "cuda"))]
        let acc = run_gpu(self, &plan.hits, &x2d, act);
        // Kernels reading these slots may be queued even if a later step failed: fence every
        // slot before letting go of it, so an in-place fill cannot overtake them. If a fence
        // cannot be recorded, wait for the device instead.
        let mut fenced = Ok(());
        for (buf, _, _) in plan.hits.iter().flatten() {
            if let Err(e) = buf.fence.record_reads(self.slot_device()) {
                fenced = fenced.and(Err(e));
            }
        }
        if fenced.is_err() {
            let _ = self.slot_device().synchronize();
        }
        drop(plan.hits);
        #[allow(unused_variables)]
        let (gpu_out, capture_bank) = acc?;
        fenced?;

        // Queue the misses' upload even if the hits failed, so the worker's buffer is released.
        let cpu_out = cpu.map(|p| p.upload()).transpose();
        let sum = match cpu_out? {
            None => gpu_out,
            Some(c) => (gpu_out + c)?,
        };
        sum.to_dtype(xs.dtype())?.reshape((batch, seq, hidden))
    }
}
