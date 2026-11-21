//! `--prefill-stream`: run prefill's routed experts on the GPU, uploading each layer's experts
//! while the previous layer computes.
//!
//! Submodules:
//! - [`planner`] — pure VRAM budget math (`plan*`, `chunk_scratch`, `kv_reserve`).
//! - [`buffers`] — layer buffer lifecycle, cache-slab borrowing, chunk begin/zero/empty, upload, downgrade.
//! - [`split`] — `--prefill-split` expert sub-range views and masked launches.
//! - [`diagnostics`] — opt-in peak and wall-clock probes.

mod buffers;
mod diagnostics;
mod planner;
mod split;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::{Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::ExpertLoadContext;
use mistralrs_core::MoEExperts;
use mistralrs_quant::Comm;
use parking_lot::RwLock;
use tracing::{info, warn};

use crate::transport::copy_stream::{CopyDone, CopyStream, CopyStreams, ReadFence};

use self::buffers::{is_out_of_memory, HostStacks, LayerBuf};
use self::diagnostics::Peak;
use self::planner::{log_plan, prompt_step_ceiling, Plan, MAX_DEPTH};

pub use self::planner::{set_prompt_step_ceiling, DEFAULT_DEPTH, DEFAULT_MIN_TOKENS};

pub struct PrefillStream {
    pub(super) device: Device,
    pub(super) copy: Arc<CopyStream>,
    pub(super) num_experts: usize,
    pub(super) top_k: usize,
    /// `--prefill-split g`: the layer's experts are consumed in `g` sub-ranges (M34 S2). 1 is the whole layer.
    pub(super) split: usize,
    pub(super) act: Activation,
    pub(super) comm: Arc<Comm>,
    pub(in crate::pipeline::prefill) layers: RwLock<Vec<Option<HostStacks>>>,
    pub(in crate::pipeline::prefill) bufs: Vec<Mutex<LayerBuf>>,
    /// `--prefill-depth`: the most buffers the plan may use.
    pub(super) depth: usize,
    /// The memory plan in force (buffers and prompt step), from warmup and re-checked at the first long chunk.
    pub(super) plan: Mutex<Option<Plan>>,
    /// Set once the stream has been refused because it cannot fit.
    pub(super) refused: AtomicBool,
    /// The first long chunk has re-checked the plan against the free VRAM it actually sees.
    pub(super) checked: AtomicBool,
    /// Chunks shorter than this stay on the CPU.
    min_tokens: usize,
    pub(in crate::pipeline::prefill) peak: Mutex<Peak>,
    last_entry: Mutex<Option<Instant>>,
    pub(super) slab: Option<Arc<crate::memory::slab::Slab>>,
    pub(super) cache: Option<Arc<crate::memory::Layers>>,
    pub(super) borrow_gate: Arc<crate::memory::slab::BorrowGate>,
}

impl PrefillStream {
    /// `None` when `device` has no copy stream (not CUDA).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        copy: Option<Arc<CopyStream>>,
        num_experts: usize,
        top_k: usize,
        act: Activation,
        comm: Arc<Comm>,
        min_tokens: usize,
        depth: usize,
        split: usize,
        slab: Option<Arc<crate::memory::slab::Slab>>,
        cache: Option<Arc<crate::memory::Layers>>,
        borrow_gate: Arc<crate::memory::slab::BorrowGate>,
    ) -> candle_core::Result<Option<Self>> {
        let Some(copy) = copy else {
            return Ok(None);
        };
        let buf = || -> candle_core::Result<Mutex<LayerBuf>> {
            Ok(Mutex::new(LayerBuf {
                loaded: None,
                queued: None,
                bytes: 0,
                copies: 0,
                memory: None,
                experts: None,
                released: ReadFence::new(device)?,
                ready: CopyDone::new(&copy)?,
            }))
        };
        let depth = depth.clamp(1, MAX_DEPTH);
        Ok(Some(Self {
            last_entry: Mutex::new(None),
            device: device.clone(),
            bufs: (0..depth)
                .map(|_| buf())
                .collect::<candle_core::Result<Vec<_>>>()?,
            depth,
            copy,
            num_experts,
            top_k,
            split: split.clamp(1, 8),
            act,
            comm,
            layers: RwLock::default(),
            plan: Mutex::new(None),
            refused: AtomicBool::new(false),
            checked: AtomicBool::new(false),
            min_tokens,
            peak: Mutex::new(Peak::default()),
            slab,
            cache,
            borrow_gate,
        }))
    }

    /// Make `layer`'s host stacks available for streaming. Only call with stacks that are
    /// page-locked: an upload from pageable memory would block the forward thread.
    pub fn register(&self, layer: usize, experts: &MoEExperts) {
        let Some((gate, up, down)) = experts.expert_qtensors() else {
            return;
        };
        let mut layers = self.layers.write();
        if layers.len() <= layer {
            layers.resize_with(layer + 1, || None);
        }
        layers[layer] = Some(HostStacks([gate, up, down]));
    }

    /// The warmup decision: run the plan once the cache, KV, and mapped buffers exist, keep
    /// it, and log the arithmetic. Returns the prompt step it chose.
    pub(super) fn plan(&self) -> Option<usize> {
        let (decision, free) = self.decide()?;
        *self.plan.lock().unwrap() = Some(decision);
        log_plan(self, decision, free, "warmup");
        Some(decision.chunk)
    }

    /// Run `layer`'s routed experts on the GPU. `None`: this chunk or layer stays on the CPU.
    pub fn forward(
        &self,
        layer: usize,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (batch, seq, _) = xs.dims3()?;
        if seq == 1 {
            return Ok(None);
        }
        if self.is_refused() {
            candle_core::bail!(
                "prefill stream refused: it did not fit at startup and NAIWA_PREFILL_ALLOW_CPU was not set, so \
                 this prompt cannot run (drop --expert-cache-gb, see the startup warning)"
            );
        }
        let rows = batch * seq;
        if rows < self.min_tokens || self.stacks(layer).is_none() {
            crate::dbg_note(
                "prefill_stream",
                &format!(
                    "layer {layer}: {rows} rows below the {}-row stream threshold - CPU path",
                    self.min_tokens
                ),
            );
            if rows >= self.min_tokens {
                crate::dbg_note(
                    "prefill_stream",
                    &format!(
                        "layer {layer}: declined {rows} rows - no host stacks registered (registered: {})",
                        self.layers.read().iter().filter(|l| l.is_some()).count()
                    ),
                );
            }
            return Ok(None);
        }

        self.note_req_time(layer);
        let entry = Instant::now();
        let gap = {
            let mut last = self.last_entry.lock().unwrap();
            let prev = if layer == 0 { None } else { *last };
            *last = Some(entry);
            prev.map(|p| entry.duration_since(p))
        };
        self.sample_peak(layer, rows);

        let mut stride = self.reserve()?;
        if stride == 0 {
            return Ok(None);
        }
        if layer == 0 && !self.borrows() {
            crate::pipeline::quiesce(true);
        }
        self.begin_chunk(layer)?;

        let loaded = match self.load(layer, stride) {
            Ok(loaded) => loaded,
            Err(e) if is_out_of_memory(&e) => {
                self.downgrade(stride)?;
                stride = stride.saturating_sub(1);
                if stride == 0 {
                    return Ok(None);
                }
                self.load(layer, stride)?
            }
            Err(e) => return Err(e),
        };
        if !loaded {
            return Ok(None);
        }

        let mut prefetch = Duration::ZERO;
        if stride > 1 {
            let t = Instant::now();
            for ahead in 1..stride {
                if let Err(e) = self.load(layer + ahead, stride) {
                    warn!(
                        "prefill stream: could not prefetch layer {}: {e}",
                        layer + ahead
                    );
                }
            }
            prefetch = t.elapsed();
        }

        let buf = self.bufs[layer % stride].lock().unwrap();
        let experts = buf.experts.as_ref().expect("loaded");
        let t_wait = Instant::now();
        buf.ready.wait_on(&self.device)?;
        let waited = t_wait.elapsed();
        if let Some(us) = buf.ready.micros() {
            crate::dbg_note(
                "prefill_copy",
                &format!(
                    "layer {layer}: {} MiB in {:.1} ms, {:.1} GB/s (own), queue-to-consumer {:?}",
                    buf.bytes >> 20,
                    us / 1e3,
                    buf.bytes as f64 / (us as f64 * 1e3) * 1e9,
                    buf.queued.map(|q| q.elapsed()).unwrap_or_default()
                ),
            );
        }

        let t_compute = Instant::now();
        let out = self.dispatch(experts, &buf, xs, topk_weights, topk_ids);
        self.log_copy_and_layer_timing(layer, rows, stride, gap, prefetch, waited, t_compute, &buf);

        // Fence even on error: kernels reading the buffer may already be queued.
        buf.released.record_reads(&self.device)?;
        if self.borrows() {
            self.copy.after_device_work()?;
            self.copy.synchronize()?;
        }
        out.map(Some)
    }
}

/// The prefill stream of each CUDA device, created by the first layer that loads on it.
pub struct PrefillStreams {
    streams: Mutex<Vec<(Device, Arc<PrefillStream>)>>,
    min_tokens: usize,
    depth: usize,
    split: usize,
}

impl PrefillStreams {
    /// Streams that run prefill chunks of `min_tokens` or more on the GPU.
    pub fn new(min_tokens: usize, depth: usize, split: usize) -> Self {
        Self {
            streams: Mutex::default(),
            min_tokens,
            depth,
            split,
        }
    }

    /// Run the memory plan for every device's stream and publish the chunk it chose to the
    /// scheduler (the smallest across devices). `None`: no stream exists, or no chunk fits.
    pub fn plan(&self) -> Option<usize> {
        let streams = self.streams.lock().unwrap();
        let chunk = streams
            .iter()
            .filter_map(|(_, stream)| stream.plan())
            .min()?;
        drop(streams);
        mistralrs_core::PROMPT_STEP_CEILING.store(chunk, Ordering::Release);
        info!(
            "prefill: the scheduler's prompt step is capped at {chunk} tokens (from \
             --max-num-batched-tokens/--max-prefill-chunk-tokens {})",
            prompt_step_ceiling()
        );
        Some(chunk)
    }

    /// Register `experts` (this block's CPU stacks) for streaming and return the device's
    /// stream. `None` for the MTP head, for stacks that are not all page-locked (`pinned`), and
    /// off CUDA: such a layer keeps its prefill on the CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn attach(
        &self,
        ctx: &ExpertLoadContext<'_>,
        experts: &MoEExperts,
        pinned: bool,
        copy_streams: &CopyStreams,
        slab: Option<Arc<crate::memory::slab::Slab>>,
        cache: Option<Arc<crate::memory::Layers>>,
        borrow_gate: Arc<crate::memory::slab::BorrowGate>,
    ) -> candle_core::Result<Option<Arc<PrefillStream>>> {
        if ctx.site.is_mtp || !pinned || experts.expert_qtensors().is_none() {
            return Ok(None);
        }
        let device = ctx.layer_device;
        let mut streams = self.streams.lock().unwrap();
        let stream = match streams.iter().find(|(d, _)| d.same_device(device)) {
            Some((_, ps)) => Arc::clone(ps),
            None => {
                let Some(ps) = PrefillStream::new(
                    device,
                    copy_streams.get(device),
                    ctx.config.num_experts,
                    ctx.config.num_experts_per_tok,
                    ctx.act,
                    Arc::clone(ctx.comm),
                    self.min_tokens,
                    self.depth,
                    self.split,
                    slab,
                    cache,
                    borrow_gate,
                )?
                else {
                    return Ok(None);
                };
                let ps = Arc::new(ps);
                streams.push((device.clone(), Arc::clone(&ps)));
                ps
            }
        };
        stream.register(ctx.site.layer, experts);
        Ok(Some(stream))
    }
}

impl Drop for PrefillStream {
    fn drop(&mut self) {
        // In-flight uploads read the host stacks this struct keeps alive.
        let _ = self.copy.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QTensor};
    use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
    use mistralrs_core::layers::Activation;
    use mistralrs_core::MoEExperts;
    use mistralrs_quant::{Comm, Id};

    fn stack(e: usize, n: usize, k: usize, seed: f32) -> Arc<QTensor> {
        let data: Vec<f32> = (0..e * n * k)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect();
        let t = Tensor::from_vec(data, (e, n, k), &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap())
    }

    #[test]
    fn from_qtensors_matches_a_dequantized_reference() {
        let (e, hidden, inter) = (4, 64, 32);
        let (gate, up, down) = (
            stack(e, inter, hidden, 0.1),
            stack(e, inter, hidden, 0.7),
            stack(e, hidden, inter, 1.3),
        );
        let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1).unwrap());
        let act = Activation::Silu;
        let experts =
            MoEExperts::from_qtensors(gate.clone(), up.clone(), down.clone(), e, 2, &comm, act)
                .unwrap();

        let tokens = 3;
        let xs: Vec<f32> = (0..tokens * hidden)
            .map(|i| (i as f32 * 0.11).cos())
            .collect();
        let xs = Tensor::from_vec(xs, (1, tokens, hidden), &Device::Cpu).unwrap();
        let expert = |q: &Arc<QTensor>, ex: usize| -> QMatMul {
            let dims = q.shape().dims();
            let data = q.data().unwrap();
            let stride = data.len() / dims[0];
            let one = qtensor_from_ggml(
                q.dtype(),
                &data[ex * stride..(ex + 1) * stride],
                vec![dims[1], dims[2]],
                &Device::Cpu,
            )
            .unwrap();
            QMatMul::QTensor(Arc::new(one))
        };
        let reference = |ids: &[[u32; 2]], ws: &[[f32; 2]], k: usize| -> Tensor {
            let mut rows = Vec::new();
            for t in 0..tokens {
                let x = xs.i((0, t)).unwrap().unsqueeze(0).unwrap();
                let mut acc = Tensor::zeros((1, hidden), DType::F32, &Device::Cpu).unwrap();
                for j in 0..k {
                    let ex = ids[t][j] as usize;
                    let gx = expert(&gate, ex).forward(&x).unwrap();
                    let ux = expert(&up, ex).forward(&x).unwrap();
                    let h = (act.forward(&gx).unwrap() * ux).unwrap();
                    let y = expert(&down, ex).forward(&h).unwrap();
                    acc = (acc + (y * ws[t][j] as f64).unwrap()).unwrap();
                }
                rows.push(acc);
            }
            Tensor::cat(&rows, 0).unwrap().unsqueeze(0).unwrap()
        };
        const TOL: f32 = 2e-2;
        let ids = [[0u32, 2], [3, 1], [2, 0]];
        let ws = [[0.6f32, 0.4], [0.3, 0.7], [0.5, 0.5]];
        let max_diff = |a: &Tensor, b: &Tensor| -> f32 {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .max_keepdim(D::Minus1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };

        let ids_t = Tensor::new(&ids, &Device::Cpu).unwrap();
        let ws_t = Tensor::new(&ws, &Device::Cpu).unwrap();
        let got = experts.forward(&xs, ws_t, &ids_t).unwrap();
        let want = reference(&ids, &ws, 2);
        assert_eq!(got.dims(), want.dims());
        assert!(max_diff(&got, &want) < TOL, "full top-k diverges");

        let sub_ids: Vec<u32> = ids.iter().map(|r| r[0]).collect();
        let sub_ws: Vec<f32> = ws.iter().map(|r| r[0]).collect();
        let got = experts
            .forward(
                &xs,
                Tensor::from_vec(sub_ws, (tokens, 1), &Device::Cpu).unwrap(),
                &Tensor::from_vec(sub_ids, (tokens, 1), &Device::Cpu).unwrap(),
            )
            .unwrap();
        let want = reference(&ids, &ws, 1);
        assert!(max_diff(&got, &want) < TOL, "subset of top-k diverges");
    }
}
