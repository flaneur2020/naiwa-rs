//! `--prefill-stream`: run prefill's routed experts on the GPU, uploading each layer's experts
//! while the previous layer computes.
//!
//! Two device buffers each hold one whole layer of experts: the three stacked `[E, N, K]`
//! projections, uploaded verbatim from the pinned host stacks (one copy per projection), and a
//! `MoEExperts` over them, so the GPU runs mistral.rs's own MoE kernels. When layer `l`'s MoE
//! runs, everything below is queued without blocking the host:
//!
//! 1. Layer `l + 1` is uploaded into the other buffer on the copy stream, after that buffer's
//!    last readers (layer `l - 1`) are done. It overlaps layer `l`'s MoE and layer `l + 1`'s
//!    attention.
//! 2. candle's stream waits for layer `l`'s upload.
//! 3. Layer `l`'s experts run on the GPU; the buffer's read fence is recorded after them.
//!
//! The first layer of each chunk has no predecessor to prefetch it, so its upload is not
//! hidden. Every chunk streams every layer's experts once, so this only pays for long chunks:
//! shorter ones (`--prefill-stream-min-tokens`) and layers whose stacks are not pinned stay on
//! the CPU.
//!
//! With `--expert-cache-gb`, the two buffers do not add to VRAM: before the first long chunk
//! the cache lends (frees) about two layers' worth of its tail slots, and after
//! `GIVE_BACK_AFTER_DECODE_STEPS` decode steps in a row the buffers are freed and the cache
//! takes the slots back, refilling them in the background. candle frees into a CUDA memory
//! pool, so both hand-overs synchronize candle's stream once in between.

use std::sync::{Arc, Mutex};

use candle_core::quantized::QTensor;
use candle_core::{Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::ExpertLoadContext;
use mistralrs_core::MoEExperts;
use mistralrs_quant::Comm;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::copy_stream::{CopyDone, CopyStream, CopyStreams, ReadFence, SlotMemory};

/// Default for `--prefill-stream-min-tokens`: chunks shorter than this stay on the CPU (Strata
/// streams every expert from 1024 tokens).
pub const DEFAULT_MIN_TOKENS: usize = 1024;

/// Decode steps in a row after which the buffers are given back to the expert cache. Several,
/// so a server alternating prompts and decodes does not hand two layers back and forth per step.
const GIVE_BACK_AFTER_DECODE_STEPS: usize = 8;

/// Device memory that can make room for the prefill buffers: the expert cache.
pub trait ExpertMemoryLender: Send + Sync {
    /// Free about `bytes` of device memory; returns the bytes actually freed.
    fn lend(&self, bytes: usize) -> candle_core::Result<usize>;
    /// Take the memory back (the buffers have been freed).
    fn give_back(&self) -> candle_core::Result<()>;
}

#[derive(Default)]
struct Loan {
    lent: bool,
    decode_steps: usize,
}

/// One layer's host stacks (gate, up, down). Holding them keeps the memory that in-flight
/// uploads read alive.
struct HostStacks([Arc<QTensor>; 3]);

struct LayerBuf {
    /// The layer uploaded (or being uploaded) into this buffer.
    loaded: Option<usize>,
    memory: Option<[SlotMemory; 3]>,
    experts: Option<MoEExperts>,
    /// Recorded on candle's stream after the kernels that read this buffer.
    released: ReadFence,
    /// Recorded on the copy stream after the upload into this buffer.
    ready: CopyDone,
}

pub struct PrefillStream {
    device: Device,
    copy: Arc<CopyStream>,
    num_experts: usize,
    top_k: usize,
    act: Activation,
    comm: Arc<Comm>,
    layers: RwLock<Vec<Option<HostStacks>>>,
    bufs: [Mutex<LayerBuf>; 2],
    /// `None`: the buffers have their own memory and are kept for good.
    lender: Option<Arc<dyn ExpertMemoryLender>>,
    loan: Mutex<Loan>,
    /// Chunks shorter than this stay on the CPU.
    min_tokens: usize,
}

impl PrefillStream {
    /// `None` when `device` has no copy stream (not CUDA).
    pub fn new(
        device: &Device,
        copy: Option<Arc<CopyStream>>,
        num_experts: usize,
        top_k: usize,
        act: Activation,
        comm: Arc<Comm>,
        lender: Option<Arc<dyn ExpertMemoryLender>>,
        min_tokens: usize,
    ) -> candle_core::Result<Option<Self>> {
        let Some(copy) = copy else {
            return Ok(None);
        };
        let buf = || -> candle_core::Result<Mutex<LayerBuf>> {
            Ok(Mutex::new(LayerBuf {
                loaded: None,
                memory: None,
                experts: None,
                released: ReadFence::new(device)?,
                ready: CopyDone::new(&copy)?,
            }))
        };
        Ok(Some(Self {
            device: device.clone(),
            bufs: [buf()?, buf()?],
            copy,
            num_experts,
            top_k,
            act,
            comm,
            layers: RwLock::default(),
            lender,
            loan: Mutex::default(),
            min_tokens,
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

    fn stacks(&self, layer: usize) -> Option<[Arc<QTensor>; 3]> {
        self.layers
            .read()
            .get(layer)
            .and_then(|s| s.as_ref())
            .map(|s| s.0.clone())
    }

    /// Bytes of the two buffers: twice the largest registered layer.
    fn buffer_bytes(&self) -> usize {
        let largest = self
            .layers
            .read()
            .iter()
            .flatten()
            .map(|s| s.0.iter().map(|t| t.storage_size_in_bytes()).sum::<usize>())
            .max()
            .unwrap_or(0);
        2 * largest
    }

    fn first_layer(&self) -> Option<usize> {
        self.layers.read().iter().position(Option::is_some)
    }

    /// Before the buffers are first allocated, have the lender make room for them.
    fn borrow(&self) -> candle_core::Result<()> {
        let Some(lender) = &self.lender else {
            return Ok(());
        };
        let mut loan = self.loan.lock().unwrap();
        if loan.lent {
            return Ok(());
        }
        let need = self.buffer_bytes();
        let freed = lender.lend(need)?;
        info!(
            "prefill stream: borrowed {} MiB of expert-cache slots for {} MiB of layer buffers",
            freed >> 20,
            need >> 20
        );
        if freed < need {
            warn!(
                "prefill stream: the expert cache could lend only {} of {} MiB; the layer \
                 buffers exceed --expert-cache-gb by the rest",
                freed >> 20,
                need >> 20
            );
        }
        // Freed slots go back to candle's memory pool; let them land before allocating.
        self.device.synchronize()?;
        loan.lent = true;
        loan.decode_steps = 0;
        Ok(())
    }

    /// Called by every decode step (one token per sequence). After enough of them in a row,
    /// free the buffers and give the memory back to the lender.
    pub fn note_decode(&self, layer: usize) -> candle_core::Result<()> {
        let Some(lender) = &self.lender else {
            return Ok(());
        };
        if self.first_layer() != Some(layer) {
            return Ok(());
        }
        let mut loan = self.loan.lock().unwrap();
        if !loan.lent {
            return Ok(());
        }
        loan.decode_steps += 1;
        if loan.decode_steps < GIVE_BACK_AFTER_DECODE_STEPS {
            return Ok(());
        }
        self.release()?;
        self.device.synchronize()?;
        lender.give_back()?;
        loan.lent = false;
        loan.decode_steps = 0;
        info!("prefill stream: gave the layer buffers back to the expert cache");
        Ok(())
    }

    /// Free both buffers, each after its last upload (readers are ordered by candle's stream).
    fn release(&self) -> candle_core::Result<()> {
        for buf in &self.bufs {
            let mut buf = buf.lock().unwrap();
            if buf.memory.is_some() {
                buf.ready.wait_on(&self.device)?;
            }
            buf.experts = None;
            buf.memory = None;
            buf.loaded = None;
        }
        Ok(())
    }

    /// Queue the upload of `layer` into its buffer, unless it is already there. `false`: the
    /// layer is not registered.
    fn load(&self, layer: usize) -> candle_core::Result<bool> {
        let Some(stacks) = self.stacks(layer) else {
            return Ok(false);
        };
        let mut buf = self.bufs[layer % 2].lock().unwrap();
        if buf.loaded == Some(layer) {
            return Ok(true);
        }
        buf.loaded = None;
        // (Re)allocate when the layer's stacks differ in shape or dtype (mixed-quant GGUFs).
        let fits = buf.memory.as_ref().is_some_and(|m| {
            m.iter()
                .zip(&stacks)
                .all(|(m, s)| m.tensor().dtype() == s.dtype() && m.tensor().shape() == s.shape())
        });
        if !fits {
            if buf.memory.is_some() {
                // The old memory is freed in candle's stream order; an upload into it may still
                // be in flight on the copy stream, so order the free after it.
                buf.ready.wait_on(&self.device)?;
            }
            buf.experts = None;
            buf.memory = None;
            let memory = [
                self.copy.alloc_like(&stacks[0])?,
                self.copy.alloc_like(&stacks[1])?,
                self.copy.alloc_like(&stacks[2])?,
            ];
            // The allocations are queued on candle's stream: the copy waits for them.
            self.copy.after_device_work()?;
            let bytes: usize = stacks.iter().map(|s| s.storage_size_in_bytes()).sum();
            debug!(
                "prefill stream: {} MiB buffer for layer {layer}'s experts",
                bytes >> 20
            );
            buf.experts = Some(MoEExperts::from_qtensors(
                Arc::clone(memory[0].tensor()),
                Arc::clone(memory[1].tensor()),
                Arc::clone(memory[2].tensor()),
                self.num_experts,
                self.top_k,
                &self.comm,
                self.act,
            )?);
            buf.memory = Some(memory);
        }
        let memory = buf.memory.as_ref().expect("allocated above");
        let data = [stacks[0].data()?, stacks[1].data()?, stacks[2].data()?];
        let parts = [
            (&memory[0], &data[0][..]),
            (&memory[1], &data[1][..]),
            (&memory[2], &data[2][..]),
        ];
        // SAFETY: registered stacks are page-locked, and `self.layers` keeps them alive and
        // unmodified for as long as this stream exists (`Drop` waits for the copies).
        unsafe { self.copy.upload_async(&buf.released, &parts, &buf.ready)? };
        buf.loaded = Some(layer);
        Ok(true)
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
            self.note_decode(layer)?;
            return Ok(None);
        }
        self.loan.lock().unwrap().decode_steps = 0;
        if batch * seq < self.min_tokens || self.stacks(layer).is_none() {
            return Ok(None);
        }
        self.borrow()?;
        if !self.load(layer)? {
            return Ok(None);
        }
        // Stream the next layer in while this one computes.
        if let Err(e) = self.load(layer + 1) {
            warn!(
                "prefill stream: could not prefetch layer {}: {e}",
                layer + 1
            );
        }
        let buf = self.bufs[layer % 2].lock().unwrap();
        let experts = buf.experts.as_ref().expect("loaded");
        buf.ready.wait_on(&self.device)?;
        let out = experts.forward(xs, topk_weights.clone(), topk_ids);
        // Fence even on error: kernels reading the buffer may already be queued.
        buf.released.record_reads(&self.device)?;
        out.map(Some)
    }
}

/// The prefill stream of each CUDA device, created by the first layer that loads on it.
pub struct PrefillStreams {
    streams: Mutex<Vec<(Device, Arc<PrefillStream>)>>,
    lender: Option<Arc<dyn ExpertMemoryLender>>,
    min_tokens: usize,
}

impl PrefillStreams {
    /// Streams that run prefill chunks of `min_tokens` or more on the GPU.
    pub fn new(min_tokens: usize) -> Self {
        Self {
            streams: Mutex::default(),
            lender: None,
            min_tokens,
        }
    }

    /// Borrow the buffers' memory from `lender` instead of keeping it for good.
    pub fn with_lender(mut self, lender: Arc<dyn ExpertMemoryLender>) -> Self {
        self.lender = Some(lender);
        self
    }

    /// Register `experts` (this block's CPU stacks) for streaming and return the device's
    /// stream. `None` for the MTP head, for stacks that are not all page-locked (`pinned`), and
    /// off CUDA: such a layer keeps its prefill on the CPU.
    pub fn attach(
        &self,
        ctx: &ExpertLoadContext<'_>,
        experts: &MoEExperts,
        pinned: bool,
        copy_streams: &CopyStreams,
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
                    self.lender.clone(),
                    self.min_tokens,
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

    /// The fork's `MoEExperts::from_qtensors` computes the routed MoE of the stacks it is
    /// given: compare against a plain dequantized reference, with the full top-k and with a
    /// subset of it (the miss path's shape).
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
        // Expert `ex` of a stack as its own QTensor (its byte slice), run through `QMatMul`.
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
                let x = xs.i((0, t)).unwrap().unsqueeze(0).unwrap(); // [1, hidden]
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
        // The experts' fused kernel and `QMatMul` quantize activations alike on aarch64 but not
        // on x86 (where the fused kernel stays in f32), so they agree only to Q8_0 rounding
        // (~0.008 here). A swapped routing weight or wrong expert moves the output by >0.3.
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

        // full top-k
        let ids_t = Tensor::new(&ids, &Device::Cpu).unwrap();
        let ws_t = Tensor::new(&ws, &Device::Cpu).unwrap();
        let got = experts.forward(&xs, ws_t, &ids_t).unwrap();
        let want = reference(&ids, &ws, 2);
        assert_eq!(got.dims(), want.dims());
        assert!(max_diff(&got, &want) < TOL, "full top-k diverges");

        // a subset of the routed experts (one per token)
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
