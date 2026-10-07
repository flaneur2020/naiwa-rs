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
//! shorter ones (`MIN_TOKENS`) and layers whose stacks are not pinned stay on the CPU.

use std::sync::{Arc, Mutex};

use candle_core::quantized::QTensor;
use candle_core::{Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::ExpertLoadContext;
use mistralrs_core::MoEExperts;
use mistralrs_quant::Comm;
use parking_lot::RwLock;
use tracing::{debug, warn};

use crate::copy_stream::{CopyDone, CopyStream, CopyStreams, ReadFence, SlotMemory};

/// Chunks shorter than this stay on the CPU (Strata streams every expert from 1024 tokens).
pub const MIN_TOKENS: usize = 1024;

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
        if batch * seq < MIN_TOKENS || !self.load(layer)? {
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
#[derive(Default)]
pub struct PrefillStreams(Mutex<Vec<(Device, Arc<PrefillStream>)>>);

impl PrefillStreams {
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
        let mut streams = self.0.lock().unwrap();
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
        // Expert `ex` of a stack as its own QTensor (its byte slice): the reference then runs
        // the same quantized matmul, with the same activation quantization, as the experts do.
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
        assert!(max_diff(&got, &want) < 1e-4, "full top-k diverges");

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
        assert!(max_diff(&got, &want) < 1e-4, "subset of top-k diverges");
    }
}
