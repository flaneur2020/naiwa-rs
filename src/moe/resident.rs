//! The drafter's resident expert pool: the MTP head's 512 experts, uploaded once and never evicted.
//!
//! The head is one block whose experts are its entire working set — a 1.7k-row prompt chunk routes
//! ~33 rows to *each* of the 512 — so there is nothing to cache and nothing to stage: the unique
//! working set **is** the layer. Streaming it would re-read the same bytes once per chunk, and the
//! CPU cannot carry it at all (167 GFLOP for that one chunk; measured 2.7-4.0 s, and a grouped CPU
//! kernel made it worse at 5.8 s because it is compute-bound, not read-bound).
//!
//! They are therefore compressed to **Q2_0** — 2.25 bit/weight, 1,382,400 bytes per expert and 675 MiB
//! for all 512, byte-for-byte the size strata reserves for this layer in the same format — and kept
//! resident, computed through the ordinary GPU expert kernels for **both** phases. Q2_0 is also the
//! dtype this model's own GGUFs use, and the one with the best CUDA coverage (`moe_grouped_gemm_q2_0`,
//! the fused gate/up and down-aggregate gemvs). Its block is 64 elements, which the head's
//! `[2560, 640]` / `[640, 2560]` matrices divide; the 256-element K-quants do not.
//!
//! Drafts are verified by the target, so this trades draft quality only, never output — the same
//! reasoning strata documents for its own 2.25-bit drafter.

use std::borrow::Cow;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{Device, Result, Shape, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::RoutedExperts;
use mistralrs_core::MoEExperts;
use mistralrs_quant::Comm;

/// The resident pool's weight format: 2.25 bit/weight, the model's own expert dtype.
const Q2_0: GgmlDType = GgmlDType::Q2_0;
/// Elements per Q2_0 block, and the packed block's size.
const QK2_0: usize = 64;
const Q2_0_BYTES: usize = 18;
/// Experts re-quantized at a time: bounds the f32 intermediate (one expert is ~6.5 MiB at this size).
const REQUANT_BATCH: usize = 64;

/// Quantize `x` (a multiple of [`QK2_0`] long) into packed Q2_0 blocks: one f16 scale per block and
/// four 2-bit codes per byte decoding to `code - 1` — the set `{-1, 0, +1, +2}` times that scale — which
/// is the convention `mistralrs_quant::gguf::q2_0::dequantize` decodes and the CUDA kernels run.
///
/// The scale is **not** fixed to the block max: `+2` is the only code above zero, so a block-max scale
/// spends one code on an unreachable range and quantizes half the block to zero. The scale is searched
/// over a small ladder instead, keeping whichever minimizes the block's squared error. It is a draft
/// layer, but acceptance is what pays for the format: 4.5 -> 2.25 bit/weight costs ~13% relative
/// acceptance at the naive scale, and the ladder buys a third of that back for free.
fn quantize_q2_0(x: &[f32], out: &mut [u8]) {
    /// Scale candidates as a fraction of the block's max absolute value.
    const LADDER: [f32; 5] = [1.0, 1.25, 1.5, 1.75, 2.0];

    debug_assert!(x.len().is_multiple_of(QK2_0));
    debug_assert_eq!(out.len(), x.len() / QK2_0 * Q2_0_BYTES);
    let mut codes = [0u8; QK2_0];
    for (block, chunk) in x.chunks_exact(QK2_0).enumerate() {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let dst = &mut out[block * Q2_0_BYTES..(block + 1) * Q2_0_BYTES];
        dst[2..].fill(0);
        if amax == 0.0 {
            // Every code decodes to `0 - 1 == -1` times a zero scale.
            dst[..2].copy_from_slice(&0u16.to_le_bytes());
            dst[2..].fill(0b01_01_01_01);
            continue;
        }

        let mut best = (f32::INFINITY, 0f32, [0u8; QK2_0]);
        for fraction in LADDER {
            let scale = half::f16::from_f32(amax / fraction).to_f32();
            if scale == 0.0 {
                continue;
            }
            let inv = 1.0 / scale;
            let mut error = 0f32;
            for (j, &v) in chunk.iter().enumerate() {
                let code = ((v * inv).round() as i32 + 1).clamp(0, 3) as u8;
                codes[j] = code;
                let residual = v - (code as f32 - 1.0) * scale;
                error += residual * residual;
            }
            if error < best.0 {
                best = (error, scale, codes);
            }
        }
        let (_, scale, chosen) = best;
        dst[..2].copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        for (j, &code) in chosen.iter().enumerate() {
            dst[2 + j / 4] |= code << (2 * (j % 4));
        }
    }
}

/// Build the resident pool from the CPU-loaded head. `Ok(None)` when the weights are not
/// `QTensor`-backed (no GGUF quantized stack), so the caller can keep the CPU path.
pub fn build(
    cpu_experts: &MoEExperts,
    device: &Device,
    num_experts: usize,
    top_k: usize,
    comm: &Arc<Comm>,
    act: Activation,
) -> Result<Option<Arc<dyn RoutedExperts>>> {
    let Some((gate, up, down)) = cpu_experts.expert_qtensors() else {
        return Ok(None);
    };
    let mut stacks = Vec::with_capacity(3);
    for src in [gate, up, down] {
        stacks.push(requantize(&src, device)?);
    }
    let experts = MoEExperts::from_qtensors(
        stacks[0].clone(),
        stacks[1].clone(),
        stacks[2].clone(),
        num_experts,
        top_k,
        comm,
        act,
    )?;
    Ok(Some(Arc::new(experts)))
}

/// Re-quantize one expert slice `[out, in]` to Q2_0 and return its packed bytes.
///
/// candle's own quantizer has no Q2_0 arm (its block type is built from GGUF data only), so the packing
/// is done here; every other dtype would be `QTensor::quantize`.
fn requantize_one(bytes: &[u8], src_dtype: GgmlDType, out: usize, inn: usize) -> Result<Vec<u8>> {
    let storage = QStorage::from_data(Cow::Borrowed(bytes), &Device::Cpu, src_dtype)?;
    let block = QTensor::new(storage, (out, inn))?;
    let dense: Tensor = block.dequantize(&Device::Cpu)?;
    let dense = dense.to_dtype(candle_core::DType::F32)?.flatten_all()?;
    let dense = dense.to_vec1::<f32>()?;
    let mut packed = vec![0u8; dense.len() / QK2_0 * Q2_0_BYTES];
    quantize_q2_0(&dense, &mut packed);
    Ok(packed)
}

/// Re-quantize a stacked `[e, out, in]` expert tensor to Q2_0 and place it on `device`, in batches of
/// experts so the f32 intermediate never holds the whole stack.
fn requantize(src: &QTensor, device: &Device) -> Result<Arc<QTensor>> {
    let (num_experts, out, inn) = src.shape().dims3()?;
    let src_dtype = src.dtype();
    let src_bytes = src.data()?;
    let src_stride = src.storage_size_in_bytes() / num_experts;
    if src_bytes.len() != src_stride * num_experts {
        candle_core::bail!(
            "expert stack has {} bytes for {num_experts} experts of {src_stride}",
            src_bytes.len()
        );
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    let mut packed: Vec<u8> = Vec::new();
    for batch in (0..num_experts).collect::<Vec<_>>().chunks(REQUANT_BATCH) {
        let per = batch.len().div_ceil(threads);
        let parts: Vec<Result<Vec<u8>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = batch
                .chunks(per)
                .map(|slice| {
                    let src_bytes = &src_bytes;
                    scope.spawn(move || -> Result<Vec<u8>> {
                        let mut out_bytes = Vec::with_capacity(slice.len() * src_stride / 2);
                        for &index in slice {
                            let bytes = &src_bytes[index * src_stride..(index + 1) * src_stride];
                            out_bytes.extend_from_slice(&requantize_one(bytes, src_dtype, out, inn)?);
                        }
                        Ok(out_bytes)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("requantize thread")).collect()
        });
        for part in parts {
            packed.extend_from_slice(&part?);
        }
    }

    // candle's `QStorage::from_data` refuses Q2_0 on CUDA; the fork's own builder is the supported
    // route (same bytes, same kernels).
    let storage = mistralrs_quant::gguf::q2_0::storage_for_device(&packed, device)?;
    let shape = Shape::from_dims(&[num_experts, out, inn]);
    Ok(Arc::new(QTensor::new(storage, shape)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packer has to be the exact inverse of the fork's Q2_0 decoder — that decoder is what the
    /// kernels run, so a mismatch would silently shift every expert's weights.
    #[test]
    fn q2_0_pack_round_trips_through_the_kernel_decoder() {
        let x: Vec<f32> = (0..4096)
            .map(|i| ((i as f32) * 0.017).sin() * 1.5 - 0.25)
            .collect();
        let mut packed = vec![0u8; x.len() / QK2_0 * Q2_0_BYTES];
        quantize_q2_0(&x, &mut packed);
        let mut back = vec![0f32; x.len()];
        mistralrs_quant::gguf::q2_0::dequantize(&packed, &mut back).expect("decodes");

        // 2 bits with scale = block max: the decoded set is {-1,0,1,2} x max, so ~0.1 of the amplitude.
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = x
            .iter()
            .zip(&back)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= amax * 0.55,
            "worst per-element error {err} exceeds half a step ({})",
            amax * 0.55
        );
        // A zero block must decode to zero, not to `-d`.
        let zeros = vec![0f32; QK2_0 * 2];
        let mut packed = vec![0u8; zeros.len() / QK2_0 * Q2_0_BYTES];
        quantize_q2_0(&zeros, &mut packed);
        let mut back = vec![0f32; zeros.len()];
        mistralrs_quant::gguf::q2_0::dequantize(&packed, &mut back).expect("decodes");
        assert!(back.iter().all(|v| *v == 0.0), "zero block decoded to {back:?}");
    }
}
