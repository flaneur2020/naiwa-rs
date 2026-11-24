//! Direct GGUF weight loader for Qwen3.8-Flash-Next (`qwen4exp`).
//!
//! Opens one or more GGUF shards / components via [`GgufArchive::open_components`], parses
//! [`Qwen4ExpConfig`] directly from metadata, and materializes every layer onto `device` with
//! pluggable [`RoutedExpertsProvider`] injection for MoE offloading.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::{quantized::QTensor, DType, Device, Tensor};
use naiwa_attn::{GatedDeltaNet, RotaryEmbedding};
use naiwa_quant::{
    Activation, ExpertLoadRequest, ExpertSite, GgufArchive, MoEExperts, MoEExpertsConfig, QLinear,
    RoutedExperts, RoutedExpertsProvider,
};

use crate::qwen4_exp::{
    config::{LayerType, Qwen4ExpConfig},
    hyper::GatedResidual,
    moe::{SharedMlp, SparseMoeBlock},
    mtp::Qwen4ExpMtpHead,
    ple::{PleHash, PleLayer, PleTable},
    qsa::QsaAttention,
    text::{DecoderLayer, Mixer, Qwen4ExpModel},
};

/// Open one or more GGUF files/components (handling split shards + merged MTP component) into a
/// single [`GgufArchive`].
pub fn open_gguf_archive<P: AsRef<Path>>(paths: &[P]) -> Result<Arc<GgufArchive>> {
    if paths.is_empty() {
        bail!("at least one GGUF file is required");
    }
    let mut components = GgufArchive::open_components(paths)
        .context("failed to open GGUF archive components")?;
    let mut base = components.remove(0);
    if !components.is_empty() {
        base.merge_components(components)
            .context("failed to merge GGUF components")?;
    }
    Ok(Arc::new(base))
}

/// Load a GGUF tensor directly into a dense `Tensor` on `device` in `target_dtype`.
///
/// For unquantized `F32`/`F16`/`BF16` tensors, reads raw mmap bytes directly without a redundant
/// GPU->CPU->GPU `QTensor::dequantize` round-trip.
fn load_dense(
    archive: &GgufArchive,
    name: &str,
    device: &Device,
    target_dtype: DType,
) -> Result<Tensor> {
    let data = archive
        .tensor_data(name)
        .with_context(|| format!("missing GGUF tensor `{name}`"))?;
    let info = data.info();
    let raw = info.dtype().raw();
    let src_dtype = match raw {
        0 => Some(DType::F32),
        1 => Some(DType::F16),
        30 => Some(DType::BF16),
        _ => None,
    };
    if let Some(dt) = src_dtype {
        let t = Tensor::from_raw_buffer(data.bytes(), dt, info.shape(), &Device::Cpu)?
            .to_device(device)?;
        if dt != target_dtype {
            return Ok(t.to_dtype(target_dtype)?);
        }
        return Ok(t);
    }
    let qt = archive.load_qtensor(name, device)?;
    Ok(qt.dequantize(device)?.to_dtype(target_dtype)?)
}

/// Load a tensor from `archive` as a `QLinear` on `device`.
///
/// Unquantized F32/F16/BF16 weights are kept as dense `QLinear::Dense` in `model_dtype` (so cuBLAS
/// GEMM runs at full speed); quantized weights stay as `QLinear::Quantized` (`QTensor`).
fn load_qlinear(
    archive: &GgufArchive,
    name: &str,
    device: &Device,
    model_dtype: DType,
) -> Result<QLinear> {
    let info = archive
        .tensor_info(name)
        .with_context(|| format!("missing GGUF tensor `{name}`"))?;
    let raw = info.dtype().raw();
    if matches!(raw, 0 | 1 | 30) {
        let dense = load_dense(archive, name, device, model_dtype)?;
        Ok(QLinear::from_dense(dense, None))
    } else {
        let qt = Arc::new(archive.load_qtensor(name, device)?);
        Ok(QLinear::from_qtensor(qt, None))
    }
}

fn load_hyper(
    archive: &GgufArchive,
    norm_name: &str,
    down_name: &str,
    up_name: &str,
    inject_name: Option<&str>,
    cfg: &Qwen4ExpConfig,
    device: &Device,
    dtype: DType,
) -> Result<GatedResidual> {
    // GGUF stores shifted norms as `(1 + w)` already — load directly as F32 without shifting!
    let norm_weight = load_dense(archive, norm_name, device, DType::F32)?;
    let down = load_qlinear(archive, down_name, device, dtype)?;
    let up = load_qlinear(archive, up_name, device, dtype)?;
    let inject = inject_name
        .map(|n| load_qlinear(archive, n, device, dtype))
        .transpose()?;
    Ok(GatedResidual {
        norm_weight,
        down,
        up,
        inject,
        hc: cfg.hc_count,
        hidden: cfg.hidden_size,
        eps: cfg.rms_norm_eps,
    })
}

fn load_qsa(
    archive: &GgufArchive,
    blk: &str,
    cfg: &Qwen4ExpConfig,
    rotary_emb: Arc<RotaryEmbedding>,
    device: &Device,
    dtype: DType,
) -> Result<QsaAttention> {
    let q_proj = load_qlinear(archive, &format!("{blk}.attn_q.weight"), device, dtype)?;
    let k_proj = load_qlinear(archive, &format!("{blk}.attn_k.weight"), device, dtype)?;
    let v_proj = load_qlinear(archive, &format!("{blk}.attn_v.weight"), device, dtype)?;
    let o_proj = load_qlinear(archive, &format!("{blk}.attn_output.weight"), device, dtype)?;
    let q_norm_weight = load_dense(archive, &format!("{blk}.attn_q_norm.weight"), device, dtype)?;
    let k_norm_weight = load_dense(archive, &format!("{blk}.attn_k_norm.weight"), device, dtype)?;
    let index_q_proj = load_qlinear(
        archive,
        &format!("{blk}.indexer.q_proj.weight"),
        device,
        dtype,
    )?;
    let index_k_proj = load_qlinear(
        archive,
        &format!("{blk}.indexer.k_proj.weight"),
        device,
        dtype,
    )?;
    let index_q_norm_weight = load_dense(
        archive,
        &format!("{blk}.indexer.q_norm.weight"),
        device,
        dtype,
    )?;
    let index_k_norm_f32 = load_dense(
        archive,
        &format!("{blk}.indexer.k_norm.weight"),
        device,
        DType::F32,
    )?;
    Ok(QsaAttention::new(
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm_weight,
        k_norm_weight,
        index_q_proj,
        index_k_proj,
        index_q_norm_weight,
        index_k_norm_f32,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.half_rot(),
        cfg.qsa,
        rotary_emb,
        cfg.rms_norm_eps,
    ))
}

fn load_gdn(
    archive: &GgufArchive,
    blk: &str,
    cfg: &Qwen4ExpConfig,
    device: &Device,
    dtype: DType,
) -> Result<GatedDeltaNet> {
    let in_proj_qkv = load_qlinear(archive, &format!("{blk}.attn_qkv.weight"), device, dtype)?;
    let in_proj_z = load_qlinear(archive, &format!("{blk}.attn_gate.weight"), device, dtype)?;
    let in_proj_b = load_qlinear(archive, &format!("{blk}.ssm_beta.weight"), device, dtype)?;
    let in_proj_a = load_qlinear(archive, &format!("{blk}.ssm_alpha.weight"), device, dtype)?;
    let out_proj = load_qlinear(archive, &format!("{blk}.ssm_out.weight"), device, dtype)?;
    let conv_weight = load_dense(archive, &format!("{blk}.ssm_conv1d.weight"), device, dtype)?;
    // GGUF stores `ssm_a` = -exp(A_log), so A_log = (-ssm_a).log() in F32.
    let ssm_a = load_dense(archive, &format!("{blk}.ssm_a"), device, DType::F32)?;
    let a_log = ssm_a.neg()?.log()?;
    let dt_bias = load_dense(archive, &format!("{blk}.ssm_dt.bias"), device, DType::F32)?;
    // `ssm_norm.weight` is NOT shifted in GGUF.
    let norm_weight = load_dense(archive, &format!("{blk}.ssm_norm.weight"), device, dtype)?;

    Ok(GatedDeltaNet {
        cfg: cfg.gdn.clone(),
        in_proj_qkv,
        in_proj_z,
        in_proj_b,
        in_proj_a,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        out_proj,
    })
}

fn load_moe(
    archive: &GgufArchive,
    blk: &str,
    site: ExpertSite,
    num_experts: usize,
    cfg: &Qwen4ExpConfig,
    provider: Option<&Arc<dyn RoutedExpertsProvider>>,
    device: &Device,
    dtype: DType,
) -> Result<SparseMoeBlock> {
    let gate_name = format!("{blk}.ffn_gate_inp.weight");
    let gate_dense = load_dense(archive, &gate_name, device, dtype)?;
    let gate = QLinear::from_dense(gate_dense.clone(), None);

    let moe_cfg = MoEExpertsConfig {
        num_experts,
        num_experts_per_tok: cfg.num_experts_per_tok,
        hidden_size: cfg.hidden_size,
        moe_intermediate_size: cfg.moe_intermediate_size,
    };

    let gate_exps = format!("{blk}.ffn_gate_exps.weight");
    let up_exps = format!("{blk}.ffn_up_exps.weight");
    let down_exps = format!("{blk}.ffn_down_exps.weight");

    let load_stacks = |target_dev: &Device| -> candle_core::Result<(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>)> {
        let g = Arc::new(archive.load_qtensor(&gate_exps, target_dev)?);
        let u = Arc::new(archive.load_qtensor(&up_exps, target_dev)?);
        let d = Arc::new(archive.load_qtensor(&down_exps, target_dev)?);
        Ok((g, u, d))
    };

    let experts: Arc<dyn RoutedExperts> = if let Some(prov) = provider {
        let req = ExpertLoadRequest {
            site,
            config: &moe_cfg,
            layer_device: device,
            act: Activation::Silu,
            router_weight: Some(gate_dense),
            load_stacks: &load_stacks,
        };
        if let Some(custom) = prov.build(req)? {
            custom
        } else {
            let (g, u, d) = load_stacks(device)?;
            Arc::new(MoEExperts::from_qtensors(
                g,
                u,
                d,
                num_experts,
                cfg.num_experts_per_tok,
                Activation::Silu,
            ))
        }
    } else {
        let (g, u, d) = load_stacks(device)?;
        Arc::new(MoEExperts::from_qtensors(
            g,
            u,
            d,
            num_experts,
            cfg.num_experts_per_tok,
            Activation::Silu,
        ))
    };

    let shared_expert = SharedMlp {
        gate_proj: load_qlinear(archive, &format!("{blk}.ffn_gate_shexp.weight"), device, dtype)?,
        up_proj: load_qlinear(archive, &format!("{blk}.ffn_up_shexp.weight"), device, dtype)?,
        down_proj: load_qlinear(archive, &format!("{blk}.ffn_down_shexp.weight"), device, dtype)?,
    };

    let sh_gate_raw = load_dense(
        archive,
        &format!("{blk}.ffn_gate_inp_shexp.weight"),
        device,
        dtype,
    )?;
    let sh_gate_2d = match sh_gate_raw.dims() {
        &[h] => sh_gate_raw.reshape((1, h))?,
        &[1, _] => sh_gate_raw,
        shape => bail!(
            "unexpected shape {shape:?} for `{blk}.ffn_gate_inp_shexp.weight`"
        ),
    };
    let shared_expert_gate = QLinear::from_dense(sh_gate_2d, None);

    Ok(SparseMoeBlock {
        gate,
        experts,
        shared_expert,
        shared_expert_gate,
        num_experts_per_tok: cfg.num_experts_per_tok,
        norm_topk_prob: cfg.norm_topk_prob,
    })
}

fn load_ple(
    archive: &Arc<GgufArchive>,
    blk: &str,
    cfg: &Qwen4ExpConfig,
    device: &Device,
    dtype: DType,
) -> Result<PleLayer> {
    let ple_cfg = cfg
        .ple
        .as_ref()
        .context("load_ple called without PLE config")?;
    let hash = PleHash::from_config(ple_cfg);
    let table = PleTable::load_gguf(Arc::clone(archive), ple_cfg.head_dim())?;
    let key_proj = load_qlinear(archive, &format!("{blk}.ple_key.weight"), device, dtype)?;
    let value_proj = load_qlinear(archive, &format!("{blk}.ple_value.weight"), device, dtype)?;
    let norm_key = load_dense(archive, &format!("{blk}.ple_norm_key.weight"), device, DType::F32)?;
    let norm_query = load_dense(
        archive,
        &format!("{blk}.ple_norm_query.weight"),
        device,
        DType::F32,
    )?;
    let norm_conv = load_dense(
        archive,
        &format!("{blk}.ple_norm_conv.weight"),
        device,
        DType::F32,
    )?;
    let conv_weight = load_dense(
        archive,
        &format!("{blk}.ple_conv1d.weight"),
        device,
        DType::F32,
    )?;
    Ok(PleLayer {
        cfg: ple_cfg.clone(),
        hash,
        table,
        key_proj,
        value_proj,
        norm_key,
        norm_query,
        norm_conv,
        conv_weight,
        hc: cfg.hc_count,
        hidden: cfg.hidden_size,
        eps: cfg.rms_norm_eps,
    })
}

fn load_mtp_head(
    archive: &GgufArchive,
    mtp_block: usize,
    cfg: &Qwen4ExpConfig,
    rotary_emb: Arc<RotaryEmbedding>,
    provider: Option<&Arc<dyn RoutedExpertsProvider>>,
    device: &Device,
    dtype: DType,
) -> Result<Qwen4ExpMtpHead> {
    let blk = format!("blk.{mtp_block}");
    let pre_fc_norm_embedding = load_dense(
        archive,
        &format!("{blk}.nextn.enorm.weight"),
        device,
        dtype,
    )?;
    let pre_fc_norm_hidden = load_dense(
        archive,
        &format!("{blk}.nextn.hnorm.weight"),
        device,
        dtype,
    )?;
    // `nextn.eh_proj.weight` has shape `[hidden, 2 * hidden]`:
    // first half along dim 1 is `fc_embedding`, second half is `fc_hidden`.
    let eh = load_dense(
        archive,
        &format!("{blk}.nextn.eh_proj.weight"),
        device,
        dtype,
    )?;
    let (h_out, fused_in) = eh.dims2()?;
    if fused_in != 2 * cfg.hidden_size || h_out != cfg.hidden_size {
        bail!(
            "unexpected shape [{h_out}, {fused_in}] for `{blk}.nextn.eh_proj.weight` (expected [{}, {}])",
            cfg.hidden_size,
            2 * cfg.hidden_size
        );
    }
    let half = cfg.hidden_size;
    let fc_embedding = QLinear::from_dense(eh.narrow(1, 0, half)?.contiguous()?, None);
    let fc_hidden = QLinear::from_dense(eh.narrow(1, half, half)?.contiguous()?, None);

    let attn = load_qsa(archive, &blk, cfg, Arc::clone(&rotary_emb), device, dtype)?;
    let attn_hc = load_hyper(
        archive,
        &format!("{blk}.hc_attn_norm.weight"),
        &format!("{blk}.hc_attn_down.weight"),
        &format!("{blk}.hc_attn_up.weight"),
        Some(&format!("{blk}.hc_attn_inject.weight")),
        cfg,
        device,
        dtype,
    )?;
    let mlp_hc = load_hyper(
        archive,
        &format!("{blk}.hc_ffn_norm.weight"),
        &format!("{blk}.hc_ffn_down.weight"),
        &format!("{blk}.hc_ffn_up.weight"),
        Some(&format!("{blk}.hc_ffn_inject.weight")),
        cfg,
        device,
        dtype,
    )?;
    let kv_layer_idx = cfg.num_hidden_layers;
    let moe = load_moe(
        archive,
        &blk,
        ExpertSite {
            layer: kv_layer_idx,
            is_mtp: true,
        },
        cfg.mtp_num_experts,
        cfg,
        provider,
        device,
        dtype,
    )?;
    let mixer = load_hyper(
        archive,
        &format!("{blk}.nextn.hc_head_norm.weight"),
        &format!("{blk}.nextn.hc_head_down.weight"),
        &format!("{blk}.nextn.hc_head_up.weight"),
        None,
        cfg,
        device,
        dtype,
    )?;

    Ok(Qwen4ExpMtpHead {
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        fc_embedding,
        fc_hidden,
        attn,
        attn_hc,
        mlp_hc,
        moe,
        mixer,
        rotary_emb,
        qsa: cfg.qsa,
        max_seq_len: cfg.max_position_embeddings,
        kv_layer_idx,
        hc: cfg.hc_count,
        hidden: cfg.hidden_size,
        eps: cfg.rms_norm_eps,
    })
}

/// Load a complete [`Qwen4ExpModel`] directly from an open [`GgufArchive`].
pub fn load_qwen4_exp_from_archive(
    archive: Arc<GgufArchive>,
    device: &Device,
    dtype: DType,
    moe_provider: Option<Arc<dyn RoutedExpertsProvider>>,
    load_mtp: bool,
) -> Result<Qwen4ExpModel> {
    let cfg = Qwen4ExpConfig::from_gguf(&archive)?;
    tracing::info!(
        layers = cfg.num_hidden_layers,
        hidden = cfg.hidden_size,
        experts = cfg.num_experts,
        topk = cfg.num_experts_per_tok,
        ple = cfg.ple.is_some(),
        mtp_present = cfg.mtp_block_idx.is_some(),
        "Loading Qwen4-Exp GGUF model"
    );

    let embed_tokens = load_qlinear(&archive, "token_embd.weight", device, dtype)?;
    let lm_head = if cfg.tie_word_embeddings {
        embed_tokens.clone()
    } else {
        load_qlinear(&archive, "output.weight", device, dtype)?
    };
    let final_mixer = load_hyper(
        &archive,
        "output_hc_norm.weight",
        "output_hc_down.weight",
        "output_hc_up.weight",
        None,
        &cfg,
        device,
        dtype,
    )?;
    let rotary_emb = Arc::new(RotaryEmbedding::new(cfg.rope_theta, cfg.rotary_dim, device)?);

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for layer_idx in 0..cfg.num_hidden_layers {
        let blk = format!("blk.{layer_idx}");
        let mixer = match cfg.layer_types[layer_idx] {
            LayerType::FullAttention => Mixer::Attention(load_qsa(
                &archive,
                &blk,
                &cfg,
                Arc::clone(&rotary_emb),
                device,
                dtype,
            )?),
            LayerType::LinearAttention => {
                Mixer::Linear(load_gdn(&archive, &blk, &cfg, device, dtype)?)
            }
        };
        let ple = match &cfg.ple {
            Some(ple_cfg) if ple_cfg.layer_idx == layer_idx => {
                Some(load_ple(&archive, &blk, &cfg, device, dtype)?)
            }
            _ => None,
        };
        let attn_hc = load_hyper(
            &archive,
            &format!("{blk}.hc_attn_norm.weight"),
            &format!("{blk}.hc_attn_down.weight"),
            &format!("{blk}.hc_attn_up.weight"),
            Some(&format!("{blk}.hc_attn_inject.weight")),
            &cfg,
            device,
            dtype,
        )?;
        let mlp_hc = load_hyper(
            &archive,
            &format!("{blk}.hc_ffn_norm.weight"),
            &format!("{blk}.hc_ffn_down.weight"),
            &format!("{blk}.hc_ffn_up.weight"),
            Some(&format!("{blk}.hc_ffn_inject.weight")),
            &cfg,
            device,
            dtype,
        )?;
        let moe = load_moe(
            &archive,
            &blk,
            ExpertSite {
                layer: layer_idx,
                is_mtp: false,
            },
            cfg.num_experts,
            &cfg,
            moe_provider.as_ref(),
            device,
            dtype,
        )?;
        layers.push(DecoderLayer {
            mixer,
            ple,
            attn_hc,
            mlp_hc,
            moe,
        });
    }

    let mtp = if load_mtp {
        let mtp_block = cfg.mtp_block_idx.context(
            "MTP requested (`--mtp`), but the GGUF checkpoint contains no `nextn.eh_proj.weight`",
        )?;
        Some(load_mtp_head(
            &archive,
            mtp_block,
            &cfg,
            Arc::clone(&rotary_emb),
            moe_provider.as_ref(),
            device,
            dtype,
        )?)
    } else {
        None
    };

    Ok(Qwen4ExpModel::from_parts(
        cfg,
        embed_tokens,
        layers,
        final_mixer,
        lm_head,
        rotary_emb,
        mtp,
        device.clone(),
        dtype,
    ))
}
