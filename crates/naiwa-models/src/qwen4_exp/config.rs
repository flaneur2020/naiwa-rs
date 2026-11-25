//! Direct GGUF metadata parser for Qwen3.8-Flash-Next (`qwen4exp`) text models.

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value;
use naiwa_attn::{GdnConfig, GdnGateActivation, GdnVHeadLayout};
use naiwa_quant::GgufArchive;

pub const ARCHITECTURE: &str = "qwen4exp";

/// Attention type per decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

/// Block-sparse attention (QSA) hyperparameters shared across every full-attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsaConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub compress_ratio: usize,
}

impl QsaConfig {
    pub fn block_topk(&self) -> usize {
        self.budget / self.compress_ratio
    }

    /// Maximum tokens any query attends to: full selected blocks plus the incomplete tail.
    pub fn max_selected_tokens(&self) -> usize {
        self.budget + self.compress_ratio - 1
    }

    pub fn aux_cache_elements_per_token(&self, rot_dim: usize) -> usize {
        self.head_dim + rot_dim + self.head_dim.div_ceil(self.compress_ratio)
    }
}

/// Explicit PLE n-gram hash constants stored in GGUF metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleHashConstants {
    pub layer_multipliers: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    pub head_offsets: Vec<u64>,
}

/// Hashed n-gram Per-Layer Embedding (PLE) parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleConfig {
    /// Zero-based decoder layer index hosting PLE.
    pub layer_idx: usize,
    pub embed_dim: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel_size: usize,
    pub eos_token_id: u32,
    pub hash: PleHashConstants,
}

impl PleConfig {
    pub fn num_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads()
    }

    pub fn conv_dilation(&self) -> usize {
        self.ngram_size
    }

    pub fn conv_state_len(&self) -> usize {
        (self.conv_kernel_size - 1) * self.conv_dilation()
    }

    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }
}

/// Complete configuration for a Qwen3.8-Flash-Next (`qwen4exp`) text GGUF model.
#[derive(Debug, Clone)]
pub struct Qwen4ExpConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub layer_types: Vec<LayerType>,
    pub gdn: GdnConfig,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub qsa: QsaConfig,
    pub ple: Option<PleConfig>,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
    /// GGUF block index holding the MTP head (`blk.<block>.nextn.eh_proj.weight`), if present.
    pub mtp_block_idx: Option<usize>,
    /// Routed expert count for the MTP block (may differ from target `num_experts`).
    pub mtp_num_experts: usize,
}

impl Qwen4ExpConfig {
    /// Parse and validate `Qwen4ExpConfig` directly from an open `GgufArchive`.
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        let arch = metadata_string(archive, "general.architecture")?
            .context("GGUF metadata missing `general.architecture`")?;
        if arch != ARCHITECTURE {
            bail!("unsupported GGUF architecture `{arch}`, expected `{ARCHITECTURE}`");
        }
        let key = |suffix: &str| format!("{ARCHITECTURE}.{suffix}");
        let usize_at = |suffix: &str| metadata_usize(archive, &key(suffix));

        let head_dim = usize_at("attention.key_length")?;
        let num_hidden_layers = usize_at("block_count")?;
        let hidden_size = usize_at("embedding_length")?;
        let num_attention_heads = usize_at("attention.head_count")?;
        let num_key_value_heads = usize_at("attention.head_count_kv")?;
        let max_position_embeddings = usize_at("context_length")?;
        let rms_norm_eps = metadata_f64(archive, &key("attention.layer_norm_rms_epsilon"))?;
        let rope_theta = metadata_f64(archive, &key("rope.freq_base"))?;
        let rotary_dim = usize_at("rope.dimension_count")?;

        let compress_ratios = u64_array(archive, &key("attention.compress_ratios"))?;
        if compress_ratios.len() != num_hidden_layers {
            bail!(
                "GGUF lists {} compress ratios for {num_hidden_layers} layers",
                compress_ratios.len()
            );
        }
        let layer_types: Vec<LayerType> = compress_ratios
            .iter()
            .map(|&r| {
                if r > 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        let compress_ratio = compress_ratios.iter().copied().max().unwrap_or(0) as usize;

        let qsa = QsaConfig {
            n_heads: usize_at("attention.indexer.head_count")?,
            head_dim: usize_at("attention.indexer.key_length")?,
            budget: usize_at("attention.indexer.top_k")?,
            compress_ratio,
        };
        if qsa.n_heads == 0
            || qsa.head_dim == 0
            || qsa.compress_ratio == 0
            || qsa.budget == 0
            || !qsa.budget.is_multiple_of(qsa.compress_ratio)
        {
            bail!("invalid QSA config: {qsa:?}");
        }
        if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim > qsa.head_dim || rotary_dim % 2 != 0
        {
            bail!("invalid rotary_dim {rotary_dim} for head_dim={head_dim}, qsa_head_dim={}", qsa.head_dim);
        }

        let vocab_size = archive
            .tensor_info("token_embd.weight")?
            .shape()
            .first()
            .copied()
            .context("token_embd.weight has empty shape")?;
        let tie_word_embeddings = !archive.contains_tensor("output.weight");

        let moe_intermediate_size = usize_at("expert_feed_forward_length")?;
        let shared_expert_intermediate_size = usize_at("expert_shared_feed_forward_length")?;
        let num_experts = usize_at("expert_count")?;
        let num_experts_per_tok = usize_at("expert_used_count")?;

        let linear_num_key_heads = usize_at("ssm.group_count")?;
        let linear_num_value_heads = usize_at("ssm.time_step_rank")?;
        let linear_key_head_dim = usize_at("ssm.state_size")?;
        let linear_inner = usize_at("ssm.inner_size")?;
        if linear_num_value_heads == 0 || !linear_inner.is_multiple_of(linear_num_value_heads) {
            bail!(
                "GDN inner_size {linear_inner} not divisible by value heads {linear_num_value_heads}"
            );
        }
        let gdn = GdnConfig {
            hidden_size,
            rms_norm_eps,
            conv_kernel_dim: usize_at("ssm.conv_kernel")?,
            num_k_heads: linear_num_key_heads,
            num_v_heads: linear_num_value_heads,
            head_k_dim: linear_key_head_dim,
            head_v_dim: linear_inner / linear_num_value_heads,
            v_head_layout: GdnVHeadLayout::Tiled,
            gate_activation: GdnGateActivation::Sigmoid,
        };

        let hc_count = usize_at("hyper_connection.count")?;
        let hc_lowrank = usize_at("hyper_connection.low_rank")?;
        if hc_count <= 1 || hc_lowrank == 0 {
            bail!("invalid hyper-connection config: count={hc_count}, low_rank={hc_lowrank}");
        }

        let (ple, eos_token_id) = if archive.metadata_value(&key("ple.layers")).is_some()
            && archive.contains_tensor("per_layer_token_embd.weight")
        {
            let ple_layers = u64_array(archive, &key("ple.layers"))?;
            if ple_layers.len() != 1 {
                bail!("expected exactly 1 PLE layer, got {ple_layers:?}");
            }
            let layer_idx = ple_layers[0] as usize;
            if layer_idx >= num_hidden_layers
                || layer_types[layer_idx] != LayerType::LinearAttention
            {
                bail!("PLE layer {layer_idx} must be a valid linear attention layer");
            }
            let ngram_size = usize_at("ple.ngram_size")?;
            let heads_per_ngram = usize_at("ple.heads_per_ngram")?;
            let row_dim = usize_at("embedding_length_per_layer_input")?;
            let conv_kernel_size = usize_at("ple.conv_kernel")?;
            let eos = usize_at("ple.eos_token_id")? as u32;
            let hash = PleHashConstants {
                layer_multipliers: u64_array(archive, &key("ple.layer_multipliers"))?,
                head_vocab_sizes: u64_array(archive, &key("ple.head_vocab_sizes"))?,
                head_offsets: u64_array(archive, &key("ple.head_offsets"))?,
            };
            let ple = PleConfig {
                layer_idx,
                embed_dim: row_dim * (ngram_size - 1) * heads_per_ngram,
                ngram_size,
                heads_per_ngram,
                conv_kernel_size,
                eos_token_id: eos,
                hash,
            };
            if ple.hash.layer_multipliers.len() != ple.ngram_size
                || ple.hash.head_vocab_sizes.len() != ple.num_heads()
                || ple.hash.head_offsets.len() != ple.num_heads()
            {
                bail!("PLE hash metadata length mismatch");
            }
            (Some(ple), eos)
        } else {
            let eos = metadata_usize(archive, "tokenizer.ggml.eos_token_id")? as u32;
            (None, eos)
        };

        let mtp_block_idx = detect_mtp_block(archive, num_hidden_layers);
        let mtp_num_experts = if archive.metadata_value(&key("mtp.expert_count")).is_some() {
            metadata_usize(archive, &key("mtp.expert_count"))?
        } else {
            num_experts
        };

        Ok(Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            rotary_dim,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob: true,
            layer_types,
            gdn,
            hc_count,
            hc_lowrank,
            qsa,
            ple,
            eos_token_id,
            tie_word_embeddings,
            mtp_block_idx,
            mtp_num_experts,
        })
    }

    pub fn hc_hidden_size(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    pub fn half_rot(&self) -> usize {
        self.rotary_dim / 2
    }

    pub fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == LayerType::FullAttention)
            .count()
    }
}

fn detect_mtp_block(archive: &GgufArchive, num_layers: usize) -> Option<usize> {
    let nextn = metadata_usize(archive, &format!("{ARCHITECTURE}.nextn_predict_layers")).ok()?;
    (1..=nextn).rev().find_map(|offset| {
        let candidate = num_layers + offset - 1;
        archive
            .contains_tensor(&format!("blk.{candidate}.nextn.eh_proj.weight"))
            .then_some(candidate)
    })
}

fn metadata_string<'a>(archive: &'a GgufArchive, key: &str) -> Result<Option<&'a str>> {
    match archive.metadata_value(key) {
        Some(Value::String(v)) => Ok(Some(v.as_str())),
        Some(_) => bail!("GGUF metadata `{key}` must be a string"),
        None => Ok(None),
    }
}

pub(crate) fn metadata_usize(archive: &GgufArchive, key: &str) -> Result<usize> {
    let value = archive
        .metadata_value(key)
        .with_context(|| format!("GGUF metadata missing `{key}`"))?;
    let v = match value {
        Value::U8(v) => u64::from(*v),
        Value::U16(v) => u64::from(*v),
        Value::U32(v) => u64::from(*v),
        Value::U64(v) => *v,
        Value::I8(v) if *v >= 0 => *v as u64,
        Value::I16(v) if *v >= 0 => *v as u64,
        Value::I32(v) if *v >= 0 => *v as u64,
        Value::I64(v) if *v >= 0 => *v as u64,
        _ => bail!("GGUF metadata `{key}` must be a nonnegative integer"),
    };
    usize::try_from(v).with_context(|| format!("GGUF metadata `{key}` overflows usize"))
}

fn metadata_f64(archive: &GgufArchive, key: &str) -> Result<f64> {
    match archive.metadata_value(key) {
        Some(Value::F32(v)) => Ok(f64::from(*v)),
        Some(Value::F64(v)) => Ok(*v),
        Some(_) => bail!("GGUF metadata `{key}` must be a float"),
        None => bail!("GGUF metadata missing `{key}`"),
    }
}

fn u64_array(archive: &GgufArchive, key: &str) -> Result<Vec<u64>> {
    let Some(Value::Array(values)) = archive.metadata_value(key) else {
        bail!("GGUF metadata `{key}` must be an integer array");
    };
    values
        .iter()
        .map(|value| match value {
            Value::U8(v) => Ok(u64::from(*v)),
            Value::U16(v) => Ok(u64::from(*v)),
            Value::U32(v) => Ok(u64::from(*v)),
            Value::U64(v) => Ok(*v),
            Value::I32(v) if *v >= 0 => Ok(*v as u64),
            Value::I64(v) if *v >= 0 => Ok(*v as u64),
            _ => bail!("GGUF metadata `{key}` must hold nonnegative integers"),
        })
        .collect()
}
