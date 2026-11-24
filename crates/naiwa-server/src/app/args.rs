//! Command-line arguments for `naiwa-rs` (preserving compatibility with existing harness scripts).

use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use candle_core::DType;
use clap::{Args, Parser, Subcommand, ValueEnum};
use naiwa_runtime::{
    MtpConfig, MtpDraftSampling, DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
};

/// Qwen3.8-Flash-Next GGUF edge inference server
#[derive(Parser)]
#[command(name = "naiwa-rs", version, about, long_about = None, propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[command(flatten)]
    pub global: GlobalOptions,
}

#[derive(Subcommand)]
pub enum Command {
    /// Serve the GGUF text model over the OpenAI- and Anthropic-compatible HTTP API
    Serve {
        #[command(flatten)]
        model: ModelOptions,

        #[command(flatten)]
        runtime: RuntimeOptions,

        #[command(flatten)]
        server: ServerOptions,
    },
}

#[derive(Args, Clone)]
pub struct GlobalOptions {
    /// Random seed for reproducibility
    #[arg(long, global = true)]
    pub seed: Option<u64>,

    /// Log all requests and responses to this file (accepted for CLI compatibility)
    #[arg(long, short, global = true)]
    pub log: Option<PathBuf>,

    /// Increase logging verbosity (-v for debug, -vv for trace)
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

/// Model activation dtype.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ModelDTypeArg {
    #[default]
    Auto,
    Bf16,
    F16,
    F32,
}

impl ModelDTypeArg {
    pub fn resolve(self, is_cuda: bool) -> DType {
        match self {
            Self::Auto => {
                if is_cuda {
                    DType::BF16
                } else {
                    DType::F32
                }
            }
            Self::Bf16 => DType::BF16,
            Self::F16 => DType::F16,
            Self::F32 => DType::F32,
        }
    }
}

/// Local GGUF checkpoint source.
#[derive(Args, Clone)]
pub struct ModelSource {
    /// Local directory containing GGUF shard(s). Optional when `-f` names absolute paths.
    #[arg(short = 'm', long)]
    pub model_id: Option<PathBuf>,

    /// GGUF filename(s), relative to `-m` or absolute (semicolon-separated for shards)
    #[arg(short = 'f', long)]
    pub quantized_file: Option<String>,

    /// Extra GGUF component file(s) merged into the archive (e.g. MTP component GGUF)
    #[arg(long)]
    pub mmproj: Option<String>,

    /// Path to optional external `tokenizer.json` override
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Activation data type (`auto`, `bf16`, `f16`, `f32`)
    #[arg(long, default_value = "auto", value_enum)]
    pub dtype: ModelDTypeArg,

    /// Optional runtime context length cap
    #[arg(long, value_parser = parse_positive_usize)]
    pub max_model_len: Option<usize>,
}

#[derive(Args, Clone)]
pub struct ModelOptions {
    #[command(flatten)]
    pub source: ModelSource,

    #[command(flatten)]
    pub device: DeviceOptions,

    #[command(flatten)]
    pub paged_attn: PagedAttentionOptions,

    /// Keep routed MoE experts in host memory with an edge GPU cache (`--expert-cache-gb`)
    #[arg(long, requires = "expert_cache_gb")]
    pub cpu_moe: bool,

    /// Disable next-layer expert lookahead prediction (deterministic for oracle/AB gates)
    #[arg(long = "no-lookahead", requires = "cpu_moe")]
    pub no_lookahead: bool,

    /// GPU expert cache budget in GiB
    #[arg(long = "expert-cache-gb", requires = "cpu_moe", value_parser = clap::value_parser!(f64))]
    pub expert_cache_gb: Option<f64>,

    /// Pre-warm expert cache from a binary strata expert profile
    #[arg(long = "expert-profile", requires = "expert_cache_gb")]
    pub expert_profile: Option<PathBuf>,

    /// Share (0.0-1.0) of decode misses to DMA over PCIe into the GPU staging ring
    #[arg(
        long = "pcie-frac",
        requires = "expert_cache_gb",
        default_value_t = 0.0,
        value_parser = parse_unit_fraction
    )]
    pub pcie_frac: f32,

    /// Max lookahead expert replacements per row and pass
    #[arg(
        long = "lookahead-swaps",
        requires = "expert_cache_gb",
        default_value_t = 2,
        value_name = "N"
    )]
    pub lookahead_swaps: usize,

    /// Stream routed experts to GPU during long prefill chunks
    #[arg(long, requires = "cpu_moe")]
    pub prefill_stream: bool,

    /// Minimum prefill chunk token length to engage `--prefill-stream`
    #[arg(
        long,
        requires = "prefill_stream",
        default_value_t = NonZeroUsize::new(naiwa_offload::pipeline::prefill::DEFAULT_MIN_TOKENS).unwrap()
    )]
    pub prefill_stream_min_tokens: NonZeroUsize,

    /// In-flight prefill stream layer buffers (1-4)
    #[arg(
        long,
        requires = "prefill_stream",
        value_parser = clap::value_parser!(u8).range(1..=4),
        default_value_t = naiwa_offload::pipeline::prefill::DEFAULT_DEPTH as u8
    )]
    pub prefill_stream_depth: u8,

    /// Expert sub-ranges per prefill layer (1-8)
    #[arg(
        long,
        requires = "prefill_stream",
        value_parser = clap::value_parser!(u8).range(1..=8),
        default_value_t = 1
    )]
    pub prefill_split: u8,
}

#[derive(Args, Clone, Default)]
pub struct DeviceOptions {
    /// Force CPU-only execution
    #[arg(long)]
    pub cpu: bool,
}

#[derive(Args, Clone)]
pub struct PagedAttentionOptions {
    /// PagedAttention mode (`auto`, `on`, `off`; accepted for CLI compatibility)
    #[arg(long = "paged-attn", default_value = "auto", value_enum)]
    pub mode: PagedAttnMode,

    /// Allocate KV cache for this many tokens
    #[arg(long = "pa-context-len", value_parser = parse_positive_usize)]
    pub context_len: Option<usize>,

    /// GPU KV cache memory in MiB
    #[arg(long = "pa-memory-mb", conflicts_with = "context_len")]
    pub memory_mb: Option<usize>,

    /// GPU KV cache utilization fraction 0.0-1.0
    #[arg(long = "pa-memory-fraction", conflicts_with_all = ["context_len", "memory_mb"])]
    pub memory_fraction: Option<f32>,

    /// Tokens per paged KV block (default 32)
    #[arg(long = "pa-block-size")]
    pub block_size: Option<usize>,

    /// KV cache dtype (`auto` only; accepted for CLI compatibility)
    #[arg(long = "pa-cache-type", default_value = "auto")]
    pub cache_type: String,
}

#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum PagedAttnMode {
    #[default]
    Auto,
    On,
    Off,
}

#[derive(Args, Clone)]
pub struct MtpOptions {
    /// Enable Built-in MTP speculative decoding
    #[arg(long)]
    pub mtp: bool,

    /// Fixed MTP draft tokens per step (default 2)
    #[arg(long, value_parser = parse_positive_usize, requires = "mtp")]
    pub mtp_n_predict: Option<usize>,

    /// MTP draft sampling policy (`auto`, `greedy`, `probabilistic`)
    #[arg(long, value_enum, default_value_t)]
    pub mtp_draft_sampling: MtpDraftSamplingArg,
}

impl MtpOptions {
    pub fn config(&self) -> Option<MtpConfig> {
        self.mtp.then(|| MtpConfig {
            n_predict: self.mtp_n_predict.unwrap_or(2),
            draft_sampling: self.mtp_draft_sampling.into(),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum MtpDraftSamplingArg {
    #[default]
    Auto,
    Greedy,
    Probabilistic,
}

impl From<MtpDraftSamplingArg> for MtpDraftSampling {
    fn from(v: MtpDraftSamplingArg) -> Self {
        match v {
            MtpDraftSamplingArg::Auto => Self::Auto,
            MtpDraftSamplingArg::Greedy => Self::Greedy,
            MtpDraftSamplingArg::Probabilistic => Self::Probabilistic,
        }
    }
}

#[derive(Args, Clone)]
pub struct RuntimeOptions {
    /// Maximum concurrent sequences
    #[arg(long, default_value_t = 4)]
    pub max_seqs: usize,

    /// Maximum tokens processed in one scheduler step
    #[arg(long, default_value_t = default_max_num_batched_tokens())]
    pub max_num_batched_tokens: NonZeroUsize,

    /// Maximum tokens per prefill chunk
    #[arg(long, default_value_t = default_max_prefill_chunk_tokens())]
    pub max_prefill_chunk_tokens: NonZeroUsize,

    /// Maximum decode steps before admitting a waiting prefill chunk
    #[arg(long, default_value_t = default_max_decode_steps_before_prefill())]
    pub max_decode_steps_before_prefill: NonZeroUsize,

    /// Accepted for CLI compatibility
    #[arg(long)]
    pub no_kv_cache: bool,

    /// Prefix cache entries (accepted for CLI compatibility)
    #[arg(long, default_value_t = 0)]
    pub prefix_cache_n: usize,

    /// Custom chat template file (.json or .jinja)
    #[arg(long, short)]
    pub chat_template: Option<PathBuf>,

    /// Explicit Jinja chat template file override
    #[arg(long, short)]
    pub jinja_explicit: Option<PathBuf>,

    #[command(flatten)]
    pub mtp: MtpOptions,
}

#[derive(Args, Clone)]
pub struct ServerOptions {
    /// HTTP server port
    #[arg(short = 'p', long, default_value_t = 1234)]
    pub port: u16,

    /// Bind address
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Disable per-request HTTP access logs
    #[arg(long)]
    pub disable_access_log: bool,

    /// Access log format (accepted for CLI compatibility)
    #[arg(long, default_value = "text")]
    pub access_log_format: String,

    /// Include health/metrics in access logs (accepted for CLI compatibility)
    #[arg(long)]
    pub access_log_health: bool,

    /// Disable x-request-id header (accepted for CLI compatibility)
    #[arg(long)]
    pub disable_request_id_header: bool,

    /// Disable the Prometheus `/metrics` endpoint
    #[arg(long)]
    pub disable_metrics: bool,
}

pub fn parse_positive_usize(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(0) => Err("value must be greater than 0".to_string()),
        Ok(v) => Ok(v),
        Err(e) => Err(e.to_string()),
    }
}

fn parse_unit_fraction(value: &str) -> Result<f32, String> {
    let v: f32 = value.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err("value must be between 0.0 and 1.0".to_string())
    }
}

fn default_max_num_batched_tokens() -> NonZeroUsize {
    NonZeroUsize::new(DEFAULT_MAX_NUM_BATCHED_TOKENS).unwrap()
}

fn default_max_prefill_chunk_tokens() -> NonZeroUsize {
    NonZeroUsize::new(DEFAULT_MAX_PREFILL_CHUNK_TOKENS).unwrap()
}

fn default_max_decode_steps_before_prefill() -> NonZeroUsize {
    NonZeroUsize::new(DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL).unwrap()
}

/// Expands a leading `~/` in each entry of a `;`-separated path list.
pub fn expand_home_in_list(value: &str) -> String {
    let home = dirs::home_dir();
    value
        .split(';')
        .map(|entry| {
            let entry = entry.trim();
            match (entry.strip_prefix("~/"), &home) {
                (Some(rest), Some(home)) => home.join(rest).to_string_lossy().into_owned(),
                _ => entry.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(";")
}

pub fn split_path_list(value: &str) -> Result<Vec<&str>> {
    let entries = value.split(';').map(str::trim).collect::<Vec<_>>();
    if entries.iter().any(|entry| entry.is_empty()) {
        bail!("path lists must contain nonempty, semicolon-separated entries");
    }
    Ok(entries)
}

pub fn ensure_local_dir(path: &Path, what: &str) -> Result<()> {
    if !path.is_dir() {
        bail!("{what} `{}` is not a local directory", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_oracle_and_mtp2_cli_invocations() {
        let cli = Cli::try_parse_from([
            "naiwa-rs",
            "serve",
            "-m",
            "/tmp/model",
            "-f",
            "a.gguf;b.gguf",
            "--cpu-moe",
            "--expert-cache-gb",
            "2.5",
            "--pcie-frac",
            "0.4",
            "--prefill-stream",
            "--prefill-stream-min-tokens",
            "64",
            "--max-num-batched-tokens",
            "8192",
            "--max-prefill-chunk-tokens",
            "8192",
            "--max-seqs",
            "4",
            "--mmproj",
            "/tmp/mtp.gguf",
            "--mtp",
            "--mtp-n-predict",
            "2",
            "--mtp-draft-sampling",
            "greedy",
            "--no-lookahead",
            "--pa-context-len",
            "4096",
            "--prefix-cache-n",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            "19980",
            "--disable-access-log",
        ])
        .unwrap();
        match cli.command {
            Command::Serve {
                model,
                runtime,
                server,
            } => {
                assert!(model.cpu_moe);
                assert!(model.no_lookahead);
                assert_eq!(model.expert_cache_gb, Some(2.5));
                assert_eq!(runtime.mtp.config().unwrap().n_predict, 2);
                assert_eq!(server.port, 19980);
            }
        }
    }
}
