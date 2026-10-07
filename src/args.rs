//! Command-line arguments. Flag names, defaults, and help text follow mistralrs-cli.

use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use mistralrs_core::{
    AutoDeviceMapParams, HfConfigOverrides, IsqOrganization, ModelDType, PagedCacheType,
    DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
};
use mistralrs_server_core::metrics::{AccessLogFormat, ObservabilityConfig};

/// Qwen3.8-Flash-Next inference server, powered by mistral.rs
#[derive(Parser)]
#[command(name = "qwen-fn-rs", version, about, long_about = None, propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[command(flatten)]
    pub global: GlobalOptions,
}

#[derive(Subcommand)]
pub enum Command {
    /// Serve the model over the OpenAI- and Anthropic-compatible HTTP API
    Serve {
        #[command(flatten)]
        model: ModelOptions,

        #[command(flatten)]
        runtime: RuntimeOptions,

        #[command(flatten)]
        server: ServerOptions,
    },

    /// Run performance benchmarks
    Bench {
        #[command(flatten)]
        model: ModelOptions,

        #[command(flatten)]
        runtime: BenchRuntimeOptions,

        /// Input lengths used to measure time to first token. Zero skips TTFT. Accepts comma-separated values for sweeps.
        #[arg(long, value_delimiter = ',', default_value = "512")]
        prompt_len: Vec<usize>,

        /// Output tokens per decode request. Values below 2 skip decode metrics.
        #[arg(long, default_value = "128")]
        gen_len: usize,

        /// Input context lengths used to measure decode TPOT. Accepts comma-separated values for sweeps.
        #[arg(long, value_delimiter = ',', default_value = "4")]
        depth: Vec<usize>,

        /// Number of benchmark iterations
        #[arg(long, default_value = "3")]
        iterations: usize,

        /// Number of warmup runs per benchmark case (discarded)
        #[arg(long, default_value = "1")]
        warmup: usize,
    },

    /// Generate UQFF quantized model files
    Quantize {
        #[command(flatten)]
        source: ModelSource,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        quantization: QuantizeOptions,

        /// Output path: a `.uqff` file, or a directory to auto-name one file per ISQ type
        #[arg(short = 'o', long)]
        output: PathBuf,
    },

    /// Inspect or verify UQFF artifacts in a local directory
    Uqff {
        #[command(subcommand)]
        command: UqffCommand,
    },

    /// Recommend quantization + device mapping for this machine.
    /// Pass `--isq <level>` to bias the recommendation toward a specific quantization target.
    Tune {
        #[command(flatten)]
        model: ModelOptions,

        /// Tuning profile (quality, balanced, fast)
        #[arg(long, value_enum, default_value = "balanced")]
        profile: TuneProfileArg,

        /// Output JSON instead of human-readable text
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum UqffCommand {
    /// Print or write a UQFF report
    Report {
        /// Local directory containing UQFF files
        #[arg(short = 'm', long)]
        model_id: PathBuf,

        /// Quant group to inspect, such as 3, q3k, afq3, or all
        #[arg(long)]
        quant: Option<String>,

        /// Write uqff_report.json beside the artifacts
        #[arg(long)]
        write: bool,

        /// Print JSON instead of human-readable text
        #[arg(long)]
        json: bool,

        /// Base model ID to include in a written report
        #[arg(long)]
        base_model: Option<String>,
    },

    /// Validate UQFF artifact structure
    Verify {
        /// Local directory containing UQFF files
        #[arg(short = 'm', long)]
        model_id: PathBuf,

        /// Quant group to inspect, such as 3, q3k, afq3, or all
        #[arg(long)]
        quant: Option<String>,

        /// Print JSON instead of human-readable text
        #[arg(long)]
        json: bool,

        /// Fail on missing report/producer metadata or fallback layers
        #[arg(long)]
        strict: bool,

        /// Allow same-major UQFF files with a newer minor version
        #[arg(long)]
        allow_newer_minor: bool,
    },
}

#[derive(Args, Clone)]
pub struct GlobalOptions {
    /// Random seed for reproducibility
    #[arg(long, global = true)]
    pub seed: Option<u64>,

    /// Log all requests and responses to this file
    #[arg(long, short, global = true)]
    pub log: Option<PathBuf>,

    /// Increase logging verbosity. Use -v for debug and -vv for trace-level internals.
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

/// Where the checkpoint lives. Everything must already be on the local filesystem.
#[derive(Args, Clone)]
pub struct ModelSource {
    /// Local model directory (config.json, tokenizer, safetensors or UQFF).
    /// Optional when `-f` names local GGUF files.
    #[arg(short = 'm', long)]
    pub model_id: Option<PathBuf>,

    /// GGUF filename(s), relative to `-m` or absolute (semicolon-separated for shards)
    #[arg(short = 'f', long)]
    pub quantized_file: Option<String>,

    /// GGUF vision projector file(s) (semicolon-separated for multiple)
    #[arg(long)]
    pub mmproj: Option<String>,

    /// Local directory overriding configuration, tokenizer, and processor assets for a GGUF model
    #[arg(long)]
    pub tok_model_id: Option<PathBuf>,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    pub dtype: ModelDType,

    /// Recursively merged JSON overrides for the model config
    #[arg(long)]
    pub hf_overrides: Option<HfConfigOverrides>,

    /// Runtime model context length
    #[arg(long, value_parser = parse_positive_usize)]
    pub max_model_len: Option<usize>,

    #[command(flatten)]
    pub multimodal: MultimodalOptions,
}

/// Model source plus the options every inference command shares.
#[derive(Args, Clone)]
pub struct ModelOptions {
    #[command(flatten)]
    pub source: ModelSource,

    #[command(flatten)]
    pub quantization: QuantizationOptions,

    #[command(flatten)]
    pub device: DeviceOptions,

    #[command(flatten)]
    pub paged_attn: PagedAttentionOptions,

    /// Keep routed MoE experts in host memory and compute them on the CPU; attention, routers,
    /// shared experts, the MTP head, and the KV cache stay on the GPU. Places every layer on the
    /// GPU unless `--device-layers` is given.
    #[arg(long)]
    pub cpu_moe: bool,

    /// With `--cpu-moe`, cache the most-used routed experts in VRAM for decode, in GiB.
    #[arg(long = "expert-cache-gb", requires = "cpu_moe", value_parser = clap::value_parser!(f64))]
    pub expert_cache_gb: Option<f64>,

    /// With `--cpu-moe`, run the routed experts of prefill chunks of 1024+ tokens on the GPU,
    /// uploading each layer's experts while the previous layer computes. Needs two layers' worth
    /// of expert memory in VRAM, borrowed from `--expert-cache-gb` while a prompt runs when that
    /// is set. Streams every expert once per chunk, so raise
    /// `--max-num-batched-tokens` (e.g. 8192) for long prompts. CUDA only.
    #[arg(long, requires = "cpu_moe")]
    pub prefill_stream: bool,
}

#[derive(Args, Clone, Default)]
pub struct MultimodalOptions {
    /// Maximum logical tensor memory retained by the multimodal encoder cache, in MiB
    #[arg(long = "encoder-cache-memory-mb")]
    pub encoder_cache_memory_mb: Option<NonZeroUsize>,

    /// Maximum edge length for image resizing (aspect ratio preserved)
    #[arg(long)]
    pub max_edge: Option<u32>,

    /// Maximum number of images per request
    #[arg(long)]
    pub max_num_images: Option<usize>,

    /// Maximum image dimension for device mapping
    #[arg(long)]
    pub max_image_length: Option<usize>,
}

#[derive(Args, Clone, Default)]
pub struct QuantizationOptions {
    /// In-situ quantization target. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or
    /// raw quant names (`q4k`, `q8_0`, etc.).
    #[arg(long = "isq", conflicts_with = "from_uqff")]
    pub in_situ_quant: Option<String>,

    /// UQFF artifact in the model directory. Accepts a filename, numeric quantization level
    /// (`2`, `3`, `4`, `5`, `6`, `8`), or quantization type (`q4k`, `afq8`, etc.).
    /// Use semicolons only to list disjoint shards manually.
    #[arg(long)]
    pub from_uqff: Option<String>,

    /// ISQ organization strategy: default or moqe
    #[arg(long)]
    pub isq_organization: Option<IsqOrganization>,

    /// imatrix file for enhanced quantization
    #[arg(long)]
    pub imatrix: Option<PathBuf>,

    /// Calibration file for imatrix generation
    #[arg(long, conflicts_with = "imatrix")]
    pub calibration_file: Option<PathBuf>,
}

#[derive(Args, Clone)]
pub struct QuantizeOptions {
    /// ISQ type(s) to write. Repeat or comma-separate for multiple outputs
    #[arg(long = "isq", required = true, value_delimiter = ',')]
    pub in_situ_quant: Vec<String>,

    /// ISQ organization strategy: default or moqe
    #[arg(long)]
    pub isq_organization: Option<IsqOrganization>,

    /// imatrix file for enhanced quantization
    #[arg(long)]
    pub imatrix: Option<PathBuf>,

    /// Calibration file for imatrix generation
    #[arg(long, conflicts_with = "imatrix")]
    pub calibration_file: Option<PathBuf>,
}

#[derive(Args, Clone, Default)]
pub struct DeviceOptions {
    /// Force CPU-only execution
    #[arg(long)]
    pub cpu: bool,

    /// Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20")
    /// Omit for automatic device mapping
    #[arg(short = 'n', long, value_delimiter = ';')]
    pub device_layers: Option<Vec<String>>,

    /// Topology YAML file for device mapping
    #[arg(long)]
    pub topology: Option<PathBuf>,

    /// Max sequence length for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
    pub max_seq_len: usize,

    /// Max batch size for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
    pub max_batch_size: usize,
}

#[derive(Args, Clone)]
pub struct PagedAttentionOptions {
    /// PagedAttention mode
    /// - auto: enabled on CUDA, disabled on CPU (default)
    /// - on: force enable (fails if unsupported)
    /// - off: force disable
    #[arg(long = "paged-attn", default_value = "auto", value_enum)]
    pub mode: PagedAttnMode,

    /// Allocate KV cache for this many tokens. Defaults to a memory budget on dedicated GPUs,
    /// capped to the model context length on CUDA unified-memory devices.
    #[arg(long = "pa-context-len", value_parser = parse_positive_usize)]
    pub context_len: Option<usize>,

    /// GPU memory to allocate in MBs (alternative to context-len)
    #[arg(long = "pa-memory-mb", conflicts_with = "context_len")]
    pub memory_mb: Option<usize>,

    /// GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb)
    #[arg(long = "pa-memory-fraction", conflicts_with_all = ["context_len", "memory_mb"])]
    pub memory_fraction: Option<f32>,

    /// Tokens per block (default: 32 on CUDA)
    #[arg(long = "pa-block-size")]
    pub block_size: Option<usize>,

    /// KV cache quantization type
    #[arg(long = "pa-cache-type", default_value = "auto", value_parser = parse_cache_type)]
    pub cache_type: PagedCacheType,
}

#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum PagedAttnMode {
    /// Automatic: enabled on CUDA, disabled on CPU
    #[default]
    Auto,
    /// Force enable (error if device doesn't support it)
    On,
    /// Force disable
    Off,
}

impl PagedAttentionOptions {
    pub fn enabled(&self) -> Option<bool> {
        match self.mode {
            PagedAttnMode::Auto => None,
            PagedAttnMode::On => Some(true),
            PagedAttnMode::Off => Some(false),
        }
    }
}

/// Built-in MTP speculative decoding. Qwen3.8-Flash-Next only drafts with its own head.
#[derive(Args, Clone)]
pub struct MtpOptions {
    /// Enable MTP speculative decoding with the head built into the model checkpoint.
    #[arg(long)]
    pub mtp: bool,

    /// Fixed MTP draft tokens per step; omit to adapt the depth automatically.
    #[arg(long, value_parser = parse_positive_usize, requires = "mtp")]
    pub mtp_n_predict: Option<usize>,

    /// MTP draft sampling policy. Auto uses probabilistic drafting when supported.
    #[arg(long, value_enum, default_value_t)]
    pub mtp_draft_sampling: MtpDraftSamplingArg,
}

impl MtpOptions {
    pub fn config(&self) -> Option<mistralrs_core::MtpConfig> {
        self.mtp.then(|| {
            mistralrs_core::MtpConfig::builtin(self.mtp_n_predict)
                .with_draft_sampling_method(self.mtp_draft_sampling.into())
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

impl From<MtpDraftSamplingArg> for mistralrs_core::MtpDraftSamplingMethod {
    fn from(value: MtpDraftSamplingArg) -> Self {
        match value {
            MtpDraftSamplingArg::Auto => Self::Auto,
            MtpDraftSamplingArg::Greedy => Self::Greedy,
            MtpDraftSamplingArg::Probabilistic => Self::Probabilistic,
        }
    }
}

#[derive(Args, Clone)]
pub struct RuntimeOptions {
    /// Maximum concurrent sequences
    #[arg(long, default_value_t = 32)]
    pub max_seqs: usize,

    /// Maximum tokens processed in one paged-attention scheduler step
    #[arg(long, default_value_t = default_max_num_batched_tokens())]
    pub max_num_batched_tokens: NonZeroUsize,

    /// CUDA prompt-token quantum used while decode is resident and for recurrent prefill batching
    #[arg(long, default_value_t = default_max_prefill_chunk_tokens())]
    pub max_prefill_chunk_tokens: NonZeroUsize,

    /// Maximum decode steps before a waiting prefill batch is admitted
    #[arg(long, default_value_t = default_max_decode_steps_before_prefill())]
    pub max_decode_steps_before_prefill: NonZeroUsize,

    /// Disable KV cache entirely
    #[arg(long)]
    pub no_kv_cache: bool,

    /// Number of prefix caches to hold (0 to disable)
    #[arg(long, default_value_t = 16)]
    pub prefix_cache_n: usize,

    /// Custom chat template file (.json or .jinja)
    #[arg(long, short)]
    pub chat_template: Option<PathBuf>,

    /// Explicit JINJA template override
    #[arg(long, short)]
    pub jinja_explicit: Option<PathBuf>,

    #[command(flatten)]
    pub mtp: MtpOptions,
}

#[derive(Args, Clone)]
pub struct BenchRuntimeOptions {
    /// Disable KV cache entirely
    #[arg(long)]
    pub no_kv_cache: bool,

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

    /// Disable per-request HTTP access logs.
    #[arg(long)]
    pub disable_access_log: bool,

    /// Format for HTTP access logs.
    #[arg(long, value_enum, default_value = "text")]
    pub access_log_format: AccessLogFormatArg,

    /// Include health and metrics requests in HTTP access logs.
    #[arg(long)]
    pub access_log_health: bool,

    /// Disable the x-request-id response header.
    #[arg(long)]
    pub disable_request_id_header: bool,

    /// Disable the Prometheus /metrics endpoint.
    #[arg(long)]
    pub disable_metrics: bool,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum AccessLogFormatArg {
    #[default]
    Text,
    Json,
}

impl ServerOptions {
    pub fn observability_config(&self) -> ObservabilityConfig {
        ObservabilityConfig {
            access_log: !self.disable_access_log,
            access_log_format: match self.access_log_format {
                AccessLogFormatArg::Text => AccessLogFormat::Text,
                AccessLogFormatArg::Json => AccessLogFormat::Json,
            },
            access_log_health: self.access_log_health,
            request_id_header: !self.disable_request_id_header,
            metrics: !self.disable_metrics,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum TuneProfileArg {
    Quality,
    Balanced,
    Fast,
}

impl From<TuneProfileArg> for mistralrs_core::TuneProfile {
    fn from(value: TuneProfileArg) -> Self {
        match value {
            TuneProfileArg::Quality => Self::Quality,
            TuneProfileArg::Balanced => Self::Balanced,
            TuneProfileArg::Fast => Self::Fast,
        }
    }
}

pub fn parse_positive_usize(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(0) => Err("value must be greater than 0".to_string()),
        Ok(value) => Ok(value),
        Err(err) => Err(err.to_string()),
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

fn parse_dtype(value: &str) -> Result<ModelDType, String> {
    value.parse()
}

fn parse_cache_type(value: &str) -> Result<PagedCacheType, String> {
    value.parse()
}

/// Expands a leading `~/` in each entry of a `;`-separated path list, since quoting the
/// list stops the shell from doing it.
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
        bail!(
            "{what} `{}` is not a local directory; qwen-fn-rs only loads models that already \
             exist on the filesystem",
            path.display()
        );
    }
    Ok(())
}
