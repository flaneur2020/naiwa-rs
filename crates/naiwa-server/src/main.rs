//! naiwa-rs: a Qwen3.8-Flash-Next-only inference server built on the mistral.rs crates.
//!
//! Models are loaded from the local filesystem only; Hugging Face Hub access is disabled.
//!
//! # Architecture (strict 4-layer DAG + cross-cutting `config` / `stats`)
//! - `transport` (L1): hardware async primitives (copy stream, mapped doorbells, CPU executor, spin pool).
//! - `memory`    (L2): host pinned stacks, VRAM slab/borrow gate, per-layer cache directory, staging rings.
//! - `pipeline`  (L3): mutually-exclusive Decode & Prefill execution pipelines + MoE provider factory.
//! - `app`       (L4): CLI arguments, model loader, and OpenAI/Anthropic HTTP server.

mod app;

use anyhow::Result;
use clap::Parser;
use mistralrs_core::{
    default_mistralrs_filter, initialize_logging_with_filter, LogVerbosity, HF_HUB_OFFLINE_ENV,
};
use tracing_subscriber::EnvFilter;

use app::{run_serve, Cli, Command};

// Tensor ops allocate fresh output buffers constantly; mimalloc removes the page-fault
// churn that dominates small-model CPU inference with the system allocator.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> Result<()> {
    // Every artifact must already be local; this keeps the mistral.rs loaders off the network.
    std::env::set_var(HF_HUB_OFFLINE_ENV, "1");
    candle_core::utils::init_global_threadpool();
    #[cfg(feature = "cuda")]
    if naiwa_offload::stats::model_partition::enabled() {
        assert!(mistralrs_core::qwen4_split::set_boundary_probe(
            naiwa_offload::stats::model_partition::dispatch
        ));
    }
    if naiwa_offload::config::runtime().cpu_spin_us_raw.is_some() {
        naiwa_quant::install_task_dispatcher(Some(
            naiwa_offload::transport::spin_pool::dispatch,
        ));
        mistralrs_quant::gguf::fused_cpu::install_task_dispatcher(Some(
            naiwa_offload::transport::spin_pool::dispatch,
        ));
    }
    naiwa_offload::pipeline::prefill::set_prompt_step_publisher(|chunk| {
        mistralrs_core::PROMPT_STEP_CEILING.store(chunk, std::sync::atomic::Ordering::Release);
    });
    let cli = Cli::parse();
    init_tracing(cli.global.verbose);

    match cli.command {
        Command::Serve {
            model,
            runtime,
            server,
        } => run_serve(model, runtime, server, cli.global).await,
    }
}

// mistral.rs's default filter only covers `mistralrs*` targets; add this crate at the same level.
fn init_tracing(verbose: u8) {
    let verbosity = LogVerbosity::from_count(verbose);
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let level = match verbosity {
            LogVerbosity::Info => "info",
            LogVerbosity::Debug => "debug",
            LogVerbosity::Trace => "trace",
        };
        default_mistralrs_filter(verbosity)
            .add_directive(
                format!("naiwa_rs={level}")
                    .parse()
                    .expect("valid directive"),
            )
            .add_directive(
                format!("naiwa_offload={level}")
                    .parse()
                    .expect("valid directive"),
            )
            .add_directive(
                format!("naiwa_server={level}")
                    .parse()
                    .expect("valid directive"),
            )
    });
    initialize_logging_with_filter(filter);
}
