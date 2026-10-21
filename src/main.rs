//! naiwa-rs: a Qwen3.8-Flash-Next-only inference server built on the mistral.rs crates.
//!
//! Models are loaded from the local filesystem only; Hugging Face Hub access is disabled.

mod args;
mod commands;
mod copy_stream;
mod doorbell;
mod executor;
mod host_memory;
mod load;
mod moe;
mod prefill_stream;
mod stats;

/// `NAIWA_STREAM_DBG=1`: the prefill stream's per-layer DMA wait against its compute, so a slow
/// prefill chunk can be attributed instead of guessed at.
pub(crate) fn dbg_note(site: &str, msg: &str) {
    if std::env::var_os("NAIWA_STREAM_DBG").is_some() {
        eprintln!("STREAM_DBG {site}: {msg}");
    }
}

use anyhow::Result;
use clap::Parser;
use mistralrs_core::{
    default_mistralrs_filter, initialize_logging_with_filter, LogVerbosity, HF_HUB_OFFLINE_ENV,
};
use tracing_subscriber::EnvFilter;

use args::{Cli, Command};
use commands::{run_bench, run_quantize, run_serve, run_tune, run_uqff, BenchRunConfig};

// Tensor ops allocate fresh output buffers constantly; mimalloc removes the page-fault
// churn that dominates small-model CPU inference with the system allocator.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> Result<()> {
    // Every artifact must already be local; this keeps the mistral.rs loaders off the network.
    std::env::set_var(HF_HUB_OFFLINE_ENV, "1");
    candle_core::utils::init_global_threadpool();
    let cli = Cli::parse();
    init_tracing(cli.global.verbose);

    match cli.command {
        Command::Serve {
            model,
            runtime,
            server,
        } => run_serve(model, runtime, server, cli.global).await,
        Command::Bench {
            model,
            runtime,
            prompt_len,
            prompt_file,
            gen_len,
            depth,
            iterations,
            warmup,
        } => {
            run_bench(
                model,
                runtime,
                cli.global,
                BenchRunConfig {
                    prompt_lens: prompt_len,
                    prompt_file,
                    gen_len,
                    depths: depth,
                    iterations,
                    warmup,
                },
            )
            .await
        }
        Command::Quantize {
            source,
            device,
            quantization,
            output,
        } => run_quantize(source, device, quantization, output).await,
        Command::Uqff { command } => run_uqff(command, cli.global.verbose > 0).await,
        Command::Tune {
            model,
            profile,
            json,
        } => run_tune(model, profile, json).await,
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
        default_mistralrs_filter(verbosity).add_directive(
            format!("naiwa_rs={level}")
                .parse()
                .expect("valid directive"),
        )
    });
    initialize_logging_with_filter(filter);
}
