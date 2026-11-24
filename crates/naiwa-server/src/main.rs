//! `naiwa-rs`: Qwen3.8-Flash-Next GGUF edge inference server built directly on `candle-core`
//! and the 6-crate `naiwa-*` workspace (zero `mistral.rs` dependency).

mod app;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use app::{run_serve, Cli, Command};

// Tensor ops allocate fresh output buffers constantly; mimalloc removes page-fault churn.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> Result<()> {
    candle_core::utils::init_global_threadpool();

    #[cfg(feature = "cuda")]
    if naiwa_offload::stats::model_partition::enabled() {
        naiwa_models::set_boundary_probe(naiwa_offload::stats::model_partition::dispatch);
    }

    if naiwa_offload::config::runtime().cpu_spin_us_raw.is_some() {
        naiwa_quant::install_task_dispatcher(Some(
            naiwa_offload::transport::spin_pool::dispatch,
        ));
    }

    // Live prompt chunk ceiling shared between `naiwa-offload::PrefillStreams` and `naiwa-runtime`.
    let prompt_ceiling = Arc::new(AtomicUsize::new(0));
    {
        static SHARED_CEILING: std::sync::OnceLock<Arc<AtomicUsize>> = std::sync::OnceLock::new();
        let _ = SHARED_CEILING.set(Arc::clone(&prompt_ceiling));
        naiwa_offload::pipeline::prefill::set_prompt_step_publisher(|chunk| {
            if let Some(cell) = SHARED_CEILING.get() {
                cell.store(chunk, Ordering::Release);
            }
        });
    }

    let cli = Cli::parse();
    init_tracing(cli.global.verbose);

    match cli.command {
        Command::Serve {
            model,
            runtime,
            server,
        } => run_serve(model, runtime, server, cli.global, prompt_ceiling).await,
    }
}

fn init_tracing(verbose: u8) {
    let level = match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!(
            "warn,naiwa_rs={level},naiwa_server={level},naiwa_offload={level},naiwa_models={level},naiwa_runtime={level}"
        ))
    });
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}
