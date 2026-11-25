//! `serve`: loads the local GGUF model, warms up the expert cache, starts the `naiwa-runtime`
//! engine worker, and serves the Axum HTTP API until graceful shutdown.

use std::sync::{atomic::AtomicUsize, Arc};

use anyhow::Result;
use naiwa_runtime::{compute_num_blocks, EngineConfig, EngineHandle, DEFAULT_BLOCK_SIZE};
use tracing::info;

use crate::app::args::{GlobalOptions, ModelOptions, RuntimeOptions, ServerOptions};
use crate::app::http::{build_router, AppState};
use crate::app::load::{load_model, query_free_vram_bytes};

pub async fn run_serve(
    model_opts: ModelOptions,
    runtime: RuntimeOptions,
    server: ServerOptions,
    global: GlobalOptions,
    prompt_ceiling: Arc<AtomicUsize>,
) -> Result<()> {
    naiwa_offload::pipeline::prefill::set_prompt_step_ceiling(
        runtime
            .max_num_batched_tokens
            .get()
            .min(runtime.max_prefill_chunk_tokens.get()),
    );

    let mtp_cfg = runtime.mtp.config();
    let chat_tmpl = runtime
        .jinja_explicit
        .as_deref()
        .or(runtime.chat_template.as_deref());

    let loaded = load_model(&model_opts, mtp_cfg.is_some(), chat_tmpl)?;

    // Warm up the expert cache & trim CUDA load-time pool BEFORE allocating KV blocks so the
    // driver returns fragmented load buffers and pre-fills `--expert-profile` slots up front.
    if let Some(provider) = &loaded.provider {
        match provider.warmup()? {
            (0, 0) => {}
            (caps, rings) => info!("warmup: {caps} capture(s), {rings} staging ring(s) fixed"),
        }
    }

    let block_size = model_opts
        .paged_attn
        .block_size
        .unwrap_or(DEFAULT_BLOCK_SIZE);
    let free_vram = query_free_vram_bytes(&loaded.device);
    let num_blocks = compute_num_blocks(
        loaded.model.config(),
        mtp_cfg.is_some(),
        model_opts.paged_attn.context_len,
        model_opts.paged_attn.memory_mb,
        model_opts.paged_attn.memory_fraction,
        block_size,
        loaded.dtype,
        free_vram,
    );
    info!(
        num_blocks,
        block_size,
        kv_tokens = num_blocks * block_size,
        max_seqs = runtime.max_seqs,
        mtp = mtp_cfg.is_some(),
        "Allocating HybridCache and starting naiwa-runtime engine"
    );

    let engine_cfg = EngineConfig {
        max_seqs: runtime.max_seqs,
        max_num_batched_tokens: runtime.max_num_batched_tokens.get(),
        max_prefill_chunk_tokens: runtime.max_prefill_chunk_tokens.get(),
        max_decode_steps_before_prefill: runtime.max_decode_steps_before_prefill.get(),
        num_blocks,
        block_size,
        default_seed: global.seed.unwrap_or(0),
        mtp: mtp_cfg,
        prompt_step_ceiling: Some(prompt_ceiling),
    };

    let engine = EngineHandle::spawn(
        loaded.display_id.clone(),
        loaded.model,
        loaded.tokenizer,
        engine_cfg,
    )?;

    let app = build_router(AppState {
        engine,
        enable_metrics: !server.disable_metrics,
    });

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", server.host, server.port)).await?;
    let listener = tcp_nodelay_listener(listener);

    info!("Server listening on http://{}:{}", server.host, server.port);
    log_api_surfaces(&server.host, server.port);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    if naiwa_offload::config::diag().stats {
        let s = naiwa_offload::stats::dump_cache_stats();
        if s.installed {
            info!(
                "expert cache (process totals): {:.1}% hit rate ({} hits, {} misses, {} staged, {} promotions); \
                 lookahead {} loaded / {} swaps / {} dropped; --pcie-frac {} misses, {:.1} MiB",
                100.0 * s.hit_rate(),
                s.hits,
                s.misses,
                s.staged,
                s.promotions,
                s.lookahead_loads,
                s.swaps,
                s.lookahead_dropped,
                s.pcie_misses,
                s.pcie_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

fn tcp_nodelay_listener(
    listener: tokio::net::TcpListener,
) -> impl axum::serve::Listener<Io = tokio::net::TcpStream, Addr = std::net::SocketAddr> {
    use axum::serve::ListenerExt;

    listener.tap_io(|stream| {
        if let Err(error) = stream.set_nodelay(true) {
            tracing::warn!("failed to set TCP_NODELAY on incoming connection: {error}");
        }
    })
}

fn log_api_surfaces(host: &str, port: u16) {
    let client_host = match host {
        "0.0.0.0" => "localhost",
        "::" => "[::1]",
        host => host,
    };
    let root = format!("http://{client_host}:{port}");
    info!("OpenAI-compatible API:    {root}/v1 (models, completions, chat/completions)");
    info!("Anthropic-compatible API: {root}/v1/messages");
    info!("Health / Metrics:         {root}/health, {root}/metrics");
}
