//! `serve`: the OpenAI- and Anthropic-compatible HTTP API.

use anyhow::Result;
use mistralrs_server_core::{
    metrics::install_prometheus_recorder,
    mistralrs_server_router_builder::MistralRsServerRouterBuilder,
    route_registry::{RouteKind, MISTRALRS_API_ROUTES},
};
use tracing::{debug, info};

use crate::app::args::{GlobalOptions, ModelOptions, RuntimeOptions, ServerOptions};
use crate::app::load::base_builder;

pub async fn run_serve(
    model: ModelOptions,
    runtime: RuntimeOptions,
    server: ServerOptions,
    global: GlobalOptions,
) -> Result<()> {
    if server.observability_config().metrics {
        install_prometheus_recorder();
    }

    // The prompt step's ceiling: the prefill path plans below it from the free VRAM it measures at
    // warmup (see `prefill_stream::plan`), so both flags are ceilings here, not settings.
    naiwa_offload::pipeline::prefill::set_prompt_step_ceiling(
        runtime
            .max_num_batched_tokens
            .get()
            .min(runtime.max_prefill_chunk_tokens.get()),
    );
    let (builder, model_id) = base_builder(&model, &global)?;
    info!("Loading `{model_id}`");
    let mistralrs = builder
        .with_max_seqs(runtime.max_seqs)
        .with_max_num_batched_tokens(runtime.max_num_batched_tokens)
        .with_max_prefill_chunk_tokens(runtime.max_prefill_chunk_tokens)
        .with_max_decode_steps_before_prefill(runtime.max_decode_steps_before_prefill)
        .with_no_kv_cache(runtime.no_kv_cache)
        .with_prefix_cache_n(runtime.prefix_cache_n)
        .with_chat_template_optional(
            runtime
                .chat_template
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
        )
        .with_jinja_explicit_optional(
            runtime
                .jinja_explicit
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
        )
        .with_mtp_config_optional(runtime.mtp.config())
        .build()
        .await?;

    // v2 M1: build every staging ring up front, after load and before the first request, so
    // ring addresses are fixed for the process lifetime (a capture needs that).
    match crate::app::load::warmup()? {
        (0, 0) => {}
        (caps, rings) => info!("warmup: {caps} capture(s), {rings} staging ring(s) fixed"),
    }

    let app = MistralRsServerRouterBuilder::new()
        .with_mistralrs(mistralrs)
        .with_observability_config(server.observability_config())
        .build()
        .await?;

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", server.host, server.port)).await?;
    let listener = tcp_nodelay_listener(listener);

    info!("Server listening on http://{}:{}", server.host, server.port);
    log_api_surfaces(&server.host, server.port);

    // **Graceful shutdown, because the counters are the instrument.** `harness/tpot/ladder.py` compares cache sizes
    // by the *mechanism* (hits, misses, PCIe bytes a token) as well as by wall time, and the process has to survive
    // long enough to print them: SIGTERM used to kill `axum::serve` outright, taking the numbers with it.
    // `NAIWA_STATS=1` is the switch, so a normal serve is unchanged.
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

/// SIGTERM or Ctrl-C, whichever comes first.
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
            // No handler available: fall back to the other arm rather than exiting early.
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

    info!("OpenAI-compatible API: {root}/v1");
    info!("Anthropic-compatible API: {root}");

    for kind in [
        RouteKind::OpenAi,
        RouteKind::Anthropic,
        RouteKind::MistralRs,
    ] {
        for route in MISTRALRS_API_ROUTES
            .iter()
            .filter(|route| route.kind == kind)
        {
            debug!("  Route: {}, Methods: {}", route.path, route.methods);
        }
    }
}
