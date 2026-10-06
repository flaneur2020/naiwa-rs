//! `serve`: the OpenAI- and Anthropic-compatible HTTP API.

use anyhow::Result;
use mistralrs_server_core::{
    metrics::install_prometheus_recorder,
    mistralrs_server_router_builder::MistralRsServerRouterBuilder,
    route_registry::{RouteKind, MISTRALRS_API_ROUTES},
};
use tracing::{debug, info};

use crate::args::{GlobalOptions, ModelOptions, RuntimeOptions, ServerOptions};
use crate::load::base_builder;

pub async fn run_serve(
    model: ModelOptions,
    runtime: RuntimeOptions,
    server: ServerOptions,
    global: GlobalOptions,
) -> Result<()> {
    if server.observability_config().metrics {
        install_prometheus_recorder();
    }

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

    axum::serve(listener, app).await?;
    Ok(())
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

    for kind in [RouteKind::OpenAi, RouteKind::Anthropic, RouteKind::MistralRs] {
        for route in MISTRALRS_API_ROUTES.iter().filter(|route| route.kind == kind) {
            debug!("  Route: {}, Methods: {}", route.path, route.methods);
        }
    }
}
