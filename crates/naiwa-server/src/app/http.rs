//! Lightweight OpenAI- and Anthropic-compatible Axum HTTP server.
//!
//! Endpoints:
//! - `GET  /health`
//! - `GET  /metrics`
//! - `GET  /v1/models`
//! - `POST /v1/completions`        (non-streaming + SSE streaming)
//! - `POST /v1/chat/completions`   (non-streaming + SSE streaming)
//! - `POST /v1/messages`           (Anthropic Messages API, non-streaming + SSE streaming)

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream;
use naiwa_runtime::{
    ChatMessage, EngineHandle, FinishReason, GenerationRequest, SamplingParams, StreamEvent,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
    pub enable_metrics: bool,
}

pub fn build_router(state: AppState) -> Router {
    let mut router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages));
    if state.enable_metrics {
        router = router.route("/metrics", get(metrics));
    }
    router.with_state(Arc::new(state))
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.engine.metrics().render_prometheus();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let now = unix_now();
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.engine.model_id(),
            "object": "model",
            "created": now,
            "owned_by": "naiwa-rs",
            "max_model_len": state.engine.max_model_len(),
        }]
    }))
}

// ---------------------------------------------------------------------------
// OpenAI /v1/completions
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CompletionRequest {
    #[allow(dead_code)]
    model: Option<String>,
    prompt: Value,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    ignore_eos: bool,
}

#[derive(Deserialize, Default)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let prompt_str = match extract_prompt_text(&req.prompt) {
        Ok(s) => s,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let prompt_tokens = match state.engine.tokenizer().encode(&prompt_str, true) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("tokenize failed: {e}")),
    };
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        seed: req.seed,
        max_tokens: req.max_tokens.unwrap_or(256),
        stop: parse_stop_list(req.stop.as_ref()),
        stop_token_ids: Vec::new(),
        ignore_eos: req.ignore_eos,
    };

    let rx = match state.engine.generate(GenerationRequest {
        prompt_tokens,
        sampling,
    }) {
        Ok(rx) => rx,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };

    let id = format!("cmpl-{}", Uuid::new_v4().simple());
    let created = unix_now();
    let model = state.engine.model_id().to_string();

    if req.stream {
        let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
        return sse_completions_stream(rx, id, created, model, include_usage);
    }

    match collect_all(rx).await {
        Ok(done) => Json(json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "text": done.text,
                "logprobs": Value::Null,
                "finish_reason": done.finish_reason.as_openai_str(),
            }],
            "usage": {
                "prompt_tokens": done.prompt_tokens,
                "completion_tokens": done.completion_tokens,
                "total_tokens": done.prompt_tokens + done.completion_tokens,
            }
        }))
        .into_response(),
        Err(msg) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &msg),
    }
}

fn sse_completions_stream(
    rx: mpsc::UnboundedReceiver<StreamEvent>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
) -> Response {
    struct Ctx {
        rx: mpsc::UnboundedReceiver<StreamEvent>,
        id: String,
        created: u64,
        model: String,
        include_usage: bool,
        finished: bool,
        sent_done_marker: bool,
    }
    let ctx = Ctx {
        rx,
        id,
        created,
        model,
        include_usage,
        finished: false,
        sent_done_marker: false,
    };
    let s = stream::unfold(ctx, |mut c| async move {
        if c.sent_done_marker {
            return None;
        }
        if c.finished {
            c.sent_done_marker = true;
            return Some((Ok::<Event, Infallible>(Event::default().data("[DONE]")), c));
        }
        match c.rx.recv().await {
            Some(StreamEvent::Token { text, .. }) => {
                let payload = json!({
                    "id": c.id,
                    "object": "text_completion",
                    "created": c.created,
                    "model": c.model,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "logprobs": Value::Null,
                        "finish_reason": Value::Null,
                    }]
                });
                Some((Ok(Event::default().data(payload.to_string())), c))
            }
            Some(StreamEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                ..
            }) => {
                c.finished = true;
                let mut payload = json!({
                    "id": c.id,
                    "object": "text_completion",
                    "created": c.created,
                    "model": c.model,
                    "choices": [{
                        "index": 0,
                        "text": "",
                        "logprobs": Value::Null,
                        "finish_reason": finish_reason.as_openai_str(),
                    }]
                });
                if c.include_usage {
                    payload["usage"] = json!({
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    });
                }
                Some((Ok(Event::default().data(payload.to_string())), c))
            }
            Some(StreamEvent::Error(e)) => {
                c.sent_done_marker = true;
                let payload = json!({ "error": { "message": e, "type": "server_error" } });
                Some((Ok(Event::default().data(payload.to_string())), c))
            }
            None => {
                c.sent_done_marker = true;
                Some((Ok(Event::default().data("[DONE]")), c))
            }
        }
    });
    Sse::new(s).keep_alive(KeepAlive::default()).into_response()
}

// ---------------------------------------------------------------------------
// OpenAI /v1/chat/completions
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    chat_template_kwargs: Option<Value>,
    #[serde(default)]
    ignore_eos: bool,
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let enable_thinking = req.enable_thinking.or_else(|| {
        req.chat_template_kwargs
            .as_ref()
            .and_then(|v| v.get("enable_thinking"))
            .and_then(Value::as_bool)
    });

    let rendered = match state.engine.tokenizer().apply_chat_template(
        &req.messages,
        true,
        enable_thinking,
        req.tools.as_ref(),
    ) {
        Ok(s) => s,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("chat_template error: {e}"),
            )
        }
    };

    let prompt_tokens = match state.engine.tokenizer().encode(&rendered, false) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("tokenize failed: {e}")),
    };

    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        seed: req.seed,
        max_tokens: req.max_completion_tokens.or(req.max_tokens).unwrap_or(512),
        stop: parse_stop_list(req.stop.as_ref()),
        stop_token_ids: Vec::new(),
        ignore_eos: req.ignore_eos,
    };

    let rx = match state.engine.generate(GenerationRequest {
        prompt_tokens,
        sampling,
    }) {
        Ok(rx) => rx,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };

    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = unix_now();
    let model = state.engine.model_id().to_string();

    if req.stream {
        let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
        return sse_chat_stream(rx, id, created, model, include_usage);
    }

    match collect_all(rx).await {
        Ok(done) => Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": done.text,
                },
                "finish_reason": done.finish_reason.as_openai_str(),
            }],
            "usage": {
                "prompt_tokens": done.prompt_tokens,
                "completion_tokens": done.completion_tokens,
                "total_tokens": done.prompt_tokens + done.completion_tokens,
            }
        }))
        .into_response(),
        Err(msg) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &msg),
    }
}

fn sse_chat_stream(
    rx: mpsc::UnboundedReceiver<StreamEvent>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
) -> Response {
    struct Ctx {
        rx: mpsc::UnboundedReceiver<StreamEvent>,
        id: String,
        created: u64,
        model: String,
        include_usage: bool,
        sent_role: bool,
        finished: bool,
        sent_done_marker: bool,
    }
    let ctx = Ctx {
        rx,
        id,
        created,
        model,
        include_usage,
        sent_role: false,
        finished: false,
        sent_done_marker: false,
    };
    let s = stream::unfold(ctx, |mut c| async move {
        if c.sent_done_marker {
            return None;
        }
        if c.finished {
            c.sent_done_marker = true;
            return Some((Ok::<Event, Infallible>(Event::default().data("[DONE]")), c));
        }
        match c.rx.recv().await {
            Some(StreamEvent::Token { text, .. }) => {
                let delta = if !c.sent_role {
                    c.sent_role = true;
                    json!({ "role": "assistant", "content": text })
                } else {
                    json!({ "content": text })
                };
                let payload = json!({
                    "id": c.id,
                    "object": "chat.completion.chunk",
                    "created": c.created,
                    "model": c.model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": Value::Null,
                    }]
                });
                Some((Ok(Event::default().data(payload.to_string())), c))
            }
            Some(StreamEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                ..
            }) => {
                c.finished = true;
                let mut payload = json!({
                    "id": c.id,
                    "object": "chat.completion.chunk",
                    "created": c.created,
                    "model": c.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason.as_openai_str(),
                    }]
                });
                if c.include_usage {
                    payload["usage"] = json!({
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    });
                }
                Some((Ok(Event::default().data(payload.to_string())), c))
            }
            Some(StreamEvent::Error(e)) => {
                c.sent_done_marker = true;
                let payload = json!({ "error": { "message": e, "type": "server_error" } });
                Some((Ok(Event::default().data(payload.to_string())), c))
            }
            None => {
                c.sent_done_marker = true;
                Some((Ok(Event::default().data("[DONE]")), c))
            }
        }
    });
    Sse::new(s).keep_alive(KeepAlive::default()).into_response()
}

// ---------------------------------------------------------------------------
// Anthropic /v1/messages
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AnthropicMessageRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    system: Option<Value>,
    max_tokens: usize,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    stream: bool,
}

async fn anthropic_messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnthropicMessageRequest>,
) -> Response {
    let mut messages = Vec::with_capacity(req.messages.len() + 1);
    if let Some(sys) = req.system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: sys,
            tool_calls: None,
            reasoning_content: None,
        });
    }
    messages.extend(req.messages);

    let rendered = match state
        .engine
        .tokenizer()
        .apply_chat_template(&messages, true, None, None)
    {
        Ok(s) => s,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("chat_template error: {e}"),
            )
        }
    };
    let prompt_tokens = match state.engine.tokenizer().encode(&rendered, false) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("tokenize failed: {e}")),
    };
    let prompt_len = prompt_tokens.len();

    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        seed: None,
        max_tokens: req.max_tokens,
        stop: req.stop_sequences.unwrap_or_default(),
        stop_token_ids: Vec::new(),
        ignore_eos: false,
    };

    let rx = match state.engine.generate(GenerationRequest {
        prompt_tokens,
        sampling,
    }) {
        Ok(rx) => rx,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };

    let id = format!("msg_{}", Uuid::new_v4().simple());
    let model = state.engine.model_id().to_string();

    if req.stream {
        return sse_anthropic_stream(rx, id, model, prompt_len);
    }

    match collect_all(rx).await {
        Ok(done) => Json(json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{
                "type": "text",
                "text": done.text,
            }],
            "stop_reason": done.finish_reason.as_anthropic_str(),
            "stop_sequence": Value::Null,
            "usage": {
                "input_tokens": done.prompt_tokens,
                "output_tokens": done.completion_tokens,
            }
        }))
        .into_response(),
        Err(msg) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &msg),
    }
}

fn sse_anthropic_stream(
    rx: mpsc::UnboundedReceiver<StreamEvent>,
    id: String,
    model: String,
    prompt_tokens: usize,
) -> Response {
    struct Ctx {
        rx: mpsc::UnboundedReceiver<StreamEvent>,
        id: String,
        model: String,
        prompt_tokens: usize,
        stage: u8, // 0=send start, 1=send block_start, 2=streaming, 3=send delta, 4=send stop, 5=done
        finish_reason: FinishReason,
        completion_tokens: usize,
    }
    let ctx = Ctx {
        rx,
        id,
        model,
        prompt_tokens,
        stage: 0,
        finish_reason: FinishReason::Stop,
        completion_tokens: 0,
    };
    let s = stream::unfold(ctx, |mut c| async move {
        match c.stage {
            0 => {
                c.stage = 1;
                let data = json!({
                    "type": "message_start",
                    "message": {
                        "id": c.id,
                        "type": "message",
                        "role": "assistant",
                        "model": c.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "usage": { "input_tokens": c.prompt_tokens, "output_tokens": 0 }
                    }
                });
                Some((
                    Ok::<Event, Infallible>(Event::default().event("message_start").data(data.to_string())),
                    c,
                ))
            }
            1 => {
                c.stage = 2;
                let data = json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                });
                Some((
                    Ok(Event::default()
                        .event("content_block_start")
                        .data(data.to_string())),
                    c,
                ))
            }
            2 => match c.rx.recv().await {
                Some(StreamEvent::Token { text, .. }) => {
                    let data = json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": { "type": "text_delta", "text": text }
                    });
                    Some((
                        Ok(Event::default()
                            .event("content_block_delta")
                            .data(data.to_string())),
                        c,
                    ))
                }
                Some(StreamEvent::Done {
                    finish_reason,
                    completion_tokens,
                    ..
                }) => {
                    c.finish_reason = finish_reason;
                    c.completion_tokens = completion_tokens;
                    c.stage = 3;
                    let data = json!({ "type": "content_block_stop", "index": 0 });
                    Some((
                        Ok(Event::default()
                            .event("content_block_stop")
                            .data(data.to_string())),
                        c,
                    ))
                }
                Some(StreamEvent::Error(e)) => {
                    c.stage = 5;
                    let data = json!({ "type": "error", "error": { "type": "api_error", "message": e } });
                    Some((
                        Ok(Event::default().event("error").data(data.to_string())),
                        c,
                    ))
                }
                None => None,
            },
            3 => {
                c.stage = 4;
                let data = json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": c.finish_reason.as_anthropic_str(), "stop_sequence": Value::Null },
                    "usage": { "output_tokens": c.completion_tokens }
                });
                Some((
                    Ok(Event::default().event("message_delta").data(data.to_string())),
                    c,
                ))
            }
            4 => {
                c.stage = 5;
                let data = json!({ "type": "message_stop" });
                Some((
                    Ok(Event::default().event("message_stop").data(data.to_string())),
                    c,
                ))
            }
            _ => None,
        }
    });
    Sse::new(s).keep_alive(KeepAlive::default()).into_response()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct Collected {
    text: String,
    finish_reason: FinishReason,
    prompt_tokens: usize,
    completion_tokens: usize,
}

async fn collect_all(mut rx: mpsc::UnboundedReceiver<StreamEvent>) -> Result<Collected, String> {
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Token { text: t, .. } => text.push_str(&t),
            StreamEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                return Ok(Collected {
                    text,
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                });
            }
            StreamEvent::Error(e) => return Err(e),
        }
    }
    Err("engine stream closed unexpectedly".to_string())
}

fn extract_prompt_text(val: &Value) -> Result<String, String> {
    match val {
        Value::String(s) => Ok(s.clone()),
        Value::Array(arr) => {
            if let Some(Value::String(first)) = arr.first() {
                Ok(first.clone())
            } else {
                Err("prompt array must contain a string".to_string())
            }
        }
        _ => Err("prompt must be a string".to_string()),
    }
}

fn parse_stop_list(val: Option<&Value>) -> Vec<String> {
    match val {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(ToOwned::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(ErrorBody {
            error: ErrorDetail {
                message,
                kind: "invalid_request_error",
            },
        }),
    )
        .into_response()
}

fn unix_now() -> u64 {
    chrono::Utc::now().timestamp().max(0) as u64
}
