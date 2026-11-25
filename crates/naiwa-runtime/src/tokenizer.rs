//! GGUF BPE tokenizer construction, incremental UTF-8 streaming decoder, and Jinja2 chat templates.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value;
use minijinja::{Environment, Value as JinjaValue};
use naiwa_quant::GgufArchive;
use serde::{Deserialize, Serialize};
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{BpeBuilder, Vocab},
    normalizers::NFC,
    pre_tokenizers::{
        byte_level::ByteLevel as ByteLevelPre,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    processors::byte_level::ByteLevel as ByteLevelPost,
    AddedToken, ModelWrapper, SplitDelimiterBehavior, Tokenizer,
};

const GPT2_REGEX: &str =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
const LLAMA3_REGEX: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN2_REGEX: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN35_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// One chat message passed to `apply_chat_template`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Tokenizer + stop-token set + Jinja2 chat template loaded directly from a GGUF archive.
pub struct GgufTokenizer {
    tokenizer: Tokenizer,
    bos_token_id: Option<u32>,
    eos_token_id: u32,
    add_bos_token: bool,
    stop_token_ids: HashSet<u32>,
    chat_template: Option<String>,
}

impl GgufTokenizer {
    /// Build tokenizer directly from GGUF metadata (or override with an external `tokenizer.json`
    /// and/or explicit Jinja chat template).
    pub fn from_gguf(
        archive: &GgufArchive,
        external_tokenizer: Option<&Path>,
        chat_template_override: Option<&Path>,
    ) -> Result<Self> {
        let tokenizer = if let Some(path) = external_tokenizer {
            Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", path.display()))?
        } else {
            build_bpe_from_gguf(archive)?
        };

        let eos_token_id = meta_u32(archive, "tokenizer.ggml.eos_token_id")
            .context("GGUF missing `tokenizer.ggml.eos_token_id`")?;
        let bos_token_id = meta_u32(archive, "tokenizer.ggml.bos_token_id");
        let add_bos_token = match archive.metadata_value("tokenizer.ggml.add_bos_token") {
            Some(Value::Bool(b)) => *b,
            _ => false,
        };

        let mut stop_token_ids = HashSet::new();
        stop_token_ids.insert(eos_token_id);
        // Also add common Qwen EOS / turn-end special tokens if present in vocabulary
        for tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
            if let Some(id) = tokenizer.token_to_id(tok) {
                stop_token_ids.insert(id);
            }
        }

        let chat_template = if let Some(path) = chat_template_override {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read chat template {}", path.display()))?;
            // Accept either raw .jinja or tokenizer_config.json with `chat_template` field
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let v: serde_json::Value = serde_json::from_str(&raw)?;
                v.get("chat_template")
                    .and_then(|c| c.as_str())
                    .map(ToOwned::to_owned)
            } else {
                Some(raw)
            }
        } else {
            match archive.metadata_value("tokenizer.chat_template") {
                Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            }
        };

        tracing::info!(
            vocab_size = tokenizer.get_vocab_size(true),
            eos_token_id,
            stop_tokens = stop_token_ids.len(),
            has_chat_template = chat_template.is_some(),
            "Initialized GGUF tokenizer"
        );

        Ok(Self {
            tokenizer,
            bos_token_id,
            eos_token_id,
            add_bos_token,
            stop_token_ids,
            chat_template,
        })
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn is_stop_token(&self, id: u32) -> bool {
        self.stop_token_ids.contains(&id)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Encode text to token IDs (`add_special_tokens` controls BOS prepending when configured).
    pub fn encode(&self, text: &str, add_bos_if_configured: bool) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)?;
        let mut ids = enc.get_ids().to_vec();
        if add_bos_if_configured && self.add_bos_token {
            if let Some(bos) = self.bos_token_id {
                if ids.first().copied() != Some(bos) {
                    ids.insert(0, bos);
                }
            }
        }
        Ok(ids)
    }

    /// Decode a full sequence of token IDs to UTF-8 text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(anyhow::Error::msg)
    }

    /// Render chat messages using the GGUF's Jinja2 `chat_template` (or ChatML fallback).
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        let Some(template_src) = self.chat_template.as_deref() else {
            return Ok(render_chatml_fallback(messages, add_generation_prompt));
        };

        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_unknown_method);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });
        env.add_filter("tojson", |val: JinjaValue| -> Result<String, minijinja::Error> {
            serde_json::to_string(&val).map_err(|e| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
            })
        });
        env.add_template("chat", template_src)
            .context("failed to compile GGUF chat_template")?;
        let tmpl = env.get_template("chat")?;

        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".to_string(), serde_json::to_value(messages)?);
        ctx.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(add_generation_prompt),
        );
        if let Some(et) = enable_thinking {
            ctx.insert("enable_thinking".to_string(), serde_json::Value::Bool(et));
        }
        if let Some(t) = tools {
            ctx.insert("tools".to_string(), t.clone());
        }
        if let Some(bos_id) = self.bos_token_id {
            if let Some(tok) = self.tokenizer.id_to_token(bos_id) {
                ctx.insert("bos_token".to_string(), serde_json::Value::String(tok));
            }
        }
        if let Some(tok) = self.tokenizer.id_to_token(self.eos_token_id) {
            ctx.insert("eos_token".to_string(), serde_json::Value::String(tok));
        }

        let rendered = tmpl
            .render(JinjaValue::from_serialize(&serde_json::Value::Object(ctx)))
            .context("failed to render chat_template")?;
        Ok(rendered)
    }
}

/// Incremental UTF-8 safe streaming token decoder (prevents emitting `\u{FFFD}` across split multi-byte UTF-8).
#[derive(Default)]
pub struct StreamDecoder {
    tokens: Vec<u32>,
    emitted_bytes: usize,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a newly generated token and return any complete UTF-8 delta text ready to stream.
    pub fn push(&mut self, tokenizer: &GgufTokenizer, token_id: u32) -> Result<String> {
        self.tokens.push(token_id);
        let full = tokenizer.decode(&self.tokens, true)?;
        if full.len() <= self.emitted_bytes {
            return Ok(String::new());
        }
        let tail = &full[self.emitted_bytes..];
        // Hold back trailing replacement char (incomplete multi-byte UTF-8 sequence)
        if tail.ends_with('\u{FFFD}') {
            let safe_len = tail.trim_end_matches('\u{FFFD}').len();
            if safe_len == 0 {
                return Ok(String::new());
            }
            let out = tail[..safe_len].to_string();
            self.emitted_bytes += safe_len;
            return Ok(out);
        }
        let out = tail.to_string();
        self.emitted_bytes = full.len();
        Ok(out)
    }

    /// Flush any remaining buffered bytes at end of generation.
    pub fn flush(&mut self, tokenizer: &GgufTokenizer) -> Result<String> {
        let full = tokenizer.decode(&self.tokens, true)?;
        if full.len() <= self.emitted_bytes {
            return Ok(String::new());
        }
        let out = full[self.emitted_bytes..].to_string();
        self.emitted_bytes = full.len();
        Ok(out)
    }
}

fn minijinja_unknown_method(
    _state: &minijinja::State<'_, '_>,
    value: &JinjaValue,
    method: &str,
    args: &[JinjaValue],
) -> std::result::Result<JinjaValue, minijinja::Error> {
    if let Some(s) = value.as_str() {
        match method {
            "strip" => {
                let chars = args.first().and_then(|a| a.as_str());
                let res = match chars {
                    Some(c) => s.trim_matches(|ch| c.contains(ch)),
                    None => s.trim(),
                };
                return Ok(JinjaValue::from(res));
            }
            "lstrip" => {
                let chars = args.first().and_then(|a| a.as_str());
                let res = match chars {
                    Some(c) => s.trim_start_matches(|ch| c.contains(ch)),
                    None => s.trim_start(),
                };
                return Ok(JinjaValue::from(res));
            }
            "rstrip" => {
                let chars = args.first().and_then(|a| a.as_str());
                let res = match chars {
                    Some(c) => s.trim_end_matches(|ch| c.contains(ch)),
                    None => s.trim_end(),
                };
                return Ok(JinjaValue::from(res));
            }
            "startswith" => {
                if let Some(p) = args.first().and_then(|a| a.as_str()) {
                    return Ok(JinjaValue::from(s.starts_with(p)));
                }
            }
            "endswith" => {
                if let Some(p) = args.first().and_then(|a| a.as_str()) {
                    return Ok(JinjaValue::from(s.ends_with(p)));
                }
            }
            "split" => {
                let sep = args.first().and_then(|a| a.as_str());
                let parts: Vec<String> = match sep {
                    Some(sp) => s.split(sp).map(ToOwned::to_owned).collect(),
                    None => s.split_whitespace().map(ToOwned::to_owned).collect(),
                };
                return Ok(JinjaValue::from(parts));
            }
            _ => {}
        }
    }
    Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod))
}

fn render_chatml_fallback(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(&m.role);
        out.push('\n');
        match &m.content {
            serde_json::Value::String(s) => out.push_str(s),
            serde_json::Value::Array(arr) => {
                for item in arr {
                    if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                        out.push_str(t);
                    }
                }
            }
            _ => {}
        }
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

fn build_bpe_from_gguf(archive: &GgufArchive) -> Result<Tokenizer> {
    let model = match archive.metadata_value("tokenizer.ggml.model") {
        Some(Value::String(s)) => s.as_str(),
        _ => bail!("GGUF missing `tokenizer.ggml.model`"),
    };
    if model != "gpt2" {
        bail!("unsupported GGUF tokenizer model `{model}` (expected `gpt2` BPE)");
    }

    let tokens: Vec<String> = match archive.metadata_value("tokenizer.ggml.tokens") {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => bail!("non-string entry in `tokenizer.ggml.tokens`"),
            })
            .collect::<Result<_>>()?,
        _ => bail!("GGUF missing `tokenizer.ggml.tokens`"),
    };
    let merges: Vec<(String, String)> = match archive.metadata_value("tokenizer.ggml.merges") {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::String(s) => {
                    let (a, b) = s
                        .split_once(' ')
                        .with_context(|| format!("invalid BPE merge `{s}`"))?;
                    Ok((a.to_string(), b.to_string()))
                }
                _ => bail!("non-string entry in `tokenizer.ggml.merges`"),
            })
            .collect::<Result<_>>()?,
        _ => bail!("GGUF missing `tokenizer.ggml.merges`"),
    };
    let pre = match archive.metadata_value("tokenizer.ggml.pre") {
        Some(Value::String(s)) => s.as_str(),
        _ => bail!("GGUF missing `tokenizer.ggml.pre`"),
    };
    let (regex, ignore_merges, use_nfc) = match pre {
        "qwen35" => (QWEN35_REGEX, false, true),
        "qwen2" | "deepseek-r1-qwen" | "hunyuan" => (QWEN2_REGEX, false, false),
        "llama3" | "llama-v3" | "llama-bpe" => (LLAMA3_REGEX, true, false),
        "gpt-2" | "phi-2" => (GPT2_REGEX, false, false),
        other => bail!("unsupported GGUF BPE pre-tokenizer `{other}`"),
    };

    let vocab: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let mut builder = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(ignore_merges);
    if let Some(unk_id) = meta_u32(archive, "tokenizer.ggml.unknown_token_id") {
        if let Some(unk) = tokens.get(unk_id as usize) {
            builder = builder.unk_token(unk.clone());
        }
    }
    let bpe = builder.build().map_err(anyhow::Error::msg)?;
    let mut tok = Tokenizer::new(ModelWrapper::BPE(bpe));
    if use_nfc {
        let _ = tok.with_normalizer(Some(NFC));
    }
    let split = Split::new(
        SplitPattern::Regex(regex.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(anyhow::Error::msg)?;
    tok.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevelPre::new(false, false, false)),
    ])));
    tok.with_decoder(Some(ByteLevelDecoder::new(false, false, false)));
    tok.with_post_processor(Some(ByteLevelPost::new(false, false, false)));

    let mut specials = Vec::new();
    if let Some(Value::Array(types)) = archive.metadata_value("tokenizer.ggml.token_type") {
        if types.len() == tokens.len() {
            for (i, ty) in types.iter().enumerate() {
                let t = ty.to_i32().unwrap_or(1);
                if t != 1 && t != 6 {
                    specials.push(AddedToken::from(tokens[i].clone(), true));
                }
            }
        }
    }
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
    ] {
        if let Some(id) = meta_u32(archive, key) {
            if let Some(s) = tokens.get(id as usize) {
                specials.push(AddedToken::from(s.clone(), true));
            }
        }
    }
    if !specials.is_empty() {
        let _ = tok.add_special_tokens(specials);
    }
    Ok(tok)
}

fn meta_u32(archive: &GgufArchive, key: &str) -> Option<u32> {
    match archive.metadata_value(key)? {
        Value::U8(v) => Some(u32::from(*v)),
        Value::U16(v) => Some(u32::from(*v)),
        Value::U32(v) => Some(*v),
        Value::U64(v) => u32::try_from(*v).ok(),
        Value::I32(v) if *v >= 0 => Some(*v as u32),
        Value::I64(v) if *v >= 0 => u32::try_from(*v).ok(),
        _ => None,
    }
}
