//! Direct forward-pass oracle verification for `naiwa-models` + `naiwa-attn` + `naiwa-quant` +
//! `naiwa-offload` (zero `mistral.rs` involved).
//!
//! Run explicitly against the local IQ1_M checkpoint:
//! ```bash
//! cargo test -p naiwa-models --offline --release --features cuda --test direct_oracle -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{quantized::gguf_file::Value, DType, Device, Tensor};
use naiwa_attn::PagedAttnMeta;
use naiwa_models::{ForwardInput, ForwardKind, HybridCache, Qwen4ExpModel};
use naiwa_offload::pipeline::GpuCacheMoeProvider;
use naiwa_quant::GgufArchive;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BpeBuilder,
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

const QWEN35_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN2_REGEX: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

fn build_gguf_tokenizer(archive: &GgufArchive) -> anyhow::Result<Tokenizer> {
    let tokens: Vec<String> = match archive.metadata_value("tokenizer.ggml.tokens") {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => anyhow::bail!("non-string token"),
            })
            .collect::<anyhow::Result<_>>()?,
        _ => anyhow::bail!("missing tokenizer.ggml.tokens"),
    };
    let merges: Vec<(String, String)> = match archive.metadata_value("tokenizer.ggml.merges") {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::String(s) => {
                    let (a, b) = s
                        .split_once(' ')
                        .ok_or_else(|| anyhow::anyhow!("bad merge `{s}`"))?;
                    Ok((a.to_string(), b.to_string()))
                }
                _ => anyhow::bail!("non-string merge"),
            })
            .collect::<anyhow::Result<_>>()?,
        _ => anyhow::bail!("missing tokenizer.ggml.merges"),
    };
    let pre = match archive.metadata_value("tokenizer.ggml.pre") {
        Some(Value::String(s)) => s.as_str(),
        _ => "qwen35",
    };
    let (regex, use_nfc) = match pre {
        "qwen35" => (QWEN35_REGEX, true),
        _ => (QWEN2_REGEX, false),
    };

    let vocab: tokenizers::models::bpe::Vocab = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(false)
        .build()
        .map_err(anyhow::Error::msg)?;
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

    if let Some(Value::Array(types)) = archive.metadata_value("tokenizer.ggml.token_type") {
        if types.len() == tokens.len() {
            let mut specials = Vec::new();
            for (i, ty) in types.iter().enumerate() {
                let t = ty.to_i32().unwrap_or(1);
                if t != 1 && t != 6 {
                    specials.push(AddedToken::from(tokens[i].clone(), true));
                }
            }
            if !specials.is_empty() {
                let _ = tok.add_special_tokens(specials);
            }
        }
    }
    Ok(tok)
}

#[test]
#[ignore]
fn direct_forward_matches_frozen_greedy_oracle() -> anyhow::Result<()> {
    let model_dir = std::env::var("MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME unset");
        format!("{home}/Models/qwen3-coder/IQ1_M")
    });
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&model_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    shards.sort();
    assert!(!shards.is_empty(), "no .gguf shards found in {model_dir}");

    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    // Match capture_oracle.sh: --cpu-moe --expert-cache-gb 2.0 --pcie-frac 0 --no-lookahead
    let provider = Arc::new(GpuCacheMoeProvider::new(
        2.0,
        None,
        0.0,
        1,
        false,
        None,
    ));
    let (model, archive) = Qwen4ExpModel::load_gguf(
        &shards,
        &device,
        dtype,
        Some(provider.clone()),
        false,
    )?;
    let (caps, rings) = provider.warmup()?;
    eprintln!("direct_oracle: loaded model, warmup caps={caps} rings={rings}");

    let tokenizer = build_gguf_tokenizer(&archive)?;
    let prompt = "The quick brown fox jumps over the lazy dog. The capital of France is";
    let enc = tokenizer.encode(prompt, false).map_err(anyhow::Error::msg)?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    let prompt_toks: Vec<_> = prompt_ids.iter().map(|&id| tokenizer.id_to_token(id).unwrap_or_default()).collect();
    eprintln!("direct_oracle: prompt ({} tokens): {prompt_ids:?} = {prompt_toks:?}", prompt_ids.len());

    let block_size = 32usize;
    let num_blocks = 16usize;
    let mut cache = HybridCache::new(model.config(), false, num_blocks, block_size, 1, dtype, &device)?;
    let slot = cache.alloc_slot()?.expect("slot 0");

    // 1. Prefill
    let n_prompt = prompt_ids.len();
    let input_ids = Tensor::from_slice(&prompt_ids, (1, n_prompt), &device)?;
    let pos_vec: Vec<u32> = (0..n_prompt as u32).collect();
    let position_ids = Tensor::from_slice(&pos_vec, (1, n_prompt), &device)?;
    let slot_mapping_vec: Vec<i64> = (0..n_prompt as i64).collect();
    let slot_mapping = Tensor::from_vec(slot_mapping_vec, (n_prompt,), &device)?;
    let block_table_vec: Vec<u32> = (0..num_blocks as u32).collect();
    let block_tables = Tensor::from_vec(block_table_vec.clone(), (1, num_blocks), &device)?;
    let context_lens = Tensor::from_vec(vec![n_prompt as u32], (1,), &device)?;
    let cu_context_lens = Tensor::from_vec(vec![0u32, n_prompt as u32], (2,), &device)?;
    let paged_meta = PagedAttnMeta {
        slot_mapping,
        block_tables: block_tables.clone(),
        context_lens,
        cu_context_lens,
        host_context_lens: vec![n_prompt],
        host_query_lens: vec![n_prompt],
        max_context_len: n_prompt,
        is_decode: false,
        row_block_tables: None,
        row_context_lens: None,
    };

    let last_idx = [(n_prompt - 1) as u32];
    let prefill_out = model.forward(
        &ForwardInput {
            tokens_host: &prompt_ids,
            input_ids: &input_ids,
            position_ids: &position_ids,
            cache_slots: &[slot],
            seq_spans: &[(0, n_prompt)],
            kv_lens: &[n_prompt],
            paged_meta: &paged_meta,
            kind: ForwardKind::Prefill,
            logit_indices: Some(&last_idx),
            capture_spec_hidden: false,
            capture_verify_stash: false,
        },
        &mut cache,
    )?;

    let mut generated = Vec::with_capacity(16);
    let mut next_tok = prefill_out
        .logits
        .argmax(1)?
        .to_vec1::<u32>()?[0];
    generated.push(next_tok);

    // 2. 15 more autoregressive decode steps (16 total tokens, matching capture_oracle.sh)
    for step in 0..15usize {
        let cur_pos = n_prompt + step;
        let cur_kv_len = cur_pos + 1;
        let tok_host = [next_tok];
        let input_ids = Tensor::from_slice(&tok_host, (1, 1), &device)?;
        let position_ids = Tensor::from_slice(&[cur_pos as u32], (1, 1), &device)?;
        let slot_mapping = Tensor::from_vec(vec![cur_pos as i64], (1,), &device)?;
        let context_lens = Tensor::from_vec(vec![cur_kv_len as u32], (1,), &device)?;
        let cu_context_lens = Tensor::from_vec(vec![0u32, cur_kv_len as u32], (2,), &device)?;
        let paged_meta = PagedAttnMeta {
            slot_mapping,
            block_tables: block_tables.clone(),
            context_lens,
            cu_context_lens,
            host_context_lens: vec![cur_kv_len],
            host_query_lens: vec![1],
            max_context_len: cur_kv_len,
            is_decode: true,
            row_block_tables: None,
            row_context_lens: None,
        };
        let out = model.forward(
            &ForwardInput {
                tokens_host: &tok_host,
                input_ids: &input_ids,
                position_ids: &position_ids,
                cache_slots: &[slot],
                seq_spans: &[(0, 1)],
                kv_lens: &[cur_kv_len],
                paged_meta: &paged_meta,
                kind: ForwardKind::Decode,
                logit_indices: None,
                capture_spec_hidden: false,
                capture_verify_stash: false,
            },
            &mut cache,
        )?;
        next_tok = out.logits.argmax(1)?.to_vec1::<u32>()?[0];
        generated.push(next_tok);
    }

    let decoded = tokenizer
        .decode(&generated, true)
        .map_err(anyhow::Error::msg)?;
    eprintln!("direct_oracle: generated ids = {generated:?}");
    eprintln!("direct_oracle: decoded       = {:?}", decoded);
    let expected = " Paris. The capital of Japan is Tokyo. The capital of Italy is Rome.";
    assert_eq!(
        decoded, expected,
        "Direct forward oracle mismatch!\n  got:      {decoded:?}\n  expected: {expected:?}"
    );
    Ok(())
}
