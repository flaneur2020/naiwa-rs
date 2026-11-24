//! End-to-end runtime oracle (non-MTP byte-identical + Built-in MTP2 high-acceptance verification).
//!
//! Run explicitly against the local IQ1_M checkpoint + MTP component:
//! ```bash
//! cargo test -p naiwa-runtime --offline --release --features cuda --test runtime_oracle -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::{atomic::Ordering, Arc};

use candle_core::{DType, Device};
use naiwa_models::Qwen4ExpModel;
use naiwa_offload::pipeline::GpuCacheMoeProvider;
use naiwa_runtime::{
    EngineConfig, EngineHandle, GenerationRequest, GgufTokenizer, MtpConfig, MtpDraftSampling,
    SamplingParams, StreamEvent,
};

async fn collect_completion(engine: &EngineHandle, prompt: &str, max_tokens: usize) -> anyhow::Result<String> {
    let prompt_tokens = engine.tokenizer().encode(prompt, true)?;
    let mut rx = engine.generate(GenerationRequest {
        prompt_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(0),
            max_tokens,
            ..Default::default()
        },
    })?;
    let mut out = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Token { text, .. } => out.push_str(&text),
            StreamEvent::Done {
                completion_tokens,
                ttft_ms,
                total_ms,
                ..
            } => {
                eprintln!(
                    "runtime_oracle: done tokens={completion_tokens} ttft={ttft_ms:.1}ms total={total_ms:.1}ms mtp_draft={} mtp_acc={}",
                    engine.metrics().mtp.drafted_tokens.load(Ordering::Relaxed),
                    engine.metrics().mtp.accepted_tokens.load(Ordering::Relaxed),
                );
                break;
            }
            StreamEvent::Error(e) => anyhow::bail!("engine error: {e}"),
        }
    }
    Ok(out)
}

#[tokio::test]
#[ignore]
async fn runtime_engine_non_mtp_and_mtp2_oracle() -> anyhow::Result<()> {
    let home = std::env::var("HOME").expect("HOME unset");
    let model_dir = std::env::var("MODEL").unwrap_or_else(|_| format!("{home}/Models/qwen3-coder/IQ1_M"));
    let mtp_path = std::env::var("MMPROJ").unwrap_or_else(|_| format!("{home}/Models/mtp/mtp-component.gguf"));

    let mut shards: Vec<PathBuf> = std::fs::read_dir(&model_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    shards.sort();
    assert!(!shards.is_empty(), "no GGUF shards in {model_dir}");

    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;
    let prompt = "The quick brown fox jumps over the lazy dog. The capital of France is";
    let expected = " Paris. The capital of Japan is Tokyo. The capital of Italy is Rome.";

    // 1. Standard non-MTP EngineHandle must match the frozen greedy oracle byte-for-byte
    {
        let provider = Arc::new(GpuCacheMoeProvider::new(2.0, None, 0.0, 1, false, None));
        let (model, archive) =
            Qwen4ExpModel::load_gguf(&shards, &device, dtype, Some(provider.clone()), false)?;
        let _ = provider.warmup()?;
        let tokenizer = Arc::new(GgufTokenizer::from_gguf(&archive, None, None)?);
        let cfg = EngineConfig {
            max_seqs: 4,
            num_blocks: 64,
            block_size: 32,
            default_seed: 0,
            mtp: None,
            ..Default::default()
        };
        let engine = EngineHandle::spawn("qwen4-exp-std".to_string(), model, tokenizer, cfg)?;
        let got = collect_completion(&engine, prompt, 16).await?;
        eprintln!("runtime_oracle (non-MTP): got = {got:?}");
        assert_eq!(got, expected, "non-MTP runtime engine must match frozen greedy oracle");
    }

    // 2. Built-in MTP2 EngineHandle must run cleanly with high draft acceptance (>50%)
    if PathBuf::from(&mtp_path).is_file() {
        let mut mtp_shards = shards.clone();
        mtp_shards.push(PathBuf::from(&mtp_path));
        let provider = Arc::new(GpuCacheMoeProvider::new(2.0, None, 0.0, 1, false, None));
        let (model, archive) =
            Qwen4ExpModel::load_gguf(&mtp_shards, &device, dtype, Some(provider.clone()), true)?;
        let _ = provider.warmup()?;
        let tokenizer = Arc::new(GgufTokenizer::from_gguf(&archive, None, None)?);
        let cfg = EngineConfig {
            max_seqs: 4,
            num_blocks: 64,
            block_size: 32,
            default_seed: 0,
            mtp: Some(MtpConfig {
                n_predict: 2,
                draft_sampling: MtpDraftSampling::Greedy,
            }),
            ..Default::default()
        };
        let engine = EngineHandle::spawn("qwen4-exp-mtp2".to_string(), model, tokenizer, cfg)?;
        let got = collect_completion(&engine, prompt, 16).await?;
        eprintln!("runtime_oracle (MTP2):    got = {got:?}");
        assert!(got.starts_with(" Paris. The "), "unexpected MTP2 output: {got:?}");
        let drafted = engine.metrics().mtp.drafted_tokens.load(Ordering::Relaxed);
        let accepted = engine.metrics().mtp.accepted_tokens.load(Ordering::Relaxed);
        assert!(
            drafted >= 8 && accepted * 2 >= drafted,
            "expected >50% MTP2 acceptance, got {accepted}/{drafted}"
        );
    }
    Ok(())
}
