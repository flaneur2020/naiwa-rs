# qwen-fn-rs

A Qwen3.8-Flash-Next-only inference server built on the
[mistral.rs](https://github.com/EricLBuehler/mistral.rs) crates (`mistralrs-core`,
`mistralrs-server-core`, `mistralrs-quant`), pinned to commit `3f2515e9b`, the commit that
added Qwen3.8-Flash-Next support.

- Loads only `Qwen4ExpForConditionalGeneration` checkpoints (safetensors/UQFF) or GGUF files
  with `general.architecture = qwen4exp`. The architecture is checked before loading, and the
  Qwen4-Exp loader is always selected explicitly.
- Local files only: `-m`/`-f` must point at the filesystem, and Hugging Face Hub access is
  disabled (`HF_HUB_OFFLINE=1`).
- No web UI, Swagger UI, agent tools, sandbox, LoRA, or interactive shell.

## Build

```bash
cargo build --release                      # CPU
cargo build --release --features cuda      # NVIDIA GPU (needs the CUDA toolkit / nvcc)
```

Optional CUDA extras: `cudnn`, `flash-attn`, `flash-attn-v3`, `nccl`, `cutile`. Optional CPU extra: `mkl`.

## Commands

```bash
# OpenAI-compatible API at http://localhost:1234/v1 (Anthropic-compatible at the root)
qwen-fn-rs serve -m /models/Qwen3.8-Flash-Next
qwen-fn-rs serve -m /models/Qwen3.8-Flash-Next --isq 4 --mtp
qwen-fn-rs serve -f /models/gguf/Qwen3.8-Flash-Next-Q4_K_M.gguf --mmproj mmproj-BF16.gguf

# Throughput / latency
qwen-fn-rs bench -m /models/Qwen3.8-Flash-Next --prompt-len 512,2048 --depth 4,4096

# Write UQFF, then serve it
qwen-fn-rs quantize -m /models/Qwen3.8-Flash-Next --isq q4k -o /models/Qwen3.8-Flash-Next
qwen-fn-rs serve -m /models/Qwen3.8-Flash-Next --from-uqff q4k

# Inspect UQFF artifacts; recommend a quantization for this machine
qwen-fn-rs uqff verify -m /models/Qwen3.8-Flash-Next
qwen-fn-rs tune -m /models/Qwen3.8-Flash-Next
```

Flag names and defaults follow `mistralrs` for the options that are kept. Run
`qwen-fn-rs <command> --help` for the full list.

## Updating mistral.rs

Bump the `rev` of the mistral.rs crates in the root `Cargo.toml`, and keep `candle-core` on the
same candle revision that mistral.rs uses. To build against a local checkout, replace the
`git`/`rev` pairs with `path = "../mistral.rs/<crate>"`.
