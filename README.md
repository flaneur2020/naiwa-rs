# naiwa-rs

A Qwen3.8-Flash-Next-only inference server built on the
[mistral.rs](https://github.com/EricLBuehler/mistral.rs) crates (`mistralrs-core`,
`mistralrs-server-core`, `mistralrs-quant`), taken from the `patch/strata` branch of
[flaneur2020/mistral.rs](https://github.com/flaneur2020/mistral.rs). That branch is upstream's
Qwen3.8-Flash-Next support (#2462) plus an external routed-experts hook
(`mistralrs_core::moe_offload`).

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
naiwa-rs serve -m /models/Qwen3.8-Flash-Next
naiwa-rs serve -m /models/Qwen3.8-Flash-Next --isq 4 --mtp
naiwa-rs serve -f /models/gguf/Qwen3.8-Flash-Next-Q4_K_M.gguf --mmproj mmproj-BF16.gguf

# Throughput / latency
naiwa-rs bench -m /models/Qwen3.8-Flash-Next --prompt-len 512,2048 --depth 4,4096

# Write UQFF, then serve it
naiwa-rs quantize -m /models/Qwen3.8-Flash-Next --isq q4k -o /models/Qwen3.8-Flash-Next
naiwa-rs serve -m /models/Qwen3.8-Flash-Next --from-uqff q4k

# Inspect UQFF artifacts; recommend a quantization for this machine
naiwa-rs uqff verify -m /models/Qwen3.8-Flash-Next
naiwa-rs tune -m /models/Qwen3.8-Flash-Next
```

Flag names and defaults follow `mistralrs` for the options that are kept. Run
`naiwa-rs <command> --help` for the full list.

## Updating mistral.rs

Push to `patch/strata` on the fork, then run `cargo update -p mistralrs-core` to move
`Cargo.lock` to the new commit. Keep `candle-core` on the same candle revision that mistral.rs
uses. To build against a local checkout, replace the `git`/`branch` pairs with
`path = "../mistral.rs/<crate>"`.
