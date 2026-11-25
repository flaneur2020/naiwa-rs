# naiwa-rs

`naiwa-rs` is a lightweight, Strata-inspired Rust inference engine for running large sparse MoE GGUF model Qwen3.8-Flash-Next (`IQ1_M`, ~60 GB) on commodity hardware (e.g., an RTX 4070 12 GB + 6-core CPU) at **~31.3 tok/s** (~31.96 ms TPOT) via GPU expert caching, async PCIe staging, a dedicated CPU MoE pool, streaming prefill, and built-in Multi-Token Prediction (MTP).

To run it serving OpenAI and Anthropic-compatible HTTP endpoints (`/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/health`, `/metrics`):

```bash
cargo build --offline --release --features cuda --workspace
./harness/serve/serve-tuned-mtp2.sh
```

Run `cargo test --offline --release --features cuda --workspace` and `./harness/oracle/capture_oracle.sh` to verify builds; see [`docs/design.md`](docs/design.md) and [`harness/README.md`](harness/README.md) for architecture and benchmarking details. MIT licensed.
