# qwen-fn-rs — benchmark plan and roadmap

Status as of 2026-10-08, `main` at `7895b94`. First measurements on the ubi box
(RTX 4070 12 GB VRAM, 30 GB RAM) with a re-quantized Qwen3.8-Flash-Next Coder GGUF.

## 1. Hardware and model

| | |
|---|---|
| Host | ubi — Ubuntu, Tailscale. Reachable as `ubi` over ssh. |
| GPU | NVIDIA RTX 4070, 12 GB VRAM |
| RAM | 30 GB usable (32 GiB physical) |
| CPU | x86_64 (used for `--cpu-moe` routed experts) |
| Model | Qwen3.8-Flash-Next **Coder**, 256 of 512 experts routed per layer |
| Source GGUF | ISTA-DASLab GGUF, IQ1_M, ~58 GB across 2 shards (shard 1 sha256 verified) |
| Working set | `~/models/qwen3-coder/requant/` — re-quantized to candle-runnable K-quants |

The original IQ1_M GGUF stores routed experts in IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL /
IQ4_XS / Q2_0 — none of which candle can run. `~/requant/requant.cpp` rewrites the
file in place using upstream ggml's own decode/re-encode (see §2).

## 2. The requant conversion

Tool: `~/requant/requant.cpp`, built against `~/requant/llama.cpp/ggml`
(commit `9c2e0e4`, sparse checkout). Streams tensor by tensor, so memory stays at
one tensor regardless of file size. Reports per-tensor relative RMSE

    sqrt(sum((decode(x) - quantize(decode(x)))^2) / sum(decode(x)^2).

Type mapping (`target_for` in `requant.cpp`):

| Source type | Target | Condition |
|---|---|---|
| IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, Q2_0, TQ1_0, TQ2_0 | Q2_K | `ne0 % 256 == 0` |
| IQ3_XXS | Q3_K (default; `--iq3xxs q2_k` to pick Q2_K) | `ne0 % 256 == 0` |
| IQ3_S | Q3_K | `ne0 % 256 == 0` |
| IQ4_NL, IQ4_XS, MXFP4, ... | Q4_K | `ne0 % 256 == 0` |
| any of the above | Q4_0 | `ne0 % 256 != 0` (block-32 rows that K-quants cannot pack) |

Result: shard 1 grew 29.6 → 31.5 GB. The PLE n-gram table
(`per_layer_token_embd.weight`) is left in its original IQ4_NL — the fork reads it
directly, so it never passes through candle.

Usage:

    ~/requant/requant IN.gguf OUT.gguf            # default, --iq3xxs q3_k
    ~/requant/requant IN.gguf OUT.gguf --dry-run  # report sizes and counts only

## 3. Bench methodology

All runs use `qwen-fn-rs bench` with the re-quantized Coder model:

    qwen-fn-rs bench \
      -m ~/models/qwen3-coder/requant \
      -f 'qwen3-coder-requant-00001-of-00002.gguf;qwen3-coder-requant-00002-of-00002.gguf' \
      --cpu-moe \
      [--prefill-stream] [--expert-cache-gb N] [--pa-context-len 8192] \
      [--max-num-batched-tokens 8192] [--max-prefill-chunk-tokens 2048] \
      --prompt-len 512 --gen-len 128 --depth 4 --iterations 3 --warmup 1

Defaults: `--prompt-len 512` (TTFT), `--gen-len 128`, `--depth 4` (decode TPOT),
3 iterations + 1 warmup. The helper `~/one-bench.sh NAME [extra flags]` rebuilds
with `--features cuda` and samples RAM/VRAM every 2 s while the bench runs.

Logged metrics: TTFT (tok/s), decode TPOT (tok/s), peak RAM, peak VRAM, min free RAM.

## 4. Results so far

| # | Flags | Prefill (tok/s) | Decode (tok/s) | Peak RAM | Peak VRAM | Notes |
|---|---|---:|---:|---:|---:|---|
| R1 | `--cpu-moe` | 12.2 | 3.7 | 25.2 GB | 11.4 GB | baseline; experts fully on CPU |
| R2 | `--cpu-moe --prefill-stream --max-prefill-chunk-tokens 512` | 12 | 3.7 | — | — | prefill stream did **not** engage (chunk < 1024 `MIN_TOKENS`) |
| R3 | `--cpu-moe --prefill-stream --max-num-batched-tokens 8192 --max-prefill-chunk-tokens 2048 --pa-context-len 8192` | **409.6** | 4.8 | 28.6 GB (min avail 2.4 GB) | 9.0 GB | prefill on GPU, 34x over R1 |
| R4 | `--cpu-moe --prefill-stream --expert-cache-gb 4 --max-prefill-chunk-tokens 2048` | — | — | — | — | **CUDA OOM** (see §5) |

Key takeaway: getting prefill onto the GPU (`R3`) is the big win — 12 → 410 tok/s.
Decode at 3.7–4.8 tok/s is the current ceiling and is the next thing to attack.

## 5. Known issues

| # | Issue | Detail |
|---|---|---|
| B1 | `--expert-cache-gb 4` OOMs with prefill stream | Only 5882 MiB free after the KV cache. The prefill stream borrows ~1380 MiB of cache slots for layer buffers, then `CUDA_ERROR_OUT_OF_MEMORY` on layer 1 prefetch. Setup log: `4.0 GB total, ~42 slots/layer (estimated 48 layers, 2112000 bytes/slot)`. **Not yet fixed** — needs a smaller cache, a smaller KV cache (`--pa-context-len`), or both. |
| B2 | PLE table lives in host memory | The `per_layer_token_embd.weight` n-gram table is large and not cached on the GPU, so every token does a slow host gather. Visible in R1/R3. |
| B3 | Decode at 3.7 tok/s is ~10x below the CPU↔host bandwidth ceiling | Suggests the bottleneck is not bus bandwidth but per-token host compute / launch overhead. Needs profiling (see roadmap §6.3). |
| B4 | `--prefill-stream` chunk size defaults to 512 | Below the `MIN_TOKENS = 1024` threshold, so the GPU path silently does nothing unless `--max-prefill-chunk-tokens` is raised (fixed in code at `7895b94`; documented here so config is deliberate). |

## 6. Roadmap

Ordered by priority. Each item is a single, testable change.

### 6.1 Make `--expert-cache-gb` fit (unblock GPU decode caching)

1. **Re-run R4 with a smaller cache + smaller KV:** try
   `--expert-cache-gb 2 --pa-context-len 4096` first, then bisect up. Goal: a
   working decode-with-cache data point, even if the cache is small.
2. **Measure hit rate** once it runs: add the per-layer hit/miss counters
   (design.md G6) or at least log `[layer l] hit X / miss Y / lookahead Z`.
3. **Tune `MAX_LOOKAHEAD_SWAPS`** and the per-layer slot split from the hit-rate
   data.

### 6.2 Decode speed (the real bottleneck)

4. **Profile one decode step with Nsight Systems** on a `--cpu-moe` run: is the
   time in host GEMV, in the H2D/D2H activation copy, or in CUDA launch overhead?
   This decides whether to optimize the CPU kernel, batch misses, or cut launches.
5. **Batch CPU misses:** when several tokens in a step miss the same expert (e.g.
   MTP verify windows, batched decode), run them together. Today `num_tokens > 1`
   is treated as prefill (design.md G4) and misses are not batched.
6. **CPU prefill grouped-by-expert (gemm) fallback** for when the GPU prefill
   stream is unavailable or OOMs: gather activations per routed expert and run a
   single f32 GEMM per expert, instead of per-token GEMV. This is the path
   mistral.rs's fallback took before the fork's `routed_gemv`; revisiting it as a
   deliberate fallback gives a CPU prefill floor that does not need VRAM.

### 6.3 Correctness and observability (no GPU strictly needed, but verify on GPU)

7. **Parity: cached vs uncached.** Same prompt, logits within ~1e-3 relative.
   CPU and CUDA kernels differ (CUDA quantizes activations to Q8_1), so never
   assert bit equality.
8. **In-place fill fence protocol** under load: watch for parity drift that grows
   with decode length — that would mean a slot was overwritten mid-read.
9. **Hit-rate / throughput counters** reported by `bench` (design.md G6).

### 6.4 Prefill stream polish

10. **Hide the first layer of each chunk** (prefetch layer 0 at the end of the
    previous chunk).
11. **Separate copy stream for prefill** if the profile shows layer uploads
    queueing behind decode slot fills (they share one copy stream today).
12. **Verify `MAX_TOKENS = 1024` threshold** suits this hardware; measure tokens/s
    at chunk sizes 1024 / 2048 / 4096 / 8192.

## 7. Commands cheat sheet (on ubi)

    # rebuild + run one bench, sampling RAM/VRAM
    ~/one-bench.sh cpumoe --cpu-moe
    ~/one-bench.sh stream --cpu-moe --prefill-stream \
        --max-num-batched-tokens 8192 --max-prefill-chunk-tokens 2048 \
        --pa-context-len 8192
    ~/one-bench.sh cache2 --cpu-moe --prefill-stream --expert-cache-gb 2 \
        --pa-context-len 4096 --max-prefill-chunk-tokens 2048

    # the requant converter (already built; rerun only if the model changes)
    ~/requant/requant ~/models/qwen3-coder/*.gguf ~/models/qwen3-coder/requant/out.gguf --dry-run

    # pull latest on ubi
    cd ~/code/qflash-rs && git pull
