# qwen-fn-rs — benchmark plan and roadmap

Status as of 2026-10-09, `main` with fork `0c2c9ad` (§4b); first measurements 2026-10-08 at `7895b94`. First measurements on the ubi box
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
      -f 'Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf;Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf' \
      --cpu-moe \
      [--prefill-stream] [--expert-cache-gb N] [--pa-context-len 8192] \
      [--max-num-batched-tokens 8192] [--max-prefill-chunk-tokens 2048] \
      --prompt-len 512 --gen-len 128 --depth 4 --iterations 3 --warmup 1

Defaults: `--prompt-len 512` (TTFT), `--gen-len 128`, `--depth 4` (decode TPOT),
3 iterations + 1 warmup. Use **release** builds: `~/rel-bench.sh NAME [bench flags]` runs
`target/release/qwen-fn-rs` (or `$BIN`), samples RAM/VRAM every second, and writes
`~/bench-rel/{bench,mem}-NAME.log`. The older `~/one-bench.sh` builds and runs the *debug*
binary with a fixed workload; its numbers (R1–R4) are not comparable.

Report the bench table's TTFT, not the engine logger's `prefill` rate: the logger divides by
the forward time only and overstated R3 by ~1.7×.

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

**Caveats on R1–R4 (found 2026-10-09):** all four ran the *debug* binary
(`~/one-bench.sh` builds `target/debug`) with `--prompt-len 2048 --gen-len 16
--iterations 1 --warmup 0`, not the §3 settings. R3's 409.6 is the engine logger's
`prefill` rate; the bench's own TTFT for that run (`bench-ps2k.log`) was 234.8 tok/s.

## 4b. Release-build results and the prefill fixes (2026-10-09)

Release build (`cargo build --release --features cuda`), `~/rel-bench.sh` (same RAM/VRAM
sampling, flags passed through), 2 iterations + 1 warmup. Logs in `~/bench-rel/`.
Hardware floors measured on ubi: pinned H2D **13.35 GB/s** (PCIe **Gen3** x16 — the
Ryzen 5 3600X board, not Gen4), pageable 5.45 GB/s. Routed experts are **25.2 GiB**, PLE
table **26.8 GiB** IQ4_NL, everything else ~4.2 GiB. So one streamed prefill chunk cannot
beat 25.2 GiB / 13.35 GB/s ≈ **2.03 s**, whatever its length.

| # | Binary | Flags (all `--cpu-moe`) | Prompt | TTFT | Decode | Notes |
|---|---|---|---:|---:|---:|---|
| S1 | release | `--prefill-stream`, chunk 2048, batched 8192, ctx 8192 | 2048 | 9.84 s (208 tok/s) | 5.9 tok/s | logger says "prefill 410" — see above |
| S2 | + PLE fix | same | 2048 | **2.08–2.58 s (794–982 tok/s)** | 7.1 tok/s | at the PCIe floor |
| S3 | + PLE fix | same | 1024 / 2100 / 4096 | 2.05 / **5.37** / 4.36 s | | 2100 = 2048 GPU + 52-token CPU tail |
| C1 | + PLE fix | no stream | 64 / 256 | 3.83 / 13.42 s (17–19 tok/s) | 4.8 tok/s | release CPU prefill rate |
| S4 | + min-tokens | S1 + `--prefill-stream-min-tokens 64` | 32 / 64 / 256 / 512 | 2.03 / 2.04 / **2.08 / 2.09 s** | 6.5 tok/s | 256: 13.4 → 2.1 s |
| S5 | + min-tokens | S4 with chunk 8192 | 2100 / 4096 / 8000 | **2.16 / 2.94 / 5.55 s** | | 974 / 1394 / 1441 tok/s; peak VRAM 11.7 GB |
| K1 | any | `--expert-cache-gb 2 --pa-context-len 4096` (± stream) | | | | **NaN logits on the first decode** (B5, fixed in §4c) |

Where the time went (nsys, S1, one 2048-token prefill, 9.2 s): H2D 2.06 s (25 GiB at
full PCIe speed), kernels 1.13 s, and **one 7.06 s window with the GPU idle** before
layer 0. That window was the host PLE gather (`PleTable::gather_host`, GGUF branch): 16
hashed rows per token, read one at a time from a 27 GiB mmap that cannot sit in page
cache (pinned experts hold 25 GiB of 30 GB RAM). `/proc/<pid>` showed ~15k major faults
and 1.5–2 GB read per prompt — 128 KiB of readahead per ~1 KiB row, at queue depth 1.

Fixes:

1. **PLE host gather** (fork, `mistralrs-core/src/vision_models/qwen4_exp/ple.rs`;
   pushed to `patch/strata` as `0c2c9ad`): gather rows with rayon
   so the NVMe sees queue depth, and `madvise(MADV_RANDOM)` the table when it stays on the
   host. The GPU-idle window drops from 7.06 s to 0.48 s. Greedy outputs are
   byte-identical to the baseline for a 15-token (CPU) and a 2100-token (stream) prompt.
2. **`--prefill-stream-min-tokens`** (this repo; default stays 1024): a streamed chunk
   costs ~2.05 s flat, CPU prefill runs at ~18 tok/s, so break-even is ~37 tokens on
   ubi. 1024 (Strata's value) leaves every 40–1023-token prompt on the CPU at up to
   ~55 s. Use ~64 here.
3. **Config, no code:** set `--max-prefill-chunk-tokens` ≥ the prompt (4096–8192), or
   every chunk re-streams all 25 GiB and a sub-threshold tail falls back to the CPU.
   8192 peaks at 11.7 of 12 GB VRAM; 4096 is the safer default.

Recommended prefill config on ubi:

    --cpu-moe --prefill-stream --prefill-stream-min-tokens 64 \
      --max-num-batched-tokens 8192 --max-prefill-chunk-tokens 4096 --pa-context-len 8192

## 4c. Expert-cache NaN: bisection and fix (2026-10-09)

Repro: `--cpu-moe --expert-cache-gb 2 --pa-context-len 4096 --prompt-len 64 --gen-len 64
--iterations 3 --warmup 1` — NaN in every run. Debug switches (env vars in a throwaway build):

| Variant | NaN | Decode |
|---|---|---:|
| as shipped | yes | — |
| hits forced to the CPU | no | 6.6 tok/s |
| lookahead off | no | 7.2 tok/s |
| lookahead admits empty slots only (`MAX_LOOKAHEAD_SWAPS = 0`) | no | 7.6 tok/s |
| prefill fills off | yes | — |
| device sync after each step / before each upload | yes | — |
| hits enqueued on the calling thread | **no** | **8.4 tok/s** |

Per-hit checks while it ran: GPU hit outputs matched the CPU reference within 5e-2 (Q8_1
activations); the slot bytes equalled the host stack's bytes even after the NaN; and the
first non-finite value appeared in a layer's *input*, after a step where 9 hits had summed
to exactly 0. So the slots and kernels were fine, and stream ordering was not the problem
(syncs did not help). What mattered was launching the hits from the scoped thread while
the calling thread did its own CUDA work (`xs.to_device(Cpu)` for the misses), with the
lookahead's swap uploads adding a third thread.

Fix (`decode_forward`): the calling thread does every CUDA call. It copies the miss inputs
to the host, spawns the CPU misses on a scoped thread, and then only enqueues the hits'
kernels, which still overlap the CPU work.

| Run | Prompt | TTFT | Decode | Peak VRAM |
|---|---:|---:|---:|---:|
| cache 2 GB, no stream (×2) | 64 | 4.0–4.2 s | **9.0 / 9.7 tok/s** | 8.6 GB |
| cache 2 GB + stream, min-tokens 64, chunk 4096 | 512 / 2048 | 2.13 / 2.39 s | **9.3 tok/s** | 11.07 GB |
| no cache (S4, for reference) | | | 6.5–7.1 tok/s | |

Parity via `serve` (`~/bench-rel/scripts/parity.sh`, chunk 4096, ctx 4096): 128 greedy
tokens byte-identical with and without `--expert-cache-gb 2`, for the 15-token and the
2100-token prompt.

## 4d. Decode: profile, fixes and residency-policy sweep (2026-10-09)

All runs: `--cpu-moe --pa-context-len 4096 --gen-len 128 --depth 4 --iterations 2 --warmup 1`,
release build. Run-to-run noise is about ±1 tok/s and a few points of hit rate (the lookahead's
timing is not deterministic), so repeated runs are given as ranges.

**Where the time went before the fixes** (nsys + main-thread timers):

- Uncached: the GPU is busy 6% of a token. The CPU routed experts take ~108 of 138 ms; they read
  10 experts × 48 layers × 2.33 MB = 1.12 GB per token at ~10 GB/s.
- Unpinned (`--cpu-moe` without `--prefill-stream` or a cache): the expert stacks swap out
  under the PLE table's page-cache pressure (8.3k pages/s swapped in, 490 MB of the process
  in swap): 4.8–5.4 tok/s vs 6.4–7.1 pinned.
- ubi's memory reads at **~17.9 GB/s** regardless of thread count (`scripts/membw.c`), so the
  CPU-expert floor is ~62 ms/token uncached.
- With a 4 GB cache (78% hits): 6,200 kernel launches per token (vs 2,050 uncached), ~11 per
  hit; launch API time ~28 ms/token, async alloc/free ~8 ms; GPU busy 30%.
- After the fused banks, main-thread time per token at 4 GB: waiting for the CPU misses 35–54
  ms, everything between MoE calls 24–45 ms, enqueueing the hits 3–4 ms, top-k readback 0.5 ms.
  The miss wait was out of proportion to ~1.7 misses per layer.

**Fixes:**

1. *Fused hit decode per slot bank* (`1231276`): kernels per token 6,200 → 2,389 at 4 GB.
2. *CPU GEMV row split* (fork `6f740bc`): the routed GEMV parallelized over (token, expert)
   only, so 2 misses used 2 of 12 threads. Microbenchmark (`~/bench-rel/gemvbench`, 1–3
   experts per call): Q4_0 down 4.1 → 11.5 GB/s, Q3_K/Q2_K gate/up 7–9 → 11–14 GB/s. At 10
   experts it changes little (12.5–15 GB/s either way, near the DRAM ceiling).

| Config | Before | Fused banks | + row-split GEMV |
|---|---:|---:|---:|
| no cache (pinned via `--prefill-stream`) | 7.1 | — | **8.2** |
| `--expert-cache-gb 2` | 9.3 | 10.2 | **11.4–11.7** (one run 12.3) |
| `--expert-cache-gb 3` | 11.0 | | |
| `--expert-cache-gb 4` | 12.4 | 13.0 | **13.8–14.7** |
| `--expert-cache-gb 5`/`6` | OOM at startup | | |

Peak VRAM: 8.6 GB at 2 GB, 11.0 GB at 4 GB. Parity: 128 greedy tokens byte-identical to
the uncached run at 2 GB (short and 2100-token prompts, with `--prefill-stream`, so two banks
and a loan cycle). `serve` cannot start with a 3–4 GB cache even without the stream (`KV
cache requires 118 MB but only 0 MB is safely available`), while `bench` runs 4 GB with
965 MiB free; this predates the fused banks (same failure on the previous binary).

**Residency policy** (2 GB, 21 slots/layer; temporary env knobs, reverted):

| Policy | Hit rate | Decode | Swaps |
|---|---:|---:|---:|
| current: top-10 lookahead, LRU victim, ≤ 2 swaps/layer/token | 63–69% | 10.3–13.1 | ~20k |
| ≤ 1 swap | 52–55% (one run 63%) | 9.6–11.4 | ~11k |
| ≤ 3 / ≤ 4 swaps | 63–68% | 11.1–11.6 | 30–39k |
| no swaps (residency frozen after prefill) | 11.9% | 6.8 | 0 |
| Strata-style decayed LFU rounds only (every 4 tokens, 1 swap, ×0.7, ≥ 2.0, +1.5) | 46.6% | 10.0 | 3k |
| LFU rounds every token, ×0.9 | 54.0% | 10.6 | 6k |
| lookahead, victim must be LFU-colder than the prediction | 37.6% | 8.2 | 1k |
| LFU-gated lookahead + LFU rounds | 57.3% | 10.5 | 11k |
| predict top-12/14/16 with 3–4 swaps | 66.9–67.8% | 10.4–11.7 | 33–46k |

Miss breakdown with the current policy at 2 GB: 53% of misses were not in the prediction, 37%
were predicted but not resident when the layer ran, 10% had no prediction for that token yet.
Prediction precision (routed experts that were in the top-10 prediction): 69% at 2 GB, 66% at
4 GB.

**Strata comparison** (github.com/Niko1221/Strata `fb58e0d`; the local clone at
`~/code/strata` is an unfinished fetch with no files). Strata keeps residency by decayed
frequency: `usage[l][e] += 1` per routed token, a round every 4 verify windows swaps in
non-residents with usage ≥ 2.0 over residents colder by ≥ 1.5, at most 96 swaps globally
(sorted by gain), then `usage *= 0.7`. Its paper credits this for ~50% → 72%. Its next-layer
router lookahead matches ours, but by default only warms OS pages; copying predictions into a
separate VRAM pool lost 0–9% decode there. Here the opposite holds: single-token decode, with
predictions loaded into the main slots, gets most of its hit rate from the lookahead, and
frequency alone or as a gate does worse. Strata also sizes per-layer slot counts from a
global frequency ranking (≈1.5× spread), saves a residency profile for warm start, and
streams a share of the misses to the GPU (`--pcie-frac`); those are on the roadmap
(design §6).

## 5. Known issues

| # | Issue | Detail |
|---|---|---|
| B1 | `--expert-cache-gb 4` OOMs with prefill stream (design G9) | Only 5882 MiB free after the KV cache. The prefill stream borrows ~1380 MiB of cache slots for layer buffers, then `CUDA_ERROR_OUT_OF_MEMORY` on layer 1 prefetch. Setup log: `4.0 GB total, ~42 slots/layer (estimated 48 layers, 2112000 bytes/slot)`. Workaround: `--expert-cache-gb 2 --pa-context-len 4096` fits (then hits B5). |
| B2 | PLE table lives in host memory | The `per_layer_token_embd.weight` n-gram table is large and not cached on the GPU, so every token does a slow host gather. Visible in R1/R3. **Mitigated** by the parallel gather + `MADV_RANDOM` (§4b): 7.06 s → 0.48 s per 2048-token prompt; still disk-bound. |
| B5 | ~~`--expert-cache-gb` produces NaN logits~~ **fixed** (§4c, design G8) | `--expert-cache-gb 2 --pa-context-len 4096` fits now (B1 was the 4 GB size), but the first decode after warmup samples NaN, with or without `--prefill-stream`, on the stock release binary too. The unit test `decode_split_matches_running_every_expert_on_the_cpu` passes, so the real-model path differs from it. Logs: `~/bench-rel/bench-cache2-*.log`, `dbg-*.log`. |
| B6 | `serve` OOMs on a 2100-token prompt with the R3 flags (design G9) | `bench` uses `max_seqs = 1`; `serve`'s default reserves more, leaving too little VRAM for the prefill buffers. `--max-seqs 1` works. |
| B3 | Decode at ~7 tok/s (release, after the PLE fix; 3.7 in the debug R1) | Not bus-bound. Needs a per-step profile (design §6 item 4). |
| B4 | `--prefill-stream` chunk size defaults to 512 | Below the 1024-token streaming threshold (now `--prefill-stream-min-tokens`), so the GPU path silently does nothing unless `--max-prefill-chunk-tokens` is raised (fixed in code at `7895b94`; documented here so config is deliberate). |

## 6. Roadmap

Moved to `design.md` §6, the single roadmap. Items there cite the B-numbers and runs above.
Done since the first version of this plan: release-build measurements, chunk-size sweep and
threshold check (old items 12), the R4 retry at 2 GB (old item 1, which exposed B5), and the
prefill profile (§4b).

## 7. Commands cheat sheet (on ubi)

    # release build, then one bench sampling RAM/VRAM (logs in ~/bench-rel/)
    source ~/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH CUDA_COMPUTE_CAP=89
    cd ~/code/qflash-rs && cargo build --release --features cuda
    ~/rel-bench.sh stream --cpu-moe --prefill-stream --prefill-stream-min-tokens 64 \
        --max-num-batched-tokens 8192 --max-prefill-chunk-tokens 4096 --pa-context-len 8192 \
        --prompt-len 256,2048,4096 --gen-len 16 --depth 4 --iterations 2 --warmup 1
    ~/rel-bench.sh cache2 --cpu-moe --expert-cache-gb 2 --pa-context-len 4096 \
        --prompt-len 64 --gen-len 128 --depth 4 --iterations 3 --warmup 1   # was B5

    # nsys profile of the measured iterations only
    MISTRALRS_BENCH_CUDA_PROFILER_RANGE=1 nsys profile -o prof -t cuda,osrt \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        target/release/qwen-fn-rs bench ...

    # greedy-output parity between two binaries (short + 2100-token prompt, via serve)
    ~/bench-rel/scripts/parity.sh BIN TAG

    # the requant converter (already built; rerun only if the model changes)
    ~/requant/requant ~/models/qwen3-coder/*.gguf ~/models/qwen3-coder/requant/out.gguf --dry-run

    # pull latest on ubi
    cd ~/code/qflash-rs && git pull
