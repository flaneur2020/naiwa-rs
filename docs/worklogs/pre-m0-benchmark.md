<!-- Salvaged 2026-11-01 from the pre-git snapshot that was sitting in ~/naiwa-validate/.plans/.
     It is the only record of the week before this repository's history starts (2026-10-16), so it is
     kept verbatim: numbers, paths and flags are the ones that were in use at the time
     (`~/models`, `~/bench-rel/mtp`, the `mistral.rs-fork` checkout) and are NOT updated. -->

# naiwa-rs — worklog: Qwen3.8-Flash-Next performance

A log, oldest first: one entry per piece of work, each stating the **anchor** it moves (the metric
and its value *before*), the data after, and what it means. Numbers are `bench`, release build,
`--cpu-moe`, on the ubi box, unless the entry says otherwise. `serve` runs are called out because
W7 and W11 showed `bench`'s synthetic prompts misjudge MTP.

Anchors at a glance (decode is TPOT from the bench table, prompts 512 in / 128 out):

| # | Date | Work | Anchor | Before -> after |
|---|---|---|---|---|
| W1 | 08 | requant to K-quants | can it run at all | no -> yes |
| W2 | 08 | first measurements (R1-R4, debug build) | prefill / decode | 12.2 / 3.7 -> 409.6 (logger) / 4.8 |
| W3 | 09 | release build, PLE gather, stream min-tokens | TTFT, 2048-token prompt | 9.84 s -> 2.08 s; decode 5.9 -> 7.1 |
| W4 | 09 | expert-cache NaN bisect and fix | decode at 2 GB | NaN -> 9.0-9.7 |
| W5 | 09 | fused hit banks + row-split GEMV | decode at 4 GB | 12.4 -> 13.8-14.7 |
| W6 | 10 | overlap CPU experts with the launches | decode at 4 GB | 13.8-14.7 -> 15.0-17.6 |
| W7 | 11 | MTP head bindings and `--mtp` for GGUF | decode, no cache | 8.71 -> 9.52 (+9%) |
| W8 | 11 | native IQ quant support + the decode fixes | decode at 4 GB | 8.0 -> 13.7 -> 15.4 |
| W9 | 12 | large-batch prefill | TTFT at 4096 / 8000 tokens | 1361 / 1318 tok/s |
| W10 | 12 | what a second memory channel would buy | decode | 7.3 / 11.4 / 15.4 (projection +10%) |
| W11 | 12 | MTP np=2 measured via `serve` | decode, no cache / 2 GB | 7.50 -> 8.60 / 11.55 -> 11.69 |
| W12 | 12 | MTP x kernel: they compound | 2 GB decode projection | 11.6 -> 20.5 (kernels + DIMM + MTP) |
| W13 | 12 | **baseline: MTP np=2 + cache 2**; expert-slot census | decode | 11.69 tok/s (serve); MTP reads 2.94x slots/token, 28% dup |
| W14 | 12 | dedup the CPU expert slots by expert | MTP np=2 decode | 11.69 -> 11.39 (no gain; reverted) |
| W15 | 12 | grid dot via a stack array | grid-type GB/s | no signal (harness is +-15-20%); reverted |
| W16 | 12 | dedup the CPU gemv by (expert, activation) | MTP np=2 + cache 2 (serve) | 11.69 -> **11.92** (+3%, exact) |
| W17 | 12 | ceiling projection: memory-bound kernels x draft depth | MTP np=2 (serve) | 11.92 -> **28.5 / 34.1 / 38.6** tok/s (np 2/3/4, 2 DIMMs) |
| W18 | 12 | perf profile of the vec-dot kernels | cycles/byte, instr/byte | grid types issue-bound (IPC 3.0-3.9); gather loses (+43% cycles) |
| W19 | 12 | share of the CPU expert cost by dtype | % of CPU expert time | grid types 68.4%, IQ4_NL 24.2%, Q2_0 6.2% |
| W20 | 12 | `--pcie-frac` (pre-existing WIP) measured | MTP np=2 + cache 2 (serve) | 12.19 -> **13.52** (+11% at frac 0.5, saturates) |
| W21 | 12 | MTP np=4 vs np=2 at a 2 GB cache | decode, cache 2 (serve) | 12.04 -> 10.81 (np4 is **10% worse**; W17 scaling falsified) |
| W22 | 12 | second memory channel, revised | best config 13.55 tok/s | -> ~14.8-15.3 tok/s (+9-13%) |
| W23 | 12 | GPU vs CPU expert split across draft depths | per round, cache 2 | 73% CPU expert / 27% GPU+serial at both np=2 and np=4 |
| W24 | 12 | 4 GB cache: runs clean, hit rate 72-80%, MTP unmeasurable | decode / hit rate | np=0 12.5 tok/s, 10.7 GB peak; cache 2's 45.6% -> 72.2% |
| W25 | 12 | `--prompt-file` (real text in bench), cache-4 np sweep | decode, cache 4 vs 2 | 2 GB np=4 loses 10%; **4 GB np=4 = np=2 = 17.4 tok/s** |
| W26 | 12 | `--pcie-frac` at a 4 GB cache | decode, cache 4 | 1 scratch slot/layer +3.3%; 3 slots OOM; **18.9-19.4 tok/s** |
| W27 | 12 | B7 resolved: 1887 MB = recurrent prefix snapshots; **serve now runs at cache 4** | `serve`, cache 4 vs 2 | 12.50 -> **16.58 tok/s** (+33%); np=4 -8% |
| W28 | 12 | `--pcie-frac` under serve at a 4 GB cache | `serve`, cache 4, frac sweep | 15.97 -> **16.64 tok/s** (+4%, 2 slots); 3 slots OOM |
| W29 | 12 | **second DIMM installed** (15.8 -> 22/32 GB/s) | decode, bench + serve | 19.4 -> **21.6** (bench) / 16.6 -> **18.9** (serve); only +9-16% |

## Setup

### Hardware and model

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
file in place using upstream ggml's own decode/re-encode (see *The requant conversion*).

### The requant conversion (W1)

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

### Methodology

All runs use `naiwa-rs bench` with the re-quantized Coder model:

    naiwa-rs bench \
      -m ~/models/qwen3-coder/requant \
      -f 'Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf;Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf' \
      --cpu-moe \
      [--prefill-stream] [--expert-cache-gb N] [--pa-context-len 8192] \
      [--max-num-batched-tokens 8192] [--max-prefill-chunk-tokens 2048] \
      --prompt-len 512 --gen-len 128 --depth 4 --iterations 3 --warmup 1

Defaults: `--prompt-len 512` (TTFT), `--gen-len 128`, `--depth 4` (decode TPOT),
3 iterations + 1 warmup. Use **release** builds: `~/rel-bench.sh NAME [bench flags]` runs
`target/release/naiwa-rs` (or `$BIN`), samples RAM/VRAM every second, and writes
`~/bench-rel/{bench,mem}-NAME.log`. The older `~/one-bench.sh` builds and runs the *debug*
binary with a fixed workload; its numbers (R1–R4) are not comparable.

Report the bench table's TTFT, not the engine logger's `prefill` rate: the logger divides by
the forward time only and overstated R3 by ~1.7×.

Logged metrics: TTFT (tok/s), decode TPOT (tok/s), peak RAM, peak VRAM, min free RAM.

## Worklog

### W1 2026-10-08 — Requant: make the model runnable at all

**Anchor:** "can it run" — before this there is nothing to measure.

**Change:** `~/requant/requant.cpp` rewrites the published IQ1_M GGUF into candle-runnable K-quants
(see Setup), which is what every measurement up to W8 used. W8 removes the need for it.

**Read:** the conversion was the price of entry; its cost is a second 60 GB artifact and the RMSE of
the re-encode, both of which W8 makes unnecessary.

### W2 2026-10-08 — First measurements (debug build)

**Anchor:** prefill and decode of the baseline `--cpu-moe` run

**Change:** none — this is the baseline

#### R1-R4

| # | Flags | Prefill (tok/s) | Decode (tok/s) | Peak RAM | Peak VRAM | Notes |
|---|---|---:|---:|---:|---:|---|
| R1 | `--cpu-moe` | 12.2 | 3.7 | 25.2 GB | 11.4 GB | baseline; experts fully on CPU |
| R2 | `--cpu-moe --prefill-stream --max-prefill-chunk-tokens 512` | 12 | 3.7 | — | — | prefill stream did **not** engage (chunk < 1024 `MIN_TOKENS`) |
| R3 | `--cpu-moe --prefill-stream --max-num-batched-tokens 8192 --max-prefill-chunk-tokens 2048 --pa-context-len 8192` | **409.6** | 4.8 | 28.6 GB (min avail 2.4 GB) | 9.0 GB | prefill on GPU, 34x over R1 |
| R4 | `--cpu-moe --prefill-stream --expert-cache-gb 4 --max-prefill-chunk-tokens 2048` | — | — | — | — | **CUDA OOM** (see Known issues) |

Key takeaway: getting prefill onto the GPU (`R3`) is the big win — 12 → 410 tok/s.
Decode at 3.7–4.8 tok/s is the current ceiling and is the next thing to attack.

**Caveats on R1–R4 (found 2026-10-09):** all four ran the *debug* binary
(`~/one-bench.sh` builds `target/debug`) with `--prompt-len 2048 --gen-len 16
--iterations 1 --warmup 0`, not the Methodology settings. R3's 409.6 is the engine logger's
`prefill` rate; the bench's own TTFT for that run (`bench-ps2k.log`) was 234.8 tok/s.

**Read:** prefill moves to the GPU with `--prefill-stream` (12 -> 410 tok/s on the engine logger); decode is the next target. All four ran the *debug* binary and are not comparable with later numbers.

### W3 2026-10-09 — Release build, PLE host gather, stream threshold

**Anchor:** TTFT for a 2048-token prompt

**Change:** release build (not debug); PLE gather parallelised with rayon + `MADV_RANDOM`; `--prefill-stream-min-tokens`

#### S1-S5, C1, K1

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
| K1 | any | `--expert-cache-gb 2 --pa-context-len 4096` (± stream) | | | | **NaN logits on the first decode** (B5, fixed in W4) |

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

**Read:** the PLE host gather was a 7.06 s GPU-idle window before layer 0; parallelising it puts TTFT at the PCIe floor.

### W4 2026-10-09 — Expert-cache NaN: bisection and fix

**Anchor:** decode at `--expert-cache-gb 2` (was NaN)

**Change:** the calling thread now does every CUDA call; hits are only enqueued while the misses run on a scoped thread

#### Bisection

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

**Read:** the cache is usable again, and decode at 2 GB beats the uncached path.

### W5 2026-10-09 — Decode profile: fused hit banks + row-split GEMV

**Anchor:** decode at `--expert-cache-gb 4`

**Change:** fused per-slot-bank hit decode (`1231276`), CPU GEMV row split (fork `6f740bc`)

#### Profile and sweep

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
(design Roadmap).

**Read:** the CPU expert GEMV was running with most threads idle when only a few experts were routed to it.

### W6 2026-10-10 — Overlap the CPU experts with the next layer's launches

**Anchor:** decode, uncached and at 2/4 GB

**Change:** `src/deferred.rs`: misses run on a persistent worker, a stream host function waits for them while the host launches the next layer

#### Overlap

`src/deferred.rs`: the decode misses (and uncached decode's experts) run on a persistent
worker; candle's stream gets a host function that waits for it, then a copy from pinned
staging, and the MoE block returns at once, so the host launches the next layer while the CPU
computes. Same workload as W5 (`--gen-len 128 --depth 4`, 2 iterations + warmup):

| Config | W5 | Overlap | Runs |
|---|---:|---:|---|
| no cache (pinned via `--prefill-stream`) | 8.2 | **8.7** | 8.7, 8.7 |
| `--expert-cache-gb 2` | ~11.5 | **12.9–13.2** | 12.9, 13.2, 13.0 |
| `--expert-cache-gb 4` | 13.8–14.7 | **15.0–17.6** | 17.6, 15.0, 16.3 (hit rate 80 / 72.5 / 77%) |

Predicted beforehand: 4 GB 17–19, 2 GB 14–15, uncached ~10. The 4 GB result is at the low end
and its spread follows the hit rate; the uncached gain is smaller than predicted. Parity: 128
greedy tokens byte-identical to the reference, uncached and with `--expert-cache-gb 2
--prefill-stream`, for the 15-token and 2100-token prompts.

**Read:** launch enqueue is no longer on the critical path — with this in place the remaining per-token cost is expert reads plus a serial per-layer chain (see W10).

### W7 2026-10-11 — MTP speculative decoding: head bindings and `--mtp` for GGUF

**Anchor:** decode, no cache, vs no MTP

**Change:** `repack.py` turns `mtp-Q4_0.gguf` into an mmproj component; five fork files bind the head (fork `patch/mtp-gguf`)

#### MTP

The published `ggml-org/Qwen3.8-Flash-Next-GGUF` repo also carries the checkpoint's MTP draft block
as `mtp-Q4_0.gguf`, which llama.cpp's `--mtp` export writes with the head's tensors under the
*upstream* `blk.48.*` names (block 48 = one past the 48 decoder blocks). That file is a complete
GGUF, so it cannot be a third `-f` shard (its `general.type` is `model` and it has no split
metadata). Instead `~/bench-rel/mtp/repack.py` rewrites it into a component the engine's
`merge_components(mmproj)` accepts:

* `general.type` becomes `mmproj` (what `validate_component_type` requires);
* `general.*` keys are dropped by the merge, and the expert numbers are moved to
  `qwen4exp.mtp.expert_count` / `.expert_used_count` — the unsuffixed keys already exist in the
  main archive with different values (256 vs the head's 512) and a conflicting value makes
  `merge_components` bail;
* `token_embd.weight` and `output.weight` are dropped (they duplicate the main model's);
* **every tensor name is kept verbatim** — the head binds from the `blk.<block>.*` names, so no
  rename is needed and the component is a pure rewrite with no byte surgery. `repack.py` verifies
  all offsets land exactly and the file has zero slack.

Model side (fork `patch/mtp-gguf`, five files, ~177 lines):

1. `mtp_num_experts: Option<usize>` on `TextConfig` (with `mtp_experts()`), so the head can route
   its own 512 experts while the pruned target keeps 256. `SparseMoeBlock::new` takes it when
   `site.is_mtp`.
2. `bind_mtp_head` in `gguf/qwen4exp.rs`: binds the head's `blk.<block>.*` tensors to the native
   `mtp.layers.0.*` names (reusing `LAYER_TENSORS`/`SHIFTED_NORMS`, so the layouts match the decoder
   block exactly), plus `nextn.hc_head_*` → `mtp.hyper_connection_mixer.*` and the two `nextn.*norm`
   → `mtp.pre_fc_norm_*`.
3. `mtp.fc_embedding` / `mtp.fc_hidden` are `Slice` bindings over the fused `blk.<block>.nextn.eh_proj.weight`
   `[5120, 2560]`. Which half is which was settled against the BF16 checkpoint, not guessed: the
   first half of the fused axis is `fc_embedding` (cosine 0.996 over sampled rows) and the second is
   `fc_hidden` (0.998).
4. **All nine of the head's norms are stored shifted** (`1 + w`): every one of them is exactly
   `checkpoint + 1.0`, so `pre_fc_norm_*` uses `bind_shifted_norm` like the rest. (An independent
   check agrees: the checkpoint's `layers.3.self_attn.q_norm.weight` has mean 0.2833 = the GGUF's
   1.2833 − 1, confirming the checkpoint stores true weights and llama.cpp adds the 1.)
5. The `Built-in MTP is not supported for GGUF or GGML models` bail is removed for `ModelSelected::GGUF`
   and the flag is threaded through a new `GGUFLoaderBuilder::with_mtp` into
   `MultimodalLoaderBuilder`.

Measured on ubi (release, `--cpu-moe`, 4096 ctx, greedy drafting):

| Run | Greedy output | Decode |
|---|---|---:|
| `serve`, 15-token and 2100-token prompts, no MTP | reference | 8.63–8.76 tok/s |
| same, `--mmproj mtp-component.gguf --mtp` | **byte-identical to the reference** | 9.62 / **10.98** tok/s |

**MTP acceptance on real text: mean 89.1%** over 7 logger windows (range 66.7–100%, mean drafted
length up to 7.00, depth up to 6.0) — the same range Strata reports for this head (~85–89%), which
is the strongest evidence the bindings are right.

Do **not** judge the head from `bench`: its prompts are synthetic pseudo-random tokens
(`BENCH_TOKEN_BASE + idx % SPAN`), which are near-unpredictable, so acceptance there reads anywhere
from 3% to 100% and is dominated by whether the draft happens to emit numeric-looking text. A short
`serve` run with one real prompt gives a stable number; `~/bench-rel/serve-r2-*.log` holds the runs
above, and greedy output parity between two `serve` runs is the correctness check (`scripts/parity.sh`).

#### MTP benchmark matrix (2026-10-11, `2da8b11` + expert-cache fix)

`serve` with `--cpu-moe --prefill-stream --prefill-stream-min-tokens 64 --pa-context-len 4096
--max-seqs 1`, greedy, 256 new tokens, three real prompts (a Fibonacci stub, the 2100-token helper
file, an `LRUCache` docstring), two requests each; decode tok/s from the response `usage`
(`~/bench-rel/mtpbench/`, script `/tmp/mtpbench.sh`).

| Config | Decode (mean of 6) | Range | MTP acceptance |
|---|---:|---:|---:|
| no cache | 8.71 | 8.60–8.75 | — |
| no cache + MTP (n_predict 6, auto) | **9.52** (+9%) | 8.88–10.35 | 89.3% |
| no cache + MTP, n_predict 2 | 9.25 | 8.54–10.31 | 97.2% |
| `--expert-cache-gb 2` | **12.49** | 11.59–13.38 | — |
| cache 2 GB + MTP | 9.99 (−20%) | 8.43–10.70 | 91.1% |
| cache 2 GB + MTP, n_predict 2 | 9.48 | 8.91–10.02 | 92.5% |

MTP drafts well (89–97% accepted) but barely pays here, and costs throughput with the cache. Decode
is bound by reading routed-expert bytes (W5), and a verify step runs one forward over n+1 tokens
whose top-10 sets mostly differ, so a layer reads the union of their experts — several tokens' worth
of bytes per step instead of one. Speculation saves the per-step fixed costs (launches, attention,
the dense weights), not the expert reads. With the cache it is worse: the union blows past the
lookahead's prediction and the per-layer slots, so most of the extra experts are CPU misses. MTP
would pay where expert reads are cheap relative to per-step overhead (all experts in VRAM, or a
dense model), not on this box.

Greedy parity: the 2100-token prompt's 256 tokens are identical in every configuration. The two
short prompts diverge after ~150 tokens at near-tie choices (`fibonacci_formula` vs
`fibonacci_generator`; one blank line vs two), and the **no-MTP** 2 GB-cache run picks the same
alternatives at the same points, so these are numeric ties flipped by computing experts differently
(GPU hits, multi-token verify), not a draft that leaks into the output. The uncached run is
deterministic run to run.

Fix found by this run: the expert-cache provider returned `None` for the MTP site, which made the
builtin loader put all 512 of the head's experts (~1.35 GB) on the GPU; `--expert-cache-gb 2 --mtp`
then failed with `KV cache requires 128 MB but only 0 MB` and 1.5 GB OOMed in the first prefill.
MTP experts now run on the CPU under the cache provider too, as they already did without it (free
VRAM after startup 2520 → 3126 MiB at 1.5→2 GB).

**Read:** the head drafts well (89-97% accepted on real text) but only pays when expert reads are cheap; with the cache it loses.

### W8 2026-10-11 — Native IQ quant support, and the decode fixes it exposed

**Anchor:** decode at `--expert-cache-gb 4` (the published IQ1_M GGUF, no requant)

**Change:** fork `patch/iq4xs`: IQ4_XS/IQ4_NL/Q2_0/IQ3_XXS/IQ2_S/IQ3_S (CPU + CUDA), then `af172c75` (restore the row split an earlier IQ commit had reverted, plus a Q2_0 AVX2 dot)

#### Native IQ1_M at a 4 GB cache

The IQ quant work (W8) lets the **original published GGUF run directly**, so the requant step is
no longer needed to benchmark this model. Everything below is `bench`, release,
`--cpu-moe --expert-cache-gb 4 --pa-context-len 4096 --prompt-len 512 --gen-len 128 --depth 4
--iterations 2 --warmup 1` — **all at 4 GB**; script `~/iq-bench.sh` (same sampling as
`~/rel-bench.sh`), logs `~/bench-rel/{bench,mem}-{E3,F3,G,K,H,I2,I4}-*.log`, table also in
`~/bench-rel/RESULTS-iq1m-c4-mtp.md`.

| `--prefill-stream` | `--mtp` | Decode | TTFT | Peak RAM | Min avail | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| no | no | 15.4 ± 0.8 | 11.8 tok/s (43.4 s) | 28.1 GB | 2.9 GB | 10.9 GB |
| yes | no | **15.4 ± 1.7** | **253.1 tok/s (2.02 s)** | 28.0 GB | 3.0 GB | 11.0 GB |
| no | auto (6) | 14.0 ± 0.8 | 10.7 tok/s (47.8 s) | 28.8 GB | 2.2 GB | 11.2 GB |
| no | `-n-predict 2` | 12.8 ± 1.6 | 10.9 tok/s (46.8 s) | 29.1 GB | 1.9 GB | 11.1 GB |
| yes | auto (6) | 11.5 ± 0.9 | 158.7 tok/s (3.23 s) | 29.2 GB | 1.8 GB | 11.3 GB |
| yes | `-n-predict 2` | 13.8 ± 0.2 | 161.5 tok/s (3.17 s) | 29.3 GB | 1.7 GB | 11.1 GB |
| yes | `-n-predict 4` | 8.5 ± 2.2 | 157.4 tok/s (3.25 s) | 29.2 GB | 1.8 GB | 11.2 GB |

Best config is `--prefill-stream` without MTP: 15.4 tok/s decode and 253 tok/s TTFT. Against the
K-quant model at the same 4 GB (W5: 13.8–14.7) the native file is slightly ahead, uses less RAM
(28.0 vs 29.4 GB) and has a better prefill.

**MTP at 4 GB loses**, consistently with W7's "with the cache it is worse": 15.4 → 14.0 uncached,
and 11.5–13.8 with the stream, at every draft depth. The acceptance `bench` logs (48–81% at depth 2)
must not be read as a real number — its prompts are synthetic (W7) — but the throughput loss is
real: MTP costs headroom (peak VRAM 11.0 → 11.3 GB, **min free RAM 3.0 → 1.7 GB**), which is why
TTFT collapses with the stream (253 → 158–161 tok/s).

Two decode fixes were needed to get here (fork `af172c75`):

1. The IQ4_XS commit had accidentally reverted `6f740bcf`'s per-row rayon split in the routed
   `gemv`; a decode step routes only a few experts to the CPU, so all but that many threads idled.
2. Q2_0's vec-dot was scalar; it is now AVX2 (four 2-bit codes → signed `-1/0/1/2` → `maddubs`).
   Microbench (`bench_vec_dot_throughput`, 2560x640, 10 experts): 1.85 → 6.46 GB/s, 2490 → 713 us.

Decode at 4 GB therefore went 8.0 (before) → 13.7 (after the fixes, but with a stale build copy) →
**15.4** (after also syncing it — the copy's `mistralrs-core` was missing `bac32a67`, and syncing
that alone moved the plain baseline 13.7 → 15.4 tok/s).

**Read:** the published file now runs unmodified at 15.4 tok/s at 4 GB, at or above the re-quantized K-quant model, and the 4 GB numbers in the table above are pre-fix for the decode column.

### W9 2026-10-12 — Large-batch prefill

**Anchor:** TTFT for 1024-8000 token prompts

**Change:** chunk 4096 or 8192 with the expert stream; a 4 GB cache does not fit a large prefill

#### Prefill sweep

`--cpu-moe --prefill-stream --prefill-stream-min-tokens 64 --pa-context-len 8192 --gen-len 8
--depth 4 --iterations 2 --warmup 1`, release, prompts up to 8000 tokens:

| run | cache | chunk / batched | 1024 | 2048 | 4096 | 8000 | peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| P5 | 2 GB | 4096 / 4096 | 529.7 | 1023.6 | **1361.0** | 1318.2 tok/s | 11.7 GB |
| P6 | 1 GB | 8192 / 8192 | 528.2 | 983.5 | 1281.1 | 1280.5 tok/s | 11.5 GB |

(TTFT in tok/s; P5's 4096-token TTFT was 3.01 s and 8000's 6.07 s.) Prefill is ~1.3k tok/s past
4096 tokens, matching the K-quant S5 numbers (1394 / 1441) within run-to-run noise; the cache size
and the chunk size barely move it.

The 4 GB cache from the matrix above does **not** fit a large prefill: chunk 8192 + ctx 8192 peaks at
11.7 GB and OOMs, and so does chunk 8192 at a 2 GB cache. The stream borrows two layers' worth of
experts from the cache, so shrink the cache (1-2 GB) or the chunk (4096) for long prompts. 4096 is
the safer chunk: at 8000 tokens it re-streams the 25 GiB of experts twice, which costs one extra
~2 s PCIe pass (6.07 s vs 6.27 s TTFT, i.e. noise).

Remaining per-dtype CPU gaps (same microbench, 640x2560, 10 experts), for reference:
IQ2_S 9.3 / IQ3_XXS 9.5 / IQ3_S 10.1 GB/s versus Q2_K 18.5 / Q3_K 20.1; IQ4_NL and IQ4_XS are
already at 22.0 / 14.5. The grid formats must gather from a codebook, so they stay behind the
K-quants; Q2_0 is now 6.5 vs Q4_0's 14.3.

**Read:** prefill is ~1.3k tok/s past 4096 tokens and is PCIe-bound (one pass of the 25 GiB expert set per chunk).

### W10 2026-10-12 — What a second memory channel would buy

**Anchor:** decode at no cache / 2 / 4 GB

**Change:** analysis only — `membw` reads 15.8 GB/s flat (one channel); the vec-dot bench gained an `EXPERTS=` working-set sweep to separate a channel limit from a kernel limit

#### Channel analysis

ubi has a single DDR4 stick: `~/bench-rel/scripts/membw.c` reads **15.8 GB/s at 1, 2, 4 and 8
threads** — one channel, saturated. A matched second stick gives dual channel (~31 GB/s) and
30 -> 62 GB of RAM.

To separate a channel limit from a kernel limit, `bench_vec_dot_throughput` now sweeps its working
set (`EXPERTS=10`, ~19 MB, fits the 3600X's 32 MB L3; `EXPERTS=128`, ~240 MB, forces DRAM). Same
kernel, same 12 threads:

| type | 10e (L3) | 128e (DRAM) | ratio | verdict |
|---|---:|---:|---:|---|
| IQ2_S | 10.40 | 10.63 | 1.02 | compute-bound |
| IQ3_XXS | 9.73 | 11.01 | 1.13 | compute-bound |
| IQ3_S | 9.22 | 10.67 | 1.16 | compute-bound |
| Q2_0 | 6.75 | 8.63 | 1.28 | compute-bound |
| IQ4_NL (down) | 18.22 | 12.86 | 0.71 | bandwidth-bound |
| IQ4_XS | 15.40 | 13.71 | 0.89 | bandwidth-bound |
| Q2_K | 16.56 | 13.89 | 0.84 | bandwidth-bound |
| Q3_K | 23.51 | 13.91 | 0.59 | bandwidth-bound |
| Q4_K | 20.13 | 14.97 | 0.74 | bandwidth-bound |
| Q4_0 | 23.19 | 14.44 | 0.62 | bandwidth-bound |

Every bandwidth-bound type converges on 13–15 GB/s, i.e. the channel; the grid types and Q2_0 do
not move. The K-quants decode their 2-bit codes arithmetically and reach the wall — the IQ grid
types look the magnitudes up in a codebook, which is exactly what costs them.

Weighing by this model's traffic (`--cpu-moe`, 10 experts x 48 layers ≈ **965 MB per token**:
gate/up IQ2_S 20 layers, IQ3_XXS 17, IQ3_S 10; down Q2_0 9, IQ4_NL 39): **63% of the expert bytes
are in compute-bound types, 37% in bandwidth-bound ones**, so a second stick can only act on a
third of the traffic.

Estimate for a matched second stick:

| config | today | +stick |
|---|---:|---:|
| `--prefill-stream`, no cache | 7.3 | 8.0–8.7 |
| cache 2 GB | 11.4 | 12.4–13.4 |
| cache 4 GB | 15.4 | 16.8–18.2 |

Split: raw bandwidth ~+3–5% (only the 37%), 30 -> 62 GB ~+5–15% (the 26.8 GiB PLE table can finally
stay in page cache beside the 25.2 GiB of pinned experts; min free RAM is 1.7–3.0 GB today), prefill
~unchanged (PCIe bound at ~1.3k tok/s).

#### Is the bandwidth wall reachable at all?

| type | DRAM GB/s | x needed | verdict |
|---|---:|---:|---|
| IQ4_NL / IQ4_XS | 12.9 / 13.7 | 1.0 | already there |
| Q2_0 | 8.6 | ~1.8 | likely: no codebook, reads the fewest bytes per element (0.28 B/el, half of Q4_0) and only expands + `maddubs`, and Q4_0 reaches 14.4 |
| IQ2_S / IQ3_XXS / IQ3_S | 10.6 / 11.0 / 10.7 | ~1.35–1.6 | hardest: the gather is intrinsic, but Q2_K reaches 13.9 with an arithmetic decode, so a pre-expanded codebook or a better gather could get there |

#### Where the non-expert time actually is

Enqueueing the GPU work is **already overlapped** with the CPU experts (W6: a persistent worker runs
the misses while the host launches the next layer), so the residual is not launch cost. Fitting the
three measured points (uncached 136.5 ms / 965 MB, cache-2 87.9 / 402, cache-4 65.0 / 248) gives a
marginal cost of **86.3 us per MB of CPU expert reads — 11.6 GB/s effective**, in line with the
microbenchmark — and a serial intercept of **53 ms uncached, 44 ms at a 4 GB cache**. That intercept
is the part of the per-layer dependency chain that cannot overlap: attention, the dense and
shared-expert weights, the norms, the router and top-k readback, the PLE host gather, the cache
bookkeeping, and contention between the CUDA main thread and the expert worker. W5 measured the same
quantity as "everything between MoE calls", 24-45 ms.

Ceiling if the kernels were also fixed (expert reads at the channel limit):

| config | today | kernels fixed, 1 channel | + second channel |
|---|---:|---:|---:|
| no cache | 7.3 | ~8.7 | **~11.8** |
| cache 4 GB | 15.4 | ~16.5 | **~18.9** |

So the ordering matters: the kernels are what stand between today's numbers and the bandwidth wall,
and only after that does the second stick pay off fully. Note that the serial intercept caps this:
even with infinite bandwidth the uncached case cannot go below ~53 ms/token (19 tok/s).

**Read:** only 37% of this model's expert bytes are in bandwidth-bound types, so a matched second stick is worth ~+10% on decode today, and its value grows once the kernels are fixed.

### W11 2026-10-12 — MTP np=2 measured via `serve`

**Anchor:** decode with real prompts, no cache and at 2 GB

**Change:** none — measurement, because `bench`'s synthetic prompts misjudge MTP in both directions

#### serve results

W7's warning holds: `bench` misjudges MTP both ways, so this was measured the way W7 recommends —
`serve`, real prompts, greedy drafting, `usage.avg_compl_tok_per_sec`. Script `/tmp/mtp2-c2.sh`,
logs `~/bench-rel/mtpc{0,2}/`. Same flags as W7's matrix (`--cpu-moe --prefill-stream
--prefill-stream-min-tokens 64 --max-num-batched-tokens 4096 --max-prefill-chunk-tokens 4096
--pa-context-len 4096 --max-seqs 1`), `--mtp --mtp-draft-sampling greedy --mtp-n-predict 2`,
IQ1_M, 3 prompts x 2 reps, 256 new tokens.

| expert cache | base | MTP np=2 | delta | acceptance (mean drafted len) |
|---|---:|---:|---:|---|
| none | 7.50 (7.38–7.70) | **8.60** (8.47–8.74) | **+14.7%** | 92–100% (2.85–2.95) |
| 2 GB | 11.55 (10.70–12.19) | 11.69 (11.18–12.34) | +1.2% | 93–98% (2.85–2.95) |

The same cache-2 pair under `bench` reads 12.0 → 10.9, i.e. bench reports a 9% loss where serve
reports a 1% win. (This also does not reproduce W7's cache-2 finding of −20% on the K-quant model;
that number predates the expert-cache MTP fix and used the requant file.)

Why the gain collapses with a cache: a verify step reads the union of the routed experts over the
accepted tokens. Uncached that is ~3.3x one token's expert bytes, and amortizing the ~50 ms serial
intercept (W10) over ~2.9 tokens still nets +15%. With a 2 GB cache the per-step expert cost grows
~5.8x instead, so the same amortization nets +1%. Hit rate is *not* the cause — MTP improves it
(61.3% -> 66.0% with the stream, 60.2% -> 69.8% without).

A cache-4 `serve` comparison is not possible: `serve` refuses to start at a 3-4 GB cache (W5, "KV
cache requires 118 MB but only 0 MB is safely available"), so MTP at 4 GB is only observable through
`bench`, which is not trustworthy for it. Note MTP also tightens the spread noticeably at cache 2
(+-0.1 vs +-0.7 tok/s).

Practical read: MTP is worth having for the uncached / low-cache regime (+15%) and for stability, but
with the expert cache in play it is roughly neutral, and the fastest single configuration remains
cache + no MTP.

**Read:** MTP is a real +15% uncached and ~neutral at 2 GB; the gain collapses as soon as the expert cache is in play.

### W12 2026-10-12 — MTP x kernel: the two compound

**Anchor:** decode at 2 GB, projected

**Change:** analysis only — cost model `S + MB*c` (no MTP) vs `(S + u*MB*c)/T` (MTP)

MTP's only extra cost is `(u - T)·MB·c`, the extra expert bytes its verify step reads, while its
benefit — amortising the serial intercept `S` over `T` tokens — does not depend on `c`. A faster
kernel therefore makes MTP *relatively* better. Model constants from W10/W11: `T` = 2.9,
`S` = 50 ms uncached / 48 ms at 2 GB, `MB` = 965 / 402 MB, `u` = 3.3 uncached / 5.8 at 2 GB,
`c` = 86.3 us/MB (11.6 GB/s effective).

| scenario | uncached: base -> +MTP | 2 GB: base -> +MTP | MTP gain at 2 GB |
|---|---:|---:|---:|
| today (11.6 GB/s) | 7.5 -> 8.9 | 12.1 -> 11.6 | -4% |
| kernels at the 1-channel wall (15.8) | 9.0 -> 11.5 | 13.6 -> 14.8 | +9% |
| kernels + a second DIMM (~25 GB/s) | 11.3 -> 16.3 | 15.6 -> 20.5 | +32% |

The same table from the other side — no kernel change at all, only the verify union factor:

| u (2 GB cache) | base | +MTP | MTP gain |
|---|---:|---:|---:|
| 5.8 (today) | 12.1 | 11.6 | -4% |
| 4.5 | 12.1 | 14.2 | +17% |
| 3.3 (the uncached value) | 12.1 | 17.8 | **+48%** |

If experts ever become free (all resident) only `S` is left: 20 tok/s without MTP, 58 tok/s with
it, so MTP is the right end-state tool.

**Read:** the largest single lever is `u`, not `c`: at a 2 GB cache a verify step costs ~5.8x one token's expert bytes against ~3.3x uncached, so first find out whether the verify batch (3 tokens x 10 experts = 30 slot reads) is processed per token instead of grouped by expert across the batch — grouping would collapse it toward the distinct-expert count. Then the kernels (toward the 15.8 wall), then the second DIMM; each raises the value of the next.

### W13 2026-10-12 — Baseline fixed at MTP np=2 + cache 2, and where the expert slots go

**Anchor:** decode with `--mmproj mtp-component.gguf --mtp --mtp-n-predict 2 --expert-cache-gb 2`
= **11.69 tok/s** (serve, real prompts, W11). This is the baseline the following work is measured
against.

**Change:** none to behaviour — `gemv` (the CPU routed-expert path) gained an env-gated counter,
`IQ_DBG_UNION=1`, printing per call the number of (token, expert) slots it ran and how many distinct
experts those slots touch (fork `326cddf5`). A `OnceLock` keeps the check off the hot path.

**Data** (`~/bench-rel/union/`, one 24-token prompt, 32 generated tokens, cache 2):

| run | path | gemv calls | slots | distinct | per generated token | dup rate |
|---|---|---:|---:|---:|---:|---:|
| base | decode | 1856 | 7895 | 7895 | 246.7 | 0% |
| base | prefill | 144 | 15456 | 5547 | — | **64%** |
| MTP np=2 | decode | 1776 | 23184 | 16725 | **724.5** | **28%** |
| MTP np=2 | prefill | 147 | 15984 | 5892 | — | **63%** |

**Read.** Two things, both actionable:

1. **MTP's verify reads 2.94x the CPU expert slots per generated token** (724.5 vs 246.7) — that is the
   real union factor `u`, not the 5.8 inferred from throughput in W10/W12. Every 3-token call carries
   ~13 slots of which ~72% are distinct experts.
2. **The CPU path never deduplicates.** A slot is one (token, expert) pair and each one re-reads its
   expert's rows; in the base decode the slots happen to be distinct, but MTP's verify and every
   prefill chunk are not (28% and 63% duplicates).

Dedup by expert — group the slots per expert and keep the weight rows in cache while the group's
activations run against them — is therefore worth roughly +15% on the MTP decode (28% of its expert
traffic) and far more on prefill. That is the next step.

### W14 2026-10-12 — Dedup by expert: implemented, measured, no gain

**Anchor:** the W13 baseline, MTP np=2 + cache 2 = 11.69 tok/s (serve, real prompts).

**Change:** `gemv` grouped its slots by expert and ran the group's activations against each weight row
while it was still in L1, so a row shared by several slots is read once instead of once per slot (the
existing test `routed_gemv_matches_per_expert_qmatmul` has a duplicate expert and still passes).

**Data** (same serve protocol as W11, `~/bench-rel/mtpc2-dedup/`):

| config | before | with dedup |
|---|---:|---:|
| base (no MTP) | 11.55 | 11.44 |
| MTP np=2 | 11.69 | 11.39 |

No gain — inside the run-to-run spread.

**Read.** The W13 census explains it: 66.7% of the MTP decode slots are gate/up (k=2560), i.e. the
grid types W10 measured as **compute-bound**. Dedup removes *bytes*, not `vec_dot` calls, and for
those types the calls are the cost. Only the 33.3% down slots can benefit, and half of those are Q2_0
(also compute-bound), so the win is bounded at a couple of percent. Reverted rather than kept for
complexity's sake.

The lever therefore has to be `vec_dot` throughput itself, not the traffic it reads — which is what
W12's model says as well, since MTP pays 2.94x the slots per generated token, so a faster dot
multiplies straight through.

### W15 2026-10-12 — First kernel attempt (grid build through a stack array): no signal, and the harness is too noisy

**Anchor:** the W13 baseline (MTP np=2 + cache 2 = 11.69 tok/s) and the CPU dot throughput behind it
(L3 working set): IQ2_S 10.4, IQ3_XXS 9.7, IQ3_S 9.2, Q2_0 6.8, IQ4_NL 18.2 GB/s.

**Change:** rebuilt each grid type's codebook words — eight `u32` for IQ3_XXS/IQ3_S, four `u64` for
IQ2_S — in a stack array and took one 32-byte load, instead of `_mm256_set_epi32`/`set_epi64x` from
memory operands, which the compiler lowers to a chain of inserts.

**Data:** 8.13 / 8.19 / 7.64 GB/s against 10.40 / 9.73 / 9.22 before. But repeating the **unmodified**
bench three times gave IQ2_S 8.21–9.39, IQ3_XXS 8.58–11.87, Q2_K 15.4–19.9 — ±15–20% run to run on
the same binary. The two ranges overlap, so the regression is not established. Reverted, since
nothing is proven either way.

**Read.** The microbenchmark cannot resolve the 10–30% changes a kernel tweak produces, so it must not
be the decision instrument. Fix the harness first — pin the clock, raise the iteration count, or read
IPC with `perf stat` (blocked here: `perf_event_paranoid=4`, needs root) — or judge changes end to end
with `serve`. Separately: the grid types run ~5x below their issue-rate ceiling (~35 uops per 32
elements), so there is real room, but the cause has to be profiled rather than guessed. The clearest
structural target is still **Q2_0**, whose dot manages 6.8–8.6 GB/s on a shape where Q4_0 reaches
14–20.

### W16 2026-10-12 — Dedup by (expert, activation): +3%

**Anchor:** the W13 baseline, MTP np=2 + cache 2 = 11.69 tok/s (serve, real prompts).

**Change:** `gemv` now groups its slots by `(expert, activation row)` instead of by the expert
alone, so slots that share both compute one dot product and scatter the value. It matters because 67%
of an MTP verify batch's slots come from calls where all three tokens share one activation row
(`x_t == 1`); where such slots also share an expert, the two dots were bit-identical. (The W14
attempt grouped by expert only — that dedupes the weight *reads* and still ran every slot, which is
why it measured nothing.)

**Data** (serve, same protocol as W11, 6 requests each, two repetitions):

| run | base | MTP np=2 |
|---|---:|---:|
| before (W11, W13, W14) | 11.55, 11.44 | 11.69, 11.39 |
| after | 11.51, 11.33 | **11.98, 11.85** |

The base is unchanged, as it should be — it has no `x_t == 1` duplicates. MTP moved from −1.5% below
the base to +4% above it.

**Read.** +3%, and exact (the outputs are bit-identical), but well below the 7–15% the census
suggested: the duplicated slots are ~19% of the MTP decode slots (31% of the verify's, and only the
`x_t == 1` part of that is a true duplicate), and the CPU expert path is ~40% of a token, so the two
multiply down. This is the payoff curve for this baseline — the remaining CPU-side levers are the
inherent 2.03x union of the verify batch and the dot throughput itself.

### W17 2026-10-12 — Ceiling projection: memory-bound kernels x draft depth

**Anchor:** the W16 baseline, MTP np=2 + cache 2 = 11.92 tok/s (serve). This entry is an estimate,
not a measurement: it asks what the baseline becomes if every CPU expert kernel ran at the memory
bound, and how that scales with the draft depth.

**Inputs** (from the W13 census and the W14/W16 fit):

| | |
|---|---|
| Expert matrix, weighted over the model's shapes/dtypes | 678 KB (gate/up 600, down 835) |
| Verify-step expert traffic, calibrated at np=2 (V=3) | 1516 matrices = **1.03 GB/step** |
| Duplicate rate at V=3 (measured) | 31% -> union = 2.07x one token's set |
| Serial intercept `S` | 67 ms/step (48 in §4h's fit; both shown) |
| Tokens per step `T` | `1 + a + ... + a^np`, `a` = 0.95 (measured 2.85-2.95 at np=2) |
| Single-channel bandwidth | 15.8 GB/s (`membw`) |

**Projection** (per-token expert traffic is ~flat in np: the union grows but so does `T`):

| np | verify | T | matrices/step | matrices/token | S/T | expert @31 | **tok/s, 2 DIMMs** | tok/s, 1 DIMM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 2.85 | 1516 | 531 | 23.5 ms | 11.6 ms | **28.5** | 21.6 |
| 3 | 4 | 3.71 | 1908 | 514 | 18.1 ms | 11.2 ms | **34.1** | 24.9 |
| 4 | 5 | 4.52 | 2300 | 508 | 14.8 ms | 11.1 ms | **38.6** | 27.3 |
| 6 (auto) | 7 | 6.03 | 3083 | 511 | 11.1 ms | 11.2 ms | **44.9** | 30.3 |

Sensitivity: with `S` = 48 ms instead of 67 everything moves up 15–25%; with the per-token acceptance
at 0.90 instead of 0.95, np=4 gives 34.9 instead of 38.6 (at 0.97, 40.1).

**Read.**

1. At the memory bound and with a second DIMM, **np=3 reaches ~34 tok/s and np=4 ~39**, against
   ~28.5 for np=2 and 11.9 today.
2. There the expert time and `S/T` are already level (23.5 vs 11.6 ms at np=2; 11.1 vs 11.2 at np=6),
   so deeper drafting only buys back `S`. It saturates around np=6; the asymptote with `S` gone is
   ~90 tok/s (the expert traffic alone).
3. **Draft depth is therefore a better lever than the second DIMM** at that point: np 2 -> 4 is +35%,
   while 15.8 -> 31 GB/s at np=2 is +32% — and the DIMM does nothing until the kernels are fixed
   (W10: two thirds of the slots are compute-bound).
4. The ordering is unchanged: kernels to the bound first (11.9 -> 22 at np=2), then the memory
   channel and the draft depth together.

**Caveats.** The union model assumes each extra verified token adds 0.535 of a single token's set (the
measured rate at V=3) and that this marginal stays flat in V; deeper drafts probably overlap more, so
the traffic is likely *lower* than modelled. Acceptance is taken from np=2. The extra GPU-side work of
a bigger verify batch (attention over more rows) is not counted, which pushes the real numbers down.

### W18 2026-10-12 — perf profile of the vec-dot kernels: the grid types are issue-bound

**Anchor:** the W16 baseline (MTP np=2 + cache 2 = 11.92 tok/s) and, behind it, the CPU dot
throughput: IQ2_S 10.6, IQ3_XXS 11.0, IQ3_S 10.7, Q2_0 8.6 GB/s (DRAM working set) against Q2_K 13.9,
Q3_K 13.9, Q4_K 15.0.

**Change:** `perf_event_paranoid` was lowered to 1 (root), and the vec-dot bench gained `ONLY=` /
`VARIANT=` so a single kernel can be counted on its own (fork `78bb0fe`). No kernel changed.

**Data** — `perf stat -e cycles,instructions`, one type, serial loop, `EXPERTS=10`:

| type | cycles/byte | instructions/byte | IPC |
|---|---:|---:|---:|
| IQ2_S | 1.96 | 7.48 | 3.82 |
| IQ3_XXS | 1.82 | 5.74 | 3.15 |
| IQ3_S | 1.72 | 6.63 | 3.85 |
| Q2_0 | 2.26 | 6.79 | 3.01 |
| Q4_K | 0.84 | 2.41 | 2.87 |
| Q3_K | 1.00 | 2.95 | 2.94 |
| Q2_K | 1.07 | 3.30 | 3.08 |

L1-dcache miss rate over the whole bench: 1.9%. LLC events are not available on this kernel.

**Read.**

1. **IPC is 3.0–3.9 everywhere** (Zen2 issues ~4/cycle), so every kernel here is *issue-bound*, not
   latency- or bandwidth-bound — consistent with W10's finding that only the K-quants and IQ4_NL/XS
   reach the channel.
2. The grid types need **1.7–2.3x the cycles per weight byte** of the K-quants, and the reason is
   visible in the instruction mix: they require **2–2.5x the instructions per byte**. `objdump` on
   IQ3_XXS shows 12 `vpinsrd` plus ~40 `mov`/`lea`/`movzbl` — the `_mm256_set_epi32` codebook build
   and its addressing dominate. The K-quants decode their codes arithmetically (shift/mask) and never
   look anything up.
3. Both cheap ways to cut those instructions have now been measured and **fail**:
   * W15 (load the eight words through a stack array): within noise.
   * `_mm256_i32gather_epi32` in place of the eight scalar loads: instructions 718M -> 588M (-18%)
     but **cycles 228M -> 325M (+43%)**, IPC 3.15 -> 1.81. Zen2's gather is microcoded; this is what
     ggml's own comment warns about.
4. So the grid types sit near a floor for this CPU: the only lever left is fewer instructions *per
   element*, and a codebook lookup per four elements does not offer one. Reaching the memory bound
   would need a **representation change** — repacking the codebook magnitudes into bytes at load
   time — which the IQ3 magnitudes would let us do at 4 bits (1.3x the bytes, ~+7 GiB of pinned
   experts here) rather than 8 (2.6x).

**Consequence for W17.** Its "kernels at the memory bound" row is not reachable by tuning these
kernels; it is a *representation* question. The projection still holds as an upper bound, but the
realistic path is: keep the grid types as they are, and take the wins from the small-block types
(whose per-block reduction overhead is worth attacking) plus the MTP-side work.

### W19 2026-10-12 — Where the CPU expert time actually sits, by dtype

**Anchor:** the CPU expert path is ~40% of a decode token (the W10 fit) and W18 showed that most of
its kernels are issue-bound. This splits that cost by dtype, which is what decides what is worth
touching at all.

**Change:** none — derived from W13's census (expert bytes per dtype, shard 1 = 23.42 GiB) and W18's
per-type counters.

**Data.** Time share = bytes x cycles/byte, normalised.

| type | GiB | % bytes | cycles/byte | **% CPU expert time** | role |
|---|---:|---:|---:|---:|---|
| IQ2_S | 5.00 | 21.3% | 1.96 | **27.0%** | gate/up, 20 layers |
| IQ3_XXS | 5.08 | 21.7% | 1.82 | **25.5%** | gate/up, 17 |
| IQ4_NL | 8.57 | 36.6% | 1.02 | 24.2% | down, 39 |
| IQ3_S | 3.36 | 14.3% | 1.72 | **15.9%** | gate/up, 10 |
| Q2_0 | 0.99 | 4.2% | 2.26 | 6.2% | down, 9 |
| IQ4_XS | 0.42 | 1.8% | 1.07 | 1.2% | 1 layer |

Grid types (IQ2_S + IQ3_XXS + IQ3_S): **13.44 GiB = 57.4% of the expert bytes and 68.4% of the CPU
expert time**, and they are the gate/up of 47 of the 48 layers. Bandwidth-bound types (IQ4_NL +
IQ4_XS) are 38.4% of the bytes and 25.4% of the time.

**Read.**

1. W18's floor applies to **about two thirds** of the CPU decode cost, not a corner of it.
2. Of the remainder, IQ4_NL (24.2% of the time) already runs at the channel with 1.02 cycles/byte, so
   it has no kernel headroom at all — only a memory channel helps it.
3. Q2_0 (6.2%) is the one type that is expensive without a codebook (2.26 cycles/byte, the worst
   here) and whose cost is per-block *reduction* overhead rather than the dot itself, so it is
   attackable — but at 6% of the cost even a 25% win there is ~1.5% end to end.
4. So the CPU kernel side is close to exhausted: the levers left are the MTP-side amplification (the
   verify union multiplies this whole cost by 2.03), the serial intercept `S`, and the second memory
   channel — which acts on the 38% of the bytes that are bandwidth-bound, and only after the two
   thirds that are compute-bound stop mattering.

### W20 2026-10-12 — `--pcie-frac` works: +11% on the MTP np=2 + cache 2 baseline

**Anchor:** the W16 baseline, MTP np=2 + cache 2, `serve` with real prompts = 11.92 tok/s.

**Change:** none — this measures the existing `--pcie-frac` implementation (naiwa-rs working tree, the
same code the earlier entries left uncommitted): a small per-layer scratch bank in VRAM, beyond the
cache budget, that a decode step DMAs its highest-weight distinct misses into so they run on the GPU
with the residents instead of on the CPU.

**Data.** `serve`, real prompts, greedy, MTP np=2 + `--expert-cache-gb 2 --prefill-stream`,
6 requests each, one session (`scripts/offload-sweep.sh`, logs `~/bench-rel/pcie/`):

| `--pcie-frac` | mean tok/s | range | peak VRAM | scratch |
|---:|---:|---|---:|---|
| 0.0 | 12.19 | 11.32–12.84 | 10.5 GB | — |
| 0.25 | 13.09 | 12.41–14.10 | 10.5 GB | 3 slots/layer (~6 MiB) |
| **0.5** | **13.52** | 12.77–14.57 | 10.9 GB | 5 slots/layer (~10 MiB) |
| 1.0 | 13.52 | 12.56–14.16 | 11.3 GB | 10 slots/layer (~20 MiB) |

**+11%**, and it saturates at 0.5: 1.0 buys the same throughput for 0.4 GB more VRAM.

**Mechanism confirmed** (`bench`, same flags, frac 1.0): `--pcie-frac: 7761 misses DMA'd to the GPU
(15156 MiB)`, and the decode hit rate rises 45.6% -> 79.8% — the DMA'd misses are re-classified as
hits, which is the intended accounting.

**Read.**

1. The feature is real and worth ~11% at the point where the CPU expert path is the bottleneck
   (W16/W19: ~40% of a token, 68.4% of it in the issue-bound grid types). It matches the estimate
   from the PCIe floor: one MTP verify step asks the CPU for ~176 ms of expert work whose bytes
   (~1.03 GB) stream at 13.35 GB/s in ~77 ms.
2. Saturating at 0.5 says the PCIe/GPU side becomes the limit, not the CPU: past half the misses
   there is nothing left to win. Half the scratch also halves the VRAM cost.
3. VRAM is the cost: +0.4 GB at frac 1.0 over the cache budget, on a 12 GB card where a 4 GB cache
   plus the prefill stream already OOMs. `0.25–0.5` is the usable range **at a 2 GB cache**; see W26
   for cache 4, where only 1-2 slots/layer fit and 1 is optimal.
4. **Correctness was not established by output comparison**, and that is a limitation of these
   prompts, not of the feature: the *baseline* is not reproducible across sessions on them (fib-1
   across five sessions: 185 / 254 / 254 / 252 / 197 characters). In the pcie session itself 4 of 6
   requests were byte-identical to frac 0 and the two that differed diverged at plausible near-ties -
   the same behaviour §4f documents for the hit/miss split (experts computed on different hardware
   round differently). A real parity check needs a long prompt through `scripts/parity.sh`.

**Usability note:** the DMA counter is a `println!` gated on `pcie_misses > 0`, and the scratch
allocation is an `INFO` log, so a mis-set fraction is silent unless `-v` is passed; the VRAM jump and
the hit-rate change are the only other signals.

### W21 2026-10-12 — MTP np=4 is *worse* than np=2 at a 2 GB cache

**Anchor:** the MTP np=2 + cache 2 baseline, 12.04 tok/s (frac 0) / 13.55 (frac 0.5) in this session.

**Change:** none — `--mtp-n-predict 4` against `2`, both at `--expert-cache-gb 2`, via
`scripts/offload-sweep.sh` (6 real prompts each, one session):

| draft depth | frac 0 | frac 0.5 | peak VRAM |
|---|---:|---:|---:|
| np=2 | 12.04 | **13.55** | 10.5 / 10.8 GB |
| np=4 | 10.81 (-10%) | 12.01 (-11%) | 10.5 / 10.9 GB |

**Read.** Deeper drafting loses ~10% at this cache size, with or without the DMA path, which
**falsifies W17's np scaling**. That model assumed each extra verified token adds only 0.535 of a
single token's expert set (the rate measured at V=3) and that the marginal stays flat. At V=5 it does
not: the union (up to 5 x top-10) is far past the ~23 slots/layer a 2 GB cache holds, so the miss set
grows faster than `S/T` shrinks, and the verify batch's extra GPU-side work lands on top. W17's
numbers should be read as an upper bound that only holds where the union fits the cache (a 4 GB cache
holds 47 slots/layer, above 3 x top-10 but not 5 x).

So the best configuration this session remains **np=2 + cache 2 + frac 0.5 = 13.55 tok/s**, and the
earlier `bench` reading of np=4 (8.5 tok/s at cache 4) was not the synthetic-prompt artefact it looked
like.

### W22 2026-10-12 — Second memory channel, revised estimate

**Anchor:** the W20 best configuration, np=2 + cache 2 + frac 0.5 = 13.55 tok/s = 73.8 ms/token.

**Change:** none — this revises W10's estimate (which predates the DMA path and the per-dtype split).

**Decomposition** of the 73.8 ms (from the W14/W16 fit plus the measured frac-0 -> frac-0.5 delta):

| | ms/token |
|---|---:|
| `S`, the serial intercept (amortised over 2.9 tokens) | 23.1 |
| CPU expert work (the 60% of misses that stay on the CPU) | 36.0 |
| PCIe/DMA (the 40% that the scratch bank streams) | 14.7 |

**Estimate** for a matched second stick (dual channel, 15.8 -> ~31 GB/s, 30 -> 62 GB):

| effect | ms saved | basis |
|---|---:|---|
| the bandwidth-bound share of the CPU expert work (W19: IQ4_NL + IQ4_XS = 25.4%) halves | 4.6 | 9.1 ms of the 36.0 |
| the PLE table (26.8 GiB) can stay in page cache, no swap | ~1.5 | W3/W10 |
| DRAM/PCIe contention: the DMA alone wants 13.35 of the 15.8 GB/s the one channel has | ~4 | inferred |
| **total** | **~10** | |

=> **65–68 ms/token, i.e. ~14.8–15.3 tok/s (+9–13%)** on the best configuration; +10–16% on the
frac-0 one (12.04 -> 13.3–14.0).

**Read.**

1. The gain is bounded by W19: the grid types are 68.4% of the CPU expert time and are compute-bound,
   so they do not move at all. Even with infinite bandwidth the CPU expert floor stays at ~27 ms of
   the 36.
2. The DMA path makes the channel matter *more*, not less: 13.35 GB/s of it is already spoken for by
   the PCIe stream, so the CPU experts and the DMA are competing for one channel — a second one
   removes that coupling as well as doubling the bandwidth.
3. The error bar is wide (±5 points) because the PLE and contention terms are inferred, not measured;
   re-running W20's sweep after the stick is installed would settle it in about ten minutes.

**Checklist to run once the stick is in** (about an hour):

```sh
~/bench-rel/scripts/membw                      # expect ~31 GB/s, not 15.8 - the premise
scripts/offload-sweep.sh after-c2-f000 2 0.0   # the W20 sweep, same session, to compare like for like
PORT=18501 scripts/offload-sweep.sh after-c2-f025 2 0.25
PORT=18502 scripts/offload-sweep.sh after-c2-f050 2 0.5
PORT=18503 scripts/offload-sweep.sh after-c2-f100 2 1.0
PORT=18504 scripts/offload-sweep.sh after-np4-f050 4 0.5   # does the extra cache headroom change W21?
```

Compare against W20 (12.19 / 13.09 / 13.52 / 13.52) and W21 (np=4: 12.01). The three things to look
for: whether the fractional gain tracks the 25.4% bandwidth-bound share or beats it (then the
contention term was real), whether the frac sweep still saturates at 0.5 (more bandwidth means the
DMA side can take more), and whether np=4 recovers (it should not - that is a cache-slot limit, not a
bandwidth one).

### W23 2026-10-12 — Inside one decode+verify round: GPU vs CPU expert time across draft depths

**Anchor:** np=2 + cache 2 = 12.04 tok/s and np=4 = 10.81 (W21), both `--pcie-frac 0`.

**Change:** none — the `IQ_DBG_UNION` census (W13) run at np=4 with the same prompt and the same 32
generated tokens as W13's np=2 run, combined with the 84 us/slot fit from W14/W16 (the same kernels
run in both).

**Census** (`~/bench-rel/../census-np{2,4}`, decode calls only):

| | np=2 | np=4 |
|---|---:|---:|
| verify rows per step | 3 | 5 |
| mean drafted length `T` | 3.00 | 5.00 |
| steps for 32 tokens | 10.67 | 6.40 |
| gemv calls / step | 166.5 | 191.3 |
| slots / step | 2178 | 4028 |
| distinct experts / step | 1563 | 2434 |
| slots / token | 726 | 806 |
| duplicate rate of (expert, activation) pairs | 28% | 40% |

**Decomposition** (CPU expert = slots/step x 84 us; the rest is GPU + the serial chain):

| | np=2 | np=4 |
|---|---:|---:|
| **per step** | **249 ms** | **463 ms** |
| - CPU expert | 183 ms (**73%**) | 338 ms (**73%**) |
| - GPU + serial | 66 ms (27%) | 124 ms (27%) |
| **per generated token** | **83.1 ms** | **92.5 ms** |
| - CPU expert | 61.0 ms | 67.7 ms |
| - GPU + serial | 22.1 ms | 24.8 ms |

**Read.**

1. The split is **73% CPU expert / 27% GPU+serial at both depths**. The CPU expert path is the
   critical path; the GPU and the per-layer serial chain (attention, dense, norms, router, PLE
   gather, the GPU hit path, the launch chain) are the minority — and the np=2 `S` of 67 ms/step from
   W14/W16 is that 66 ms, which is a good independent check on the fit.
2. np=4 loses because **both halves grow ~85–90% per step while `T` grows only 67%** (3 -> 5 rows):
   slots/step 2178 -> 4028 and GPU/serial 66 -> 124 ms, so everything is ~+11% per token.
3. The census explains why the CPU half grows faster than `T`: distinct experts/step grow *slower*
   than `T` (+56%), but the duplicate rate of (expert, activation) pairs rises 28% -> 40%, so the
   *executed* slots grow faster than the token count. Deeper drafting adds pairs, not just tokens.
   (W16's dedup only removes pairs that are exactly equal; across 5 rows more pairs share an expert
   with different activations.)
4. Practical: with 73% of the round in the CPU expert path and that path growing with the batch, the
   three levers stay the same — a bigger cache (fewer misses), `--pcie-frac` (moves ~half of it onto
   PCIe for W20's +11%), and kernel throughput (W18: little headroom left).

### W24 2026-10-12 — A 4 GB cache: what runs, what cannot be measured

**Anchor:** the MTP np=2 + cache 2 baseline (12.19 / 13.55 tok/s) and W21/W23's finding that a 2 GB
cache cannot hold a verify batch's expert union.

**Change:** none — five `bench` runs at `--expert-cache-gb 4 --pa-context-len 4096`, prompt 64 in / 32
out, peak VRAM sampled by `~/iq-bench.sh`. Run in increasing order of risk, checking peak VRAM and
the exit code between each because this size has OOM'd before (B1/R4).

| config (cache 4) | decode tok/s | peak VRAM | hit rate | OOM |
|---|---:|---:|---:|---|
| np=0, no stream | **12.5** | 10.7 GB | 60.5% | no |
| np=2, no stream | (9.1)* | 10.9 GB | 72.2% | no |
| np=4, no stream | (6.1)* | 10.9 GB | 80.0% | no |
| np=2, chunk 4096 stream | (7.5)* | 11.0 GB | - | no |
| np=0, stream (W8's E3, prompt 512) | 15.4 | 11.0 GB | - | no |

\* not usable: `bench`'s synthetic prompts leave MTP acceptance at 5.6% (np=2) and 10.9% (np=4), so
those rows time the drafting overhead rather than MTP's payoff.

**Read.**

1. **No OOM in any of the five** (peak 10.7–11.0 GB). B1/R4's OOM is specific to cache 4 + the stream
   with a 2048-token prefill chunk; 4096 was fine here, and so was MTP on top of it.
2. The solid result is the **hit rate**: 45.6% at cache 2 (np=2, W20) -> **72.2% at cache 4**, and
   80.0% at np=4. The 4 GB cache does hold most of the verify batch's union, which is the bottleneck
   W21 and W23 identified.
3. Cache 4 *without* the stream is 12.5 tok/s against 15.4 with it (E3), so the stream's host-side
   pinning is worth ~23% of decode on its own, not only the prefill win.
4. **Cache 4 + MTP timing cannot be measured on this box** — see B7. Four routes were tried and all
   are blocked. This is the one configuration that could overturn W21's "np=2 beats np=4", since 47
   slots/layer is much closer to a 5-row union than 23 is, so unlocking it is worth more than the
   kernel work that is left.

### W25 2026-10-12 — `--prompt-file`: real text in `bench`, and what it says about cache 4 and np=4

**Anchor:** W24's blocked measurement — cache 4 + MTP could not be timed, because `bench`'s
synthetic prompt (`BENCH_TOKEN_BASE + idx % SPAN`) leaves MTP acceptance at 5.6% (np=2) / 10.9%
(np=4) and `serve` cannot start at a 3-4 GB cache (B7).

**Change:** `bench --prompt-file <PATH>` (commit `550cf15`). The file's text is sent as
`RequestMessage::Completion`, so the engine tokenizes it exactly as `serve` does; the prompt-length
sweep collapses to one case (the file's token count is only known after tokenization) and TTFT is
skipped through the existing "zero skips TTFT" convention, leaving the decode cases to run against
real text. Acceptance on a 24-token code prompt goes **5.6% -> 100%**.

**Data.** `bench` with `--prompt-file ~/bench-rel/pf/prompt-fib.txt` (a 24-token Python prompt),
`--gen-len 32 --depth 4 --iterations 3 --warmup 1`, `--pcie-frac 0`, one prompt (so this is a
*different harness* from the `serve` entries - compare within this table, not across). Logs
`~/bench-rel/pf/w25-cache{2,4}-np{0,2,4}.log`; the 30%-variance 1-iteration attempt that this
`--iterations 3` protocol replaced is `w25-noisy-1iter-cache4-np2.log`:

| expert cache | np=0 | np=2 | np=4 |
|---|---:|---:|---:|
| 4 GB | 14.6 | **17.4 ±0.5** | **17.4 ±1.1** |
| 2 GB | 11.8 | 14.2 ±0.0 | 12.9 ±1.1 |

**Read.**

1. **At a 4 GB cache, np=4 equals np=2** (17.4 both); the ~10% penalty W21 measured at 2 GB is gone.
   That is exactly what W21/W23 predicted: 23 slots/layer cannot hold a five-row verify union, 47
   can. So the np=4 verdict is a cache-capacity statement, not a property of MTP.
2. Cache 4 is worth **+23%** over cache 2 at np=2 (14.2 -> 17.4) with everything else equal, which is
   the largest single win measured in this log so far - and it confirms the hit-rate evidence of W24
   (45.6% -> 72.2%).
3. MTP itself is worth ~+19% over np=0 at cache 4 (14.6 -> 17.4) and ~+20% at cache 2 (11.8 -> 14.2).
4. Two caveats: this is a single-prompt `bench` run without the prefill stream, so it is not directly
   comparable with the `serve` numbers (W20's 13.55 at cache 2 + frac 0.5 is a different harness);
   and `serve` at cache 4 was impossible at the time of writing (B7; resolved in W27, where the
   `serve` counterpart is 16.58 tok/s).

### W26 2026-10-12 — `--pcie-frac` at a 4 GB cache: **one scratch slot/layer** is the sweet spot

**Anchor:** W20 validated the flag at a 2 GB cache (frac 0.5 = 5 scratch slots/layer, +11%, saturating);
W25 supplied real prompts, so the flag can now be measured against the cache-4 baseline that replaced it.

**How the knob works.** `scratch_slots(frac, top_k) = clamp(ceil(frac * top_k), 1, top_k)` reserves that
many extra slots *per layer* beyond the cache (`top_k = 10`, so fractions 0.01-0.1 = 1 slot, 0.11-0.2 = 2,
...); `streamed_misses` then DMAs the highest-total-gate-weight of the step's misses into them (capped by
the scratch size) so those run on the GPU instead of the CPU. Cost: 1 slot = 48 x 1.88 MB = ~90 MB,
3 slots = ~270 MB.

**The VRAM frontier at cache 4** (log: `~47 slots/layer, 1881600 bytes/slot`; only `available_gb=2` left):

| scratch slots | `--pcie-frac` | result |
|---|---:|---|
| 0 | 0.0 | runs, 18.3 tok/s |
| 1 | 0.05-0.1 | runs, **18.9**, 3.2 GB DMA'd |
| 2 | 0.15-0.2 | runs, 18.3-18.8, 3.3 GB DMA'd |
| 3 | 0.25 | **`CUDA_ERROR_OUT_OF_MEMORY` during weight load** |

**Cache GB vs scratch slots** (np=2, real prompt, `--iterations 3 --warmup 1`; one batch, so compare within
the table - cross-batch drift in this harness is 4-8%):

| expert cache | scratch slots | decode tok/s | DMA'd |
|---:|---:|---:|---:|
| 4.0 GB | 0 | 18.8 | - |
| 4.0 GB | 1 | **19.4** | 3.2 GB (1659 misses) |
| 3.5 GB | 3 | 18.5 | 10.8 GB (5502) |
| 3.0 GB | 3 | 18.2 | 11.6 GB (5950) |
| 3.0 GB | 10 | 17.7 | 26.9 GB (13715) |

Interleaved confirmation of the first row pair: frac 0.0 = 18.2 / 18.4, frac 0.05 = 19.0 / 18.8, i.e.
**+3.3% reproducible**; the effective decode hit rate rises 78.9% -> 81.3% as the scratch absorbs the
top misses.

Logs: `~/bench-rel/pf/w26-{4-0.0,4-0.05,4-0.1,3.5-0.5,3-0.5,3-1.0}.log` for the trade-off table,
`w26-rep-*.log` for the interleaved repeats (incl. the 2-slot runs 0.15 / 0.2),
`w26-oom-cache4-frac0.25.log` for the load-time OOM, and
`w26-best-cache4-frac0.05-np{2,4}.log` for the best configuration.

**Read.**

1. **The first scratch slot is worth ~+3%, a second adds nothing, and a third does not fit.** W20's
   "0.25-0.5 is the usable range" is a cache-2 statement: 5 slots fit a 2 GB cache but OOM a 4 GB one, so
   the optimum is a function of cache size, not a fixed fraction. Express the knob in slots/layer.
2. **Cache GB still dominate scratch slots:** 4 GB + 1 slot (18.9-19.4) beats 3.5 GB + 3 slots (18.5) and
   3 GB + 10 slots (17.7). Trading cache for scratch is a loss.
3. np=4 does not overtake np=2 here either (17.3 +-1.9 vs 18.3 +-0.9), consistent with W25.
4. **Best configuration measured so far: cache 4 + `--pcie-frac 0.05` (1 slot) + np=2 = 18.9-19.4 tok/s**,
   in the bench harness (W25 caveats apply: single prompt, no prefill stream, not comparable with the
   `serve` numbers; `serve` at cache 4 is now possible via `--prefix-cache-n 2-3`, see W27).

### W27 2026-10-12 — B7 resolved: the 1887 MB is the recurrent prefix snapshot cache, and `serve` runs at a 4 GB cache

**Anchor:** B7 — `serve` + MTP could not start at a 3-4 GB expert cache, so W25/W26 had no `serve` counterpart.

**Root cause.** The "fixed 1887 MB" is not an activation reservation. It is the **recurrent prefix snapshot
cache**: `PagedAttentionConfig::recurrent_prefix_capacity` (set from `--prefix-cache-n`, default 16) is
multiplied into `CacheMemoryReservations` by `add_recurrent_prefix_memory_reservations` as
`bytes_per_snapshot x (capacity + 1)`. One snapshot is 111 MB on this model, so 16 cached prefixes reserve
1887 MB - more than the KV cache itself - and it is subtracted before the KV request is fitted:

    available = device_memory_cap(free) - reserved - affine           (all post-load)
    0 MB      = (1204 - 512) - 1887 - 0                               (4 GB cache, --pa-context-len 4096)

while the failure named only the KV cache ("requires 128 MB but only 0 MB is safely available"), i.e. the
wrong knob.

**The reservation is a worst case, not real usage.** Idle VRAM after startup is **identical** (8525 MiB)
with `--prefix-cache-n 0` and `16` at a 2 GB cache, even though the reservation differs by 1887 MB: nothing
is allocated until a prefix is actually cached.

**Fix** (fork `9fa42b41`, branch `patch/strata`). `CacheMemoryReservations` now carries the snapshot share
(bytes per snapshot, and how many the reservation stands for) and the sizing failure reports the whole
budget:

    PagedAttention KV cache requires 128 MB but only 0 MB is safely available after model loading.
    Budget: 1204 MB free - 512 MB device reserve - 1887 MB post-load reservations (1887 MB of them for
    the recurrent prefix snapshot cache: 16 cached prefixes x 111 MB) - 0 MB packed GGUF affine weights.
    The snapshot cache is the only tunable part: with this 128 MB KV cache at most 4 cached prefixes fit,
    so lower `--prefix-cache-n` (currently 16).

The answer is a flag value (`3` starts the server, `4` is one snapshot too many - the free-memory reading
moves ~100 MB between runs), not a search through `--max-num-batched-tokens`, which never had anything to
do with it.

The share is deliberately **not shed automatically**: a snapshot commit that cannot allocate rejects the
request with `Response::InternalError` (the `recurrent_prefix_*_failure_*` scheduler tests pin this), and the
snapshot capacity also bounds the KV prefix cache, so dropping the reservation without propagating a smaller
prefix capacity would trade a startup failure for failed requests. Threading the effective capacity into the
prefix validator is the remaining piece of B7.

**`serve` at a 4 GB cache** (`scripts/offload-sweep.sh`, three real prompts x 2 reps, `--pcie-frac 0.05`,
same protocol as W20/W21):

| configuration | tok/s | range | peak VRAM |
|---|---:|---|---:|
| cache 2, np=2 (control) | 12.50 | 11.69-13.41 | 10580 MB |
| cache 4, np=2, `--prefix-cache-n 2` | **16.58** | 16.01-17.49 | 11604 MB |
| cache 4, np=2, `--prefix-cache-n 3` | 16.00 | 13.18-18.37 | 11604 MB |
| cache 4, np=4, `--prefix-cache-n 2` | 15.32 | 14.16-15.74 | 11666 MB |

**Read.**

1. **+33% for cache 4 over cache 2 in the `serve` harness** (12.50 -> 16.58), and **+23% over the previous
   best `serve` number** anywhere in this log (W20's 13.52 at cache 2 + frac 0.5).
2. np=4 stays 8% behind np=2 under `serve` at cache 4 (15.32 vs 16.58), where `bench` measured parity (W25).
   For a difference this small the two harnesses disagree in sign, so np=2 remains the safe choice.
3. These `serve` numbers sit ~15% below `bench` at the same cache (16.6 vs 18.9-19.4 in W26), which is the
   harness gap W25 warned about (varied real prompts, prefill stream, HTTP, prefix reuse). Use `bench` for
   A/B and `serve` for absolute claims.
4. `--prefix-cache-n` is a real trade, not free: at a 4 GB cache it drops to 2-3, which effectively disables
   prefix reuse. Spending the last GB on experts rather than on prefix reuse is the choice this makes.
5. Peak VRAM lands at 11604-11666 MB of 12282 MB, so cache 4 + MTP + one scratch slot is within ~600 MB of
   the card's limit; anything else that grows with the batch will not fit.

Logs: `~/bench-rel/sweep/w27-{c2-f050-np2,c4-f050-np2-pcn2,c4-f050-np2-pcn3,c4-f050-np4}.log` with their
`vram-*.log`; `w27-c4-f050-np2.log` is the `--prefix-cache-n 4` failure that shows the new message, and
`~/bench-rel/pf/w27-{lazy-snapshots-pcn0,lazy-snapshots-pcn16,new-message}.log` hold the lazy-snapshot
evidence (identical idle VRAM at 0 and 16 prefixes).

### W28 2026-10-12 — `--pcie-frac` under `serve` at a 4 GB cache: two scratch slots, +4%

**Anchor:** W27 established the serving configuration at a 4 GB cache (np=2, `--prefix-cache-n 2`), and its
numbers already included `--pcie-frac 0.05`. W26 fixed the flag's optimum in the `bench` harness (one scratch
slot/layer); this is the `serve` side of the same question.

`scripts/offload-sweep.sh` (three real prompts x 2 reps, cache 4, np=2, `--prefix-cache-n 2`):

| `--pcie-frac` | scratch slots/layer | tok/s | range | peak VRAM |
|---:|---:|---:|---|---:|
| 0.00 | 0 | 15.97 | 14.33-17.35 | 11508 MB |
| 0.05 | 1 | 16.26 | 13.92-17.55 | 11636 MB |
| 0.15 | 2 | **16.64** | 14.64-17.74 | 11700 MB |
| 0.25 | 3 | **OOM at startup** | - | - |

**Read.**

1. The flag is worth **+4% over off** at cache 4 (15.97 -> 16.64) with the mean rising monotonically with the
   scratch count. The steps are ~2% each, so the 1-vs-2 slot choice is inside this harness's spread
   (ranges overlap heavily); the sign of the trend is the reliable part.
2. **Three slots OOM during allocation, before serving a single request**, exactly as in `bench` (W26), so the
   VRAM ceiling - not the harness - sets the limit. Peak VRAM walks 11508 -> 11700 MB for 0 -> 2 slots, of
   12282 MB, and 3 slots (~+96 MB) does not fit because sizing happens when the PagedAttention cache and the
   PLE gather have already taken their share.
3. The gain is small because at a 4 GB cache the decode hit rate is already 72-80% (W24): there are few
   misses left to DMA. The 2 GB cache, where misses dominate, got +11% from frac 0.5 (W20).
4. `bench` (W26) said one slot, `serve` says two; both are within ~2% and the ranking inside a run is
   monotone, so treat this as "1-2 slots, never 3". For a 4 GB + MTP configuration I would run 0.15 (two
   slots) and keep 0.05 in reserve if anything else grows.

Logs: `~/bench-rel/sweep/w28-c4-np2-pcn2-f{000,005,015,025}.log` with `vram-*.log`; the 0.25 run is the one
that OOMs at load (`--pcie-frac: 3 scratch slots/layer` immediately followed by `CUDA_ERROR_OUT_OF_MEMORY`,
zero requests served).

### W29 2026-10-12 — the second DIMM landed: +9-16% decode, and why that is all it is worth

**Anchor:** W10's channel analysis (single channel, `membw` 15.8 GB/s flat) and W22's projection (+9-13% from a
matched second stick).

**Hardware.** RAM is now 60 GB (was 30). `~/bench-rel/scripts/membw.c`:

| threads | before (1 channel) | now (2 channels) |
|---|---:|---:|
| 1 | 15.8 | **22.0 GB/s** |
| 2 | 15.8 | **32.0 GB/s** |
| 4 | 15.8 | 31.6 GB/s |
| 8 | 15.8 | 29.3 GB/s |

so +39% at one thread and roughly double once two threads run.

**Per-kernel evidence** (`EXPERTS=128` forces DRAM, same test and the same *split* column as W10):

| type | 1 channel | 2 channels | change |
|---|---:|---:|---:|
| Q3_K | 13.91 | 26.78 | +93% |
| IQ4_XS | 13.71 | 25.22 | +84% |
| Q4_K | 14.97 | 27.42 | +83% |
| Q2_K | 13.89 | 25.45 | +83% |
| IQ4_NL (down) | 12.86 | 20.37 | +58% |
| Q4_0 (down) | 14.44 | 20.84 | +44% |
| IQ3_S | 10.67 | 15.24 | +43% |
| **IQ2_S** | 10.63 | 12.14 | **+14%** |
| **IQ3_XXS** | 11.01 | 12.27 | **+11%** |
| **Q2_0 (down)** | 8.63 | 9.60 | **+11%** |

**Decode** (bench protocol of W25/W26, byte-identical prompt file, so directly comparable):

| configuration | 1 channel | 2 channels | change |
|---|---:|---:|---:|
| bench cache 4, np=2, frac 0.05 | 19.4 | **21.6 ±1.4** | +11% |
| bench cache 4, np=2, frac 0.15 | 18.3-18.8 | **21.2 ±0.6** | +13-16% |
| bench cache 2, np=2, frac 0.00 | 14.2 | **15.5 ±1.1** | +9% |

`serve` (W27/W28 protocol; fib+long only, n=4, and `usage.prompt_tokens` = 28 / 2100 confirms the prompts are
the originals):

| configuration | 1 channel | 2 channels | change |
|---|---:|---:|---:|
| cache 4, np=2, `--prefix-cache-n 2`, frac 0.15 | 16.22 | **18.79** | +16% |
| cache 2, np=2, frac 0.5 | 13.32 | **14.57** | +9% |
| (all prompts) cache 4 / cache 2 | 16.63 / 13.54 | 18.90 / 14.99 | +14% / +11% |

#### Why only +9-16% for a doubled memory channel

1. **Most of the CPU expert time does not scale with bandwidth.** The grid formats (IQ2_S, IQ3_XXS, IQ3_S)
   gain only 11-43% here while the K-quants and IQ4_XS gain 83-93% - the same split W10 found (they sit at
   10-12 GB/s against a 15.8 GB/s channel, i.e. issue-bound, not channel-bound; W18 put them at IPC 3.0-3.9
   and 5.7-7.5 instructions/byte). Worse, they are **68.4% of the CPU expert time** (W19), so the large
   multipliers land on the smaller 31.6%.
2. **The CPU expert path is only about half of a token.** W18's breakdown of the MTP np=2 + frac 0.5 round
   (73.8 ms) is 23.1 ms serial/GPU + 36.0 ms CPU expert + 14.7 ms PCIe. Amdahl then predicts
   `0.316 x 36 ms = 11.4 ms` of a 73.8 ms token (15%) is bandwidth-sensitive; speeding *that* up 1.8x buys
   5.1 ms, i.e. **+7-8%**. The measured +9-16% brackets it, and the extra likely comes from the per-step host
   gather of the 28 GB PLE table (see the `qwen4_exp::ple` warning), which every configuration pays.
3. Nothing about this changes the serial/GPU part (23.1 ms) or the cache-management cost, which is why the
   cache-2 and cache-4 gains come out within a few points of each other despite very different hit rates.

**Caveats.**

- At cache 4 with two scratch slots (frac 0.15) **one of the six requests OOM'd** (peak 11551 MB against 12282
  and a desktop session holding ~440 MB). frac 0.05 (one slot) ran 6/6 clean at 18.53, so the recommendation
  at cache 4 is now one scratch slot.
- W17's projection that two DIMMs plus MTP would reach 28.5 tok/s is still far from observed (18.9); memory
  was one of several ceilings, not the binding one.
- The reboot that installed the DIMM also cleared `/tmp`, which held the `offload-sweep.sh` prompts. They now
  live at `~/bench-rel/prompts/req-{fib,long,class}.json`; fib and long are byte-identical to the originals
  (175 B / 4936 B, and `prompt_tokens` 28 / 2100), while `class` is a substitute (67 -> 32 prompt tokens) and
  is therefore excluded from the comparisons above.

Logs: `~/bench-rel/ram2/bench-c{2,4}-f*.log`, `~/bench-rel/ram2/ram2-*.log` (+ `vram-*`), the old side in
`~/bench-rel/sweep/*.json`; the microbenchmark is `EXPERTS=128 cargo test --release -p mistralrs-quant --lib
bench_vec_dot_throughput -- --ignored --nocapture`.

## 5. Known issues

| # | Issue | Detail |
|---|---|---|
| B1 | `--expert-cache-gb 4` OOMs with prefill stream (design G9) | Only 5882 MiB free after the KV cache. The prefill stream borrows ~1380 MiB of cache slots for layer buffers, then `CUDA_ERROR_OUT_OF_MEMORY` on layer 1 prefetch. Setup log: `4.0 GB total, ~42 slots/layer (estimated 48 layers, 2112000 bytes/slot)`. Workaround: `--expert-cache-gb 2 --pa-context-len 4096` fits (then hits B5). |
| B2 | PLE table lives in host memory | The `per_layer_token_embd.weight` n-gram table is large and not cached on the GPU, so every token does a slow host gather. Visible in R1/R3. **Mitigated** by the parallel gather + `MADV_RANDOM` (W3): 7.06 s → 0.48 s per 2048-token prompt; still disk-bound. |
| B5 | ~~`--expert-cache-gb` produces NaN logits~~ **fixed** (W4, design G8) | `--expert-cache-gb 2 --pa-context-len 4096` fits now (B1 was the 4 GB size), but the first decode after warmup samples NaN, with or without `--prefill-stream`, on the stock release binary too. The unit test `decode_split_matches_running_every_expert_on_the_cpu` passes, so the real-model path differs from it. Logs: `~/bench-rel/bench-cache2-*.log`, `dbg-*.log`. |
| B6 | `serve` OOMs on a 2100-token prompt with the R3 flags (design G9) | `bench` uses `max_seqs = 1`; `serve`'s default reserves more, leaving too little VRAM for the prefill buffers. `--max-seqs 1` works. |
| B3 | Decode at ~7 tok/s (release, after the PLE fix; 3.7 in the debug R1) | Not bus-bound. Needs a per-step profile (design Roadmap item 4). |
| B7 | **RESOLVED by W27**: a 3-4 GB expert cache combines with MTP on `serve` via `--prefix-cache-n 2-3` | The 1887 MB is the recurrent prefix snapshot reservation, `111 MB x (--prefix-cache-n + 1)`, not an activation cost: it looked fixed because none of `--max-num-batched-tokens`, `--max-seqs` or `--max-model-len` touch `--prefix-cache-n`. It is also a lazily materialized worst case (idle VRAM is identical at `--prefix-cache-n` 0 and 16), and the engine now reports the breakdown and the prefix capacity that fits alongside the requested KV cache (W27). |
| B4 | `--prefill-stream` chunk size defaults to 512 | Below the 1024-token streaming threshold (now `--prefill-stream-min-tokens`), so the GPU path silently does nothing unless `--max-prefill-chunk-tokens` is raised (fixed in code at `7895b94`; documented here so config is deliberate). |

## 6. Roadmap

Moved to `design.md` Roadmap, the single roadmap. Items there cite the B-numbers and runs above.
Done since the first version of this plan: release-build measurements, chunk-size sweep and
threshold check (old items 12), the R4 retry at 2 GB (old item 1, which exposed B5), and the
prefill profile (W3).

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
        target/release/naiwa-rs bench ...

    # greedy-output parity between two binaries (short + 2100-token prompt, via serve)
    ~/bench-rel/scripts/parity.sh BIN TAG

    # one offload configuration, measured with real prompts (W11/W20 protocol)
    export PROMPTS=~/bench-rel/prompts/req   # durable; /tmp copies died in a reboot (W29)
    scripts/offload-sweep.sh TAG NP_PREDICT PCIE_FRAC    # e.g. np4-f050 4 0.5 (cache 2)
    #   at cache 4 use frac 0.05: 1 scratch slot/layer is the sweet spot, 3 slots OOM (W26)
    #   serve at a 4 GB cache needs --prefix-cache-n 2..3: the default 16 reserves 1887 MB of
    #   recurrent prefix snapshots, more than the KV cache; the engine now prints the breakdown (W27)
    CACHE_GB=4 scripts/offload-sweep.sh TAG 2 0.15 --prefix-cache-n 2
    #   frac 0.15 = 2 scratch slots/layer is the best measured under serve; 3 slots OOMs at load (W28)

    # bench with real text instead of the synthetic ramp (W25), e.g. to time MTP where serve cannot run
    cp ~/bench-rel/pf/prompt-fib.txt /tmp/p.txt   # the W25/W26 prompt
    naiwa-rs bench -m ... -f '...' --cpu-moe --expert-cache-gb 4 --prompt-file /tmp/p.txt \
        --mmproj mtp-component.gguf --mtp --mtp-n-predict 2 \
        --gen-len 32 --depth 4 --iterations 3 --warmup 1
    #   env: CACHE_GB (default 2), PORT, OUT, PROMPTS (default /tmp/mb)

    # the requant converter (already built; rerun only if the model changes)
    ~/requant/requant ~/models/qwen3-coder/*.gguf ~/models/qwen3-coder/requant/out.gguf --dry-run

    # pull latest on ubi
    cd ~/code/qflash-rs && git pull
