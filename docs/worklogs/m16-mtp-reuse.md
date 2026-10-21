# M16 — chat with MTP: prefix reuse under speculation

Status: **in progress** (2026-10-21). One mechanism this milestone: **a speculative decode that does not
cost cross-turn prefix reuse**, so the per-layer host round trip can be amortised over a multi-row step.
Parent: [m10-chat.md](m10-chat.md).

## Why this, now

M10 left one measured term standing: chat spends **~36 ms/token of host↔device round trip**
(`dry 1.2 ms/layer × 48`, `busy 0.000` — the GPU is idle at every layer entry) on top of a 19 ms GPU chain.
Two things were closed on the way: the CPU miss path is *not* on that path (`--pcie-frac 1.0` removes it with
no TPOT change), and the resident weights are already at the file's cost (4,300 MiB, 1,299 MiB of it BF16 in
the file). So the only lever left is **rows per step**: one round trip per layer per *step*, not per token.

MTP is that lever, and the measurements say it works *when acceptance is high* — but it also sets
`SpeculativePrefixReplay::Full`, and M10 measured prefix reuse at **0.00% with MTP vs 33.33% without** on the
same conversation. On `doc`/`long` that costs 20–30 s of TTFT per turn, which is a bigger term than the decode
gain. Hence this milestone.

## What is established (measured, M10.3 items 3b and 6)

| fact | number |
|---|---|
| TPOT at 25–30% acceptance (`--mtp-n-predict 4`) | 59–62 ms, no better than no-MTP 55 ms |
| TPOT at 41–80% acceptance (same arm, later turns) | 48 → **35.8 ms**, under Strata's 38 |
| expert hit rate with 5 rows | 78–81% (vs 51% at one row) |
| host cost per layer | `dry = 1.00 + ~0.25 × rows` ms (1 row 1.21, 5 rows 2.27) |
| `--pcie-frac 1.0` with MTP | worse: 2 668 MiB staged per window vs 848 → PCIe saturation |

So MTP's gain is `48 × dry / accept_len`, i.e. it is fighting a host term that grows with rows.

## Root cause of the 0% reuse (traced)

`MISTRALRS_PREFIX_DBG=1` (fork `bea0e450`) shows the constant kills **capture as well as restore**, which
was not visible from the call site:

```
replay_for = Full
  → clamp_speculative_prefix_cache_hit(cached, 32, Full) = 0
  → recurrent_checkpoint_boundary(...) = None
  → build_prompt_chunk_plan(..., replay, ...) receives no boundary → the prompt stays one chunk
  → snapshot_paged_recurrent_prefix is never called (not one `snapshot` line)
  → every restore prints `no_replay_boundary` and stages a recurrent reset
```

| arm | snapshots | restore | prefix hit rate |
|---|---|---|---|
| no MTP | seq 2 @32, seq 3 @96 | `no_replay_boundary cached=0` | **33.33%** |
| MTP, `Full` | none | `cached=0/32/96` | 0.00% |
| MTP, `MISTRALRS_MTP_PREFIX_REPLAY=none` | seq 2 @32, seq 3 @96 | `cached=0` | **33.33%** |

The override (`full|none|suffix:<n>`, default `Full`) is the fork's `builtin_mtp_prefix_replay()`;
`qwen4_exp` is the wrapper this model uses, not `qwen3_5`.

**Why `Full` is at least partly over-conservative.** The documented hazard is that a block hash covers target
tokens only, while a block written during a speculative step can hold draft rows in its tail that later become
interior. But capture is prompt-only (`is_prompt && !scheduler_visible_prompt_step`) and the boundary is
strictly inside the prompt, where every row is a committed target token; the drafts live in the drafter's own
paged cache (`drafter_prefill_chunk`, placeholder row `0`). The hazard as stated applies to a capture path this
build does not take.

## Two determinism sources (found, must be controlled before any gate means anything)

| source | effect | control |
|---|---|---|
| CPU vs GPU expert path | which runs depends on the **lookahead's timing**, and the two reduce in different orders → the same greedy prompt differs run to run (90 vs 102 tokens), even with `NAIWA_CPU_MOE_THREADS=1` | `--no-lookahead --pcie-frac 1.0` |
| prompt length | boundary is `align_down(total_len - 1, 32)`, so a **≤32-token prompt has no boundary at all** | item 1 below |

The second is also the standing `short` TTFT puzzle (M10 open question): those turns cannot checkpoint, so
their cross-turn reuse is 0 for a reason unrelated to MTP.

## Work items

1. **A prompt that ends on a block boundary must be checkpointable.** `total_len - 1` excludes the end of the
   prompt, so a one-block prompt (a lot of `short` chat turns) has no boundary. Gate: a 40-token prompt shows
   `snapshot ... tokens=32` (already true) *and* a 32-token prompt shows `tokens=32` (today: nothing), with the
   same output as without reuse.
2. **Deterministic equivalence gate for `=none`.** `--no-lookahead --pcie-frac 1.0`, prompt ≥ 70 tokens sent
   three times, arms `full` vs `none`. Gate: a `snapshot` line exists, and the arms are byte-identical.
3. **Promote MTP into the chat arm** only after 1–2: `harness/chat/engines/naiwa.json` gains
   `--mtp --mmproj … --mtp-n-predict 4` plus the replay override. Gate: full 3-round corpus, `doc`/`long`
   TTFT≥2 within the M10 baseline (3.67 s / 6.13 s turn 1) *and* TPOT ≤ 38 ms on at least one class.
4. **Flatten the row-proportional host term** (`0.25 ms/row/layer`), which is what makes the amortisation
   pay at all: `dry` must stop scaling with rows. This is M10 item 3b's device-side dispatch, re-aimed at np5.

Non-goals here: KV int8, host-mapped embedding, the prefill blob-ring. Those are memory/`e2e` terms, tracked
in M10.

## Results

### item 1 — aligned prompts checkpoint (done)

`recurrent_checkpoint_boundary` now takes `align_down(total_len, 32)` and accepts `boundary <= total_len`.
Oracle re-verified after the change, both variants byte-identical to `harness/m1/reference.oracle.txt`
(`frac 0 --no-lookahead`, with and without `--prefill-stream`); the oracle runs `--prefix-cache-n 0`, so this
path is inactive there by construction.

### item 2 — the `=none` equivalence gate (done, passing)

`--no-lookahead --pcie-frac 1.0` (deterministic), MTP on, 96-token prompt sent three times:

| arm | `PREFIX_DBG` | three responses |
|---|---|---|
| `none` | `snapshot: seq 1 tokens=64 blocks=2`, restore never reports `no_replay_boundary` | `09cc90a793cd` ×3 |
| `full` | no snapshot, `no_replay_boundary cached=64` twice | `09cc90a793cd` ×3 |

So reuse demonstrably happens with `none`, does not with `full` (the must-fail control), and the outputs are
byte-identical either way.

### item 3 — MTP in the chat arm (measured, gate partly met)

`harness/chat/engines/naiwa.json` now carries `--mtp --mtp-n-predict 4` and
`env: {MISTRALRS_MTP_PREFIX_REPLAY: none}`; `run.py` learned an `env` passthrough and prints it in the launch
line so a run reproduces from its own log. 3 rounds, `short`+`doc`+`long`, naiwa only (medians):

| class | TTFT t1 | TTFT ≥2 p50 | TPOT p50 | reuse p50 | e2e/conv | M10 baseline (no MTP) |
|---|---:|---:|---:|---:|---:|---|
| doc | 9 528 ms | 4 396 ms | **43 ms** | **0.96** | **50.7 s** | 6 060 / 3 670 / 55 / 0.83–0.94 / 52.1 |
| long | 9 084 ms | 3 560 ms | **44 ms** | **0.89** | **46.6 s** | 6 130 / 3 800 / 55 / 0.83–0.94 / 53.5 |
| short | 3 050 ms | 4 443 ms | **52 ms** | **0.76** | **60.5 s** | — / 4 570 / 55 / — / 63.6 |

**What the milestone was for:** reuse is back (0.76–0.96 vs **0.00% with MTP** before) and TPOT drops
55–56 → 43/44/52 ms, i.e. −22%; end-to-end is better on all three classes.

**What it did not get:** the gate asked for TPOT ≤ 38 ms on one class (got 43) and `doc` TTFT≥2 within the
baseline (got 4 396 vs 3 670 ms). Two separate causes, both now measured:

| new cost | number | where |
|---|---|---|
| turn-1 prefill with MTP | 6.06 → **9.53 s** (284 → 181 tok/s) on `doc` | the drafter prompt path: the chunk plan still picks 8192 (log confirms), so it is not chunking; candidate is the speculative prompt chunk forcing the unpacked attention policy |
| row-proportional host term | `dry = 1.00 + 0.25 × rows` ms/layer | item 4: at 5 rows the round trip is only half amortised, and it is what keeps TPOT at 43 instead of ~20 |

## Turn-1 prefill: decomposed (2026-10-21)

A 1,725-token `doc` turn 1, one request, `--prefill-stream --prefill-stream-min-tokens 64`, MTP on, timed through
`PREFIX_DBG` (`prefill:` per chunk, `prefill_step:` per step, `drafter_prefill:` inside the drafter):

| piece | time | rate | note |
|---|---:|---:|---|
| target forward, 1,696 rows | 3.50 s | 485 tok/s | same as the no-MTP arm (3.48 s) — unaffected by MTP |
| **drafter prefill, 1,696 rows** | **2.73–4.01 s** | 424–620 tok/s | `mtp_forward`; embed 6 ms, hidden/positions µs |
| tail chunk, 29 rows | 2.32–2.63 s | ~12 tok/s | below `--prefill-stream-min-tokens`, so it is a CPU prefill; the **no-MTP arm pays this too** |
| total | 9.0–10.4 s | 166–190 tok/s | vs the no-MTP 6.1 s (3.48 + 2.3) |

Three separate causes, and the accounting matters because they are paid differently:

1. **The drafter prefill is a CPU MoE, once per conversation.** The MTP head's experts are always loaded to the
   host (`moe/provider.rs`: `if ctx.site.is_mtp` returns `CpuExperts` before `attach_prefill` runs, so it never
   gets a prefill stream either — `prefill_stream.rs::attach` also excludes `is_mtp`). For a prompt chunk it
   falls through to `MoEExperts::forward` → `FastExpertsWeights::forward_gather`, which on the CPU pays one
   weight-block read **per (token, expert) pair**: 1 696 × 8 × ~2.6 MB ≈ **35 GB** at the ~9–10 GB/s the
   one-row kernel gets. That is the whole 3–4 s. A grouped CPU kernel (one GEMM per expert, ~26 rows each)
   would read the same 1.35 GB once: **≈ 0.2 s**.
2. **It is paid once, not per turn.** With reuse enabled, a repeated request restored 1 696 tokens and its
   drafter prefill covered only the 29 new rows (`mtp_forward` 50–65 ms). So this is a cold-conversation cost:
   −4 s on `e2e/conv` for `doc`/`long`, not on TTFT≥2.
3. **The 29-row tail is a pre-existing defect, not MTP's.** `align_down(total, 32)` splits 1 725 as
   1 696 + 29, and a 29-row chunk touches ~all 128 experts of every layer, so it is expensive either way
   (CPU 2.63 s at `min-tokens 64`; stream 2.32 s at `min-tokens 8` — 128 experts × 1.88 MB × 48 layers ≈
   11.5 GB of PCIe). It exists in the no-MTP arm too and is the bulk of the M10 "TTFT≥2 3.67 s" number.

## Drafter prefill: the grouped CPU kernel was tried and is *not* the answer (2026-10-21)

The plan was a grouped CPU kernel: `gemv` groups by (expert, activation row), so an expert serving 33 rows
has its block read 33 times — 45 GiB of reads for 1.35 GiB of unique weights. A variant that groups by expert
only (one read, exact f32 dots, 4-way unrolled, rayon over experts) was implemented behind the same dispatch
and pinned by a new test against the dequantized reference (Q4_0/Q8_0/Q4K/Q6K, all within 1e-4).

It made the prefill **slower**, 2.73 s → **5.82 s**, which settles what the kernel is actually bound by:

| kernel | reads | arithmetic | time | rate |
|---|---:|---:|---:|---|
| per-slot `gemv` (grouped by (expert, row)) | 45 GiB | 167 GFLOP | 2.73 s | 17.7 GB/s **and** 61 GFLOPS (int8 `vec_dot`) |
| grouped by expert (one read, f32 dots) | 1.35 GiB | 167 GFLOP | 5.82 s | 28.6 GFLOPS — compute-bound |
| to reach 0.2 s | — | 167 GFLOP | — | **834 GFLOPS** — out of reach on 6 cores |

The drafter's prompt prefill is **167 GFLOP for one layer** (1696 rows × 10 experts × 3 matrices × 640 ×
2560). For scale: the *whole* 48-layer target prefill is 32 TFLOP and takes 3.5 s, i.e. **9.1 TFLOPS on the
GPU**. So the drafter prefill belongs on the GPU by two orders of magnitude — the earlier "grouped CPU kernel"
recommendation was wrong, and the code was reverted (`mistralrs-quant/src/gguf/cpu.rs` is unchanged).

What the GPU path needs, from reading the stream (`src/prefill_stream.rs`):

1. `moe/provider.rs`: the `is_mtp` early return keeps the head out of `attach_prefill`; route it through the
   same 0-slot path (`CpuExperts` + prefill stream) so decode still runs on the CPU.
2. `prefill_stream.rs::attach`: the `ctx.site.is_mtp` exclusion.
3. The stream assumes one expert count: `PrefillStream::new` stores `num_experts`/`top_k` from the first
   attached layer and `MoEExperts::from_qtensors` is built with them, so a 512-expert head behind a 128-expert
   model needs the counts stored per layer. The buffer is allocated per layer via `alloc_like`, so it has to be
   re-allocated when the MTP layer's stride (1.35 GiB) exceeds the main layers' (650 MiB), and
   `prefill_stream::plan` must reserve `max(stride)` rather than the main-layer stride.

## item 5 — the drafter's experts resident in Q2_0 (done, 2026-10-21)

Implemented as `src/moe/resident.rs`: the MTP head's 512 experts are re-quantized at load to **Q2_0**
(2.25 bit/weight, 1,382,400 bytes per expert, **675 MiB** for the layer — byte-for-byte the size and format
strata reserves for its `MtpDrafter`) and kept on the GPU, computed through the ordinary expert kernels for
both phases. `provider.rs`'s `is_mtp` branch builds that pool instead of returning `CpuExperts`; the CPU
path remains as the fallback for non-`QTensor` weights. The host copies are dropped afterwards, which also
returns 1.35 GiB of RAM.

Two details worth recording:

1. **candle cannot quantize Q2_0** (`GgmlDType::cpu_zeros` is `unreachable!` for it) and its
   `QStorage::from_data` refuses Q2_0 on CUDA, so the packing lives here
   (`quantize_q2_0`, pinned by a test that round-trips through the fork's
   `gguf::q2_0::dequantize`), and the upload goes through the fork's `q2_0::storage_for_device`.
2. **The naive block-max scale wastes the format.** Codes decode to `{-1, 0, +1, +2} x d`, so a block-max
   `d` spends one code on an unreachable range and quantizes half the block to zero. The scale is searched
   over `amax / {1.0, 1.25, 1.5, 1.75, 2.0}` per block, keeping the smallest squared error. That is what
   decides acceptance: 50.8% with the Q4_0 drafter, 44.2% with the naive Q2_0 pack, **51.6%** with the
   ladder — i.e. the format is free at the acceptance level once the scale is chosen properly.

| measurement | before (Q4_0 drafter, CPU) | after (Q2_0, resident) |
|---|---:|---:|
| drafter prefill, 1,696 rows (`mtp_forward`) | 2.73–4.01 s | **10.4 ms** |
| drafter forward, decode (`NAIWA_MTP_DBG head_us`) | 1.0–2.4 ms | **0.55 ms** |
| drafter weights: VRAM / host | 0 / 1.35 GiB | **675 MiB / 0** |
| `MTP accept` (3 rounds) | 50.8% | 51.6% |
| output vs the no-MTP arm (deterministic config, 3 requests) | `09cc90a793cd` | `09cc90a793cd` — identical |

3 rounds, `short`+`doc`+`long`, naiwa only, medians, cache 2.3 GB:

| class | TTFT t1 | TTFT ≥2 | TPOT p50 | reuse | e2e/conv | before this item |
|---|---:|---:|---:|---:|---:|---|
| doc | **6 266 ms** | 4 765 ms | **40 ms** | 0.95 | **48.5 s** | 9 528 / 4 396 / 43 / 50.7 |
| long | **6 092 ms** | 3 765 ms | 47 ms | 0.90 | 47.7 s | 9 084 / 3 560 / 44 / 46.6 |
| short | 3 015 ms | **2 819 ms** | 52 ms | 0.73 | **53.3 s** | 3 050 / 4 439 / 52 / 60.5 |

**The 675 MiB has to come out of the main cache.** With `--expert-cache-gb 3` the pool left too little free
VRAM, `prefill_stream::plan` declined, and the whole main prefill fell back to the CPU (13 tok/s, 133 s for
one turn). `engines/naiwa.json` now uses **2.3 GB**: the log's plan line confirms chunk 8192 on 1 984 MiB
free. This is the cheap end of the trade because M10 measured main-cache size 1 → 3 GB as TPOT-neutral at
chat.

## item 6 — the prompt tail belongs to the decode split (done, 2026-10-21)

The checkpoint boundary is `align_down(prompt, 32)`, so **every** prompt is prefilled as
`[0, boundary)` plus a remainder of **at most 31 rows**. At `PREFILL_THRESHOLD = 16` that remainder was
wider than the threshold and took the whole-chunk path — CPU, or the prefill stream — where a handful of rows
still touches nearly every expert of every layer (128 x 3.9 MiB x 48 = 24 GB of reads for 29 rows). Measured
2.3 s per `doc`/`long` turn.

`PREFILL_THRESHOLD` is now **32**, so the tail takes the decode split (cache hits on the GPU, staging for
`--pcie-frac`, the rest on the CPU) — and the cache has just been warmed by the preceding chunk, which is why
it is cheap. Two things had to change with it:

- `MAX_DECODE_ROWS` follows the threshold, and the **lookahead channel is sized for 5 rows** (the widest
  verify window). A 29-row chunk made `LookaheadChannel::publish` bail with *"lookahead publish of 74240 f32
  exceeds the channel slot (40960)"*, which failed the whole request. The publish is now skipped when the
  chunk is wider than the slot: a 5-row prediction could not cover a chunk's routing anyway, and the chunk's
  own hit counts fill the cache through `spawn_fill`.
- Oracle re-verified after the change (the 17-token oracle prompt now takes the decode path instead of the
  chunk path): byte-identical, both variants.

| measurement | before | after |
|---|---:|---:|
| `doc` turn-1 TTFT | 6 266 ms | **4 400 ms** |
| `doc` tail chunk (29 rows) | 2.3 s | **0.81 s** |
| `short` turn-1 TTFT | 3 015 ms | **1 076 ms** |

3 rounds, medians, naiwa only, cache 2.3 GB:

| class | TTFT t1 | TTFT ≥2 | TPOT p50 | reuse | e2e/conv | M16 start |
|---|---:|---:|---:|---:|---:|---|
| doc | **4 400 ms** | 2 984 ms | 42 ms | 0.96 | **40.7 s** | 9 528 / 4 396 / 43 / 50.7 |
| long | **4 300 ms** | 3 106 ms | 44 ms | 0.89 | **42.0 s** | 9 084 / 3 560 / 44 / 46.6 |
| short | **1 076 ms** | 2 792 ms | 48 ms | 0.74 | **53.8 s** | 3 050 / 4 439 / 52 / 60.5 |

Whole milestone so far: `doc` turn-1 9.5 → 4.4 s, TTFT≥2 4.4 → 3.0 s, e2e 50.7 → 40.7 s; TPOT 43 → 42 with
`doc` at 40–42 ms against strata's 39.

## The mid-size chunk is a bandwidth floor, not a threshold (measured, 2026-10-21)

The remaining `TTFT>=2` gap (3.0 s vs strata's 0.63 s) has the same shape as item 6 one size up: a turn's new
tokens are ~70-200 rows, so they take the chunk path and must read ~all 128 experts of every layer
(128 x 3.9 MiB x 48 = 24 GB). The obvious knob is the threshold: route those rows through the decode split,
which reads only what the cache does not hold.

`PREFILL_THRESHOLD` 32 -> 256, same 3-turn `doc` conversation on one binary each:

| threshold | turn 1 | turn 2 (86 new rows) | turn 3 | `ROUND_STATS` hit / staged / cpu |
|---|---:|---:|---:|---|
| 32 | 4.57 s | **4.11 s** | 3.83 s | 68.8 / 9.9 / 21.4% |
| 256 | 4.59 s | **4.08 s** | 3.86 s | 66.3 / 10.7 / 23.0% |

**No change, and no wonder**: the hit rate barely moves (the cache holds ~12 slots/layer against 128 experts,
so ~90% of the *distinct* experts are still cold), and the miss side then pays the same 24 GB — on the CPU
instead of over PCIe (measured 10 GB/s vs 13.4). The threshold only moves traffic between two paths that are
both bounded by the same read. Reverted; the constant stays at 32.

Closing that gap requires the conversation's experts to be resident (cache size, or a resident set that
survives the turn) or a faster path than either — not a dispatch knob.

**Correction to the numbers above, and to an earlier note in this file**: the model has **256 experts per
layer** (`qwen4exp.expert_count = 256`, `expert_used_count = 10`, `expert_feed_forward_length = 640`), not 128.
An expert is `500 MiB / 256 = 1.95 MiB`, which is strata's `1028 slots, 1.98 GiB` -> 1.93 MiB per slot to
within rounding — so there is no 2x slot-size difference to chase. The totals in this file stay right because
they are per *layer* (expert count x per-expert bytes = 500 MiB either way): a full-layer read is still 24 GB
for 48 layers, our 2.3 GB cache is ~1 178 slots (~24.5/layer, 9.6% of 256), and strata's 1 028 is ~21/layer ""(8.4%).

## Gates

- Oracle unchanged: `frac 0 + --no-lookahead` byte-identical to `harness/m1/reference.oracle.txt`, with and
  without `--prefill-stream`. MTP is *not* in the oracle path (it is a chat-only arm), so item 3 additionally
  needs its own equivalence gate (item 2).
- One mechanism per run; medians of ≥ 3 turns; failed rounds are recorded, not retried.
- Status of the three gates, 2026-10-21: oracle ✅ (both variants); reuse equivalence ✅ (item 2 table);
  chat corpus ⚠️ partly (TPOT −22% and e2e better on all classes, but TPOT > 38 ms and `doc` TTFT≥2 above
  baseline — see the two costs above). Item 4 is what closes it.

## Next

1. ~~Grouped CPU MoE for prefill chunks~~ — **falsified by measurement**, see the section above: the kernel is
   compute-bound (167 GFLOP), so the drafter prefill has to run on the GPU. Next item is that GPU path, which
   needs the three changes listed there.
2. **Item 4, the row term**: `dry` must stop scaling with rows, or MTP's amortisation stays capped at
   `1.0/(1.0+0.25·R)`. This is the M10 3b device-dispatch work, now with a measured payoff curve.
3. ~~The drafter's experts~~ — done above; resident Q2_0 removes the drafter from both phases.
4. ~~The 29-row tail~~ — done above.
5. ~~The remaining TTFT≥2 gap via the threshold~~ — measured, no effect (see above); it is a read floor.
   The open lead there is strata's **1.93 MiB per expert slot against our 3.9 MiB** for the same model.
6. Re-run the 3-round **paired** corpus against strata (this item was measured naiwa-only).
