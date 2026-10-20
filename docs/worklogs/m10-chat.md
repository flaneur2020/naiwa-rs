# M10 — multi-round chat: naiwa-rs vs Strata

Status: **M10.1–2 done, M10.3 items 1–2 done (2026-10-19/20)**. Next: "Next (M10.3 items 3–6)" below —
decode (TPOT 1.6–2.2×) is the open term.
Parent: [m9c.md](m9c.md). This is the roadmap's **M10** (`docs/roadmap.md` §M10).
(`worklogs/m10.md` is the older *MoE naming and ownership* milestone from the pre-rescope numbering; it is
unrelated and stays.)

Scope: the **measurement**, not the alignment. Harness: [`harness/chat/`](../../harness/chat/README.md).

## Why rounds, not another microbenchmark

Every number in `design.md` §5 is a single request at `--depth 4`: MoE-dominated, ~no attention, no prefix reuse.
A chat session is the opposite. Turn *n*'s prompt is turns 1..*n*'s transcript, so

- (a) prompt processing grows every turn,
- (b) almost all of that prompt is a prefix the engine already holds,
- (c) per-turn TTFT is decided by prefix reuse and prefill batching, not by decode throughput.

**The measured unit is a conversation, not a request.** That is also the only shape in which a parity claim
means anything: Strata quotes single-shot "writes answers" / "reads your prompt" numbers, and long chats are a
separate table in its own docs.

## Feasibility on this box (checked, then used)

| fact | evidence |
|---|---|
| Strata runs on the same card | `~/code/strata`, RTX 20–50 support; `engine/strata` + `.venv` present |
| same model artifact | both load `~/models/qwen3-coder/IQ1_M/…-0000{1,2}.gguf`; strata additionally needs `~/code/Strata-data/packs/coder-iq1_m` |
| one client view of both | both are OpenAI-compatible and both report `prompt_tokens_details.cached_tokens` (strata maps its `cache_n`/`reused`) |
| thinking pinned off on both | `enable_thinking: false` (naiwa) / `reasoning_effort: "none"` (strata) |
| engines cannot co-reside | 12 GB VRAM / 60 GB RAM; sequential, PID-only stop, order alternated per round |

## Protocol

### Corpus (`harness/chat/corpus.jsonl`, committed, hash in `run.json`)

One line = one conversation: `{"id","class","system","turns":[{"user","max_tokens"}]}`. `{{DOC}}` on line 1's
`_constants` expands the 1.7k-token document used by `doc` and `long`.

| class | shape | isolates |
|---|---|---|
| `short` | 8 turns × 96 tok | per-turn TTFT with reuse; decode |
| `doc` | 1.7k-token document, then 6 questions | prefill + prefix reuse dominate |
| `code` | 4 turns × 160 tok | long outputs; draft acceptance |
| `long` | 1.7k turn, then 2 × 400 tok | KV growth near the context limit (M10.4) |

**The transcript grows with each engine's own streamed answer**, exactly as a chat client does.
A fixed transcript of *authored* assistant turns cannot measure reuse: mistralrs snapshots the
recurrent prefix at the end of a **prompt**, so a request whose shared prefix ends in text the engine never
generated never matches. Measured both ways: authored transcripts → zero reuse on naiwa for seven turns
(`--prefix-cache-n 1` and `2`); live answers → the same engine still reports zero, and strata reports 0.95–0.99
(see findings 1 and 3). Turn 1 is identical on both engines, which is where prompt-token equality is checked
(it was: `short` turn 1 = 40 tokens on both, `doc` turn 1 = 1723 on both).

### Runs

- **Round** = cold engine process, wait for `/v1/models`, one discarded 2-turn warmup conversation, every
  conversation in the corpus, stop by PID. Order alternated per round.
- `short`+`code`: **3 rounds**, both engines. `doc`+`long`: **1 round**, both engines — naiwa's `doc` cost
  1,136 s per conversation, so three rounds would have been an hour of pure re-prefill.
- Concurrency **1**: strata serves one sequence behind a FIFO, so any other number is not like-for-like.
- VRAM/RSS sampled at 1 Hz.

### Per-turn metrics (from the SSE stream, not engine logs)

TTFT (request write → first content delta); **TPOT** (decode window / gaps — a speculative decoder delivers
several tokens per step, so an ITL *median* reads 0); decode tok/s; end-to-end; `prompt_tokens`,
`completion_tokens`, `cached_tokens`, `reuse = cached/prompt`. Turn 1 has nothing to reuse, so the headline is
turns ≥2, and the per-turn curve is the deliverable.

### Pinned configuration (both sides, recorded per run)

| | naiwa-rs | strata |
|---|---|---|
| model | IQ1_M native, both shards | same shards via `packs/coder-iq1_m` |
| context | `--pa-context-len 4096` | `--max-context 4096` |
| KV | BF16 paged, 32-token blocks | `--kv int8` |
| expert cache | `--expert-cache-gb 1.5` | `--expert-cache 800` blobs (~1.5 GB) |
| speculation | `--mtp --mtp-n-predict 4` | `--spec 4 --spec-min-p 0.5` |
| `pcie_frac` | 0.2 | its own startup probe picked 0.37 |
| prefix reuse | `--prefix-cache-n 1` | one resident sequence + conversation checkpoint |
| peak VRAM / RSS | 11.55 GiB / 56.0 GiB | 7.04 GiB / 26.5 GiB |

Strata's `--expert-cache auto` fills the card and is not a comparison, so it is pinned to naiwa's budget.
**Speculation cannot be paired off**: strata refuses to start a native (IQ) pack without `--spec T, T >= 2`
("it needs `--native SHARD1, --spec T (T >= 2)`"), so the paired arm has speculation on both sides.
`--prefix-cache-n` on this model is a **recurrent state snapshot of ~111 MB**, not a few KV blocks: the default
16 needs 1.9 GB and does not fit beside a 1.5 GB cache at 4096 context. Pinned to 1.

## M10.2 baseline — 2026-10-19/20

Three arms, concurrency 1, thinking off, medians. Raw runs: `~/bench-rel/m10/2026-10-19` (arm A,
3 rounds short/code), `2026-10-19-long` (arm A, 1 round doc/long), `2026-10-20-nospec` (arm B,
3 rounds short/code + 1 round doc/long). 138 turns, 0 failed turns.

| arm | naiwa | strata | why |
|---|---|---|---|
| **A** paired | `--mtp --mtp-n-predict 4`, 1.5 GB cache | `--spec 4`, 800 blobs | each engine's shipped speculation; strata requires it for a native IQ pack |
| **B** no speculation | no `--mtp`, 2 GB cache | *(cannot run)* | measures what speculation costs a chat session. Superseded by the stream arm below (`engines/naiwa.json`) |

```bash
python3 harness/chat/run.py --engines naiwa,strata   --rounds 3 --classes short,code --out ~/bench-rel/m10/2026-10-19
python3 harness/chat/run.py --engines naiwa,strata   --rounds 1 --classes doc,long   --out ~/bench-rel/m10/2026-10-19-long
python3 harness/chat/run.py --engines naiwa-nospec   --rounds 3 --classes short,code --out ~/bench-rel/m10/2026-10-20-nospec
python3 harness/chat/run.py --engines naiwa-nospec   --rounds 1 --classes doc,long   --out ~/bench-rel/m10/2026-10-20-nospec
```

| class | engine | TTFT t1 | TTFT ≥2 p50 | p90 | TPOT p50 | decode tok/s | reuse p50 | e2e/conv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `short` | naiwa A | 3,095 ms | 40,630 ms | 66,333 ms | 59 ms | 17.0 | **–** | 333.2 s |
| `short` | naiwa B | 3,081 ms | **11,389 ms** | 13,195 ms | 57 ms | 17.5 | **0.73** | 121.1 s |
| `short` | strata | 1,030 ms | **614 ms** | 809 ms | 38 ms | 26.1 | **0.95** | 29.1 s |
| `code` | naiwa A | 3,041 ms | 31,950 ms | 36,524 ms | 40 ms | 25.0 | – | 106.6 s |
| `code` | naiwa B | 2,852 ms | 16,399 ms | 19,053 ms | 58 ms | 17.2 | 0.47 | 72.6 s |
| `code` | strata | 985 ms | 1,150 ms | 1,287 ms | 27 ms | 36.5 | 0.67 | 21.2 s |
| `doc` | naiwa A | 136,060 ms | 164,196 ms | 175,027 ms | 51 ms | 19.5 | – | 1,136.4 s |
| `doc` | naiwa B | 151,782 ms | **10,846 ms** | 12,368 ms | 60 ms | 16.7 | **0.94** | 238.5 s |
| `doc` | strata | 2,321 ms | **620 ms** | 710 ms | 39 ms | 25.9 | **0.99** | 22.1 s |
| `long` | naiwa A | 135,588 ms | 182,142 ms | 192,893 ms | 52 ms | 19.3 | – | 550.5 s |
| `long` | naiwa B | 150,530 ms | 13,377 ms | 15,123 ms | 60 ms | 16.8 | 0.92 | 188.6 s |
| `long` | strata | 2,251 ms | 777 ms | 827 ms | 38 ms | 26.6 | 0.99 | 34.7 s |

Per-turn TTFT, `short`, all three arms (turn 1 is the same prompt everywhere):

| turn | naiwa A | naiwa B | strata | reuse A / B / strata |
|---:|---:|---:|---:|---|
| 1 | 3,095 ms | 3,081 ms | 1,030 ms | – / – / 0.00 |
| 2 | 12,929 ms | 11,094 ms | 811 ms | – / 0.19 / 0.80 |
| 3 | 23,794 ms | 11,270 ms | 609 ms | – / 0.56 / 0.92 |
| 4 | 35,936 ms | 13,285 ms | 632 ms | – / 0.63 / 0.94 |
| 5 | 45,064 ms | 9,037 ms | 590 ms | – / 0.78 / 0.96 |
| 6 | 55,204 ms | 5,410 ms | 660 ms | – / 0.89 / 0.95 |
| 7 | 66,333 ms | 13,284 ms | 572 ms | – / 0.78 / 0.96 |
| 8 | 76,894 ms | 5,754 ms | 616 ms | – / 0.89 / 0.97 |

## M10.3 baseline — 2026-10-20 (after items 1 and 2, 3 rounds, all classes)

Same corpus and protocol as M10.2; `engines/naiwa.json` is now the stream arm (no MTP, 1 GB cache),
strata unchanged. 132 turns, 0 failed turns, 0 OOMs. Arm `naiwa-mtp` (MTP 4, 1.5 GB, no stream) keeps
its M10.2 numbers: speculation and the stream cannot both fit on this card.

| class | engine | TTFT t1 | TTFT ≥2 p50 | p90 | TPOT p50 | decode tok/s | reuse p50 | e2e/conv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `short` | naiwa | 3,026 ms | **3,231 ms** | 5,079 ms | 61 ms | 16.5 | 0.78 | 67.0 s |
| `short` | strata | 1,043 ms | 635 ms | 817 ms | 39 ms | 25.6 | 0.95 | 28.8 s |
| `code` | naiwa | 2,969 ms | 4,287 ms | 4,700 ms | 61 ms | 16.4 | 0.47 | 43.8 s |
| `code` | strata | 975 ms | 1,160 ms | 1,170 ms | 28 ms | 35.2 | 0.67 | 21.6 s |
| `doc` | naiwa | **11,881 ms** | **4,403 ms** | 5,037 ms | 63 ms | 16.0 | 0.94 | 64.1 s |
| `doc` | strata | 2,400 ms | 628 ms | 731 ms | 38 ms | 26.1 | 0.99 | 24.6 s |
| `long` | naiwa | 11,844 ms | 2,832 ms | 4,076 ms | 60 ms | 16.6 | 0.83 | 83.6 s |
| `long` | strata | 2,244 ms | 1,003 ms | 1,299 ms | 38 ms | 26.2 | 0.89 | 39.7 s |

Against the pre-fix arm B: `short` TTFT ≥2 **11.4 → 3.2 s**, `code` 16.4 → 4.3 s, `doc` 10.8 → 4.4 s,
`long` 13.4 → 2.8 s; `doc` turn 1 **151.8 → 11.9 s**; e2e per conversation 4–11× → 1.9–2.8×.
TPOT edged up 57 → 61 ms: 0.5 GB of the expert cache went to the stream buffers.

## M10.3 item 1 — resolved: two configuration causes, no code change

**1. `--mtp` disables cross-turn prefix reuse by policy.** `BUILTIN_MTP_PREFIX_REPLAY =
SpeculativePrefixReplay::Full` (`mistralrs-core/src/speculative/builtin_mtp.rs:32`, with the fork's
own comment: "target-only block hashes can alias MTP rows with different next-token inputs,
including interior blocks"), so `effective_recurrent_prefix_boundary` returns 0 for any sequence
with drafts and every turn replays the whole prompt. Measured: arm A reports **no** `cached_tokens`
on any turn of any class (21/21 `short` turns ≥2); arm B, same harness and same card, reuses
0.19 → 0.89 over eight turns. It is a correctness choice, not a bug — but it is a *cost that was
never priced*: on `short`, `doc` and `long` MTP also buys **no decode speed** (TPOT 59 vs 57 ms,
51 vs 60 ms, 52 vs 60 ms), so it is strictly worse for chat. Only `code` pays for itself on decode
(TPOT 40 vs 58 ms, +45%) and still loses end-to-end (107 vs 73 s) because of the lost reuse.

**2. A checkpoint needs a block-aligned prompt *step*.** `snapshot_paged_recurrent_prefix` bails
when `num_computed_tokens % block_size != 0` (`mistralrs-core/src/engine/mod.rs`, the prompt-step
snapshot loop), and the query backs off a further block. With the default
`--max-num-batched-tokens 4096`, a ≤4k prompt is one step whose length is essentially never a
multiple of 32 → no checkpoint → the next turn starts from zero. Capping the step at one PA block
(`--max-num-batched-tokens 32`, `--max-prefill-chunk-tokens 512`) creates a checkpoint at every
boundary, costs nothing measurable on this CPU-bound prefill (11–13 tok/s either way), and turns
the 1.9k-token `doc` turn from 141.9 s into 5.6 s (cached 1,856 of 1,920); a 641-token turn goes
from 56.5 s to 2.2 s (cached 640).

Both are now in the arm-B configuration and recorded per run. The harness keeps `--mtp` in arm A
because that is the shipped configuration; **arm B is the configuration to ship for chat**.

### M10.3 item 2 — done: prefill runs on the GPU

Serve-mode prefill was **11–13 tok/s** on the CPU experts regardless of `--prefill-stream`; it is now
**147–285 tok/s** and `doc` turn 1 goes **151.8 s → 11.9 s**. Two independent defects, both measured:

1. **The lender does not return what it reports.** `PrefillStream::borrow` asks the expert cache to
   lend `2 × largest layer` (1300 MiB) and the lender credits 1107–1863 MiB, but device free memory
   does not move: at a 2 GB cache, `free before lend 770 MiB → free after lend 770 MiB → 1058 MiB
   after a context synchronise + `cuMemPoolTrimTo``, i.e. **288 MiB of 1863**; at a 1 GB cache it
   credits 1107 and returns **0** (2018 → 2018 MiB). So the stream's 1.3 GB of layer buffers has to
   come from VRAM that is free *before* it starts, not from the cache.
2. **candle's pool keeps freed blocks by size.** A dropped tensor goes back to CUDA's stream-ordered
   pool, which reuses it only at the same size; a mixed-quant GGUF changes expert shape per layer
   group, so reallocating a whole-layer buffer per shape accumulated one ~500 MB block per shape and
   OOM'd. `copy_stream::trim_device_pool` (context synchronise, then `cuMemPoolTrimTo(0)`) is now
   called at both hand-overs — the lend and each mixed-quant reallocation. The new unit test
   `pool_trim_returns_freed_tensors` pins that behaviour: a 400 MiB tensor's memory is gone from
   `mem_get_info` until the trim runs.

The working configuration on this 12 GB card, now in `engines/naiwa.json`:

```
--expert-cache-gb 1                 # leaves ~2 GB free; 2 GB leaves 770 MiB and OOMs
--prefill-stream --prefill-stream-min-tokens 64
--max-num-batched-tokens 512 --max-prefill-chunk-tokens 512
```

Measured prefill with it (single request, `max_tokens 1`): 108 tokens 2.8 s (39 tok/s), 945 tokens
6.4 s (**147 tok/s**), 1875 tokens 6.6 s (**285 tok/s**). The upload is per *chunk* — every chunk
streams all 48 layers, ~31 GB — so small chunks pay it repeatedly; 64 tokens is the smallest chunk
where the stream still wins, and 512 both amortises the upload and keeps every checkpoint boundary
block-aligned (item 1).

Two limits, measured rather than assumed:
- **Stream and MTP are mutually exclusive on 12 GB.** At a 1 GB cache, `--mtp` + the stream loads and
  then OOMs on the first request; at 1.5 GB the stream cannot fit at all. So the paired arm keeps
  MTP *without* the stream, and the chat arm keeps the stream *without* MTP.
- **0.5 GB less cache costs decode**: TPOT 57–60 → 60–63 ms (fewer resident experts). That is the
  price of the 1.3 GB stream buffers on this card.

### What is still missing for parity (updated after item 2)

| term | naiwa | strata | gap | where it goes |
|---|---|---|---|---|
| prefill (turn 1, 1.7k) | **11.9 s** | 2.4 s | 5× | 4 chunks × 31 GB of layer uploads at 512-token chunks; a bigger chunk divides the upload count |
| TTFT ≥2 | 2.8–4.4 s | 0.6–1.2 s | 3–7× | the per-turn tail (67–213 tokens) reprinted at 150–285 tok/s, plus the chunk's fixed upload |
| **TPOT** | 60–63 ms | 28–39 ms | **1.6–2.2×** | the miss path: at 42% hit rate 3.4 experts/token run on the CPU (perfect cache → 19 ms) |
| e2e/conv | 44–84 s | 22–40 s | 1.9–2.8× | dominated by turns ≥2 |

**Why TPOT is the term to attack, and what it is made of** (measured in "Next" item 3): with every expert
resident the same conversation runs at **~19 ms/token** — faster than strata's 38 — and `NAIWA_STEP_DBG`
shows the GPU already idle at every layer entry (`busy 0.000`, `dry 1.2 ms/layer` × 48 = the whole step).
So the step is host-side wall time spent on the **miss path**: 42.4% hits / 15.7% staged / 41.9% CPU at a
1 GB cache, i.e. ~3.4 experts per token computed on the CPU. It is not the launch chain (M8 was right for
np8: `--no-lookahead` does not reduce it) and not the PLE gather.

## Judging

1. **M10.1 — reproducible harness.** ✅ Same corpus + config → same schema and turn count; 270 turns written
   across three runs (138 before the prefill fix, 132 after), none failed; turn-1 `prompt_tokens` equal
   across engines (40 = 40, 1723 = 1723); 3 rounds for every class after the prefill fix (1 round for
   `doc`/`long` before it, labelled); medians throughout.
2. **M10.2 — honest baseline.** ✅ Table above, no tuning, `spec` arm paired, engine order alternated.
3. **M10.3 — align** (open): items 1 (prefix reuse) and 2 (prefill) done; decode (TPOT 1.6–2.2×)
   is now the open term, and the lender's accounting is the open defect — see the agenda above.
4. **M10.4 — long context.** `doc` and `long` are measured (1,723–2,491 prompt tokens): strata reuses
   0.89–0.99 and holds 0.6–1.0 s TTFT; naiwa reuses 0.83–0.94 and holds 2.8–4.4 s, with an 11.8 s turn 1.
   The residual is the chunk upload, not the cache. Contexts beyond 4096 stay out of scope until the
   KV budget question in "Open questions" is settled.

## Gates (executed)

- Harness: 3 rounds for `short`/`code` (1 for `doc`/`long` before the prefill fix, 3 after), medians, order
  alternated, 0 failed turns in 270, 0 OOMs after the fix. Model load is intermittently OOM on this 12 GB
  card (the PLE host-gather fallback fragments the allocator); the driver retries *startup* up to twice and
  never retries a turn.
- Engine changed in item 2 only (`prefill_stream.rs`, `copy_stream.rs`), and the oracle was re-verified after
  it: `frac 0` + `--no-lookahead` byte-identical to `harness/m1/reference.oracle.txt` at `d0ef46d`; 35/35 tests
  (the new `pool_trim_returns_freed_tensors` included); 0 naiwa warnings. Items 1 and 2 of M10.3 touched no
  engine code except that fix.
- Every number above carries the comparability record in `run.json` and is reproduced by
  `harness/chat/summarize.py`.

## Work items (done)

- `harness/chat/corpus.jsonl`, `run.py` (stdlib-only driver: launch by PID, readiness, warmup, SSE timestamps,
  VRAM/RSS samplers, per-round config copy for strata), `summarize.py`, `README.md`.
- `harness/chat/engines/{naiwa,naiwa-mtp,strata,strata-engine}.json` — launch configs with a `record` block;
  `naiwa` is the chat arm (stream, no MTP, 1 GB cache), `naiwa-mtp` the shipped-speculation reference
  (MTP 4, 1.5 GB, no stream; strata has no nospec counterpart because a native IQ pack requires `--spec`).
- This table, generated by `summarize.py`, not by hand.

## Next (M10.3 items 3–6)

Ordered by payoff per unit of work. Each item states its gate; none of them may run inside the same
measurement as the baseline (one mechanism per run, `harness/chat/README.md`).

1. ✅ **Chunk-size sweep — done, configuration only.** `doc` turn 1: **512 → 11.9 s, 1024 → 7.4 s,
   2048 → OOM** (the 2048-row MoE forward needs the memory the stream buffers occupy: it also OOMs at a
   0.8 GB cache). Reuse is unchanged (0.92–0.98) and turns ≥2 are 2.5–5.6 s at both 512 and 1024, so
   `engines/naiwa.json` now uses **1024**. Gate met on the reuse side, missed on turn 1 (5 s); the 2048-row
   forward's memory is the blocker, not the upload.
2. **Fix the lender's accounting — 0.5–1 day, `src/moe/cached_layer.rs`.** `lend` credits
   `slots_lent × slot_bytes` (1107–1863 MiB) but returns 0–288 MiB; per layer it does free something
   (layer 0: 770 → 802 MiB of a 43 MiB credit), so most banks still have a strong reference. First move:
   print `Arc::strong_count` on each bank inside `CachedLayer::lend` and find the extra owner (candidates:
   a fill-thread clone, `Bank.proj` vs `SlotMemory.tensor`, slots outside `st.usable` sharing the bank).
   *Gate:* `lend` returns ≥ 90% of what it credits, 2 GB cache + stream does not OOM, TPOT back to ≤ 58 ms.
3. ✅ **Break down the 60 ms step — done. It is the miss path, not the host chain and not the PLE.**
   Two probes, one `short`-shaped conversation, `max_tokens 64`, 3 turns:

   | variant | TPOT per turn |
   |---|---|
   | as shipped (1 GB cache, stream, lookahead) | 54.3, 55.4, 65.8 ms |
   | `--no-lookahead` | 67.2, 65.4, 68.1 ms |
   | `NAIWA_GPU_ONLY_EXPERTS=1` (perfect cache, **lossy**) | **18.8, 19.0, 24.2 ms** |
   | perfect cache + `--no-lookahead` | 20.2, 14.0, 16.5 ms |

   `NAIWA_STEP_DBG=1` on the shipped config says `busy 0.000 ms/layer, dry 1.17–1.32 ms/layer` over 48
   layers: **the GPU is already idle at every layer entry**, so the token step is host-side wall time, and
   the line is 48 × ~1.2 = 56–63 ms. Turning the lookahead off does not reduce it (it slightly increases
   it), so the lookahead thread's CPU matmul is not the cost. Making every expert resident collapses it to
   **~19 ms, i.e. faster than strata's 38 ms**.

   Conclusion: the decode machinery is fine; the ~39 ms between 19 and 58 is what the *miss path* costs —
   CPU expert gemv on 41.9% of the experts and staged DMA on 15.7% (`NAIWA_ROUND_STATS` on the same run:
   hit 42.4% / staged 15.7% / cpu 41.9%, ~130 ms of CPU work per token summed over the pool). This flips
   the M6–M9 conclusion for *this* workload: the host chain is not short of launches, it is waiting on
   experts the 1 GB cache does not hold.
4. **Attack the miss path — the two levers, in this order.**
   a. **Hit rate.** 42.4% at a 1 GB cache. Half a gigabyte back (item 2) is the cheapest point available on
      this card; beyond that a bigger card. Measure the same `short` conversation at 1 / 1.5 / 2 GB
      (2 GB needs item 2) and report TPOT against hit rate — that curve says whether the CPU path or the
      cache is the lever.
   b. **CPU gemv cost.** ~130 ms of pooled CPU per token for ~3.4 miss experts. One expert-row is ~5.7 MB of
      IQ1_M weights, i.e. ~0.1 ms of memory traffic, so the fused decode is 5–10× off its memory bound at
      1 row — the same fused path that the np8 numbers praised at 8 rows. A 1-row-specialised kernel, or
      batching a layer's misses across the verify window, is where the remaining CPU time is.
   Do (a) before (b): if the hit-rate curve says 2 GB takes TPOT under 40 ms, the gemv work is second order.
5. **MTP: keep it out of chat, or make it reuse-aware — 0–2 days, fork optional.** Today `--mtp` costs all
   prefix reuse for +45% TPOT on `code` only. Either ship "MTP off for chat" (already the default arm) or
   add draft-aware checkpoints to the fork. *Gate:* if MTP is re-enabled for chat, reuse must stay ≥ 0.7.
6. **Scope and hardware — needs a decision.** A ≥24 GB card would allow stream+MTP, 8k `long`, a 4 GB expert
   cache with the stream (hit rate up, TPOT down), and the PLE residency question; on 12 GB the stream and
   MTP are mutually exclusive and `long` stays at 4096.

**Not now** (measured dead ends for this workload): tuning MoE/GEMM before item 3 names the term; chasing
the ±10% band while turn 1 is 5× off; `--prefix-cache-n 16` at 4096 context; stream+MTP or 2 GB+stream on
12 GB; retrying a failed *turn* (startup retries are separate and bounded).

## Open questions

- Should `--mtp` be smarter than `SpeculativePrefixReplay::Full`? The fork's reason is sound (block hashes
  alias MTP rows with different next-token inputs), but the cost is every turn's reuse. A draft-aware
  checkpoint (store the next-token input with the snapshot) would let chat keep both; that is fork work.
- The last-turn slack: reuse stops one block before the prompt end, so 0.73–0.94 instead of strata's
  0.95–0.99. Is that only the assistant-turn re-render, or also the block boundary rule? Measured but not
  explained.
- `--prefill-stream`'s defaults contradict each other (`--prefill-stream-min-tokens` 1024 >
  `--max-prefill-chunk-tokens` 512, so it never engages in serve mode). The benchmark pins 64/512; the
  defaults themselves should move, and the "1024" comment claims Strata streams from 1024 — with strata's
  2.4 s turn 1 that claim is worth re-checking.
- Where is the strong reference that keeps a lent bank alive? `lend` credits 1107–1863 MiB and returns
  0–288; per layer it frees some (770 → 802 MiB for a 43 MiB credit), so most banks are still owned
  elsewhere. Answering this gives 0.5 GB back to the cache (item 2 of "Next").
- Strata's startup probe picks `pcie_frac 0.37`; naiwa's `--pcie-frac 0.2` is a hand-tuned constant. Whether
  that matters for chat (it did not for np8) is unmeasured here.
- Host RSS: naiwa 56.0 GiB vs strata 26.5 GiB. Pinned expert stacks + host PLE gather is the likely cause; not
  investigated, and it is the difference between "runs on 64 GB" and "runs on 32 GB".
- Concurrency >1 is unmeasured: strata is a one-sequence FIFO. Any concurrency claim needs strata's slot/batch
  path configured first.
