# M10 — multi-round chat: naiwa-rs vs Strata

Status: **M10.1–2 done; M10.3 items 1–3 done (2026-10-19/20)**. The open term is the 1-row CPU gemv
(0.73 ms/layer, 48 layers); next: "Next" item 4 below.
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

## M10.3 baseline — 2026-10-20, lend era (superseded)

Same corpus and protocol as M10.2; `engines/naiwa.json` is the stream arm (no MTP, 1 GB cache), strata
unchanged. 132 turns, 0 failed turns, 0 OOMs in this run — but the stream then declined in 2 of 3 later
runs because of the lend churn (see "Next" item 2), which is why the memory model was replaced below.
Arm `naiwa-mtp` (MTP 4, 1.5 GB, no stream) keeps its M10.2 numbers: speculation and the stream cannot
both fit on this card.

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

## M10.3 baseline — 2026-10-20, final (chunk 1024, stream owns its memory)

3 rounds, 132 turns, 0 failed turns, 0 OOMs; `engines/naiwa.json` unchanged except the chunk size.

| class | engine | TTFT t1 | TTFT ≥2 p50 | p90 | TPOT p50 | decode tok/s | reuse p50 | e2e/conv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `short` | naiwa | 3,023 ms | **2,749 ms** | 5,193 ms | 58 ms | 17.2 | 0.75 | 61.4 s |
| `short` | strata | 1,050 ms | 636 ms | 824 ms | 39 ms | 25.5 | 0.95 | 29.9 s |
| `code` | naiwa | 2,993 ms | 4,420 ms | 5,039 ms | 60 ms | 16.7 | 0.47 | 49.2 s |
| `code` | strata | 984 ms | 1,165 ms | 1,175 ms | 29 ms | 34.7 | 0.67 | 21.9 s |
| `doc` | naiwa | **8,387 ms** | 4,343 ms | 5,200 ms | 60 ms | 16.6 | 0.95 | 57.2 s |
| `doc` | strata | 2,405 ms | 638 ms | 733 ms | 38 ms | 26.5 | 0.99 | 24.3 s |
| `long` | naiwa | 8,378 ms | 3,062 ms | 3,841 ms | 59 ms | 17.0 | 0.82 | 80.3 s |
| `long` | strata | 2,241 ms | 1,007 ms | 1,297 ms | 38 ms | 26.1 | 0.89 | 39.5 s |

Against the lend-era run: `doc` turn 1 7.6 → 8.4 s (one buffer instead of two), `doc` e2e 56.3 →
57.2 s, `short` TTFT ≥2 **4.04 → 2.75 s** and e2e 66.9 → 61.4 s (the cache is no longer emptied during
prefill). TPOT 58–60 ms unchanged. What remains is item 4: 19 ms of GPU chain + 48 × 0.73 ms of 1-row
CPU expert gemv.

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

### How Strata does the same thing (read from its source)

Strata has the identical problem and solves it by **letting the prompt path borrow the tail of the
expert cache**, not by giving it memory of its own (`src/program/generate.cpp:3203`):

> THE PROMPT PATH'S BUFFERS ARE BORROWED FROM THE CACHE, NOT WITHHELD BESIDE IT. With borrowing the
> cache is sized first and at full size, and the prompt path is laid out in the tail of it
> (`Prefill::relayout`), so it withholds no VRAM of its own and this reserve is zero.

| | Strata | naiwa, now (owned) | naiwa, before (lend by free/alloc) |
|---|---|---|---|
| where the buffers come from | a **bump allocation inside the cache's own device memory** (`Prefill::init(borrow, borrow_bytes)`, `src/prefill/prefill.cpp:363`) | a separate allocation, resident for the session | cache slots freed → CUDA pool → new allocation |
| how they go back | re-laid out in place (`relayout`), no `cudaMalloc`/free | never | `give_back` → `alloc_bank` + background refill |
| what adapts | **the chunk**: bisection on a 256-token grid from 8192 down to 256 (`biggest_chunk`, `generate.cpp:3505`) | the buffer count, once (2 → 1 → 0) | nothing → it turns itself off |
| how much is lent | `bytes_needed(chunk)` exactly (`lend_slots`, `generate.cpp:5712`), capped at 90% of the slots (85% when host copies are the limit) | n/a | `2 × largest layer`, whole banks |
| when it is decided | at startup for the cache (with `--vram-reserve-mib`, default 700, deducted first), per request for the chunk | once, at the first long chunk | per chunk, with a latched refusal |
| failure posture | shrink the chunk (or the cache, `--expert-cache auto` re-checks *after* the slots are written) | OOM inside the stream → drop a buffer | a 3 MiB miss turned the stream off for the run |
| extra VRAM | **0** | 650 MiB beside the cache | 0, but unstable |

Two measured numbers from their comments worth keeping:

- **The chunk is worth more than the slots it costs**: on a 4-way IQ3_S rig, a streamed-ring slot is
  worth ~0.53 tok/s and a chunk token ~0.05, so the rule is "the largest chunk that still leaves the
  ring full"; on a 12 GB card with a 32K Q2_0 prompt, chunk 4096 → 791 tok/s, 6144 → 878, 8192 → 973
  (69% of the slots lent).
- **Borrowing costs nothing on decode**: 58.4 tok/s owned vs 58.5 borrowing (K=28, RX 9070 XT + R9700),
  which is why borrowing is their default and why the "owned" layout is reserved for
  `--no-prefill-borrow` / no expert profile.

Their no-borrow reserve is `160 + 680/1024 × chunk` MiB (`generate.cpp:3216`) — for our 1024-token
chunk that is ~840 MiB, within 10% of the 768 MiB margin we arrived at empirically. The margin is a
property of the model, not a tune.

### What is still missing for parity (updated after item 2)

| term | naiwa | strata | gap | where it goes |
|---|---|---|---|---|
| prefill (turn 1, 1.7k) | **11.9 s** | 2.4 s | 5× | 4 chunks × 31 GB of layer uploads at 512-token chunks; a bigger chunk divides the upload count |
| TTFT ≥2 | 2.8–4.4 s | 0.6–1.2 s | 3–7× | the per-turn tail (67–213 tokens) reprinted at 150–285 tok/s, plus the chunk's fixed upload |
| **TPOT** | 60–63 ms | 28–39 ms | **1.6–2.2×** | 19 ms GPU chain + **48 × 0.73 ms** of 1-row CPU miss-experts on the per-layer critical path |
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
- Engine changed in items 2, 3 and 5 (`prefill_stream.rs`, `copy_stream.rs`, the lend removal in `moe/`,
  and a 10-line `PROMPT_STEP_CEILING` hook in the fork's scheduler: `mistralrs-core` `cc61c38c`). The oracle was re-verified after each: `frac 0` + `--no-lookahead` byte-identical to
  `harness/m1/reference.oracle.txt` — and, after the memory rework, **also with `--prefill-stream`**, which
  is the first time that path is covered by the oracle (it exercises one-buffer prefill and, at the oracle's
  2 GB cache, the *decline* path). Re-verified after item 5 as well, both variants. 35/35 tests, 0 naiwa
  warnings.
- Every number above carries the comparability record in `run.json` and is reproduced by
  `harness/chat/summarize.py`.

## Work items (done)

- `harness/chat/corpus.jsonl`, `run.py` (stdlib-only driver: launch by PID, readiness, warmup, SSE timestamps,
  VRAM/RSS samplers, per-round config copy for strata), `summarize.py`, `README.md`.
- `harness/chat/engines/{naiwa,naiwa-mtp,strata,strata-engine}.json` — launch configs with a `record` block;
  `naiwa` is the chat arm (stream, no MTP, 1 GB cache), `naiwa-mtp` the shipped-speculation reference
  (MTP 4, 1.5 GB, no stream; strata has no nospec counterpart because a native IQ pack requires `--spec`).
- This table, generated by `summarize.py`, not by hand.

## Next (M10.3 items 3–8)

Ordered by payoff per unit of work. Each item states its gate; none of them may run inside the same
measurement as the baseline (one mechanism per run, `harness/chat/README.md`).

1. ✅ **Chunk-size sweep — done, configuration only.** `doc` turn 1: **512 → 11.9 s, 1024 → 7.4 s,
   2048 → OOM** (the 2048-row MoE forward needs the memory the stream buffers occupy: it also OOMs at a
   0.8 GB cache). Reuse is unchanged (0.92–0.98) and turns ≥2 are 2.5–5.6 s at both 512 and 1024, so
   `engines/naiwa.json` now uses **1024**. Gate met on the reuse side, missed on turn 1 (5 s); the 2048-row
   forward's memory is the blocker, not the upload.
2. ✅ **Memory management — done, by removing the lending instead of fixing it.** With the pool trim
   inside `Layers::lend` the lend *is* measurable at a 1 GB cache (1107 MiB freed of 1107 credited), so
   the estimate was not the problem; the **cycle** was. Per turn: borrow → chunk → eight decode steps →
   give back (re-allocate 1.1 GB of cache banks and re-fill them) → next turn borrows again. Measured
   free VRAM across one conversation: 1887 → 1787 → 1785 → 1649 → **1489 MiB**, and at 1489 it missed
   the 1492 MiB requirement by 3 MiB, which latched the stream **off for the rest of the run** (the
   `long` class then prefilled at 13 tok/s: 135 s for a 1.7k turn, in 2 of 3 rounds).

   The redesign deletes the time-sharing: `ExpertMemoryLender`, `Layers::lend`/`give_back`,
   `CachedLayer::lend`/`slot_bytes`, the bank-tail split (`loan_bank_slots`) and the lend unit test are
   gone. The stream reserves its own buffers once, from free VRAM, and keeps them:

   | free VRAM at the first long chunk | choice |
   |---|---|
   | ≥ 2 × layer + 768 MiB | two buffers (the next layer's upload overlaps this layer's MoE) |
   | ≥ layer + 192 MiB | one buffer (uploads serial, ~8% slower on turn 1) |
   | else | off, one warning with the numbers and the flag to lower |

   The 768 MiB margin is the chunk's MoE intermediates at 1024-token chunks; 512 and 192 MiB margins
   both OOM'd when the *model's* allocation (not the stream's) grew. As a safety net, a CUDA OOM raised
   inside the stream's own path downgrades a buffer (2 → 1 → off) and leaves that layer on the CPU
   rather than failing the request — the driver's own answer is the only reliable statement about what
   fits, so it is used as the signal instead of a size heuristic.

   Result, 3 rounds, 132 turns: **0 OOMs, the same reservation in every round** ("one 650 MiB layer
   buffer reserved"), prefill 195–416 tok/s, and the cache is never emptied during prefill any more.
   The cost is explicit: one buffer instead of two on this card, so `doc` turn 1 is 8.4 s instead of the
   7.6 s the lend version managed when it did not drift.

3. ✅ **Break down the 60 ms step — done. It is the 1-row CPU gemv, on the critical path per layer.**
   Four probes on the same `short`-shaped conversation (`max_tokens 64`, 3 turns):

   | probe | reading |
   |---|---|
   | `NAIWA_STEP_DBG` | `busy 0.000 ms/layer`, `dry 1.17–1.32 ms/layer` × 48 → **the GPU is idle at every layer entry**; the step is host wall time, 56–63 ms |
   | `NAIWA_RET_DBG` | per CPU job: **queue 7.2–7.5 µs, compute 667–729 µs** — it is the gemv, not the hand-off |
   | `NAIWA_ROUND_STATS` | 1 GB: hit 35.7–38.2% / staged 16.5–17.4% / cpu 45.3–46.9%, ~140 ms of pooled CPU per token |
   | perfect cache (`NAIWA_GPU_ONLY_EXPERTS=1`, lossy) | **18.8 / 19.0 / 24.2 ms** TPOT — faster than strata's 38 ms |

   So: ~19 ms is the GPU chain, ~5 ms is the rest of the host path, and **~35 ms = 48 layers × 0.73 ms of
   CPU miss-experts** — the GPU blocks on the result publish point for exactly that long, and the host's
   next-layer top-k read waits behind it. Two things that do *not* move it:

   | knob | TPOT (3 turns) | hit rate |
   |---|---|---|
   | cache 1 GB → 1.5 GB | 53.6/57.1/64.4 → 52.5/54.2/70.0 | 35.7–38.2 → 37.2–42.2% |
   | `--lookahead-swaps` 2 → 4 | 53.5/58.3/65.3 → 58.0/62.4/68.0 | 37.2–38.2 → 43.8% |
   | `--lookahead-swaps` 0 | 69.2/70.8/62.5 | 20.1% |
   | `--pcie-frac` 0.2 → 0.5 → 1.0 | 53.5–65.3 → 51.0/52.3/56.1 → 59.8/67.1/79.1 | — |
   | `--no-lookahead` | 65.4–68.1 | — |
   | `NAIWA_CPU_MOE_THREADS` 1 → 2 → 4 → 8 (default) | job 3.22 → 1.70 → 1.24 → **0.73 ms**, TPOT 129 → 82 → 65 → 55 ms | — |

   More cache, more swaps, more staging and less lookahead all fail to help for the same reason: **any miss
   in a layer pays the whole 0.73 ms round trip**, so the cost is per *layer with a miss*, not per miss.
   Halving the miss count does not halve the wait; eliminating it does (perfect cache = 19 ms). Thread
   scaling (last row) says the gemv itself is the cost and that 8 threads is still the right pool: the kernel
   moves ~6.4 MB of IQ1_M weights in 0.73 ms ≈ 9 GB/s, well under what the host can stream, because a 1-row
   dot pays the dequantisation per block and uses each weight once.

4. **Shrink the per-layer CPU gemv — now the only lever that changes the critical path.** 0.73 ms for ~3.4
   expert-rows is ~5× its memory bound (~6.4 MB of IQ1_M weights per layer at 1 row → ~0.13 ms). The fused
   two-pass decode is tuned for the np8 batch (8 rows reuse each weight block); at 1 row it streams the same
   weights with none of that reuse. Targets, in order of expected payoff:
   a. **A 1-row path** that streams each expert's weights once (single pass over gate/up/down with the
      activation applied in-line, no intermediate quantize), measured against the 0.73 ms baseline. Gate:
      TPOT ≤ 40 ms on the same conversation (i.e. beating strata's 38 ms), oracle unaffected (it is the
      `frac 0 --no-lookahead` path, which also runs CPU experts — byte-identical is the check).
   b. **Cross-layer batching**: compute layer L's misses with layer L+1's (the ids are known one layer
      early through the router), so one gemv call serves two layers. Bigger change; only if (a) stalls.
   c. **Not** more cache or more staging at 12 GB: both measured as a wash (table above). A ≥24 GB card with a
      4 GB cache would be the real test of "keep them resident", and it is a hardware decision.
5. ✅ **Adapt the chunk instead of the buffer count — done, one decision in one place.**
   `prefill_stream::plan(free, layer, ceiling)` is a pure function (unit-tested): it bisects the
   256-token grid for the largest prompt step whose footprint fits, where
   `footprint = stride × layer + activation_bytes(chunk) × (1 + 1/8) + HEADROOM` and
   `activation_bytes` is Strata's `160 MiB + 680 MiB per 1024 tokens`. A bigger chunk beats the
   second buffer (see the measured note in the source), so the second buffer is taken only when the
   *same* chunk affords it. It runs twice, Strata-style:

   | when | what it sees | decision |
   |---|---|---|
   | warmup, once the cache/KV/buffers exist | 2050 MiB free | chunk 1280, one 650 MiB buffer (1852 MiB) |
   | the first long chunk, which sees ~200 MiB less (staging ring, KV growth) | 1888 MiB free | **chunk 1024**, one buffer (1682 MiB) — publishes the smaller step |

   `--max-num-batched-tokens`/`--max-prefill-chunk-tokens` are now **ceilings** (8192 in the engine
   config), not settings; the scheduler reads the published step every prompt step, so a mid-session
   shrink is a smaller next chunk rather than a broken one (a new `PROMPT_STEP_CEILING` atomic in
   `mistralrs-core/src/paged_attention/scheduler.rs`, clamped with `min`, so a host that never sets
   it behaves exactly as before). A CUDA OOM inside the stream's own path still drops a buffer.

   3 rounds / 132 turns: **0 OOMs** and the same decision in every round; `doc` turn 1 8.5 s,
   `doc` TTFT ≥2 3.35 s, TPOT 58–60 ms, `short` e2e 61.2 s — i.e. the plan reproduces the hand-tuned
   1024 on this card, and on a card with ~3 GB free it would pick 2560, on 12 GB free 8192 with two
   buffers. Gate met apart from the one-machine part of it (this card cannot demonstrate the larger
   chunks).
6. **Carve the buffers inside the cache — 1–2 days, own repo, after the CPU gemv work (open).** A layer buffer
   is exactly the `[E, N, K]` stack a cache bank holds, so the stream's buffer could *be* a bank:
   allocated as a bank when the stream is first needed, used as the `MoEExperts` stack (what `load`
   already does), and handed back to the cache as slots between prompts. Then the 650 MiB stops being
   extra VRAM — on this card the cache goes from 1 GB to ~1.65 GB at the same footprint, and Strata
   measured that owning vs borrowing costs nothing on decode (58.4 vs 58.5 tok/s). Risks: candle
   cannot mint a `QCudaStorage` over a foreign pointer, so the bank must be allocated *as* a bank from
   the start; the mixed-quant shape changes must not re-enter the CUDA-pool trap (`trim_device_pool`).
   *Gate:* same prefill tok/s as today, +650 MiB of cache slots measured, TPOT ≤ 57 ms, oracle
   identical with `--prefill-stream`.
7. **MTP: keep it out of chat, or make it reuse-aware — 0–2 days, fork optional.** Today `--mtp` costs all
   prefix reuse for +45% TPOT on `code` only. Either ship "MTP off for chat" (already the default arm) or
   add draft-aware checkpoints to the fork. *Gate:* if MTP is re-enabled for chat, reuse must stay ≥ 0.7.
8. **Scope and hardware — needs a decision.** A ≥24 GB card would allow stream+MTP, 8k `long`, a 4 GB expert
   cache with the stream (hit rate up, TPOT down), and the PLE residency question; on 12 GB the stream and
   MTP are mutually exclusive and `long` stays at 4096.

**Not now** (measured dead ends for this workload): more cache by *budget* at 12 GB (1 → 1.5 GB was a
wash; item 6 is the way to get cache back), more staging at 12 GB (a wash — see item 3),
`--no-lookahead` (worse), `--prefix-cache-n 16` at 4096 context, stream+MTP or 2 GB+stream on 12 GB, retrying
a failed *turn* (startup retries are separate and bounded). And no MoE/GEMM tuning that is not aimed at the
0.73 ms 1-row CPU path from item 4: the GPU chain is already 19 ms, faster than strata.

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
