# M10 — multi-round chat: naiwa-rs vs Strata

Status: **M10.1 done, M10.2 measured, M10.3 item 1 resolved (2026-10-19/20)**. Alignment not started.
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
| **B** no speculation | no `--mtp`, 2 GB cache | *(cannot run)* | measures what speculation costs a chat session |

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

### What is still missing for parity (the real M10.3 agenda)

| term | naiwa B | strata | gap | where it goes |
|---|---|---|---|---|
| reuse at turn ≥2 | 0.73–0.94 | 0.95–0.99 | 5–23 points | block-aligned *last user turn*: one block of slack per turn, plus the assistant turn's re-render |
| prefill rate | 11–13 tok/s | ~750 tok/s (1.7k in 2.3 s) | **60×** | turn 1 of every conversation: 142–152 s vs 2.2–2.3 s. `--prefill-stream` never engages at its defaults (min 1024 > chunk 512) and OOMs when forced at a 1.5 GB cache |
| TPOT | 57–60 ms | 27–39 ms | 1.5–2.2× | decode/MoE, the only term the np8 work optimized |
| e2e/conv | 121–238 s | 21–35 s | 4–11× | dominated by turn 1's prefill |

So parity needs, in this order: **a prefill path that runs at GPU speed** (60× on turn 1, and it
carries every conversation), **reuse of the last turn** (23 points of reuse ≈ 4× TTFT at turn 8),
and only then **decode** (1.5×). The MoE/decode path that M6–M9 spent five milestones on is the
smallest of the three gaps for this workload.

## Judging

1. **M10.1 — reproducible harness.** ✅ Same corpus + config → same schema and turn count; 138/138 turns
   written; turn-1 `prompt_tokens` equal across engines (40 = 40, 1723 = 1723); ≥3 rounds for the head-line
   classes; medians; every failure recorded (none).
2. **M10.2 — honest baseline.** ✅ Table above, no tuning, `spec` arm paired, engine order alternated.
3. **M10.3 — align** (open): item 1 (prefix reuse) resolved by configuration; items 2–3 (prefill
   rate, decode) open — see the agenda above.
4. **M10.4 — long context.** `doc` and `long` are measured (1,723–2,491 prompt tokens): strata reuses
   0.99 and holds 620–777 ms TTFT; naiwa reuses 0.92–0.94 and holds 10.8–13.4 s — the residual is the
   prefill rate, not the cache. Deeper contexts stay out of scope until the context/KV budget question in
   "Open questions" is settled.

## Gates (executed)

- Harness: 3 rounds for `short`/`code`, 1 for `doc`/`long` (labelled), medians, order alternated, 0 failed
  turns in 138. Model load is intermittently OOM on this 12 GB card (the PLE host-gather fallback fragments
  the allocator); the driver retries *startup* up to twice and never retries a turn.
- Engine untouched: no `src/` change; `frac 0` + `--no-lookahead` oracle is unaffected and was last verified at
  `21cd122`.
- Every number above carries the comparability record in `run.json` and is reproduced by
  `harness/chat/summarize.py`.

## Work items (done)

- `harness/chat/corpus.jsonl`, `run.py` (stdlib-only driver: launch by PID, readiness, warmup, SSE timestamps,
  VRAM/RSS samplers, per-round config copy for strata), `summarize.py`, `README.md`.
- `harness/chat/engines/{naiwa,naiwa-nospec,strata,strata-engine}.json` — launch configs with a `record` block;
  `naiwa-nospec` exists but was not run (strata cannot pair it).
- This table, generated by `summarize.py`, not by hand.

## Open questions

- Should `--mtp` be smarter than `SpeculativePrefixReplay::Full`? The fork's reason is sound (block hashes
  alias MTP rows with different next-token inputs), but the cost is every turn's reuse. A draft-aware
  checkpoint (store the next-token input with the snapshot) would let chat keep both; that is fork work.
- The last-turn slack: reuse stops one block before the prompt end, so 0.73–0.94 instead of strata's
  0.95–0.99. Is that only the assistant-turn re-render, or also the block boundary rule? Measured but not
  explained.
- `--prefill-stream`'s two contradictory defaults: which one moves, and can the stream run inside a 1.5–2 GB
  cache on a 12 GB card at all?
- Strata's startup probe picks `pcie_frac 0.37`; naiwa's `--pcie-frac 0.2` is a hand-tuned constant. Whether
  that matters for chat (it did not for np8) is unmeasured here.
- Host RSS: naiwa 56.0 GiB vs strata 26.5 GiB. Pinned expert stacks + host PLE gather is the likely cause; not
  investigated, and it is the difference between "runs on 64 GB" and "runs on 32 GB".
- Concurrency >1 is unmeasured: strata is a one-sequence FIFO. Any concurrency claim needs strata's slot/batch
  path configured first.
