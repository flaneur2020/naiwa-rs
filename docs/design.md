# naiwa-rs — design

**Goal: a Rust implementation of Strata that runs Qwen3.8-Next-Flash on commodity hardware.**

Reference target: RTX 4070 12 GB, Ryzen 5 3600X 6C/12T, Qwen3.8-Next-Flash IQ1_M (60 GB,
48 decoder layers + MTP head). The hard constraint is memory: the expert working set is ~26 GB
against a 12 GB card, so the engine must stream experts without putting the host on the critical
path.

Design principle:

> **The GPU spins on mapped flags; the host publishes data and never leads the device by a
> pass.**

The shipped path is eager MoE and eager attention. Plan and evidence:
[`roadmap.md`](roadmap.md), [`worklogs/`](worklogs/).

### Shipped configuration

| item | value |
|---|---|
| model / card | Qwen3.8-Flash-Next IQ1_M, 48 decoder layers + 1 MTP head / RTX 4070 12 GB |
| expert cache | 2 GB default; 4 GB only with deep draft (np8); without it 4 GB is slower |
| `--pcie-frac` | 0.2 optimum at np3/np8; bandwidth knob, not a TPOT lever |
| experts | fork rayon pool, 8 threads |
| prediction | one dedicated event-driven thread; issue-only, never performs the DMA |
| staging | residents run **before** the staging DMA's completion event; staged experts after it |
| MoE / attention | eager; no capture |
| PLE | host-only table; not implemented on decode |
| MTP | works in steady state on short/benchmark prompts; long-prompt drafter prefill is an open fault |
| reference | np8, `frac 0.2`, cache 4, counting prompt: **170 ms, 44 tok/s, 87.7% hit** |
| gate | `frac 0` + `--no-lookahead` **byte-identical** to `harness/m1/reference.oracle.txt` |

`--depth N` is decode context length (`4` in all current numbers); `--mtp-n-predict N` is draft
depth (np8 = 8 drafts, 9 verify rows). All performance numbers are short-context and
MoE-dominated; long-context runs at 2k/4k are still required.

---

## 2. Execution model

```
token thread                  GPU stream                         prediction thread / rayon
────────────                  ──────────                         ─────────────────────────
dispatch kernel    ────────→  resident / staged / CPU lists (mapped)
read 3 lists (no hashing)
issue staging DMA  ────────→  copy stream
submit CPU job     ────────→                                     rayon: CPU expert gemv
launch residents   ────────→  [resident experts]
wait staging event ────────→  cuStreamWaitEvent(staging DMA done)
launch staged      ────────→  [staged experts]
wait CPU payload   ────────→  wait_copy_slot, add to acc
                              next layer's attention (eager)
```

**Dispatch is host-side:** the M6 P2 device dispatch (`dispatch_topk` / `DispatchBuf`) measured
neutral and was deleted in M9.1. The host builds the resident/staged/CPU lists from the doorbell
top-k; the only host-side guard is the shape check. One fixed CPU path: fused when the
activation/format is supported, generic otherwise.

Stream order inside one layer:

```
wait preload claim (event)        released as soon as the lookahead's copies land
[resident MoE]                    starts immediately
wait staging event                DMA lands under the resident compute
[staged MoE]                      only now reads the staging ring
wait CPU payload, add             completes the MoE output
```

Resident-first staging is unconditional: **78.9 → 62.8 ms/pass (−20%)**, staging wait p50 858 → 11 µs. The CPU add
stays last because its result belongs to this layer's output, which the next layer consumes.

The preload wait has to come **before** the resident compute: a preload may have *replaced* a
resident slot, so the resident hits can include slots the lookahead is still copying into. It was
previously published as a gate word on candle's stream while the copies ran on the copy stream —
i.e. it did not order anything, and a resident hit could be read mid-copy.

The prediction thread runs only in the gap **after** a CPU-expert batch (beside it cost
10 ms/step). Workers make **no CUDA calls**; a CUDA call from a worker delays predictions and
collapses the hit rate (measured 74% → 38%).

---

## 3. Synchronisation

### Completion events (staging and preload)

There are no gate words and no epoch (deleted in M9c, see
[`worklogs/m9c.md`](worklogs/m9c.md)). Both barriers are CUDA events:

| barrier | event | producer | consumer |
|---|---|---|---|
| staging DMA | one `CopyDone` per layer, created at build | copy stream, recorded after the layer's copies | `cuStreamWaitEvent` on candle's stream, after the resident compute |
| preload | one `CopyDone` per layer + a claim counter (`PassCell`) | lookahead thread, recorded once after the batch of fills, then claimed | `cuStreamWaitEvent` on candle's stream, before the resident compute |

Rules:

1. **Record before claim.** The lookahead records the batch's completion and only then publishes
   the claim, so a consumer that sees a claim always has an event to wait on.
2. **Wait for the claim you see, not for a pass of your own.** The cell owns its sequence; the
   token thread waits for whatever was claimed last. That drains a copy the lookahead queued late
   (a stale claim is waited for instead of skipped) and deletes the pass number, the device epoch
   and the "every path must publish or the GPU spins" rule.
3. **One persistent event per layer, one record per claim.** Re-recording is safe: a new claim for
   a layer comes from a job the token thread scheduled one layer earlier, i.e. after its wait for
   the previous claim was enqueued (`cuStreamWaitEvent` binds to the most recent record at call
   time). A *fresh* event per claim costs ~1 ms per lookahead job — destroying an event whose
   record is still queued on the copy stream blocks — and measurably drops the hit rate.
4. **Failure closes.** A layer without its staging event does not stage (`staging_enabled()`);
   mapped memory that cannot be allocated at warmup is an error, not a silently unordered path.
5. Every stream shared by a producer and a consumer means "event completed = everything queued
   before it on that stream completed". The preload event therefore also waits for any staging DMA
   queued earlier on the same copy stream. Accepted: the alternative is a second stream.

### CPU return

One path: `ResultDoorbell` (`doorbell/h2d/result.rs`) — a seq, a slot ring and a device-written
`consumed` word. The worker publishes seq; the stream queues a wait+copy kernel that sets
`consumed = seq` once it has read the slot out, so the worker's ring depth is *checked* (bounded
wait, then a counted drop) instead of assumed. Ring: depth 8, sized once at warmup; the schedule
`CPU_EXPERTS` point, `wait_add`, `publish_cpu`, `deferred::start_schedule` and the per-layer
`Channel` were deleted in M9.2.

### PLE

`PleHash::rows` is a **pure function of the last `ngram-1` token ids**; PLE is **one layer**,
before its attention. The ~100 GB hashed table stays on the host, never VRAM. If implemented,
PLE is the third host-produced DMA: compute rows (µs), mmap → pinned → async H2D on the copy
stream, record a completion event a pass early. It is ~0.1–0.3% of a pass and not needed now.

---

## 4. Memory and load order

```
[VRAM]
  expert_banks      per-layer; mixed quant (7 expert shapes in IQ1_M)
  staging_ring      global, shape-keyed (16 slots; 64 measured identical)
  result_ring       global ring depth 8, seq + consumed (host -> device CPU results)
  activation_rings  per-layer ring depth 8 (device -> host activations)
  topk_doorbells    per-layer, one publish outstanding (device -> host router ids)
[host pinned]
  expert stacks     per-layer, three projections, cuMemHostRegister(PORTABLE)
```

Load: weights → **warmup** (all staging rings and per-layer banks once) → serve. The ring set is
asserted frozen across decode. Static addresses keep the steady state allocation-free.

Banks are per-layer because the model is mixed-quant (7 expert shapes); shared structures carry
counts/offsets/pointers only.

**A prompt step runs in row batches of 256** (`SUB_ROWS` in `prefill_stream.rs`), so the MoE
activation footprint is set by the batch and not by the chunk: an 8192-token chunk costs the same
activations as a 256-token one, and only attention still grows with the chunk (24 KiB a token). That
is what lets the plan below buy a large chunk instead of a second buffer — the same "ubatch" trade
Strata makes.

**A sequence slot costs ~112 MiB of recurrent state** on this hybrid model, and the engine defaults to
`--max-seqs 32`: ~3.5 GiB of the card for slots a one-request-at-a-time server never uses. A chat server
sets `--max-seqs 4`, which funds a 3 GB expert cache instead of 1 GB (TPOT 58 → 55 ms). The weights
themselves are 4.3 GiB on the device (2.8 GiB of it quantized verbatim) — measured with
`MISTRALRS_VRAM_DUMP=1` (`mistralrs_quant::dump_weights`), which is the way to attribute the footprint
instead of subtracting from `nvidia-smi`.

**The prefill stream's memory is one decision, taken twice.** `prefill_stream::plan` picks the prompt
step and the buffer count together from the free VRAM it measures: the largest chunk on the 256-token
grid whose `stride × layer + activations(chunk) + headroom` fits, where the activation term is
Strata's `160 MiB + 680 MiB per 1024 tokens` plus an eighth of slack. It runs at warmup and again at
the first long chunk (which sees ~200 MiB less), publishing the smaller of the two to the scheduler —
`--max-num-batched-tokens`/`--max-prefill-chunk-tokens` are ceilings, not settings. On this card the
warmup says 1280 and the first chunk corrects it to 1024. A bigger chunk beats a second buffer
(both measured), so the second buffer is only taken when the same chunk affords it, and a CUDA OOM
inside the stream's own path drops one (2 → 1 → off).

Its layer buffers (~650 MiB at IQ1_M) are *not* lent from the
expert cache: that time-sharing cycle was measured to be unstable — the lend returns whole banks
(1107 MiB of a 1300 MiB need), the buffers then need the rest from free VRAM, and the
give-back/re-borrow churn per turn (re-allocating the cache's banks and re-filling them) leaked
~400 MiB of free memory per conversation until a 3 MiB miss turned the stream off for the rest of
the run. So the stream reserves once and keeps:

- two buffers if `free ≥ 2 × layer + 768 MiB` (the next layer's upload overlaps this layer's MoE),
- one buffer if `free ≥ layer + 192 MiB` (uploads serial, ~8% slower on the first turn),
- otherwise off, with one warning naming the numbers and the flag to lower.

The 768 MiB margin is the chunk's MoE intermediates at `--max-prefill-chunk-tokens 1024`; a smaller
margin OOMs when the *model's* own allocation (not the stream's) grows. As a safety net, any CUDA
OOM raised inside the stream's own path drops a buffer (2 → 1 → off) and leaves that layer on the
CPU instead of failing the request. Measured with `--expert-cache-gb 1`: one 650 MiB buffer, 555 /
1095 / 3615-token prefill at 195 / 356 / 416 tok/s (CPU path: 11–13 tok/s). `--expert-cache-gb` is
therefore the cache's own budget; the stream adds its buffers on top and says so when they do not
fit.

The alternative — Strata's — is to lay the prompt buffers **inside** the cache's own allocation (a
bump allocation in its tail, re-laid out per chunk, so nothing is ever freed or re-allocated) and to
adapt the *chunk* to the region instead of the buffer count. That gives the same footprint with ~650
MiB more cache on this card; it is tracked as "Next" item 6 in
[`worklogs/m10-chat.md`](worklogs/m10-chat.md), to be done after the CPU-expert gemv work.

---

## 5. Performance envelope

| configuration | round p50 | tok/s | tokens/round |
|---|---|---|---|
| np8, counting prompt, best | 184.7 ms | 43.3 | 8.00 (23.1 ms/token) |
| np8, reference build | **170.0 ms** | **44.0** | 8.0 |
| np3 | 98.1 ms | 40.0 | 3.92 |
| np2 | 82–90 ms | 33.8 | 2.77 |
| no MTP | 63.2 ms | 15.8 | 1.0 |
| code continuation | - | 25–26 | 2.5–2.9 |

Target ≤ 35 ms/token is met on the counting prompt; acceptance decides tokens/round.

np8 round (~185 ms): GPU 40 ms (22%), host launch chain + sampler ~70 ms (38%, **not** CUDA
launches; profile needed), CPU-payload wait 47 ms (25%), staging wait 27 ms (15%), copy stream
overlapped at 53% duty.

np8 traffic: 2.68 GB/round (CPU 745 MB, staged 1000 MB, preload 935 MB) = 14.5 GB/s, 51% of the
28.6 GB/s measured DRAM ceiling; PCIe 12.8 GB/s busy / 6.4 average. A perfect cache buys only
**+8%**; the whole miss path is under a tenth of the round.

Measured dead ends are recorded in the roadmap's ablation table; do not re-litigate them.

`--lookahead-swaps`: `1` = −11% round at np8 with ~3 points less hit; `2` default; `0` collapses
hit rate to 26.5%.

### MTP

Head is local (`~/bench-rel/mtp/mtp-component.gguf`, `--mmproj`). Steady-state speculation is the
biggest lever (×2.5 tokens/pass). Open fault: prompt 512 faults with
`CUDA_ERROR_ILLEGAL_ADDRESS` in `drafter_prefill_chunk` before `drafter_forward`; prompt 8 works.
Suspects: drafter KV cache vs `context_lens`, and the MTP site's CPU-expert rows. The MTP site
now uses the global result doorbell, so the old schedule sizing bug (MTP site = layer 48) is gone.

---

## 6. Correctness

- **Oracle:** `frac 0` + `--no-lookahead` byte-identical to the frozen reference.
- **`frac > 0` is not byte-exact:** lookahead changes the resident/miss split, hence the
  `atomicAdd` terms. `frac 0.55` + `--no-lookahead` has exactly **two attractors** (2/20 diverge,
  both to the same text); `frac 0` is 0/10. A third variant is the regression signal.
- **Structural checks** and a must-fail control for every wait (a stale stamp must hang). Correct
  values alone are not evidence.
- **NaN fix:** two uninitialised buffers — the old schedule payload (`cudaHostAlloc` does not zero;
  `MappedBuf` now zeroes every mapping) and paged-attention KV blocks (`Tensor::empty`). Zeroing
  both: 2–3/5 → **0/6**.
- **Measurement:** medians of ≥ 3 runs; report round p50 + tokens/round + hit rate + flags;
  liveness ≥ 6 runs; no mean where a median exists.
- **One fixed configuration per decision.** Env vars are probes/overrides, not A/B arms. Current
  surface (6): `NAIWA_STEP_DBG`, `NAIWA_ROUND_STATS`, `NAIWA_RET_DBG`, `NAIWA_FUSED_DBG`,
  `NAIWA_LOOKAHEAD_DBG`, `NAIWA_GPU_ONLY_EXPERTS` (lossy bound). The oracle switch is the `--no-lookahead` flag.
- **Pitfalls:** host-written per-step targets race (the claim cell owns its sequence); publish FIFO
  needs the mapped **device view**; the DMA and its wait must share `staging_enabled()` (a copy
  stream exists at `frac 0`, so the copy stream's existence is not the predicate).

---

## 7. Fork boundary

Upstream `bac32a67`: **41 commits, 77 files, +19 448 / −152**; `mistralrs-quant` 65 files /
+18 742, `mistralrs-core` 10 / +696.

- **Stay:** native IQ/Q formats (`mistralrs-quant` ~14k lines, block tables, oracles, MMQ/MMVQ
  kernels, `gguf/ffi.rs`); CPU fused MoE (`fused_cpu.rs`, `fast_mmq.rs`, `fast_mmvq.rs`);
  GPU MoE kernels (`indexed_moe.cu`, `moe_grouped.cu`); quant CUDA wiring (`gguf/cuda.rs`);
  model/pipeline plumbing; speculative drafting.
- **Done:** the doorbell moved to `src/doorbell/` (−1660 lines).

The doorbell was the only clean extraction; everything else is a kernel or model plumbing.

---

## 8. Layout and commands

```
~/code/naiwa-rs            engine + docs/ + harness/
  src/doorbell/            mapped doorbells + kernels
    mod.rs                 ffi + re-exports
    mapped.rs              single mapped-allocation owner
    payload_ring.rs        seq + slot-ring layout
    d2h/topk.rs            TopkDoorbell (device -> host)
    d2h/activation.rs      ActivationDoorbell (device -> host)
    h2d/result.rs          ResultDoorbell (host -> device)
    handoff.rs             PassCell (cross-thread completion claim)
    tests.rs
  src/moe/                 MoE execution + GPU expert cache
    mod.rs                 module docs, constants, exports
    provider.rs            GPU-cache provider + layer construction + staging rings + attach_prefill
    expert.rs              CpuExperts fallback + per-expert byte slicing
    cached_layer.rs        Layers + LayerState + CachedLayer + CachedExperts + lookahead prediction
    slot.rs                Slot/SlotBuf/Bank/Projections + slot byte estimate
    staging.rs             StagingRings/StagingRing + alloc_bank
    decode.rs              decode orchestrator
    stages/                route, gpu, cpu-miss (+ fused decode), dma-miss, preload
  src/load.rs              model loading + provider install + warmup
  src/stats/               counters + NAIWA_* measurement probes
    mod.rs                 shared gate/window/header/units; re-exports record_*/dump_*
    cache.rs               expert-cache counters (`dump_cache_stats`)
    lookahead.rs           LOOKAHEAD_DBG
    round.rs               ROUND_STATS
    step.rs                STEP_DBG + ROUND_DBG
    fused.rs               FUSED_DBG
    retired.rs             RET_DBG
  src/executor.rs          single-thread CPU executor + result doorbell + idle signal
  harness/                  primitives, oracle, bandwidth harnesses
~/code/mistral.rs-iq4xs    fork
~/naiwa-rs/target          crate target
~/bench-rel/naiwa-v2/      binaries/logs
```

```bash
# oracle (must be IDENTICAL)
BIN=$HOME/naiwa-rs/target/release/naiwa-rs TAG=X FRAC=0 NO_LOOKAHEAD=1 \
  OUT=$HOME/bench-rel/naiwa-v2/m4 timeout 400 ./harness/m1/capture_oracle.sh
diff ~/bench-rel/naiwa-v2/m4/reference.X.txt harness/m1/reference.oracle.txt

cargo test --offline --release --features cuda     # 33/33
```

Reference bench (np8):

```bash
$BIN bench -m ~/models/qwen3-coder/requant -f '<shard1>;<shard2>' --cpu-moe \
  --expert-cache-gb 4 --pcie-frac 0.2 --depth 4 --mtp-n-predict 8 \
  --prompt-file <predictable-prompt> --gen-len 96
```

---

## 9. Pointers

- Plan and gates: [`roadmap.md`](roadmap.md)
- Measurements and per-milestone evidence: [`worklogs/`](worklogs/)
