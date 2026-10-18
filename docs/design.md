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
| staging | residents run **before** the staging gate; staged experts after it |
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
launch staging gate ───────→  wait SYNC_STAGING
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
[resident MoE]                    starts immediately
wait_ready(layer, SYNC_STAGING)   DMA lands under the resident compute
[staged MoE]                      only now reads the staging ring
wait CPU payload, add             completes the MoE output
```

Resident-first staging is unconditional: **78.9 → 62.8 ms/pass (−20%)**, staging-gate p50 858 → 11 µs. The CPU add
stays last because its result belongs to this layer's output, which the next layer consumes.

The prediction thread runs only in the gap **after** a CPU-expert batch (beside it cost
10 ms/step). Workers make **no CUDA calls**; a CUDA call from a worker delays predictions and
collapses the hit rate (measured 74% → 38%).

---

## 3. Synchronisation

### Schedule (staging and preload)

```
epoch    u32                 device-written once per pass (layer 0 prologue)
ready    [K] u32             one word per sync point; value = pass whose data is ready
payload  [slots][max_n] f32  single CPU-result block (see CPU return)

PRELOAD = 0, STAGING = 1, CPU_EXPERTS = 2
sync_id(layer, kind) = layer * 3 + kind
slots = max_decoder_layer + 2 = 49     // includes the MTP site at layer 48
K = 49 * 3 = 147
```

Node:

```
e = *epoch
spin while (ready[id] < e)
__threadfence_system()
CPU_EXPERTS -> acc += payload[layer]
PRELOAD / STAGING -> continue    // the DMA already wrote VRAM
```

`epoch` is **device-written only**, so the host can lag or match, never lead. The single payload
is safe because the host's pass advance is gated by a GPU-produced signal (the per-layer router
doorbell, after all layers of pass k). If that stops holding, add double buffering or a `consumed`
handshake.

Rules:

1. **FIFO per point.** CPU publications, including “no misses → zeros”, go through the single
   worker queue; a synchronous publish can overtake the previous pass and move `ready` backwards.
2. **`ready` is a level.** Preload uses one-thread `atomicMax(ready[id], pass)`; staging uses
   `cuStreamWriteValue32` after the copies. A stale publish must be harmless.
3. **Every staging path publishes** (nothing / staged / failed); its gate is replayed every pass.
   Preload's gate is replayed only when `inflight == pass`.
4. **Failure advances the point** (publish zeros), or the GPU spins forever.
5. `u32` wrap is ~4.29B passes; no handling needed.
6. Device epoch is fenced before/after; host payload/ready uses release; the node fences after
   the spin. Ada did not need the fences here, but they stay for RDNA.

### CPU return (open)

The schedule's `CPU_EXPERTS` point is currently unused. The live path is the per-layer `Channel`
(`FlagBuf`: seq + payload ring) with `wait_copy_slot` on the stream; the MTP site uses the same
path.

The intended end state is **one** protocol: move decoder/MTP CPU return onto `ScheduleBuf` and
delete `Channel` (~200 lines), or document why `Channel` stays. Do not add a third path.

### PLE

`PleHash::rows` is a **pure function of the last `ngram-1` token ids**; PLE is **one layer**,
before its attention. The ~100 GB hashed table stays on the host, never VRAM. If implemented,
PLE is the third host-produced DMA: compute rows (µs), mmap → pinned → async H2D on the copy
stream, publish `SYNC_PLE` a pass early. It is ~0.1–0.3% of a pass and not needed now.

---

## 4. Memory and load order

```
[VRAM]
  expert_banks      per-layer; mixed quant (7 expert shapes in IQ1_M)
  staging_ring      global, shape-keyed (16 slots; 64 measured identical)
  flag_host_to_gpu  per-layer ring depth 8 (descriptors, DMA lists, CPU results)
  schedule          epoch, ready[147], payload[49][max_n]
[host pinned]
  expert stacks     per-layer, three projections, cuMemHostRegister(PORTABLE)
```

Load: weights → **warmup** (all staging rings and per-layer banks once) → serve. The ring set is
asserted frozen across decode. Static addresses keep the steady state allocation-free.

Banks are per-layer because the model is mixed-quant (7 expert shapes); shared structures carry
counts/offsets/pointers only.

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
Suspects: drafter KV cache vs `context_lens`, and the MTP site's CPU-expert rows. The schedule
sizing bug (MTP site = layer 48) is fixed.

---

## 6. Correctness

- **Oracle:** `frac 0` + `--no-lookahead` byte-identical to the frozen reference.
- **`frac > 0` is not byte-exact:** lookahead changes the resident/miss split, hence the
  `atomicAdd` terms. `frac 0.55` + `--no-lookahead` has exactly **two attractors** (2/20 diverge,
  both to the same text); `frac 0` is 0/10. A third variant is the regression signal.
- **Structural checks** and a must-fail control for every wait (a stale stamp must hang). Correct
  values alone are not evidence.
- **NaN fix:** two uninitialised buffers — schedule payload (`cudaHostAlloc` does not zero) and
  paged-attention KV blocks (`Tensor::empty`). Zeroing both: 2–3/5 → **0/6**.
- **Measurement:** medians of ≥ 3 runs; report round p50 + tokens/round + hit rate + flags;
  liveness ≥ 6 runs; no mean where a median exists.
- **One fixed configuration per decision.** Env vars are probes/overrides, not A/B arms. Current
  surface (6): `NAIWA_STEP_DBG`, `NAIWA_ROUND_STATS`, `NAIWA_RET_DBG`, `NAIWA_FUSED_DBG`,
  `NAIWA_LOOKAHEAD_DBG`, `NAIWA_GPU_ONLY_EXPERTS` (lossy bound). The oracle switch is the `--no-lookahead` flag.
- **Pitfalls:** host-written per-step targets race (use device epoch); publish FIFO; `bump_epoch`
  needs the mapped **device view**; publish/gate replay must share `staging_enabled()` (a copy
  stream exists at `frac 0`).

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
  src/doorbell/            Doorbell/FlagBuf/ScheduleBuf + kernels
    mod.rs ffi, topk.rs, flag.rs, schedule.rs, tests.rs
  src/expert_cache/        GPU expert cache
    mod.rs                 module docs, constants, slot/bank/layer types
    provider.rs            RoutedExpertsProvider + layer construction
    layer.rs               slots, fills, LRU replacement, lookahead prediction
    decode.rs              run_hits, decode_forward
    staging.rs             --pcie-frac ring + DMA of misses
    registry.rs            layer lookups + prefill loan
    expert_bytes.rs        per-expert byte slicing
  src/stats/               counters + NAIWA_* measurement probes
    mod.rs                 shared gate/window/header/units; re-exports record_*/dump_*
    cache.rs               expert-cache counters (`dump_cache_stats`)
    lookahead.rs           LOOKAHEAD_DBG
    round.rs               ROUND_STATS
    step.rs                STEP_DBG + ROUND_DBG
    fused.rs               FUSED_DBG
    retired.rs             RET_DBG
  src/deferred.rs          CPU worker + CPU-return channel
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
