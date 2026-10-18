# naiwa-rs v2 — design

Canonical **current** design. History, milestone gates, and evidence are in
[`roadmap.md`](roadmap.md) and [`worklogs/`](worklogs/). Last updated 2026-10-18 after M9.1.

---

## 1. Goal

The old engine was **host-bound**, not kernel-bound: `STEP_DBG` measured
`busy 0.000 ms/layer, dry 0.9–1.3 ms/layer` — every layer the GPU had already drained the
stream and was waiting for the host. 48 layers × ~1 ms ≈ TPOT.

The design changes the control model, not the kernels:

> **The GPU spins on mapped flags; the host publishes data and never leads the device by a
> pass.**

CUDA graphs were part of the original plan. They were built and measured **neutral**, then
deleted in M9.1. The shipped path is eager MoE segments and eager attention.

### Shipped configuration

| item | value |
|---|---|
| model / card | Qwen3.8-Flash-Next IQ1_M, 48 decoder layers + 1 MTP head / RTX 4070 12 GB |
| expert cache | 2 GB default (4 GB is slower) |
| `--pcie-frac` | 0.2 optimum at np3/np8; bandwidth knob, not a TPOT lever |
| experts | fork rayon pool, 8 threads |
| prediction | one dedicated event-driven thread; issue-only, never performs the DMA |
| staging | `S2`: residents run **before** the staging gate; staged experts after it |
| MoE / attention | eager; no capture |
| PLE | host-only table; not implemented on decode |
| MTP | works in steady state; long-prompt drafter prefill is an open fault |
| reference | np8, `frac 0.2`, cache 4, counting prompt: **170 ms round, 44 tok/s, 87.7% hit** |
| gate | `frac 0` + `NO_LOOKAHEAD` **byte-identical** to `harness/m1/reference.oracle.txt` |

`--depth N` is the **decode context length** (`4` in all current numbers); `--mtp-n-predict N`
is **draft depth** (np8 = 8 drafts, 9 verify rows). All performance numbers are at a 4-token
context, i.e. MoE-dominated; M8.4 will re-measure at 2k/4k.

---

## 2. Execution model

```
token thread                  GPU stream                         prediction thread / rayon
────────────                  ──────────                         ─────────────────────────
read top-k (doorbell)
split hits / misses
issue staging DMA  ────────→  copy stream
submit CPU job     ────────→                                     rayon: CPU expert gemv
launch residents   ────────→  [resident experts]
launch staging gate ───────→  wait SYNC_STAGING
launch staged      ────────→  [staged experts]
wait CPU payload   ────────→  wait_copy_slot, add to acc
                              next layer's attention (eager)
```

One layer, in stream order:

```
[resident MoE segment]            <- starts immediately
wait_ready(layer, SYNC_STAGING)   <- DMA lands under the resident compute
[staged MoE segment]              <- only now reads the staging ring
wait CPU payload, add             <- CPU contribution completes the MoE output
```

`S2` is unconditional: it measured **78.9 → 62.8 ms/pass (−20%)**, and the staging gate's p50
wait fell from 858 µs to 11 µs. The CPU add stays last: the CPU contribution belongs to this
layer's output, which the next layer consumes; moving it earlier removes cover.

The prediction thread runs only in the gap **after** a CPU-expert batch (running beside it cost
10 ms/step), and workers make **no CUDA calls** (W99: late predictions, hit rate 74% → 38%).

---

## 3. Synchronisation

### Schedule (staging and preload)

```
epoch    u32                 device-written once per pass (layer 0 prologue)
ready    [K] u32             one word per sync point; value = pass whose data is ready
payload  [slots][max_n] f32  CPU result block (legacy; see below)

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

`epoch` is **device-written only**, so the host can lag or match, never lead. The single-buffered
payload is safe because the host's pass advance is gated by a GPU-produced signal (the per-layer
router doorbell, after all layers of pass k). If that ever stops holding, add double buffering or
a `consumed` handshake.

Publication rules:

1. **Strictly FIFO per point.** All CPU publications, including “no misses → zeros”, go through
   the single worker queue. A synchronous publish can overtake the previous pass and move `ready`
   backwards.
2. **`ready` is a level.** Preload publishes with a one-thread `atomicMax(ready[id], pass)`;
   staging publishes with `cuStreamWriteValue32` after the copies. A stale publish must be
   harmless.
3. **Every staging path publishes** (nothing staged / staged / failed); its gate is replayed every
   pass. Preload's gate is replayed only when `inflight == pass`.
4. **Failure advances the point** (publish zeros), or the GPU spins forever.
5. `u32` wrap is ~4.29B passes; no handling needed.
6. Memory order: device epoch with fence before/after; host payload/ready with release; node
   `__threadfence_system()` after the spin. Ada did not need the fences here (all four fence arms
   passed), but they are kept for RDNA.

### CPU return (current state, M9.2 open)

After capture was deleted, the scheduler's `CPU_EXPERTS` point is unused. The live path is the
legacy per-layer `Channel` (`FlagBuf`: seq + payload ring), with `wait_copy_slot` on the stream.
The MTP site uses the same path.

`M9.2` must collapse this to **one** protocol: either move the MTP/decoder CPU return onto
`ScheduleBuf` and delete `Channel` (~200 lines), or document why `Channel` stays. Do not add a
third path.

### PLE

`PleHash::rows` is a **pure function of the last `ngram-1` token ids**; PLE is **one layer**,
before its attention. The ~100 GB hashed table stays on the host — never VRAM. If implemented,
PLE is the third host-produced DMA: compute rows (µs), mmap read → pinned → async H2D on the copy
stream, publish `SYNC_PLE` a pass early. It is **~0.1–0.3%** of a pass and is not needed now.

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

Load order: load weights → **warmup** (create all staging rings and per-layer banks once) → serve.
The ring set is asserted frozen across decode. Static addresses keep the steady state
allocation-free and the protocol auditable.

Mixed quant is why banks are per-layer: a shared captured graph could only hold one dtype/geometry
(historical; capture is now deleted). Shared structures carry counts/offsets/pointers only.

---

## 5. Capture policy

**Do not resurrect per-layer MoE graphs or attention-region capture.** Measured with capture on
vs off: **43.5 vs 44.1 tok/s (np8)**, i.e. neutral; attention capture was also neutral (43.9 vs
43.0). The machinery was deleted.

Rules if a future region is considered:

- capture pays only when it replaces **many** launches: `cuGraphLaunch` 12.7–15.4 µs vs
  `cuLaunchKernel` 5.1–6.0 µs; one-node graphs are a loss, a 5-launch MoE segment is neutral;
- a whole-step graph is **unreachable**: `decode_forward` contains a host spin on device output
  every layer, and device-initiated H2D does not exist; forcing the fork's step graph hangs;
- any captured region needs fixed input buffers, zero allocation/H2D/host control flow inside the
  window, and one shared memory pool with bounded retention (60 guarded regions OOM).

M8 also refuted “the host launch chain is the missing 70 ms”: with capture off the round was
identical. The residual host term must be found by **profiling the token thread** (candle per-op
overhead, sampler, handshake work), not by counting launches.

---

## 6. Performance envelope

### Reference

| configuration | round p50 | tok/s | tokens/round |
|---|---|---|---|
| np8, counting prompt, best | 184.7 ms | 43.3 | 8.00 (23.1 ms/token) |
| np8, M9 baseline | **170.0 ms** | **44.0** | 8.0 |
| np3 | 98.1 ms | 40.0 | 3.92 |
| np2 | 82–90 ms | 33.8 | 2.77 |
| no MTP | 63.2 ms | 15.8 | 1.0 |
| code continuation | - | 25–26 | 2.5–2.9 |
| Strata, 4070 Ti SUPER 16 GB, spec 4 + MTP | ~33 ms | 76.0 | ~2.5 |

M7 target (≤ 35 ms/token) is met on the counting prompt; acceptance, not the engine, decides
tokens/round. The gap to Strata is ~×2.5 speculation and ~×2.6 pass latency
(`11.6 × 2.5 = 29; 29 × 2.6 = 76`).

### np8 round budget (~185 ms)

```
GPU compute (MoE + attention + PLE)   40 ms   22%
host launch chain + sampler          ~70 ms   38%   <- not CUDA launches; profile needed
CPU-payload wait                      47 ms   25%   p50 611 us
staging wait                          27 ms   15%   p50 11 us, p90 1.06 ms
copy stream                           97 ms   overlapped, 53% duty
```

### Memory (np8, `frac 0.2`, cache 4)

```
CPU experts  745 MB (28%) | staged 1000 MB (37%) | preload 935 MB (35%) = 2.68 GB/round
14.5 GB/s = 51% of the 28.6 GB/s measured DRAM ceiling
PCIe 12.8 GB/s busy / 6.4 average (~25 ceiling); VRAM effectively unused
```

The limit is **consumption rate**, not byte supply. The CPU Q4_K dot is fine in isolation
(16.7 GB/s, 106 µs/expert cache-resident); the engine's 264 µs/expert is ~60% memory stall, so
fewer CPU experts or more staging is the only lever, and `frac` has an optimum at 0.2 because
staging and the CPU gemv contend for host DRAM.

A perfect cache buys only **+8%** (47.4 vs 43.9 tok/s). The entire miss path — CPU gemvs, waits,
DMA, preload — is under a tenth of the round.

### Ruled out (measured)

bigger expert cache (4 GB slower; 5.5 GB OOM) · second copy stream (~3.4 ms, not worth the
cross-stream change) · staging ring 16→64 (no reuse, only VRAM) · L3 warming (−7%) · stack
interleaving (1.00×) · huge pages (already 2 MB) · pinning (already pinned) · intra-expert
parallelism (already 24–120 tasks/job) · CPU threads > 8 · lower-bit CPU copies (breaks oracle) ·
GPU kernel micro-optimisation (GPU idle ~98%) · batching the CPU gemv (already one call).

`--lookahead-swaps` is a real knob: `1` = −11% round at np8 with ~3 points less hit; `2` default;
`0` collapses hit rate to 26.5%.

### MTP

The head is local (`~/bench-rel/mtp/mtp-component.gguf`, loaded via `--mmproj`). Steady-state
speculation is the biggest lever (×2.5 tokens/pass). Open fault: prompt 512 faults with
`CUDA_ERROR_ILLEGAL_ADDRESS` in `drafter_prefill_chunk` before `drafter_forward`; prompt 8 works.
Suspects: drafter KV cache vs `context_lens`, and the MTP site's CPU-experts rows. The schedule
sizing bug (MTP site = layer 48) is fixed.

---

## 7. Correctness

- **Oracle:** `frac 0` + `NAIWA_NO_LOOKAHEAD=1` is byte-identical to the frozen reference.
- **`frac > 0` is not byte-exact:** the lookahead changes the resident/miss split, hence the
  `atomicAdd` term set. `frac 0.55` + `NO_LOOKAHEAD` has exactly **two attractors** (2/20 diverge,
  both to the same text); `frac 0` is 0/10. A third variant is the regression signal.
- **Structural checks:** launch/capture/allocation/spin counts, and a must-fail control for every
  wait (a stale stamp must hang). Correct values alone are not evidence.
- **NaN history:** two uninitialised buffers — schedule payload (`cudaHostAlloc` does not zero) and
  paged-attention KV blocks (`Tensor::empty`). Zeroing both took failures from 2–3/5 to **0/6**.
- **Measurement:** medians of ≥ 3 runs; report round p50 + tokens/round + hit rate + flags;
  liveness ≥ 6 runs; no mean where a median exists.
- **One fixed configuration per decision.** Env vars are read-only probes, not A/B arms. Current
  probes: `NAIWA_STEP_DBG`, `NAIWA_ROUND_STATS`, `NAIWA_RET_DBG`, `NAIWA_FUSED_DBG`,
  `NAIWA_LOOKAHEAD_DBG`, `NAIWA_LH_SWAPS`; plus the lossy bound `NAIWA_GPU_ONLY_EXPERTS`.
  `NAIWA_NO_LOOKAHEAD` is the oracle determinism switch.
- **Pitfalls:** host-written per-step targets race (use device epoch); publish FIFO per point;
  `bump_epoch` needs the mapped **device view**; capture/publish/replay must share
  `staging_enabled()` (a copy stream exists at `frac 0`); zero accumulators inside any replayed
  segment; a one-node graph is slower than a plain kernel.

---

## 8. Fork boundary

Measured against upstream `bac32a67`: **41 commits, 77 files, +19 448 / −152**;
`mistralrs-quant` 65 files / +18 742, `mistralrs-core` 10 / +696.

| area | contents | verdict |
|---|---|---|
| native IQ/Q formats | `mistralrs-quant` ~14k lines, block tables, oracles, `mmq_instance_*.cu`, `mmvq_gguf.cu`, `iq*_grid.cuh`, `gguf/ffi.rs` | **must stay** |
| CPU fused MoE | `fused_cpu.rs`, `fast_mmq.rs`, `fast_mmvq.rs`, `cpu.rs` | **must stay** |
| GPU MoE kernels | `kernels/indexed_moe.cu`, `kernels/moe_grouped.cu` | **must stay** |
| quant CUDA wiring | `gguf/cuda.rs` (+701) | stays; `MoeDecodeCapture` remains but the engine no longer uses it |
| model/pipeline plumbing | `moe/experts/backends.rs`, `pipeline/mod.rs`, `paged_attention/mod.rs` | **must stay** |
| speculative drafting | `speculative/` +456 | stays |
| W-era probes | ~150 lines over 9 sites (`PLE_DBG`, `MOE_DBG`, `MTP_DBG`) | delete (M9.4) |
| PLE resident table | `make_table_resident`, `ResidentTable`, `quantize_shards`, planning constants (~300 lines) | dead; delete (M9.4) |
| doorbell | moved to `src/doorbell/` (−1660 lines) | done |

The only clean extraction was the doorbell. Everything else is a kernel or model plumbing.

---

## 9. Layout and commands

```
~/code/naiwa-rs            engine + docs/ + harness/
  src/doorbell/            Doorbell/FlagBuf/ScheduleBuf + kernels
  src/expert_cache.rs      banks, run_hits, decode_forward
  src/deferred.rs          CPU worker + CPU-return channel
  harness/m0..m7/          primitives, oracle, region-capture and bandwidth harnesses
~/code/mistral.rs-iq4xs    fork
~/code/qflash-rs           old eager oracle (read-only)
~/naiwa-rs/target          this crate's target
~/bench-rel/naiwa-v2/      binaries/logs; m9-baseline is the ablation reference
```

```bash
# oracle (must be IDENTICAL)
BIN=$HOME/naiwa-rs/target/release/naiwa-rs TAG=X FRAC=0.55 NAIWA_NO_LOOKAHEAD=1 \
  OUT=$HOME/bench-rel/naiwa-v2/m4 timeout 400 ./harness/m1/capture_oracle.sh
diff ~/bench-rel/naiwa-v2/m4/reference.X.txt harness/m1/reference.oracle.txt

# tests
cargo test --offline --release --features cuda     # 33/33

# reference bench (np8)
$BIN bench -m ~/models/qwen3-coder/requant -f '<shard1>;<shard2>' --cpu-moe \
  --expert-cache-gb 4 --pcie-frac 0.2 --depth 4 --mtp-n-predict 8 \
  --prompt-file <predictable-prompt> --gen-len 96
```

---

## 10. Pointers

- Current plan, gates, milestone history: [`roadmap.md`](roadmap.md)
- Latest cleanup/ablation: [`worklogs/m9.md`](worklogs/m9.md),
  [`worklogs/m9-baseline.md`](worklogs/m9-baseline.md)
- Per-milestone evidence: [`worklogs/`](worklogs/)
- Original architecture sketch (historical): [`architecture.png`](architecture.png)
