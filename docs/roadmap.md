# naiwa-rs — roadmap

Plan of record: what is built, what was measured, what is judged, and what comes next.
Goal: a Rust implementation of Strata that runs Qwen3.8-Next-Flash on commodity hardware.
Design and current-state detail live in [`design.md`](design.md); per-milestone evidence lives in
[`worklogs/`](worklogs/).

Recorded 2026-10-16; last updated 2026-10-19 after M9c (doorbell events) and the M10 chat harness plan.

---

## Judging principles

Every step must be **feasible on its own** and **independently verifiable**:

1. **Numeric oracle** — a frozen eager reference: fixed prompt set, greedy decode.
   - `--pcie-frac 0` is bit-reproducible (8/8 across two dispatcher configs).
   - `frac 0` + `--no-lookahead` is **byte-identical** to
     `harness/m1/reference.oracle.txt`. This is the gate that means “no bug”.
   - `frac > 0` is not byte-exact: the lookahead changes the resident/miss split, hence the
     `atomicAdd` term set, so near-ties flip. `frac 0.55` + `--no-lookahead` has exactly **two
     recorded variants** (2/20 diverge to the same alternative text; `frac 0` is 0/10). A
     third variant, or the reference missing, is the regression signal.
2. **Structural criteria** — launch/capture/allocation counts, spin/`waited` counts, dry
   bucket. These do not depend on the 5–8% run-to-run noise. Performance is recorded, never
   used as a gate.
3. **One new mechanism per milestone** — so a failure has a single cause.
4. **A test that cannot fail proves nothing** — every wait/capture step carries a must-fail
   control (M0 baked target; M4 stale stamp).
5. **One fixed configuration per decision** (`design.md` §2, §6.5). Env switches are
   read-only probes, not A/B arms. Real flags (`--lookahead-swaps`, `--expert-cache-gb`,
   `--pcie-frac`, `--depth`, `--mtp-n-predict`) are workload knobs and stay.

Measurement protocol: medians of ≥ 3 runs; report round p50 + tokens/round + hit rate + exact
flags; liveness/memory claims need ≥ 6 runs; never report a mean where a median exists.

---

## Status summary

| milestone | mechanism | status | decisive result |
|---|---|---|---|
| **M0** | zero-argument doorbell primitives | ✅ 2026-10-16 | 1000/1000 visible, 200/200 replays, baked-target control fails |
| **M1** | static layout + descriptor-driven kernel | ✅ 2026-10-16 | bitwise kernel, byte-exact engine at `frac 0`; rings frozen |
| **M2** | per-layer MoE segment graph | ✅ 2026-10-16 | replay bitwise; 150 captures 11.6 ms |
| **M3** | CPU return path into the graph | ✅ 2026-10-16 | device epoch protocol; `frac 0` + `--no-lookahead` 5/5 |
| **M3.1** | move doorbell out of the fork | ✅ 2026-10-16 | fork −1660 lines; 5/5 oracle |
| **M4** | PCIe staging leg into the graph | ✅ 2026-10-17 | 5/5 at `frac 0.55`, 2/2 at `frac 0`; stale stamp hangs |
| **M5** | lookahead thread + async preload | ✅ 2026-10-17 | 81.3–83.8% hit, 78.2–86.7 ms TPOT; 11/11 oracle |
| **M6** | PLE decision + attention capture | ✅ re-scoped | Q2 settled; whole-step graph impossible; attention capture neutral |
| **M7** | memory throughput / per-token cost | ✅ 2026-10-18 | np8 43.3 tok/s, 23.1 ms/token; perfect cache only +8% |
| **M8** | cut the host launch chain | ✅ refuted | capture on/off identical; host term is not launches |
| **M9** | cleanup, simplification, ablation | 🔶 M9.1 done | S2 kept (−20%); graphs/probes/knobs deleted; `NAIWA_*` names 38 → 7 |
| **M10** | chat workload parity with Strata | 🔶 M10.3 items 1–3 done | reuse + prefill fixed (e2e 1.9–2.8×); TPOT broken down: 19 ms GPU chain + 48 × 0.73 ms 1-row CPU gemv. That gemv is the last term |

Current shipped path: eager MoE + eager attention, `S2` split, predictive preload, one
lookahead thread, rayon experts, device-epoch sync. **No per-layer graphs and no attention
capture** — both were measured neutral and deleted in M9.1.

---

## Dependency chain (historical)

```
M0 ── M1 ── M2 ── M3 ── M4 ── M5        M6 (originally: Q2 + F1 + F5)
                    └────── M5 measurement

M6 (re-scoped) ── M7 ── M8 ── M9 ── M10
```

M5's implementation only needed M2; its measurement needed M4 (host off the critical path).
M9 depends on all measurements above it: it deletes what M6–M8 proved worthless.
M10 measured the chat gap before any parity claim: TTFT is 28–265× off (no cross-turn prefix reuse,
13 tok/s serve-mode prefill); decode is 1.4–1.6×.

---

## M0 — zero-argument doorbell primitives — ✅ DONE 2026-10-16

- **Did**: proved host→GPU visibility of `wait_ge_ptr` / `wait_copy_ptr` with `target` and
  `slot_off` in mapped memory, not kernel arguments (A); and the same kernel captured into
  one CUDA graph and replayed 200 times while the host rewrites mapped values between
  replays (B).
- **Depends on**: nothing. Hard blocker for everything.
- **Judged by**: ① 1000/1000 visible and correct; ② 200/200 replays correct; ③ the kernel
  demonstrably waited (`waited == iters/replays`).
- **Result**: all pass on RTX 4070 (Ada). Two findings carried forward:
  - on Ada neither fence is load-bearing for this pattern (kept for RDNA portability);
  - **correct values are not evidence of no arg baking** — the baked-target control still
    produced correct values; only `waited == 0` caught it.
- **Artifact**: `harness/m0/`; binaries/logs `~/bench-rel/naiwa-v2/m0/`.
- **Evidence**: [`worklogs/m0.md`](worklogs/m0.md).

---

## M1 — static address layout + descriptor-driven kernel (no graph) — ✅ DONE 2026-10-16

- **Did**: ① `load → 7-ring warmup` (all quant shapes' staging rings and per-layer banks
  created once); ② all `FlagBuf` / descriptor / staging buffers allocated at load, addresses
  fixed for the process lifetime; ③ `indexed_moe_fused_decode` became descriptor-driven
  (`MoeDecodeCapture`, fixed grid `(⌈n/8⌉, max_topk, max_batch)`).
- **Depends on**: M0 + F4.
- **Judged by**: ① no new allocation / address change across a decode (assertion); ② under
  eager at `frac 0`, old host-computed geometry vs descriptor-driven is token-identical —
  bitwise at kernel level and byte-identical at engine level; ③ no graph/PCIe machinery.
- **Result**: `MoeDecodeCapture` bitwise identical to the host-geometry path; engine
  `serve --pcie-frac 0` byte-identical to the frozen oracle with 48 captures engaged; ring
  set frozen (3 rings, 0 created during decode). The F4 target was the **GPU**
  `indexed_moe_fused_decode`, not the CPU `fused_moe_decode`. No kernel change was needed:
  the launchers already derive the grid from value arguments.
- **Evidence**: [`worklogs/m1.md`](worklogs/m1.md).

---

## M2 — per-layer MoE graph skeleton (resident path only) — ✅ DONE 2026-10-16

- **Did**: the narrow `MoEGraphKey` registry, capture of the four-kernel MoE segment per
  bank, replay with live staging data; persistent `[max_batch, max_topk]` descriptors.
- **Depends on**: M1.
- **Judged by**: ① fully-resident config: greedy byte-identical to eager; ② the **MoE
  segment** goes from ~5 raw launches to **1 `cuGraphLaunch`** (not the whole layer: attention
  stayed eager); ③ the one-time cost of 150 captures measured — **11.6 ms**; ④ pad rows use
  a valid slot + weight 0 (`real_rows` deferred and never needed).
- **Result**: capture/replay bitwise (diff 0) for two different activations; `serve
  --pcie-frac 0` byte-identical (48 captures); 380 segment replays in the bench. Bugs found:
  the output accumulator must be zeroed *inside* the segment (`atomicAdd` doubled the result);
  `quantize_input_q8_1_into` allocates inside capture; event tracking must not be toggled
  globally.
- **Evidence**: [`worklogs/m2.md`](worklogs/m2.md).

---

## M3 — CPU result return path into the graph — ✅ DONE 2026-10-16

- **Did**: replaced the argument-taking wait with the **device epoch protocol**:
  `epoch` (device-written once per pass), one `ready` word per sync point, single-buffered
  payload, publication strictly FIFO per point, `wait_add_cpu` node per single-bank layer.
- **Depends on**: M2.
- **Judged by**: ① with misses, greedy == eager; ② misses handled **inside the graph** (a
  counter proves no eager path); ③ zero sync points on the token thread; ④ failure injection
  — a worker error still advances the point.
- **Result**: `--pcie-frac 0` + `--no-lookahead` → **5/5 byte-identical** to the frozen
  oracle. With lookahead on, the split itself varies per step, so checks compare the variant
  set. Three earlier variants (per-bank host cells, global dynamic seq, per-layer dynamic seq)
  all raced or hung; the epoch protocol removes the class.
- **Evidence**: [`worklogs/m3.md`](worklogs/m3.md); doorbell move in
  [`worklogs/m3-1.md`](worklogs/m3-1.md).

---

## M3.1 — move the doorbell out of the fork into the engine — ✅ DONE 2026-10-16

- **Did**: `src/doorbell/{mod.rs,doorbell.cu}` + own `build.rs` (`cudaforge`); 14 consumers
  switched; fork deleted `cuda/doorbell.{rs,cu}`, the FFI decls, and the re-export.
- **Result**: doorbell tests 5/5 here; `frac 0` + `--no-lookahead` 5/5 byte-identical; fork
  builds without the module; **−1660 lines**. Capture is per stream, not per crate, so this is
  safe even if a future fork-driven capture needs a doorbell node.
- **Evidence**: [`worklogs/m3-1.md`](worklogs/m3-1.md).

---

## M4 — PCIe staging leg enters the graph — ✅ DONE 2026-10-17

- **Did**: the 55% leg. The token thread issues the H2D; the copy stream publishes
  `SYNC_STAGING` with `cuStreamWriteValue32`; a per-layer one-node gate spins for it before
  the MoE segment. `CopyStream::queue_async` records no event; the per-streamed-expert
  `cudaStreamWaitEvent` is gone.
- **Depends on**: M3.
- **Judged by**: ① hit + PCIe + CPU == eager: **5/5 at `frac 0.55`**, 2/2 at `frac 0`; ② no
  recapture for any miss combination — all gates captured at warmup, ring set frozen; ③
  `frac` is only a bandwidth knob (`frac 0` captures/replays nothing, same output).
- **Must-fail control**: publishing a stale stamp (`pass - 1`) hangs, killed at 200 s. The
  spin is live.
- **Bugs/fixes**: the gate predicate must be `staging_enabled()` (`pcie_frac > 0 &&
  copy_stream.is_some()`); `copy_stream` exists at `frac 0` (cache fill), so the first version
  waited for a point nothing published and hung.
- **Evidence**: [`worklogs/m4.md`](worklogs/m4.md).

---

## M5 — lookahead on its own thread, experts on rayon — ✅ DONE 2026-10-17

- **Did**: reversed the W-era dispatch to the fixed configuration: expert batches on the
  fork's **rayon**; predictions on **one dedicated event-driven thread**. `expert_pool.rs`
  deleted (−700 lines) with its three A/B knobs. **M5b**: the prediction job only decides and
  *issues*; the preload is asynchronous on the copy stream and publishes `SYNC_PRELOAD` with a
  monotonic `atomicMax` kernel. Each layer replays a one-node preload gate only when that pass
  was claimed.
- **Depends on**: implementation M2; measurement M4.
- **Judged by**: ① hit rate **81.3–83.8%** (bar 78%; old pool 82.4%); ② STEP_DBG dry fell
  2.03–2.13 → **1.53–1.63 ms/layer** and TPOT 101.4 → **78.2–86.7 ms** (9.9 → 12.4–12.8 T/s);
  ③ no A/B switch — the configuration is in the code.
- **Gates**: `frac 0.55` + `--no-lookahead` 11/11 byte-identical; `frac 0` 2/2; 96 gates captured
  at warmup (48 preload + 48 staging), 0 launched warm, 0 predictions dropped.
- **Open risk carried forward**: one unexplained divergence in ~12 runs on an intermediate
  build; the final build was 11/11 clean and 27 consecutive clean runs across M4/M5b. Later
  characterised as a two-attractor rounding variant (see `design.md` §7.1).
- **Evidence**: [`worklogs/m5.md`](worklogs/m5.md).

---

## M6 — PLE + whole-step graph — ✅ RE-SCOPED / CLOSED

Originally: `SYNC_PLE` + whole-step graph, judged by `1 launch/step`.

What was learned and settled:

1. **Q2 answered**: `PleHash::rows` is pure (last `ngram-1` token ids), PLE is **one layer**,
   and the ~100 GB table stays on the host. PLE is not a new mechanism; if implemented it is
   the third host-produced DMA (`SYNC_PLE`, one word for the pass). The fork's resident-table
   path becomes dead (~300 lines).
2. **Whole-step graph is unreachable**: forcing the fork's decode graph on hangs because
   `decode_forward` contains a host spin (`db.wait_and_read`) while the ring kernel is only
   recorded. Device-initiated H2D does not exist, and staging depends on the just-computed
   top-k. The capture window must stop at the MoE segment.
3. **Per-layer gate graphs deleted** (M6 step 1): a one-node `cuGraphLaunch` (12.7 µs) is
   more expensive than `cuLaunchKernel` (5.1–6.0 µs). 384 warm waits/run become plain launches.
4. **Attention-region capture**: mix/inject capture worked bit-exactly; the mixer window
   needed the full paged-metadata resident set (cos_sin, mask, every metadata map, paged KV
   metadata; `TokenLayout` is tensor-free for decode). 12 regions verified; 60 regions OOM on
   first replay because live `CudaGraphMemoryPoolGuard`s stop pool release. Ultimately measured
   **neutral** and deleted in M9.1.
5. **PLE has nothing to do with the per-layer MoE graph**; it sits before its layer's
   attention, so it would only enter an attention/whole-step capture.

Consequence: M6's target `1 launch/step` is withdrawn. The re-scoped target was
`48 MoE + 48 attention + O(1)`; M8 then showed even the attention half does not move the round.

Evidence: [`worklogs/m6-graph.md`](worklogs/m6-graph.md) and
[`worklogs/m6.md`](worklogs/m6.md).

---

## M7 — memory throughput / per-token cost — ✅ DONE 2026-10-18

- **Headline**: predictable counting prompt reaches **40.0 tok/s at np3 / 25.0 ms per token**
  and **43.3 tok/s at np8 / 23.1 ms per token**; np4 40.9, np2 33.8. Code continuation:
  25–26 tok/s. The M7 target (≤ 35 ms/token) is met on the counting prompt.
- **Bytes**: np8 round moves **2.68 GB** (CPU 745 MB, staged 1000 MB, preload 935 MB) =
  14.5 GB/s over 185 ms, 51% of the measured DRAM ceiling. Links: DRAM pure read 28.6 GB/s,
  CPU dots 8.2 GB/s, PCIe 12.8 busy / 6.4 average, VRAM unused.
- **The bound**: `NAIWA_GPU_ONLY_EXPERTS` (perfect cache) buys only **+8%** (47.4 vs
  43.9 tok/s). The whole miss path is under a tenth of the round; 92% is GPU compute (~40 ms)
  + the host term (~70 ms).
- **Kept**: `--lookahead-swaps` as a real flag (`1` = −11% round at np8, ~3 points hit;
  `2` default; `0` collapses hit rate to 26.5%); `--lookahead-swaps`/cache/`frac` knobs.
- **Ruled out**: bigger cache (slower), second copy stream (~3.4 ms, dropped after
  measurement), ring 16→64 (no change), L3 warming (−7%), stack interleaving (1.00×), huge
  pages (already 2 MB), pinning (already pinned), intra-expert parallelism (already
  implemented), CPU threads > 8, lower-bit CPU copies (breaks oracle).
- **CPU diagnosis**: the Q4_K dot kernel is fine (16.71 GB/s, 105.9 µs/expert when
  cache-resident); the engine's 264 µs/expert is **60% memory stall**. No AVX2 change fixes
  it; only fewer CPU experts or more staging helps.
- Evidence: [`worklogs/m7.md`](worklogs/m7.md) (bytes) and
  [`worklogs/m7-per-token.md`](worklogs/m7-per-token.md) (per-token, depth sweep, M7.1–M7.5).

---

## M8 — cut the host launch chain — ✅ PREMISE REFUTED

- **Hypothesis**: 30–40 CUDA calls/layer × ~6 µs = ~70 ms/round, serialised against the GPU.
- **Decisive experiment**: `NAIWA_NO_CAPTURE=1` (eager segments) gives **44.1 tok/s /
  170.2 ms** vs graphs-on **43.5 tok/s** — identical within noise. Attention-region capture
  (M8.1) was also neutral: 43.9 tok/s / 173.6 ms vs 43.0 / 171.4. Therefore the ~70 ms “host
  launch chain” was a residual, not launches.
- **Consequences**:
  - M8.2 (fold one-node gates into segments) **dropped** — same refuted reasoning.
  - The host term must be found by profiling the host thread (candle per-op overhead,
    sampler, handshake work), not by counting CUDA calls.
  - The long-context re-measurement originally tracked as M8.4 is now covered by M10.4: every
    M6/M7/M8 number is at a 4-token context.
- **Target table withdrawn**: “30–40 → 1–2 calls/layer, 185 → ~125 ms, 23.1 → 15.6 ms/token”
  assumed the launch chain was the round; the measurement refutes that assumption. Do not
  plan against it until the host profile exists.
- Evidence: [`worklogs/m8.md`](worklogs/m8.md).

---

## M9 — cleanup, simplification, ablation — 🔶 M9.1 DONE

### Why

M6–M8 tested many mechanisms; several were **neutral or negative**, but the code and probes
still carried them. The engine was 9.6k lines (`expert_cache.rs` 3.3k) with **38 `NAIWA_*`
mentions**. The rule: a switch still in the tree is a decision that has not been made.

### Ablation table (measured; the agenda)

| mechanism | flag | measured effect | verdict |
|---|---|---|---|
| **S2: residents launch before the staging gate** | `NAIWA_STAGE_SPLIT` | **78.9 → 62.8 ms/pass (−20%)** | **keep, unconditional** |
| predictive preload / lookahead | `--no-lookahead` to disable | hit 85% → 20%, −25% throughput | keep, unconditional |
| `--lookahead-swaps 1` at deep draft | `--lookahead-swaps` | −11% round at np8 (165.5 vs 185.8 ms) | keep as a real flag |
| per-layer MoE segment graphs | `NAIWA_NO_CAPTURE` | **neutral** (44.1 vs 43.5 tok/s) | **deleted** |
| attention region capture | `NAIWA_ATTN_REGION` | **neutral** (43.9 vs 43.0) | **deleted** (~450 lines) |
| dispatch-on-device flip (M6 P2) | `NAIWA_DISPATCH_USE` | **neutral** (77.4 vs 78.0 ms) | **deleted** |
| L3 warming of CPU experts | `NAIWA_CPU_PREFETCH` | **−7%** | **deleted** |
| staging ring size | `NAIWA_SCRATCH_SLOTS` | 16 vs 64 identical | keep 16, knob deleted |
| expert cache size | `--expert-cache-gb` | 4 GB slower than 2 GB | keep 2 GB default + flag |
| one-node gate graphs | (deleted in M6) | −140 lines, +0 perf | done |
| CPU MoE worker count | `NAIWA_CPU_MOE_THREADS` | 8 optimal (10 was default) | keep, default 8 |
| GPU-only probe | `NAIWA_GPU_ONLY_EXPERTS` | lossy bound at +8% | keep as probe |

### M9.1 — done

| item | result |
|---|---|
| L3-warming probe (`NAIWA_CPU_PREFETCH`) | measured −7%, deleted with `host_memory::stack_bytes` |
| ring-size knob (`NAIWA_SCRATCH_SLOTS`) | 16 vs 64 identical, now a constant |
| attention region (`NAIWA_ATTN_REGION`, M8.1) | reverted to mix/inject-only, then deleted; np8 42.6/43.0/43.8 vs baseline 44.0, oracle identical |
| segment graphs | disengaged: 0 warmup captures, `fused_decode_capture` returns `None`; np8 44.6/43.1 vs 44.0, oracle identical |
| `attn_capture.rs` + M6 mix/inject region | deleted (~450 lines); eager mix/inject and mixer restored; np8 39.7/43.6 vs 44.0 (noise), oracle identical |
| capture code | `fused_decode_capture`, `capture_will_serve`, `Bank::create_capture`, `warmup_captures`, `waitadd_replay`, `fast`/`capture_bank` deleted; `expert_cache.rs` 3282 → 2989 lines; np8 43.0/43.4 vs 44.0, oracle identical |
| `NAIWA_*` env names | **38 → 7** unique names in `src/` + `harness/` (target < 12 met; M9 wrote 11 before the last deletions) |

**Acceptance for M9.1**: `cargo test` green; `frac 0` + `--no-lookahead` byte-exact; np8 round
p50 within 2% of 170 ms; < 12 `NAIWA_*` mentions. Met.

### M9.2 — one decision point, one CPU-return protocol (open)

Already landed:

- **device dispatch is the only path** (`c6578b0`): the GPU emits resident/staged/CPU lists into
  `DispatchBuf`; the host reads three lists and never hashes ids. The `NAIWA_DISPATCH_USE` probe
  is gone; the startup dummy run keeps the shape-check fallback.
- **S2 and the fused CPU path are unconditional** (`bbd1485`): `NAIWA_STAGE_SPLIT` and
  `NAIWA_CPU_MOE_GENERIC` are gone.
- **`--no-lookahead` is a CLI flag** (`b70ca7b`), not `NAIWA_NO_LOOKAHEAD`.

Remaining: two CPU-return paths still coexist — `ScheduleBuf`'s `CPU_EXPERTS` point (unused
since capture was deleted) and the live per-layer `Channel` + `wait_copy_slot` used by both the
decoder and the MTP site. Decide which is canonical:

- move decoder/MTP CPU return onto `ScheduleBuf`, delete `Channel` (~200 lines) and its
  fallbacks; or
- document *why* `Channel` stays.

Do not add a third path.

### M9.3 — one measurement harness, not eleven (open)

Keep exactly three probes and move them into one `probes.rs` with one env prefix and one doc
comment each:

- `NAIWA_STEP_DBG` — per-layer dry/busy + `ROUND_DBG`;
- `NAIWA_ROUND_STATS` — hit/staged/cpu split, staged MiB, CPU ms;
- `NAIWA_RET_DBG` — CPU job queue/compute/total.

Everything else is derivable or was a one-off.

### M9.4 — the model-level cleanup the fork is owed (open)

- Delete the W-era probes (~150 lines over 9 sites: `PLE_DBG`, `MOE_DBG`, `MTP_DBG`, the
  prefetch counter, `NAIWA_LH_MODE`).
- Delete the dead PLE resident-table path (~300 lines: `make_table_resident`,
  `ResidentTable`, `copy_iq4nl`, `quantize_shards`, `resident_bytes`,
  `planned_resident_table_bytes`). The table lives on the CPU only (`design.md` §5.3).

### Ablation protocol

For every deletion, record the number **before and after**, with the flag doing the before:

```
NAIWA_STEP_DBG=1 <flag>=0  -> round p50 baseline
NAIWA_STEP_DBG=1 <flag>=1  -> round p50 with the mechanism
```

A mechanism within the ~5% run-to-run spread is deleted; the pair of numbers goes into the
table. A mechanism that pays is made unconditional and its flag removed. “Keep the flag in
case” is not an acceptable outcome.

Gates for every commit: `frac 0` + `--no-lookahead` byte-exact; `cargo test` green; np8 round
p50 recorded.

### M9 baseline

```
file     ~/bench-rel/bin/naiwa-rs-m9-baseline
sha256   66f8577f3e294011c47d08e6dc9eeb616eb529472a7f5d1bed1ec34a8a1b5e95
built    naiwa-rs 2f24b5a  (M9.1 part 1: L3-warming probe and ring-size knob deleted)
fork     85913adf           (fused CPU MoE worker default 8, MTP head probe)
candle   59b60645           (q4k_micro)
size     274 549 200 bytes
```

Reference numbers (counting prompt, `--depth 4`, gen 96):

| configuration | round p50 | tok/s | hit |
|---|---|---|---|
| np8, `frac 0.2`, cache 4 | **170.0 ms** | **44.0** | 87.7% |
| + `NAIWA_NO_CAPTURE=1` | 170.2 ms | 44.1 | 87.1% |
| + `NAIWA_ATTN_REGION=1` | 173.6 ms | 43.9 | 87.0% |
| + `NAIWA_GPU_ONLY_EXPERTS=1` (lossy) | - | 47.4 | 100% |
| np2, `frac 0.2`, cache 4 | ~90 ms | 23–25 | ~82% |

After every deletion, compare against this table. A move outside 162–179 ms is a regression;
2–5% behind is investigated before acceptance.

Evidence: [`worklogs/m9.md`](worklogs/m9.md) and
[`worklogs/m9-baseline.md`](worklogs/m9-baseline.md).

---

## M10 — chat workload parity with Strata — 🔶 M10.1–2 DONE

Worklog: [`worklogs/m19-shared-expert-vram.md`](worklogs/m19-shared-expert-vram.md) — M19 (one expert-VRAM
arena: the prefill stream borrows the expert cache's top slots, strata's `Prefill::init(borrow)`/`relayout`
shape, so decode gets ~2 200 slots and prefill keeps two buffers instead of trading one for the other).
Worklog: [`worklogs/m18-prefill-transport.md`](worklogs/m18-prefill-transport.md) — M18 (the prefill stream's
transport: ~8 GB/s against a 13.4 GB/s link, a second buffer worth -16%, and the probe that has to say why).
Worklog: [`worklogs/m17-prefill-cost.md`](worklogs/m17-prefill-cost.md) — M17 (what a prompt chunk costs:
`chunk ≈ 2.93 s + 0.46 ms × rows`, i.e. 23 GiB of expert reads per chunk plus a per-row PLE/attention tax;
the ≤31-row checkpoint tail is the same term one size down). Worklog: [`worklogs/m16-mtp-reuse.md`](worklogs/m16-mtp-reuse.md) — M16 (chat with MTP: prefix reuse
under speculation; TPOT 55 → 43/44/52 ms, reuse restored to 0.76–0.96, turn-1 prefill 6.06 → 9.53 s is the
open cost). Worklog: [`worklogs/m10-chat.md`](worklogs/m10-chat.md) (multi-round chat harness + baseline; `worklogs/m10.md`
is the older MoE-naming milestone and unrelated). Harness: `harness/chat/`.

**Baseline (2026-10-19/20, cache-matched, concurrency 1):** turns ≥2 TTFT `short` 40.6 s vs 0.61 s with the
shipped `--mtp` (0% reuse), 11.4 s vs 0.61 s without MTP (0.73 vs 0.95 reuse); TPOT 59 vs 38 ms; end-to-end per
conversation 4–11×. Two configuration causes were found and fixed: `--mtp` forces
`SpeculativePrefixReplay::Full` (no reuse at all), and a checkpoint needs a block-aligned prompt step. The
prefill stream was then fixed too (its expert-cache lend does not return what it credits, and candle's pool
keeps freed blocks by size): `doc` turn 1 151.8 → 11.9 s, prefill 11–13 → 147–285 tok/s, e2e per conversation
1.9–2.8×. What is left is TPOT, now attributed: with every expert resident the same conversation runs at 19 ms/token
(faster than Strata's 38), and the real 60 ms is 48 layers × 0.73 ms of 1-row CPU expert gemv blocking the
pipeline. `NAIWA_RET_DBG` says the queue wait is 7 µs, so it is the kernel, not the hand-off. Next steps,
ordered and gated: worklog §"Next (M10.3 items 3–8)" — the 1-row gemv (approved fork work), and, behind it, whatever
the CPU expert path needs next. Prefill memory is settled: a warmup + first-chunk plan publishes the
prompt step from measured free VRAM, and row batching (256 rows a call) keeps the chunk's activation
footprint flat, so it picks the 8192 ceiling here (turn 1 8.5 → 6.0 s). Carving the buffers inside the
cache is not reachable in this architecture — it needs Strata's blob-ring prefill kernel; the measured
reason is in the worklog.

### Why

The current numbers are decode microbenchmarks at `--depth 4` (MoE-dominated, trivial
attention). They do not answer the product question: **does this engine serve a real chat
workload at Strata's performance?** M10 builds the measurement first, then closes the gap.

### Method: a chat harness, not another microbenchmark

Build `harness/chat/` around the OpenAI-compatible streaming endpoint, usable against both
engines:

- **Fixed corpus** (`corpus.jsonl`), replayed turn by turn with the full conversation history:
  short chat, multi-turn, long-context (≥2k), and code continuation. Each record fixes the
  messages, max tokens, temperature, seed, and stop conditions. The corpus is committed; the
  results are not.
- **Streaming timestamps** per request: first token (TTFT), every subsequent token (ITL/TPOT),
  final token, finish reason, usage. Tokens are counted from the stream, not from the prompt.
- **Same driver for both engines**: start the server, wait for readiness, warm up out of the
  measurement window, replay the corpus, repeat ≥3 times. Store one JSON result per run and a
  side-by-side summary.
- **Comparability record**: model artifact, quantization, card, driver, context length, cache
  size, `frac`, MTP depth, sampling, concurrency, prefix-cache state. A parity claim requires
  the same model and card; if Strata needs more VRAM, report the difference and do not call it
  parity.

Primary metrics, per workload class and aggregate:

| metric | definition |
|---|---|
| TTFT p50 / p90 | request start → first streamed token (prefill + scheduling) |
| ITL p50 / p90 | inter-token latency during decode |
| output tok/s | generated tokens / decode window (per request and aggregate) |
| end-to-end | request start → final token |
| MTP acceptance | accepted drafts / proposed drafts, when enabled |
| memory | peak VRAM and host RSS |

Concurrency: at least 1, 2, 4; report request throughput separately from per-request latency.
Measured at **1** so far: Strata serves one sequence behind a FIFO, so nothing else is like-for-like
until its slot/batch path is configured. Correctness is not measured here: the `frac 0` +
`--no-lookahead` oracle remains the gate.
Performance runs use temperature 0 and the same seed so the compared text is stable.

### Judging

1. **M10.1 — harness exists and is reproducible.** The same command produces the same schema
   for naiwa-rs and Strata; the corpus is fixed; warmup is excluded; results include the
   comparability record.
2. **M10.2 — baseline both engines.** Record the side-by-side table. Do not tune during this
   step; the point is an honest gap, per class and per metric.
3. **M10.3 — align.** Close the gap on the primary metrics. Initial parity band: naiwa-rs
   within **10%** of Strata on TTFT p50, ITL p50, and output tok/s at concurrency 1, and within
   **15%** at concurrency 4, with no unexplained per-class regression. If the first baseline
   shows the band is unrealistic, revise it **before** tuning, not after.
4. **M10.4 — long context.** The corpus includes ≥2k-token turns; this subsumes M8.4 and
   replaces the 4-token extrapolation with measured chat behaviour.

### Work items

- ✅ `harness/chat/corpus.jsonl`, `run.py`, `summarize.py`, `engines/*.json` and `README.md`.
- A server wrapper that pins the configuration (model, cache, `frac`, MTP, context) and emits
  the comparability record.
- A results schema and summary table: per class × concurrency × engine, p50/p90, tokens,
  acceptance, memory.
- Strata adapter: same corpus, same sampling; record its build/commit and flags.
- Bottleneck pass only after M10.2: TTFT → prefill/attention; ITL → decode/MoE/host term;
  throughput → batching/concurrency. Do not propose a mechanism before the per-class baseline
  says which term is short.

### Gates

- Harness reproducibility: same corpus + config → same schema; ≥3 runs; medians reported.
- Correctness unaffected: `frac 0` + `--no-lookahead` stays byte-exact.
- Alignment: the parity band above, or a documented, measured reason why the remaining gap is
  outside this engine's control (card, model, Strata-only kernel).

### Risks

- **Comparability:** different cards/models/flags can make a “parity” claim meaningless; the
  comparability record is mandatory.
- **Prefix cache:** multi-turn chat changes the prefill/cache state between turns; reset it per
  conversation and report it.
- **MTP:** the long-prompt drafter fault must be fixed before MTP-on chat parity is claimed.
- **Concurrency:** the current engine is tuned for batch 1; c=2/4 may expose a different
  bottleneck than c=1. Report both, never aggregate them into one number.

---

## Next actions, in order

1. **Finish M9.2/M9.3/M9.4** — one sync protocol, three probes, fork probe/dead-PLE deletion.
   These are hygiene and they remove ambiguity from every later measurement.
2. **MTP long-prompt fault** (`CUDA_ERROR_ILLEGAL_ADDRESS` in `drafter_prefill_chunk` at
   prompt 512). This is the largest throughput lever: speculation is ×2.5 tokens/pass.
   Steady state works; the schedule-size bug is fixed. Next diagnostic: targeted logging in
   `mtp.rs::forward` (shapes/devices, KV slot), then the drafter KV cache vs `context_lens`
   and the MTP `CpuExperts` path under draft row counts. `compute-sanitizer` is unusable on
   the 60 GB load.
3. **M10 — chat workload parity with Strata.** Build the harness (M10.1), baseline both
   engines without tuning (M10.2), then close the gap (M10.3). M10.4's ≥2k-token turns replace
   the short-context extrapolation, so there is no separate long-context task; start M10.1
   after M9.1, and claim parity only after the MTP fix.
4. **Root-cause the host term by profiling the token thread.** M8 showed it is not CUDA
   launch count. Candidates: candle per-op overhead (allocation, shape/layout bookkeeping as
   attention/PLE walk op by op), the sampler, per-layer handshake work. This profile is an
   input to M10.3 if ITL is the shortfall.
5. **Adaptive draft depth** — acceptance is workload-dependent: counting prompt peaks at
   np8 (43.3 tok/s), code continuation at np2–np3 (25–26 tok/s). Choose a policy that is
   within ~5% of the better fixed depth per prompt class, so depth is not a user knob; fold
   the result into M10.3.
6. **Optional / deferred**:
   - PLE host gather + `SYNC_PLE` (one word/pass, ~0.1–0.3% of a pass) — only after the
     larger items; not needed for correctness or the current throughput.
   - Attention-region capture — only if the host profile proves launches matter and a shared
     bounded capture pool exists. It was neutral and is deleted.
   - Per-layer MoE graphs — do not resurrect. They were neutral, and capture costs more per
     one-node/5-node segment than plain launches.

---

## Open risks

- **MTP prefill fault** at prompt 512 remains unexplained; steady-state is fine.
- **One unexplained divergence** was seen once on an intermediate M5 build (not reproducible;
  27 consecutive clean runs afterward). It is currently characterised as a two-attractor
  rounding variant, but the MTP/whole-step path would multiply sync points and make such a
  bug harder to localise. A ×50-run soak at `frac 0.55` is still the right guard before any
  large protocol change.
- **All performance numbers are short-context** (`--depth 4`); M10.4's long-context chat
  corpus is the correction.
- **The ~70 ms host term is unlocated.** M8 refuted the launch-count explanation; the next
  step is a host profile, not another mechanism.
- **The MTP head's schedule sizing** was an out-of-bounds bug and is fixed, but the same class
  (a site past the decoder layers) should be asserted at warmup.
- **`frac > 0` is never byte-exact**; only the variant set is. Any new protocol must keep the
  `frac 0` + `--no-lookahead` gate byte-exact or it is unverifiable.
- **Chat parity is unmeasured.** All current numbers are short-context decode microbenchmarks;
  M10 is the first workload-level comparison and its comparability record is mandatory.

---

## Verification matrix (historical)

| step | new mechanism | independent criterion (not performance) |
|---|---|---|
| M0 | mapped zero-arg spin primitives | visibility 1000/1000; replay 200/200; kernel demonstrably waited |
| M1 | static layout + descriptor kernel | eager token-identical; no new allocation / addresses fixed |
| M2 | per-layer capture | segment ~5 launches → 1; token-identical to eager |
| M3 | CPU return path in graph | misses handled in-graph (counter); dry bucket drops |
| M4 | PCIe path in graph | three-way sum == eager; capture count independent of misses |
| M5 | dedicated lookahead thread | hit rate holds the knee; dispatch not on critical path |
| M6 | PLE + whole-step graph | **withdrawn** — whole-step impossible; PLE deferred |
| M7 | memory/per-token | round p50 + tokens/round; never a single tok/s number |
| M8 | cut host launch chain | **refuted** — capture on/off identical |
| M9 | ablation | before/after round p50 per deletion; oracle byte-exact |
| M10 | chat harness | fixed corpus; reproducible schema; side-by-side TTFT/ITL/tok/s; parity band |
