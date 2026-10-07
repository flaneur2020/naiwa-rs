# qwen-fn-rs — design and roadmap

Status as of 2026-10-07, `main` with the prefill stream (§3.4); fork pinned to `781f149f`.

## 1. What this project is

A Qwen3.8-Flash-Next-only inference server built on the mistral.rs crates. The target is
running a model whose routed experts do not fit in VRAM on one consumer GPU, using Strata
(`~/code/Strata`) as the reference for how to place and stream those experts.

- Only `Qwen4ExpForConditionalGeneration` checkpoints (safetensors/UQFF) and GGUF files with
  `general.architecture = qwen4exp` are accepted. Local files only; `HF_HUB_OFFLINE=1`.
- Commands: `serve` (OpenAI + Anthropic compatible API), `bench`, `quantize`, `uqff`, `tune`.
- No web UI, LoRA, agent tools, sandbox or interactive shell.

### Dependencies

| Crate | Source |
|---|---|
| `mistralrs-core`, `mistralrs-server-core`, `mistralrs-quant` | `flaneur2020/mistral.rs`, branch `patch/strata`, pinned in `Cargo.lock` to `781f149f` |
| `candle-core` | `huggingface/candle` rev `66a8cf18`, the revision mistral.rs uses |

`patch/strata` = upstream Qwen3.8-Flash-Next support (#2462) + these fork commits:

| Fork commit | Change |
|---|---|
| `6e7a09aa` | `mistralrs_core::moe_offload`: a `RoutedExpertsProvider` hook that can take over a MoE block's routed experts (Qwen4-Exp decoder layers and MTP head) |
| `f069ce87` | `MoEExperts::expert_qtensors()` exposes the stacked gate/up/down `QTensor`s (gather backend, replicated experts only) |
| `f069ce87` | `forward_gather` reads top-k from `topk_ids` instead of the config, so a caller can pass a subset of the routed experts |
| `781f149f` | `MoEExperts::from_qtensors(gate, up, down, ..)` builds routed experts over stacks the caller holds (shared, not copied), on any device; used for GPU prefill |

**Fork policy: keep it minimal.** Everything else lives in this repo. The router is read
through the existing `ExpertLoadContext::vb`, not a new accessor.

**Fork workflow.** Edit `~/code/mistral.rs`, and for local iteration add a `[patch]` block at
the **end** of `Cargo.toml`. A block inserted mid-file swallows every dependency after it.

```toml
[patch."ssh://git@github.com/flaneur2020/mistral.rs.git"]
mistralrs-core = { path = "../mistral.rs/mistralrs-core" }
mistralrs-server-core = { path = "../mistral.rs/mistralrs-server-core" }
mistralrs-quant = { path = "../mistral.rs/mistralrs-quant" }
```

To land a fork change: commit it, then `git push flaneur2020 patch/strata` (`origin` is
upstream EricLBuehler). Remove the `[patch]` block, then run `cargo update` so the lock moves
to the new commit. `cargo update -p mistralrs-core` does not work while the patch is active.

## 2. Source layout

| File | Role |
|---|---|
| `src/main.rs` | CLI entry, logging, `HF_HUB_OFFLINE` |
| `src/args.rs` | clap definitions; `ModelOptions` shared by serve/bench/tune |
| `src/load.rs` | model selection, Qwen3.8-Flash-Next architecture checks (formerly `guard.rs`), device-layer map, `install_cpu_moe` call |
| `src/offload.rs` | `--cpu-moe` provider wiring; plain `CpuExperts` when no cache is requested |
| `src/expert_cache.rs` | the GPU expert cache (`GpuCacheMoeProvider`) |
| `src/host_memory.rs` | huge pages and CUDA pinning for the host expert stacks |
| `src/copy_stream.rs` | dedicated CUDA copy stream, in-place slot memory and read fences |
| `src/prefill_stream.rs` | `--prefill-stream`: GPU prefill experts from two streamed layer buffers |
| `src/commands/*` | one file per subcommand |

## 3. Expert placement, as implemented

### 3.1 `--cpu-moe`

Routed experts load into host RAM via mistral.rs's own `MoEExperts` and run on the CPU.
Attention, DeltaNet, routers, shared experts, the MTP head and the KV cache stay on the GPU.
Each MoE step moves activations and top-k ids to the CPU and the `[tokens, hidden]` result
back. Without `-n/--device-layers`, every layer is mapped to the GPU
(`ALL_LAYERS_ON_DEVICE`), because auto mapping still counts the taken-over expert bytes.
`tune` refuses `--cpu-moe`.

### 3.2 `--expert-cache-gb N` (requires `--cpu-moe`)

A per-layer VRAM cache of routed experts, layered on 3.1.

**Weights are never dequantized.** A slot holds an expert's GGML block bytes, sliced out of
the stacked `[E, N, K]` `QTensor` (`expert e = bytes [e·stride, (e+1)·stride)`). They are
rebuilt as a `[N, K]` `QTensor` with `qtensor_from_ggml` and run as `QMatMul::QTensor`, never
through `QMatMul::from_arc`, which would densify. A slot therefore costs the same in VRAM as
in RAM.

**Budget.** `N` GiB is split evenly over an assumed 48 decoder layers. Each layer gets
`min(budget / slot_bytes, num_experts)` slots, where `slot_bytes` is the expert's
gate + up + down bytes. MTP-head experts and CPU-mapped layers are not cached.

**Prefill** (more than one token in the step):
1. Every routed expert runs on the CPU, since all of them are needed anyway, unless
   `--prefill-stream` takes the chunk (§3.4).
2. Before that, the layer starts a background fill of layer *l+1*'s empty slots (and its
   own, which covers layer 0). At most one fill runs per layer; a full layer is a no-op.
   On CUDA the copy runs on a dedicated copy stream, so it overlaps this layer's CPU
   experts and the next layer's attention (§3.3). Elsewhere it goes through candle's stream.
3. Fill order is routing count, then expert id, so the first prefill fills in id order.

**Decode** (one token):
1. Layer *l* submits its activation to the lookahead thread. That thread runs layer *l+1*'s
   router on the CPU and loads the predicted top-k into *l+1*: empty slots first, then over
   the least-recently-used non-predicted resident, at most `MAX_LOOKAHEAD_SWAPS = 2` per
   layer per token. The queue is one deep and a busy worker drops the request, so decode
   never waits on it.
2. Top-k is split into hits (resident) and misses. Hits run via `QMatMul` on the GPU and
   misses via `MoEExperts::forward` on the CPU with only the miss ids. The two run at once
   (`std::thread::scope`) and are summed. Router weights are applied once per side.
3. **A miss is never copied to the GPU on demand.** If an expert is not resident in time, it
   runs on the CPU.

**Concurrency invariants:**
- A slot is reserved under the layer lock before its copy and published to readers only
  after the copy completes. A failed copy frees the reservation.
- A replaced slot is unpublished and re-reserved in one critical section.
- A token copies its hits' weight handles (`Arc<QTensor>`) under the lock, so a slot
  replaced mid-token cannot change what that token computes with.
- `LayerRegistry` holds weak references; the model owns the layers.

### 3.3 Host memory and slot uploads (`src/host_memory.rs`, `src/copy_stream.rs`)

**Why this exists.** candle's CUDA device has one stream that every thread uses, and a
`QTensor` upload (`qtensor_from_ggml` → `load_quantized` → `memcpy_htod`) is a
`cuMemcpyHtoDAsync` on it. From pageable memory, CUDA performs a **stream sync before the
copy** and stages the bytes through a driver buffer. A slot fill therefore used to wait for
all queued model work, then copy at roughly half bandwidth, in line with compute.

**Implemented (Phase A):**
- **No intermediate buffer.** A slot fill copies expert *e*'s byte range straight out of the
  stack (`expert_range`), instead of `to_vec()` first.
- **Huge pages** (Linux): every `--cpu-moe` stack is advised `MADV_HUGEPAGE`, and the pages
  already populated are collapsed with `MADV_COLLAPSE` (Linux 6.1+; older kernels leave it to
  khugepaged). Only whole 2 MiB pages inside a stack are touched. This helps the CPU GEMV for
  misses (TLB reach).
- **Pinning** (CUDA, cache layers only): each stack is registered with
  `cuMemHostRegister(PORTABLE)`. A slot fill is then a true async DMA. Pinning happens after
  the huge-page advice, because pinned pages cannot be collapsed. `PinnedStack` holds an
  `Arc` of its stack, so the memory stays valid while it is registered, and unregisters on
  drop.
- Both are best effort. A failure leaves the stack pageable, logs once, and is otherwise
  harmless.

Phase A makes copies faster, but on its own they stay ordered on candle's stream, between
kernels.

**Implemented (Phase B, `src/copy_stream.rs`): copy/compute overlap.**
- One extra CUDA stream per device, shared by its cached layers.
- At load, each slot's three projections are allocated once (`QCudaStorage::zeros`). The copy
  stream waits on an event recorded after those allocations.
- A fill writes expert *e*'s bytes into the slot's memory in place
  (`memcpy_htod_async` on the copy stream). It returns only after the copy stream has
  synchronized, so a slot is published fully written, and candle's stream never waits on
  the copy stream.
- **In-place hazard.** Overwriting a slot while queued kernels still read it would corrupt
  them. Two guards:
  1. Each slot has a `ReadFence` event. A decode thread records it on candle's stream after
     enqueueing the kernels that read the slot, even if computing a hit failed partway (if
     the record itself fails, it waits for the device). The copy stream waits on it before
     the overwrite.
  2. Replacement unpublishes the slot under the layer lock. The fill thread then waits until
     no reader still holds the slot's `Arc`, i.e. every earlier reader has recorded its fence.
- Off CUDA, or if the copy stream cannot be created, fills allocate a fresh buffer each time
  as before; old buffers are freed when their last reader lets go.
- Cost: slot VRAM is allocated eagerly at load (the same budget), instead of on first fill.

### 3.4 `--prefill-stream` (requires `--cpu-moe`, CUDA only)

Runs prefill's routed experts on the GPU, streaming each layer's experts in while the previous
layer computes. Works with or without `--expert-cache-gb`. Strata (`src/prefill/prefill.cpp`)
does the same job with a ring of per-expert slots and its own grouped kernels; this is the
simplest version that keeps the overlap:

- **Two whole-layer buffers.** Each holds one layer's three stacked `[E, N, K]` projections in
  VRAM and a `MoEExperts` over them (`from_qtensors`, fork `781f149f`). The GPU therefore runs
  mistral.rs's own MoE path: grouped MMQ for 32+ tokens, which reads each expert once per chunk.
- **Per layer *l*, nothing blocks the host:**
  1. Layer *l+1* is uploaded into the other buffer. That is one `memcpy_htod_async` per
     projection, straight from the pinned host stack, issued after the buffer's `released`
     fence (layer *l-1* done) and followed by recording `ready`. It overlaps layer *l*'s MoE
     and layer *l+1*'s attention.
  2. candle's stream waits on layer *l*'s `ready`.
  3. `forward` runs on the GPU, then `released` is recorded.
- **Not hidden:** the first layer of each chunk, which has no predecessor to prefetch it.
- **Stays on the CPU:** chunks under `MIN_TOKENS` (1024, as in Strata), layers whose stacks are
  not all pinned (an async upload from pageable memory would block the forward thread), and
  the MTP head.
- **Mixed-quant GGUFs:** when a layer's dtype or shape differs from what a buffer holds, the
  buffer is reallocated. The old memory is freed only after its in-flight upload completes.
- **Cost:** two layers of experts in VRAM, separate from the cache budget. Bandwidth: every
  chunk streams every layer's experts once, so this pays only for long chunks. Strata uses
  about 8K-token chunks. Raise `--max-num-batched-tokens` (e.g. 8192); the 512-token
  `--max-prefill-chunk-tokens` default may also bound chunks, which is still to be verified.
- **Shared copy stream:** decode-cache slot fills and layer uploads share one copy stream, so
  during prefill a layer upload can queue behind slot fills.
- **Tested here:** `from_qtensors` on CPU against per-expert `QMatMul` references, for the
  full top-k and a subset. The result is bit-identical (max diff 0).

### 3.5 Format support — actual, not intended

| Experts as | Cached? |
|---|---|
| GGUF block-quantized (Q4_0/1, Q5_0/1, Q8_0, Q2K–Q6K) | yes |
| `--isq` | expected yes (ISQ leaves a `QTensor`); not yet confirmed on a real load |
| BF16 safetensors | **no.** `expert_qtensors()` is `None` and every load gives up, so the layer never hits. It still takes its share of the budget. The module doc's "dense fallback" does not exist. |
| Fused / Cutlass / Cutile backends, sharded experts | no (`expert_qtensors()` is `None`) |

## 4. Verification status

| Check | Status |
|---|---|
| `cargo check --all-targets`, `cargo fmt --check` | clean (default features only) |
| `#[cfg(feature = "cuda")]` code (`host_memory`, `copy_stream`) | **not compiled by this build.** `candle-kernels` needs `nvcc`. Every cudarc call is type-checked in a scratch crate against cudarc 0.19.10; the candle glue (`Device::Cuda`, `cuda_stream()`, `QCudaStorage::zeros`, `QTensor::device_ptr`) is not |
| Linux-only code (`madvise`) | type-checked in a scratch crate for `riscv64gc-unknown-linux-gnu` |
| `cargo test` | 21 pass: architecture checks, device-layer map, expert byte ranges, single-expert `QTensor` equality, slot-byte accounting, fill order, top-k, LRU victim choice, huge-page span math, host-byte borrowing, reader wait, `from_qtensors` vs per-expert reference |
| In-place copy-stream path | **cannot run here.** Only exercised on CUDA; the non-CUDA build always takes the fresh-upload path |
| Prefill stream | **cannot run here** (CUDA only). Its buffer and event logic are unexercised until a CUDA run |
| Any CUDA run | **never.** No CUDA host available; the dev machine is an M1 Mac without `nvcc` |
| Numeric parity: cached vs uncached | not done |
| Hit rate, tokens/s | not measured |

Things that only a CUDA run will show:
- **CUDA from non-main threads.** The decode hit thread, fill threads and lookahead thread
  all touch the CUDA device. This is the riskiest untested assumption.
- **In-place fills.** Whether the fence protocol holds under load: look for corrupted output
  after slot replacements, which would show as parity drift that grows with decode length.
- **Pinning.** Registering tens of GiB takes seconds at load and may be refused (lockable
  memory limits); check the log for the "could not be pinned" warning.
- **Parity.** CPU and CUDA kernels differ (CUDA quantizes activations to Q8_1), so compare
  logits with a tolerance (~1e-3 relative), never for bit equality.

## 5. Known gaps

| # | Gap | Effect |
|---|---|---|
| G1 | `--expert-cache-gb` uses `value_parser!(f64)` | accepts `0`, negatives, `NaN`, `inf` |
| G2 | Layer count hardcoded to 48 for the budget split | wrong per-layer quota for any other layer count |
| G3 | BF16 experts: budget charged, nothing cached | silent no-op cache; module doc claims a dense fallback |
| G4 | `num_tokens > 1` is treated as prefill | batched decode (several sequences) and MTP verify windows run all-CPU and get no cache hits |
| G5 | Cache flags undocumented | README does not mention `--cpu-moe` or `--expert-cache-gb` |
| G6 | No hit-rate / timing counters | cache effectiveness cannot be observed |
| G7 | First prefill fills in expert-id order | cold-start residency is arbitrary until routing counts or the lookahead correct it |

## 6. Roadmap

Ordered by priority. Small, local, testable work comes first.

### Next — correctness and hygiene (no GPU needed)
1. **G1:** `parse_positive_f64` for `--expert-cache-gb`, plus clap tests in the
   `Cli::try_parse_from` style already used in `load.rs`.
2. **G3:** refuse BF16 experts with a clear message, or skip charging the budget for them,
   and fix the module doc.
3. **G2:** take the layer count from the model config / GGUF metadata before install and pass
   it to the provider. Refuse if it is unknown rather than guess.
4. **G6:** per-layer counters for hits, misses, lookahead loads, swaps and dropped lookahead
   requests, logged at debug level and reported by `bench`.
5. **G5:** document the two flags in the README.

### Then — first CUDA run (needs a CUDA host)
6. Build with `--features cuda` and run `serve` and `bench` with and without
   `--expert-cache-gb`. Confirm there are no CUDA context/thread errors.
7. Parity: same prompt, cached vs uncached, logits within tolerance.
8. Measure hit rate and tokens/s, then tune `MAX_LOOKAHEAD_SWAPS` and the per-layer split.

### Then — Phase B follow-ups (needs a CUDA host)
9. Profile a decode step (Nsight Systems): confirm fills run on the copy stream concurrently
   with compute kernels, and that candle's stream never waits on them.
10. Copy-stream count: one shared stream serializes all layers' fills; try one per layer if
    the profile shows fills queueing behind each other.
11. Batched DMA (`cuMemcpyBatchAsync`, CUDA 12.8+) for the three projections and several
    experts at once. Mostly a Windows/WDDM win per Strata; only worth it if profiling shows
    per-copy overhead.

### Then — prefill stream follow-ups (needs a CUDA host)
12. Measure tokens/s for long prompts with and without `--prefill-stream`, at several chunk
    sizes. Find which setting bounds the chunk the MoE sees, and whether 1024 is the right
    threshold for this hardware.
13. Profile: confirm layer uploads overlap attention, and whether they queue behind slot
    fills. If they do, give prefill its own copy stream.
14. Hide the first layer of each chunk, e.g. prefetch layer 0 at the end of the previous chunk.

### Later — extend the cache's reach
15. **G4:** run the hit/miss split per token so batched decode and MTP verify windows (a few
   tokens) use the cache, not just single-token decode.
16. **G7:** warm start, e.g. save learned residency on exit and reload it (Strata's
    `--expert-profile-save`), or fill from a static frequency profile.
17. Adaptive admission: decide whether lookahead swaps should be gated by routing frequency
    (Strata's LFU-style adaptive tier) once hit-rate data exists.

### Out of scope
- **Streaming misses to the GPU** (Strata's PCIe share: H2D copy + GPU compute for missed
  experts). Misses always run on the CPU.
- An expert arena owned by qwen-fn-rs (Strata's `PinnedArena`, loading the expert file
  itself). The stacks stay owned by mistral.rs; we only advise and pin them.
- Multi-GPU expert tiers, remote experts.
