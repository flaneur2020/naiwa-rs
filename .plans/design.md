# naiwa-rs — design and roadmap

Status as of 2026-10-09, `main` at the fused-bank decode commit; fork pinned to `6f740bc`.
First CUDA runs done on ubi (RTX 4070 12 GB, PCIe Gen3, 30 GB RAM); measurements and logs are in
`benchmark.md`. Prefill runs at the PCIe floor. Decode: 8.7 tok/s uncached, ~13 with a 2 GB
cache, 15–17.6 with 4 GB (`benchmark.md` §4d–e). The CPU experts now overlap the next layer's
launches (`src/deferred.rs`); single-channel DRAM (~18 GB/s) bounds the CPU side.

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
| `mistralrs-core`, `mistralrs-server-core`, `mistralrs-quant` | `flaneur2020/mistral.rs`, branch `patch/strata`, pinned in `Cargo.lock` to `6f740bc` |
| `candle-core` | `huggingface/candle` rev `66a8cf18`, the revision mistral.rs uses |

`patch/strata` = upstream Qwen3.8-Flash-Next support (#2462) + these fork commits:

| Fork commit | Change |
|---|---|
| `6e7a09aa` | `mistralrs_core::moe_offload`: a `RoutedExpertsProvider` hook that can take over a MoE block's routed experts (Qwen4-Exp decoder layers and MTP head) |
| `f069ce87` | `MoEExperts::expert_qtensors()` exposes the stacked gate/up/down `QTensor`s (gather backend, replicated experts only) |
| `f069ce87` | `forward_gather` reads top-k from `topk_ids` instead of the config, so a caller can pass a subset of the routed experts |
| `781f149f` | `MoEExperts::from_qtensors(gate, up, down, ..)` builds routed experts over stacks the caller holds (shared, not copied), on any device; used for GPU prefill |
| `5cd26a9f` | CPU MoE fallback runs a GEMV over only the routed experts instead of every expert |
| `3e281961` | the routed-expert GEMV accepts the model's 4D activations |
| `0c2c9ad` | host PLE gather: rows read in parallel (rayon), table advised `MADV_RANDOM` (§3.6) |
| `6f740bc` | CPU routed GEMV splits each expert's rows across threads (32-row tasks), so 1–3 decode misses use every core |

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

Alternative that leaves `Cargo.toml` alone (used on ubi, which has no fork checkout): copy the
pinned checkout from `~/.cargo/git/checkouts/mistral.rs-*/<rev>`, edit it, and build with
`--config 'patch."ssh://git@github.com/flaneur2020/mistral.rs.git".mistralrs-core.path="…"'`
(same for `-server-core`, `-quant`). Restore `Cargo.lock` afterwards (`git checkout
Cargo.lock`). Switching between the git and path sources rebuilds the fork's CUDA kernels
(~18 min release on ubi); a later edit rebuilds in ~2 min.

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
2. Top-k is split into hits (resident) and misses. Hits run on the GPU, misses via
   `MoEExperts::forward` on the CPU with only the miss ids. In copy-stream mode a layer's
   slots live in **banks**: one `[slots, N, K]` allocation per projection (a main bank, and
   with `--prefill-stream` a tail bank of `loan_bank_slots` ≈ 14 slots that the loan lends
   and returns whole). The hits are grouped by bank and each group runs through
   `mistralrs_quant::indexed_moe_fused_decode` (4 kernels: quantize, gate+up+act, quantize,
   down+weighted sum), indexing by `slot - bank.first`. Fresh-upload mode (no copy stream,
   the CPU tests) runs one `QMatMul` chain per hit.
   **The misses do not block the step** (`src/deferred.rs`): they run on a persistent worker
   thread, and candle's stream gets a host function that waits for the worker, then a copy
   of the result from a pinned staging buffer. `decode_forward` returns right after queueing
   that, so the host launches the next layer's attention/norms/router while the CPU computes,
   and only blocks at the next layer's top-k read-back. The worker makes no CUDA calls; a
   failed computation is reported by the next `deferred::start`. Uncached decode
   (`CpuExperts`, `seq == 1`) uses the same path. The two run at once
   and are summed; router weights are applied once per side. **Every CUDA call stays on the
   calling thread**: it copies the miss inputs to the host, spawns a scoped thread for the
   CPU misses, then enqueues the hits' kernels (asynchronous, so they still overlap). The
   earlier layout, hits launched from the scoped thread while the caller copied the
   activation to the host, produced NaN logits once the lookahead swapped slots (G8).
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
- **Stays on the CPU:** chunks under `--prefill-stream-min-tokens` (default 1024, as in
  Strata; ~64 on ubi, see *Measured* below), layers whose stacks are not all pinned (an async upload from pageable memory would block the forward thread), and
  the MTP head.
- **Mixed-quant GGUFs:** when a layer's dtype or shape differs from what a buffer holds, the
  buffer is reallocated. The old memory is freed only after its in-flight upload completes.
- **VRAM, without `--expert-cache-gb`:** two layers of experts, allocated on first use and
  kept.
- **VRAM, with `--expert-cache-gb`: lent by the cache, no extra memory** (Strata's "loan").
  candle cannot view part of one allocation as another tensor, so lending means freeing one
  set of allocations and allocating another:
  1. *Borrow*, before the first long chunk. Each cached layer gives up an equal share of
     about two layers' bytes, taken from its tail slots (`CacheLoan`, `CachedLayer::lend`).
     Lent slots are emptied and unpublished, and `usable` shrinks so no admission, fill or
     lookahead touches them. An upload in flight into a lent slot holds that slot's lock, so
     the lend waits for it, and its result is discarded instead of published. The memory is
     freed in candle's stream order, so after any queued reader.
  2. Chunks then run as above. The buffers stay across chunks, including a short tail chunk.
  3. *Give back*, after 8 decode steps in a row (`seq == 1`); a long chunk resets the count.
     The buffers are freed after their last upload, the lent slots get fresh memory, and
     `spawn_fill` refills them in the background by routing count, so learned popularity
     survives. Until then, those experts are CPU misses.
  4. candle frees into a CUDA memory pool (`cuMemFreeAsync`), and freed memory returns at the
     next synchronize. Each hand-over therefore syncs candle's stream once between the free
     and the allocation. Whether peak VRAM really stays at `N` (pool fragmentation) is to be
     checked on hardware.
  5. If the cache is too small to lend two layers, the shortfall is warned about and comes on
     top of `N`.
- **Bandwidth:** every chunk streams every layer's experts once (25.2 GiB for the requant
  Coder GGUF), so a chunk costs a flat ~2.05 s on PCIe Gen3 whatever its length. The chunk
  the MoE sees is bounded by **both** `--max-num-batched-tokens` and
  `--max-prefill-chunk-tokens` (default 512, below the default threshold, so the stream
  silently never engaged until both were raised). Set the chunk ≥ the prompt; a tail chunk
  under the threshold falls back to the CPU (2100 tokens with chunk 2048: 5.4 s vs 2.2 s).
- **Measured (ubi, 2026-10-09):** 2048 tokens in 2.08 s (982 tok/s), 4096 in 2.94 s, 8000
  in 5.55 s (chunk 8192, peak VRAM 11.7 / 12 GB). Short prompts cost the same ~2.05 s, which
  beats CPU prefill (~18 tok/s) from ~37 tokens, hence the lower threshold. The nsys profile
  shows uploads at full PCIe speed and overlapping compute; the first-layer upload and the
  shared copy stream are not visible costs at this scale.
- **Shared copy stream:** decode-cache slot fills and layer uploads share one copy stream, so
  during prefill a layer upload can queue behind slot fills.
- **Tested here:** `from_qtensors` on CPU against per-expert `QMatMul` references, for the
  full top-k and a subset (bit-identical, max diff 0). Also lend and give back on a CPU
  cached layer: lent slots leave service and come back refilled by routing count.

### 3.5 Format support — actual, not intended

| Experts as | Cached? |
|---|---|
| GGUF block-quantized (Q4_0/1, Q5_0/1, Q8_0, Q2K–Q6K) | yes |
| `--isq` | expected yes (ISQ leaves a `QTensor`); not yet confirmed on a real load |
| BF16 safetensors | **no.** `expert_qtensors()` is `None` and every load gives up, so the layer never hits. It still takes its share of the budget. The module doc's "dense fallback" does not exist. |
| Fused / Cutlass / Cutile backends, sharded experts | no (`expert_qtensors()` is `None`) |
| GGUF IQ*/Q2_0/TQ*/MXFP4 experts | **refused at load** (`f6e5c78`, names the types): candle cannot run them. Convert with `~/requant/requant` to Q2_K/Q3_K/Q4_K (`benchmark.md` §2) |

### 3.6 The PLE n-gram table (fork, `qwen4_exp/ple.rs`)

Qwen3.8-Flash-Next adds per-layer embeddings looked up from a hashed n-gram table: 16 rows of
~1 KiB per token (`per_layer_token_embd.weight`, 26.8 GiB IQ4_NL in the Coder GGUF). It is
made device-resident only if it fits beside a 2 GiB reserve, which on a 12 GB card it never
does, so every step gathers rows on the host from the GGUF mmap and uploads them.

With `--cpu-moe` the pinned expert stacks hold ~25 of 30 GB RAM, so the table cannot stay in
page cache and every row is a major fault. Read one at a time, that left the GPU idle for
7.06 s of a 9.2 s 2048-token prefill. Since `0c2c9ad` the rows are gathered with rayon (disk
queue depth) and the table is advised `MADV_RANDOM` (no 128 KiB readahead per 1 KiB row):
0.48 s. It is still disk-bound and also costs decode a share of each token.

## 4. Verification status

| Check | Status |
|---|---|
| `cargo build --release --features cuda`, `cargo test` | clean on ubi (CUDA 13.1, sm_89); 23 tests pass |
| CUDA from non-main threads (decode hit thread, fill threads, lookahead) | runs without context/thread errors |
| Pinning ~25 GiB of expert stacks | works on ubi (no "could not be pinned" warning) |
| Prefill stream, no cache | **works**, at the PCIe floor (§3.4). Greedy output byte-identical between the pre- and post-PLE-fix binaries, short (CPU) and 2100-token (stream) prompts |
| Prefill stream + cache loan | works: 2048 tokens in 2.39 s with `--expert-cache-gb 2`, peak VRAM 11.07 GB |
| Expert cache decode | works since the G8 fix: 9.0–9.7 tok/s at 2 GB (7.1 uncached), no NaN over 2×3×128 tokens |
| Numeric parity: cached vs uncached | **128 greedy tokens byte-identical** (`serve`, short and 2100-token prompts, lookahead swapping throughout) |
| Hit rate | `bench` reports it: 63–69% at 2 GB, 74–81% at 4 GB on the synthetic bench prompt |
| Decode tokens/s | 8.2 uncached, ~11.5 at 2 GB, ~14 at 4 GB; profiled (`benchmark.md` §4d) |
| Fused bank decode | 128 greedy tokens byte-identical to uncached at 2 GB (two banks, a loan cycle) |

Things still to watch for on CUDA:
- **In-place fills.** No drift over 128 tokens; recheck over thousands of tokens and with
  larger caches (more swaps per token).
- **Parity.** CPU and CUDA kernels differ (CUDA quantizes activations to Q8_1), so compare
  logits with a tolerance (~1e-3 relative), never for bit equality.
- **Shutdown.** Some runs log `CUDA_ERROR_DEINITIALIZED` from mistral.rs's memory maintenance
  after the bench finishes (G11).

## 5. Known gaps

| # | Gap | Effect |
|---|---|---|
| G1 | `--expert-cache-gb` uses `value_parser!(f64)` | accepts `0`, negatives, `NaN`, `inf` |
| G2 | Layer count hardcoded to 48 for the budget split | wrong per-layer quota for any other layer count |
| G3 | BF16 experts: budget charged, nothing cached | silent no-op cache; module doc claims a dense fallback |
| G4 | `num_tokens > 1` is treated as prefill | batched decode (several sequences) and MTP verify windows run all-CPU and get no cache hits |
| G5 | Cache flags undocumented | README does not mention `--cpu-moe`, `--expert-cache-gb`, `--prefill-stream*` |
| G6 | ~~No hit-rate counters~~ done | `bench` prints hits, misses, lookahead swaps, dropped requests; no per-layer split or timing yet |
| G7 | First prefill fills in expert-id order | cold-start residency is arbitrary until routing counts or the lookahead correct it |
| G8 | ~~`--expert-cache-gb` yields NaN logits~~ **fixed** | cause: decode hits launched from a scoped thread while the calling thread did CUDA work too; bisection in `benchmark.md` §4c |
| G9 | Prefill-stream VRAM is not reserved up front | `serve` with default `--max-seqs` OOMs on a long prompt (B6); 4 GB cache + loan OOMs (B1). Failures surface mid-request, not at startup |
| G10 | `--prefill-stream-min-tokens` default is Strata's 1024 | on ubi the break-even is ~37 tokens; the right value is upload time × CPU prefill rate, i.e. hardware-specific |
| G11 | `CUDA_ERROR_DEINITIALIZED` warnings at exit | cosmetic so far; exit code is 0 |
| G12 | PLE rows come from disk every step | 0.48 s per 2048-token prefill and part of every decode step (§3.6) |

## 6. Roadmap

Ordered by priority. `benchmark.md` holds the measurements behind each item.

### Done
1. ~~G8, the expert-cache NaN~~ (§3.2, `benchmark.md` §4c).
2. ~~Parity, cached vs uncached~~: 128 greedy tokens identical.

### Done (decode, 2026-10-09)
3. ~~G6~~ counters; ~~profile a decode step~~; ~~CPU GEMV parallelism~~ (fork `6f740bc`);
   ~~fused per-bank hit decode~~; ~~residency-policy sweep~~ (lookahead swaps, Strata-style
   decayed LFU, LFU-gated swaps, wider prediction): the current policy (top-10 lookahead,
   LRU victim, 2 swaps) is at the knee; see `benchmark.md` §4d.

### Next — decode speed (~11.5 tok/s at 2 GB, ~14 at 4 GB)
4. ~~Host overhead between MoE calls~~: hidden behind the CPU experts by the deferred
   upload (`benchmark.md` §4e). Piecewise CUDA graphs (fork work) would now only help layers
   without misses; re-profile before considering them.
5. **Stream part of the misses to the GPU** (Strata's PCIe share): DMA a miss into a
   staging slot and run it in the same fused call. Host DRAM (~18 GB/s on ubi) is shared by
   the CPU GEMV and the DMA, so expect at most the DRAM headroom the GEMV leaves (~25%).
6. **More VRAM for the cache:** keep `token_embd` (341 MiB) on the host; let `serve` start
   with 3–4 GB (its startup headroom check fails where `bench` fits, G9).
7. **Prediction:** precision is ~66–69% (top-10 from layer *l*'s MoE input through layer
   *l+1*'s router); 53% of misses were not predicted at all, 37% were predicted but not
   resident in time. Ideas not yet tried: predict from layer *l*'s post-MoE residual (one
   step closer to *l+1*'s input), and prioritize copies by router weight.

### Next — prefill robustness
8. **G9:** account for the two prefill layer buffers (~1.35 GiB) when sizing the KV cache
   and expert cache, or fail at startup with the numbers, instead of OOM mid-request.
9. **G10:** lower the default threshold (e.g. 128), or derive it at load from the measured
   pinned H2D rate and stack bytes.
10. **Short chunks: upload only routed experts.** A 64-token chunk touches a fraction of
   each layer's experts but pays the full 2 s. Measure routed-expert coverage per chunk
   size first; if it is low below ~512 tokens, upload per expert (Strata's slot ring).
11. **Long prompts beyond one chunk** pay 2 s per chunk: keep chunks as large as VRAM
    allows, and consider lending more cache slots so some layers stay resident across
    chunks.

### Later
12. **G12:** a host-RAM or VRAM cache of hot PLE rows (n-gram hashes of a code prompt
    repeat), or a smaller table encoding, to take the disk out of the step.
13. Hide the first layer of each chunk; give prefill its own copy stream. Both small at
    current scale (one layer ≈ 42 ms), revisit after 9–10.
14. Hygiene: G1, G2, G3, G5.
15. **G4:** per-token hit/miss split so batched decode and MTP verify windows use the cache.
16. **G7:** warm start (save/restore residency; Strata's `STRP` profile format). Static
    residency alone is weak here (12% hit rate with no swaps), so this only shortens the
    first tokens.

### Out of scope
- **Streaming misses to the GPU** (Strata's PCIe share). Misses always run on the CPU.
- An expert arena owned by naiwa-rs (Strata's `PinnedArena`). The stacks stay owned by
  mistral.rs; we only advise and pin them.
- Multi-GPU expert tiers, remote experts.
