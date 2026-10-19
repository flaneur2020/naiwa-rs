# M9b — expert_cache decode components

Status: **done**. Branch `m9b-decode-components` (`7e20a3b`), merged to main after the gates below.
Parent: [m9a.md](m9a.md). Goal: split the decode path into `components/` with one entry, one exit and
explicit doorbell ownership per stage. No behaviour or performance change.

## Pipeline

```
x2d ──► Preload ──► Route ──► DmaMiss ──► CpuMiss ──► Gpu(residents) ──► gates ──► Gpu(staged)
         │          │         │            │
         ▼          ▼         ▼            ▼
   SYNC_PRELOAD  RoutePlan  SYNC_STAGING  CpuPending
                             (plan mutated)
```

`src/decode.rs` becomes the orchestrator: it calls the components in this order and replays the two gates.
The order is the current `decode_forward` order; the split must not reorder it.

## Components

| component | entry | exit | owns |
|---|---|---|---|
| `Preload` | `submit(layer, x2d, rows, hidden, pass)` | `SYNC_PRELOAD` published by the lookahead worker after the next-layer copies | `ActivationDoorbell` publish; the lookahead worker calls `CachedLayer::prefetch_predicted` |
| `Route` | `split(layer, topk_ids, weights, rows)` | `RoutePlan { hits, misses }` | `TopkDoorbell` read, slot hit lookup, tick/hit_counts/last_use update |
| `DmaMiss` | `stage(layer, &mut RoutePlan, pass)` | `SYNC_STAGING` on the copy stream; `plan.hits` gains the staged experts | `CopyStream`, staging gate publish; uses the cache's scratch ring |
| `CpuMiss` | `submit(&plan.misses, x2d)` | `Option<CpuPending>` | CPU job body; `ResultDoorbell` is owned by `executor result channel` |
| `GpuExperts` | `run(layer, hits, x2d, act)` | output tensor + read fences on every slot | fused decode / per-expert `QMatMul`; no doorbell |

`CpuPending::upload()` is the CPU-result exit: it queues the `ResultDoorbell` wait+copy and returns the
tensor to add. `Decode` performs the add, not the components.

## Types

```rust
// components/mod.rs
type Hit = (Arc<SlotBuf>, usize, f32);

struct RoutePlan {
    hits: Vec<Vec<Hit>>,
    misses: Vec<Vec<(u32, f32)>>,
}

struct CpuPending {
    inner: executor::Pending,
}

impl CpuPending {
    fn upload(self) -> Result<Tensor>; // queues the ResultDoorbell wait+copy
}
```

`RoutePlan` is plain data; components do not hold each other. `CpuPending` is private to
`components::cpu_miss`; the orchestrator only calls `upload()`.

## Files

```
src/main.rs          // mod components; mod decode;
src/decode.rs        // orchestrator only (no stage bodies)
src/components/
  mod.rs             // exports + pipeline order docs
  route.rs           // RoutePlan, Route::split, hit lookup
  preload.rs         // Preload::submit (ActivationDoorbell publish)
  dma_miss.rs        // DmaMiss::stage and CachedLayer::stream_pcie/stage_misses
  cpu_miss.rs        // CpuMiss::submit, CpuPending::upload, CPU job construction
  gpu.rs             // GpuExperts::run, run_expert, fused_decode
src/expert_cache/
  mod.rs             // private submodules + pub(crate) facade
  layer.rs           // CachedLayer lifecycle + LookaheadChannel/predict/prefetch
  staging.rs         // scratch ring + alloc_bank (cache state used by DmaMiss)
  provider_gpu.rs    // GPU-cache provider + build/warmup
  provider_cpu.rs    // no-cache CPU-only provider
  registry.rs        // LayerRegistry lookup + lender
```

Dependency direction is one-way:

```
components/ + decode.rs  ──►  expert_cache/ (pub(crate) facade)
expert_cache/            ──X──►  components/ or decode.rs
```

`stream_pcie`/`stage_misses` moved into `src/components/dma_miss.rs`; the scratch ring and
`alloc_bank` stay in `expert_cache/staging.rs` because they are cache state. `LookaheadChannel`,
`predict` and `prefetch_predicted` stay in `expert_cache/layer.rs` for the same reason; `Preload`
owns only the activation publish and worker submit.

### expert_cache facade

`expert_cache/mod.rs` keeps its submodules private and exports only what the components need:

- types: `CachedLayer`, `LayerState`, `Slot`, `SlotBuf`, `SlotData`, `Projections`, `Bank`,
  `LayerRegistry`, `LookaheadChannel`
- `CachedLayer` accessors: `state()`, `cpu()`, `slot_device()`, `hidden()`, `num_experts()`, `top_k()`,
  `act()`, `router()`, `lookahead()`, `lookahead_chan()`, `copy_stream()`, `pcie_frac()`,
  `registry()`, `layer()`, `preload_claimed()`
- `CachedLayer` mutation used by components/decode: `admit()`, `replace()`, `prefetch_predicted()`,
  `replay_gate()`, `staging_enabled()`
- `LayerRegistry`: `get()`, `all()`, `insert()`, `scratch_ring()`

`LayerState` and `Slot` fields are `pub(crate)` too: the route component updates `tick`,
`hit_counts` and `last_use` under the layer's state lock, and the gpu/dma components read slot
buffers. They are the shared state vocabulary, not a public API. `Hit` and the `*Plan` data types
belong to `components`, not `expert_cache`.

## Doorbell ownership

- `TopkDoorbell` — `Route`, lazily acquired in `LayerState`; the only blocking host read.
- `ActivationDoorbell` — `Preload`, one per layer; worker reads it and runs the router.
- `ResultDoorbell` — `CpuMiss` via the executor's result channel; the worker publishes, the stream wait+copy is
  queued by `CpuPending::upload`.
- `GateSet` — shared primitive. `Preload` publishes `SYNC_PRELOAD` with `publish_stream_max`; `DmaMiss`
  publishes `SYNC_STAGING` with `publish_stream`; `Decode` replays both gates.
- `GpuExperts` has no doorbell; it records `SlotBuf` fences after launching its kernels.

## Invariants

1. Every producer publishes its gate on every path, including empty/failure, or the GPU spins.
2. `Preload` publishes with `atomicMax`; a stale producer must not move `SYNC_PRELOAD` backwards.
3. `DmaMiss` publishes after the copies on the copy stream, so the gate is stream-ordered after the DMA.
4. Residents run before the staging gate; staged experts run after it. Do not merge the two `GpuExperts`
   calls behind the gate.
5. `CpuMiss` has no gate: its `ResultDoorbell` seq is the completion point; `upload` queues the wait+copy.
6. `GpuExperts` fences every `Arc<SlotBuf>` it read before the caller drops the hits.
7. Components stay synchronous and don't own scheduling; `Decode` owns the order and the gate replay.

## Non-goals

- No new doorbell or sync primitive; reuse `TopkDoorbell`, `ActivationDoorbell`, `ResultDoorbell`, `GateSet`.
- No generic `Component` trait; plain structs with one public method are enough and keep concepts small.
- No change to `route` selection, quantization, cache admission, or lookahead policy.
- No change to the current `--pcie-frac`, `--no-lookahead`, or prefill-stream behaviour.

## Gates

- `cargo check --features cuda`: 0 naiwa warnings.
- `cargo test --features cuda --bin naiwa-rs`: 33/33.
- `frac 0` + `NO_LOOKAHEAD`: byte-identical to `harness/m1/reference.oracle.txt`.
- `frac 0` + `NO_LOOKAHEAD` + `--prefill-stream`: byte-identical (exercises lend/give_back).
- np8 counting prompt: round p50 / tok/s inside the current 52–53 tok/s band.

## Tasks

- [x] Add `mod components; mod decode;` to `src/main.rs`.
- [x] Add the `pub(crate)` facade in `expert_cache/mod.rs`; keep submodules private.
- [x] Add `src/components/mod.rs`, `Hit`, `RoutePlan`.
- [x] Move top-k read and hit/miss split to `route.rs`.
- [x] Move `run_hits` / `run_expert` / `fused_decode` to `gpu.rs`.
- [x] Move CPU job construction and `Pending` upload to `cpu_miss.rs`.
- [x] Move `stream_pcie` / `stage_misses` to `dma_miss.rs`; keep the scratch ring in `staging.rs`.
- [x] Move the preload submit block to `preload.rs`; keep `LookaheadChannel`/`predict`/`prefetch` in the cache.
- [x] Move `decode.rs` out of `expert_cache/`; it is now 198 lines of orchestration.
- [x] Gates: 0 warnings, 33/33 tests, oracle identical (also `--prefill-stream`), np8 53.3 / 52.2 tok/s.
