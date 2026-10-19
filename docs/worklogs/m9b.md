# M9b — expert_cache decode components

Status: **planned**. Parent: [m9a.md](m9a.md). Goal: split the decode path into `components/` with one
entry, one exit and explicit doorbell ownership per stage. No behaviour or performance change.

## Pipeline

```
x2d ──► Preload ──► Route ──► DmaMiss ──► CpuMiss ──► Gpu(residents) ──► gates ──► Gpu(staged)
         │          │         │            │
         ▼          ▼         ▼            ▼
   SYNC_PRELOAD  RoutePlan  CpuPending  SYNC_STAGING + StagedHits
```

`decode.rs` becomes the orchestrator: it calls the components in this order and replays the two gates.
The order is the current `decode_forward` order; the split must not reorder it.

## Components

| component | entry | exit | owns |
|---|---|---|---|
| `Preload` | `submit(layer, x2d, pass)` on the token thread | `SYNC_PRELOAD` published by the lookahead worker after the next-layer copies | `ActivationDoorbell` / `LookaheadChannel`, next-layer router + slot admit/replace |
| `Route` | `split(topk_ids, weights)` | `RoutePlan { hits, misses }` | `TopkDoorbell` read, slot hit lookup, tick/hit_counts/last_use update |
| `DmaMiss` | `stage(&mut RoutePlan, pass)` | `StagedHits` + `SYNC_STAGING` on the copy stream | scratch ring, `CopyStream`, staging gate publish; removes staged ids from `plan.misses` |
| `CpuMiss` | `submit(&plan.misses, x2d)` | `Option<CpuPending>` | CPU job body; `ResultDoorbell` is owned by `deferred::Channel` |
| `GpuExperts` | `run(hits, x2d, act)` | output tensor + read fences on every slot | fused decode / per-expert `QMatMul`; no doorbell |

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

struct StagedHits {
    hits: Vec<Vec<Hit>>,
}

struct CpuPending {
    inner: deferred::Pending,
}

impl CpuPending {
    fn upload(self) -> Result<Tensor>; // queues the ResultDoorbell wait+copy
}
```

`RoutePlan` and `StagedHits` are plain data; components do not hold each other.

## Files

```
src/components/
  mod.rs          // exports + pipeline order docs
  route.rs        // RoutePlan, Route::split, hit lookup
  preload.rs      // LookaheadChannel, Preload::submit, predict/prefetch
  dma_miss.rs     // DmaMiss::stage, ScratchRing, alloc_bank, LayerRegistry::scratch_ring
  cpu_miss.rs     // CpuMiss::submit, CpuPending::upload, CPU job construction
  gpu.rs          // GpuExperts::run, run_expert, fused_decode
src/expert_cache/
  mod.rs          // #[path = "../components/mod.rs"] mod components;
  decode.rs       // orchestrator only (no stage bodies)
  layer.rs        // CachedLayer lifecycle: admit/replace/load/lend/give_back
  provider.rs     // provider + build/warmup
  registry.rs     // LayerRegistry lookup + lender
```

`staging.rs` disappears: its scratch-ring and `stream_pcie`/`stage_misses` bodies move into
`src/components/dma_miss.rs`.

`src/components/` is declared from `expert_cache/mod.rs` with
`#[path = "../components/mod.rs"] mod components;`, so the components stay descendants of
`expert_cache` (they need `CachedLayer`/`Slot` private fields) while the files live at the requested
path. A crate-level `mod components;` would need a narrow `pub(crate)` facade from `expert_cache`;
that is a larger API change and is not part of this plan.

## Doorbell ownership

- `TopkDoorbell` — `Route`, lazily acquired in `LayerState`; the only blocking host read.
- `ActivationDoorbell` — `Preload`, one per layer; worker reads it and runs the router.
- `ResultDoorbell` — `CpuMiss` via `deferred::Channel`; the worker publishes, the stream wait+copy is
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

- [ ] Add `src/components/mod.rs` (declared from `expert_cache` via `#[path]`), `Hit`, `RoutePlan`,
  `StagedHits`.
- [ ] Move top-k read and hit/miss split to `route.rs`.
- [ ] Move `run_hits` / `run_expert` / `fused_decode` to `gpu.rs`.
- [ ] Move CPU job construction and `Pending` upload to `cpu_miss.rs`.
- [ ] Move `stream_pcie` / `stage_misses` / scratch ring to `dma_miss.rs`; delete `staging.rs`.
- [ ] Move `LookaheadChannel` / `predict` / `prefetch_predicted` to `preload.rs`.
- [ ] Reduce `decode.rs` to orchestration and gate replay.
- [ ] Run the gates above and merge only if all pass.
