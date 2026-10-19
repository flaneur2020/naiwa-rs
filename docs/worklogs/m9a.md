# M9a — expert_cache concept reduction

Status: **done**. Branch `m9a-expert-cache-concepts` (`7601fbf`, `08c369a`), merged to main after the gates
below. Baseline: `f9b862e` (expert_cache has private submodules and a single export surface).

## Why

The module split fixed *where* things live; the next problem is *how many concepts* the cache exposes
internally. Today one cache slot is represented by five parallel tables, and two protocol wrappers only forward
to another object. Every slot mutation has to keep those tables consistent by hand.

Goal: make a slot a single state machine, make the two slot-buffer modes a type-level choice, and delete the
forwarding types. No behaviour change.

## Current concepts

| concept | role | overlap |
|---|---|---|
| `SlotWeights` | gate/up/down `Arc<QTensor>` | duplicates `Bank`'s three fields |
| `Bank` | `first` + stacked gate/up/down | same three fields as `SlotWeights` |
| `SlotBuf` | `weights: Option`, `memory: Option`, `bank: Option`, `fence` | three options encode two legal modes |
| `SlotEntry` | `Mutex<Option<Arc<SlotBuf>>>` | a field type with its own name |
| `ScratchSlot` | `Arc<SlotBuf>` | pure wrapper |
| `CachedLayerState` | `slots` map + `resident` + `published` + `last_use` + `slot_bufs` + `usable` + `tick` + `hit_counts` | five per-slot tables |
| `CacheLoan` | holds `Arc<LayerRegistry>` for `ExpertMemoryLender` | pure forwarder |
| `LookaheadChannel` / `LookaheadShared` | channel wrapper plus shared counters | one `Arc` indirection |

The five per-slot tables are `resident[slot]`, `published[slot]`, `last_use[slot]`, `slot_bufs[slot]`, and the
reverse `slots: HashMap<expert, slot>`.

## Target model

```rust
#[derive(Clone)]
struct Projections {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
}

struct Bank {
    first: usize,
    proj: Projections,
}

enum SlotData {
    Owned(Projections),                              // fresh per-expert QTensor
    Banked { bank: Arc<Bank>, memory: [SlotMemory; 3] }, // view into a copy-stream bank
}

struct SlotBuf {
    data: SlotData,
    fence: ReadFence,
}

struct Slot {
    expert: Option<usize>,       // reserved or published
    buf: Option<Arc<SlotBuf>>,   // published; readers clone this Arc
    storage: Arc<Mutex<Option<Arc<SlotBuf>>>>, // copy-stream permanent buffer, None while lent
    last_use: u64,
}

struct LayerState {
    doorbell: Option<Arc<TopkDoorbell>>,
    doorbell_seq: u32,
    slots: Vec<Slot>,
    hit_counts: Vec<u64>,
    tick: u64,
    usable: usize,
}
```

Slot states become explicit:

```
Empty      expert=None, buf=None
Reserved   expert=Some, buf=None
Published  expert=Some, buf=Some
Lent       expert=None, buf=None, storage=None
```

`storage` is an `Arc<Mutex<...>>` so `upload_in_place` and `lend` can clone it under the state lock and then
release that lock before waiting for readers. Lock order stays: `state lock -> clone storage Arc -> release
state -> storage lock`.

### Slot operations

```rust
fn admit(&self, expert: usize, sync: bool) -> Result<Option<usize>> {
    let slot = {
        let mut st = self.state.lock().unwrap();
        if st.slots.iter().any(|s| s.expert == Some(expert)) {
            return Ok(None);
        }
        let Some(slot) = st.slots[..st.usable].iter().position(|s| s.expert.is_none()) else {
            return Ok(None);
        };
        st.slots[slot].expert = Some(expert);
        slot
    };
    self.load_reserved(expert, slot, sync)
}

fn replace(&self, expert: usize, protect: &[usize], sync: bool) -> Result<Option<usize>> {
    let slot = {
        let mut st = self.state.lock().unwrap();
        if st.slots.iter().any(|s| s.expert == Some(expert)) {
            return Ok(None);
        }
        let Some((_victim, slot)) = lru_victim(&st.slots, protect) else {
            return Ok(None);
        };
        st.slots[slot].expert = Some(expert);
        st.slots[slot].buf = None;
        slot
    };
    self.load_reserved(expert, slot, sync)
}
```

`lru_victim` takes `&[Slot]`; `hit` lookup scans `LayerState::slots` for `expert == id && buf.is_some()`.
With ~42–47 slots the scan is negligible against the GEMMs; if it ever profiles, add a derived map without
changing `Slot`.

## What gets deleted / merged

- `SlotWeights` -> `Projections`.
- `SlotEntry` -> `Slot::storage`.
- `ScratchSlot` -> `ScratchRing { slots: Vec<Arc<SlotBuf>>, cursor }`.
- `resident` / `published` / `last_use` / `slot_bufs` / `slots` map -> `Vec<Slot>`.
- `CacheLoan` -> `impl ExpertMemoryLender for LayerRegistry`.
- `LookaheadChannel` + `LookaheadShared` -> one `LookaheadChannel` with `Arc<ActivationDoorbell>` and the two
  counters.

Result: ~17 named types -> 14, five per-slot tables -> one. The diff is +246/−216 (net +30) because the
explicit `Slot` helpers cost a few lines; the win is the state machine and the invariants, not line count. The module layout
stays as is: `mod.rs` remains the state/type hub so children keep private field access; splitting the types into
another file would force `pub(in crate::expert_cache)` on fields without reducing concepts.

## Result

- `Slot { expert, buf, storage, last_use }` is the only per-slot state; `LayerState.slots` is the only table.
- `Projections` / `Bank` / `SlotData` make the two buffer modes explicit.
- `SlotEntry`, `ScratchSlot`, `CacheLoan` and `LookaheadShared` are gone.
- Hit lookup scans the slot vector; the np8 run shows no measurable cost.

## Phases

### Phase 1 — slot data (low risk)

- [x] `Projections`; `Bank` uses it.
- [x] `SlotData` + `SlotBuf`.
- [x] `ScratchSlot` removed; `ScratchRing` holds `Vec<Arc<SlotBuf>>`.
- [x] Adapt `alloc_bank`, `run_hits`, `stage_misses`, `upload_in_place`.

### Phase 2 — slot state (main win)

- [x] `Slot` and `LayerState`.
- [x] Remove the five per-slot tables and the reverse map.
- [x] Rewrite `admit`, `replace`, `load_reserved`, `lend`, `give_back`, hit lookup.
- [x] Adapt `fill_candidate`, `lru_victim`, `CachedLayer::spawn_fill`, tests.

### Phase 3 — wrappers

- [x] `impl ExpertMemoryLender for LayerRegistry`; delete `CacheLoan`.
- [x] Merge `LookaheadChannel` and `LookaheadShared`.
- [ ] Optional: rename `ProviderInner` to `Budget` (not taken; naming only).

## Gates after each phase

- [x] `cargo check --features cuda`: 0 naiwa warnings.
- [x] `cargo test --features cuda --bin naiwa-rs`: 33/33.
- [x] `frac 0` + `NO_LOOKAHEAD`: oracle byte-identical to `harness/m1/reference.oracle.txt`; also with
  `--prefill-stream` (exercises `lend`/`give_back`).
- [x] np8 counting prompt: 52.2 tok/s, inside the current 52–53 tok/s band.
- [x] `lent_slots_leave_service_and_come_back_refilled`, `fill_candidate` and `lru_victim` tests pass after
  the state rewrite.

## Non-goals

- Do not merge `Bank` into `Projections`: a bank is a shared allocation; projections are one expert's weights.
- Do not remove `CachedExperts`: `RoutedExperts::forward(&self)` needs an `Arc<CachedLayer>` for `spawn_fill`,
  and an `Arc::new_cyclic` + `Weak<Self>` field would add a self-reference concept instead.
- Do not remove `LayerRegistry` or `ProviderInner`: next-layer lookup + scratch-ring ownership and the build
  budget are real invariants.
- Do not move the core types out of `mod.rs` just to shrink the file; that would turn private fields into
  `pub(in crate::expert_cache)` fields.
