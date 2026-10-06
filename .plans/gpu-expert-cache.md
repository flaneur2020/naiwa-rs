# GPU expert cache for `--cpu-moe` (`--expert-cache-gb N`)

## Context

`--cpu-moe` (commit `ecf95e2`) keeps routed MoE experts in host RAM and computes them on the CPU.
It works, but the cost is exactly the bytes that must cross PCIe every token. The GPU expert cache
buys that back: keep the most-used experts resident in VRAM so decode moves only the misses.

`d2a712d ("feat: add gpu cache")` attempted this and **never compiled**. Two independent reasons:

1. **Wrong build.** `Cargo.lock` pinned mistral.rs to git commit `6e7a09aa`, so cargo built from
   `~/.cargo/git/checkouts/mistral.rs-00c9898cf8584438/6e7a09a/` — not `~/code/mistral.rs`. Every
   fork edit was invisible to the compiler.
2. **Nonexistent API.** It called `MoEExperts::dequantize_experts()`, which exists nowhere in the
   fork. `MoEExperts` dereferences to a *private* `backend` field and exposes only `new`,
   `new_direct`, and `forward`.

**Decision taken: never dequantize.** A GPU slot holds the expert's *quantized* bytes (GGML block
format), byte-identical to the CPU copy, so both sides run the same kernels and slot VRAM cost
equals CPU RAM cost. This invalidates the deleted design outright: it sized its slots in BF16, and
candle's CUDA quantized path *rejects* BF16 (`fast_mmvq::supports` / `fast_mmq::supports` match only
Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2K..Q6K). BF16/F16/F32 have `block_size` 1, so there are no blocks to copy
— dense experts take a separate copying path (see "Format matrix").

**Decisions taken:** per-layer slot ranges (Strata measured global admission at **2.97%** vs
**21.4%** for per-layer at the same VRAM — a global pool fills entirely with the first layer's
experts and never recovers); **misses fall through to the CPU** (never swap-then-retry on the token
path — keeps the path non-blocking and always correct); all three model formats in scope.

**Acceptance for this stage is a clean `cargo check`.** There is no CUDA host available (this is an
M1 Mac; no `nvcc`, no `nvidia-smi`). All real GPU verification is deferred, and the plan says so
explicitly rather than implying the feature is proven.

## Verified API facts

Established by reading the pinned candle rev (`66a8cf18`) and the fork, not assumed:

| Fact | Where |
|---|---|
| `QMatMul::QTensor(Arc<QTensor>)` is a **public** variant | candle `quantized/mod.rs:812-813` |
| `candle_core::quantized` is a **public** module | candle `lib.rs:78` |
| `QTensor` has **no** `narrow`, **no** `cat`/`stack`, **no** `to_device` | `quantized/mod.rs:536-903` |
| `QTensor::data() -> Result<Cow<[u8]>>` round-trips exactly (CUDA D2H excludes padding) | `mod.rs:758`, `cuda.rs:880` |
| `qtensor_from_ggml(dtype, &[u8], dims, &Device)` is public | `quantized/ggml_file.rs:138` |
| `QStorage::from_data(Cow<[u8]>, &Device, GgmlDType)` is public | `mod.rs:98` |
| Expert `e` of `[E,N,K]` is contiguous bytes `[e·stride, (e+1)·stride)` | `quantized.cu:4701`; `indexed_moe.cu:834` |
| Byte-concat == quantizing the whole stack (row-block formats never cross experts) | `gguf/mod.rs:200-202` |
| `QuantMethod::get_qtensor() -> Option<Arc<QTensor>>`, public | `mistralrs-quant/src/lib.rs:1862` |
| GGUF override returns `Some` | `gguf/mod.rs:518-522` |

Two traps this creates:

- **`QMatMul::from_arc` silently dequantizes BF16** (`mod.rs:841-845`) — the same shortcut the fork
  itself avoids via `from_qtensor` (`gguf/mod.rs:190-198`). Build `QMatMul::QTensor(..)` directly.
- **CPU and CUDA are not bit-identical.** CUDA quantizes activations to Q8_1 with a 512-element
  row-padded stride (`cuda.rs:384`, `:457`); CPU uses repacked AVX/NEON GEMV. Same weights, same
  block formats, different accumulation order. **Parity checks must compare with a tolerance**, and
  a parity test that asserts bit equality will fail for a correct implementation.

## The blocking fork gap

`FastExpertsWeights::forward_gather` reshapes `topk_ids` with `config.num_experts_per_tok`, a value
fixed at load (`backends.rs:1266-1272`). Passing only the *misses* — a variable subset of top-k —
breaks that reshape. This is unavoidable: running all top_k on CPU and zeroing the cached weights
would still read every expert's bytes on the host, defeating the cache entirely.

Note the CUDA flattened path (`uses_flattened_gather`, `backends.rs:1232-1241`) uses `topk_ids`
directly and would already accept a subset. **Only the CPU path needs the fix**, which is the path
this machine can actually run.

## Plan

### Stage 0 — plan location
Create `./.plans/` and keep this file there. (`.gitignore` currently lists only `/target`; add
`.plans/` to it if these are not meant to be committed.)

### Stage 1 — fork: two additive changes (already partly done)
Work in `~/code/mistral.rs` on `patch/strata`. Both changes must be **purely additive** so upstream
callers are bit-for-bit unaffected.

1. **Expose the quantized expert weights.** Add to `MoEExperts` a public accessor returning the
   three gate/up/down `Arc<QTensor>`s when the backend is gather-based:
   - `FastExpertsWeights` → `fused_gate_proj.get_qtensor()`, `fused_up_proj..`, `fused_down_proj..`
   - `Fused`/`Cutlass`/`Cutile` → return `None` (documented: those hold stacked `Tensor`s, not
     `QTensor`s; the cache declines rather than mis-slicing).
   - Return `None` for any dtype where `fast_mmvq::supports`/`fast_mmq::supports` is false, and log
     once. A cache that admits experts it cannot run would report a hit rate for slots it cannot use.
   - Do **not** resurrect `dequantize_experts`.

2. **Derive top_k from `topk_ids`.** In `FastExpertsWeights::forward_gather`, replace the
   `config.num_experts_per_tok` reshape with the actual trailing extent of `topk_ids`. Guard it: if
   the extent equals the config value, behaviour is unchanged.

Also needed: `qflash-rs/src/offload.rs` must reach the CPU `MoEExperts`'s weights. Confirm whether
`ctx.load_builtin` hands back accessors or whether the provider builds experts itself via `ctx.vb`.

### Stage 2 — qflash-rs: the cache (the bulk of the work)
New module `src/expert_cache.rs`; `src/offload.rs` keeps only the provider wiring.

**Format matrix.** `ExpertStorage` is an enum, chosen per block at load from what
`get_qtensor()`/the backend reports:

| Model | Storage | Slot bytes per expert | Compute |
|---|---|---|---|
| GGUF (`Q4_K` etc.) | `Quantized(Arc<QTensor>)` | `(N·K/block)·type_size` — e.g. Q4K `[640,2560]`: 640·2560/256·144 = **921,600 B** | `QMatMul::QTensor` |
| `--isq 4` | `Quantized(Arc<QTensor>)` | same (post-load QTensor) | `QMatMul::QTensor` |
| BF16 safetensors | `Dense(Tensor)` | `N·K·2` — **4× larger** | dense `matmul` |

The dense arm is why sizing must be computed from the actual storage, never from a fixed constant:
a BF16 slot costs 4× a Q4K slot, so the same `--expert-cache-gb` yields ~4× fewer slots.

**Weight plumbing (never dequantize).** For each expert `e` and projection `p`:
1. `p.data()?` → `Cow<[u8]>`.
2. Slice `[e·stride, (e+1)·stride)`, where
   `stride = N · (K / block_size) · type_size` (verify `K % block_size == 0` first; refuse the
   whole cache otherwise rather than mis-indexing).
3. `qtensor_from_ggml(dtype, &bytes, vec![N, K], &slot_device)`.
   Wrap as `QMatMul::QTensor(Arc::new(qt))` — **never `from_arc`**.
4. Cache one `QMatMul` per (expert, projection) in the slot.

Refuse rather than guess when: `K % block_size != 0`; dtype is a float fallback (`F32`/`F16`/`BF16`
arrived via a `Skip` behaviour — `gguf/mod.rs:199-215` shows ISQ can produce `F32`); `get_qtensor()`
is `None`; or the shard is not replicated (partial sums would need an all-reduce per slot).

**Admission — per-layer slots.** Layer `l` may use only `[l·q, (l+1)·q)`, `q = slots / n_layers`,
last layer taking the remainder (the layout Strata's `layer_slot_range` uses). Within its range a
layer fills empty slots in arrival order; **no eviction in this stage** — a fixed-per-layer policy
that also evicts has two effects moving at once, and the hit rate then cannot be attributed to
either. Count-based promotion is a follow-up.

**Device-layer count.** The provider needs `n_layers` to divide the budget, but it sees one block at
a time. Resolve this without guessing: read `num_hidden_layers` (+ MTP layers) from the model config
(`qwen4_exp/config.rs:50`) before install, and pass it into the provider. If it cannot be
determined, refuse — do not divide by a guessed number, since a wrong `q` silently starves layers.

**Hit/miss dispatch** (`RoutedExperts::forward`):
- **Prefill** (`num_tokens > 1`): all experts are needed, so route everything through the CPU and
  use the pass to update usage counts. No GPU work — a cache cannot help when every expert is used.
- **Decode** (1 token): split top-k into hits (in a resident slot) and misses. **Hits** run on the
  slot device; **misses fall through to the CPU**, as decided. Sum with weights applied **once, on
  one side** — applying the router weight on both and summing double-counts, which is invisible in
  one layer and compounds over 48.

**Concurrency.** Hits and misses are independent, but the previous attempt's threading was unsound:
it cast `&self.gpu_gate[slot]` to `*mut Tensor` and wrote through it while a spawned thread read the
same `Vec` — an aliasing violation and a real data race, not merely awkward. Slots must live behind
the lock (or be `Arc`-shared and immutable once filled). **Prefer the simple correct thing first**
(sequential), and only overlap if a measured profile justifies it — `std::thread::scope` is not free
and the CPU side already uses candle's threadpool.

### Stage 3 — CLI and guards
`--expert-cache-gb` already exists in the working tree. Fixes needed:
- **`value_parser!(f64)` accepts `0`, negatives, `NaN`, `inf`.** Add `parse_positive_f64` in the
  shape of `parse_positive_usize` (`args.rs:541-547`).
- `src/commands/tune.rs:11-13` bails on `cpu_moe`; extend it to name the cache flag explicitly, since
  `tune` never calls `base_builder` and the flag would otherwise be silently ignored.
- `install_cpu_moe` currently `bail!`s on the flag — replace with the real implementation. Keep the
  refusal for genuinely unsupported combinations (BF16-only stacks, sharded experts, unknown
  `n_layers`), worded in the repo's house style: lowercase, no trailing period, backticks.
- Update the module doc in `offload.rs` (it currently says the cache "is not implemented yet") and
  the README if the flag becomes user-visible.

### Stage 4 — build
`Cargo.toml` carries a TEMPORARY `[patch."ssh://git@github.com/flaneur2020/mistral.rs.git"]` block
pointing at `../mistral.rs`, so fork edits compile without a push. **Remove it (or convert to the
README's `path =` form) before committing**, and note the repo's README documents
`cargo update -p mistralrs-core` as the real update flow.

## Verification

**On this machine (M1, no CUDA) — all that is achievable now:**
```bash
cargo check --all-targets          # must be clean
cargo fmt --check
cargo test                         # unit tests only; no test loads a model
```
- **Refusal tests** (the pattern at `load.rs:340-351`, via `Cli::try_parse_from`): `--expert-cache-gb`
  requires `--cpu-moe`; `0` / negative / `NaN` are usage errors; `tune` refuses the flag. These run
  without a model or a GPU and exercise the real clap definition.
- **Byte-slicing unit tests**: build a tiny synthetic GGML `[E,N,K]` (the fixture style at
  `guard.rs:90-94` uses `candle_core::quantized::gguf_file::write`), slice expert `e`, and assert the
  extracted bytes equal `data()[e·stride..(e+1)·stride]` and that re-assembling all experts
  reproduces the original buffer exactly. This tests the one genuinely error-prone piece — the
  stride arithmetic — with no GPU.
- **Slot-arithmetic tests**: per-layer `q` and remainder distribution cover all slots exactly, with
  no layer able to reach another's range (the property Strata's `layer_slot_range` exposes for the
  same reason).

**Deferred to a CUDA host (cannot be done here, and must not be claimed as done):**
- Numeric parity: cached vs uncached decode on the same prompt, **compared with a tolerance**
  (~1e-3 relative on logits), never for bit equality.
- Hit-rate measurement and whether the cache actually reduces per-token time.
- Whether overlap between GPU hits and CPU misses helps — measure before threading.

## Open items resolved during this stage
- Whether `ctx.load_builtin` exposes weights or the provider must build experts from `ctx.vb`.
- Exact `n_layers` availability at install time (including MTP).
- Whether the ISQ path leaves a `QTensor` reachable at build time, or only a dense `Tensor` — if the
  latter, `--isq` falls into the dense arm and its slots cost 4× until investigated.
