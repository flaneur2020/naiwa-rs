# M26 - gate_up before down (splitting a layer's staging into waves)

**Status:** **closed negative**, same day it opened. Built, oracle-clean, measured: **no gain**.
**Parent:** M24 (§7 the decode is PCIe-bound at 394 MB per step, 71% of the link), M25 (closed negative: the
denominator cannot be raised by another drafter).
**One mechanism:** stage a layer's misses in **waves** - the copies of the first wave are followed by their own
completion event, so the GPU can start that wave's experts while the next wave is still crossing PCIe. The extreme
form of the idea is "upload `gate`+`up` first and `down` later", because the fused kernel needs `gate`/`up` before
`down`; a wave is the same trade at expert granularity, and it needs no change inside the fused kernel.

## 1. What was built

| layer | change |
|---|---|
| `mistralrs-quant/src/gguf/cuda.rs` | the fused decode was split into **`indexed_moe_gate_up`** (quantize input + fused gate+up+act, returns the activation) and **`indexed_moe_down_aggregate`** (quantize activation + fused down+weighted aggregate); `indexed_moe_fused_decode` is now their composition, so there is one implementation of each half and the projection-level split is available for a future milestone |
| `src/moe/cached_layer.rs` | per-wave completion events (`staging_waves`), `ensure_wave_events`, `wait_staging_wave`, and `wave_of_slot` |
| `src/moe/stages/dma_miss.rs` | `NAIWA_STAGING_WAVES` (1 = old behaviour, 2 = split); the copies of wave `w` are followed by event `w`, the layer's existing `staging_done` still closes the batch |
| `src/moe/decode.rs` | submit the staged experts wave by wave: wait wave `w`, submit it, and only then wait the next |

## 2. Gates

- **oracle**: `frac 0` + `--no-lookahead` is **byte-identical** to `harness/m1/reference.oracle.txt` (the wave path is
  inert at `frac 0`, but the capture was run anyway).
- `cargo test` 53 passed / 4 ignored.

## 3. Measurement: nothing moves, and the waits get worse

`code`, three 40-token turns, `--mtp-n-predict 1` (2 rows), cache 2.2 GB:

| arm | TPOT | step | `staging_wait` | `expert_dev` |
|---|---|---:|---:|---:|
| waves = 1 (shipped) | 37.5 / 38.5 / 38.9 ms | 43.9-58.4 ms | 3.0-3.5 ms | 20.5-28.2 ms |
| waves = 2 | 37.6 / 38.1 / 38.2 ms | 43.9-56.6 ms | **4.6-5.3 ms** | 22.5-33.0 ms |

TPOT is flat inside the +-8% noise, and the only clear effect is that `staging_wait` grows by ~1.6 ms: each extra
boundary costs a wait, and nothing is gained on the other side.

## 4. Why it cannot work while the link is saturated

Per step the decode moves **394 MB** at 12.8 GB/s = **31 ms of copy-engine time inside an 80 ms step**: the copy engine
is **71% busy**. A wave boundary can only hide the *tail* of a layer's DMA behind that layer's kernels, and there is no
idle copy-engine capacity to reorder into - the stream is already full of the predictive preloads for later layers,
which is why the copies were interleaved per expert in the first place.

The arithmetic is the same one M24 established: **reordering does not reduce bytes**. The numerator is 394 MB per step
and only a smaller numerator (fewer misses, or fewer bytes per miss) can move TPOT.

Shipped: `NAIWA_STAGING_WAVES` default **1**, i.e. exactly the previous behaviour. The wave path and the quant-crate
split stay in the tree as infrastructure for the numerator milestone, which is the only lever left.
