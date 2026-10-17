# Where the 6x vs Strata comes from

Measured on 2026-10-17. Two engines, **the same model family** (Qwen3.8-Flash-Next), two cards.

## The numbers

| | Strata | naiwa-rs v2 |
|---|---|---|
| model | Qwen3.8-Flash-Next IQ3_XXS / IQ2_XS | Qwen3.8-Flash-Next GSQ-RCO **IQ1_M** |
| card | RTX 4070 Ti SUPER, 16 GB, sm_89 | RTX 4070, **12 GB**, sm_89 |
| decode | **76.0** tok/s (IQ3_XXS, 32k) / **97.9** (IQ2_XS) | **11.6-13.3** tok/s |
| per pass | ~33 ms (76 tok/s at ~2.5 tok/pass) | **78-86 ms, 1 token/pass** |
| speculation | `--spec 4 --mtp`, ~62-68% accepted | **none** - the artifact has no MTP head |

Strata's figures are from `~/code/strata/bench/results/2026-10-05-community-rtx-4070-ti-super/README.md`
(engine-reported `predicted_per_second`, temperature 0, 192 tokens, three SSE calls). Ours:
`bench --prompt-len 512 --gen-len 32 --depth 4 --pcie-frac 0 --expert-cache-gb 2` with
`NAIWA_NO_LOOKAHEAD=1`.

## The gap decomposes into two factors, not one

```
11.6 tok/s   x2.5 (speculation: 1 -> ~2.5 tokens/pass)   = 29 tok/s
             x2.6 (pass latency: 86 ms -> 33 ms)         = 76 tok/s   <- strata
```

**1. Tokens per pass: ~2.5x. The artifact problem is solved; a crash is not.**

The MTP head exists locally - it was downloaded and repacked separately:

```
~/bench-rel/mtp/mtp-component.gguf      1.47 GB, 32 tensors: blk.48.* (the head block, nextn.*)
~/bench-rel/mtp/mtp-Q4_0.gguf           2.20 GB, the published source
~/bench-rel/mtp/repack.py                rewrites names/metadata, drops token_embd/output,
                                         keeps blk.48.* (the fork binds those) - no tensor bytes touched
```

and it loads through the mmproj channel:

```bash
--mmproj ~/bench-rel/mtp/mtp-component.gguf --mtp --mtp-n-predict 2
```

Measured today:

| configuration | result |
|---|---|
| the two shards alone, `--mtp` | `Error: checkpoint has no built-in MTP head (mtp.fc_embedding.weight)` |
| **shards + `--mmproj` component, no `--mtp`** | **fine: 12.5 tok/s, 79.8 ms TPOT** (the component merges cleanly) |
| shards + `--mmproj` + `--mtp` | **`CUDA_ERROR_ILLEGAL_ADDRESS`** in the first draft/verify step |

So the blocker is no longer the artifact but the draft path itself.

**Found and fixed while chasing it: the schedule was sized for the decoder only.** `ScheduleBuf` is
`8 + k*4 + slots*max_n*4` with `k = slots * WAITS_PER_LAYER` and `slots = layers.len()`, and
`sync_id(layer, kind) = layer * WAITS_PER_LAYER + kind`. The MTP head is a site of its own with
`layer == layers.len()` (48), so its `SYNC_CPU_EXPERTS` id (144..146) and its payload slot (48) both sat
past the end of the mapping - a guaranteed illegal access the first time the head published. The schedule
now reserves one slot past the highest decoder layer (`slots = max_layer + 2`). That is a real bug fixed
and it leaves the decoder byte-identical (oracle re-checked).

It is not the only one: `--mtp` still faults with `CUDA_ERROR_ILLEGAL_ADDRESS`. Remaining suspects, in
order:

1. **the component's `nextn.eh_proj` slicing** - `repack.py` passes `blk.48.nextn.eh_proj.weight` through
   whole and the fork binds `fc_embedding`/`fc_hidden` as two halves of it; a half-offset mismatch reads
   past the tensor;
2. **the MTP block's own attention/state** - `MtpAttentionInputs` takes its own KV/state path, and the
   hybrid cache may not carry a slot for layer 48;
3. **the legacy `CpuExperts` channel under MTP row counts** (the head's experts are host-resident by
   design, `expert_cache.rs` `ctx.site.is_mtp` branch).

Ruled out in this round:

- **the `nextn.eh_proj` slice** - the component stores it as **Q4_0, dims (5120, 2560)**, so the loader's
  `shape()` reads `[2560, 5120]`, `half = 2560`, and 2560 rows of 2880 bytes is a whole number of 32-element
  Q4_0 blocks: the slice offset is block-aligned, not a fractional-quant offset;
- **the prefill-stream layer table** - `PrefillStream::register` resizes to `layer + 1`, and the MTP site
  returns before it anyway (`ctx.site.is_mtp`).

**Localised by experiment**: it is not steady-state drafting but the drafter's **prompt-chunk catch-up**.

| config | result |
|---|---|
| prompt 8, `--mtp --mtp-n-predict 1` | **works**: 2-row draft pass, 106.2 ms TPOT, 9.4 tok/s |
| prompt 8, gen 1 (prefill only) | works (no draft step runs) |
| prompt 512, `--mtp` (any depth) | `CUDA_ERROR_ILLEGAL_ADDRESS`, and **zero `MTP-DBG` lines** - the fault happens *before* `drafter_forward`, i.e. in `drafter_prefill_chunk` |

So `drafter_forward`'s sizing is fine at the lengths it reaches (`rows=2`, `kv_blocks=2520`,
`blocks_needed=1`, `block_table_width=1`), and the crash is in the catch-up that runs a whole prompt chunk
through the drafter with the *target's* attention inputs. Also measured while it worked: the 2-row pass
costs 106 ms against 82 ms for a 1-row pass, i.e. **+29% per pass for one draft token**, so MTP only pays
off if acceptance is well above one token in three - exactly what the host-bound pass cost makes harder
than on Strata.

From reading the engine's draft path
(`speculative/builtin_mtp.rs:120-150`): the drafter builds its own paged metadata
(`make_paged_rows_metadata(seq_ids, context_lens = position+1, ..)`) and passes `kv_cache.clone()` plus
`AttentionMask::None` / `FlashParams::empty(false)`. Two concrete suspects, in order:

1. **the drafter's KV cache against `context_lens`** - the block table is indexed by the *target's*
   positions (hundreds of blocks during a 512-token prompt), while the drafter's own cache is sized for
   draft rows; a write past its last block is exactly an illegal address;
2. **our `CpuExperts` path under draft row counts** - the MTP site's experts are host-resident by design
   and reach the GPU through the deferred queue at `layer = 48`, so its rows/top_k and the queue's device
   buffer have to agree with the draft batch rather than the decoder batch.

Everything else around the head is in the head's own attention/state path: `MtpAttentionInputs` and the
KV/state slot the head uses (the engine's own errors mention "release speculative sequence state" and
"recurrent state for sequence 2, slot 0" right after the fault). `compute-sanitizer` is unusable here (it
turns the 60 GB model load into an hours-long run; killed at 900 s), so the next diagnostic is targeted
logging in `mtp.rs::forward` - print the shapes/devices it receives and the KV slot it resolves, then the
kernel that faults is identified by the last log line before the error.

With the draft path working, the same 86 ms pass carries ~2.5 tokens; minus the extra cost of the verify
rows (2-3 rows instead of 1 - our per-layer *host* work barely moves, the GPU segments grow) that is
roughly **20-29 tok/s** without touching anything else, and one more rung of draft depth gets near 40. That
is also why "mtp2 -> 40 tok/s" is remembered: speculation is worth more than every kernel optimisation on
this list put together.

**2. Pass latency: 86 ms against ~33 ms, ~2.6x.** And we know exactly where ours goes, because it was
measured: `NAIWA_STEP_DBG` dry time is **1.53-1.63 ms per layer x 48 = 74-78 ms**, and the GPU is idle 98%
of that. The token thread is the bottleneck: per layer it reads the top-k back through the doorbell,
computes the miss set, issues the staging DMAs and writes the CPU payload. Strata's design instead keeps
the host *out* of the critical path - `BATCHED_DMA`, `KV_PREFETCH`, `DISJOINT_EXPERT_CACHE`, 23 expert-pool
workers - and the GPU runs the pass as one graph.

Card caveat, stated honestly: strata had 16 GB to our 12 GB, so it holds more experts resident; ours is
2 GB of expert cache (~21 slots/layer) with an **82-84% hit rate**, so ~1.5-3 experts per layer go to the
CPU, and that CPU leg is what the 1.6 ms/layer is made of. Its card also fits IQ3_XXS where we run IQ1_M
(fewer bytes per weight, which should favour *us* on the bandwidth-bound part).

## What this means for the roadmap

| lever | expected | cost | verdict |
|---|---|---|---|
| **MTP/spec decoding** | x2.5 tokens/pass (~20-29 tok/s) | head is **local and loads** (`~/bench-rel/mtp/mtp-component.gguf`, mmproj channel); fix the `ILLEGAL_ADDRESS` in the draft path - the MTP site's experts are outside the engine's schedule | **by far the biggest**; also the only one that gets us near 40 |
| **host-loop pipelining** (batch the staging DMAs, prefetch the top-k a layer ahead, cut the doorbell readback) | 86 -> ~50-60 ms/pass (x1.4-1.7) | medium: the doorbell protocol already supports "issue, don't perform" | second |
| **bigger expert cache / better hit rate** | fewer CPU experts -> less host work | VRAM-limited at 12 GB; IQ1_M leaves headroom | third |
| **M6 2c (attention capture)** | 1-4 ms/pass (5%) | ~150 lines, residents half-done | last |

So M6 is *not* where the 6x lives. It is worth doing for the mechanism (and it deletes work), but the
honest ranking is: get the MTP head, then take the host out of the inner loop.
