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

So the blocker is no longer the artifact but the draft path itself, and there is already a pointer to
where: the MTP site's experts are not in the engine's schedule - they go through the legacy global
`CpuExperts` channel (`prefill_stream.rs:373` has an explicit `ctx.site.is_mtp` branch). An illegal
address on the first MTP step is what an off-schedule expert site looks like when its buffers were never
staged.

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
