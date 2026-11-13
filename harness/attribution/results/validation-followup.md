# GPU sampling validation follow-up

Status: second no-lookahead identity/logprobs parity positive; chat TTFT and fallback performance gates NOT passed. Parent single-gpu-sampling.md. No engine/default changes this turn.

Harness now supports explicit chat_followups, sequentially appending each arm's actual full assistant text and configured user followup; saves exact per-request bodies. This is end-to-end conversational policy benchmark, not same history when text diverges. Request3role chain manually verified. common.capture retains streamed logprobs payloads (previously discarded); does not infer usage from chunks. Python15tests pass after mock config compatibility fix (optional messages on nonchat mock). No fresh Rust tests needed/claimed, engine binary unchanged.

All new runs3AB/BA/AB restarted pairs x3requests x128completiontokens; no failures/retries; CPU-first/budget1/Rayon/frac .4; GPU sampling only arm difference. Clean no probes.

## Second prompt, no-lookahead

Config verify-gpu-second-nolook.json, raw /tmp/attr-verify-second-nolook-01. TPOT improves2.702/2.088/1.957ms, median2.088. ALL full text hashes equal across both arms/all restarts/all request positions. SSEp99 delta -2.283/+.300/-3.710ms (per-process median of requestp99s, not decode-step tails). TTFT delta+66.6/-55.2/-144.4ms. Text equality necessary not sufficient for expert-work equality. Strengthens second-prompt parity.

## Three-turn conversation, lookahead on

Config verify-gpu-chat.json, raw /tmp/attr-verify-chat-01. Preserves own assistant replies, request tokens fixed128 each. TPOT improves2.439/1.042/1.421ms, median1.421. SSEp99 decreases3.217/2.174/1.284ms. BUT TTFT median regresses22.6/183.3/345.7ms. Histories differ between arms, so no causal same-input TTFT slowdown claim, but full chat adoption NOT passed. Only one3turn conversation, not corpus-level chat gate. Future fixed transcript crossover needed to isolate TTFT from content/prefix effects.

## Logprobs fallback

Config verify-gpu-logprobs.json adds logprobs=true/top_logprobs1 with no-lookahead. First /tmp/attr-verify-logprobs-01 text equal by position but collector discarded logprobs, inadequate parity evidence. Extended capture then new /tmp/attr-verify-logprobs-02. ALL9pairs logprobs payload nonempty and byte-equivalent JSON values; full text matches per position. TPOT paired median candidate regression+.447ms (initial run+.164); fallback performance non-regression NOT established. Guard expected to leave path unchanged, but no active-branch trace in clean; cannot dismiss difference as noise or claim optimization active. Need independent fallback diagnostic reasons plus repetition/cornercase testing. No change defaults.

Evidence validation-followup.json contains reports, TTFT/SSEp99 and parity checks; validation-{second-nolook,chat,logprobs}-manifest.json archives provenance. Oracle/ties/NaNs/tool/penalty cases and MTP/chat gates remain. Recommendation remains opt-in for no-drafting supported greedy use only, not default promotion.
