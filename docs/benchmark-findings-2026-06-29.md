# Hermes model-simulator — first benchmark findings (2026-06-29)

Run managed autonomously overnight. The headline numbers come from run `balanced-4`
(qwen3.6 full Stage-2, 2 seeds) plus a focused `qwen3:32b` run; `gemma4` and
`hermes3` eliminations reproduced cleanly after the harness fixes below.

> **Status of each claim:** results are stated as facts with the run they came
> from. Where I extrapolate ("this suggests…"), it's marked.

## Ranked result

```
 #  Model                 Cap   Mem   Rel    Cost$    Tokens   Composite
 1  Qwen3.6 (local)       0.50  0.67  1.00   0.0996   497,820  0.717

Eliminated:
  - Gemma4 (local):       pre-filter pass rate 10% (tools loaded; genuinely weak)
  - Hermes3 8B (64K):     tool-call format — does not reliably emit tool calls
  - Qwen3 8B:             context window 40,960 below the 64K floor
  - Qwen3 32B:            capable (uses tools, passes Stage-1 tasks) BUT Ollama caps
                          it at 40,960 — not a true 64K model; also 20GB/slow to self-host
```

`qwen3.6` is the only candidate to reach Stage 2; the rest are ruled out before it.
`qwen3:32b`'s full Stage 2 was not run — at ~60s/call on 48GB it could not complete
within the run window — but its disqualifier (sub-64K context) is independent of that.

Cap = behavioral adherence (0–1). Mem = memory exam (recall/update/abstention).
Rel = pass^k over seeds. Cost = imputed local $ (configurable). Tokens =
mean tokens-to-complete the 6-day persona. Composite = weighted (0.40/0.25/0.20/0.15).

## What each model showed

**Qwen3.6 — the only genuinely viable local model, but with one clear weakness.**
- ✅ **Recall**: remembered the standing rule ("no appointments before 9am") — both seeds.
- ✅ **Abstention**: declined to invent Theo's never-scheduled dentist appointment — both seeds.
- ❌ **Knowledge-update**: when the coach's day-2 email moved soccer Thursday→Wednesday,
  qwen3.6 kept asserting **Thursday**. Two independent graders agree:
  - memory exam scored `knowledge_update` *stale* (both seeds);
  - the behavioral scan caught it still saying "Thursday" on day 3 (seed 0) and days 3/5/6 (seed 1).
- ✅ **Preference adherence**: never scheduled a meeting before 9am, even when the school
  proposed 8am (both seeds).
- ⚠️ **Cost instability**: 254K vs **741K** tokens for the same 6 days across two seeds — ~3×
  variance, driven by its `temperature 1.0` default. Token spend (hence $) is unpredictable run-to-run.
- Note on `Rel=1.00`: both seeds *completed* and cleared the 0.5 capability bar, so pass^k=1.
  This measures completion-reliability, not quality consistency — the token variance shows real
  run-to-run instability a 0.5 threshold doesn't capture.

*Inference:* the failure mode that matters for a personal assistant is the
knowledge-update miss — an assistant that keeps acting on a stale schedule after
you've been told it changed is a real liability for running a household.

**Gemma4 — not viable (genuine).** With tools loaded and given two attempts per
task, it passed only 1/10 pre-filter tasks. Combined with `gemma3` (format-incompatible),
the **Gemma family is out** on the binding constraint (reliable tool use).

**Hermes3 8B — not viable at this size.** Lineage-matched to the harness and it
*can* emit the tool-call format, but inconsistently: across repeats it often had the
tools loaded and simply answered without calling them, and it failed the Stage-1 format
gate. *Inference:* 8B is too small for reliable agentic tool use; the 70B Hermes would
need API hosting (won't fit 64K on 48GB) and is worth testing there.

**Qwen3 8B / Qwen3 32B — context-floor problem.** Both are natively ≤40,960 tokens.
`qwen3:32b` can be coaxed to run (a `num_ctx` Ollama variant stops Hermes refusing it),
but Ollama caps it at 40,960 — so it is **not a true 64K model**, the same disqualifier
as qwen3:8b. Among local Qwen models, **only qwen3.6 genuinely meets the 64K floor.**

## The engineering story (why the first runs were misleading)

The first full run eliminated *everything* and finished suspiciously fast. Diagnosing
it — not the scores — was the real value of the night. Five harness issues were found
from run data and fixed (all committed/pushed, +regression tests; 144 tests green):

1. **MCP cold-start race (dominant error source).** When Ollama loads a model, that load
   races the three mock-world MCP servers' startup; Hermes sometimes ran the agent with
   **no world tools**, and every tool task failed as an artifact (signature: ~8–12K input,
   0 tool calls). This had corrupted most of run 1 — including 3 of qwen3.6's "failures"
   and gemma4's first "format" verdict. Fixes: **warm the model** before registering tools,
   and **retry tool-starved runs** (only when input is low — schemas absent — not when the
   model merely chose not to call, which is a real result).
2. **Context-refusal misclassification.** Hermes refuses under-context models on stdout with
   exit 0 in a phrasing the detector missed → context failures mislabeled as format failures.
3. **Ollama context defaults.** Models load at 32–40K by default, below the 64K floor;
   addressed with per-model `num_ctx` variants (no server restart).
4. **Single-shot Stage-1 gate** was noise-sensitive for temperature-1.0 models → best-of-N.
5. **Interrupted-run resilience.** Per-track memory-exam answers are now persisted and
   `scripts/build_report.py` rebuilds the ranked report from disk (the overnight runs kept
   getting killed at ~40 min — apparently a background-process duration cap — so this
   mattered: qwen3.6's full result was recovered from disk + a post-hoc exam).

## Answer to "should we download more open models?" — now evidence-backed

- **Don't add more Gemma.** The family fails the binding constraint (tool use).
- **8B is too small** for reliable agentic tool use (Hermes3 8B); the viable band starts higher.
- **The local field is genuinely thin:** only `qwen3.6` meets 64K *and* uses tools reliably,
  and even it fails knowledge-update. One usable local model is not a basis for a business decision.
- **Therefore:** the highest-value next models are (a) tool-tuned models in the **24B–70B**
  range from **different families** (Hermes 70B, Mistral-Small/Large, Qwen 72B, Command-R) and
  (b) **API hosting** for anything that needs reliable 64K context — where context is a config
  parameter, not an Ollama memory fight. This run strengthens, rather than changes, that plan.

## Recommended next steps

1. **Test the 64K, tool-tuned tier via API** (Together/Fireworks/OpenRouter): Hermes-70B,
   Qwen2.5/3-72B, Mistral-Large, Command-R. The simulator already supports API candidates (R13);
   this is the path where context and reliability are controllable.
2. **Re-weight knowledge-update.** Every local model that got far enough failed it; it's the
   discriminating dimension for a life-running assistant. Consider a persona with more mid-run
   changes and a heavier memory weight in the composite.
3. **Add the LLM judge** (cross-family, e.g. a frontier API model) to score tone/proactivity —
   off in these runs (no key wired). It rounds out the capability picture.
4. **Persistent-MCP-gateway** (deferred in the plan) would eliminate the cold-start race at the
   source and speed runs up; the warm+retry mitigation works but costs reruns.
