# Family persona + cost-forward, latency-aware model eval — requirements

**Date:** 2026-07-01
**Status:** Ready for planning
**Scope:** Deep — feature (extends existing personas, model field, and composite scoring)
**Sources:** `~/Downloads/hermes_interactions_clean.md` (136 real sessions),
`~/Downloads/openrouter_activity_2026-07-01.csv` (2,433 real calls, $30.21),
`/tmp/hermes-simulator-handoff-2026-07-02-model-selection.md`

## Problem

The user stopped using Hermes because their model (**Qwen 3.7 Plus**, `qwen/qwen3.7-plus`)
"could be better" and cost too much. But **Qwen 3.7 Plus has never been benchmarked here** —
the field is GLM-5.2, Mistral-Large, Llama-3.3, Qwen2.5-72B (a different, older Qwen). So the
current "Mistral-Large leads" result does not answer the user's question. Two gaps:

1. **No representative persona.** The synthetic `dana` persona doesn't reflect the user's real
   workload — a multilingual (EN/NL/PT) Amsterdam family assistant doing email triage, daily
   briefings, and family-calendar edits, ~80% of it via cron automation.
2. **The scoring under-weights what the user actually cares about now.** Cost is weighted 0.10
   (memory-heavy weights from the prior question), latency isn't scored at all, and the cost
   model doesn't reflect the user's real input-token-dominated bill.

### Evidence (from real usage, not inferred)

| Model | Calls | Real $ | Input tok/call | Output tok/call | Avg gen time |
|---|---|---|---|---|---|
| **qwen3.7-plus** (user's main) | 675 | **$15.81** | 77,311 | 626 | **15.9s** |
| glm-5.2 | 512 | $7.46 | 25,024 | 377 | 13.0s |
| gemini-3.5-flash | 219 | $4.49 | 35,353 | 277 | 2.5s |
| llama-3.3-70b | 289 | $0.59 | 14,975 | 47 | 2.1s |
| mistral-large | 148 | $0.64 | 15,156 | 119 | 3.4s |
| qwen-2.5-72b | 122 | $0.75 | 16,932 | 128 | 7.6s |

Three findings that reframe the eval design:

- **Cost is input-token-dominated.** ~77K input vs ~626 output per call for Qwen 3.7 Plus →
  a model's *input* price matters far more than its output price for this workload. (Inference:
  the input is memory context + re-sent email-triage payloads; 35M of Qwen's tokens were cached,
  already softening the bill.)
- **Qwen 3.7 Plus is the slowest model** (15.9s/call vs Gemini's 2.5s). The user's "could be
  better" is plausibly partly *latency*, which the current composite ignores.
- **Live prices drift materially.** Gemini 3.5 Flash was cheap in the logged usage but is now
  **$1.50 in / $9.00 out** on OpenRouter (verified live 2026-07-01). Any recommendation must be
  dated and re-verified at run time.

## Goals

- Add a **PII-scrubbed Amsterdam-family persona** that mirrors the user's real, multilingual,
  automation-heavy workload — so capability, memory, latency, and cost are measured on
  representative tasks.
- Benchmark **Qwen 3.7 Plus** and an expanded field, and produce a **cost-forward,
  latency-aware** comparison.
- Deliver a **per-dimension evidence table** (Capability / Memory / Reliability / Speed /
  Cost-per-task) with **three labelled picks** — best accuracy, best value, cheapest-viable —
  and **no single composite winner**.

## Non-goals

- No single "the winner is X" composite ranking. (User chose "show all, no single winner.")
- No re-plan of the gateway / API-track infrastructure — done and merged.
- No replacement of `dana` — the new persona is added alongside it.
- No commit of real PII — all identities become `*.test` stand-ins.

## Requirements

### R1 — New persona: `amsterdam` (working name)

A multi-day, memory-on persona modelled on the real interactions, following the `dana.py`
schema (`Persona`, `DayPlan`, `ExogenousEvent`, `answer_key` with memory probes + behavioral
signals). It must exercise:

- **Family-calendar coordination** — kids' recurring activities (violin, kickboxing, swim,
  dance), dentist appointments, schedule conflicts to detect.
- **Email triage** — an inbox with appointment-bearing emails (some in Dutch) the agent must
  read, translate, summarize, and turn into calendar/to-do actions.
- **Daily briefing** — a morning summary of the day's events (the real cron pattern), in the
  user's stated default language.
- **Memory mechanics** (keep dana's discriminating structure): at least one *recall* fact, one
  mid-run *knowledge-update* (an activity that changes day/time), and one *abstention trap* (a
  plausible-but-never-scheduled appointment the agent must not invent).
- **Multilingual** (R2) and **heavy-context cost fidelity** (R3) as below.

**PII:** replace real names, addresses, and identifiers (Amsterdam family, dentist, schools)
with `*.test` stand-ins. Keep the *shape* of the real tasks, not the real data.

### R2 — Multilingual dimension

The persona must include Dutch- and Portuguese-language content and language-preference
behavior, testing:
- Translating a Dutch email and surfacing the action items in the user's default language.
- Honoring a stated default-language preference across days (a recall/behavioral signal).
- Understanding a Portuguese-language request. (English is the default reply language per the
  real "switch briefings to English" instruction.)

The answer key must score whether the model handled the non-English content correctly, so
weaker-multilingual models are discriminated.

### R3 — Heavy-context cost fidelity

The persona's per-day context (accumulated memory + inbox payloads) must be large enough that
cost-per-task reflects the user's real **input-dominated** bill, not a lean synthetic one. This
rewards cheap-*input* models and is pure persona design — no harness change. The report already
records cache-read/write tokens, so prompt-caching effects are captured where the model/provider
supports them.

### R4 — Latency as a scored dimension

Latency is already computed (`metrics.latency_seconds`) and surfaced in reports, but is **not**
in the composite. Add **Speed** as a first-class reported dimension and make it available to the
weighted composite (a new weight, or folded into reliability — a planning decision). The
2.5s→16s real-world spread must be visible and count against slow models.

### R5 — Expanded model field, run via the two-stage funnel

Run every candidate through **Stage 1** (cheap pre-filter: context-floor + tool-format gates +
single-shot tasks), and promote only survivors to the expensive **Stage 2** multi-day persona
sim. Field:

| Model | OpenRouter id | Live price in/out ($/1M) | Why |
|---|---|---|---|
| Qwen 3.7 Plus | `qwen/qwen3.7-plus` | 0.32 / 1.28 | **Mandatory** — user's model, never benchmarked |
| Qwen-Plus | `qwen/qwen-plus` | 0.26 / 0.78 | Cheaper Qwen sibling (user pick) |
| GPT-5.4-mini | `openai/gpt-5.4-mini` | 0.75 / 4.50 | Cross-vendor baseline (user pick) |
| Gemini 3.5 Flash | `google/gemini-3.5-flash` | **1.50 / 9.00** | Fast, proven on real workload (user pick) — ⚠️ now costly |
| DeepSeek V3.2 | `deepseek/deepseek-v3.2` | 0.23 / 0.34 | Cheap-output DeepSeek flagship (grounded add) |
| MiniMax M2.5 | `minimax/minimax-m2.5` | 0.12 / 0.48 | Cheap agentic model, 204K ctx (grounded add) |
| Qwen 3.5 Flash | `qwen/qwen3.5-flash` | **0.065 / 0.26** | Cheapest-input serious Qwen; likely cost-per-task winner (grounded add) |
| GLM-5.2 | `z-ai/glm-5.2` | 0.93 / 3.00 | Continuity — prior field |
| Mistral-Large | `mistralai/mistral-large-2512` | 0.50 / 1.50 | Continuity — current leader |
| Llama-3.3-70B | `meta-llama/llama-3.3-70b-instruct` | 0.10 / 0.32 | Continuity — cheap sticker, token-verbose |
| Qwen2.5-72B | `qwen/qwen-2.5-72b-instruct` | 0.36 / 0.40 | Continuity — prior field |

- **Owl Alpha:** requested but **not available on OpenRouter as of 2026-07-01** (pulled again,
  as the handoff warned) — excluded; note it in the report so the exclusion isn't mistaken for
  a failure.
- All listed ids are **tool-capable** (verified against the live `/models` list). Re-verify
  prices at run time (they drift — see Gemini).

### R6 — Output: evidence table + three labelled picks

- A per-dimension table: **Capability / Memory / Reliability / Speed / Cost-per-completed-task**
  (tokens-to-complete × live price, from metered `actual_cost_usd`), one row per model.
- Three labelled recommendations derived from the table:
  - **Best accuracy** — highest capability+memory at acceptable reliability.
  - **Best value** — best accuracy-per-dollar above a stated quality floor.
  - **Cheapest-viable** — lowest cost-per-task that clears the floor.
- Show the ranking under **≥2 weightings** (the memory-heavy default and a cost-forward one) to
  make the trade-off explicit. `CompositeWeights` is the single knob; columns are always
  reported regardless.
- **Reconcile with real costs:** compare the sim's per-task cost *shape* against the user's real
  aggregate bill (input-dominated), not absolutes — the persona drives token volume.

### R7 — Rigor and provenance

- **≥5 seeds** for any decision (the 3-seed run wrongly crowned GLM-5.2; reproducibility caught
  it). Reliability is `pass^k`, k=5, so `len(seeds) ≥ k`.
- Per-model chunking into a shared run-id under the ~40-min session cap, then
  `scripts/build_report.py <run_id> results` to stitch a judge-corrected table.
- Record the outcome in `docs/benchmark-findings-2026-07-01.md` and update `STRATEGY.md`'s
  leader line. Recommendation must be **dated** and note prices were live-verified.

## Success criteria

- The user can read one table and answer "which model should I run, and what does it cost me,"
  with Qwen 3.7 Plus placed against cheaper and faster alternatives on *their* workload.
- The persona is representative enough that its cost-per-task shape matches the real bill's
  input-dominated shape.
- Slow models are visibly penalized; the latency gap the user felt is quantified.
- No real PII in committed files.

## Open questions (for planning)

- **Speed scoring shape:** own composite dimension vs folded into reliability, and how to
  normalize seconds→0-1 (linear cap? log?). Also whether to score p50 or mean latency.
- **Persona size vs run cost:** how large to make the heavy-context load before Stage-2 run cost
  becomes painful across ~7-11 models × 5 seeds × multi-day. (Stage-1 funnel mitigates.)
- **Quality floor for "viable":** what capability+memory+reliability threshold gates the
  best-value / cheapest-viable picks.
- **DeepSeek/MiniMax variant choice:** if run cost must shrink, `deepseek/deepseek-v4-flash`
  ($0.098/$0.196, 1M ctx) is a cheaper-input alternative to v3.2; confirm which DeepSeek to run.
- **Multilingual grading:** keyword/substring checks (like dana) vs judge-scored translation
  quality.

## Dependencies / assumptions

- `OPENROUTER_API_KEY` in gitignored `.env`; `claude` CLI logged in for the judge; `hermes` on
  PATH. Full prior 4×5 run cost ~$0.40; the expanded field is a larger but still-modest run —
  budget history is ~$5/session.
- **Assumption:** the real input-token load is memory context + re-sent triage payloads
  (inferred from the data, not stated). If wrong, the cost-fidelity target (R3) may over- or
  under-shoot the real shape.
- **Assumption:** latency captured from `state.db` session timestamps is a fair proxy for the
  user-felt latency (it's wall-clock per session, which includes tool round-trips — arguably
  *better* than raw model gen-time for a UX judgment).
