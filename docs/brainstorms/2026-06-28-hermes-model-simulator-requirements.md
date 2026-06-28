---
date: 2026-06-28
topic: hermes-model-simulator
---

# Hermes Model Simulator — Requirements

## Summary

Build a benchmark simulator that runs candidate open-weight models *inside the fixed Nous `hermes-agent` harness* and produces a trustworthy ranking of which model to found a personal-AI-assistant business on. Models are judged on capability **and** true cost — tokens-to-complete × price, plus reliability and latency — over a multi-day, memory-on simulation of a synthetic person's life. A cheap single-shot pre-filter eliminates weak models before the expensive simulation runs.

## Problem Frame

The business plan is to offer personal AI assistants (parents, business people) that run interlinked parts of someone's life: email triage, calendar, kids' activities, spouse coordination. The value lives in the seams between those domains, and — because the chosen harness self-improves — in how well the assistant builds a model of the user over time.

Founding this on a frontier subscription (Anthropic, OpenAI) is a sustainability risk: access can be restricted or de-subsidized, and the whole business would sit on a foundation it doesn't control. The intended hedge is an open-weight model the operator can self-host or rent from any provider. But "which open model is good enough, and what does it actually cost to complete real tasks?" can't be answered from published benchmarks or per-token sticker prices — a cheap model that needs three times the tokens, or fails one run in three, is not cheap and not viable for running someone's life. Answering it by hand across many models and many tasks is impractical. The simulator is the apparatus that answers it.

## Key Decisions

- **The harness is a fixed input; only the model varies.** `hermes-agent` (Nous Research, MIT) is a real model-agnostic agent harness — swap models via config, no code change. The simulator holds the harness constant and treats the model as the single variable under test. The harness is never modified or benchmarked.

- **Multi-day, memory-on simulation — not single-session.** Hermes's persistent self-improving memory is the product's core value, so the headline test runs memory-on over multiple simulated days. How well each model builds a usable model of the user over time is itself a measured outcome.

- **Two-stage funnel.** Stage 1 is a cheap single-shot, memory-off screen that eliminates models which can't handle a basic cross-domain tool chain. Stage 2 is the full multi-day memory-on simulation, run only on the 3–4 survivors. This keeps the expensive test off models that were never viable. Stage 1 also enforces hard eligibility gates the spike surfaced: a model must have ≥64K context (a Hermes minimum) and must reliably emit Hermes's tool-call format (some models, e.g. `gemma3`, run but produce no usable final response).

- **Each model runs on its own isolated memory track.** Memory stores never cross between models; one model must not inherit skills or user-knowledge another model built. Tracks are independent and reset per model.

- **Fixed exogenous event stream.** The external events of the simulated life (incoming emails, spouse texts, school notices, cancellations) are scripted and fired on an identical timeline for every model's track. Only the agent's own state (memory, calendar edits) is allowed to diverge. Every model is dealt the same hand on the same days.

- **Personas are curricula, not diaries.** A persona deliberately plants learnable regularities (recurring schedules, stated preferences revealed early) and later fires events that only an agent who remembered them handles well. Random event streams teach nothing and don't test memory.

- **Grade memory by outcome, not mechanism.** Whether a model "remembers" via its weights, learned procedural skills, or searching its own conversation history is irrelevant — only the resulting behavior and answers are scored.

- **Hybrid grading.** Deterministic checks for crisp end-states and memory-exam answers; LLM-as-judge for genuinely fuzzy behavior (email tone, prioritization, coordination quality).

## Actors

- A1. **Candidate agent** — the `hermes-agent` harness running one model under test, acting on the mock world.
- A2. **Simulated counterparties** — spouse, school, kids' organizers, played by a cheap, fixed LLM from a per-persona script so multi-day coordination has something consistent to coordinate with.
- A3. **Grader** — the deterministic checker plus LLM-judge that scores end-states, memory-exam answers, and behavioral quality.
- A4. **Operator** — the business owner running the simulation and reading the comparison report.

## Key Flows

- F1. **Stage 1 — pre-filter screen**
  - **Trigger:** Operator starts a run for a set of candidate models.
  - **Steps:** Each model faces a handful of single-shot cross-domain tasks (memory-off) against a frozen mock world; grader checks the end-state; models below a capability/cost floor are dropped.
  - **Outcome:** A shortlist of 3–4 viable models passes to Stage 2.

- F2. **Stage 2 — multi-day persona simulation**
  - **Trigger:** A model clears Stage 1.
  - **Actors:** A1, A2, A3
  - **Steps:** The model runs the persona's scripted life day by day with memory on and its own isolated memory track; the fixed exogenous event stream fires each day; the agent acts via mock MCP tools; counterparties respond from script; per-run tokens, cost, and latency are captured.
  - **Outcome:** A completed track with full behavioral logs and an accumulated memory store.

- F3. **Memory exam**
  - **Trigger:** A persona track completes its final day.
  - **Steps:** The agent is quizzed on facts only a good memory would know (recurring schedules, stated preferences, counterparty patterns), with ground-truth answers from the persona script.
  - **Outcome:** A deterministic memory score per model.

- F4. **Aggregation and report**
  - **Trigger:** All tracks for all models complete.
  - **Steps:** Grader combines capability, memory-exam, behavioral-improvement, reliability (across seeds), and cost into a per-model comparison.
  - **Outcome:** A ranked, side-by-side report the operator uses to choose a model and hosting strategy.

## Requirements

**Mock world and fixtures**
- R1. The simulator provides a sandboxed fake life — email, calendar, contacts — implemented as mock tools the harness calls, with inspectable post-run state. No real user data is used.
- R2. Each persona defines an initial world state plus a fixed, timestamped exogenous event stream applied identically across every model's track.
- R3. Personas embed learnable regularities and early preference reveals, with later events that depend on those being remembered.
- R4. Simulated counterparties respond consistently from a per-persona script, driven by a cheap fixed model held constant across all candidate models.

**Run orchestration**
- R5. The simulator runs candidate models through a two-stage funnel: a cheap single-shot screen, then the multi-day simulation for survivors only.
- R6. Each model runs on an isolated memory track; no memory, skills, or user-knowledge cross between models.
- R7. Each (model × persona) configuration runs across multiple seeds so reliability, not a single lucky run, is what's measured.

**Grading and metrics**
- R8. Grading is hybrid: deterministic checks for crisp end-states and memory-exam answers, LLM-judge for fuzzy behavior.
- R9. A final memory exam scores each model on user-knowledge with ground-truth answers.
- R10. Behavioral improvement over time is measured (e.g. stops repeating a corrected mistake, proactively applies a learned preference), distinct from the final exam.
- R11. Cost is reported as tokens-to-complete combined with price, not raw per-token rate, and is comparable across models. Latency is reported alongside.
- R12. Reliability is reported as a pass-across-repeated-runs metric (a model that succeeds 6/10 is flagged as unfit for unattended life-management).

**Model and hosting coverage**
- R13. The simulator can point the harness at both self-hosted models (local endpoint) and open-weights-via-API providers, so hosting strategy is an output of the comparison rather than a precondition.

**Reporting**
- R14. The output is a ranked, side-by-side per-model report spanning capability, memory, reliability, and cost, sufficient for the operator to choose a model and a hosting approach.

## Acceptance Examples

- AE1. **Covers R2, R6.** Given Model X and Model Y run the same persona, when Model X deletes an email on Day 1 that Model Y keeps, then both still receive the identical scripted events on Day 2 — only their own memory/calendar state differs.
- AE2. **Covers R3, R9.** Given the persona stated on Day 1 "never schedule meetings before 9am" and that fact is not repeated, when the Day 5 memory exam asks for the earliest acceptable meeting time, then a model with good memory answers 9am and is scored correct.
- AE3. **Covers R10.** Given the agent double-booked a kids' activity on Day 2 and was corrected, when a similar conflict arises on Day 6, then a model that learned surfaces the conflict instead of silently double-booking, and scores higher on behavioral improvement.

## Success Criteria

- The ranking is reproducible: re-running the same models and seeds yields the same ordering within noise.
- Results discriminate — the simulator visibly separates strong models from weak ones rather than scoring everything similarly.
- The cost figure reflects task completion (tokens × price across the whole task), so a verbose-but-cheap model and a terse-but-pricey model are compared fairly.
- The operator can act on the report directly: pick a model and a hosting strategy without further manual testing.

## Scope Boundaries

**Deferred for later**
- Breadth across many personas — start with one richly-authored persona to prove the apparatus discriminates, then add a second and third.
- Real integrations (live Gmail, real calendars) and any production assistant product.

**Outside this product's identity**
- This is an evaluation simulator, not the assistant product itself. It does not ship features to end users.
- It does not benchmark or modify the `hermes-agent` harness, and does not evaluate frontier subscription models as the foundation (they are the dependency being avoided), though one may be used as the LLM-judge.

## Dependencies / Assumptions

The three formerly-blocking feasibility assumptions are confirmed against the installed `hermes-agent` v0.16.0 — see `docs/spikes/2026-06-28-hermes-feasibility-findings.md`.

- **Headless driving — confirmed.** `hermes -z/--oneshot` runs one prompt, prints only the final response to stdout, returns an exit code; `HERMES_ACCEPT_HOOKS=1` for unattended runs.
- **Token + cost capture — confirmed.** Per-run tokens and cost (incl. `estimated_cost_usd`, cache and reasoning tokens) are written to `$HERMES_HOME/state.db`, table `sessions` — queried directly via SQLite.
- **Isolated memory per track — confirmed.** A profile is a `HERMES_HOME` directory; one home per (model × persona × seed) gives fully isolated memory, and `hermes memory reset` wipes it.
- **Per-run cost carries a large fixed input overhead** — Hermes injects its system prompt + 40+ tool schemas every turn (~11.8K input tokens even for a trivial task), uncached on local Ollama. This is a primary cost driver the simulator measures.
- **Cold-start overhead** — each one-shot re-runs plugin discovery (38 plugins). At hundreds of runs, prefer a persistent gateway process or trimmed toolsets over cold one-shots.
- Assumes the M5 Max (48GB) can host the local candidate models (e.g. `qwen3.6:latest` ~23GB loads and runs); larger models may need API hosting or a separate GPU box.
- Self-hosted "cost" has near-zero marginal token price; comparability with API models relies on a stated convention (e.g. tokens + latency + an amortized-hardware figure) to be decided in planning.

## Outstanding Questions

All blocking feasibility questions are resolved (see `docs/spikes/2026-06-28-hermes-feasibility-findings.md`), including the mock world: a custom stdio MCP server registered via `hermes mcp add` was called by the agent in a headless run, with a sentinel value proving the tool supplied the data.

**Deferred to planning**
- Which model plays the LLM-judge and the simulated counterparties, and how to keep judge bias and cost in check.
- Exact weighting of capability vs memory vs reliability vs cost in the final ranking.
- Scenario/persona count and number of seeds per configuration.
- The convention for making self-hosted and API costs directly comparable.

## Sources / Research

- `hermes-agent` (Nous Research, MIT): model-agnostic agent harness; `hermes model` swapping; 40+ tools and custom MCP server support (the mechanism for the mock world); persistent self-improving memory; cron; gateway and serverless run modes. github.com/nousresearch/hermes-agent
- τ-bench (tau-bench, Sierra) — prior art for tool-agent evaluation with a simulated user, sandboxed world, deterministic post-run state checks, and a pass^k reliability metric. Closest existing pattern to adapt. To verify for currency/license in planning.
- AgentBench — broader multi-environment agent benchmark; alternative reference pattern.
