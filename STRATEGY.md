---
name: Hermes Model Simulator
last_updated: 2026-06-30
---

# Hermes Model Simulator Strategy

## Target problem

An operator building a personal-assistant business needs to run it on an open-weight
model — to escape the cost, withdrawal risk, and eventual de-subsidization of closed
models like Claude or ChatGPT. But which open model is *good enough* (tool-calling,
reliability, low hallucination, memory) and what it *truly* costs can't be read off
leaderboards or per-token prices, and testing candidates by hand — one agent per
model, many models — is impractically slow.

## Our approach

Measure each model the way it will actually be used: inside one fixed agent harness
(Hermes), varying only the model, on a multi-day memory-on simulation of one person's
life — graded against ground truth the agent can't reach and on true
tokens-to-complete cost. So the ranking reflects the model on the real job, not a
leaderboard proxy. Select one best model by default; pursue a multi-model combination
only if Hermes can integrate it practically and it measurably beats the single best.

## Who it's for

**Primary:** The founder (you) — picking the open model to build this
personal-assistant business on. Hiring the simulator to choose that model with
trustworthy evidence on capability and true cost, without weeks of manual testing.

## Key metrics

- **Artifact rate** — fraction of runs that are clean (tools loaded, completed) vs
  corrupted by infrastructure (e.g. MCP cold-start starvation); a high rate means the
  rankings are fiction. From run logs/trajectories.
- **Discrimination** — whether a run's ranking actually separates models (real spread,
  not all-pass or all-eliminated). Per run.
- **Ranking reproducibility** — whether re-running would flip the winner; tracked via
  seed-to-seed score and token variance. Across seeds.
- **Time-and-cost per candidate** — wall-clock and dollars to push one model through
  the full funnel. Per run.
- **Predictive validity** *(post-launch, lagging)* — whether the model the simulator
  picks actually performs in the real deployed assistant. Not measurable until ship.

## Tracks

### Harness & world fidelity
The fixed Hermes wrapper, the mock MCP world, the run orchestrator, and their
robustness. The MCP cold-start race that corrupted early runs is **resolved**: the
mock-world servers now run as a persistent per-track HTTP gateway, so tool
discovery is deterministic for both local and API fields (model warmup and the
tool-starvation retry are kept as safety nets).

_Why it serves the approach:_ keeps measurement fair (one harness), realistic, and
reliable — without it, the rankings are noise.

### Scenario realism
The personas (multi-day curricula), Stage-1 cross-domain tasks, and the simulated
counterparty.

_Why it serves the approach:_ the closer the simulated life is to the real assistant
use-case, the more the ranking predicts real performance.

### Grading & cost integrity
Deterministic state-diff, the cross-family LLM judge, the forgetting-aware memory
exam, behavioral checks, and the metrics/report rollup.

_Why it serves the approach:_ scores stay trustworthy and ungameable (read
out-of-band), and the cost and reliability math is honest.

### Candidate & hosting coverage
The field of candidate models (local and API) running validly at the 64K context
floor, with real cost-per-hosting profiles. The **API field over OpenRouter is
operational** (GLM-5.2, Llama-3.3, Qwen2.5-72B, Mistral-Large), with a
tool-support guard that drops models whose providers can't do agentic tool use.
Current leader: **GLM-5.2** (the first model to adopt a mid-run knowledge update);
see `docs/benchmark-findings-2026-06-30.md`.

_Why it serves the approach:_ makes "true cost" real and spans the deployment options
actually worth choosing between.

## Not working on

- Generalizing the simulator for other operators — it's a personal decision tool for now.
- A multi-model routing/combination solution, unless Hermes can integrate it practically and it beats the single best model.
- Building the assistant product itself in this repo — this is the evaluation tool; the product is separate.
- Evaluating frontier/closed models as the business foundation — they serve only as the eval judge, never as what gets shipped.
