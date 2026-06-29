# hermes-simulator

A benchmark harness that runs candidate **open-weight models** inside the fixed
Nous `hermes-agent` harness and ranks them for a personal-assistant business on
**capability, memory quality, reliability, and true cost**.

Published benchmarks and per-token sticker prices don't answer the real question:
*which open model is good enough to run someone's life, and what does it actually
cost to complete real tasks?* A cheap model that triples token use, fails one run
in three, or forgets the user is not viable. This simulator measures that
directly: it holds the agent harness constant and varies only the model.

## How it works

One harness, many models. Each `(model × persona × seed)` runs in its own
isolated `HERMES_HOME` so memory tracks never cross. A two-stage funnel keeps the
expensive part cheap:

- **Stage 1 — pre-filter.** Hard eligibility gates (≥64K context, tool-call-format
  compatibility) plus ~10 single-shot cross-domain tasks. Non-viable models are
  dropped with a recorded reason.
- **Stage 2 — multi-day simulation.** Survivors run a memory-on, multi-day
  persona: a fixed exogenous event stream (emails, calendar, kids' activities)
  replays identically across tracks while the agent coordinates with a simulated
  spouse/coach. Memory must survive across days.

The mock world (email / calendar / contacts) runs as **MCP servers** the agent
calls; the grader reads the same backing store **out-of-band** — never through an
agent-reachable tool — which closes the reward-hack / answer-leak hole.

Grading is hybrid: deterministic state-diff for crisp outcomes, a cross-family
frontier **LLM judge** for fuzzy behavior (tone, proactivity, surfacing
remembered context), a forgetting-aware **memory exam** (recall / knowledge-update
/ abstention), and **behavioral-improvement** checks over days. Reliability is
`pass^k`; cost is always tokens-to-complete plus a comparable dollar figure.

## Layout

```
simulator/
  config.py          candidate models, hosting profiles, run params
  harness.py         the only module that drives the hermes CLI
  runner.py          two-stage funnel orchestrator
  pipeline.py        run_full: funnel -> exam -> evaluate -> rollup -> report
  world/             mock MCP servers + out-of-band SQLite store
  scenarios/         stage-1 tasks, persona schema + first persona (dana)
  counterparty.py    fixed-model spouse/coach stand-in (partial observability)
  grading/           deterministic, behavioral, memory_exam, judge
  metrics.py         pass^k, cost normalization, per-model rollup
  report.py          ranked per-dimension + weighted composite report
tests/               unit tests (default) + live tests (`-m live`)
results/             trajectories + reports (gitignored)
```

## Requirements

- A working `hermes` (Hermes Agent, validated against **v0.16.0**) on `PATH`.
- **Ollama** for local candidate models and the counterparty.
- Python ≥ 3.11. The mock-world MCP servers run under this project's venv (which
  carries the `mcp` SDK), launched as `python -m simulator.world.<server>`.
- An API key only if you use API-hosted candidates or the LLM judge.
  - For repo-safe persistence, store it in a local `.env` file (see below).

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Run the default candidate field against the 'dana' persona:
.venv/bin/python -m simulator

# Include the Stage-1 pre-filter suite:
.venv/bin/python -m simulator --with-stage1
```

Repo-safe persistent API key setup (one-time):

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=...
```

`python -m simulator` auto-loads `.env` from the project root at startup (without
overwriting already-exported environment variables), so API runs can work without
re-exporting the key every new shell session.

Programmatic use:

```python
from simulator.config import default_run_config
from simulator.pipeline import run_full
from simulator.scenarios.personas import ALL_PERSONAS

report, table = run_full(default_run_config(), list(ALL_PERSONAS.values()))
print(table)
```

## Testing

```bash
.venv/bin/pytest            # fast unit tests (no model needed; uses a fake hermes)
.venv/bin/pytest -m live    # live tests against the real hermes + Ollama (slow)
```

The default suite stubs the `hermes` binary and the LLM calls, so it runs in
seconds with no model. Each implementation unit also carries a `live` test that
exercises the real binary end-to-end.

## Scope

This is an **evaluation simulator**, not the assistant product, and it ships
nothing to end users. It does not benchmark or modify the `hermes-agent` harness,
and it does not evaluate frontier subscription models as the business foundation
— a frontier model is used only as the eval judge.
