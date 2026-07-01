# hermes-simulator

A Python benchmark harness that runs candidate open-weight models inside the fixed
Nous `hermes-agent` CLI and ranks them for a personal-assistant business on
capability, memory, reliability, and cost. See `README.md` for the full picture.

## Knowledge stores

- **`CONCEPTS.md`** — shared domain vocabulary (Track, Persona, Counterparty,
  two-stage funnel, mock world, out-of-band grading, …). Relevant when orienting to
  the codebase or discussing domain concepts.
- **`docs/solutions/`** — documented solutions to past problems (bugs, best
  practices, workflow patterns), organized by category with YAML frontmatter
  (`module`, `tags`, `problem_type`). Relevant when implementing or debugging in a
  documented area — e.g. MCP tool-loading / cold-start races.
- **`docs/`** also holds the brainstorm, plan, spike, and benchmark-findings docs.

## Working in this repo

- Python ≥ 3.11; the venv lives at `.venv`. Install with `.venv/bin/pip install -e ".[dev]"`.
- Tests: `.venv/bin/pytest` runs the fast suite (stubs the `hermes` binary and LLM
  calls — no model needed). `.venv/bin/pytest -m live` runs the slow tests against the
  real `hermes` + Ollama; each implementation unit carries one.
- Run the benchmark: `.venv/bin/python -m simulator` (default = local Ollama field,
  `dana` persona). `--candidates api` runs the OpenRouter field (needs
  `OPENROUTER_API_KEY` in `.env`); `--candidates api-free` validates at $0;
  `--seeds N` / `--model <id>` tune the run; the judge runs via the Claude Code
  subscription (`claude -p`). `scripts/build_report.py <run_id>` rebuilds a ranked
  report from on-disk results if a run was interrupted. Long unattended runs belong
  on a VPS — see `docs/guides/vps-unattended-run.md`.
- `simulator/harness.py` is the only module that shells out to `hermes`; the grader
  reads world state out-of-band (never through an agent tool). The mock-world MCP
  servers run as a persistent per-track HTTP gateway (`simulator/world/gateway.py`)
  so tool discovery never races the agent's first turn.
- Latest benchmark state: `docs/benchmark-findings-2026-06-30.md` (API field, 5-seed
  definitive: **Mistral-Large leads**; the 3-seed GLM-5.2 lead did not reproduce).
  `docs/benchmark-findings-2026-06-29.md` is the local field.
