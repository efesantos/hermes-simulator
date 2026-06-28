# Hermes Feasibility Spike — Findings (2026-06-28)

Goal: confirm the three assumptions the simulator design depends on, against the
real `hermes-agent` already installed on this machine (no installs needed).

Environment: macOS Apple Silicon (M5 Max, 48GB). Hermes Agent **v0.16.0**
(`~/.local/bin/hermes`, upstream `23021be2`, ~13k commits behind). Ollama
**0.30.6**. All tests ran in throwaway isolated homes under `~/hermes-spike/`;
the real `~/.hermes` profile was never modified.

## Q1 — Headless / non-interactive execution → **YES**

`hermes -z/--oneshot "<prompt>"` sends one prompt and prints **only** the final
response to stdout (no banner, no spinner), auto-bypasses approvals, returns a
real exit code. Built for scripts/pipes.

- **Evidence:** `hermes -z "Reply ... SPIKE_OK"` → stdout `SPIKE_OK`, exit `0`.
- **Reproduce:**
  ```bash
  HERMES_HOME=~/hermes-spike/homeA HERMES_ACCEPT_HOOKS=1 \
    hermes -z "Reply with exactly: SPIKE_OK"
  ```
- Model/provider overridable per-invocation via `-m` / `--provider` (or
  `config.yaml`). `HERMES_ACCEPT_HOOKS=1` needed for unattended runs.

## Q2 — Per-run token + cost accounting → **YES (machine-readable)**

Every run is recorded in `$HERMES_HOME/state.db` (SQLite), table `sessions`, with
columns: `model, input_tokens, output_tokens, cache_read_tokens,
cache_write_tokens, reasoning_tokens, api_call_count, tool_call_count,
billing_provider, estimated_cost_usd, actual_cost_usd, cost_status,
pricing_version, started_at, ended_at`.

- **Evidence:** the SPIKE_OK run recorded `input_tokens=11873, output_tokens=24,
  estimated_cost_usd=0.0` (local Ollama → free). For an API provider the cost
  fields populate automatically.
- **Reproduce:**
  ```bash
  sqlite3 ~/hermes-spike/homeA/state.db \
    "select model,input_tokens,output_tokens,estimated_cost_usd from sessions;"
  ```
- `hermes insights --days N` gives a human summary; the SQLite table is the
  runner's data source.

## Q3 — Isolated + resettable memory → **YES**

A profile **is** a `HERMES_HOME` directory; `HERMES_HOME` is the supported
per-invocation relocation var. Built-in memory lives in
`$HERMES_HOME/memories/{MEMORY.md,USER.md}`.

- **Persistence:** agent saved "Mia has soccer Thursday 4pm" to
  `homeA/memories/USER.md`; a later homeA run recalled "Thursday at 4pm" from
  injected memory alone.
- **Isolation:** a fresh `homeB` (same model) answered **UNKNOWN** to the same
  question — zero cross-track contamination.
- **Reset:** `echo yes | HERMES_HOME=... hermes memory reset` wiped it.
- **Design fit:** one `HERMES_HOME` per (model × persona × seed) gives fully
  isolated memory tracks for free.

## Q4 — Mock world via custom MCP server → **YES**

A custom stdio MCP server (the mechanism for the fake email/calendar/contacts
world) registers and is called by the agent in a headless run.

- **Evidence:** a FastMCP server exposing `get_calendar` (hardcoded data with
  sentinel `ZX9QPLUM`) was registered via `hermes mcp add`; a `-z` run produced
  an answer containing `ZX9QPLUM` with `tool_call_count=1` in `state.db` —
  proving the data came from the tool, not the model.
- **Reproduce:**
  ```bash
  hermes mcp add mockcal --command <venv-python> --args mock_calendar_mcp.py   # confirm with 'y'
  HERMES_HOME=... hermes -z "Look up my calendar for 2026-07-02 using your tools"
  ```
- The MCP Python SDK (`mcp.server.fastmcp.FastMCP`) is already in Hermes's venv;
  `hermes mcp add` prompts to enable discovered tools (pipe `y` for unattended).

## Other findings that shape the build

- **≥64K context required.** `qwen3:8b` (40,960) was rejected:
  *"context window below the minimum 64,000 required."* Override via
  `model.context_length`, or use large-context models. `gemma3:12b` (128K) and
  `qwen3.6:latest` passed. This narrows the candidate field and interacts with cost.
- **Large fixed per-run input overhead.** A one-word task cost **11,873 input
  tokens** — Hermes's system prompt + 40+ tool schemas injected each turn, with
  no prompt caching on local Ollama. This is a primary cost driver and a core
  reason the simulator must measure tokens-to-complete, not sticker price.
- **Model compatibility is not a given.** `gemma3:12b` ran but produced *"no
  final response"* (does not emit Hermes's tool-call format reliably).
  `qwen3.6:latest` worked cleanly. Tool-call-format compatibility belongs in the
  Stage-1 pre-filter.
- **Cold-start overhead per run.** Each one-shot re-runs plugin discovery (38
  plugins). For hundreds of runs, prefer a persistent process (gateway) or
  disabling unused plugins/toolsets over cold one-shots.
- **Local Ollama provider:** bare `provider: ollama` = local, auto-detects
  `localhost:11434`; set `model.base_url: http://localhost:11434/v1` explicitly
  to be safe.

## Verdict

All four mechanisms hold — no remaining feasibility blockers. The simulator's
runner is viable: `HERMES_HOME` per track + custom MCP servers for the mock world
+ `-z` one-shot + read `state.db` for tokens/cost + `memory reset` between runs.

## Cleanup

Throwaway artifacts live in `~/hermes-spike/` (homeA, homeB, report). Safe to
delete: `rm -rf ~/hermes-spike`.
