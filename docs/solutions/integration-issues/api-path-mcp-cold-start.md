---
title: API-path MCP cold-start — fast remote models fail the tool smoke deterministically
date: 2026-06-29
category: integration-issues
module: simulator/harness.py
problem_type: integration_issue
component: tooling
symptoms:
  - "Tool-capable API models (GLM-5.2, Llama-3.3-70B) eliminated at Stage-1 smoke for 'did not call any tool'"
  - "The same models DO emit native tool_calls when called directly over OpenRouter"
  - "Owl Alpha passes the smoke in the same run; GLM/Llama fail every retry"
  - "HERMES_DUMP_REQUESTS shows the request carries only hermes' 16 built-in tools — the mock-world MCP tools are absent"
root_cause: async_timing
resolution_type: pending
severity: high
tags:
  - mcp
  - cold-start
  - race-condition
  - hermes-agent
  - openrouter
  - api-models
  - benchmark-harness
---

# API-path MCP cold-start — fast remote models fail the tool smoke deterministically

## Problem

When the candidate is an **API model** (OpenRouter), the Stage-1 format smoke
eliminates tool-capable models (GLM-5.2, Llama-3.3-70B) for "did not call any
tool" — while **Owl Alpha passes in the very same run**. This is the
[[mcp-cold-start-race]] again, but the local mitigations (warm + token-gated
retry) **do not work on the API path**, and here they fail *deterministically*.

## Evidence chain (how we know it's not the models)

1. **Direct OpenRouter probe**: GLM-5.2 and Llama-3.3-70B both return standard
   native `tool_calls` for a tool-requiring prompt — including the exact smoke
   prompt and a FastMCP-style nullable-union (`anyOf`/null) schema. The models
   tool-call fine over OpenRouter.
2. **hermes always sends native tools**: `agent/transports/chat_completions.py`
   adds `api_kwargs["tools"]` for every model on the chat_completions path (the
   OpenRouter path) and reads `message.tool_calls` natively — no prompted/XML
   format, no per-family gating, no config flag. So it is not a tool-call-format
   mismatch.
3. **The decisive artifact** — set `HERMES_DUMP_REQUESTS=1` and re-run; hermes
   dumps each request to `<home>/sessions/*.json` (`reason: preflight`). For a
   failed GLM smoke the request carried **only hermes' 16 built-in tools**
   (`clarify`, `terminal`, `read_file`, …) — the **mock-world calendar/email/
   contacts MCP tools were absent**. The smoke says "use your calendar tools",
   but no calendar tool was present, so GLM correctly answered in prose (zero
   tool calls) and failed the "called any tool" gate. Owl Alpha eagerly calls a
   *built-in* tool on turn 1 and passes the same gate.

## Root cause

The oneshot (`hermes -z`) path **backgrounds** MCP discovery and waits only
**0.75s** before the first tool snapshot:

- `hermes_cli/main.py` → `_should_background_mcp_startup(args)` is True for the
  oneshot command, so `start_background_mcp_discovery()` runs the connect+list in
  a daemon thread.
- `hermes_cli/mcp_startup.py:wait_for_mcp_discovery(timeout=0.75)` joins that
  thread for at most 0.75s before the agent's first turn.

Three freshly-spawned **Python** stdio MCP servers (the mock world) take longer
than 0.75s to import, boot, and register, so the first turn fires with only
built-in tools. Then:

- **`warm()` no-ops for API models** (no Ollama load), so the incidental
  multi-second delay that lets MCP win the race for *local* models is gone.
- **The token-gated retry can't recover it.** Two reasons: (a) the API no-tools
  baseline is ~12-13K input (hermes' 16 built-in tool schemas), which sits *near*
  `TOOLS_LOADED_MIN_INPUT = 15_000` — the gap the gate relies on (no-tools ~8-12K
  vs with-tools ~20K+) does not hold on the API path; (b) more fundamentally, the
  race is **deterministic** here — with no model-load jitter, every retry hits the
  same fixed 0.75s window and the Python servers lose it every time. GLM failed all
  4 attempts; Owl passed because it calls a built-in tool, not because it won the
  MCP race.

So the smoke is **biased toward eager tool-callers**, not a clean capability gate,
on the API path. Each Stage-2 day is also a oneshot, so the same race affects the
whole benchmark for non-eager remote models — they never get to demonstrate
capability.

## Secondary finding — OpenRouter tool-use availability varies

`nousresearch/hermes-3-llama-3.1-70b` returns HTTP 404 *"No endpoints found that
support tool use"* on OpenRouter — its providers expose no tool-calling endpoint,
so it cannot run as an agent there at all (distinct from the race above; it never
makes a successful call). Guard added: `simulator/openrouter.py` +
`__main__` check filters candidates by `supported_parameters` containing `tools`
before a run. Replaced with `meta-llama/llama-3.3-70b-instruct` (tools=True).

## Fixes (pending a decision)

Neither is applied yet — both touch the harness contract and warrant a call.

**A. Env-gate the discovery wait (smallest).** Make
`wait_for_mcp_discovery` honor `HERMES_MCP_DISCOVERY_WAIT` (a join with a larger
ceiling returns as soon as discovery finishes, so there is no cost when it is
fast), then have the harness export e.g. `HERMES_MCP_DISCOVERY_WAIT=20` for every
run. This de-biases the smoke for *all* models equally (consistent with the
fixed-harness principle). **Caveat:** it patches the out-of-repo hermes install
(`~/.hermes/hermes-agent/hermes_cli/mcp_startup.py`), so it is lost on hermes
reinstall and must be re-applied or upstreamed. (An automated edit was correctly
blocked as an out-of-repo modification; apply by hand or via an install hook.)

**B. Persistent MCP gateway (the real fix, already flagged as deferred).** Run the
three mock-world servers as long-lived HTTP/SSE servers started once by the runner;
register them with hermes as remote servers. hermes then only connects+lists an
*already-running* server (fast), so discovery wins the 0.75s window without any
hermes patch. Eliminates the race at its source for local and API alike, and makes
the token gate a safety net rather than load-bearing. Larger change to
`simulator/world/registration.py` + server lifecycle.

## Diagnostic quick-reference

- `HERMES_MCP cold-start on API?` → set `HERMES_DUMP_REQUESTS=1`, run, and inspect
  `<home>/sessions/*.json` `request.body.tools`. If the mock-world tool names
  (`list_events`, …) are missing and only built-ins are present, it's this race.
- Confirm the model itself is fine with a direct OpenRouter `/chat/completions`
  call carrying a `tools=[...]` param — if it returns `tool_calls`, the model is
  not the problem.

## Related

- [[mcp-cold-start-race]] — the local-path version and the warm + token-gate
  mitigation that this documents the API-path limits of.
- `docs/benchmark-findings-2026-06-29.md` — the persistent-gateway fix was flagged
  here as deferred; this finding raises its priority for the API track.
- Source: `simulator/harness.py` (`warm()` no-ops for API; `run_oneshot` retry),
  `simulator/openrouter.py` (tool-support guard), `~/.hermes/hermes-agent/hermes_cli/mcp_startup.py`
  (the 0.75s wait).
