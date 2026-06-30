---
title: "feat: Persistent MCP gateway — kill the cold-start race for reliable API benchmarking"
type: feat
status: active
date: 2026-06-30
---

# feat: Persistent MCP gateway — kill the cold-start race for reliable API benchmarking

## Summary

Run the three mock-world MCP servers (`mockcal`, `mockemail`, `mockcontacts`) as **persistent HTTP servers started once per track** and register them with hermes **by URL**, instead of spawning them as fresh stdio subprocesses on every `hermes -z` call. Because the servers are already listening when each day's agent connects, hermes' tool discovery completes before the first turn — deterministically, for the Stage-1 smoke and every Stage-2 day, for all models regardless of how eagerly they call tools. This removes the nondeterministic gate that currently makes the API-model field unbenchmarkable.

---

## Problem Frame

Today `simulator/world/registration.py` registers each mock-world server as a **stdio** server (`hermes mcp add <name> --command <python> --args -m <module> <world_db>`). hermes then spawns all three as fresh Python subprocesses on **every** `hermes -z` invocation — the Stage-1 smoke and each persona-day. Those Python servers take longer to import and register than hermes' ~0.75s first-turn discovery wait, so the agent's first turn fires with only hermes' 16 built-in tools and none of the mock-world tools.

For **local** models, `Harness.warm()` masks this — the multi-second Ollama load incidentally gives discovery time to land. For **API** models, `warm()` is a no-op, so the race is lost, and:

- The Stage-1 "format smoke" becomes a **nondeterministic gate**. Eager models (Owl Alpha) call a built-in tool and pass; borderline models (GLM-5.2) flip-flop run to run; polite models (Llama-3.3) decline a non-existent calendar tool and always fail. Verified with `HERMES_DUMP_REQUESTS=1` dumps showing only the 16 built-in tools at the smoke's first turn, even though the MCP stderr shows the servers starting and answering `ListToolsRequest`.
- The same race affects **every Stage-2 day**, so even models that pass the smoke get quality-degraded, eagerness-biased results.

An env-gated wait patch (`HERMES_MCP_DISCOVERY_WAIT`, applied via `scripts/patch_hermes_mcp_wait.py`) helped multi-turn stage runs but did **not** fix the single-turn smoke snapshot. The race must be removed at its source.

Full diagnosis: `docs/solutions/integration-issues/api-path-mcp-cold-start.md` and `docs/solutions/integration-issues/mcp-cold-start-race.md`.

**In scope:** mock-world server transport (stdio → HTTP), the harness registration seam, a per-track server-lifecycle manager, runner wiring, and the per-day clock delivery that the transport change breaks. **Out of scope:** hermes source (treated as fixed/vendored), grading/scoring, candidate/judge wiring, and the persona/world data model.

---

## Key Technical Decisions

**KTD1 — Persistent HTTP servers, one set per track (not per process, not one global set).**
Each track already gets its own `world.db` for memory isolation. The gateway starts the three servers bound to that track's `world.db` before the track's first day, keeps them alive across all the track's days, and tears them down after. This preserves per-track isolation while making discovery a fast connect-to-a-running-server instead of a subprocess boot race. A single global set is rejected: it cannot serve per-track worlds without per-connection world routing, which MCP does not give us cleanly.

**KTD2 — `streamable-http` transport via FastMCP.**
The installed `mcp` SDK's `FastMCP.run` accepts `transport='streamable-http'` with `host`/`port`/`streamable_http_path` settings (default path `/mcp`). Servers bind to `127.0.0.1:<port>` and are registered with `hermes mcp add <name> --url http://127.0.0.1:<port>/mcp`. (`sse` is the legacy alternative; `streamable-http` is the current default HTTP transport and what we target.)

**KTD3 — Unify on the HTTP path for all candidate runs (local and API).**
The three world servers are per-track regardless of whether the candidate is local Ollama or a remote API. Rather than maintain two registration paths, all runs use the HTTP gateway. This removes the race for local runs too (so the warm-vs-MCP coupling stops being load-bearing) at the cost of a one-time live retest of the local field. Trade-off accepted for a single code path.

**KTD4 — Per-day simulated clock via a per-track sidecar file (replaces `HERMES_SIM_NOW` env).**
A persistent server captures process env once at startup, so the per-day `HERMES_SIM_NOW` mechanism breaks. Instead, the runner writes the current simulated time to a per-track file (e.g. `<track_home>/sim_now`) before each day, and `sim_now()` reads that file per call. The file path is passed to each server at startup (argv/env). This needs no `world.db` schema change and stays invisible to the agent (it is not an MCP tool). Alternatives — a `sim_now` column in `world.db`, or a control endpoint/tool — are heavier or risk agent visibility.

**KTD5 — A dedicated, injectable lifecycle manager.**
Server startup, readiness polling, URL registration, and guaranteed teardown live in one new module exposed as a context manager, and are injected into the runner the same way `HarnessFactory` is, so the fast test suite can substitute a fake and never spawn real servers.

**KTD6 — Deterministic readiness replaces the race; existing mitigations become safety nets.**
The gateway polls each server's HTTP endpoint until it responds before registering URLs and running day 1. Once discovery is deterministic, `warm()` and the `TOOLS_LOADED_MIN_INPUT` token-gate retry are no longer load-bearing for tool availability. They are kept as cheap safety nets (a server that dies mid-run still surfaces), with comments downgrading their role; they are not removed in this plan.

---

## High-Level Technical Design

Per-track lifecycle (replaces today's per-`hermes -z` stdio spawn):

```mermaid
sequenceDiagram
    participant R as Runner (_prepared_harness / track loop)
    participant G as WorldGateway (new)
    participant S as 3 HTTP servers (mockcal/email/contacts)
    participant H as hermes -z (per day + smoke)
    participant A as Agent (model under test)

    R->>G: start(world_db, clock_file)  [once per track]
    G->>S: spawn on free 127.0.0.1:<port> (streamable-http)
    G->>S: poll /mcp until ready (deterministic)
    G-->>R: URLs [http://127.0.0.1:p/mcp ...]
    R->>H: add_remote_mcp_server(name, url)  (hermes mcp add --url)
    loop each persona day (and the smoke)
        R->>G: set_clock(day.clock())  -> write sim_now file
        R->>H: hermes -z <prompt>
        H->>S: connect + ListTools  (servers already up -> wins)
        A->>S: list_events / create_event / ...  (tools present turn 1)
    end
    R->>G: stop()  (try/finally; also on error & KeyboardInterrupt)
    G->>S: terminate + reap
```

The key difference from the current architecture: discovery is a connect-to-running-server (sub-second, deterministic) instead of a per-invocation subprocess boot that races a fixed wait window.

---

## Implementation Units

### U1. HTTP transport for the mock-world servers

**Goal:** Let each mock-world server run as a `streamable-http` server on a given host/port, while keeping stdio runnable for ad-hoc manual use.

**Dependencies:** none.

**Files:** `simulator/world/calendar_server.py`, `simulator/world/email_server.py`, `simulator/world/contacts_server.py`, `simulator/world/_server_common.py`, `tests/test_world_servers.py` (new or extended).

**Approach:** Add a shared bootstrap in `_server_common.py` that reads transport config (host, port, world_db path, clock-file path) from argv/env and constructs `FastMCP(name, host=..., port=...)`. Each server module calls a shared `run_server(mcp)` that selects `transport='streamable-http'` when a port is configured, else falls back to stdio. Tool functions are unchanged. Keep the `world_from_argv()` contract working for the stdio fallback.

**Patterns to follow:** the existing `FastMCP("mockcal")` + `@mcp.tool()` definitions; `_server_common.world_from_argv()` arg handling.

**Test scenarios:**
- Happy path: starting a server module with a port binds and serves `/mcp`; `list_events` returns seeded events from the bound `world.db`.
- Edge: missing port → stdio mode still works (argv world_db only).
- Error: an already-bound port surfaces a clear startup failure (not a silent hang).

**Verification:** a server process started on a free port answers an MCP `ListTools`/`list_events` call against a seeded `world.db`.

### U2. Per-day clock via sidecar file

**Goal:** Decouple the simulated clock from process env so persistent servers reflect the current day.

**Dependencies:** U1.

**Files:** `simulator/world/_server_common.py`, `simulator/runner.py`, `tests/test_world_clock.py` (new).

**Approach:** `sim_now()` reads the simulated time from a clock file whose path is provided at server startup (argv/env), re-reading per call so updates land without restart. Fall back to `HERMES_SIM_NOW` then `_FALLBACK_NOW` when no clock file is configured (preserves stdio/manual use). The runner writes `day.clock()` to the track's clock file before each day's run (and before the smoke), replacing the per-day `extra_env={"HERMES_SIM_NOW": ...}` injection.

**Patterns to follow:** current `sim_now()` env read; runner `_run_day` clock injection at the `extra_env` call site.

**Test scenarios:**
- Happy path: writing time T to the clock file makes `sim_now()` return T; rewriting to T2 makes the next call return T2 (no restart).
- Edge: no clock file configured → falls back to `HERMES_SIM_NOW`, then `_FALLBACK_NOW`.
- Integration: a write performed by a stand-in for the day loop is observed by a server reading the same file (proves the runner↔server channel).

**Verification:** across two simulated days, a running server stamps writes with each day's time without being restarted.

### U3. Remote (URL) MCP registration on the harness

**Goal:** Give `Harness` a way to register a remote MCP server by URL.

**Dependencies:** none (parallel to U1/U2).

**Files:** `simulator/harness.py`, `tests/test_harness.py`.

**Approach:** Add `add_remote_mcp_server(name, url)` mirroring `add_mcp_server` but invoking `hermes mcp add <name> --url <url>` (auto-confirming the enable prompt as the stdio variant does). Keep `add_mcp_server` for the stdio fallback.

**Patterns to follow:** existing `Harness.add_mcp_server` (subprocess construction, `input="y\n"`, `HarnessResult`).

**Test scenarios:**
- Happy path: `add_remote_mcp_server` issues `hermes mcp add <name> --url <url>` and returns a `HarnessResult` (assert against the fake-binary harness used in the suite).
- Edge: a non-zero hermes exit is surfaced in the `HarnessResult`, not swallowed.

**Verification:** registering a URL writes a `url`-transport server into the home's hermes config.

### U4. World gateway lifecycle manager

**Goal:** Start/poll/stop the three per-track HTTP servers as one injectable, exception-safe unit.

**Dependencies:** U1, U2.

**Files:** `simulator/world/gateway.py` (new), `tests/test_world_gateway.py` (new).

**Approach:** A `WorldGateway` context manager that, given a `world_db` and clock-file path, picks three free loopback ports (probe-and-bind to avoid collisions), spawns the three servers under the project venv python, polls each `/mcp` endpoint until ready (bounded timeout → clear error), and exposes `urls` (name→URL) and `set_clock(t)`. `__exit__` (and explicit `stop()`) terminate and reap all children, and must run on success, exception, and `KeyboardInterrupt`. Expose a factory type so the runner can inject a fake in tests.

**Test scenarios:**
- Happy path: entering the context starts three servers and yields three reachable URLs; exiting stops all three (no orphaned processes).
- Readiness: a server slow to bind is waited for up to the timeout; exceeding the timeout raises a clear error and tears down any already-started servers.
- Failure/cleanup: an exception raised inside the `with` block still terminates all children (assert no live PIDs afterward).
- Edge: three free ports are chosen even when some candidate ports are occupied.

**Verification:** after a gateway context exits (normally or via exception), no server processes remain and the ports are free.

### U5. Wire the gateway into the runner; retire per-process stdio registration

**Goal:** Make tracks use the persistent gateway for the smoke and all days.

**Dependencies:** U3, U4.

**Files:** `simulator/runner.py`, `simulator/world/registration.py`, `tests/test_runner.py` (extend), `tests/test_integration.py` (extend).

**Approach:** Replace the per-`_prepared_harness` `register_world` stdio call with a per-track flow: start the `WorldGateway` for the track's `world.db`, register its URLs via `add_remote_mcp_server`, run the smoke + day loop inside the gateway context (so `set_clock` is called per day), and stop the gateway in a `finally`. Repoint `register_world` to URL registration (or fold it into the gateway start), keeping the same return shape so callers/tests stay stable. The gateway factory is injected through the runner like `HarnessFactory`, defaulting to the real one and overridden to a fake in the fast suite. The smoke must run inside the same gateway context as the days so its first turn also sees the world tools.

**Patterns to follow:** `HarnessFactory` injection; the existing `_prepared_harness` → `register_world` call site; the `for seed`/day-loop structure.

**Test scenarios:**
- Integration (fast, faked gateway+harness): a track runs the smoke + days against injected fakes; URLs are registered once per track; `set_clock` is called once per day; the gateway is stopped exactly once even when a day raises.
- Edge: a gateway start failure for a track is recorded as a track failure (degraded evaluation) without crashing the whole matrix, mirroring the existing per-track isolation.
- Verification that the smoke runs within the gateway context (not before start / after stop).

**Verification:** the fast suite passes with faked gateway/harness; the runner registers URLs and drives `set_clock` per day.

### U6. Live end-to-end verification + mitigation downgrade + docs

**Goal:** Prove the race is gone against the real stack and record the resolution.

**Dependencies:** U5.

**Files:** `tests/test_live_world.py` (extend or add), `simulator/harness.py` (comment-level downgrade of `warm()`/token-gate role), `docs/solutions/integration-issues/api-path-mcp-cold-start.md` (mark resolved), `docs/benchmark-findings-2026-06-29.md` (note the fix).

**Execution note:** This unit's core assertion is a live test (`pytest -m live`) against real hermes + servers; treat the live check as the unit's definition of done.

**Approach:** Add a `-m live` test that, for a tool-capable model, asserts the smoke's first-turn request includes mock-world tools (via `HERMES_DUMP_REQUESTS` dump inspection) across repeated runs — i.e. deterministic, not eagerness-dependent. Downgrade the `warm()` and `TOOLS_LOADED_MIN_INPUT` comments from "load-bearing mitigation" to "safety net." Update the cold-start docs to "resolved by the persistent gateway."

**Test scenarios:**
- Live: repeated smoke runs for a previously-polite model (e.g. Llama-3.3) all show mock-world tools at the first turn and pass the smoke deterministically.
- Live: a full `--candidates api --seeds N` run completes with GLM-5.2, Llama-3.3, and Owl all reaching Stage-2 (none eliminated for "did not call any tool").
- `Test expectation: none` for the doc/comment edits.

**Verification:** `python -m simulator --candidates api --seeds N` ranks all three API models with no race-driven eliminations, repeatably.

---

## Scope Boundaries

- **In scope:** server transport, clock delivery, remote registration, gateway lifecycle, runner wiring, live verification, mitigation comment-downgrade, doc updates.
- **Out of scope (non-goals):** hermes source changes; grading/scoring; candidate/judge selection; persona/world schema.

### Deferred to Follow-Up Work
- Parallelizing tracks (the gateway already picks free ports defensively, but the runner stays sequential here).
- Removing `warm()` / the token-gate entirely (kept as safety nets this round).
- Re-running and refreshing `docs/benchmark-findings` with a full reliable API field (a benchmarking task, not this infra change).

---

## Risks & Dependencies

- **Port flakiness / orphaned processes.** Mitigated by probe-and-bind free-port selection and guaranteed teardown (U4) covering success, exception, and `KeyboardInterrupt`.
- **`streamable-http` path/version drift in the `mcp` SDK.** Grounded against the installed SDK (`FastMCP.run` accepts `streamable-http`; default `/mcp` path); U1 tests pin the served path.
- **Per-day clock correctness.** A stale clock file would mis-stamp writes; U2 tests assert per-day re-read without restart.
- **Local-field regression** from KTD3 (local also moves to HTTP). Mitigated by a live retest of the local field as part of U6.
- **Startup latency per track.** Three HTTP servers + readiness poll add a few seconds per track; acceptable versus the eliminated reruns, and the gateway is per-track (amortized across the track's days).

---

## Verification Strategy

1. Fast suite (`.venv/bin/pytest`) stays green with faked gateway/harness — no real servers spawned.
2. `.venv/bin/pytest -m live` proves deterministic first-turn tool availability for a polite model across repeated runs.
3. `python -m simulator --candidates api --seeds N` ranks GLM-5.2, Llama-3.3, and Owl with zero race-driven eliminations, repeatably (the original failure reproduced as a pass).
4. `HERMES_DUMP_REQUESTS=1` smoke dump shows `list_events` et al. present at the first turn.

---

## Sources & Research

- `docs/solutions/integration-issues/api-path-mcp-cold-start.md` — root-cause diagnosis (this session).
- `docs/solutions/integration-issues/mcp-cold-start-race.md` — the local-path version and warm+token-gate mitigation.
- Installed `mcp` SDK: `FastMCP.run(transport='stdio'|'sse'|'streamable-http')`, settings `host`/`port`/`streamable_http_path` (default `/mcp`).
- hermes `mcp add --url` remote registration (`hermes_cli/subcommands/mcp.py`, `hermes_cli/mcp_config.py`) — treated as fixed/vendored.
