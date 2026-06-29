---
title: MCP cold-start race — tool-starved agent runs corrupt benchmark results
date: 2026-06-29
category: integration-issues
module: simulator/harness.py
problem_type: integration_issue
component: tooling
symptoms:
  - "Models wrongly eliminated at Stage 1 for 'did not call any tool' when they are in fact capable"
  - "Starved-run token signature: ~8-12K input (no tool schemas in context) with 0 tool calls"
  - "Healthy run by contrast: ~20K+ input with at least one tool call"
  - "Non-deterministic: re-running the same model sometimes passes, sometimes starves"
root_cause: async_timing
resolution_type: code_fix
severity: high
tags:
  - mcp
  - ollama
  - cold-start
  - race-condition
  - hermes-agent
  - benchmark-harness
---

# MCP cold-start race — tool-starved agent runs corrupt benchmark results

## Problem

When Ollama cold-loads a model, that load races the three mock-world MCP servers'
boot. Hermes sometimes starts the agent before the servers are ready, so the agent
runs with **no world tools** and every tool-requiring task fails as a silent
*artifact* rather than a real model verdict. The first full benchmark eliminated
every candidate model — mostly to this infrastructure race, not genuine incapability.

## Symptoms

- **Artifact eliminations**: models dropped at Stage 1 (format smoke / pre-filter)
  for "did not call any tool" when they were actually capable — the entire first
  benchmark run wiped out.
- **Token signature of a starved run**: ~8–12K input tokens (no tool schemas in
  context) and `tool_call_count == 0`. A healthy run shows ~20K+ input and
  `tool_call_count >= 1`.
- The agent's final text reflects the missing tools — e.g. saying it "doesn't have
  a calendar tool" — instead of acting on the world.
- **Non-deterministic**: re-running the same model sometimes passed and sometimes
  starved, depending on who won the load-vs-boot race. A live world test had to be
  made explicitly "resilient to MCP cold-start nondeterminism".

## What Didn't Work

**1. `warm()` alone.** Preloading the model fixed the *slow*-loading big models
(qwen3.6 went 4/10 → 10/10) because their long load no longer overran MCP boot. But
it exposed an **inverse race**: *fast*-loading small models (Hermes-3 8B) now started
the agent almost instantly — before the freshly spawned MCP servers finished booting
— and still ran tool-starved. Warming addresses one side of the race but cannot by
itself guarantee the servers win.

**2. A blunt "retry on ANY 0-tool run."** Retrying every tool-less run recovered
starved runs, but was wrong for a different failure mode: weak models (e.g.
Hermes-3 8B) frequently *have* the tools loaded and simply answer without calling
one. Those runs are real results, and blindly retrying burned up to 3× the runs on
them for no benefit. This was refined with an input-token gate.

## Solution

Two coordinated changes in `simulator/harness.py`, wired through `simulator/runner.py`.

**1. `Harness.warm()`** — preload the local model into Ollama, at the forced
context, before any tools are registered or any run happens:

```python
def warm(self, *, keep_alive: str = "2m", timeout: float = 300.0) -> bool:
    """Preload a local model into Ollama so it's resident before a run."""
    if self.model.hosting != Hosting.LOCAL:
        return False  # no-op for API models
    endpoint = self.model.base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[: -len("/v1")]
    payload = {
        "model": self.model.id, "prompt": "", "stream": False,
        "keep_alive": keep_alive, "options": {"num_ctx": self.model.context_length},
    }
    req = urllib.request.Request(
        f"{endpoint}/api/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False  # best-effort; the run still proceeds
```

**2. `run_oneshot(expect_tools=True)`** — a token-gated retry. Retry a 0-tool run
*only* when the input is small enough that the schemas clearly weren't loaded:

```python
TOOLS_LOADED_MIN_INPUT = 15_000  # clear gap between no-tools base (~8-12K)
                                 # and with-tools input (~20K+)

def run_oneshot(self, prompt, *, extra_env=None, expect_tools=False, tool_retries=3):
    result = self._run_once(prompt, extra_env)
    if not expect_tools:
        return result
    for _ in range(tool_retries):
        session = self.latest_session()
        # None session => can't verify (fake binary / no state.db); don't spin.
        if session is None or session.tool_call_count >= 1:
            return result
        # Zero tool calls: retry ONLY if schemas weren't loaded (low input =>
        # MCP cold-start starvation). High input means tools WERE present and the
        # model chose not to use them — a real result; retrying would burn runs.
        if session.input_tokens >= TOOLS_LOADED_MIN_INPUT:
            return result
        result = self._run_once(prompt, extra_env)
    return result
```

**3. Wiring (`simulator/runner.py`).** `_prepared_harness` warms *before* registering
the world, so the load can never race MCP startup; and every tool-requiring entry
point passes `expect_tools=True`:

```python
def _prepared_harness(self, model, home, world_db) -> Harness:
    harness = self.harness_factory(home, model)
    harness.setup()
    # Warm the model BEFORE registering/running so its load can't race MCP
    # server startup (which otherwise leaves the agent with no world tools).
    if self.warm_models:
        harness.warm()
    register_world(harness, str(world_db), python_exe=self.python_exe)
    return harness
```

`_format_smoke`, `_attempt_stage1_task`, and `_run_day` all call
`run_oneshot(..., expect_tools=True)`. `warm_models` defaults on for real runs and
off when a custom `harness_factory` is injected (tests/fakes don't talk to Ollama).

## Why This Works

The root cause is an **asynchronous startup race** between two independent boot
sequences Hermes does not synchronize: Ollama loading the model, and the three stdio
MCP servers spawning and registering their tool schemas. Whoever loses, the agent's
first turn fires without tools in context.

- **`warm()`** removes the model-load side of the race: by the time tools are
  registered and the run starts, the model is already resident in Ollama (held by
  `keep_alive`), so there's no multi-second load to overrun MCP boot.
- **The input-token gate is the key discriminator** between the two reasons a run
  shows 0 tool calls. The mock-world schemas add ~10–12K tokens of input, creating a
  clean separation: a starved run sits at the model's no-tools base (~8–12K), a
  healthy-but-non-calling run sits at with-tools input (~20K+). `TOOLS_LOADED_MIN_INPUT
  = 15_000` lands in that gap. Below it, the schemas demonstrably weren't in context
  → infrastructure starvation → a retry can recover. At or above it, the tools *were*
  present and the model chose not to call one → a genuine result → do not retry. This
  is what lets the same mechanism rescue capable models without wasting 3× runs on
  weak ones.
- Retrying is **safe** because a tool-less run made no world changes — there's no
  state to roll back.

## Prevention

- **Always warm before running.** Any new entry point that spawns MCP servers and
  then invokes the model must route through `_prepared_harness` (or call
  `harness.warm()` before `register_world`).
- **Use the token signature as a diagnostic.** When a model is eliminated for "did
  not call any tool", check `session.input_tokens`. ~8–12K ⇒ suspect infra starvation
  (artifact), not the model. ~20K+ ⇒ a real no-call verdict. Fastest triage for "real
  failure or artifact?"
- **Pass `expect_tools=True`** on any run where a tool call is expected; never trust a
  bare 0-tool run as a verdict for tool-requiring tasks.
- **Recalibrate the threshold if the tool surface changes.** `TOOLS_LOADED_MIN_INPUT`
  is tuned to the ~10–12K schema cost of the three mock-world servers; changing the
  number or size of MCP tools may move the no-tools/with-tools gap.
- **Deeper fix (not yet implemented): a persistent MCP gateway.** The warm+retry
  combination mitigates a race that exists *because* the three MCP servers are spawned
  fresh per run. Running them as long-lived processes Hermes connects to would
  eliminate the boot race at its source and make the token gate a safety net rather
  than a load-bearing mechanism.

## Related Issues

- `docs/benchmark-findings-2026-06-29.md` — the run that surfaced this; §"The
  engineering story" item 1 documents the blast radius (most of run 1's eliminations
  were artifacts) and the deferred persistent-gateway fix.
- Source: `simulator/harness.py` (`warm()`, `run_oneshot`, `TOOLS_LOADED_MIN_INPUT`),
  `simulator/runner.py` (`_prepared_harness`, `warm_models`), `simulator/world/registration.py`
  (registers the three racing MCP servers).
