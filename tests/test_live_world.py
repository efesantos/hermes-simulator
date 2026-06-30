"""Live verification that the persistent gateway killed the cold-start race (U6).

Run with ``.venv/bin/pytest -m live`` against a real ``hermes`` + a tool-capable
model. The decisive assertion is the one from the diagnosis
(``docs/solutions/integration-issues/api-path-mcp-cold-start.md``): with
``HERMES_DUMP_REQUESTS=1``, hermes dumps each request to ``<home>/sessions/*.json``
with ``request.body.tools``. Before this fix, the smoke's first-turn request
carried only hermes' 16 built-in tools — the mock-world calendar/email/contacts
tools were absent, so polite models were eliminated for "did not call any tool".

With the gateway, the three world servers are already running when hermes connects,
so discovery wins deterministically: the mock-world tools must be present at the
first turn on *every* run, regardless of how eagerly the model calls tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulator.config import LOCAL_OLLAMA, CandidateModel
from simulator.harness import Harness
from simulator.runner import SMOKE_PROMPT
from simulator.world.gateway import WorldGateway
from simulator.world.registration import register_world_urls
from simulator.world.state import WorldState

# A mock-world tool that does NOT exist among hermes' built-ins — its presence in
# the request proves the MCP servers' tools landed before the first turn. hermes
# namespaces MCP tools as ``mcp_<server>_<tool>`` (e.g. ``mcp_mockcal_list_events``),
# so we match on the bare suffix.
_WORLD_TOOL = "list_events"


def _has_world_tool(names: set[str]) -> bool:
    return any(n == _WORLD_TOOL or n.endswith(f"_{_WORLD_TOOL}") for n in names)


def _model() -> CandidateModel:
    return CandidateModel(
        id="qwen3.6:latest", hosting_profile=LOCAL_OLLAMA, context_length=65_536
    )


def _find_tool_names(obj) -> list[str]:
    """Recursively collect function names from any ``tools`` list in a dump."""
    names: list[str] = []
    if isinstance(obj, dict):
        tools = obj.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    fn = tool.get("function", tool)
                    name = fn.get("name") if isinstance(fn, dict) else None
                    if name:
                        names.append(name)
        for value in obj.values():
            names += _find_tool_names(value)
    elif isinstance(obj, list):
        for value in obj:
            names += _find_tool_names(value)
    return names


def _requests_with_tools(home: Path) -> list[set[str]]:
    """Tool-name sets for each dumped request that carried any tools."""
    out: list[set[str]] = []
    for dump in sorted((home / "sessions").glob("*.json")):
        names = set(_find_tool_names(json.loads(dump.read_text())))
        if names:
            out.append(names)
    return out


@pytest.mark.live
@pytest.mark.parametrize("run", [1, 2, 3])
def test_smoke_first_turn_sees_world_tools_deterministically(tmp_path: Path, run: int):
    """Repeated smoke runs all carry the mock-world tools at the first turn.

    Parametrized to run several times: the original failure was *deterministic*
    on the API path (every retry lost the race), so a single green run is not
    enough — the point is that it's now green *every* time.
    """
    home = tmp_path / f"home{run}"
    world_db = tmp_path / f"world{run}.db"
    clock_file = tmp_path / f"sim_now{run}"
    WorldState.create(world_db).close()

    harness = Harness(home, _model(), timeout=400)
    harness.setup()
    with WorldGateway(world_db, clock_file) as gateway:
        reg = register_world_urls(harness, gateway.urls)
        assert all(r.ok for r in reg.values())  # URL registration is deterministic
        result = harness.run_oneshot(
            SMOKE_PROMPT, extra_env={"HERMES_DUMP_REQUESTS": "1"}
        )

    assert result.ok
    tool_sets = _requests_with_tools(home)
    assert tool_sets, "hermes dumped no request carrying tools"
    # Every request that carried tools carried the mock-world tools — the race is
    # gone, so the first-turn snapshot is never tool-less.
    for names in tool_sets:
        assert _has_world_tool(names), f"mock-world tools absent from a request: {names}"


@pytest.mark.live
def test_full_api_field_reaches_stage2_without_race_eliminations(tmp_path: Path):
    """A small ``--candidates api`` run ranks the tool-capable models with no
    race-driven 'did not call any tool' eliminations.

    This is the plan's headline verification (it bills OpenRouter). Skipped unless
    ``OPENROUTER_API_KEY`` is set so it never runs accidentally in CI.
    """
    import os

    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("set OPENROUTER_API_KEY to run the paid API-field verification")

    from simulator.config import API_CANDIDATES, RunConfig
    from simulator.runner import Runner
    from simulator.scenarios.personas.dana import PERSONA as DANA

    cfg = RunConfig(candidates=API_CANDIDATES, seeds=(0,), k=1)
    runner = Runner(cfg, results_root=tmp_path)
    matrix = runner.run_matrix([DANA], stage1_tasks=[], run_id="live-api")

    race_eliminated = [
        o.model_id for o in matrix.stage1
        if not o.survived and "did not call any tool" in o.reason
    ]
    assert not race_eliminated, f"race-driven eliminations remain: {race_eliminated}"
