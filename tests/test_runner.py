"""Tests for the orchestrator and two-stage funnel (U3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulator.config import LOCAL_OLLAMA, CandidateModel, RunConfig
from simulator.harness import Harness, HarnessResult, SessionRow
from simulator.runner import Runner, evaluate_format_smoke
from simulator.scenarios.types import DayPlan, ExogenousEvent, Persona, Stage1Task
from simulator.world.state import WorldState


def _model(model_id="qwen3.6:latest", ctx=65_536) -> CandidateModel:
    return CandidateModel(id=model_id, hosting_profile=LOCAL_OLLAMA, context_length=ctx)


def _factory(fake_hermes: str):
    return lambda home, model: Harness(home, model, hermes_bin=fake_hermes, timeout=60)


def _session(tool_calls: int) -> SessionRow:
    return SessionRow("m", 100, 10, 0, 0, 0, 1, tool_calls, 0.0, 0.0, "", "")


# --- format-smoke gate (pure) ------------------------------------------------


def test_smoke_passes_with_response_and_tool_call():
    r = evaluate_format_smoke(HarnessResult("DONE", "", 0), _session(1))
    assert r.passed


def test_smoke_fails_on_empty_response():
    # The spike's gemma3:12b "no final response" case.
    r = evaluate_format_smoke(HarnessResult("   ", "", 0), None)
    assert not r.passed and r.reason == "no final response"


def test_smoke_fails_on_nonzero_exit():
    r = evaluate_format_smoke(HarnessResult("anything", "", 1), None)
    assert not r.passed


def test_smoke_fails_when_no_tool_called():
    r = evaluate_format_smoke(HarnessResult("DONE", "", 0), _session(0))
    assert not r.passed and "tool" in r.reason


# --- Stage 1 gates -----------------------------------------------------------


def test_stage1_drops_below_floor_model_without_running(tmp_path: Path):
    runner = Runner(RunConfig(candidates=(_model(ctx=40_960),), seeds=(0,), k=1))
    out = runner.run_stage1(_model(ctx=40_960), [], tmp_path / "run")
    assert not out.eligible and not out.survived
    assert "below floor" in out.reason
    # Reason is persisted (no silent caps).
    saved = json.loads((tmp_path / "run" / "stage1" / "qwen3.6_latest" / "outcome.json").read_text())
    assert saved["reason"] == out.reason


def test_stage1_format_gate_failure_eliminates(tmp_path, fake_hermes, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", "")  # empty => "no final response"
    runner = Runner(
        RunConfig(candidates=(_model(),), seeds=(0,), k=1),
        harness_factory=_factory(fake_hermes),
    )
    out = runner.run_stage1(_model(), [], tmp_path / "run")
    assert not out.eligible and not out.survived
    assert "format" in out.reason


def test_stage1_passes_gates_and_grades_prefilter(tmp_path, fake_hermes, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", "DONE")
    tasks = [
        Stage1Task(id="t1", prompt="do x", expected_state={"ok": True}),
        Stage1Task(id="t2", prompt="do y", expected_state={"ok": True}),
    ]
    # Inject a grader: first task passes, second fails -> 50% < 60% -> dropped.
    calls = {"n": 0}
    def seq_grader(world, expected):
        calls["n"] += 1
        return (calls["n"] == 1, "ok" if calls["n"] == 1 else "wrong")

    runner = Runner(
        RunConfig(candidates=(_model(),), seeds=(0,), k=1),
        harness_factory=_factory(fake_hermes),
        stage1_grader=seq_grader,
    )
    out = runner.run_stage1(_model(), tasks, tmp_path / "run")
    assert out.eligible  # gates passed
    assert out.pass_rate == 0.5
    assert not out.survived  # below 0.6 threshold
    assert {t.task_id for t in out.task_results} == {"t1", "t2"}


# --- Stage 2 day loop --------------------------------------------------------


def _mini_persona() -> Persona:
    return Persona(
        name="mini",
        world_seed={"contacts": [{"name": "Sam", "email": "sam@home.test", "relation": "spouse"}]},
        days=(
            DayPlan(
                day=1, date="2026-07-02", user_prompt="handle today",
                inbound=(
                    ExogenousEvent("event", {"title": "Soccer", "start": "2026-07-02T16:00:00",
                                              "end": "2026-07-02T17:00:00"}),
                ),
            ),
            DayPlan(
                day=2, date="2026-07-03", user_prompt="handle today",
                inbound=(
                    ExogenousEvent("email", {"from_addr": "school@x.test", "to_addr": "me@example.com",
                                             "subject": "Trip", "timestamp": "2026-07-03T08:00:00"}),
                ),
            ),
        ),
    )


def test_stage2_track_completes_and_writes_trajectory(tmp_path, fake_hermes, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", "ok")
    runner = Runner(
        RunConfig(candidates=(_model(),), seeds=(0,), k=1),
        harness_factory=_factory(fake_hermes),
    )
    track = runner.run_stage2_track(_model(), _mini_persona(), seed=0, run_dir=tmp_path / "run")
    assert track.status == "completed"
    assert len(track.days) == 2

    track_dir = Path(track.trajectory_dir)
    assert (track_dir / "day_1.json").exists()
    assert (track_dir / "day_2.json").exists()
    assert (track_dir / "track.json").exists()

    # Inbound exogenous events were applied to the world out-of-band.
    final = json.loads((track_dir / "final_world.json").read_text())
    assert any(e["title"] == "Soccer" for e in final["events"])
    assert any(e["subject"] == "Trip" for e in final["emails"])


def test_stage2_track_failure_is_captured_not_raised(tmp_path, fake_hermes, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", "boom")
    monkeypatch.setenv("FAKE_EXIT", "3")  # every day's run fails
    runner = Runner(
        RunConfig(candidates=(_model(),), seeds=(0,), k=1),
        harness_factory=_factory(fake_hermes),
    )
    track = runner.run_stage2_track(_model(), _mini_persona(), seed=0, run_dir=tmp_path / "run")
    assert track.status == "failed"
    assert "day 1" in track.reason
    assert len(track.days) == 1  # stopped after the failing day


def test_ae1_identical_event_stream_across_seeds(tmp_path, fake_hermes, monkeypatch):
    # Same persona, two seeds: the exogenous stream lands identically regardless
    # of which track it is (the fake agent makes no world changes).
    monkeypatch.setenv("FAKE_STDOUT", "ok")
    runner = Runner(
        RunConfig(candidates=(_model(),), seeds=(0, 1), k=2),
        harness_factory=_factory(fake_hermes),
    )
    persona = _mini_persona()
    t0 = runner.run_stage2_track(_model(), persona, seed=0, run_dir=tmp_path / "run")
    t1 = runner.run_stage2_track(_model(), persona, seed=1, run_dir=tmp_path / "run")

    def world_events(track):
        snap = json.loads((Path(track.trajectory_dir) / "final_world.json").read_text())
        return [(e["title"], e["start"]) for e in snap["events"]]

    assert world_events(t0) == world_events(t1)


# --- counterparty step -------------------------------------------------------


def test_counterparty_step_seeds_replies_for_outbound_mail(tmp_path):
    world_db = tmp_path / "world.db"
    w = WorldState.create(world_db)
    # The agent "sent" an email this day (id 1, > before_id 0).
    w.send_email(to_addr="sam@home.test", subject="Pickup?", body="grab Mia?",
                 timestamp="2026-07-02T12:00:00")
    w.close()

    class ScriptedSpouse:
        def reply(self, outbound_email, persona, *, sim_now):
            # Partial observability: only the email is visible, no tool calls.
            assert "subject" in outbound_email
            return {"from_addr": "sam@home.test", "to_addr": "me@example.com",
                    "subject": "Re: " + outbound_email["subject"], "body": "yes",
                    "timestamp": sim_now}

    runner = Runner(
        RunConfig(candidates=(_model(),), seeds=(0,), k=1),
        counterparty=ScriptedSpouse(),
    )
    day = _mini_persona().days[0]
    n = runner._counterparty_step(world_db, day, _mini_persona(), before_id=0)
    assert n == 1
    inbox = WorldState(world_db).list_emails(folder="inbox")
    assert inbox[0]["subject"] == "Re: Pickup?"


def test_no_counterparty_means_no_replies(tmp_path):
    world_db = tmp_path / "world.db"
    WorldState.create(world_db).close()
    runner = Runner(RunConfig(candidates=(_model(),), seeds=(0,), k=1))
    n = runner._counterparty_step(world_db, _mini_persona().days[0], _mini_persona(), 0)
    assert n == 0


# --- full matrix -------------------------------------------------------------


def test_run_matrix_drops_ineligible_then_runs_survivors(tmp_path, fake_hermes, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", "ok")
    cfg = RunConfig(
        candidates=(_model("good:latest"), _model("toosmall:latest", ctx=40_960)),
        seeds=(0,), k=1,
    )
    runner = Runner(cfg, harness_factory=_factory(fake_hermes), results_root=tmp_path)
    result = runner.run_matrix([_mini_persona()], stage1_tasks=[], run_id="t")

    by_model = {o.model_id: o for o in result.stage1}
    assert by_model["good:latest"].survived
    assert not by_model["toosmall:latest"].survived
    # Only the survivor produced Stage-2 tracks.
    assert {t.model_id for t in result.tracks} == {"good:latest"}
    assert (tmp_path / "t" / "matrix.json").exists()


# --- live (real hermes + Ollama; run with `-m live`) -------------------------


@pytest.mark.live
def test_live_two_day_track_runs_to_completion(tmp_path: Path):
    """U3 verification: a 2-day track runs end-to-end and writes trajectory+metrics.

    Asserts mechanical completion and that per-day token metrics are captured.
    Whether the model *recalls* across days is a grading concern (U7/U8), not the
    harness's — the home is shared across days so memory can persist when saved.
    """
    persona = _mini_persona()
    runner = Runner(RunConfig(candidates=(_model(),), seeds=(0,), k=1), results_root=tmp_path)
    track = runner.run_stage2_track(_model(), persona, seed=0, run_dir=tmp_path / "run")
    assert track.status == "completed"
    assert len(track.days) == 2
    assert all(d.exit_code == 0 for d in track.days)
    assert all(d.session is not None and d.session.total_tokens > 0 for d in track.days)
