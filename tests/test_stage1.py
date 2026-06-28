"""Tests for the Stage-1 task suite and the deterministic engine it grades with (U4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from simulator.grading.deterministic import check_world, grade_task
from simulator.scenarios.stage1 import STAGE1_TASKS
from simulator.world.state import WorldState


def _task(task_id: str):
    return next(t for t in STAGE1_TASKS if t.id == task_id)


def _world(tmp_path: Path, name: str = "world.db") -> WorldState:
    return WorldState.create(tmp_path / name)


# --- suite hygiene -----------------------------------------------------------


def test_suite_has_unique_ids_and_required_fields():
    ids = [t.id for t in STAGE1_TASKS]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    assert len(STAGE1_TASKS) >= 10
    for t in STAGE1_TASKS:
        assert t.prompt and t.expected_state, f"{t.id} missing prompt/expected_state"


def test_categories_span_the_seams():
    cats = {t.category for t in STAGE1_TASKS}
    assert {"booking", "conflict", "coordination", "cross_domain", "abstention"} <= cats


# --- engine predicates -------------------------------------------------------


def test_time_of_day_predicates():
    snap = {"events": [{"title": "x", "start": "2026-07-15T08:30:00", "end": "2026-07-15T09:00:00"}]}
    assert check_world(snap, {"events_present": [{"start_time_before": "09:00"}]}) == []
    assert check_world(snap, {"events_present": [{"start_time_at_or_after": "09:00"}]}) != []


def test_overlaps_predicate_in_engine():
    snap = {"events": [{"title": "call", "start": "2026-07-16T16:30:00", "end": "2026-07-16T17:00:00"}]}
    window = {"start": "2026-07-16T16:00:00", "end": "2026-07-16T17:00:00"}
    assert check_world(snap, {"events_absent": [{"overlaps": window}]}) != []  # it overlaps


def test_unknown_expected_key_raises():
    with pytest.raises(ValueError, match="unknown expected_state key"):
        check_world({"events": []}, {"bogus": []})


def test_unknown_event_predicate_raises():
    with pytest.raises(ValueError, match="unknown event predicate"):
        check_world({"events": [{"title": "x", "start": "2026-07-15T10:00:00"}]},
                    {"events_present": [{"nope": 1}]})


# --- each expected end-state is reachable AND rejects a wrong one -------------


def test_book_after_9am_pass_and_fail(tmp_path):
    task = _task("book_after_9am")

    good = _world(tmp_path, "good.db")
    good.create_event(title="Dental cleaning", start="2026-07-15T10:00:00",
                      end="2026-07-15T10:30:00")
    passed, _ = grade_task(good, task.expected_state)
    assert passed

    bad = _world(tmp_path, "bad.db")  # booked too early
    bad.create_event(title="Dental cleaning", start="2026-07-15T08:00:00",
                     end="2026-07-15T08:30:00")
    passed, detail = grade_task(bad, task.expected_state)
    assert not passed and "before" in detail.lower() or "09:00" in detail


def test_avoid_double_book_pass_and_fail(tmp_path):
    task = _task("avoid_double_book")

    good = _world(tmp_path, "good.db")
    good.seed(task.world_seed)  # soccer 16:00-17:00
    good.create_event(title="Contractor call", start="2026-07-16T10:00:00",
                      end="2026-07-16T10:30:00")  # no clash
    assert grade_task(good, task.expected_state)[0]

    bad = _world(tmp_path, "bad.db")
    bad.seed(task.world_seed)
    bad.create_event(title="Contractor call", start="2026-07-16T16:30:00",
                     end="2026-07-16T17:00:00")  # double-booked over soccer
    passed, detail = grade_task(bad, task.expected_state)
    assert not passed and "contractor" in detail.lower()


def test_flag_conflict_to_spouse_pass_and_fail(tmp_path):
    task = _task("flag_conflict_to_spouse")

    good = _world(tmp_path, "good.db")
    good.seed(task.world_seed)
    good.send_email(to_addr="sam@home.test", subject="Clash",
                    body="Can't do the dentist then — it clashes with Mia's soccer.",
                    timestamp="2026-07-16T09:00:00")
    assert grade_task(good, task.expected_state)[0]

    bad = _world(tmp_path, "bad.db")  # booked the dentist over soccer, no email
    bad.seed(task.world_seed)
    bad.create_event(title="Dentist", start="2026-07-16T16:00:00",
                     end="2026-07-16T16:30:00")
    assert not grade_task(bad, task.expected_state)[0]


def test_do_not_invent_appointment_pass_and_fail(tmp_path):
    task = _task("do_not_invent_appointment")

    clean = _world(tmp_path, "clean.db")  # left the calendar alone
    assert grade_task(clean, task.expected_state)[0]

    hallucinated = _world(tmp_path, "bad.db")
    hallucinated.create_event(title="Doctor", start="2026-07-18T11:00:00",
                              end="2026-07-18T11:30:00")
    assert not grade_task(hallucinated, task.expected_state)[0]


# --- a suite run yields a per-model pass/fail vector --------------------------


def test_grading_a_world_produces_a_pass_fail_vector(tmp_path):
    # A "model" that only ever creates a generic 10am event on the asked day:
    # passes some tasks, fails others -> a discriminating vector, not all-pass.
    results = {}
    for task in STAGE1_TASKS:
        w = _world(tmp_path, f"{task.id}.db")
        w.seed(task.world_seed)
        w.create_event(title="Generic", start="2026-07-15T10:00:00",
                       end="2026-07-15T10:30:00")
        results[task.id] = grade_task(w, task.expected_state)[0]
    assert any(results.values()) and not all(results.values())


# --- live (real hermes + Ollama; run with `-m live`) -------------------------


@pytest.mark.live
def test_live_known_good_model_passes_a_task(tmp_path):
    """U4 verification: a known-good model reaches a task's expected end-state."""
    import sys

    from simulator.config import LOCAL_OLLAMA, CandidateModel, RunConfig
    from simulator.runner import Runner

    model = CandidateModel(id="qwen3.6:latest", hosting_profile=LOCAL_OLLAMA,
                           context_length=65_536)
    runner = Runner(RunConfig(candidates=(model,), seeds=(0,), k=1),
                    results_root=tmp_path, python_exe=sys.executable,
                    stage1_grader=grade_task)
    result = runner._run_stage1_task(model, _task("book_after_9am"), tmp_path / "s1")
    assert result.passed, result.detail
