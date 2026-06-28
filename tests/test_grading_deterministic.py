"""Tests for deterministic state-diff grading reproducibility and immunity (U7)."""

from __future__ import annotations

from pathlib import Path

from simulator.grading.deterministic import check_world, grade_task
from simulator.world.state import WorldState


def _world(tmp_path: Path, name="w.db") -> WorldState:
    return WorldState.create(tmp_path / name)


def test_scores_are_reproducible_across_reruns(tmp_path):
    w = _world(tmp_path)
    w.create_event(title="Dental", start="2026-07-15T10:00:00", end="2026-07-15T10:30:00")
    expected = {"events_present": [{"title_contains": "dental", "start_time_at_or_after": "09:00"}]}
    first = grade_task(w, expected)
    second = grade_task(WorldState(w.path), expected)  # re-open, re-grade
    assert first == second
    assert first[0] is True


def test_wrong_field_is_reported_by_key(tmp_path):
    w = _world(tmp_path)
    w.create_event(title="Dental", start="2026-07-15T08:00:00", end="2026-07-15T08:30:00")
    passed, detail = grade_task(
        w, {"events_present": [{"title_contains": "dental", "start_time_at_or_after": "09:00"}]}
    )
    assert not passed
    assert "dental" in detail.lower() or "09:00" in detail


def test_partial_completion_is_not_silently_passed(tmp_path):
    # Task wants BOTH: an event created AND a notification email sent.
    expected = {
        "events_present": [{"title_contains": "meeting"}],
        "emails_present": [{"to_contains": "sam@home.test"}],
    }
    w = _world(tmp_path)
    w.create_event(title="Team meeting", start="2026-07-15T10:00:00", end="2026-07-15T11:00:00")
    # ... but the spouse was never emailed.
    passed, detail = grade_task(w, expected)
    assert not passed
    assert "sam@home.test" in detail  # the missing half is named


def test_grader_reads_world_not_agent_claims(tmp_path):
    # The grader has no transcript input: a world that lacks the event fails even
    # if the agent "claimed" success, and a world that has it passes regardless.
    expected = {"events_present": [{"title_contains": "dentist"}]}

    lying = _world(tmp_path, "lying.db")  # agent said "booked!" but did nothing
    assert grade_task(lying, expected)[0] is False

    honest = _world(tmp_path, "honest.db")
    honest.create_event(title="Dentist", start="2026-07-15T10:00:00", end="2026-07-15T10:30:00")
    assert grade_task(honest, expected)[0] is True


def test_check_world_pure_function_is_deterministic():
    snap = {"events": [{"title": "x", "start": "2026-07-15T10:00:00", "end": "2026-07-15T10:30:00"}]}
    expected = {"events_present": [{"title_contains": "x"}]}
    assert check_world(snap, expected) == check_world(snap, expected) == []
