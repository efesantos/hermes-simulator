"""Tests for behavioral-improvement grading (U7), including AE3."""

from __future__ import annotations

import json
from pathlib import Path

from simulator.grading.behavioral import (
    grade_behavioral,
    grade_track_dir,
    load_track,
    score_no_event_before,
    score_not_after_day,
)
from simulator.scenarios.personas.dana import PERSONA as DANA


# --- individual signal scorers -----------------------------------------------


def test_no_event_before_passes_clean_calendar():
    snap = {"events": [{"title": "Soccer", "start": "2026-07-09T16:00:00", "end": "2026-07-09T17:00:00"}]}
    passed, _ = score_no_event_before(snap, "09:00")
    assert passed


def test_no_event_before_fails_on_early_booking():
    snap = {"events": [{"title": "Sync", "start": "2026-07-09T08:00:00", "end": "2026-07-09T08:30:00"}]}
    passed, detail = score_no_event_before(snap, "09:00")
    assert not passed and "Sync" in detail


def test_not_after_day_passes_when_keyword_dropped():
    texts = {1: "got it", 2: "soccer moved to Wednesday", 3: "pickup is Wednesday", 4: "all set"}
    passed, _ = score_not_after_day(texts, learned_on_day=2, forbidden_keyword="thursday")
    assert passed


def test_not_after_day_fails_when_keyword_repeated():
    texts = {2: "soccer moved to Wednesday", 3: "see you Thursday for soccer"}
    passed, detail = score_not_after_day(texts, learned_on_day=2, forbidden_keyword="thursday")
    assert not passed and "3" in detail


def test_not_after_day_exempts_days_before_correction():
    # Mentioning Thursday on day 1 (before the change) is fine.
    texts = {1: "Mia has soccer Thursday at 4pm"}
    assert score_not_after_day(texts, learned_on_day=2, forbidden_keyword="thursday")[0]


# --- AE3: improver scores higher than repeater -------------------------------


def test_ae3_adopter_outscores_repeater():
    clean_world = {"events": [{"title": "Soccer", "start": "2026-07-15T16:00:00", "end": "2026-07-15T17:00:00"}]}

    adopter_texts = {2: "noted, soccer is Wednesday now", 3: "pickup Wednesday", 5: "Wednesday practice"}
    repeater_texts = {2: "noted", 3: "soccer is Thursday", 5: "Thursday practice as usual"}

    adopter = grade_behavioral(DANA, clean_world, adopter_texts)
    repeater = grade_behavioral(DANA, clean_world, repeater_texts)
    assert adopter.score > repeater.score
    # The specific signal flips.
    adopt_signal = next(r for r in adopter.results if r.signal_id == "adopts_soccer_change")
    repeat_signal = next(r for r in repeater.results if r.signal_id == "adopts_soccer_change")
    assert adopt_signal.passed and not repeat_signal.passed


def test_early_meeting_violation_lowers_score():
    early_world = {"events": [{"title": "Sync", "start": "2026-07-09T08:00:00", "end": "2026-07-09T08:30:00"}]}
    texts = {3: "Wednesday"}  # adopts the change, but booked an early meeting
    report = grade_behavioral(DANA, early_world, texts)
    pref = next(r for r in report.results if r.signal_id == "respects_no_early_meetings")
    assert not pref.passed
    assert report.score < 1.0


# --- loading persisted trajectories ------------------------------------------


def test_load_track_attributes_sent_emails_to_their_day(tmp_path: Path):
    track = tmp_path / "track"
    track.mkdir()
    (track / "day_1.json").write_text(json.dumps(
        {"day": 1, "date": "2026-07-06", "stdout": "acknowledged"}))
    (track / "day_3.json").write_text(json.dumps(
        {"day": 3, "date": "2026-07-08", "stdout": "replied to Sam"}))
    (track / "final_world.json").write_text(json.dumps({
        "events": [],
        "emails": [
            {"folder": "sent", "to_addr": "sam@home.test", "subject": "Soccer",
             "body": "It's on Thursday", "timestamp": "2026-07-08T09:00:00"},
        ],
    }))
    snapshot, texts = load_track(track)
    assert "Thursday" in texts[3]  # the sent email folds into day 3's text
    # And the convenience path grades it: the stale keyword on day 3 fails the signal.
    report = grade_track_dir(DANA, track)
    soccer = next(r for r in report.results if r.signal_id == "adopts_soccer_change")
    assert not soccer.passed
