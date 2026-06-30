"""Tests for the forgetting-aware memory exam (U8)."""

from __future__ import annotations

from simulator.grading.memory_exam import (
    grade_memory_exam,
    score_probe,
)
from simulator.scenarios.personas.dana import PERSONA as DANA
from simulator.scenarios.personas.schema import (
    PROBE_ABSTENTION,
    PROBE_RECALL,
    PROBE_UPDATE,
)


def _probe(persona, kind):
    return next(p for p in persona.answer_key["memory_probes"] if p["kind"] == kind)


# --- recall ------------------------------------------------------------------


def test_recall_correct_only_when_keywords_match():
    probe = _probe(DANA, PROBE_RECALL)  # "no appointments before 9am" -> ["9","am"]
    assert score_probe("You never book anything before 9am.", probe).correct
    assert not score_probe("You prefer mornings free.", probe).correct


# --- knowledge update (the heart of forgetting-aware scoring) -----------------


def test_update_post_change_answer_is_correct():
    probe = _probe(DANA, PROBE_UPDATE)  # expected wednesday, stale thursday
    r = score_probe("Mia's soccer is on Wednesday now.", probe)
    assert r.correct and r.label == "correct"


def test_update_stale_answer_is_wrong_not_merely_missing():
    probe = _probe(DANA, PROBE_UPDATE)
    r = score_probe("Mia has soccer on Thursday at 4pm.", probe)
    assert not r.correct
    assert r.label == "stale"  # remembered the OLD fact -> worse than missing


def test_update_mentioning_history_alongside_new_still_correct():
    probe = _probe(DANA, PROBE_UPDATE)
    r = score_probe("It moved from Thursday to Wednesday.", probe)
    assert r.correct  # new fact present; old mentioned only as history


def test_update_missing_when_neither_present():
    probe = _probe(DANA, PROBE_UPDATE)
    r = score_probe("I'm not sure about her schedule.", probe)
    assert not r.correct and r.label == "missing"


# --- abstention --------------------------------------------------------------


def test_abstention_correct_when_declines():
    probe = _probe(DANA, PROBE_ABSTENTION)
    r = score_probe("There's no dentist appointment scheduled for Theo.", probe)
    assert r.correct and r.label == "abstained"


def test_abstention_wrong_when_fabricated():
    probe = _probe(DANA, PROBE_ABSTENTION)
    r = score_probe("Theo's dentist appointment is on July 22 at 3pm.", probe)
    assert not r.correct and r.label == "fabricated"


def test_abstention_correct_when_declines_naming_the_topic():
    # Regression (observed with Owl Alpha): a refusal that restates the subject
    # ("I have no information about Theo's dentist appointment") is still a decline,
    # not a fabrication — it asserts no concrete event details.
    probe = _probe(DANA, PROBE_ABSTENTION)
    r = score_probe(
        "I have no information about a Theo or their dentist appointment in any "
        "of my context.",
        probe,
    )
    assert r.correct and r.label == "abstained"


def test_abstention_not_fooled_by_leading_no_then_fabrication():
    # Regression: "No, ..." is not a decline; a confident fabrication must fail.
    probe = _probe(DANA, PROBE_ABSTENTION)
    r = score_probe("No, your dentist appointment is Friday at 3pm.", probe)
    assert not r.correct


def test_abstention_wrong_when_declines_then_fabricates_specifics():
    # Regression: hedging then inventing a concrete time is still a fabrication.
    probe = _probe(DANA, PROBE_ABSTENTION)
    r = score_probe("I don't have the exact date, but it's Thursday at 3pm.", probe)
    assert not r.correct  # a specific time/weekday means it fabricated


def test_recall_keyword_not_matched_by_incidental_substrings():
    # Regression: the old ['9','am'] keys matched "I am"/"9 emails"; "before 9" doesn't.
    probe = _probe(DANA, PROBE_RECALL)
    assert not score_probe("I am not sure, but you have 9 unread emails.", probe).correct
    assert score_probe("You never book anything before 9am.", probe).correct


# --- whole-exam report -------------------------------------------------------


def test_grade_memory_exam_aggregates_and_breaks_down_by_kind():
    answers = {
        "recall_no_early_meetings": "No appointments before 9am.",
        "update_soccer_day": "Soccer is on Wednesday now.",
        "update_swim_day": "Theo's swim class is on Monday now.",
        "abstain_theo_dentist": "I don't see any dentist appointment for Theo.",
    }
    report = grade_memory_exam(DANA, answers)
    assert report.score == 1.0
    by_kind = report.by_kind()
    assert by_kind[PROBE_UPDATE] == 1.0  # both soccer and swim updates correct
    assert by_kind[PROBE_ABSTENTION] == 1.0


def test_missing_answer_scores_zero_for_that_probe():
    answers = {"update_soccer_day": "Wednesday"}  # others unanswered
    report = grade_memory_exam(DANA, answers)
    assert 0.0 < report.score < 1.0  # only one of four correct
    abstain = next(r for r in report.results if r.kind == PROBE_ABSTENTION)
    assert not abstain.correct  # empty answer doesn't decline -> fabricated/missing
