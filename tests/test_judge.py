"""Tests for the LLM-as-judge bias mitigations (U8).

All tests inject a fake chat_fn; none hits a live frontier model.
"""

from __future__ import annotations

import json

import pytest

from simulator.grading.judge import (
    Judge,
    JudgeConfig,
    JudgeFamilyError,
    Verdict,
)

CFG = JudgeConfig(model="judge-x", family="anthropic", base_url="http://x/v1")


def _scoring_chat(scores: dict[str, int], rationale="ok"):
    def chat(messages, *, temperature):
        return json.dumps({"scores": scores, "rationale": rationale})
    return chat


# --- cross-family enforcement ------------------------------------------------


def test_same_family_judge_is_rejected():
    judge = Judge(CFG, chat_fn=_scoring_chat({"tone": 4, "proactivity": 4, "memory_surfacing": 4}))
    with pytest.raises(JudgeFamilyError):
        judge.score("transcript", candidate_family="anthropic")


def test_cross_family_judge_is_allowed():
    judge = Judge(CFG, chat_fn=_scoring_chat({"tone": 4, "proactivity": 3, "memory_surfacing": 5}))
    verdict = judge.score("transcript", candidate_family="qwen")
    assert isinstance(verdict, Verdict)
    assert verdict.scores["memory_surfacing"] == 5


# --- rubric scoring ----------------------------------------------------------


def test_score_returns_all_rubric_dimensions():
    judge = Judge(CFG, chat_fn=_scoring_chat({"tone": 5, "proactivity": 2, "memory_surfacing": 4}))
    v = judge.score("t", candidate_family="qwen")
    assert set(v.scores) == {"tone", "proactivity", "memory_surfacing"}
    assert v.mean == pytest.approx((5 + 2 + 4) / 3)


def test_rubric_is_included_in_the_prompt():
    captured = {}

    def chat(messages, *, temperature):
        captured["system"] = messages[0]["content"]
        return json.dumps({"scores": {"tone": 3, "proactivity": 3, "memory_surfacing": 3}})

    Judge(CFG, chat_fn=chat).score("t", candidate_family="qwen")
    assert "Rubric:" in captured["system"]
    assert "proactivity" in captured["system"]


def test_majority_median_across_judges():
    # Three judges disagree on tone; median is robust to the outlier.
    seq = [
        {"scores": {"tone": 5, "proactivity": 4, "memory_surfacing": 4}, "rationale": "a"},
        {"scores": {"tone": 4, "proactivity": 4, "memory_surfacing": 4}, "rationale": "b"},
        {"scores": {"tone": 1, "proactivity": 4, "memory_surfacing": 4}, "rationale": "c"},
    ]
    calls = {"n": 0}

    def chat(messages, *, temperature):
        out = json.dumps(seq[calls["n"]])
        calls["n"] += 1
        return out

    judge = Judge(CFG, chat_fn=chat, n_judges=3)
    v = judge.score("t", candidate_family="qwen")
    assert v.scores["tone"] == 4  # median(5,4,1)


def test_even_judge_count_rejected():
    with pytest.raises(ValueError):
        Judge(CFG, chat_fn=lambda *a, **k: "{}", n_judges=2)


# --- position-bias guard (compare) -------------------------------------------


def test_identical_responses_tie_regardless_of_order():
    judge = Judge(CFG, chat_fn=lambda *a, **k: json.dumps({"winner": "FIRST"}))
    assert judge.compare("same", "same", candidate_family="qwen") == "tie"


def test_compare_is_order_stable():
    # The judge always prefers whatever is shown FIRST (a maximally position-biased
    # judge). Our content-based ordering must still pick the SAME physical response
    # whichever way the arguments are passed — even though its a/b label flips.
    judge = Judge(CFG, chat_fn=lambda *a, **k: json.dumps({"winner": "FIRST"}))
    ab = judge.compare("alpha response", "beta response", candidate_family="qwen")
    ba = judge.compare("beta response", "alpha response", candidate_family="qwen")
    winner_when_ab = "alpha" if ab == "a" else "beta"
    winner_when_ba = "beta" if ba == "a" else "alpha"
    assert winner_when_ab == winner_when_ba  # same physical response wins both ways


def test_compare_majority_vote():
    votes = ["FIRST", "FIRST", "SECOND"]
    calls = {"n": 0}

    def chat(messages, *, temperature):
        out = json.dumps({"winner": votes[calls["n"]]})
        calls["n"] += 1
        return out

    judge = Judge(CFG, chat_fn=chat, n_judges=3)
    result = judge.compare("aaa", "bbb", candidate_family="qwen")
    assert result in {"a", "b"}  # majority FIRST -> whichever was shown first


def test_compare_also_enforces_cross_family():
    judge = Judge(CFG, chat_fn=lambda *a, **k: json.dumps({"winner": "TIE"}))
    with pytest.raises(JudgeFamilyError):
        judge.compare("x", "y", candidate_family="anthropic")


# --- persona-scoped multilingual rubric (KTD7) -------------------------------


def test_rubric_for_persona_maps_amsterdam_to_multilingual():
    from simulator.grading.judge import (
        DEFAULT_RUBRIC,
        MULTILINGUAL_RUBRIC,
        rubric_for_persona,
    )
    assert rubric_for_persona("amsterdam") is MULTILINGUAL_RUBRIC
    assert "multilingual" in rubric_for_persona("amsterdam")
    # dana (and any other persona) keeps the default rubric — comparability preserved.
    assert rubric_for_persona("dana") is DEFAULT_RUBRIC
    assert "multilingual" not in rubric_for_persona("dana")


def test_default_rubric_untouched_by_multilingual_addition():
    from simulator.grading.judge import DEFAULT_RUBRIC
    assert "multilingual" not in DEFAULT_RUBRIC  # dana's dimensions are unchanged


def test_score_rubric_override_adds_multilingual_dimension():
    from simulator.grading.judge import MULTILINGUAL_RUBRIC
    scores = {"tone": 4, "proactivity": 3, "memory_surfacing": 5, "multilingual": 2}
    judge = Judge(CFG, chat_fn=_scoring_chat(scores))
    v = judge.score("t", candidate_family="qwen", rubric=MULTILINGUAL_RUBRIC)
    assert set(v.scores) == set(MULTILINGUAL_RUBRIC)
    assert v.scores["multilingual"] == 2


def test_score_override_surfaces_multilingual_in_prompt():
    from simulator.grading.judge import MULTILINGUAL_RUBRIC
    captured = {}

    def chat(messages, *, temperature):
        captured["system"] = messages[0]["content"]
        return json.dumps({"scores": {d: 3 for d in MULTILINGUAL_RUBRIC}})

    judge = Judge(CFG, chat_fn=chat)
    judge.score("t", candidate_family="qwen", rubric=MULTILINGUAL_RUBRIC)
    assert "multilingual" in captured["system"]


def test_score_without_override_uses_default_rubric():
    # A judge built with the default rubric, scored without override, must NOT emit
    # a multilingual dimension (this is what a dana track sees).
    captured = {}

    def chat(messages, *, temperature):
        captured["system"] = messages[0]["content"]
        return json.dumps({"scores": {"tone": 3, "proactivity": 3, "memory_surfacing": 3}})

    judge = Judge(CFG, chat_fn=chat)
    v = judge.score("t", candidate_family="qwen")
    assert "multilingual" not in captured["system"]
    assert "multilingual" not in v.scores
