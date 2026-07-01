"""Tests for the candidate fields (U3), incl. the api-family expansion and the
build_report rebuild-costing regression guard (the P1 cost bug)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from simulator.config import (
    API_CANDIDATES,
    API_FAMILY_CANDIDATES,
    Hosting,
    run_config_for,
)

_EXPECTED_NEW = {
    "qwen/qwen3.7-plus",
    "qwen/qwen-plus",
    "openai/gpt-5.4-mini",
    "google/gemini-3.5-flash",
    "deepseek/deepseek-v3.2",
    "minimax/minimax-m2.5",
    "qwen/qwen3.5-flash",
}


def test_api_family_field_selectable_and_complete():
    cfg = run_config_for("api-family")
    ids = {c.id for c in cfg.candidates}
    # Seven new models plus the four continuity models from API_CANDIDATES.
    assert _EXPECTED_NEW <= ids
    assert {c.id for c in API_CANDIDATES} <= ids
    assert len(cfg.candidates) == len(_EXPECTED_NEW) + len(API_CANDIDATES)


def test_api_family_candidates_are_api_priced_and_eligible():
    for c in API_FAMILY_CANDIDATES:
        assert c.hosting == Hosting.API, c.id
        assert c.price_per_1m_input > 0 and c.price_per_1m_output > 0, c.id
        assert c.meets_context_floor, c.id


def test_api_family_families_are_explicit_and_correct():
    fam = {c.id: c.family_name for c in API_FAMILY_CANDIDATES}
    assert fam["deepseek/deepseek-v3.2"] == "deepseek"
    assert fam["minimax/minimax-m2.5"] == "minimax"
    assert fam["google/gemini-3.5-flash"] == "google"
    assert fam["openai/gpt-5.4-mini"] == "openai"


def test_glm_price_corrected():
    glm = next(c for c in API_CANDIDATES if c.id == "z-ai/glm-5.2")
    assert glm.price_per_1m_input == pytest.approx(0.93)  # was 0.95, drifted


def test_existing_api_field_unchanged():
    # Continuity guard: the prior 4-model api field keeps its exact membership.
    assert {c.id for c in API_CANDIDATES} == {
        "z-ai/glm-5.2",
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "mistralai/mistral-large-2512",
    }


def test_unknown_field_raises():
    with pytest.raises(ValueError):
        run_config_for("bogus")


# --- build_report rebuild-costing regression guard (P1) ----------------------


def _load_build_report():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_report.py"
    spec = importlib.util.spec_from_file_location("_build_report_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_model_for_resolves_api_family_to_api_candidate():
    """The P1 guard: an api-family id must NOT fall back to a synthesized local $0
    candidate on the rebuild path — that mis-costs cost-per-task, the deliverable."""
    br = _load_build_report()
    model = br._model_for("qwen/qwen3.7-plus")
    assert model.hosting == Hosting.API
    assert model.price_per_1m_input == pytest.approx(0.32)
    assert model.price_per_1m_output == pytest.approx(1.28)


def test_model_for_still_resolves_local_and_unknown():
    br = _load_build_report()
    # A genuinely unknown id still synthesizes a local candidate (unchanged behavior).
    unknown = br._model_for("totally/unknown-model")
    assert unknown.hosting == Hosting.LOCAL
