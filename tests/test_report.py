"""Tests for the ranked comparison report (U9)."""

from __future__ import annotations

import pytest

from simulator.config import CompositeWeights
from simulator.metrics import ModelRollup
from simulator.report import build_report, render_table


def _roll(model_id, *, cap, mem, rel, cost, name=None):
    return ModelRollup(model_id=model_id, display_name=name or model_id, eliminated=False,
                       capability=cap, memory=mem, reliability=rel, cost_usd=cost,
                       tokens=10000, n_tracks=3)


def _eliminated(model_id, reason):
    return ModelRollup(model_id=model_id, display_name=model_id, eliminated=True, reason=reason)


# --- composite & ranking -----------------------------------------------------


def test_per_dimension_columns_present_regardless_of_weights():
    rolls = [_roll("a", cap=0.9, mem=0.8, rel=0.7, cost=0.02),
             _roll("b", cap=0.5, mem=0.5, rel=0.5, cost=0.01)]
    report = build_report(rolls, CompositeWeights(1, 0, 0, 0))  # capability-only weighting
    for row in report.ranked:
        # Every dimension is still reported even though only capability is weighted.
        assert row.capability is not None and row.memory is not None
        assert row.reliability is not None and row.cost_usd is not None
        assert row.composite is not None


def test_changing_weights_reorders_ranking():
    # 'cheap' is weak but cheapest; 'strong' is best on quality but priciest.
    strong = _roll("strong", cap=0.95, mem=0.9, rel=0.9, cost=0.10)
    cheap = _roll("cheap", cap=0.55, mem=0.5, rel=0.5, cost=0.001)

    quality_first = build_report([strong, cheap], CompositeWeights(0.6, 0.3, 0.1, 0.0))
    assert quality_first.ranked[0].model_id == "strong"

    cost_first = build_report([strong, cheap], CompositeWeights(0.0, 0.0, 0.0, 1.0))
    assert cost_first.ranked[0].model_id == "cheap"  # cheapest wins on a cost-only weighting


def test_cost_score_favors_cheaper_model():
    pricey = _roll("pricey", cap=0.7, mem=0.7, rel=0.7, cost=1.00)
    cheap = _roll("cheap", cap=0.7, mem=0.7, rel=0.7, cost=0.01)
    # Identical on quality; cost-only weights must rank the cheaper one first.
    report = build_report([pricey, cheap], CompositeWeights(0, 0, 0, 1))
    assert report.ranked[0].model_id == "cheap"
    assert report.ranked[0].composite > report.ranked[1].composite


# --- eliminated models -------------------------------------------------------


def test_eliminated_models_appear_with_reason_not_omitted():
    rolls = [_roll("good", cap=0.8, mem=0.8, rel=0.8, cost=0.02),
             _eliminated("toosmall", "context window 40960 below floor")]
    report = build_report(rolls, CompositeWeights())
    assert {r.model_id for r in report.rows} == {"good", "toosmall"}
    elim = report.eliminated
    assert len(elim) == 1
    assert elim[0].composite is None and elim[0].rank is None
    assert "below floor" in elim[0].reason


# --- rendering ---------------------------------------------------------------


def test_render_table_has_all_dimensions_and_lists_eliminated():
    rolls = [_roll("good", cap=0.8, mem=0.7, rel=0.9, cost=0.02, name="Good Model"),
             _eliminated("gemma3:12b", "tool-call format: no final response")]
    text = render_table(build_report(rolls, CompositeWeights()))
    for col in ("Cap", "Mem", "Rel", "Cost$", "Tokens", "Composite"):
        assert col in text
    assert "Good Model" in text
    assert "Eliminated in Stage 1:" in text
    assert "no final response" in text


def test_full_small_matrix_renders_ranked_report():
    # Verification: four dimensions + composite for every survivor, eliminated shown.
    rolls = [
        _roll("qwen3.6", cap=0.85, mem=0.7, rel=0.8, cost=0.015, name="Qwen3.6"),
        _roll("qwen3-32b", cap=0.9, mem=0.75, rel=0.85, cost=0.03, name="Qwen3 32B"),
        _eliminated("qwen3:8b", "context window 40960 below floor"),
        _eliminated("gemma3:12b", "tool-call format: no final response"),
    ]
    report = build_report(rolls, CompositeWeights())
    assert [r.rank for r in report.ranked] == [1, 2]
    assert len(report.eliminated) == 2
    text = render_table(report)
    assert "Composite" in text and "Eliminated in Stage 1:" in text
