"""Tests for the ranked comparison report (U9)."""

from __future__ import annotations

import pytest

from simulator.config import CompositeWeights
from simulator.metrics import ModelRollup
from simulator.report import build_report, render_table


def _roll(model_id, *, cap, mem, rel, cost, latency=0.0, name=None):
    return ModelRollup(model_id=model_id, display_name=name or model_id, eliminated=False,
                       capability=cap, memory=mem, reliability=rel, cost_usd=cost,
                       latency_s=latency, tokens=10000, n_tracks=3)


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


# --- speed dimension (U4) ----------------------------------------------------


def test_normalized_sums_across_five_dimensions():
    w = CompositeWeights(0.30, 0.30, 0.15, 0.15, 0.10).normalized()
    assert w.capability + w.memory + w.reliability + w.cost + w.speed == pytest.approx(1.0)


def test_zero_sum_weights_still_raise():
    with pytest.raises(ValueError):
        CompositeWeights(0, 0, 0, 0, 0).normalized()


def test_default_weights_reproduce_prior_four_dimension_ranking():
    # Back-compat guard: the DEFAULT weighting (speed 0.0) must produce the same
    # composites/order as an explicit 4-dimension weighting — no ranking shift.
    rolls = [
        _roll("a", cap=0.9, mem=0.8, rel=0.7, cost=0.02, latency=12.0),
        _roll("b", cap=0.6, mem=0.6, rel=0.6, cost=0.01, latency=2.0),
        _roll("c", cap=0.8, mem=0.9, rel=0.8, cost=0.05, latency=8.0),
    ]
    default = build_report(rolls, CompositeWeights())  # speed defaults to 0.0
    explicit_4d = build_report(rolls, CompositeWeights(0.35, 0.35, 0.20, 0.10))
    assert [r.model_id for r in default.ranked] == [r.model_id for r in explicit_4d.ranked]
    for d, e in zip(default.ranked, explicit_4d.ranked):
        assert d.composite == pytest.approx(e.composite)


def test_speed_score_favors_faster_model():
    slow = _roll("slow", cap=0.7, mem=0.7, rel=0.7, cost=0.02, latency=16.0)
    fast = _roll("fast", cap=0.7, mem=0.7, rel=0.7, cost=0.02, latency=2.0)
    # Identical on everything but latency; a speed-only weighting ranks the faster first.
    report = build_report([slow, fast], CompositeWeights(0, 0, 0, 0, 1))
    assert report.ranked[0].model_id == "fast"
    assert report.ranked[0].composite > report.ranked[1].composite


def test_render_table_has_speed_column():
    rolls = [_roll("good", cap=0.8, mem=0.7, rel=0.9, cost=0.02, latency=5.5, name="Good Model")]
    text = render_table(build_report(rolls, CompositeWeights()))
    assert "Speed" in text
    assert "5.5" in text  # the latency value is rendered


# --- labelled picks + multi-weighting (U5) -----------------------------------

from simulator.report import (  # noqa: E402
    NAMED_WEIGHTINGS,
    Picks,
    pick_labels,
    render_picks,
    render_weightings,
)


def _rows(*rolls):
    return build_report(list(rolls), CompositeWeights()).ranked


def test_best_accuracy_ignores_floor():
    # High cap+mem but reliability below floor — still wins best-accuracy.
    hot = _roll("hot", cap=0.95, mem=0.95, rel=0.10, cost=0.05)
    safe = _roll("safe", cap=0.7, mem=0.7, rel=0.99, cost=0.05)
    picks = pick_labels(_rows(hot, safe))
    assert picks.best_accuracy.model_id == "hot"


def test_value_and_cheapest_exclude_floor_failers():
    below = _roll("below", cap=0.55, mem=0.4, rel=0.5, cost=0.001)  # fails floor, cheapest
    ok = _roll("ok", cap=0.8, mem=0.7, rel=0.9, cost=0.05)
    picks = pick_labels(_rows(below, ok))
    assert picks.n_floor_passers == 1
    assert picks.cheapest_viable.model_id == "ok"  # not 'below', despite it being cheaper
    assert picks.best_value.model_id == "ok"


def test_cheapest_viable_vs_best_value_differ():
    # cheap_ok: passes floor, cheapest. rich_ok: passes floor, best accuracy-per-dollar.
    cheap_ok = _roll("cheap_ok", cap=0.62, mem=0.52, rel=0.80, cost=0.010)
    rich_ok = _roll("rich_ok", cap=0.95, mem=0.95, rel=0.95, cost=0.012)
    picks = pick_labels(_rows(cheap_ok, rich_ok))
    assert picks.cheapest_viable.model_id == "cheap_ok"
    assert picks.best_value.model_id == "rich_ok"  # (0.95+0.95)/0.012 > (0.62+0.52)/0.010


def test_zero_cost_guard_in_best_value():
    free = _roll("free", cap=0.9, mem=0.9, rel=0.9, cost=0.0)  # $0 smoke-style track
    paid = _roll("paid", cap=0.7, mem=0.6, rel=0.8, cost=0.02)
    picks = pick_labels(_rows(free, paid))  # must not raise ZeroDivisionError
    # $0 row is excluded from the value ratio; the paid floor-passer wins best-value.
    assert picks.best_value.model_id == "paid"


def test_empty_floor_passers_reports_none():
    a = _roll("a", cap=0.3, mem=0.3, rel=0.3, cost=0.01)
    b = _roll("b", cap=0.4, mem=0.2, rel=0.2, cost=0.02)
    picks = pick_labels(_rows(a, b))
    assert picks.n_floor_passers == 0
    assert picks.best_value is None and picks.cheapest_viable is None
    assert picks.best_accuracy is not None  # accuracy pick ignores the floor
    assert "(none)" in render_picks(picks)


def test_named_weightings_reorder_top_rank():
    # strong+pricey+slow vs weak+cheap+fast: memory_heavy favors strong, cost_forward the cheap one.
    strong = _roll("strong", cap=0.95, mem=0.95, rel=0.95, cost=0.20, latency=16.0)
    thrifty = _roll("thrifty", cap=0.62, mem=0.55, rel=0.80, cost=0.005, latency=2.0)
    text = render_weightings([strong, thrifty])
    assert "weighting: memory_heavy" in text
    assert "weighting: cost_forward" in text
    mh = build_report([strong, thrifty], NAMED_WEIGHTINGS["memory_heavy"]).ranked[0].model_id
    cf = build_report([strong, thrifty], NAMED_WEIGHTINGS["cost_forward"]).ranked[0].model_id
    assert mh == "strong"
    assert cf == "thrifty"


def test_render_weightings_includes_picks_once():
    ok = _roll("ok", cap=0.8, mem=0.7, rel=0.9, cost=0.05)
    text = render_weightings([ok])
    assert text.count("Picks (no single winner") == 1
