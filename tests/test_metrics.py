"""Tests for metrics rollup, pass^k, and cost normalization (U9)."""

from __future__ import annotations

import pytest

from simulator.config import (
    LOCAL_OLLAMA,
    CandidateModel,
    Hosting,
    HostingProfile,
    RunConfig,
)
from simulator.harness import SessionRow
from simulator.metrics import (
    TrackEvaluation,
    compute_pass_k,
    impute_cost,
    normalize_cost,
    rollup,
)


def _local(model_id="qwen3.6:latest", ctx=65_536):
    return CandidateModel(id=model_id, hosting_profile=LOCAL_OLLAMA, context_length=ctx)


_API_PROFILE = HostingProfile("Together", Hosting.API, "together", "https://api.together.xyz/v1")


def _api(model_id="big-open-model", price_in=0.6, price_out=0.9):
    return CandidateModel(id=model_id, hosting_profile=_API_PROFILE, context_length=128_000,
                          price_per_1m_input=price_in, price_per_1m_output=price_out)


def _session(in_tok=1000, out_tok=100, reasoning=0, est=0.0, act=0.0,
             started="100.0", ended="110.0"):
    return SessionRow("m", in_tok, out_tok, 0, 0, reasoning, 1, 0, est, act, started, ended)


# --- pass^k ------------------------------------------------------------------


def test_pass_k_uniform_reduces_to_p_to_the_k():
    # Every task has the same success rate p=0.5 over its seeds -> E[p^k] = 0.5^k.
    success = {"t1": [True, False], "t2": [False, True], "t3": [True, False]}
    assert compute_pass_k(success, k=3) == pytest.approx(0.5 ** 3)


def test_pass_k_all_pass_is_one_all_fail_is_zero():
    assert compute_pass_k({"t": [True, True, True]}, k=5) == 1.0
    assert compute_pass_k({"t": [False, False]}, k=5) == 0.0


def test_pass_k_averages_across_tasks():
    # t1: p=1 -> 1^2=1 ; t2: p=0.5 -> 0.25 ; mean = 0.625
    success = {"t1": [True, True], "t2": [True, False]}
    assert compute_pass_k(success, k=2) == pytest.approx((1.0 + 0.25) / 2)


def test_pass_k_empty_is_zero():
    assert compute_pass_k({}, k=5) == 0.0


# --- cost normalization ------------------------------------------------------


def test_local_zero_dollar_run_yields_imputed_nonzero_cost():
    cfg = RunConfig(candidates=(_local(),), local_price_per_1m_input=0.2,
                    local_price_per_1m_output=0.2)
    sessions = [_session(in_tok=12000, out_tok=200, est=0.0, act=0.0)]
    cost = normalize_cost(sessions, _local(), cfg)
    assert cost > 0  # local metered $0 -> imputed from the price assumption
    assert cost == pytest.approx((12000 * 0.2 + 200 * 0.2) / 1e6)


def test_api_run_uses_metered_state_db_cost():
    cfg = RunConfig(candidates=(_api(),))
    sessions = [_session(in_tok=12000, out_tok=200, est=0.0, act=0.0123)]
    assert normalize_cost(sessions, _api(), cfg) == pytest.approx(0.0123)


def test_api_run_without_metered_cost_falls_back_to_model_price():
    cfg = RunConfig(candidates=(_api(),))
    sessions = [_session(in_tok=1_000_000, out_tok=1_000_000, est=0.0, act=0.0)]
    # 1M input @ $0.6 + 1M output @ $0.9 = $1.50
    assert normalize_cost(sessions, _api(price_in=0.6, price_out=0.9), cfg) == pytest.approx(1.5)


def test_impute_includes_cache_and_reasoning():
    s = SessionRow("m", 100, 50, cache_read_tokens=200, cache_write_tokens=300,
                   reasoning_tokens=25, api_call_count=1, tool_call_count=0,
                   estimated_cost_usd=0, actual_cost_usd=0, started_at="0", ended_at="1")
    # input_side = 100+200+300=600 ; output_side = 50+25=75
    assert impute_cost([s], 1.0, 1.0) == pytest.approx((600 + 75) / 1e6)


# --- rollup ------------------------------------------------------------------


def _eval(model_id, persona, seed, *, completed=True, cap=0.8, mem=0.7, cost=0.01):
    return TrackEvaluation(model_id, persona, seed, completed=completed,
                           capability=cap, memory=mem, tokens=10000, cost_usd=cost,
                           latency_s=5.0)


def test_rollup_aggregates_per_model_and_computes_reliability():
    cfg = RunConfig(candidates=(_local("good"),), seeds=(0, 1), k=2)
    evals = [
        _eval("good", "dana", 0, cap=0.9),
        _eval("good", "dana", 1, cap=0.9),
    ]
    [r] = rollup(evals, cfg)
    assert r.model_id == "good" and not r.eliminated
    assert r.capability == pytest.approx(0.9)
    assert r.reliability == pytest.approx(1.0)  # both seeds succeed -> p=1, p^2=1
    assert r.n_tracks == 2


def test_rollup_reliability_drops_when_a_seed_underperforms():
    cfg = RunConfig(candidates=(_local("mid"),), seeds=(0, 1), k=2)
    evals = [
        _eval("mid", "dana", 0, cap=0.9),  # success
        _eval("mid", "dana", 1, cap=0.2),  # below threshold -> failure
    ]
    [r] = rollup(evals, cfg, success_threshold=0.5)
    assert r.reliability == pytest.approx(0.5 ** 2)  # p=0.5 over the two seeds


def test_rollup_surfaces_eliminated_models_with_reason():
    cfg = RunConfig(candidates=(_local("good"), _local("toosmall", ctx=40_960)),
                    seeds=(0,), k=1)
    evals = [_eval("good", "dana", 0)]
    rolls = {r.model_id: r for r in rollup(
        evals, cfg, eliminated={"toosmall": "context window 40960 below floor"})}
    assert not rolls["good"].eliminated
    assert rolls["toosmall"].eliminated
    assert "below floor" in rolls["toosmall"].reason


def test_track_latency_seconds_sums_positive_day_elapsed():
    from simulator.metrics import track_latency_seconds
    # Sums per-day wall-clock; non-positive (skipped/failed day) entries are ignored.
    assert track_latency_seconds([3.0, 4.5, 2.5]) == pytest.approx(10.0)
    assert track_latency_seconds([5.0, 0.0, -1.0]) == pytest.approx(5.0)
    assert track_latency_seconds([]) == 0.0
