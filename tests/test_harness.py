"""Tests for config + harness wrapper (U1)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
import yaml

from simulator.config import (
    LOCAL_OLLAMA,
    CandidateModel,
    CompositeWeights,
    RunConfig,
    default_run_config,
)
from simulator.harness import ContextWindowError, Harness


def _model(model_id: str = "qwen3.6:latest", ctx: int = 65_536) -> CandidateModel:
    return CandidateModel(id=model_id, hosting_profile=LOCAL_OLLAMA, context_length=ctx)


# --- config ------------------------------------------------------------------


def test_meets_context_floor():
    assert _model(ctx=65_536).meets_context_floor is True
    assert _model(ctx=40_960).meets_context_floor is False


def test_candidate_hosting_passthrough():
    m = _model()
    assert m.provider == "local-ollama"
    assert m.base_url == "http://localhost:11434/v1"
    assert m.display_name == "qwen3.6:latest"  # falls back to id when label empty


def test_composite_weights_normalize_to_one():
    w = CompositeWeights(capability=2, memory=1, reliability=1, cost=1).normalized()
    assert pytest.approx(w.capability + w.memory + w.reliability + w.cost) == 1.0
    assert pytest.approx(w.capability) == 0.4  # 2 of total 5


def test_composite_weights_reject_nonpositive():
    with pytest.raises(ValueError):
        CompositeWeights(0, 0, 0, 0).normalized()


def test_runconfig_requires_at_least_k_seeds():
    with pytest.raises(ValueError):
        RunConfig(candidates=(_model(),), seeds=(0, 1), k=5)


def test_default_run_config_includes_ineligible_candidates():
    # Ineligible models stay in the field so Stage 1 can demonstrate dropping them.
    cfg = default_run_config()
    ids = {c.id for c in cfg.candidates}
    assert "qwen3:8b" in ids  # below floor
    assert "gemma3:12b" in ids  # format-incompatible


# --- harness setup -----------------------------------------------------------


def test_setup_writes_config_pinning_model(tmp_path: Path):
    h = Harness(tmp_path / "home", _model(ctx=65_536))
    h.setup()
    cfg = yaml.safe_load(h.config_path.read_text())
    assert cfg["model"]["default"] == "qwen3.6:latest"
    assert cfg["model"]["provider"] == "local-ollama"
    assert cfg["model"]["context_length"] == 65_536
    assert cfg["providers"]["local-ollama"]["api"] == "http://localhost:11434/v1"
    assert cfg["hooks_auto_accept"] is True


def test_two_homes_are_isolated(tmp_path: Path, fake_hermes: str):
    a = Harness(tmp_path / "homeA", _model(), hermes_bin=fake_hermes)
    b = Harness(tmp_path / "homeB", _model(), hermes_bin=fake_hermes)
    a.setup()
    b.setup()

    os.environ["FAKE_STDOUT"] = "ok"
    try:
        a.run_oneshot("remember: A-only fact")
    finally:
        del os.environ["FAKE_STDOUT"]

    # The fake hermes appended A's prompt to homeA's memory only.
    assert "A-only fact" in (a.memory_dir / "USER.md").read_text()
    assert not (b.memory_dir / "USER.md").exists()


def test_reset_memory_empties_only_this_home(tmp_path: Path, fake_hermes: str):
    h = Harness(tmp_path / "home", _model(), hermes_bin=fake_hermes)
    h.setup()
    os.environ["FAKE_STDOUT"] = "ok"
    try:
        h.run_oneshot("a planted fact")
    finally:
        del os.environ["FAKE_STDOUT"]
    assert (h.memory_dir / "USER.md").read_text().strip() != ""

    h.reset_memory()
    assert (h.memory_dir / "USER.md").read_text().strip() == ""


# --- harness running ---------------------------------------------------------


def test_run_oneshot_happy_path(tmp_path: Path, fake_hermes: str, monkeypatch):
    h = Harness(tmp_path / "home", _model(), hermes_bin=fake_hermes)
    h.setup()
    monkeypatch.setenv("FAKE_STDOUT", "SPIKE_OK")
    result = h.run_oneshot("Reply with exactly: SPIKE_OK")
    assert result.ok
    assert result.exit_code == 0
    assert result.stdout == "SPIKE_OK"


def test_run_oneshot_nonzero_exit_returns_result_not_raise(
    tmp_path: Path, fake_hermes: str, monkeypatch
):
    h = Harness(tmp_path / "home", _model(), hermes_bin=fake_hermes)
    h.setup()
    monkeypatch.setenv("FAKE_STDOUT", "boom")
    monkeypatch.setenv("FAKE_EXIT", "3")
    result = h.run_oneshot("do a thing")
    assert not result.ok
    assert result.exit_code == 3


def test_context_window_error_is_typed_and_catchable(
    tmp_path: Path, fake_hermes: str, monkeypatch
):
    h = Harness(tmp_path / "home", _model(ctx=40_960), hermes_bin=fake_hermes)
    h.setup()
    # Hermes emits this on a sub-floor model; the fake puts it on stderr.
    monkeypatch.setenv("FAKE_STDERR", "Error: context window below the minimum 64,000 required.")
    monkeypatch.setenv("FAKE_EXIT", "1")
    with pytest.raises(ContextWindowError):
        h.run_oneshot("anything")


def test_runtime_context_refusal_is_detected(tmp_path: Path, fake_hermes: str, monkeypatch):
    # Regression: Hermes refuses an under-context model on STDOUT with exit 0 using
    # a phrasing the original regex missed ("...only N tokens of runtime context,
    # but Hermes needs at least 64,000 tokens..."). It must still be classified as
    # a context-floor failure, not a format failure.
    h = Harness(tmp_path / "home", _model("qwen3:32b", 65_536), hermes_bin=fake_hermes)
    h.setup()
    monkeypatch.setenv(
        "FAKE_STDOUT",
        "Ollama loaded qwen3:32b with only 40,960 tokens of runtime context, "
        "but Hermes needs at least 64,000 tokens for reliability.",
    )
    with pytest.raises(ContextWindowError):
        h.run_oneshot("anything")  # exit 0, message on stdout


def test_successful_run_mentioning_phrase_does_not_raise(
    tmp_path: Path, fake_hermes: str, monkeypatch
):
    # Regression: a clean run (exit 0) whose reply merely mentions the phrase must
    # NOT be misclassified as a context-floor failure.
    h = Harness(tmp_path / "home", _model(), hermes_bin=fake_hermes)
    h.setup()
    monkeypatch.setenv(
        "FAKE_STDOUT", "Note: a context window below the minimum can truncate long chats."
    )
    result = h.run_oneshot("explain context windows")  # exit 0, phrase only on stdout
    assert result.ok


# --- state.db accounting -----------------------------------------------------


def _seed_state_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE sessions (
            model TEXT, input_tokens INT, output_tokens INT,
            cache_read_tokens INT, cache_write_tokens INT, reasoning_tokens INT,
            api_call_count INT, tool_call_count INT, billing_provider TEXT,
            estimated_cost_usd REAL, actual_cost_usd REAL, cost_status TEXT,
            pricing_version TEXT, started_at TEXT, ended_at TEXT
        )
        """
    )
    for r in rows:
        cols = ", ".join(r.keys())
        ph = ", ".join("?" for _ in r)
        conn.execute(f"INSERT INTO sessions ({cols}) VALUES ({ph})", list(r.values()))
    conn.commit()
    conn.close()


def test_warm_is_noop_for_non_local_model(tmp_path: Path):
    from simulator.config import Hosting, HostingProfile

    api = HostingProfile("API", Hosting.API, "together", "https://api.together.xyz/v1")
    h = Harness(tmp_path / "home", CandidateModel("m", api, 128_000))
    assert h.warm() is False  # API models aren't warmed; never touches Ollama


def test_runner_warm_defaults_off_with_custom_factory(tmp_path: Path):
    # Custom factory (tests/fakes) -> warming auto-disabled so we never hit Ollama.
    from simulator.config import RunConfig
    from simulator.runner import Runner, _default_harness_factory

    cfg = RunConfig(candidates=(_model(),), seeds=(0,), k=1)
    assert Runner(cfg, harness_factory=lambda h, m: None).warm_models is False
    assert Runner(cfg, harness_factory=_default_harness_factory).warm_models is True
    assert Runner(cfg, warm_models=True, harness_factory=lambda h, m: None).warm_models is True


def test_read_sessions_empty_when_no_db(tmp_path: Path):
    h = Harness(tmp_path / "home", _model())
    h.setup()
    assert h.read_sessions() == []
    assert h.latest_session() is None


def test_read_sessions_parses_tokens_and_cost(tmp_path: Path):
    h = Harness(tmp_path / "home", _model())
    h.setup()
    _seed_state_db(
        h.state_db_path,
        [
            {
                "model": "qwen3.6:latest",
                "input_tokens": 11873,
                "output_tokens": 24,
                "reasoning_tokens": 0,
                "tool_call_count": 1,
                "estimated_cost_usd": 0.0,
                "started_at": "2026-06-28T10:00:00",
                "ended_at": "2026-06-28T10:00:11",
            },
            {
                "model": "qwen3.6:latest",
                "input_tokens": 5000,
                "output_tokens": 100,
                "reasoning_tokens": 50,
                "estimated_cost_usd": 0.0,
                "started_at": "2026-06-28T10:05:00",
                "ended_at": "2026-06-28T10:05:30",
            },
        ],
    )
    sessions = h.read_sessions()
    assert len(sessions) == 2
    first = sessions[0]
    assert first.input_tokens == 11873
    assert first.output_tokens == 24
    assert first.total_tokens == 11897  # input + output + reasoning
    assert first.tool_call_count == 1

    latest = h.latest_session()
    assert latest is not None
    assert latest.total_tokens == 5150  # 5000 + 100 + 50


# --- live (real hermes + Ollama; run with `-m live`) -------------------------


@pytest.mark.live
def test_live_oneshot_end_to_end(tmp_path: Path):
    """U1 verification: a real model runs end-to-end and tokens read from state.db."""
    h = Harness(tmp_path / "home", _model("qwen3.6:latest", 65_536), timeout=300)
    h.setup()
    result = h.run_oneshot("Reply with exactly: SPIKE_OK")
    assert result.ok
    assert "SPIKE_OK" in result.stdout
    session = h.latest_session()
    assert session is not None
    assert session.total_tokens > 0  # the ~12K fixed-overhead input the spike found
