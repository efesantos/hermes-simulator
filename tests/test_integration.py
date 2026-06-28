"""End-to-end pipeline: runner -> evaluate -> rollup -> report (U9 verification).

Uses the fake hermes so the whole funnel runs without a live model: a small
matrix (one survivor, one below-floor model) produces trajectories, which are
scored into per-track evaluations, rolled up per model, and rendered as a ranked
report with all four dimensions plus the eliminated model and its reason.
"""

from __future__ import annotations

from pathlib import Path

from simulator.config import CompositeWeights, LOCAL_OLLAMA, CandidateModel, RunConfig
from simulator.harness import Harness
from simulator.metrics import evaluate_track, rollup
from simulator.report import build_report, render_table
from simulator.runner import Runner
from simulator.scenarios.personas.dana import PERSONA as DANA


def _model(model_id, ctx=65_536):
    return CandidateModel(id=model_id, hosting_profile=LOCAL_OLLAMA, context_length=ctx)


def test_full_matrix_produces_ranked_report(tmp_path, fake_hermes, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", "ok")
    good = _model("qwen3.6:latest")
    toosmall = _model("qwen3:8b", ctx=40_960)
    cfg = RunConfig(candidates=(good, toosmall), seeds=(0, 1), k=2)

    runner = Runner(
        cfg,
        harness_factory=lambda home, model: Harness(home, model, hermes_bin=fake_hermes, timeout=60),
        results_root=tmp_path,
    )
    matrix = runner.run_matrix([DANA], stage1_tasks=[], run_id="it")

    # Stage 1 kept the survivor, dropped the below-floor model with a reason.
    by_model = {o.model_id: o for o in matrix.stage1}
    assert by_model["qwen3.6:latest"].survived
    assert not by_model["qwen3:8b"].survived
    eliminated = {o.model_id: o.reason for o in matrix.stage1 if not o.survived}

    # Score each completed track out-of-band, then roll up.
    evaluations = [
        evaluate_track(
            DANA, good, track_dir=t.trajectory_dir, sessions=t.sessions,
            seed=t.seed, completed=(t.status == "completed"), run_config=cfg,
        )
        for t in matrix.tracks
    ]
    assert len(evaluations) == 2  # two seeds for the one survivor

    rollups = rollup(evaluations, cfg, eliminated=eliminated)
    report = build_report(rollups, CompositeWeights())

    # The survivor is ranked; the eliminated model is present with its reason.
    assert [r.model_id for r in report.ranked] == ["qwen3.6:latest"]
    assert report.ranked[0].composite is not None
    elim = report.eliminated
    assert len(elim) == 1 and elim[0].model_id == "qwen3:8b"
    assert "below floor" in elim[0].reason

    text = render_table(report)
    for col in ("Cap", "Mem", "Rel", "Cost$", "Composite"):
        assert col in text
    assert "Eliminated in Stage 1:" in text
