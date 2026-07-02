"""Tests for the full-run pipeline glue (run_full)."""

from __future__ import annotations

from pathlib import Path

from simulator.config import LOCAL_OLLAMA, CandidateModel, RunConfig
from simulator.harness import Harness
from simulator.pipeline import run_full
from simulator.scenarios.personas.dana import PERSONA as DANA


def _model(model_id, ctx=65_536):
    return CandidateModel(id=model_id, hosting_profile=LOCAL_OLLAMA, context_length=ctx)


def test_run_full_drives_funnel_to_a_written_report(tmp_path, fake_hermes, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", "ok")
    cfg = RunConfig(
        candidates=(_model("qwen3.6:latest"), _model("qwen3:8b", ctx=40_960)),
        seeds=(0, 1), k=2,
    )
    factory = lambda home, model: Harness(home, model, hermes_bin=fake_hermes, timeout=60)

    report, rendered = run_full(
        cfg, [DANA], stage1_tasks=[], results_root=tmp_path,
        harness_factory=factory, run_id="full",
    )

    # Survivor ranked, below-floor model eliminated with a reason.
    assert [r.model_id for r in report.ranked] == ["qwen3.6:latest"]
    assert report.ranked[0].composite is not None
    assert report.eliminated[0].model_id == "qwen3:8b"
    assert "below floor" in report.eliminated[0].reason

    # Report persisted next to the trajectories.
    assert (tmp_path / "full" / "report.txt").read_text() == rendered
    assert "Composite" in rendered


def test_run_full_survives_a_raising_judge(tmp_path, fake_hermes, monkeypatch):
    # Regression: an exception in the per-track grading (here, the judge) must not
    # abort the whole run — the track is folded in as a degraded evaluation.
    from simulator.grading.judge import Judge, JudgeConfig

    monkeypatch.setenv("FAKE_STDOUT", "ok")
    cfg = RunConfig(candidates=(_model("qwen3.6:latest"),), seeds=(0, 1), k=2)
    factory = lambda home, model: Harness(home, model, hermes_bin=fake_hermes, timeout=60)

    # A judge whose model returns unparseable output -> JudgeError inside the loop.
    bad_judge = Judge(
        JudgeConfig(model="j", family="anthropic", base_url="http://x/v1"),
        chat_fn=lambda *a, **k: "not json at all",
    )

    report, _ = run_full(
        cfg, [DANA], stage1_tasks=[], results_root=tmp_path,
        harness_factory=factory, judge=bad_judge, run_id="badjudge",
    )
    # The run still produced a report with the model present (degraded tracks folded in).
    assert report.ranked and report.ranked[0].model_id == "qwen3.6:latest"


def test_judge_score_persisted_and_rebuilt_from_disk(tmp_path, fake_hermes, monkeypatch):
    # The judge's capability contribution must survive a disk rebuild: the pipeline
    # writes judge.json per track, and scripts/build_report.py reads it back (so a
    # chunked multi-seed field can be stitched together without losing the judge).
    import json
    import subprocess
    import sys

    from simulator.grading.judge import Judge, JudgeConfig

    monkeypatch.setenv("FAKE_STDOUT", "ok")
    cfg = RunConfig(candidates=(_model("qwen3.6:latest"),), seeds=(0, 1), k=2)
    factory = lambda home, model: Harness(home, model, hermes_bin=fake_hermes, timeout=60)
    good_judge = Judge(  # cross-family vs the qwen candidate; returns a valid verdict
        JudgeConfig(model="j", family="anthropic", base_url="http://x/v1"),
        chat_fn=lambda *a, **k: '{"scores": {"tone": 4, "proactivity": 4, '
        '"memory_surfacing": 4}, "rationale": "ok"}',
    )

    run_full(cfg, [DANA], stage1_tasks=[], results_root=tmp_path,
             harness_factory=factory, judge=good_judge, run_id="jp")

    judge_files = list((tmp_path / "jp" / "stage2").glob("*/*/seed*/judge.json"))
    assert judge_files, "expected judge.json written per Stage-2 track"
    payload = json.loads(judge_files[0].read_text())
    assert 0.0 <= payload["judge_mean_0_1"] <= 1.0
    assert payload["scores"]  # rubric dimensions preserved for inspection

    def _rebuild():
        r = subprocess.run(
            [sys.executable, "scripts/build_report.py", "jp", str(tmp_path)],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        assert r.returncode == 0, r.stderr
        return r.stdout

    with_judge = _rebuild()
    assert "Qwen3.6" in with_judge and "Composite" in with_judge
    # Removing the persisted judge changes the rebuilt capability -> proves the
    # script actually reads judge.json (not that it merely runs).
    for jf in judge_files:
        jf.unlink()
    assert _rebuild() != with_judge


def test_family_name_inference():
    assert _model("qwen3.6:latest").family_name == "qwen"
    assert _model("gemma3:12b").family_name == "gemma"
    assert CandidateModel("llama-3.3-70b", LOCAL_OLLAMA, 128_000, family="meta").family_name == "meta"


def test_same_family_candidate_falls_back_to_deterministic_capability(tmp_path, fake_hermes, monkeypatch):
    # A candidate sharing the judge's family (e.g. an Anthropic model under the
    # Anthropic subscription judge) must NOT be zeroed by the cross-family guard —
    # it should be scored deterministic-only (judge_mean skipped), not degraded.
    import json
    from simulator.grading.judge import Judge, JudgeConfig

    monkeypatch.setenv("FAKE_STDOUT", "ok")
    # Candidate family "anthropic" == the judge's family below.
    anthropic_model = CandidateModel(
        id="anthropic/claude-sonnet-5", hosting_profile=LOCAL_OLLAMA,
        context_length=65_536, family="anthropic",
    )
    cfg = RunConfig(candidates=(anthropic_model,), seeds=(0,), k=1)
    factory = lambda home, model: Harness(home, model, hermes_bin=fake_hermes, timeout=60)
    judge = Judge(  # same family as the candidate -> score() would raise JudgeFamilyError
        JudgeConfig(model="j", family="anthropic", base_url="http://x/v1"),
        chat_fn=lambda *a, **k: json.dumps({"scores": {"tone": 5}, "rationale": "x"}),
    )

    report, _ = run_full(
        cfg, [DANA], stage1_tasks=[], results_root=tmp_path,
        harness_factory=factory, judge=judge, run_id="samefam",
    )

    # The model is present and NOT crashed into a 0-everything degraded row.
    assert report.ranked, "same-family candidate should still be ranked, not dropped"
    row = report.ranked[0]
    assert row.model_id == "anthropic/claude-sonnet-5"
    # judge.json records the skip reason; judge_mean is None (deterministic-only).
    jf = list((tmp_path / "samefam" / "stage2").glob("*/*/seed*/judge.json"))
    assert jf, "judge.json should still be written for the skipped-judge track"
    payload = json.loads(jf[0].read_text())
    assert payload["judge_mean_0_1"] is None
    assert "same-family" in payload.get("skipped", "")
