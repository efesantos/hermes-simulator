"""Rebuild a comparison report from a run's on-disk artifacts.

Usage: python scripts/build_report.py <run_id> [results_root]

Reads ``results/<run_id>/stage1/*/outcome.json`` and the per-track
``stage2/.../track.json`` + ``memory_exam.json`` written during a run, and
renders the ranked report. Lets a run that was interrupted (e.g. the machine
slept mid-run) still be reported from whatever completed — no re-running.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from simulator.config import (
    API_CANDIDATES,
    DEFAULT_CANDIDATES,
    CandidateModel,
    LOCAL_OLLAMA,
    RunConfig,
)
from simulator.harness import SessionRow
from simulator.metrics import (
    TrackEvaluation,
    evaluate_track,
    latency_seconds,
    normalize_cost,
    rollup,
    tokens_to_complete,
)
from simulator.report import build_report, render_table
from simulator.scenarios.personas import ALL_PERSONAS


def _session_from_dict(d: dict) -> SessionRow:
    fields = ("model", "input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens", "reasoning_tokens", "api_call_count",
              "tool_call_count", "estimated_cost_usd", "actual_cost_usd",
              "started_at", "ended_at")
    return SessionRow(*(d.get(f, 0 if "tokens" in f or "count" in f or "cost" in f else "")
                        for f in fields))


def _model_for(model_id: str) -> CandidateModel:
    # Search both fields so API candidates keep their real provider + prices when
    # a report is rebuilt from disk (else an API model is mis-costed as local $0).
    for c in (*DEFAULT_CANDIDATES, *API_CANDIDATES):
        if c.id == model_id:
            return c
    # Unknown id (e.g. an ad-hoc variant): synthesize a local 64K candidate.
    return CandidateModel(id=model_id, hosting_profile=LOCAL_OLLAMA, context_length=65_536)


def main() -> None:
    run_id = sys.argv[1]
    results_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    run_dir = results_root / run_id
    if not run_dir.exists():
        raise SystemExit(f"no run at {run_dir}")

    # Recovery report: reliability is computed at k=1 (mean pass) so it never
    # fails regardless of how many seeds completed on disk. The authoritative
    # reliability (pass^k) comes from a single live run; this rebuilds whatever
    # exists. seeds is generous purely to satisfy the len(seeds) >= k invariant.
    cfg = RunConfig(candidates=DEFAULT_CANDIDATES, seeds=(0, 1, 2, 3, 4), k=1)

    # Stage-1 outcomes -> eliminated map.
    eliminated: dict[str, str] = {}
    survived: set[str] = set()
    for f in sorted(run_dir.glob("stage1/*/outcome.json")):
        o = json.loads(f.read_text())
        (survived.add(o["model_id"]) if o.get("survived") else
         eliminated.__setitem__(o["model_id"], o.get("reason", "eliminated")))

    # Stage-2 tracks -> evaluations.
    evaluations = []
    for tf in sorted(run_dir.glob("stage2/*/*/seed*/track.json")):
        track = json.loads(tf.read_text())
        track_dir = tf.parent
        persona = ALL_PERSONAS.get(track["persona"])
        if persona is None:
            continue
        sessions = [_session_from_dict(d["session"]) for d in track.get("days", [])
                    if d.get("session")]
        answers_path = track_dir / "memory_exam.json"
        answers = json.loads(answers_path.read_text()) if answers_path.exists() else None
        judge_path = track_dir / "judge.json"
        judge_mean = (json.loads(judge_path.read_text()).get("judge_mean_0_1")
                      if judge_path.exists() else None)
        model = _model_for(track["model_id"])
        # Per-track isolation (mirrors the live pipeline): a single track that
        # fails grading — e.g. a model wrote a non-ISO event time — folds in as a
        # degraded evaluation instead of crashing the whole rebuild.
        try:
            evaluations.append(evaluate_track(
                persona, model, track_dir=str(track_dir),
                sessions=sessions, seed=track["seed"],
                completed=(track["status"] == "completed"), run_config=cfg,
                memory_answers=answers, judge_mean_0_1=judge_mean,
            ))
        except Exception as exc:  # noqa: BLE001 - report robustness over strictness
            print(f"warning: track {track_dir} failed grading ({exc}); "
                  "folding in as degraded", file=sys.stderr)
            evaluations.append(TrackEvaluation(
                model_id=model.id, persona=persona.name, seed=track["seed"],
                completed=False, capability=0.0, memory=0.0,
                tokens=tokens_to_complete(sessions),
                cost_usd=normalize_cost(sessions, model, cfg),
                latency_s=latency_seconds(sessions),
            ))

    rollups = rollup(evaluations, cfg, eliminated=eliminated)
    report = build_report(rollups, cfg.weights)
    print(f"=== report rebuilt from {run_dir} ===")
    print(f"(survivors with tracks: {len({e.model_id for e in evaluations})}, "
          f"eliminated: {len(eliminated)})\n")
    print(render_table(report))


if __name__ == "__main__":
    main()
