"""The full run: compose every unit into one ranked report.

This is the top-level glue the units were built for. It drives the two-stage
funnel (U3), then for each completed Stage-2 track it administers the memory exam
on the track's still-warm home (U8), optionally scores qualitative behavior with
the judge (U8), evaluates the track (U7/U9), rolls up per model (U9), and renders
the comparison report (U9).

``run_full`` is deliberately thin — all the real work lives in the units it calls.
The harness factory is injectable so the whole pipeline can run on the fake binary
in tests; the default uses the real ``hermes``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from .config import RunConfig
from .grading.judge import Judge
from .grading.memory_exam import administer_exam
from .harness import Harness
from .metrics import (
    TrackEvaluation,
    evaluate_track,
    latency_seconds,
    normalize_cost,
    rollup,
    tokens_to_complete,
)
from .report import Report, build_report, render_table
from .runner import HarnessFactory, Runner, _default_harness_factory
from .scenarios.types import Counterparty, Persona, Stage1Task


def run_full(
    run_config: RunConfig,
    personas: list[Persona],
    *,
    stage1_tasks: Optional[list[Stage1Task]] = None,
    results_root: str | Path = "results",
    harness_factory: HarnessFactory = _default_harness_factory,
    python_exe: Optional[str] = None,
    counterparty: Optional[Counterparty] = None,
    judge: Optional[Judge] = None,
    run_id: Optional[str] = None,
    success_threshold: float = 0.5,
    stage1_attempts: int = 1,
    stage1_pass_threshold: float = 0.6,
) -> tuple[Report, str]:
    """Run the whole simulator and return ``(report, rendered_table)``.

    Also writes ``report.txt`` next to the run's trajectories.
    """
    python_exe = python_exe or sys.executable
    runner = Runner(
        run_config,
        results_root=results_root,
        harness_factory=harness_factory,
        python_exe=python_exe,
        counterparty=counterparty,
        stage1_attempts=stage1_attempts,
        stage1_pass_threshold=stage1_pass_threshold,
    )
    matrix = runner.run_matrix(personas, stage1_tasks or [], run_id=run_id)

    persona_by_name = {p.name: p for p in personas}
    model_by_id = {c.id: c for c in run_config.candidates}

    evaluations = []
    for track in matrix.tracks:
        persona = persona_by_name[track.persona]
        model = model_by_id[track.model_id]
        completed = track.status == "completed"

        # Per-track isolation: a failing exam/judge/evaluation for one track must
        # not discard every other track's expensive Stage-2 work (mirrors the
        # runner's own per-day guard). On failure, record a degraded evaluation.
        try:
            memory_answers = None
            if completed:
                # The track's home still holds accrued memory + registered world.
                exam_harness = harness_factory(Path(track.trajectory_dir) / "home", model)
                memory_answers = administer_exam(exam_harness, persona)
                # Persist so the run is fully reportable from disk even if a later
                # track is interrupted (e.g. the machine sleeps mid-run).
                (Path(track.trajectory_dir) / "memory_exam.json").write_text(
                    json.dumps(memory_answers, indent=2)
                )

            judge_mean = None
            if judge is not None and track.days:
                transcript = "\n\n".join(d.stdout for d in track.days)
                verdict = judge.score(transcript, candidate_family=model.family_name)
                judge_mean = verdict.mean / 5.0  # 1..5 -> 0..1

            evaluations.append(evaluate_track(
                persona, model, track_dir=track.trajectory_dir, sessions=track.sessions,
                seed=track.seed, completed=completed, run_config=run_config,
                memory_answers=memory_answers, judge_mean_0_1=judge_mean,
            ))
        except Exception:
            # Degraded: keep the track in the rollup as an incomplete failure
            # rather than crashing the whole report.
            evaluations.append(TrackEvaluation(
                model_id=model.id, persona=persona.name, seed=track.seed,
                completed=False, capability=0.0, memory=0.0,
                tokens=tokens_to_complete(track.sessions),
                cost_usd=normalize_cost(track.sessions, model, run_config),
                latency_s=latency_seconds(track.sessions),
            ))

    eliminated = {o.model_id: o.reason for o in matrix.stage1 if not o.survived}
    rollups = rollup(evaluations, run_config, eliminated=eliminated,
                     success_threshold=success_threshold)
    report = build_report(rollups, run_config.weights)
    rendered = render_table(report)

    (Path(matrix.results_dir) / "report.txt").write_text(rendered)
    return report, rendered
