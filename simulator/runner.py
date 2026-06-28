"""The orchestrator: drive the (model x scenario/persona x seed) matrix through a
two-stage funnel, capturing trajectories and metrics.

Stage 1 (cheap, per model): hard eligibility gates — context floor and tool-call
*format* compatibility — then a single-shot pre-filter task set. Models that fail
any gate are dropped with a recorded reason (no silent caps).

Stage 2 (expensive, per surviving model x persona x seed): a day loop replays the
persona's fixed exogenous event stream into the world, invokes the harness once
per day in a persistent ``HERMES_HOME`` (so memory accrues across days), and runs
the counterparty between agent turns. Every track writes its trajectory and
per-day ``state.db`` metrics under ``results/``.

This module produces trajectories and metrics; it does **not** grade Stage 2 —
the memory exam (U8), behavioral checks (U7), and rollup (U9) read what it writes.
The only grading wired in here is the Stage-1 deterministic pre-filter, injected
as ``stage1_grader`` (U7 supplies the real one; tests inject a stub).
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import CandidateModel, RunConfig
from .grading.deterministic import grade_task
from .harness import ContextWindowError, Harness, HarnessResult, SessionRow
from .scenarios.types import Counterparty, DayPlan, Persona, Stage1Task
from .world.registration import register_world
from .world.state import WorldState

# A prompt that *requires* a tool call: a format-incompatible model (spike:
# gemma3:12b "no final response") fails this gate.
SMOKE_PROMPT = (
    "Use your calendar tools to list events for 2026-07-02, "
    "then reply with the single word DONE."
)

# Grades one finished Stage-1 task: (world after the run, expected_state) -> (passed, detail).
Stage1Grader = Callable[[WorldState, dict], "tuple[bool, str]"]
# Builds a Harness for a track; overridable in tests to point at a fake binary.
HarnessFactory = Callable[[Path, CandidateModel], Harness]


def _default_harness_factory(home: Path, model: CandidateModel) -> Harness:
    return Harness(home, model)


# The default Stage-1 grader is the deterministic state-diff engine (U7).
_default_stage1_grader = grade_task


# --- Stage-1 gate evaluation (pure; unit-tested directly) --------------------


@dataclass(frozen=True)
class FormatSmokeResult:
    passed: bool
    reason: str


def evaluate_format_smoke(
    result: HarnessResult, session: Optional[SessionRow]
) -> FormatSmokeResult:
    """Decide whether a model emits Hermes's tool-call format usably.

    Fails on: non-zero exit, an empty final response (the spike's "no final
    response"), or a completed run that called no tool at all when one was needed.
    """
    if not result.ok:
        return FormatSmokeResult(False, f"non-zero exit ({result.exit_code})")
    if not result.stdout.strip():
        return FormatSmokeResult(False, "no final response")
    if session is not None and session.tool_call_count == 0:
        return FormatSmokeResult(False, "did not call any tool")
    return FormatSmokeResult(True, "")


# --- result records ----------------------------------------------------------


@dataclass
class Stage1TaskResult:
    task_id: str
    passed: bool
    detail: str


@dataclass
class Stage1Outcome:
    """Per-model Stage-1 verdict, with the reason when dropped."""

    model_id: str
    eligible: bool  # passed context + format gates
    survived: bool  # eligible AND pre-filter pass-rate met threshold
    reason: str  # why dropped (empty when survived)
    task_results: list[Stage1TaskResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.task_results:
            return 1.0
        return sum(t.passed for t in self.task_results) / len(self.task_results)


@dataclass
class DayRecord:
    day: int
    date: str
    user_prompt: str
    exit_code: int
    stdout: str
    session: Optional[SessionRow]
    inbound_count: int
    counterparty_replies: int
    error: str = ""


@dataclass
class TrackResult:
    """One (model x persona x seed) Stage-2 run."""

    model_id: str
    persona: str
    seed: int
    status: str  # "completed" | "failed"
    trajectory_dir: str
    days: list[DayRecord] = field(default_factory=list)
    reason: str = ""

    @property
    def sessions(self) -> list[SessionRow]:
        return [d.session for d in self.days if d.session is not None]


@dataclass
class MatrixResult:
    run_id: str
    results_dir: str
    stage1: list[Stage1Outcome] = field(default_factory=list)
    tracks: list[TrackResult] = field(default_factory=list)


# --- JSON persistence --------------------------------------------------------


def _jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, default=str))


# --- the runner --------------------------------------------------------------


class Runner:
    """Drives the whole funnel and writes results under ``results_root/run_id``."""

    def __init__(
        self,
        run_config: RunConfig,
        *,
        results_root: str | Path = "results",
        harness_factory: HarnessFactory = _default_harness_factory,
        python_exe: str | None = None,
        stage1_grader: Stage1Grader = _default_stage1_grader,
        counterparty: Optional[Counterparty] = None,
        stage1_pass_threshold: float = 0.6,
    ) -> None:
        self.cfg = run_config
        self.results_root = Path(results_root)
        self.harness_factory = harness_factory
        self.python_exe = python_exe or sys.executable
        self.stage1_grader = stage1_grader
        self.counterparty = counterparty
        self.stage1_pass_threshold = stage1_pass_threshold

    # --- top level -----------------------------------------------------------

    def run_matrix(
        self,
        personas: list[Persona],
        stage1_tasks: list[Stage1Task] | None = None,
        *,
        run_id: str | None = None,
    ) -> MatrixResult:
        run_id = run_id or f"run-{int(time.time())}"
        run_dir = self.results_root / run_id
        result = MatrixResult(run_id=run_id, results_dir=str(run_dir))

        for model in self.cfg.candidates:
            outcome = self.run_stage1(model, stage1_tasks or [], run_dir)
            result.stage1.append(outcome)
            if not outcome.survived:
                continue
            seeds = self.cfg.seeds[: self.cfg.k]
            for persona in personas:
                for seed in seeds:
                    track = self.run_stage2_track(model, persona, seed, run_dir)
                    result.tracks.append(track)

        _write_json(run_dir / "matrix.json", result)
        return result

    # --- Stage 1 -------------------------------------------------------------

    def run_stage1(
        self, model: CandidateModel, tasks: list[Stage1Task], run_dir: Path
    ) -> Stage1Outcome:
        stage_dir = run_dir / "stage1" / _safe(model.id)

        # Gate 1: context floor (cheap pre-check before spending a run).
        if not model.meets_context_floor:
            outcome = Stage1Outcome(
                model.id, eligible=False, survived=False,
                reason=f"context window {model.context_length} below floor",
            )
            _write_json(stage_dir / "outcome.json", outcome)
            return outcome

        # Gate 2: tool-call format smoke (also catches a runtime context refusal).
        try:
            smoke = self._format_smoke(model, stage_dir / "smoke")
        except ContextWindowError as exc:
            outcome = Stage1Outcome(
                model.id, eligible=False, survived=False, reason=str(exc)
            )
            _write_json(stage_dir / "outcome.json", outcome)
            return outcome
        if not smoke.passed:
            outcome = Stage1Outcome(
                model.id, eligible=False, survived=False,
                reason=f"tool-call format: {smoke.reason}",
            )
            _write_json(stage_dir / "outcome.json", outcome)
            return outcome

        # Gate 3: single-shot pre-filter tasks.
        task_results = [self._run_stage1_task(model, t, stage_dir) for t in tasks]
        outcome = Stage1Outcome(
            model.id, eligible=True, survived=False, reason="", task_results=task_results
        )
        survived = outcome.pass_rate >= self.stage1_pass_threshold
        outcome.survived = survived
        if not survived:
            outcome.reason = (
                f"pre-filter pass rate {outcome.pass_rate:.0%} "
                f"< threshold {self.stage1_pass_threshold:.0%}"
            )
        _write_json(stage_dir / "outcome.json", outcome)
        return outcome

    def _format_smoke(self, model: CandidateModel, home_dir: Path) -> FormatSmokeResult:
        world_db = home_dir / "world.db"
        WorldState.create(world_db).close()  # empty world; listing is still a tool call
        harness = self._prepared_harness(model, home_dir / "home", world_db)
        result = harness.run_oneshot(SMOKE_PROMPT)
        return evaluate_format_smoke(result, harness.latest_session())

    def _run_stage1_task(
        self, model: CandidateModel, task: Stage1Task, stage_dir: Path
    ) -> Stage1TaskResult:
        task_dir = stage_dir / "tasks" / _safe(task.id)
        world_db = task_dir / "world.db"
        world = WorldState.create(world_db)
        world.seed(task.world_seed)
        world.close()

        harness = self._prepared_harness(model, task_dir / "home", world_db)
        try:
            result = harness.run_oneshot(task.prompt)
        except ContextWindowError as exc:  # shouldn't reach here post-gate, but be safe
            return Stage1TaskResult(task.id, False, f"context error: {exc}")

        graded_world = WorldState(world_db)
        try:
            passed, detail = self.stage1_grader(graded_world, task.expected_state)
        finally:
            graded_world.close()
        _write_json(
            task_dir / "result.json",
            {"task_id": task.id, "prompt": task.prompt, "stdout": result.stdout,
             "exit_code": result.exit_code, "passed": passed, "detail": detail},
        )
        return Stage1TaskResult(task.id, passed, detail)

    # --- Stage 2 -------------------------------------------------------------

    def run_stage2_track(
        self, model: CandidateModel, persona: Persona, seed: int, run_dir: Path
    ) -> TrackResult:
        track_dir = (
            run_dir / "stage2" / _safe(model.id) / _safe(persona.name) / f"seed{seed}"
        )
        world_db = track_dir / "world.db"
        world = WorldState.create(world_db)
        world.seed(persona.world_seed)
        world.close()

        harness = self._prepared_harness(model, track_dir / "home", world_db)
        track = TrackResult(
            model_id=model.id, persona=persona.name, seed=seed,
            status="completed", trajectory_dir=str(track_dir),
        )

        for day in persona.days:
            try:
                record = self._run_day(harness, world_db, day, persona, track_dir)
                track.days.append(record)
                if record.exit_code != 0:
                    track.status = "failed"
                    track.reason = f"day {day.day} exited {record.exit_code}"
                    break
            except Exception as exc:  # harness blew up — capture, don't abort matrix
                track.days.append(
                    DayRecord(day.day, day.date, day.user_prompt, exit_code=-1,
                              stdout="", session=None, inbound_count=len(day.inbound),
                              counterparty_replies=0, error=repr(exc))
                )
                track.status = "failed"
                track.reason = f"day {day.day} raised {type(exc).__name__}"
                break

        # Final out-of-band snapshot for downstream graders.
        snap_world = WorldState(world_db)
        try:
            _write_json(track_dir / "final_world.json", snap_world.inspect())
        finally:
            snap_world.close()
        _write_json(track_dir / "track.json", track)
        return track

    def _run_day(
        self,
        harness: Harness,
        world_db: Path,
        day: DayPlan,
        persona: Persona,
        track_dir: Path,
    ) -> DayRecord:
        world = WorldState(world_db)
        try:
            # 1. Apply the day's exogenous events (identical across all tracks).
            self._apply_inbound(world, day)
            # 2. Checkpoint outbound mail so we can find what the agent sends today.
            before_id = world.max_email_id()
        finally:
            world.close()

        # 3. Agent acts, with the day's simulated clock visible to the servers.
        result = harness.run_oneshot(
            day.user_prompt, extra_env={"HERMES_SIM_NOW": day.clock()}
        )
        session = harness.latest_session()

        # 4. Counterparty replies to anything the agent sent (partial observability).
        replies = self._counterparty_step(world_db, day, persona, before_id)

        record = DayRecord(
            day=day.day, date=day.date, user_prompt=day.user_prompt,
            exit_code=result.exit_code, stdout=result.stdout, session=session,
            inbound_count=len(day.inbound), counterparty_replies=replies,
        )
        _write_json(track_dir / f"day_{day.day}.json", record)
        return record

    @staticmethod
    def _apply_inbound(world: WorldState, day: DayPlan) -> None:
        for event in day.inbound:
            if event.kind == "email":
                world.add_email(**event.data)
            elif event.kind == "event":
                world.create_event(**event.data)
            elif event.kind == "contact":
                world.add_contact(**event.data)
            else:
                raise ValueError(f"unknown exogenous event kind: {event.kind!r}")

    def _counterparty_step(
        self, world_db: Path, day: DayPlan, persona: Persona, before_id: int
    ) -> int:
        if self.counterparty is None:
            return 0
        world = WorldState(world_db)
        try:
            outbound = world.emails_since(before_id, folder="sent")
            count = 0
            for email in outbound:
                reply = self.counterparty.reply(email, persona, sim_now=day.clock())
                if reply:
                    world.add_email(
                        from_addr=reply["from_addr"], to_addr=reply["to_addr"],
                        subject=reply["subject"], body=reply.get("body", ""),
                        timestamp=reply.get("timestamp", day.clock()), folder="inbox",
                    )
                    count += 1
            return count
        finally:
            world.close()

    # --- shared --------------------------------------------------------------

    def _prepared_harness(
        self, model: CandidateModel, home: Path, world_db: Path
    ) -> Harness:
        harness = self.harness_factory(home, model)
        harness.setup()
        register_world(harness, str(world_db), python_exe=self.python_exe)
        return harness


def _safe(name: str) -> str:
    """Filesystem-safe slug for ids that contain ':' etc. (e.g. 'qwen3.6:latest')."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
