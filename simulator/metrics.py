"""Metrics rollup: turn per-track results into per-model numbers.

Three jobs (KTD7, R11/R12):

- **cost** — always report tokens-to-complete; make dollars comparable. API
  models use the dollars Hermes metered into ``state.db``; local models (which
  meter ``$0``) get an imputed figure from a configurable price-per-1M assumption,
  including cache and reasoning tokens.
- **reliability** — ``pass^k = E_task[p_task^k]``: estimate each task's success
  probability from its repeated seeds, raise to the k-th power, average over
  tasks. Uniform ``p`` reduces to ``p^k``.
- **rollup** — aggregate a model's tracks into one row per dimension (capability,
  memory, reliability, cost), ready for the report.

The core functions take plain data so they unit-test without a live anything;
:func:`evaluate_track` is the thin glue that scores a persisted track.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from .config import Hosting, RunConfig
from .grading.behavioral import grade_track_dir
from .grading.memory_exam import grade_memory_exam
from .harness import SessionRow
from .scenarios.types import Persona


@dataclass(frozen=True)
class TrackEvaluation:
    """Per-track scores, the unit the rollup consumes."""

    model_id: str
    persona: str
    seed: int
    completed: bool
    capability: float  # 0..1
    memory: float  # 0..1
    tokens: int
    cost_usd: float
    latency_s: float


@dataclass(frozen=True)
class ModelRollup:
    """One model's aggregated numbers across all its tracks."""

    model_id: str
    display_name: str
    eliminated: bool
    reason: str = ""
    capability: float = 0.0
    memory: float = 0.0
    reliability: float = 0.0  # pass^k
    cost_usd: float = 0.0  # mean per track
    tokens: float = 0.0  # mean per track
    latency_s: float = 0.0
    n_tracks: int = 0


# --- cost --------------------------------------------------------------------


def impute_cost(sessions: list[SessionRow], price_in_per_1m: float, price_out_per_1m: float) -> float:
    """Dollar cost from token counts and a price assumption (incl. cache + reasoning)."""
    input_side = sum(s.input_tokens + s.cache_read_tokens + s.cache_write_tokens for s in sessions)
    output_side = sum(s.output_tokens + s.reasoning_tokens for s in sessions)
    return input_side * price_in_per_1m / 1e6 + output_side * price_out_per_1m / 1e6


def normalize_cost(sessions: list[SessionRow], model, run_config: RunConfig) -> float:
    """Comparable dollar cost for a track's sessions.

    API: the metered ``state.db`` dollars (imputed from the model's own price card
    if, unusually, nothing was metered). Local: always imputed from the run's
    self-hosting price assumption, since Ollama meters ``$0``.
    """
    metered = sum((s.actual_cost_usd or s.estimated_cost_usd) for s in sessions)
    if model.hosting == Hosting.API:
        if metered > 0:
            return metered
        return impute_cost(sessions, model.price_per_1m_input, model.price_per_1m_output)
    # LOCAL
    return impute_cost(
        sessions, run_config.local_price_per_1m_input, run_config.local_price_per_1m_output
    )


def tokens_to_complete(sessions: list[SessionRow]) -> int:
    return sum(s.total_tokens for s in sessions)


def latency_seconds(sessions: list[SessionRow]) -> float:
    """Wall-clock summed over sessions, where both timestamps parse (else skipped)."""
    total = 0.0
    for s in sessions:
        try:
            total += float(s.ended_at) - float(s.started_at)
        except (TypeError, ValueError):
            continue  # ended_at can be empty mid-accounting; count what we can
    return total


# --- reliability -------------------------------------------------------------


def compute_pass_k(success_by_task: dict[str, list[bool]], k: int) -> float:
    """``pass^k = E_task[p_task^k]``.

    ``success_by_task`` maps a task (here: a persona) to its per-seed pass/fail
    list. Tasks with no observations are ignored. With uniform ``p`` this returns
    ``p^k`` exactly.
    """
    rates = [sum(v) / len(v) for v in success_by_task.values() if v]
    if not rates:
        return 0.0
    return statistics.mean(p ** k for p in rates)


# --- rollup ------------------------------------------------------------------


def rollup(
    evaluations: list[TrackEvaluation],
    run_config: RunConfig,
    *,
    eliminated: Optional[dict[str, str]] = None,
    success_threshold: float = 0.5,
) -> list[ModelRollup]:
    """Aggregate per-track evaluations into one :class:`ModelRollup` per candidate.

    ``eliminated`` maps a model id to its Stage-1 drop reason; those models appear
    in the output marked eliminated (never silently omitted). A track counts as a
    reliability *success* when it completed and its capability meets
    ``success_threshold``.
    """
    eliminated = eliminated or {}
    by_model: dict[str, list[TrackEvaluation]] = {}
    for ev in evaluations:
        by_model.setdefault(ev.model_id, []).append(ev)

    name_of = {c.id: c.display_name for c in run_config.candidates}
    rollups: list[ModelRollup] = []

    # Survivors with tracks.
    for model_id, evs in by_model.items():
        success_by_persona: dict[str, list[bool]] = {}
        for ev in evs:
            success = ev.completed and ev.capability >= success_threshold
            success_by_persona.setdefault(ev.persona, []).append(success)
        rollups.append(ModelRollup(
            model_id=model_id,
            display_name=name_of.get(model_id, model_id),
            eliminated=False,
            capability=statistics.mean(e.capability for e in evs),
            memory=statistics.mean(e.memory for e in evs),
            reliability=compute_pass_k(success_by_persona, run_config.k),
            cost_usd=statistics.mean(e.cost_usd for e in evs),
            tokens=statistics.mean(e.tokens for e in evs),
            latency_s=statistics.mean(e.latency_s for e in evs),
            n_tracks=len(evs),
        ))

    # Eliminated models (no tracks) — surfaced with their reason.
    for model_id, reason in eliminated.items():
        if model_id in by_model:
            continue
        rollups.append(ModelRollup(
            model_id=model_id,
            display_name=name_of.get(model_id, model_id),
            eliminated=True,
            reason=reason,
        ))
    return rollups


# --- glue: score a persisted track -------------------------------------------


def evaluate_track(
    persona: Persona,
    model,
    *,
    track_dir: str,
    sessions: list[SessionRow],
    seed: int,
    completed: bool,
    run_config: RunConfig,
    memory_answers: Optional[dict[str, str]] = None,
    judge_mean_0_1: Optional[float] = None,
) -> TrackEvaluation:
    """Build a :class:`TrackEvaluation` from a persisted track and its graders.

    Capability is the behavioral-adherence score, blended with the judge's
    qualitative mean when supplied. Memory is the exam score (0 if no answers were
    collected). Cost/tokens/latency come from ``state.db`` sessions.
    """
    capability = grade_track_dir(persona, track_dir).score
    if judge_mean_0_1 is not None:
        capability = statistics.mean([capability, judge_mean_0_1])
    memory = grade_memory_exam(persona, memory_answers).score if memory_answers else 0.0
    return TrackEvaluation(
        model_id=model.id,
        persona=persona.name,
        seed=seed,
        completed=completed,
        capability=capability,
        memory=memory,
        tokens=tokens_to_complete(sessions),
        cost_usd=normalize_cost(sessions, model, run_config),
        latency_s=latency_seconds(sessions),
    )
