"""The comparison report (R14).

Takes per-model rollups and renders a ranked, side-by-side table: every
dimension shown separately (capability, memory, reliability, cost) **and** a
single configurable weighted composite. The per-dimension columns are always
present regardless of the weights — the weights only decide the composite and the
ranking. Eliminated models are listed with their Stage-1 reason, never dropped.

Composite normalization: capability/memory/reliability are already 0..1 (higher
is better) and used directly. Cost (dollars, lower is better) and speed (latency
seconds, lower is better) are each min-max inverted across the ranked models so
the cheapest / fastest scores 1.0 and the priciest / slowest 0.0. The composite
is the weight-normalized sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import CompositeWeights
from .metrics import ModelRollup


@dataclass(frozen=True)
class ReportRow:
    model_id: str
    display_name: str
    eliminated: bool
    reason: str
    capability: float
    memory: float
    reliability: float
    cost_usd: float
    tokens: float
    latency_s: float
    composite: Optional[float]  # None for eliminated models
    rank: Optional[int]  # None for eliminated models


@dataclass(frozen=True)
class Report:
    rows: list[ReportRow]  # survivors first (ranked), then eliminated
    weights: CompositeWeights

    @property
    def ranked(self) -> list[ReportRow]:
        return [r for r in self.rows if not r.eliminated]

    @property
    def eliminated(self) -> list[ReportRow]:
        return [r for r in self.rows if r.eliminated]


def _invert_min_max(values: list[float]) -> list[float]:
    """Min-max invert to a 0..1 score (lowest=1, highest=0). Degenerate range -> all 1.

    Used for cost (cheapest=1) and speed (fastest=1) — both "lower is better".
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(hi - v) / (hi - lo) for v in values]


# Back-compat alias; cost and speed share the same "lower is better" inversion.
_cost_scores = _invert_min_max
_speed_scores = _invert_min_max


def build_report(rollups: list[ModelRollup], weights: CompositeWeights) -> Report:
    """Compute composites and rank. Survivors are ranked by composite descending."""
    w = weights.normalized()
    survivors = [r for r in rollups if not r.eliminated]
    eliminated = [r for r in rollups if r.eliminated]

    cost_scores = _cost_scores([r.cost_usd for r in survivors])
    speed_scores = _speed_scores([r.latency_s for r in survivors])
    scored: list[tuple[ModelRollup, float]] = []
    for r, cost_score, speed_score in zip(survivors, cost_scores, speed_scores):
        composite = (
            w.capability * r.capability
            + w.memory * r.memory
            + w.reliability * r.reliability
            + w.cost * cost_score
            + w.speed * speed_score
        )
        scored.append((r, composite))

    scored.sort(key=lambda rc: rc[1], reverse=True)

    rows: list[ReportRow] = []
    for rank, (r, composite) in enumerate(scored, start=1):
        rows.append(ReportRow(
            model_id=r.model_id, display_name=r.display_name, eliminated=False, reason="",
            capability=r.capability, memory=r.memory, reliability=r.reliability,
            cost_usd=r.cost_usd, tokens=r.tokens, latency_s=r.latency_s,
            composite=composite, rank=rank,
        ))
    for r in eliminated:
        rows.append(ReportRow(
            model_id=r.model_id, display_name=r.display_name, eliminated=True, reason=r.reason,
            capability=r.capability, memory=r.memory, reliability=r.reliability,
            cost_usd=r.cost_usd, tokens=r.tokens, latency_s=r.latency_s,
            composite=None, rank=None,
        ))
    return Report(rows=rows, weights=w)


def render_table(report: Report) -> str:
    """Render the report as a fixed-width text table."""
    header = (
        f"{'#':>2}  {'Model':<22} {'Cap':>5} {'Mem':>5} {'Rel':>5} "
        f"{'Cost$':>8} {'Speed(s)':>9} {'Tokens':>9} {'Composite':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in report.ranked:
        lines.append(
            f"{row.rank:>2}  {row.display_name:<22} "
            f"{row.capability:>5.2f} {row.memory:>5.2f} {row.reliability:>5.2f} "
            f"{row.cost_usd:>8.4f} {row.latency_s:>9.1f} {row.tokens:>9.0f} "
            f"{row.composite:>9.3f}"
        )
    if report.eliminated:
        lines.append("")
        lines.append("Eliminated in Stage 1:")
        for row in report.eliminated:
            lines.append(f"  - {row.display_name}: {row.reason}")
    return "\n".join(lines)
