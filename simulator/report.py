"""The comparison report (R14).

Takes per-model rollups and renders a ranked, side-by-side table: every
dimension shown separately (capability, memory, reliability, cost) **and** a
single configurable weighted composite. The per-dimension columns are always
present regardless of the weights — the weights only decide the composite and the
ranking. Eliminated models are listed with their Stage-1 reason, never dropped.

Composite normalization: capability/memory/reliability are already 0..1 (higher
is better) and used directly. Cost (dollars, lower is better) is min-max inverted
across the ranked models so the cheapest scores 1.0 and the priciest 0.0. The
composite is the weight-normalized sum.
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


def _cost_scores(costs: list[float]) -> list[float]:
    """Min-max invert cost to a 0..1 score (cheapest=1). Degenerate range -> all 1."""
    if not costs:
        return []
    lo, hi = min(costs), max(costs)
    if hi == lo:
        return [1.0 for _ in costs]
    return [(hi - c) / (hi - lo) for c in costs]


def build_report(rollups: list[ModelRollup], weights: CompositeWeights) -> Report:
    """Compute composites and rank. Survivors are ranked by composite descending."""
    w = weights.normalized()
    survivors = [r for r in rollups if not r.eliminated]
    eliminated = [r for r in rollups if r.eliminated]

    cost_scores = _cost_scores([r.cost_usd for r in survivors])
    scored: list[tuple[ModelRollup, float]] = []
    for r, cost_score in zip(survivors, cost_scores):
        composite = (
            w.capability * r.capability
            + w.memory * r.memory
            + w.reliability * r.reliability
            + w.cost * cost_score
        )
        scored.append((r, composite))

    scored.sort(key=lambda rc: rc[1], reverse=True)

    rows: list[ReportRow] = []
    for rank, (r, composite) in enumerate(scored, start=1):
        rows.append(ReportRow(
            model_id=r.model_id, display_name=r.display_name, eliminated=False, reason="",
            capability=r.capability, memory=r.memory, reliability=r.reliability,
            cost_usd=r.cost_usd, tokens=r.tokens, composite=composite, rank=rank,
        ))
    for r in eliminated:
        rows.append(ReportRow(
            model_id=r.model_id, display_name=r.display_name, eliminated=True, reason=r.reason,
            capability=r.capability, memory=r.memory, reliability=r.reliability,
            cost_usd=r.cost_usd, tokens=r.tokens, composite=None, rank=None,
        ))
    return Report(rows=rows, weights=w)


def render_table(report: Report) -> str:
    """Render the report as a fixed-width text table."""
    header = (
        f"{'#':>2}  {'Model':<22} {'Cap':>5} {'Mem':>5} {'Rel':>5} "
        f"{'Cost$':>8} {'Tokens':>9} {'Composite':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in report.ranked:
        lines.append(
            f"{row.rank:>2}  {row.display_name:<22} "
            f"{row.capability:>5.2f} {row.memory:>5.2f} {row.reliability:>5.2f} "
            f"{row.cost_usd:>8.4f} {row.tokens:>9.0f} {row.composite:>9.3f}"
        )
    if report.eliminated:
        lines.append("")
        lines.append("Eliminated in Stage 1:")
        for row in report.eliminated:
            lines.append(f"  - {row.display_name}: {row.reason}")
    return "\n".join(lines)
