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


# --- labelled picks (R6) -----------------------------------------------------
# The eval deliberately crowns no single composite winner; instead it surfaces
# three labelled picks against a viability floor, plus the full table under more
# than one weighting, so the user makes the accuracy-vs-cost trade-off themselves.

# Viability floor gating best-value and cheapest-viable (KTD3). Tunable.
CAP_FLOOR = 0.60
MEM_FLOOR = 0.50
REL_FLOOR = 0.75


def _passes_floor(row: ReportRow) -> bool:
    return (row.capability >= CAP_FLOOR
            and row.memory >= MEM_FLOOR
            and row.reliability >= REL_FLOOR)


@dataclass(frozen=True)
class Picks:
    """Three labelled recommendations derived from the per-dimension rows."""

    best_accuracy: Optional[ReportRow]  # max capability+memory, floor ignored
    best_value: Optional[ReportRow]  # max (cap+mem)/cost among floor-passers ($>0)
    cheapest_viable: Optional[ReportRow]  # min cost among floor-passers
    n_floor_passers: int


def pick_labels(rows: list[ReportRow]) -> Picks:
    """Compute the three labelled picks from the (survivor) rows.

    Picks are weighting-independent — they read raw dimensions and cost, not the
    composite — so any built report's ``ranked`` rows can feed this. best-value
    guards ``cost_usd <= 0`` (the api-free $0 smoke and any metered-$0 track would
    otherwise divide by zero), excluding such rows from the value ratio.
    """
    survivors = [r for r in rows if not r.eliminated]
    if not survivors:
        return Picks(None, None, None, 0)

    best_accuracy = max(survivors, key=lambda r: r.capability + r.memory)
    passers = [r for r in survivors if _passes_floor(r)]

    value_eligible = [r for r in passers if r.cost_usd > 0]
    best_value = (max(value_eligible, key=lambda r: (r.capability + r.memory) / r.cost_usd)
                  if value_eligible else None)
    cheapest_viable = min(passers, key=lambda r: r.cost_usd) if passers else None

    return Picks(best_accuracy, best_value, cheapest_viable, len(passers))


def render_picks(picks: Picks) -> str:
    """Human-readable summary of the three picks."""
    def _line(label: str, row: Optional[ReportRow]) -> str:
        if row is None:
            return f"  {label:<16} (none)"
        return (f"  {label:<16} {row.display_name}  "
                f"(cap {row.capability:.2f} / mem {row.memory:.2f} / "
                f"rel {row.reliability:.2f} / ${row.cost_usd:.4f}/task / "
                f"{row.latency_s:.1f}s)")

    lines = [
        "Picks (no single winner — pick by your priority):",
        f"  floor: cap>={CAP_FLOOR:.2f}, mem>={MEM_FLOOR:.2f}, rel>={REL_FLOOR:.2f} "
        f"— {picks.n_floor_passers} model(s) clear it",
        _line("best accuracy", picks.best_accuracy),
        _line("best value", picks.best_value),
        _line("cheapest viable", picks.cheapest_viable),
    ]
    if picks.n_floor_passers == 0:
        lines.append("  (no model clears the floor — best-value/cheapest-viable are (none))")
    return "\n".join(lines)


# --- multiple named weightings (R6) ------------------------------------------
# memory_heavy is the unchanged default; cost_forward raises cost+speed and lowers
# memory. This named pair is the ONLY home for the speed/cost rebalance — the
# global default (CompositeWeights()) stays memory-heavy with speed 0.0.
NAMED_WEIGHTINGS: dict[str, CompositeWeights] = {
    "memory_heavy": CompositeWeights(),  # 0.35 / 0.35 / 0.20 / 0.10 / 0.0
    "cost_forward": CompositeWeights(
        capability=0.25, memory=0.20, reliability=0.15, cost=0.25, speed=0.15
    ),
}


def render_weightings(
    rollups: list[ModelRollup],
    weightings: Optional[dict[str, CompositeWeights]] = None,
) -> str:
    """Render the ranked table once per named weighting, plus the picks once.

    Picks are weighting-independent, so they are computed from the first weighting's
    rows and shown a single time after the tables.
    """
    weightings = weightings or NAMED_WEIGHTINGS
    blocks: list[str] = []
    first_report: Optional[Report] = None
    for name, weights in weightings.items():
        report = build_report(rollups, weights)
        if first_report is None:
            first_report = report
        blocks.append(f"== weighting: {name} ==\n{render_table(report)}")
    if first_report is not None:
        blocks.append(render_picks(pick_labels(first_report.rows)))
    return "\n\n".join(blocks)
