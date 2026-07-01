"""Behavioral-improvement grading.

Distinct from the end-of-run memory exam (U8): this measures *whether behavior
improves over the days* of a track, against the ``behavioral_signals`` a persona
declares. It reads out-of-band artifacts only — the final world snapshot and the
per-day trajectory the runner persisted — never trusting an agent's own claims.

Two signal kinds (see ``personas/schema.py``):

- ``no_event_before`` — a standing preference (e.g. "no meetings before 9am").
  Passes iff the final calendar has no event starting before that time-of-day.
- ``not_after_day`` — a corrected fact (e.g. soccer moved off Thursday on day 2).
  Passes iff, *after* the learning day, the agent stops emitting the now-stale
  keyword. This is the AE3 signal: a model that keeps surfacing the corrected
  Thursday scores worse than one that adopts Wednesday.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

from ..scenarios.personas.schema import (
    SIGNAL_NO_EVENT_BEFORE,
    SIGNAL_NOT_AFTER_DAY,
    behavioral_signals,
)
from ..scenarios.types import Persona


@dataclass(frozen=True)
class BehavioralResult:
    signal_id: str
    kind: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class BehavioralReport:
    results: list[BehavioralResult]

    @property
    def score(self) -> float:
        """Fraction of behavioral signals passed (1.0 if none defined)."""
        if not self.results:
            return 1.0
        return sum(r.passed for r in self.results) / len(self.results)


def _parse_time(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


def score_no_event_before(
    snapshot: dict[str, list[dict]],
    hhmm: str,
    *,
    exempt_starts: frozenset[str] = frozenset(),
) -> tuple[bool, str]:
    """Pass iff no *agent-created* calendar event starts before ``hhmm``.

    ``exempt_starts`` lists the start datetimes of exogenous events (seeded or
    inbound) — those were not scheduled by the agent, so they don't count against
    a "never books before 9am" preference even if they fall early.
    """
    floor = _parse_time(hhmm)
    early = []
    for e in snapshot.get("events", []):
        if e.get("start") in exempt_starts:
            continue
        try:
            start_time = datetime.fromisoformat(e["start"]).time()
        except (ValueError, TypeError, KeyError):
            # A model may write a non-ISO start (e.g. "this week 10:00 AM"); it
            # can't be assessed against the floor, so skip it rather than crash the
            # grader (and, downstream, the whole report rebuild).
            continue
        if start_time < floor:
            early.append(e)
    if early:
        titles = [e.get("title") for e in early]
        return False, f"events scheduled before {hhmm}: {titles}"
    return True, f"no agent-created events before {hhmm}"


def score_not_after_day(
    agent_text_by_day: dict[int, str],
    learned_on_day: int,
    forbidden_keyword: str,
    *,
    corrected_keyword: Optional[str] = None,
) -> tuple[bool, str]:
    """Pass iff the agent stops *relying on* the stale keyword after correction.

    A day after ``learned_on_day`` counts as a repeat only when the forbidden
    keyword appears AND (when ``corrected_keyword`` is given) the corrected one
    does not — so "moved from Thursday to Wednesday" is fine (it states the new
    fact and merely references the old as history), while "see you Thursday" is a
    repeat. Days at or before ``learned_on_day`` are exempt.
    """
    forbidden = forbidden_keyword.lower()
    corrected = corrected_keyword.lower() if corrected_keyword else None
    repeats = []
    for day, text in sorted(agent_text_by_day.items()):
        if day <= learned_on_day:
            continue
        low = (text or "").lower()
        if forbidden in low and (corrected is None or corrected not in low):
            repeats.append(day)
    if repeats:
        return False, f"still relied on {forbidden_keyword!r} on day(s) {repeats} after correction"
    return True, f"dropped {forbidden_keyword!r} after day {learned_on_day}"


def exogenous_event_starts(persona: Persona) -> frozenset[str]:
    """Start datetimes of every non-agent event (seeded + inbound), for exemption."""
    starts = {e.get("start") for e in persona.world_seed.get("events", []) if e.get("start")}
    for day in persona.days:
        for ev in day.inbound:
            if ev.kind == "event" and ev.data.get("start"):
                starts.add(ev.data["start"])
    return frozenset(starts)


def grade_behavioral(
    persona: Persona,
    snapshot: dict[str, list[dict]],
    agent_text_by_day: dict[int, str],
) -> BehavioralReport:
    """Evaluate every behavioral signal the persona declares."""
    exempt = exogenous_event_starts(persona)
    results: list[BehavioralResult] = []
    for signal in behavioral_signals(persona):
        kind = signal["kind"]
        if kind == SIGNAL_NO_EVENT_BEFORE:
            passed, detail = score_no_event_before(
                snapshot, signal["time"], exempt_starts=exempt
            )
        elif kind == SIGNAL_NOT_AFTER_DAY:
            passed, detail = score_not_after_day(
                agent_text_by_day, signal["learned_on_day"], signal["forbidden_keyword"],
                corrected_keyword=signal.get("corrected_keyword"),
            )
        else:  # pragma: no cover - schema validation forbids this
            raise ValueError(f"unknown behavioral signal kind {kind!r}")
        results.append(BehavioralResult(signal["id"], kind, passed, detail))
    return BehavioralReport(results)


# --- loading persisted trajectories (out-of-band) ----------------------------


def load_track(track_dir: str | Path) -> tuple[dict, dict[int, str]]:
    """Read a persisted track into ``(final_world_snapshot, agent_text_by_day)``.

    ``agent_text_by_day`` combines each day's agent stdout with the bodies of any
    emails the agent sent that day (matched by timestamp date) — the agent's
    observable output for the behavioral keyword scan.
    """
    track_dir = Path(track_dir)
    snapshot = json.loads((track_dir / "final_world.json").read_text())

    # date -> day number, to attribute sent emails to a day.
    day_files = sorted(track_dir.glob("day_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    texts: dict[int, str] = {}
    date_to_day: dict[str, int] = {}
    for f in day_files:
        rec = json.loads(f.read_text())
        texts[rec["day"]] = rec.get("stdout", "")
        date_to_day[rec.get("date", "")] = rec["day"]

    for email in snapshot.get("emails", []):
        if email.get("folder") != "sent":
            continue
        date = str(email.get("timestamp", ""))[:10]
        day = date_to_day.get(date)
        if day is not None:
            texts[day] = texts.get(day, "") + "\n" + email.get("subject", "") + "\n" + email.get("body", "")
    return snapshot, texts


def grade_track_dir(persona: Persona, track_dir: str | Path) -> BehavioralReport:
    """Convenience: load a persisted track and grade its behavioral signals."""
    snapshot, texts = load_track(track_dir)
    return grade_behavioral(persona, snapshot, texts)
