"""Persona schema: the structure of a persona's ``answer_key`` and the validation
that a persona is well-formed enough to grade.

A persona's ``answer_key`` has two parts:

- ``memory_probes`` — end-of-run questions for the memory exam (U8). Three kinds:

  - ``recall`` — a planted fact. Correct iff the answer contains every keyword in
    ``expected`` (case-insensitive).
  - ``knowledge_update`` — a fact that *changed* mid-run. Correct iff the answer
    contains the ``expected`` (new) keywords AND none of the ``stale`` (old) ones
    — forgetting-aware: a confidently-stale answer is wrong, not merely missing.
  - ``abstention`` — an event that never happened. Correct iff the answer
    declines (matches an abstention phrase) and asserts none of ``trap_keywords``.

- ``behavioral_signals`` — improvement/adherence checks over the trajectory (U7).
  Each names a ``kind`` the behavioral grader knows how to evaluate.

This module owns only the *shape* and validation; the scoring lives in U7/U8.
A valid persona must carry at least one ``knowledge_update`` and one
``abstention`` probe — the two abilities that most distinguish real memory from
lucky recall.
"""

from __future__ import annotations

from typing import Any

from ..types import Persona

PROBE_RECALL = "recall"
PROBE_UPDATE = "knowledge_update"
PROBE_ABSTENTION = "abstention"
PROBE_KINDS = {PROBE_RECALL, PROBE_UPDATE, PROBE_ABSTENTION}

# Behavioral signal kinds the U7 grader understands (see behavioral.py).
SIGNAL_NO_EVENT_BEFORE = "no_event_before"  # preference: no calendar event before a time
SIGNAL_NOT_AFTER_DAY = "not_after_day"  # stops using a stale keyword after it's corrected
SIGNAL_KINDS = {SIGNAL_NO_EVENT_BEFORE, SIGNAL_NOT_AFTER_DAY}


class PersonaError(ValueError):
    """A persona is structurally invalid and cannot be graded reliably."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PersonaError(msg)


def _validate_probe(probe: dict[str, Any]) -> None:
    pid = probe.get("id", "<no id>")
    _require("id" in probe, "memory probe missing 'id'")
    kind = probe.get("kind")
    _require(kind in PROBE_KINDS, f"probe {pid!r}: bad kind {kind!r}")
    _require(bool(probe.get("question")), f"probe {pid!r}: missing question")
    if kind == PROBE_RECALL:
        _require(bool(probe.get("expected")), f"probe {pid!r}: recall needs 'expected' keywords")
    elif kind == PROBE_UPDATE:
        _require(bool(probe.get("expected")), f"probe {pid!r}: update needs 'expected' (new) keywords")
        _require(bool(probe.get("stale")), f"probe {pid!r}: update needs 'stale' (old) keywords")
    elif kind == PROBE_ABSTENTION:
        # An abstention probe needs the trap details it must NOT assert.
        _require(bool(probe.get("trap_keywords")), f"probe {pid!r}: abstention needs 'trap_keywords'")


def _validate_signal(signal: dict[str, Any]) -> None:
    sid = signal.get("id", "<no id>")
    _require("id" in signal, "behavioral signal missing 'id'")
    _require(signal.get("kind") in SIGNAL_KINDS, f"signal {sid!r}: bad kind {signal.get('kind')!r}")
    _require(bool(signal.get("description")), f"signal {sid!r}: missing description")
    if signal["kind"] == SIGNAL_NO_EVENT_BEFORE:
        _require(bool(signal.get("time")), f"signal {sid!r}: needs 'time' (e.g. '09:00')")
    elif signal["kind"] == SIGNAL_NOT_AFTER_DAY:
        _require("learned_on_day" in signal, f"signal {sid!r}: needs 'learned_on_day'")
        _require(bool(signal.get("forbidden_keyword")), f"signal {sid!r}: needs 'forbidden_keyword'")


def memory_probes(persona: Persona) -> list[dict[str, Any]]:
    return list(persona.answer_key.get("memory_probes", []))


def behavioral_signals(persona: Persona) -> list[dict[str, Any]]:
    return list(persona.answer_key.get("behavioral_signals", []))


def validate_persona(persona: Persona) -> None:
    """Raise :class:`PersonaError` if the persona is not well-formed."""
    _require(bool(persona.name), "persona needs a name")
    _require(len(persona.days) >= 1, "persona needs at least one day")
    # Days must be numbered 1..N in order — the runner replays them in sequence.
    for i, day in enumerate(persona.days, start=1):
        _require(day.day == i, f"day {i}: out-of-order day number {day.day}")
        _require(bool(day.date), f"day {i}: missing date")
        _require(bool(day.user_prompt), f"day {i}: missing user_prompt")
        for ev in day.inbound:
            _require(ev.kind in {"email", "event", "contact"}, f"day {i}: bad event kind {ev.kind!r}")

    probes = memory_probes(persona)
    _require(bool(probes), "answer_key needs at least one memory probe")
    for probe in probes:
        _validate_probe(probe)
    kinds = {p["kind"] for p in probes}
    _require(PROBE_UPDATE in kinds, "persona must include a knowledge_update probe")
    _require(PROBE_ABSTENTION in kinds, "persona must include an abstention probe")

    for signal in behavioral_signals(persona):
        _validate_signal(signal)


def replay_events(persona: Persona) -> list[tuple[int, str, dict]]:
    """Flatten the exogenous stream to ``(day, kind, data)`` tuples, in order.

    Pure function over static persona data, so calling it twice yields identical
    output — the determinism the runner relies on (R2/AE1).
    """
    out: list[tuple[int, str, dict]] = []
    for day in persona.days:
        for ev in day.inbound:
            out.append((day.day, ev.kind, dict(ev.data)))
    return out
