"""End-of-run memory exam.

After the day loop, the agent (memory intact, in the same HERMES_HOME) is asked
the persona's probe questions; this module scores the answers **forgetting-aware**
against the answer key (KTD5):

- ``recall`` — correct iff every expected keyword is present.
- ``knowledge_update`` — correct iff the *new* fact is present. A confidently
  *stale* answer (old fact, no new) is scored ``stale`` (wrong), not merely
  ``missing`` — a model that remembers the wrong thing is worse than one that
  forgot. Mentioning the old fact as history alongside the new one still counts.
- ``abstention`` — correct iff the agent declines (an event that never happened);
  a fabricated specific answer is wrong.

Scoring is a pure function of (answer text, probe), so it is fully testable
without a live model. Administering the exam (asking the harness) is separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..scenarios.personas.schema import (
    PROBE_ABSTENTION,
    PROBE_RECALL,
    PROBE_UPDATE,
    memory_probes,
)
from ..scenarios.types import Persona

# Phrases that signal the agent is declining / saying no such thing exists.
_ABSTAIN_PHRASES = (
    "no such", "not aware", "no record", "no appointment", "nothing scheduled",
    "don't have", "do not have", "don't see", "do not see", "couldn't find",
    "could not find", "can't find", "cannot find", "isn't any", "aren't any",
    "there is no", "there's no", "no dentist", "not scheduled", "no scheduled",
    "i don't", "i do not", "none ", "no, ",
)


@dataclass(frozen=True)
class ProbeResult:
    probe_id: str
    kind: str
    correct: bool
    label: str  # correct | stale | missing | abstained | fabricated
    answer: str


@dataclass(frozen=True)
class MemoryExamReport:
    results: list[ProbeResult]

    @property
    def score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.correct for r in self.results) / len(self.results)

    def by_kind(self) -> dict[str, float]:
        """Per-ability accuracy (recall / knowledge_update / abstention)."""
        out: dict[str, float] = {}
        for kind in (PROBE_RECALL, PROBE_UPDATE, PROBE_ABSTENTION):
            kind_results = [r for r in self.results if r.kind == kind]
            if kind_results:
                out[kind] = sum(r.correct for r in kind_results) / len(kind_results)
        return out


def _all_present(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return all(k.lower() in a for k in keywords)


def _any_present(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return any(k.lower() in a for k in keywords)


def score_probe(answer: str, probe: dict) -> ProbeResult:
    """Score one answer against one probe (pure, forgetting-aware)."""
    kind = probe["kind"]
    pid = probe["id"]

    if kind == PROBE_RECALL:
        correct = _all_present(answer, probe["expected"])
        return ProbeResult(pid, kind, correct, "correct" if correct else "missing", answer)

    if kind == PROBE_UPDATE:
        new_present = _all_present(answer, probe["expected"])
        stale_present = _any_present(answer, probe["stale"])
        if new_present:
            return ProbeResult(pid, kind, True, "correct", answer)
        if stale_present:  # remembered the wrong (old) thing -> worse than missing
            return ProbeResult(pid, kind, False, "stale", answer)
        return ProbeResult(pid, kind, False, "missing", answer)

    if kind == PROBE_ABSTENTION:
        declined = _any_present(answer, list(_ABSTAIN_PHRASES))
        if declined:
            return ProbeResult(pid, kind, True, "abstained", answer)
        return ProbeResult(pid, kind, False, "fabricated", answer)

    raise ValueError(f"unknown probe kind {kind!r}")  # pragma: no cover


def grade_memory_exam(persona: Persona, answers: dict[str, str]) -> MemoryExamReport:
    """Grade collected answers (probe_id -> answer text) against the answer key."""
    results = []
    for probe in memory_probes(persona):
        answer = answers.get(probe["id"], "")
        results.append(score_probe(answer, probe))
    return MemoryExamReport(results)


def administer_exam(
    harness, persona: Persona, *, extra_env: Optional[dict[str, str]] = None
) -> dict[str, str]:
    """Ask each probe question to the (memory-intact) harness; return the answers.

    Each probe is a fresh one-shot in the same home, so the agent answers from
    accrued memory (and may consult its tools — a model that trusts a stale
    calendar over a remembered correction earns the penalty that implies).
    """
    answers: dict[str, str] = {}
    for probe in memory_probes(persona):
        result = harness.run_oneshot(probe["question"], extra_env=extra_env)
        answers[probe["id"]] = result.stdout
    return answers
