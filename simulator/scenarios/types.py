"""Shared scenario data contracts.

These are the shapes the orchestrator (U3) consumes and that later units fill:

- :class:`Stage1Task` — a single-shot pre-filter task (authored in U4, graded by
  the deterministic grader U7).
- :class:`ExogenousEvent` / :class:`DayPlan` / :class:`Persona` — the multi-day
  curriculum (schema/validation + first persona in U5; ``answer_key`` consumed by
  the memory exam U8 and behavioral checks U7).
- :class:`Counterparty` — the protocol the day loop calls between agent turns to
  let a spouse/school/kid stand-in reply (implemented in U6).

``world_seed`` everywhere is a :meth:`WorldState.seed` spec
(``{"emails": [...], "events": [...], "contacts": [...]}``). ``answer_key`` and
``expected_state`` are intentionally free-form dicts here — their internal shape
is owned by the grading units, keeping this module loosely coupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

# A world slice: maps "emails"/"events"/"contacts" to lists of row dicts.
WorldSeed = dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class Stage1Task:
    """A single-shot, memory-off cross-domain task with a deterministic end-state.

    The agent is given ``prompt`` against a world seeded with ``world_seed``;
    success is decided by comparing the resulting world to ``expected_state``
    (the deterministic grader's input). ``category`` groups tasks (e.g.
    ``"conflict"``, ``"coordination"``) for reporting.
    """

    id: str
    prompt: str
    world_seed: WorldSeed = field(default_factory=dict)
    expected_state: dict[str, Any] = field(default_factory=dict)
    category: str = "general"


@dataclass(frozen=True)
class ExogenousEvent:
    """Something that happens *to* the user on a given day, independent of the agent.

    Applied to the world at the start of its day, identically across every track
    (R2/AE1). ``kind`` selects the table; ``data`` is the row to seed.
    """

    kind: str  # "email" | "event" | "contact"
    data: dict[str, Any]
    description: str = ""


@dataclass(frozen=True)
class DayPlan:
    """One simulated day: exogenous events arrive, then the user gives a task."""

    day: int  # 1-based
    date: str  # "YYYY-MM-DD"
    user_prompt: str  # the instruction the user gives the agent this day
    inbound: tuple[ExogenousEvent, ...] = ()
    # Wall-clock the servers stamp writes with this day (defaults to 09:00 on date).
    sim_now: Optional[str] = None

    def clock(self) -> str:
        return self.sim_now or f"{self.date}T09:00:00"


@dataclass(frozen=True)
class Persona:
    """A multi-day curriculum for one synthetic person.

    ``world_seed`` is the day-0 world. ``days`` is the fixed exogenous stream.
    ``answer_key`` carries ground truth for the memory exam (U8) and behavioral
    checks (U7); its structure is owned by those units. ``counterparty_brief``
    configures the U6 stand-in.
    """

    name: str
    world_seed: WorldSeed
    days: tuple[DayPlan, ...]
    answer_key: dict[str, Any] = field(default_factory=dict)
    counterparty_brief: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Counterparty(Protocol):
    """A spouse/school/kid stand-in the day loop consults between agent turns.

    Given one outbound email the agent sent, return a reply email dict
    (``from_addr``, ``to_addr``, ``subject``, ``body``, ``timestamp``) to seed
    into the inbox, or ``None`` to stay silent. Implementations see only the
    agent's *messages* (this email), never its tool calls (partial
    observability, per tau-bench).
    """

    def reply(
        self, outbound_email: dict[str, Any], persona: Persona, *, sim_now: str
    ) -> Optional[dict[str, Any]]: ...
