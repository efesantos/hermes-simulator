"""Deterministic state-diff grading.

The crisp half of the hybrid grader (KTD4): compare the world's end-state to an
``expected_state`` spec made of key-value assertions (tau-bench-style). No LLM,
no agent transcript — only the out-of-band world snapshot, so an agent cannot
talk its way to a pass (KTD2).

An ``expected_state`` is a dict with any of these keys:

- ``events_present`` — list of *event matchers*; each must match >= 1 event.
- ``events_absent``  — list of event matchers; each must match 0 events.
- ``emails_present`` — list of *email matchers* over sent mail; each must match >= 1.
- ``emails_absent``  — list of email matchers over sent mail; each must match 0.

An **event matcher** is a dict of predicates, all of which must hold:
  ``title_contains`` / ``location_contains`` (case-insensitive substring),
  ``day`` ("YYYY-MM-DD" prefix of start), ``start`` (exact),
  ``start_time_at_or_after`` / ``start_time_before`` ("HH:MM" time-of-day),
  ``overlaps`` ({"start","end"} ISO window).

An **email matcher** (over the sent folder) supports:
  ``to_contains`` / ``from_contains`` / ``subject_contains`` / ``body_contains``.

This keeps tasks crisp enough to grade without judgment, which is the whole point
of the Stage-1 pre-filter and the deterministic dimension of Stage 2.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any

from ..world.state import WorldState


def _ci_contains(haystack: Any, needle: str) -> bool:
    return needle.lower() in str(haystack or "").lower()


def _parse_time(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


def _overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    sa, ea = datetime.fromisoformat(a_start), datetime.fromisoformat(a_end)
    sb, eb = datetime.fromisoformat(b_start), datetime.fromisoformat(b_end)
    return sa < eb and sb < ea


def _event_matches(event: dict, matcher: dict) -> bool:
    for key, val in matcher.items():
        if key == "title_contains":
            if not _ci_contains(event.get("title"), val):
                return False
        elif key == "location_contains":
            if not _ci_contains(event.get("location"), val):
                return False
        elif key == "day":
            if not str(event.get("start", "")).startswith(val):
                return False
        elif key == "start":
            if event.get("start") != val:
                return False
        elif key == "start_time_at_or_after":
            if datetime.fromisoformat(event["start"]).time() < _parse_time(val):
                return False
        elif key == "start_time_before":
            if datetime.fromisoformat(event["start"]).time() >= _parse_time(val):
                return False
        elif key == "overlaps":
            if not _overlaps(event["start"], event["end"], val["start"], val["end"]):
                return False
        else:
            raise ValueError(f"unknown event predicate {key!r}")
    return True


def _email_matches(email: dict, matcher: dict) -> bool:
    field_map = {
        "to_contains": "to_addr",
        "from_contains": "from_addr",
        "subject_contains": "subject",
        "body_contains": "body",
    }
    for key, val in matcher.items():
        field = field_map.get(key)
        if field is None:
            raise ValueError(f"unknown email predicate {key!r}")
        if not _ci_contains(email.get(field), val):
            return False
    return True


def check_world(snapshot: dict[str, list[dict]], expected: dict[str, Any]) -> list[str]:
    """Return a list of human-readable failures (empty list == pass).

    ``snapshot`` is a :meth:`WorldState.inspect` result. Unknown top-level keys in
    ``expected`` are an error, not a silent skip — a typo in a task should fail
    loudly rather than vacuously pass.
    """
    events = snapshot.get("events", [])
    sent = [e for e in snapshot.get("emails", []) if e.get("folder") == "sent"]
    failures: list[str] = []

    for key, specs in expected.items():
        if key == "events_present":
            for matcher in specs:
                if not any(_event_matches(e, matcher) for e in events):
                    failures.append(f"no event matching {matcher}")
        elif key == "events_absent":
            for matcher in specs:
                hits = [e for e in events if _event_matches(e, matcher)]
                if hits:
                    failures.append(f"unexpected event matching {matcher}: {[h.get('title') for h in hits]}")
        elif key == "emails_present":
            for matcher in specs:
                if not any(_email_matches(m, matcher) for m in sent):
                    failures.append(f"no sent email matching {matcher}")
        elif key == "emails_absent":
            for matcher in specs:
                if any(_email_matches(m, matcher) for m in sent):
                    failures.append(f"unexpected sent email matching {matcher}")
        else:
            raise ValueError(f"unknown expected_state key {key!r}")
    return failures


def grade_task(world: WorldState, expected: dict[str, Any]) -> tuple[bool, str]:
    """Grade one task's end-state. Matches the runner's ``Stage1Grader`` signature.

    Returns ``(passed, detail)`` where detail names the first offending
    assertion(s) — so a failure is diagnosable, not just a red mark.
    """
    failures = check_world(world.inspect(), expected)
    if not failures:
        return True, "all assertions passed"
    return False, "; ".join(failures)
