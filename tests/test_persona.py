"""Tests for the persona schema, validation, and the first persona (U5)."""

from __future__ import annotations

import pytest

from simulator.scenarios.personas import ALL_PERSONAS, get_persona
from simulator.scenarios.personas.dana import PERSONA as DANA
from simulator.scenarios.personas.schema import (
    PROBE_ABSTENTION,
    PROBE_UPDATE,
    PersonaError,
    behavioral_signals,
    memory_probes,
    replay_events,
    validate_persona,
)
from simulator.scenarios.types import DayPlan, ExogenousEvent, Persona


# --- the shipped persona is valid -------------------------------------------


def test_dana_persona_validates():
    validate_persona(DANA)


def test_dana_is_registered():
    assert get_persona("dana") is DANA
    assert "dana" in ALL_PERSONAS


def test_dana_has_required_probe_kinds_structurally():
    kinds = {p["kind"] for p in memory_probes(DANA)}
    assert PROBE_UPDATE in kinds  # the soccer day change
    assert PROBE_ABSTENTION in kinds  # Theo's non-existent dentist appointment


def test_dana_knowledge_update_tracks_old_and_new():
    update = next(p for p in memory_probes(DANA) if p["kind"] == PROBE_UPDATE)
    assert "wednesday" in [k.lower() for k in update["expected"]]
    assert "thursday" in [k.lower() for k in update["stale"]]


def test_dana_behavioral_signals_present():
    ids = {s["id"] for s in behavioral_signals(DANA)}
    assert "respects_no_early_meetings" in ids
    assert "adopts_soccer_change" in ids


# --- the change actually appears in the exogenous stream ---------------------


def test_soccer_change_is_delivered_as_an_event():
    # The knowledge-update must be *learnable*: the change arrives in the inbox.
    stream = replay_events(DANA)
    moving = [
        data for _day, kind, data in stream
        if kind == "email" and "wednesday" in data.get("body", "").lower()
    ]
    assert moving, "the soccer-moves-to-Wednesday email must be in the event stream"


def test_replay_is_deterministic():
    assert replay_events(DANA) == replay_events(DANA)


# --- validation rejects malformed personas -----------------------------------


def _persona_dict() -> dict:
    return {
        "name": "x",
        "world_seed": {},
        "days": (DayPlan(day=1, date="2026-01-01", user_prompt="hi"),),
        "answer_key": {
            "memory_probes": [
                {"id": "u", "kind": PROBE_UPDATE, "question": "q?",
                 "expected": ["new"], "stale": ["old"]},
                {"id": "a", "kind": PROBE_ABSTENTION, "question": "q?",
                 "trap_keywords": ["nope"]},
            ]
        },
    }


def test_valid_minimal_persona_passes():
    validate_persona(Persona(**_persona_dict()))


def test_missing_knowledge_update_is_rejected():
    d = _persona_dict()
    d["answer_key"]["memory_probes"] = [
        {"id": "a", "kind": PROBE_ABSTENTION, "question": "q?", "trap_keywords": ["x"]}
    ]
    with pytest.raises(PersonaError, match="knowledge_update"):
        validate_persona(Persona(**d))


def test_missing_abstention_is_rejected():
    d = _persona_dict()
    d["answer_key"]["memory_probes"] = [
        {"id": "u", "kind": PROBE_UPDATE, "question": "q?",
         "expected": ["new"], "stale": ["old"]}
    ]
    with pytest.raises(PersonaError, match="abstention"):
        validate_persona(Persona(**d))


def test_out_of_order_days_rejected():
    d = _persona_dict()
    d["days"] = (
        DayPlan(day=1, date="2026-01-01", user_prompt="a"),
        DayPlan(day=3, date="2026-01-02", user_prompt="b"),  # gap
    )
    with pytest.raises(PersonaError, match="out-of-order"):
        validate_persona(Persona(**d))


def test_malformed_event_kind_rejected():
    d = _persona_dict()
    d["days"] = (
        DayPlan(day=1, date="2026-01-01", user_prompt="a",
                inbound=(ExogenousEvent("telegram", {}),)),
    )
    with pytest.raises(PersonaError, match="bad event kind"):
        validate_persona(Persona(**d))


def test_update_probe_missing_stale_rejected():
    d = _persona_dict()
    d["answer_key"]["memory_probes"][0] = {
        "id": "u", "kind": PROBE_UPDATE, "question": "q?", "expected": ["new"]
    }  # no 'stale'
    with pytest.raises(PersonaError, match="stale"):
        validate_persona(Persona(**d))
