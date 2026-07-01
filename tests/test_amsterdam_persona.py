"""Tests for the multilingual Amsterdam persona (U1).

Covers structural validity, the learnable/updated regularities appearing in the
exogenous stream, the multilingual recall probes, and — critically — a
pattern-based **PII guard** that keeps real identifiers out of the committed
persona file. The guard deliberately checks *shapes* (email domains, phone /
postal patterns) rather than a denylist of real names, because embedding real
names in the test would itself leak the PII the guard exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

from simulator.scenarios.personas import ALL_PERSONAS, get_persona
from simulator.scenarios.personas.amsterdam import PERSONA as AMSTERDAM
from simulator.scenarios.personas.schema import (
    PROBE_ABSTENTION,
    PROBE_RECALL,
    PROBE_UPDATE,
    behavioral_signals,
    memory_probes,
    replay_events,
    validate_persona,
)

_PERSONA_FILE = Path(__file__).resolve().parents[1] / "simulator" / "scenarios" / "personas" / "amsterdam.py"


# --- validity + registration -------------------------------------------------


def test_amsterdam_persona_validates():
    validate_persona(AMSTERDAM)


def test_amsterdam_is_registered():
    assert get_persona("amsterdam") is AMSTERDAM
    assert "amsterdam" in ALL_PERSONAS


def test_amsterdam_has_required_probe_kinds():
    kinds = {p["kind"] for p in memory_probes(AMSTERDAM)}
    assert PROBE_RECALL in kinds
    assert PROBE_UPDATE in kinds
    assert PROBE_ABSTENTION in kinds


def test_knowledge_update_tracks_old_and_new():
    update = next(p for p in memory_probes(AMSTERDAM) if p["kind"] == PROBE_UPDATE)
    assert "sunday" in [k.lower() for k in update["expected"]]
    assert "saturday" in [k.lower() for k in update["stale"]]


def test_abstention_probe_has_trap():
    ab = next(p for p in memory_probes(AMSTERDAM) if p["kind"] == PROBE_ABSTENTION)
    assert ab["trap_keywords"]


def test_behavioral_signals_present():
    ids = {s["id"] for s in behavioral_signals(AMSTERDAM)}
    assert "respects_no_early_meetings" in ids
    assert "adopts_swim_change" in ids


# --- multilingual probes -----------------------------------------------------


def test_two_multilingual_recall_probes_present():
    recalls = {p["id"]: p for p in memory_probes(AMSTERDAM) if p["kind"] == PROBE_RECALL}
    # Language-preference probe (English default) + Dutch-email comprehension probe.
    assert "recall_default_language" in recalls
    assert "recall_dutch_school_closure" in recalls
    for pid in ("recall_default_language", "recall_dutch_school_closure"):
        assert recalls[pid]["expected"], f"{pid} needs non-empty expected keywords"


def test_dutch_email_is_in_the_stream_with_action_content():
    stream = replay_events(AMSTERDAM)
    dutch = [
        data for _day, kind, data in stream
        if kind == "email" and "gesloten" in data.get("body", "").lower()
    ]
    assert dutch, "the Dutch school-closure email must be in the event stream"


def test_swim_change_is_delivered_as_an_event():
    stream = replay_events(AMSTERDAM)
    moving = [
        data for _day, kind, data in stream
        if kind == "email" and "sunday" in data.get("body", "").lower()
    ]
    assert moving, "the swim-moves-to-Sunday email must be in the event stream"


def test_replay_is_deterministic():
    assert replay_events(AMSTERDAM) == replay_events(AMSTERDAM)


# --- PII guard (anti-leak) ---------------------------------------------------


def test_all_persona_emails_use_test_domain():
    """Every email address in the persona ends with `.test` — no real inboxes."""
    text = _PERSONA_FILE.read_text()
    addrs = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", text)
    assert addrs, "expected some email addresses in the persona"
    bad = [a for a in addrs if not a.endswith(".test")]
    assert not bad, f"non-.test email addresses found (possible real PII): {bad}"


def test_persona_has_no_real_looking_contact_identifiers():
    """Pattern guard: no real email TLDs, phone numbers, or NL postal codes.

    Checks shapes, not a real-name denylist (embedding real names here would leak
    them). Catches the most common ways real PII slips in via copy-paste.
    """
    text = _PERSONA_FILE.read_text()
    real_domains = re.findall(r"@[A-Za-z0-9.-]+\.(?:com|net|org|nl|io|co)\b", text)
    assert not real_domains, f"real-looking email domains found: {real_domains}"
    # NL postal code: 4 digits + optional space + 2 letters (e.g. "1071 AB").
    postal = re.findall(r"\b\d{4}\s?[A-Z]{2}\b", text)
    assert not postal, f"NL-postal-code-shaped strings found: {postal}"
    # Phone-ish: 7+ consecutive digits, or +31 country code.
    phones = re.findall(r"\+31\b|\d{7,}", text)
    assert not phones, f"phone-number-shaped strings found: {phones}"
