"""Tests for the simulated counterparty (U6)."""

from __future__ import annotations

import pytest

from simulator.counterparty import (
    CounterpartyConfig,
    LLMCounterparty,
    ScriptedCounterparty,
    _build_messages,
    _resolve_brief,
)
from simulator.scenarios.personas.dana import PERSONA as DANA
from simulator.scenarios.types import Counterparty


def _email(to_addr="sam@home.test", subject="Soccer pickup?", body="Which day is practice?"):
    return {"from_addr": "dana@home.test", "to_addr": to_addr,
            "subject": subject, "body": body}


# --- protocol conformance ----------------------------------------------------


def test_both_implementations_satisfy_the_protocol():
    assert isinstance(ScriptedCounterparty({}), Counterparty)
    assert isinstance(LLMCounterparty(chat_fn=lambda *a, **k: "hi"), Counterparty)


# --- recipient resolution ----------------------------------------------------


def test_resolve_brief_matches_seeded_contact_email():
    resolved = _resolve_brief(DANA, "sam@home.test")
    assert resolved is not None
    name, brief = resolved
    assert name == "Sam"
    assert brief["relation"] == "spouse"


def test_resolve_brief_none_for_unbriefed_or_unknown():
    # Dr. Patel is a seeded contact but has no counterparty brief -> silence.
    assert _resolve_brief(DANA, "frontdesk@pateldental.test") is None
    assert _resolve_brief(DANA, "stranger@nowhere.test") is None


# --- partial observability ---------------------------------------------------


def test_prompt_contains_only_the_email_never_tool_calls():
    msgs = _build_messages("Sam", DANA.counterparty_brief["Sam"], _email())
    blob = " ".join(m["content"] for m in msgs)
    assert "Which day is practice?" in blob  # the email body is visible
    # No agent internals leak in (there are none to leak — assert the obvious guard).
    for forbidden in ("tool_call", "create_event", "send_email", "function"):
        assert forbidden not in blob


def test_llm_counterparty_only_sees_the_email(monkeypatch):
    seen = {}

    def spy_chat(messages, *, temperature, seed):
        seen["messages"] = messages
        return "Sounds good, I'll handle pickup."

    cp = LLMCounterparty(chat_fn=spy_chat)
    cp.reply(_email(body="Mia's practice moved to Wednesday — can you do pickup?"),
             DANA, sim_now="2026-07-08T09:00:00")
    blob = " ".join(m["content"] for m in seen["messages"])
    assert "Wednesday" in blob
    assert "create_event" not in blob and "tool" not in blob.lower()


# --- determinism -------------------------------------------------------------


def test_temperature_and_seed_are_passed_through():
    captured = {}

    def spy_chat(messages, *, temperature, seed):
        captured["temperature"] = temperature
        captured["seed"] = seed
        return "ok"

    cp = LLMCounterparty(CounterpartyConfig(temperature=0.0, seed=7), chat_fn=spy_chat)
    cp.reply(_email(), DANA, sim_now="2026-07-08T09:00:00")
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 7


def test_same_input_yields_same_reply_with_deterministic_chat():
    # A deterministic chat_fn stands in for a temp-0/seeded model.
    cp = LLMCounterparty(chat_fn=lambda messages, **k: "fixed reply")
    r1 = cp.reply(_email(), DANA, sim_now="2026-07-08T09:00:00")
    r2 = cp.reply(_email(), DANA, sim_now="2026-07-08T09:00:00")
    assert r1 == r2


# --- reply shaping -----------------------------------------------------------


def test_reply_is_addressed_back_and_threaded():
    cp = LLMCounterparty(chat_fn=lambda *a, **k: "Yes, I can.")
    reply = cp.reply(_email(subject="Pickup?"), DANA, sim_now="2026-07-08T09:00:00")
    assert reply["from_addr"] == "sam@home.test"  # replies from where mail was sent
    assert reply["to_addr"] == "dana@home.test"
    assert reply["subject"] == "Re: Pickup?"
    assert reply["timestamp"] == "2026-07-08T09:00:00"


def test_reply_does_not_double_prefix_re():
    cp = LLMCounterparty(chat_fn=lambda *a, **k: "ok")
    reply = cp.reply(_email(subject="Re: Pickup?"), DANA, sim_now="2026-07-08T09:00:00")
    assert reply["subject"] == "Re: Pickup?"


# --- scripted beats ----------------------------------------------------------


def test_scripted_counterparty_canned_reply_and_silence():
    cp = ScriptedCounterparty({"sam@home.test": "On it."})
    r = cp.reply(_email(), DANA, sim_now="2026-07-08T09:00:00")
    assert r["body"] == "On it."
    # No script for the coach -> silence.
    assert cp.reply(_email(to_addr="rivera@youthsoccer.test"), DANA,
                    sim_now="2026-07-08T09:00:00") is None


def test_scripted_override_bypasses_the_model():
    def boom(*a, **k):
        raise AssertionError("model should not be called for a scripted recipient")

    cp = LLMCounterparty(chat_fn=boom, scripted={"sam@home.test": "Pinned beat."})
    r = cp.reply(_email(), DANA, sim_now="2026-07-08T09:00:00")
    assert r["body"] == "Pinned beat."


def test_unknown_recipient_is_silent_not_an_error():
    cp = LLMCounterparty(chat_fn=lambda *a, **k: "should not be used")
    assert cp.reply(_email(to_addr="stranger@nowhere.test"), DANA,
                    sim_now="2026-07-08T09:00:00") is None


# --- live (real Ollama; run with `-m live`) ----------------------------------


@pytest.mark.live
def test_live_counterparty_in_character_and_seed_stable():
    """U6 verification: a coordination beat returns an in-character, seed-stable reply."""
    cp = LLMCounterparty(CounterpartyConfig(model="qwen3:8b", temperature=0.0, seed=7))
    email = _email(body="Mia's practice moved to Wednesday at 4pm. Can you grab her after?")
    r1 = cp.reply(email, DANA, sim_now="2026-07-08T09:00:00")
    r2 = cp.reply(email, DANA, sim_now="2026-07-08T09:00:00")
    assert r1["from_addr"] == "sam@home.test"
    assert r1["body"].strip()
    assert r1["body"] == r2["body"]  # temp 0 + fixed seed -> stable
