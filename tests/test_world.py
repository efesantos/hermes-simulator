"""Tests for the mock world store and MCP servers (U2)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from simulator.world.state import WorldState, _overlaps


@pytest.fixture
def world(tmp_path: Path) -> WorldState:
    return WorldState.create(tmp_path / "world.db")


_SEED = {
    "contacts": [
        {"name": "Sam", "email": "sam@home.test", "relation": "spouse"},
        {"name": "Lincoln Elementary", "email": "office@lincoln.test", "relation": "school"},
    ],
    "events": [
        {"title": "Soccer practice", "start": "2026-07-02T16:00:00",
         "end": "2026-07-02T17:00:00", "location": "Field 3"},
    ],
    "emails": [
        {"from_addr": "office@lincoln.test", "to_addr": "me@example.com",
         "subject": "Field trip permission", "body": "Please reply by Friday.",
         "timestamp": "2026-07-01T08:00:00"},
    ],
}


# --- store basics ------------------------------------------------------------


def test_seed_and_inspect_round_trip(world: WorldState):
    world.seed(_SEED)
    snap = world.inspect()
    assert {c["name"] for c in snap["contacts"]} == {"Sam", "Lincoln Elementary"}
    assert snap["events"][0]["title"] == "Soccer practice"
    assert snap["events"][0]["attendees"] == []  # JSON decoded to a list
    assert snap["emails"][0]["folder"] == "inbox"


def test_create_event_lands_in_store_and_inspect_sees_it(world: WorldState):
    # Mirrors the happy-path scenario: an agent 'create_event' is visible out-of-band.
    eid = world.create_event(
        title="Dentist", start="2026-07-03T09:00:00", end="2026-07-03T09:30:00"
    )
    assert eid > 0
    assert world.inspect()["events"][-1]["title"] == "Dentist"


def test_send_email_goes_to_sent_folder(world: WorldState):
    world.send_email(
        to_addr="sam@home.test", subject="Pickup?", body="Can you grab Mia?",
        timestamp="2026-07-02T12:00:00",
    )
    assert world.list_emails(folder="sent")[0]["subject"] == "Pickup?"
    assert world.list_emails(folder="inbox") == []


def test_get_contact_exact_and_substring(world: WorldState):
    world.seed(_SEED)
    assert world.get_contact("sam")["email"] == "sam@home.test"  # case-insensitive
    assert world.get_contact("Lincoln")["relation"] == "school"  # substring
    assert world.get_contact("nobody") is None


# --- conflict detection (ground truth for double-booking) --------------------


def test_overlap_predicate():
    assert _overlaps("2026-07-02T16:00:00", "2026-07-02T17:00:00",
                     "2026-07-02T16:30:00", "2026-07-02T17:30:00")
    # Touching at the boundary is not an overlap (half-open ranges).
    assert not _overlaps("2026-07-02T16:00:00", "2026-07-02T17:00:00",
                         "2026-07-02T17:00:00", "2026-07-02T18:00:00")


def test_find_conflicts_flags_double_booking(world: WorldState):
    world.seed(_SEED)  # soccer 16:00-17:00
    conflicts = world.find_conflicts("2026-07-02T16:30:00", "2026-07-02T17:30:00")
    assert len(conflicts) == 1
    assert conflicts[0]["title"] == "Soccer practice"
    # A non-overlapping slot is clear.
    assert world.find_conflicts("2026-07-02T18:00:00", "2026-07-02T19:00:00") == []


def test_list_events_filtered_by_day(world: WorldState):
    world.create_event(title="A", start="2026-07-02T10:00:00", end="2026-07-02T11:00:00")
    world.create_event(title="B", start="2026-07-03T10:00:00", end="2026-07-03T11:00:00")
    assert [e["title"] for e in world.list_events(day="2026-07-02")] == ["A"]


# --- AE1: divergent agent action, identical exogenous seed -------------------


def test_two_tracks_diverge_but_share_seeded_events(tmp_path: Path):
    # Two tracks get the SAME seeded world, then act differently. Their stores
    # diverge on the agent's writes but both retain the identical scripted event.
    a = WorldState.create(tmp_path / "a" / "world.db")
    b = WorldState.create(tmp_path / "b" / "world.db")
    a.seed(_SEED)
    b.seed(_SEED)

    a.create_event(title="A-only meeting", start="2026-07-02T09:00:00",
                   end="2026-07-02T09:30:00")
    b.send_email(to_addr="sam@home.test", subject="B-only", body="hi",
                 timestamp="2026-07-02T09:00:00")

    a_titles = {e["title"] for e in a.inspect()["events"]}
    b_titles = {e["title"] for e in b.inspect()["events"]}
    assert "A-only meeting" in a_titles and "A-only meeting" not in b_titles
    # The scripted exogenous event is present and identical in both tracks.
    assert "Soccer practice" in a_titles and "Soccer practice" in b_titles


# --- isolation / security ----------------------------------------------------


@pytest.mark.parametrize(
    "module_name, expected_tools",
    [
        ("simulator.world.email_server",
         {"list_emails", "get_email", "search_emails", "send_email", "mark_read"}),
        ("simulator.world.calendar_server",
         {"list_events", "get_event", "find_conflicts", "create_event", "delete_event"}),
        ("simulator.world.contacts_server",
         {"list_contacts", "get_contact"}),
    ],
)
def test_servers_expose_only_world_tools_never_inspect(
    tmp_path: Path, monkeypatch, module_name: str, expected_tools: set[str]
):
    # The out-of-band inspect() must NOT be reachable as a tool. Import each
    # server with argv pointing at a real db, then list its registered tools.
    db = tmp_path / "world.db"
    WorldState.create(db).close()
    monkeypatch.setattr(sys, "argv", ["server", str(db)])
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    try:
        names = {t.name for t in module.mcp._tool_manager.list_tools()}
        assert names == expected_tools
        assert "inspect" not in names
        assert "seed" not in names
    finally:
        sys.modules.pop(module_name, None)


def test_world_db_lives_outside_any_hermes_home(tmp_path: Path):
    # Design invariant: the store path the runner uses is not under HERMES_HOME,
    # so no filesystem tool inside the agent's env can read it. We assert the
    # contract that registration passes an explicit external path (not derived
    # from the home) — here by constructing both and checking non-containment.
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    world_db = tmp_path / "world" / "world.db"
    WorldState.create(world_db).close()
    assert hermes_home not in world_db.parents


# --- live (real hermes + Ollama; run with `-m live`) -------------------------


@pytest.mark.live
def test_live_agent_operates_world_grader_reads_out_of_band(tmp_path: Path):
    """U2 verification: agent acts through tools; grader sees end-state out-of-band."""
    from simulator.config import LOCAL_OLLAMA, CandidateModel
    from simulator.harness import Harness
    from simulator.world.registration import register_world

    home = tmp_path / "home"
    world_db = tmp_path / "world" / "world.db"  # outside the home
    WorldState.create(world_db).close()

    model = CandidateModel(
        id="qwen3.6:latest", hosting_profile=LOCAL_OLLAMA, context_length=65_536
    )
    h = Harness(home, model, timeout=400)
    h.setup()
    reg = register_world(h, str(world_db), python_exe=sys.executable)
    assert all(r.ok for r in reg.values())

    res = h.run_oneshot(
        "Using your calendar tools, create an event titled 'Dentist' on "
        "2026-07-03 from 09:00 to 09:30, then confirm it."
    )
    assert res.ok
    events = WorldState(world_db).inspect()["events"]
    assert any(e["title"] == "Dentist" for e in events)
