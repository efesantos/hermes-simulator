"""Tests for the per-track world gateway lifecycle (U4).

These spawn real loopback servers (no model, no hermes) to prove start/readiness/
teardown for real; the runner wiring (U5) uses a fake gateway so the fast funnel
tests never spawn anything.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest

from simulator.world.gateway import GatewayError, WorldGateway, free_ports
from simulator.world.state import WorldState

from ._serverkit import call_tool, wait_listening


def _seeded_world(tmp_path: Path) -> Path:
    world_db = tmp_path / "world.db"
    w = WorldState.create(world_db)
    w.seed({"events": [{"title": "Soccer practice", "start": "2026-07-02T16:00:00",
                        "end": "2026-07-02T17:00:00"}]})
    w.close()
    return world_db


def _port_listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# --- happy path --------------------------------------------------------------


def test_gateway_starts_three_reachable_servers_and_tears_down(tmp_path: Path):
    world_db = _seeded_world(tmp_path)
    clock = tmp_path / "sim_now"

    with WorldGateway(world_db, clock) as gw:
        assert set(gw.urls) == {"mockcal", "mockemail", "mockcontacts"}
        procs = list(gw._procs.values())
        assert len(procs) == 3
        # Each URL is reachable; the calendar one serves the seeded world.
        for url in gw.urls.values():
            port = int(url.rsplit(":", 1)[1].split("/")[0])
            assert wait_listening(port)
        _, result = asyncio.run(call_tool(gw.urls["mockcal"], "list_events", {}))
        assert "Soccer practice" in str(result.structuredContent or result.content)

    # After the context exits, no server processes remain.
    assert all(p.poll() is not None for p in procs)


def test_set_clock_stamps_the_clock_file(tmp_path: Path):
    world_db = _seeded_world(tmp_path)
    clock = tmp_path / "sim_now"
    gw = WorldGateway(world_db, clock, port_picker=lambda n: free_ports(n))
    gw.set_clock("2026-07-02T09:00:00")
    assert clock.read_text().strip() == "2026-07-02T09:00:00"


def test_world_gateway_satisfies_the_gateway_protocol(tmp_path: Path):
    from simulator.world.gateway import Gateway

    gw = WorldGateway(tmp_path / "world.db", tmp_path / "sim_now")
    assert isinstance(gw, Gateway)  # runtime-checkable structural match


# --- readiness failure & teardown of already-started -------------------------


def test_readiness_timeout_raises_and_tears_down_started(tmp_path: Path):
    world_db = _seeded_world(tmp_path)
    clock = tmp_path / "sim_now"

    # Occupy a port so the second server can't bind; the first (free) one comes up.
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    occupied = blocker.getsockname()[1]
    good = free_ports(1)[0]
    try:
        gw = WorldGateway(
            world_db, clock,
            servers={"a": "simulator.world.calendar_server",
                     "b": "simulator.world.calendar_server"},
            port_picker=lambda n: [good, occupied],
            readiness_timeout=5,
        )
        with pytest.raises(GatewayError):
            gw.start()
        # The already-started "a" server was torn down (its port is free again).
        assert not _port_listening(good)
        assert gw._procs == {}
    finally:
        blocker.close()


# --- cleanup on an exception inside the context ------------------------------


def test_exception_in_context_still_terminates_all_children(tmp_path: Path):
    world_db = _seeded_world(tmp_path)
    clock = tmp_path / "sim_now"

    procs = []
    with pytest.raises(RuntimeError, match="boom"):
        with WorldGateway(world_db, clock) as gw:
            procs = list(gw._procs.values())
            assert all(p.poll() is None for p in procs)  # all alive mid-context
            raise RuntimeError("boom")

    assert procs and all(p.poll() is not None for p in procs)  # all reaped


# --- free-port selection -----------------------------------------------------


def test_free_ports_are_distinct_and_avoid_occupied():
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    occupied = blocker.getsockname()[1]
    try:
        ports = free_ports(3)
        assert len(ports) == 3
        assert len(set(ports)) == 3          # distinct
        assert occupied not in ports         # occupied port avoided
    finally:
        blocker.close()
