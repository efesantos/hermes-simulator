"""Tests for the per-day simulated clock via a sidecar file (U2).

A persistent HTTP server captures process env once at startup, so the old per-day
``HERMES_SIM_NOW`` env injection can't reach it. Instead the runner stamps a
per-track clock file that ``sim_now()`` re-reads on every call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from simulator.world import _server_common
from simulator.world._server_common import sim_now, write_clock
from simulator.world.state import WorldState

from ._serverkit import call_tool, free_port, serve, wait_listening


@pytest.fixture(autouse=True)
def _clear_clock_env(monkeypatch):
    """Keep clock env hermetic per test."""
    monkeypatch.delenv("HERMES_SIM_NOW", raising=False)
    monkeypatch.delenv("HERMES_SIM_NOW_FILE", raising=False)


# --- in-process read path ----------------------------------------------------


def test_sim_now_reads_clock_file_and_reflects_rewrites(tmp_path: Path, monkeypatch):
    clock = tmp_path / "sim_now"
    monkeypatch.setenv("HERMES_SIM_NOW_FILE", str(clock))

    write_clock(clock, "2026-07-02T09:00:00")
    assert sim_now() == "2026-07-02T09:00:00"
    # Rewriting (the next simulated day) lands on the very next call — no restart.
    write_clock(clock, "2026-07-03T09:00:00")
    assert sim_now() == "2026-07-03T09:00:00"


def test_sim_now_falls_back_to_env_then_default(tmp_path: Path, monkeypatch):
    # No clock file configured → HERMES_SIM_NOW wins.
    monkeypatch.setenv("HERMES_SIM_NOW", "2026-05-05T12:00:00")
    assert sim_now() == "2026-05-05T12:00:00"
    # Neither configured → the module's fallback.
    monkeypatch.delenv("HERMES_SIM_NOW", raising=False)
    assert sim_now() == _server_common._FALLBACK_NOW


def test_sim_now_missing_clock_file_falls_back(tmp_path: Path, monkeypatch):
    # Edge: a configured-but-absent clock file must not raise; it falls back.
    monkeypatch.setenv("HERMES_SIM_NOW_FILE", str(tmp_path / "does_not_exist"))
    monkeypatch.setenv("HERMES_SIM_NOW", "2026-05-05T12:00:00")
    assert sim_now() == "2026-05-05T12:00:00"


def test_empty_clock_file_falls_back(tmp_path: Path, monkeypatch):
    clock = tmp_path / "sim_now"
    write_clock(clock, "")  # nothing stamped yet
    monkeypatch.setenv("HERMES_SIM_NOW_FILE", str(clock))
    monkeypatch.setenv("HERMES_SIM_NOW", "2026-05-05T12:00:00")
    assert sim_now() == "2026-05-05T12:00:00"


# --- the runner↔server channel (live server, no model) -----------------------


def test_running_server_stamps_each_days_clock_without_restart(tmp_path: Path):
    # U2 verification: across two simulated days, ONE long-lived server stamps its
    # writes with each day's time — proving per-call re-read of the clock file.
    world_db = tmp_path / "world.db"
    WorldState.create(world_db).close()
    clock = tmp_path / "sim_now"

    port = free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    with serve(
        "simulator.world.email_server", world_db, port,
        extra_env={"HERMES_SIM_NOW_FILE": str(clock)},
    ):
        assert wait_listening(port), "email server never bound its port"
        # Day 1: stamp, then the agent (us) sends mail.
        write_clock(clock, "2026-07-02T09:00:00")
        asyncio.run(call_tool(url, "send_email", {
            "to_addr": "sam@home.test", "subject": "day1", "body": "x"}))
        # Day 2: re-stamp the SAME file; no restart.
        write_clock(clock, "2026-07-03T09:00:00")
        asyncio.run(call_tool(url, "send_email", {
            "to_addr": "sam@home.test", "subject": "day2", "body": "y"}))

    sent = {e["subject"]: e["timestamp"] for e in WorldState(world_db).list_emails(folder="sent")}
    assert sent["day1"].startswith("2026-07-02")
    assert sent["day2"].startswith("2026-07-03")
