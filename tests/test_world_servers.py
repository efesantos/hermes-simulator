"""Tests for the mock-world servers' HTTP transport (U1).

The transport-selection logic is unit-tested pure; the actual bind+serve is
exercised by spawning a real server subprocess (no model, no hermes — just the
project venv python) and driving it with the MCP streamable-http client.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from simulator.world import _server_common
from simulator.world.state import WorldState

_SEED = {
    "events": [
        {"title": "Soccer practice", "start": "2026-07-02T16:00:00",
         "end": "2026-07-02T17:00:00", "location": "Field 3"},
    ],
}


# --- transport selection (pure) ----------------------------------------------


def test_chosen_transport_is_http_when_port_set(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_PORT", "8123")
    assert _server_common.chosen_transport() == "streamable-http"


def test_chosen_transport_is_stdio_without_port(monkeypatch):
    monkeypatch.delenv("HERMES_MCP_PORT", raising=False)
    assert _server_common.chosen_transport() == "stdio"


def test_make_server_binds_host_port_when_configured(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_PORT", "8123")
    monkeypatch.setenv("HERMES_MCP_HOST", "127.0.0.1")
    mcp = _server_common.make_server("mockcal")
    assert mcp.settings.port == 8123
    assert mcp.settings.host == "127.0.0.1"
    # Default served path is /mcp (what the gateway registers and polls).
    assert mcp.settings.streamable_http_path == "/mcp"


def test_make_server_no_port_falls_back_to_stdio(monkeypatch):
    # Edge: no port configured → stdio mode (server still constructs fine).
    monkeypatch.delenv("HERMES_MCP_PORT", raising=False)
    mcp = _server_common.make_server("mockcal")
    assert _server_common.chosen_transport() == "stdio"
    assert mcp.name == "mockcal"


# --- live-on-the-loopback subprocess (no model needed) -----------------------


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_listening(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


@contextmanager
def _serve(module: str, world_db: Path, port: int):
    env = dict(os.environ)
    env["HERMES_MCP_PORT"] = str(port)
    env["HERMES_MCP_HOST"] = "127.0.0.1"
    proc = subprocess.Popen(
        [sys.executable, "-m", module, str(world_db)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


async def _list_tools_and_events(url: str):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("list_events", {})
            return {t.name for t in tools.tools}, result


def test_http_server_serves_seeded_world(tmp_path: Path):
    # Happy path: a server on a free port answers ListTools + list_events with the
    # events seeded into the bound world.db.
    world_db = tmp_path / "world.db"
    w = WorldState.create(world_db)
    w.seed(_SEED)
    w.close()

    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    with _serve("simulator.world.calendar_server", world_db, port):
        assert _wait_listening(port), "calendar server never bound its port"
        tool_names, result = asyncio.run(_list_tools_and_events(url))

    assert "list_events" in tool_names
    assert "create_event" in tool_names
    # The seeded event is visible through the tool against the bound world.db.
    assert "Soccer practice" in str(result.structuredContent or result.content)


def test_http_server_on_occupied_port_fails_fast(tmp_path: Path):
    # Error path: a second server on an already-bound port surfaces a clear startup
    # failure (non-zero exit) instead of hanging silently.
    world_db = tmp_path / "world.db"
    WorldState.create(world_db).close()

    port = _free_port()
    with _serve("simulator.world.calendar_server", world_db, port):
        assert _wait_listening(port), "first server never bound its port"
        with _serve("simulator.world.calendar_server", world_db, port) as second:
            # The loser must exit (not block forever); a clean bind would stay up.
            exit_code = second.wait(timeout=20)
    assert exit_code is not None and exit_code != 0
