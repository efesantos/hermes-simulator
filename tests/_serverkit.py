"""Shared helpers for spawning real mock-world servers in tests.

Not a ``test_*`` module, so pytest does not collect it. These helpers spawn the
servers as real loopback HTTP processes (no model, no hermes — just the project
venv python) and drive them with the MCP streamable-http client.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def free_port() -> int:
    """An ephemeral loopback port, released immediately (small TOCTOU window)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_listening(port: int, timeout: float = 20.0) -> bool:
    """Block until something accepts on ``port`` (or the timeout elapses)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


@contextmanager
def serve(module: str, world_db: Path, port: int, *, extra_env: dict | None = None):
    """Spawn ``python -m <module> <world_db>`` as an HTTP server; reap on exit."""
    env = dict(os.environ)
    env["HERMES_MCP_PORT"] = str(port)
    env["HERMES_MCP_HOST"] = "127.0.0.1"
    if extra_env:
        env.update(extra_env)
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


async def call_tool(url: str, tool: str, args: dict):
    """Initialize an MCP session against ``url`` and call one tool."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool(tool, args)
            return {t.name for t in tools.tools}, result
