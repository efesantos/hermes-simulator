"""Shared bootstrap for the mock-world MCP servers.

Each server is launched by Hermes as a stdio subprocess:

    hermes mcp add mockemail --command <python> --args <email_server.py> <world.db>

so the store path arrives as ``sys.argv[1]``. The simulated clock (what "now"
means inside the persona day loop) arrives via the ``HERMES_SIM_NOW`` env var the
runner sets per day. Both are read here so the three servers stay consistent.
"""

from __future__ import annotations

import os
import sys

from .state import WorldState

# Used only if the runner hasn't set a simulated clock (e.g. ad-hoc manual runs).
_FALLBACK_NOW = "2026-01-01T09:00:00"


def world_from_argv() -> WorldState:
    """Open the world store whose path was passed as the first CLI argument."""
    if len(sys.argv) < 2:
        raise SystemExit("usage: <server>.py <world_db_path>")
    return WorldState(sys.argv[1])


def sim_now() -> str:
    """The simulated wall-clock time, as an ISO string, for stamping writes."""
    return os.environ.get("HERMES_SIM_NOW", _FALLBACK_NOW)
