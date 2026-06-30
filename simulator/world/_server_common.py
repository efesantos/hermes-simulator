"""Shared bootstrap for the mock-world MCP servers.

Each server can run two ways:

- **streamable-http** (the benchmark path): the :class:`~simulator.world.gateway`
  starts the server bound to ``127.0.0.1:<port>`` once per track and registers it
  with Hermes *by URL*. Transport config (host, port, clock-file) arrives via env
  the gateway sets; the world-store path still arrives as ``sys.argv[1]``.
- **stdio** (ad-hoc/manual use): launched by Hermes as a stdio subprocess with no
  port configured, so ``world_from_argv()`` + ``mcp.run()`` behave as before.

The simulated clock (what "now" means inside the persona day loop) is read here so
the three servers stay consistent. A persistent HTTP server captures process env
once at startup, so the clock can no longer ride on a per-day env var; instead the
runner writes the current simulated time to a per-track *clock file* whose path is
passed at startup, and :func:`sim_now` re-reads it per call (see U2).
"""

from __future__ import annotations

import os
import sys
import tempfile

from mcp.server.fastmcp import FastMCP

from .state import WorldState

# Used only if neither a clock file nor HERMES_SIM_NOW is set (e.g. manual runs).
_FALLBACK_NOW = "2026-01-01T09:00:00"

# Env vars the gateway sets when launching a server over HTTP. Absent => stdio.
# Public so the gateway (which sets them) and the server (which reads them) share
# one definition — the names are the contract between the two.
HOST_ENV = "HERMES_MCP_HOST"
PORT_ENV = "HERMES_MCP_PORT"
# Path to the per-track sidecar file the runner stamps with each day's clock (U2).
CLOCK_FILE_ENV = "HERMES_SIM_NOW_FILE"

DEFAULT_HOST = "127.0.0.1"


def world_from_argv() -> WorldState:
    """Open the world store whose path was passed as the first CLI argument."""
    if len(sys.argv) < 2:
        raise SystemExit("usage: <server>.py <world_db_path>")
    return WorldState(sys.argv[1])


def _port_configured() -> bool:
    """True when an HTTP port is configured (and so the server runs over HTTP)."""
    return bool(os.environ.get(PORT_ENV))


def make_server(name: str) -> FastMCP:
    """Construct a ``FastMCP`` for ``name``, binding host/port when configured.

    With ``HERMES_MCP_PORT`` set the server is built to listen on
    ``HERMES_MCP_HOST``:<port> (default path ``/mcp``); :func:`run_server` then
    runs it as ``streamable-http``. With no port it is a plain stdio server.
    """
    if _port_configured():
        return FastMCP(
            name,
            host=os.environ.get(HOST_ENV, DEFAULT_HOST),
            port=int(os.environ[PORT_ENV]),
        )
    return FastMCP(name)


def chosen_transport() -> str:
    """``'streamable-http'`` when a port is configured, else ``'stdio'``."""
    return "streamable-http" if _port_configured() else "stdio"


def run_server(mcp: FastMCP) -> None:
    """Run ``mcp`` over the configured transport (HTTP when a port is set)."""
    mcp.run(transport=chosen_transport())


def sim_now() -> str:
    """The simulated wall-clock time, as an ISO string, for stamping writes.

    Reads the per-track clock file (path in ``HERMES_SIM_NOW_FILE``) on *every*
    call so the runner's per-day updates land in a long-lived server without a
    restart. Falls back to the ``HERMES_SIM_NOW`` env var, then ``_FALLBACK_NOW``,
    so stdio/manual use keeps working.
    """
    clock_file = os.environ.get(CLOCK_FILE_ENV)
    if clock_file:
        try:
            with open(clock_file, encoding="utf-8") as fh:
                stamped = fh.read().strip()
        except OSError:  # not written yet / unreadable → fall back below
            stamped = ""
        if stamped:
            return stamped
    return os.environ.get("HERMES_SIM_NOW", _FALLBACK_NOW)


def write_clock(path: str | os.PathLike[str], value: str) -> None:
    """Atomically stamp the per-track clock file with ``value`` (an ISO time).

    Written by the runner/gateway before each day so a long-lived server reflects
    the current simulated time. The write is atomic (temp file + ``os.replace``)
    so a concurrent :func:`sim_now` read can never observe a torn half-write.
    """
    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(value)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
