"""Wire the three mock-world servers into a Hermes home.

The servers use package-relative imports, so they must be launched as modules
(``python -m simulator.world.email_server <world_db>``) by an interpreter that
can import the ``simulator`` package — i.e. *this* project's venv python, which
also carries the ``mcp`` SDK. That interpreter is normally ``sys.executable``.
"""

from __future__ import annotations

import sys

from ..harness import Harness, HarnessResult

# server registration name -> module to run as `python -m <module>`
WORLD_SERVERS: dict[str, str] = {
    "mockemail": "simulator.world.email_server",
    "mockcal": "simulator.world.calendar_server",
    "mockcontacts": "simulator.world.contacts_server",
}


def register_world(
    harness: Harness,
    world_db: str,
    *,
    python_exe: str | None = None,
) -> dict[str, HarnessResult]:
    """Register email/calendar/contacts servers into ``harness``'s home.

    Each server is launched as ``<python> -m <module> <world_db>``. Returns the
    per-server ``hermes mcp add`` results so the caller can verify each enabled.
    """
    python_exe = python_exe or sys.executable
    results: dict[str, HarnessResult] = {}
    for name, module in WORLD_SERVERS.items():
        results[name] = harness.add_mcp_server(
            name,
            command=python_exe,
            args=["-m", module, world_db],
        )
    return results
