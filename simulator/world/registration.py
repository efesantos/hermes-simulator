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
    """Register email/calendar/contacts servers into ``harness``'s home (stdio).

    Each server is launched as ``<python> -m <module> <world_db>``. Returns the
    per-server ``hermes mcp add`` results so the caller can verify each enabled.

    This is the **stdio** path, kept for ad-hoc/manual use. The benchmark runner
    uses :func:`register_world_urls` against a persistent :class:`WorldGateway`
    instead, so tool discovery doesn't race a per-run subprocess boot.
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


def register_world_urls(
    harness: Harness, urls: dict[str, str],
) -> dict[str, HarnessResult]:
    """Register already-running world servers into ``harness``'s home by URL.

    ``urls`` is the gateway's name→URL map. Returns the per-server
    ``hermes mcp add --url`` results so the caller can verify each enabled.
    """
    return {name: harness.add_remote_mcp_server(name, url) for name, url in urls.items()}
