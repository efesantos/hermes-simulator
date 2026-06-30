"""Patch the local hermes install so MCP discovery can wait longer before turn 1.

Hermes' oneshot path waits only 0.75s for background MCP discovery
(`hermes_cli/mcp_startup.py:wait_for_mcp_discovery`). That is too short for the
three freshly-spawned Python mock-world servers, so fast API models get a
tool-less first turn and are falsely failed at the Stage-1 smoke. See
docs/solutions/integration-issues/api-path-mcp-cold-start.md.

This makes the wait honor ``HERMES_MCP_DISCOVERY_WAIT`` (the harness sets it to
20s). The join returns the instant discovery finishes, so a generous ceiling
costs nothing when servers boot quickly. Idempotent and reversible-by-reinstall;
re-run after upgrading/reinstalling hermes.

Usage:
    python scripts/patch_hermes_mcp_wait.py [path/to/hermes_cli/mcp_startup.py]
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_PATH = Path.home() / ".hermes" / "hermes-agent" / "hermes_cli" / "mcp_startup.py"

ORIGINAL = '''def wait_for_mcp_discovery(timeout: float = 0.75) -> None:
    """Briefly wait for background MCP discovery before the first tool snapshot."""
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=timeout)'''

PATCHED = '''def wait_for_mcp_discovery(timeout: float = 0.75) -> None:
    """Briefly wait for background MCP discovery before the first tool snapshot.

    ``HERMES_MCP_DISCOVERY_WAIT`` (seconds) overrides the default ceiling. The
    wait is a thread join, so it returns the instant discovery finishes — a larger
    value only raises the ceiling, never the typical cost. Needed when MCP servers
    are slow-booting subprocesses (e.g. fresh Python stdio servers) that would
    otherwise miss the first turn. Added for hermes-simulator.
    """
    import os

    _w = os.environ.get("HERMES_MCP_DISCOVERY_WAIT")
    if _w:
        try:
            timeout = float(_w)
        except ValueError:
            pass
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=timeout)'''


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"ERROR: hermes mcp_startup.py not found at {path}", file=sys.stderr)
        print("Pass the path explicitly: python scripts/patch_hermes_mcp_wait.py <path>", file=sys.stderr)
        return 2

    text = path.read_text()
    if "HERMES_MCP_DISCOVERY_WAIT" in text:
        print(f"Already patched: {path}")
        return 0
    if ORIGINAL not in text:
        print(f"ERROR: expected wait_for_mcp_discovery() body not found in {path}.", file=sys.stderr)
        print("Hermes may have changed; apply the env-var override by hand (see the docstring).", file=sys.stderr)
        return 3

    path.write_text(text.replace(ORIGINAL, PATCHED, 1))
    print(f"Patched {path}")
    print("wait_for_mcp_discovery now honors HERMES_MCP_DISCOVERY_WAIT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
