"""Mock contacts MCP server (FastMCP, stdio).

Exposes the contact list (spouse, school, kids, …) backing the coordination
tasks. Lookups are how the agent resolves a name to an email before sending mail.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ._server_common import world_from_argv

mcp = FastMCP("mockcontacts")
_world = world_from_argv()


@mcp.tool()
def list_contacts() -> list[dict]:
    """List every contact with their email, phone, and relationship."""
    return _world.list_contacts()


@mcp.tool()
def get_contact(name: str) -> dict | None:
    """Look up a contact by name (case-insensitive; falls back to substring match)."""
    return _world.get_contact(name)


if __name__ == "__main__":
    mcp.run()
