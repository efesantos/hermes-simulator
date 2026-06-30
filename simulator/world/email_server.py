"""Mock email MCP server (FastMCP).

Exposes inbox/sent tools backed by the shared world store. Registered into a
track's HERMES_HOME via ``hermes mcp add`` (by URL on the gateway path). The agent
reaches email *only* through these tools; the grader reads the same store
out-of-band. Runs over streamable-http when a port is configured, else stdio.
"""

from __future__ import annotations

from ._server_common import make_server, run_server, sim_now, world_from_argv

mcp = make_server("mockemail")
_world = world_from_argv()


@mcp.tool()
def list_emails(folder: str = "inbox", unread_only: bool = False) -> list[dict]:
    """List emails in a folder ('inbox' or 'sent'). Set unread_only to see only unread mail."""
    return _world.list_emails(folder=folder, unread_only=unread_only)


@mcp.tool()
def get_email(email_id: int) -> dict | None:
    """Fetch one email by id, including its full body. Returns null if not found."""
    return _world.get_email(email_id)


@mcp.tool()
def search_emails(query: str) -> list[dict]:
    """Search inbox and sent mail for a substring in the subject or body."""
    return _world.search_emails(query)


@mcp.tool()
def send_email(to_addr: str, subject: str, body: str) -> dict:
    """Send an email to a recipient. Records it in the sent folder and returns its id."""
    email_id = _world.send_email(
        to_addr=to_addr, subject=subject, body=body, timestamp=sim_now()
    )
    return {"id": email_id, "status": "sent"}


@mcp.tool()
def mark_read(email_id: int) -> dict:
    """Mark an email as read."""
    ok = _world.mark_read(email_id)
    return {"id": email_id, "marked_read": ok}


if __name__ == "__main__":
    run_server(mcp)
