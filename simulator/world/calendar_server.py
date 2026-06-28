"""Mock calendar MCP server (FastMCP, stdio).

Exposes calendar tools backed by the shared world store. ``find_conflicts`` gives
double-booking tasks a ground-truth check the agent can use before creating an
event on an occupied slot.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ._server_common import world_from_argv

mcp = FastMCP("mockcal")
_world = world_from_argv()


@mcp.tool()
def list_events(day: str | None = None) -> list[dict]:
    """List calendar events. Pass day as 'YYYY-MM-DD' to filter to one day; omit for all."""
    return _world.list_events(day=day)


@mcp.tool()
def get_event(event_id: int) -> dict | None:
    """Fetch one event by id. Returns null if not found."""
    return _world.get_event(event_id)


@mcp.tool()
def find_conflicts(start: str, end: str) -> list[dict]:
    """List existing events that overlap a proposed time window.

    start/end are ISO datetimes, e.g. '2026-07-02T16:00:00'. Use this before
    creating an event to avoid double-booking.
    """
    return _world.find_conflicts(start, end)


@mcp.tool()
def create_event(
    title: str,
    start: str,
    end: str,
    location: str = "",
    attendees: list[str] | None = None,
    notes: str = "",
) -> dict:
    """Create a calendar event. start/end are ISO datetimes. Returns the new event id."""
    event_id = _world.create_event(
        title=title, start=start, end=end, location=location,
        attendees=attendees or [], notes=notes,
    )
    return {"id": event_id, "status": "created"}


@mcp.tool()
def delete_event(event_id: int) -> dict:
    """Delete a calendar event by id."""
    ok = _world.delete_event(event_id)
    return {"id": event_id, "deleted": ok}


if __name__ == "__main__":
    mcp.run()
