"""The single backing store for the mock world.

One SQLite file holds the inbox, the calendar, and the contact list. Three kinds
of caller touch it:

- the **MCP servers** (``email_server.py`` etc.) call the mutation/read methods
  to service agent tool calls;
- the **grader / runner** call :meth:`WorldState.inspect` and the read methods to
  check end-state — *out-of-band*, never through a tool;
- the **runner** calls :meth:`WorldState.seed` to lay down a scenario's world
  slice before a track runs.

SQLite (not JSON) because the three server subprocesses and the grader are
separate OS processes hitting the same file; WAL + a busy timeout gives safe
concurrent access a JSON file could not. The store deliberately lives *outside*
any ``HERMES_HOME`` so no filesystem tool inside the agent's environment can read
it directly — the only legitimate path in is an MCP tool.

Security boundary: this store contains only world data the agent is *meant* to
operate on. The ground-truth answer key (what the agent should have done, planted
facts for the memory exam) is **not** stored here — it lives in the persona files
and is read only by the grader. So there is nothing here for an agent to exploit
even if it could read the file.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT,
    from_addr TEXT NOT NULL,
    to_addr   TEXT NOT NULL,
    subject   TEXT NOT NULL,
    body      TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    folder    TEXT NOT NULL DEFAULT 'inbox',  -- inbox | sent
    unread    INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title     TEXT NOT NULL,
    start     TEXT NOT NULL,   -- ISO 8601, e.g. 2026-07-02T16:00:00
    end       TEXT NOT NULL,
    location  TEXT NOT NULL DEFAULT '',
    attendees TEXT NOT NULL DEFAULT '[]',  -- JSON list of emails/names
    notes     TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL,
    email    TEXT NOT NULL DEFAULT '',
    phone    TEXT NOT NULL DEFAULT '',
    relation TEXT NOT NULL DEFAULT '',  -- spouse | school | child | ...
    notes    TEXT NOT NULL DEFAULT ''
);
"""


@dataclass(frozen=True)
class Conflict:
    """Two events whose time ranges overlap."""

    a_id: int
    b_id: int
    a_title: str
    b_title: str


def _overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    """Half-open overlap: [a_start, a_end) intersects [b_start, b_end)."""
    sa, ea = datetime.fromisoformat(a_start), datetime.fromisoformat(a_end)
    sb, eb = datetime.fromisoformat(b_start), datetime.fromisoformat(b_end)
    return sa < eb and sb < ea


class WorldState:
    """Read/write access to one mock-world SQLite store.

    Cheap to construct (one per process is fine). Methods return plain
    dict/list/str/int so MCP servers can return them directly and tests can
    assert on them without ORM noise.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        # WAL + busy timeout: multiple server processes + the grader share this file.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")

    @classmethod
    def create(cls, path: str | Path) -> "WorldState":
        """Create (or open) the store and ensure the schema exists."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = cls(path)
        state.conn.executescript(_SCHEMA)
        state.conn.commit()
        return state

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "WorldState":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- seeding (runner only) ----------------------------------------------

    def seed(self, spec: dict[str, Iterable[dict[str, Any]]]) -> None:
        """Populate the store from a scenario world slice.

        ``spec`` maps ``"emails" | "events" | "contacts"`` to lists of row dicts.
        Unknown keys per row are ignored so scenarios can carry extra annotation
        without breaking the loader. Idempotent only in the sense that it appends;
        seed into a fresh store.
        """
        for email in spec.get("emails", []):
            self.add_email(
                from_addr=email["from_addr"],
                to_addr=email["to_addr"],
                subject=email["subject"],
                body=email.get("body", ""),
                timestamp=email["timestamp"],
                folder=email.get("folder", "inbox"),
                unread=bool(email.get("unread", True)),
                thread_id=email.get("thread_id"),
            )
        for event in spec.get("events", []):
            self.create_event(
                title=event["title"],
                start=event["start"],
                end=event["end"],
                location=event.get("location", ""),
                attendees=event.get("attendees", []),
                notes=event.get("notes", ""),
            )
        for contact in spec.get("contacts", []):
            self.add_contact(
                name=contact["name"],
                email=contact.get("email", ""),
                phone=contact.get("phone", ""),
                relation=contact.get("relation", ""),
                notes=contact.get("notes", ""),
            )

    # --- email ---------------------------------------------------------------

    def add_email(
        self,
        *,
        from_addr: str,
        to_addr: str,
        subject: str,
        body: str = "",
        timestamp: str,
        folder: str = "inbox",
        unread: bool = True,
        thread_id: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO emails (thread_id, from_addr, to_addr, subject, body, "
            "timestamp, folder, unread) VALUES (?,?,?,?,?,?,?,?)",
            (thread_id, from_addr, to_addr, subject, body, timestamp, folder, int(unread)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_emails(self, folder: str = "inbox", unread_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM emails WHERE folder = ?"
        params: list[Any] = [folder]
        if unread_only:
            sql += " AND unread = 1"
        sql += " ORDER BY timestamp"
        return [dict(r) for r in self.conn.execute(sql, params)]

    def get_email(self, email_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM emails WHERE id = ?", (email_id,)).fetchone()
        return dict(row) if row else None

    def search_emails(self, query: str) -> list[dict]:
        like = f"%{query}%"
        rows = self.conn.execute(
            "SELECT * FROM emails WHERE subject LIKE ? OR body LIKE ? ORDER BY timestamp",
            (like, like),
        )
        return [dict(r) for r in rows]

    def send_email(self, *, to_addr: str, subject: str, body: str, timestamp: str,
                   from_addr: str = "me@example.com") -> int:
        """Record an outgoing email in the 'sent' folder; returns its id."""
        return self.add_email(
            from_addr=from_addr, to_addr=to_addr, subject=subject, body=body,
            timestamp=timestamp, folder="sent", unread=False,
        )

    def max_email_id(self) -> int:
        """Highest email id, or 0 if empty — to detect mail created since a checkpoint."""
        row = self.conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM emails").fetchone()
        return int(row["m"])

    def emails_since(self, email_id: int, *, folder: str = "sent") -> list[dict]:
        """Emails in ``folder`` with id greater than ``email_id`` (oldest first)."""
        rows = self.conn.execute(
            "SELECT * FROM emails WHERE id > ? AND folder = ? ORDER BY id",
            (email_id, folder),
        )
        return [dict(r) for r in rows]

    def mark_read(self, email_id: int) -> bool:
        cur = self.conn.execute("UPDATE emails SET unread = 0 WHERE id = ?", (email_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # --- calendar ------------------------------------------------------------

    def create_event(
        self,
        *,
        title: str,
        start: str,
        end: str,
        location: str = "",
        attendees: list[str] | None = None,
        notes: str = "",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO events (title, start, end, location, attendees, notes) "
            "VALUES (?,?,?,?,?,?)",
            (title, start, end, location, json.dumps(attendees or []), notes),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_events(self, *, day: str | None = None) -> list[dict]:
        """List events, optionally filtered to a single ``YYYY-MM-DD`` day."""
        rows = self.conn.execute("SELECT * FROM events ORDER BY start")
        events = [self._event_row(r) for r in rows]
        if day is not None:
            events = [e for e in events if e["start"].startswith(day)]
        return events

    def get_event(self, event_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return self._event_row(row) if row else None

    def delete_event(self, event_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def find_conflicts(self, start: str, end: str) -> list[dict]:
        """Existing events overlapping the proposed ``[start, end)`` window.

        Ground truth for double-booking tasks: a correct agent calls this (or
        reasons equivalently) before creating an event on an occupied slot.
        """
        out = []
        for row in self.conn.execute("SELECT * FROM events ORDER BY start"):
            if _overlaps(start, end, row["start"], row["end"]):
                out.append(self._event_row(row))
        return out

    @staticmethod
    def _event_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["attendees"] = json.loads(d.get("attendees") or "[]")
        return d

    # --- contacts ------------------------------------------------------------

    def add_contact(self, *, name: str, email: str = "", phone: str = "",
                    relation: str = "", notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO contacts (name, email, phone, relation, notes) VALUES (?,?,?,?,?)",
            (name, email, phone, relation, notes),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_contacts(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM contacts ORDER BY name")]

    def get_contact(self, name: str) -> dict | None:
        """Case-insensitive lookup by exact name, then by substring."""
        row = self.conn.execute(
            "SELECT * FROM contacts WHERE lower(name) = lower(?)", (name,)
        ).fetchone()
        if row is None:
            row = self.conn.execute(
                "SELECT * FROM contacts WHERE name LIKE ? ORDER BY name LIMIT 1",
                (f"%{name}%",),
            ).fetchone()
        return dict(row) if row else None

    # --- out-of-band inspection (grader / runner only) -----------------------

    def inspect(self) -> dict[str, list[dict]]:
        """Full snapshot of world state for the grader.

        Deliberately **not** wrapped as an MCP tool: the agent must never be able
        to call this. Returns every table so a state-diff grader can assert on the
        end-state directly.
        """
        return {
            "emails": [dict(r) for r in self.conn.execute("SELECT * FROM emails ORDER BY id")],
            "events": [self._event_row(r) for r in self.conn.execute("SELECT * FROM events ORDER BY id")],
            "contacts": self.list_contacts(),
        }
