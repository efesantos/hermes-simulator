"""Wrapper around the Hermes CLI.

This is the *only* module that shells out to ``hermes``. It encapsulates the
spike recipe (``docs/spikes/2026-06-28-hermes-feasibility-findings.md``):

- one disposable ``HERMES_HOME`` per track, with a minimal ``config.yaml``
  pointing at the candidate model/provider;
- headless runs via ``hermes -z`` with ``HERMES_ACCEPT_HOOKS=1``;
- ``hermes memory reset`` to wipe a track's memory;
- per-run token/cost read back from ``$HERMES_HOME/state.db`` (table ``sessions``).

Known eligibility failures (context window too small) are raised as typed,
catchable exceptions so the runner can drop a model with a recorded reason
instead of seeing a raw crash.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import CandidateModel

# Default hermes executable. Overridable per-Harness for tests / alternate installs.
DEFAULT_HERMES_BIN = "hermes"

# Hermes prints this (substring) when a model's context window is under the floor.
_CONTEXT_ERROR_RE = re.compile(r"context window .*below the minimum", re.IGNORECASE)


class HarnessError(Exception):
    """Base class for harness failures."""


class ContextWindowError(HarnessError):
    """The model's context window is below Hermes's minimum — an eligibility failure.

    Raised (rather than returned) because a context-floor failure means the model
    cannot run at all; the runner catches this to drop the candidate at Stage 1.
    """


@dataclass(frozen=True)
class SessionRow:
    """One row of the ``sessions`` table — a single Hermes invocation's accounting."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    api_call_count: int
    tool_call_count: int
    estimated_cost_usd: float
    actual_cost_usd: float
    started_at: str
    ended_at: str

    @property
    def total_tokens(self) -> int:
        """Tokens-to-complete: input + output + reasoning (KTD7).

        Cache reads/writes are tracked separately on the row; they are not added
        here because for local Ollama there is no prompt caching (spike), and on
        API providers cache reads are billed differently from fresh input.
        """
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


@dataclass(frozen=True)
class HarnessResult:
    """Outcome of one ``hermes -z`` invocation."""

    stdout: str
    stderr: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# Columns we read from the sessions table, in SessionRow order. We map by name
# (not position) so the read tolerates upstream column reordering/additions.
_SESSION_FIELDS = (
    ("model", str, ""),
    ("input_tokens", int, 0),
    ("output_tokens", int, 0),
    ("cache_read_tokens", int, 0),
    ("cache_write_tokens", int, 0),
    ("reasoning_tokens", int, 0),
    ("api_call_count", int, 0),
    ("tool_call_count", int, 0),
    ("estimated_cost_usd", float, 0.0),
    ("actual_cost_usd", float, 0.0),
    ("started_at", str, ""),
    ("ended_at", str, ""),
)


class Harness:
    """Drives one isolated Hermes home for one candidate model.

    Construct with the target ``HERMES_HOME`` directory and the model to run;
    call :meth:`setup` once, then :meth:`run_oneshot` per prompt. The home is
    disposable — point it at a fresh directory per ``(model x persona x seed)``
    track to keep memory isolated (KTD1).
    """

    def __init__(
        self,
        home: str | os.PathLike[str],
        model: CandidateModel,
        *,
        hermes_bin: str = DEFAULT_HERMES_BIN,
        timeout: float = 600.0,
    ) -> None:
        self.home = Path(home)
        self.model = model
        self.hermes_bin = hermes_bin
        self.timeout = timeout

    # --- lifecycle -----------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.home / "config.yaml"

    @property
    def state_db_path(self) -> Path:
        return self.home / "state.db"

    @property
    def memory_dir(self) -> Path:
        return self.home / "memories"

    def setup(self) -> None:
        """Create the home directory and write a minimal ``config.yaml``.

        Idempotent: safe to call on an existing home (overwrites config only).
        """
        self.home.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(self._config_dict(), sort_keys=False))

    def _config_dict(self) -> dict:
        """Minimal Hermes config pinning provider + model + forced context window.

        ``model.context_length`` is the override mechanism the spike used to lift a
        model above the 64K floor; we set it on both the top-level model block and
        the per-model entry to match the shape of a real ``config.yaml``.
        """
        m = self.model
        return {
            "model": {
                "context_length": m.context_length,
                "default": m.id,
                "provider": m.provider,
            },
            "providers": {
                m.provider: {
                    "api": m.base_url,
                    "default_model": m.id,
                    "name": m.hosting_profile.name,
                    "models": [
                        {"name": m.id, "context_length": m.context_length},
                    ],
                },
            },
            # Auto-approve unseen shell hooks; required for unattended runs.
            "hooks_auto_accept": True,
        }

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["HERMES_HOME"] = str(self.home)
        env["HERMES_ACCEPT_HOOKS"] = "1"
        return env

    # --- running -------------------------------------------------------------

    def run_oneshot(self, prompt: str) -> HarnessResult:
        """Run a single headless prompt; return stdout/stderr/exit code.

        Raises :class:`ContextWindowError` when Hermes refuses the model for being
        below the context floor — an eligibility failure the runner records and
        drops, rather than a crash to propagate.
        """
        completed = subprocess.run(
            [self.hermes_bin, "-z", prompt],
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        result = HarnessResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )
        if _CONTEXT_ERROR_RE.search(result.stdout) or _CONTEXT_ERROR_RE.search(
            result.stderr
        ):
            raise ContextWindowError(
                f"{self.model.id}: context window {self.model.context_length} "
                f"below Hermes minimum"
            )
        return result

    def reset_memory(self) -> None:
        """Wipe this home's memory (``hermes memory reset``).

        Pipes ``yes`` to auto-confirm the destructive prompt. Only affects this
        home — other tracks' memory is untouched.
        """
        subprocess.run(
            [self.hermes_bin, "memory", "reset"],
            env=self._env(),
            input="yes\n",
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )

    # --- accounting ----------------------------------------------------------

    def read_sessions(self) -> list[SessionRow]:
        """Read every ``sessions`` row for this home, oldest first.

        Returns an empty list if no run has happened yet (no ``state.db``).
        Tolerates extra/reordered columns by selecting ``*`` and mapping by name.
        """
        if not self.state_db_path.exists():
            return []
        # Open read-only so a concurrently-running hermes process isn't disturbed.
        uri = f"file:{self.state_db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute("SELECT * FROM sessions ORDER BY started_at")
            except sqlite3.OperationalError:
                return []  # table not created yet
            return [self._row_to_session(dict(r)) for r in cursor.fetchall()]
        finally:
            conn.close()

    def latest_session(self) -> SessionRow | None:
        """The most recent session row, or None if there are none."""
        sessions = self.read_sessions()
        return sessions[-1] if sessions else None

    @staticmethod
    def _row_to_session(row: dict) -> SessionRow:
        values = {}
        for name, caster, default in _SESSION_FIELDS:
            raw = row.get(name, default)
            if raw is None:
                raw = default
            try:
                values[name] = caster(raw)
            except (TypeError, ValueError):
                values[name] = default
        return SessionRow(**values)
