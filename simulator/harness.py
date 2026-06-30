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

import json
import os
import re
import sqlite3
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import CandidateModel, Hosting

# Default hermes executable. Overridable per-Harness for tests / alternate installs.
DEFAULT_HERMES_BIN = "hermes"

# If a run made zero tool calls but its input is at least this large, the
# mock-world tool schemas (~10-12K tokens) WERE loaded and the model simply chose
# not to call one — retrying won't help. Below it, the MCP servers hadn't booted
# (cold-start starvation) and a retry can recover. Sits in the clear gap between a
# model's no-tools base (~8-12K) and its with-tools input (~20K+).
TOOLS_LOADED_MIN_INPUT = 15_000

# Hermes refuses a model whose usable context is under the floor, in a few
# phrasings. All cite the token threshold (a number), which is what keeps these
# from matching incidental agent prose that merely mentions context windows:
#   "...context window below the minimum 64,000 required."
#   "Ollama loaded `m` with only 40,960 tokens of runtime context, but Hermes
#    needs at least 64,000 tokens..."
_CONTEXT_ERROR_RE = re.compile(
    r"context window .*below the minimum [\d,]+"
    r"|only [\d,]+ tokens of runtime context"
    r"|needs at least [\d,]+ tokens",
    re.IGNORECASE,
)


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

    def warm(self, *, keep_alive: str = "2m", timeout: float = 300.0) -> bool:
        """Preload a local model into Ollama so it's resident before a run.

        Critical for correctness, not just speed: when Ollama cold-loads a large
        model, that load races the MCP servers' startup and Hermes's tool
        discovery sometimes fires first — the agent then runs with NO mock-world
        tools and every tool-requiring task fails as an artifact. Warming the
        model (at the forced context) removes the race. Best-effort and a no-op
        for API-hosted models; returns True if the warm call succeeded.
        """
        if self.model.hosting != Hosting.LOCAL:
            return False
        endpoint = self.model.base_url.rstrip("/")
        if endpoint.endswith("/v1"):
            endpoint = endpoint[: -len("/v1")]
        payload = {
            "model": self.model.id, "prompt": "", "stream": False,
            "keep_alive": keep_alive, "options": {"num_ctx": self.model.context_length},
        }
        req = urllib.request.Request(
            f"{endpoint}/api/generate", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except Exception:
            return False  # best-effort; the run still proceeds

    def _config_dict(self) -> dict:
        """Minimal Hermes config pinning provider + model + forced context window.

        ``model.context_length`` is the override mechanism the spike used to lift a
        model above the 64K floor; we set it on both the top-level model block and
        the per-model entry to match the shape of a real ``config.yaml``.
        """
        m = self.model
        provider_block: dict = {
            "api": m.base_url,
            "default_model": m.id,
            "name": m.hosting_profile.name,
            "models": [
                {"name": m.id, "context_length": m.context_length},
            ],
        }
        # API providers authenticate via a bearer key Hermes reads from the named
        # env var; the harness injects that var into the subprocess (see _run_once).
        if m.hosting_profile.key_env:
            provider_block["key_env"] = m.hosting_profile.key_env
        return {
            "model": {
                "context_length": m.context_length,
                "default": m.id,
                "provider": m.provider,
            },
            "providers": {
                m.provider: provider_block,
            },
            # Auto-approve unseen shell hooks; required for unattended runs.
            "hooks_auto_accept": True,
        }

    def _env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env["HERMES_HOME"] = str(self.home)
        env["HERMES_ACCEPT_HOOKS"] = "1"
        # Give the freshly-spawned mock-world MCP servers time to register before
        # the agent's first turn, so fast API models aren't handed a tool-less
        # turn and falsely failed. Hermes joins the discovery thread with this as
        # an upper bound, returning as soon as discovery finishes — so a generous
        # value costs nothing when servers boot quickly. Requires the one-line
        # patch to hermes_cli/mcp_startup.py to honor this var (see
        # docs/solutions/integration-issues/api-path-mcp-cold-start.md). An explicit
        # value already in the environment wins.
        env.setdefault("HERMES_MCP_DISCOVERY_WAIT", "20")
        if extra:
            env.update(extra)
        return env

    # --- running -------------------------------------------------------------

    def run_oneshot(
        self,
        prompt: str,
        *,
        extra_env: dict[str, str] | None = None,
        expect_tools: bool = False,
        tool_retries: int = 3,
    ) -> HarnessResult:
        """Run a single headless prompt; return stdout/stderr/exit code.

        ``extra_env`` is added to the subprocess environment (and so to any MCP
        server Hermes launches) — used to pass the simulated clock per day.

        When ``expect_tools`` is set, the run is retried (up to ``tool_retries``
        times) if it completed without calling **any** tool. The mock-world MCP
        servers are spawned fresh per run and, for fast-loading models, the agent
        can start before they finish booting — running tool-less. Retrying lets a
        genuinely capable model get its tools on a later attempt, while a model
        that truly never emits tool calls (e.g. a format-incompatible one) still
        ends with zero and fails correctly. Retrying is safe: a tool-less run made
        no world changes.

        Raises :class:`ContextWindowError` when Hermes refuses the model for being
        below the context floor — an eligibility failure the runner records and
        drops, rather than a crash to propagate.
        """
        result = self._run_once(prompt, extra_env)
        if not expect_tools:
            return result
        for _ in range(tool_retries):
            session = self.latest_session()
            # None session => can't verify (e.g. fake binary / no state.db); don't
            # spin. A real tool-using run has tool_call_count >= 1.
            if session is None or session.tool_call_count >= 1:
                return result
            # Zero tool calls: retry ONLY if the schemas weren't loaded (low input
            # => MCP cold-start starvation). High input means the tools were
            # present and the model chose not to use them — a real result, not
            # infra; retrying would just burn runs.
            #
            # The input-token proxy is only trustworthy for LOCAL models, where it
            # was calibrated (no prompt caching; ~10-12K base, ~20K+ with tools).
            # On the API path token counts run lower and vary by provider/caching,
            # so a tools-loaded run can sit under the threshold; there we retry on
            # any zero-tool run to give the cold-start race its full set of chances.
            if self.model.hosting is Hosting.LOCAL and session.input_tokens >= TOOLS_LOADED_MIN_INPUT:
                return result
            result = self._run_once(prompt, extra_env)
        return result

    def _run_once(
        self, prompt: str, extra_env: dict[str, str] | None
    ) -> HarnessResult:
        completed = subprocess.run(
            [self.hermes_bin, "-z", prompt],
            env=self._env(extra_env),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        result = HarnessResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )
        # Hermes emits the runtime-context refusal on stdout with exit 0, so match
        # either stream regardless of exit. Safe from false positives because the
        # regex requires the cited token threshold (a number), which a successful
        # agent reply mentioning context windows in prose won't contain.
        if _CONTEXT_ERROR_RE.search(result.stdout) or _CONTEXT_ERROR_RE.search(
            result.stderr
        ):
            raise ContextWindowError(
                f"{self.model.id}: context window {self.model.context_length} "
                f"below Hermes minimum"
            )
        return result

    def add_mcp_server(
        self,
        name: str,
        *,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> HarnessResult:
        """Register a stdio MCP server into this home (``hermes mcp add``).

        Auto-confirms the discovery/enable prompt by piping ``y``. ``command`` +
        ``args`` are how Hermes will launch the server; pass the server's own
        Python interpreter (one that can import the server module) as ``command``.
        """
        cmd = [self.hermes_bin, "mcp", "add", name, "--command", command]
        for kv in (env or {}).items():
            cmd += ["--env", f"{kv[0]}={kv[1]}"]
        # --args must be last; it greedily consumes the remainder.
        cmd += ["--args", *args]
        completed = subprocess.run(
            cmd,
            env=self._env(),
            input="y\n",
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        return HarnessResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )

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
