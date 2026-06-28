"""Shared test fixtures.

The key fixture is :func:`fake_hermes`, a stand-in for the real ``hermes`` binary
so the default suite never needs a live model or Ollama. It is a small bash
script whose behavior is steered by environment variables the test sets:

- ``-z <prompt>``     → prints ``$FAKE_STDOUT`` to stdout, ``$FAKE_STDERR`` to
                        stderr, exits ``$FAKE_EXIT`` (default 0). Also appends the
                        prompt to ``$HERMES_HOME/memories/USER.md`` to simulate
                        memory persistence within a home.
- ``memory reset``    → truncates ``$HERMES_HOME/memories/USER.md``.

Because the script reads ``$HERMES_HOME`` (which the Harness sets per call), it
naturally exercises home isolation.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

_FAKE_HERMES = r"""#!/usr/bin/env bash
set -u
mem="$HERMES_HOME/memories"
mkdir -p "$mem"
if [ "${1:-}" = "-z" ]; then
  printf '%s' "${FAKE_STDOUT:-}"
  if [ -n "${FAKE_STDERR:-}" ]; then printf '%s' "$FAKE_STDERR" >&2; fi
  # Simulate the agent persisting something to memory for this home.
  printf '%s\n' "${2:-}" >> "$mem/USER.md"
  exit "${FAKE_EXIT:-0}"
elif [ "${1:-}" = "memory" ] && [ "${2:-}" = "reset" ]; then
  : > "$mem/USER.md"
  exit 0
fi
exit 0
"""


@pytest.fixture
def fake_hermes(tmp_path: Path) -> str:
    """Write the fake hermes script and return its path (for ``hermes_bin=``)."""
    script = tmp_path / "fake_hermes.sh"
    script.write_text(_FAKE_HERMES)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)
    return str(script)


@pytest.fixture(autouse=True)
def _clear_fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure FAKE_* knobs don't leak between tests."""
    for var in ("FAKE_STDOUT", "FAKE_STDERR", "FAKE_EXIT"):
        monkeypatch.delenv(var, raising=False)
