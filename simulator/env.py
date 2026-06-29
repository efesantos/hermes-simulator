"""Project-local environment loading for CLI runs.

This repo keeps real secrets out of git via an untracked ``.env`` file in the
project root. ``load_project_env`` reads that file into ``os.environ`` on
startup so API-backed workflows can run without manual ``export`` each session.
Existing environment variables always win.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


def _strip_quotes(value: str) -> str:
    """Drop matching single/double quotes around a value, if present."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _iter_env_pairs(text: str) -> Iterator[tuple[str, str]]:
    """Yield ``KEY=VALUE`` pairs from dotenv-like text.

    Supports blank lines, comments, and optional ``export`` prefix.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        yield key, _strip_quotes(value.strip())


def load_env_file(path: Path, *, override: bool = False) -> int:
    """Load environment variables from ``path``.

    Returns the count of variables set in ``os.environ``.
    """
    if not path.exists():
        return 0
    loaded = 0
    for key, value in _iter_env_pairs(path.read_text()):
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        loaded += 1
    return loaded


def load_project_env(
    *,
    project_root: Path | None = None,
    filename: str = ".env",
    override: bool = False,
) -> int:
    """Load ``project_root/filename`` into ``os.environ``.

    Defaults to this package's repo root and ``.env``. Existing exported
    variables are not overwritten unless ``override=True``.
    """
    root = project_root or Path(__file__).resolve().parents[1]
    return load_env_file(root / filename, override=override)
