"""Tests for project-local .env loading."""

from __future__ import annotations

import os
from pathlib import Path

from simulator.env import load_env_file, load_project_env


def test_load_env_file_missing_is_noop(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    loaded = load_env_file(tmp_path / ".env")
    assert loaded == 0
    assert "OPENROUTER_API_KEY" not in os.environ


def test_load_env_file_sets_values(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-or-test\n")
    loaded = load_env_file(env_file)
    assert loaded == 1
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-test"


def test_load_env_file_does_not_override_existing_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-dotenv\n")
    loaded = load_env_file(env_file, override=False)
    assert loaded == 0
    assert os.environ["OPENROUTER_API_KEY"] == "from-shell"


def test_load_env_file_parses_export_and_quotes(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("export OPENROUTER_API_KEY='sk-or-quoted'\n")
    loaded = load_env_file(env_file)
    assert loaded == 1
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-quoted"


def test_load_project_env_uses_project_root_and_filename(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env.test").write_text("OPENROUTER_API_KEY=sk-or-project\n")
    loaded = load_project_env(project_root=tmp_path, filename=".env.test")
    assert loaded == 1
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-project"
