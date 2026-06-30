"""Tests for the OpenRouter tool-capability guard (injected fetch, no network)."""
from __future__ import annotations

from simulator.openrouter import tool_capable_ids

_CATALOG = [
    {"id": "openrouter/owl-alpha", "supported_parameters": ["tools", "temperature"]},
    {"id": "z-ai/glm-5.2", "supported_parameters": ["tools"]},
    {"id": "nousresearch/hermes-3-llama-3.1-70b", "supported_parameters": ["temperature"]},
]


def _fetch():
    return _CATALOG


def test_tool_capable_distinguishes_support():
    caps = tool_capable_ids(
        ["openrouter/owl-alpha", "z-ai/glm-5.2", "nousresearch/hermes-3-llama-3.1-70b"],
        fetch=_fetch,
    )
    assert caps == {
        "openrouter/owl-alpha": True,
        "z-ai/glm-5.2": True,
        "nousresearch/hermes-3-llama-3.1-70b": False,  # the real 2026-06-29 failure
    }


def test_unknown_id_is_not_capable():
    caps = tool_capable_ids(["does/not-exist"], fetch=_fetch)
    assert caps == {"does/not-exist": False}


def test_missing_supported_parameters_key_is_false():
    caps = tool_capable_ids(["x/y"], fetch=lambda: [{"id": "x/y"}])
    assert caps == {"x/y": False}
