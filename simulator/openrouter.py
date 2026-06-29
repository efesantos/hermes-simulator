"""OpenRouter model-capability checks.

Lets a run fail fast when an API candidate cannot do what the benchmark needs —
agentic **tool use** — instead of discovering it mid-run as a confusing
``did not call any tool`` elimination. (Observed 2026-06-29: a 70B whose
OpenRouter providers expose no tool-use endpoint returned HTTP 404
``No endpoints found that support tool use`` on every attempt.)

The catalog fetch is injectable so tests never hit the network.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable, Iterable

MODELS_URL = "https://openrouter.ai/api/v1/models"

# Returns the OpenRouter ``data`` list: each item a dict with at least ``id`` and
# ``supported_parameters``.
FetchFn = Callable[[], list[dict]]


def _default_fetch() -> list[dict]:
    req = urllib.request.Request(MODELS_URL, headers={"User-Agent": "hermes-simulator"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["data"]


def tool_capable_ids(
    model_ids: Iterable[str], *, fetch: FetchFn = _default_fetch
) -> dict[str, bool]:
    """Map each model id to whether OpenRouter advertises ``tools`` support.

    Unknown ids map to ``False`` (we cannot confirm capability). Propagates the
    fetch exception on network failure so the caller can decide to warn-and-proceed.
    """
    catalog = {m["id"]: m for m in fetch()}
    result: dict[str, bool] = {}
    for mid in model_ids:
        entry = catalog.get(mid)
        result[mid] = bool(entry and "tools" in (entry.get("supported_parameters") or []))
    return result
