"""Stage-1 pre-filter task suite.

Single-shot, memory-off, cross-domain tasks with deterministic end-states. Each
cheaply tests one email/calendar/coordination seam; together they separate viable
models from non-viable ones before the expensive Stage-2 simulation.
"""

from __future__ import annotations

from .tasks import STAGE1_TASKS

__all__ = ["STAGE1_TASKS"]
