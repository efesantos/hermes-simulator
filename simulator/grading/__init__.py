"""Grading: out-of-band scoring of what the agent did.

- :mod:`deterministic` — the state-diff engine (key-value assertions over the
  world's end-state). Shared by Stage-1 pre-filter grading (U4) and Stage-2
  deterministic grading (U7).
- ``behavioral`` (U7) — improvement-over-days checks.
- ``memory_exam`` / ``judge`` (U8) — memory quality and fuzzy behavior.

Every grader reads the world store or trajectory directly; none trusts
agent-produced claims (KTD2).
"""
