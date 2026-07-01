"""Multi-day personas: curricula that exercise memory over many days.

Each persona is a :class:`~simulator.scenarios.types.Persona` validated by
:func:`~simulator.scenarios.personas.schema.validate_persona`. The registry below
is the entry point for the runner and graders.
"""

from __future__ import annotations

from ..types import Persona
from .amsterdam import PERSONA as AMSTERDAM
from .dana import PERSONA as DANA

ALL_PERSONAS: dict[str, Persona] = {DANA.name: DANA, AMSTERDAM.name: AMSTERDAM}


def get_persona(name: str) -> Persona:
    if name not in ALL_PERSONAS:
        raise KeyError(f"unknown persona {name!r}; have {sorted(ALL_PERSONAS)}")
    return ALL_PERSONAS[name]
