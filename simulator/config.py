"""Declarative configuration for the simulator.

Three things are declared here and nowhere else:

- :class:`CandidateModel` — a model under test, plus the hosting facts needed to
  run it (provider, base_url, context window) and to cost it (price per 1M
  tokens, or zero for local).
- :class:`HostingProfile` — local (Ollama) vs API; decides how cost is derived
  (KTD7: API dollars come from ``state.db``; local dollars are imputed from a
  configurable price-per-1M assumption).
- :class:`RunConfig` — run-wide knobs: seeds, the reliability exponent ``k``, and
  the composite-score weights.

Nothing here shells out or touches the filesystem; this module is pure data so
tests can import it without a live Hermes. The eligibility floor
(``MIN_CONTEXT_LENGTH``) is the spike's observed hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Hermes rejects any model whose context window is below this floor (spike: qwen3:8b
# at 40,960 was refused with "context window below the minimum 64,000 required").
MIN_CONTEXT_LENGTH = 64_000


class Hosting(str, Enum):
    """How a candidate model is served, which decides how it is costed."""

    LOCAL = "local"  # Ollama on this machine; dollar cost is imputed, not metered.
    API = "api"  # Remote provider; dollar cost is metered into state.db.


@dataclass(frozen=True)
class HostingProfile:
    """A named way to reach models of one hosting kind.

    ``provider`` is the key written into Hermes ``config.yaml`` under ``providers:``
    and referenced by ``model.provider``. ``base_url`` is the OpenAI-compatible
    endpoint (Ollama exposes ``/v1``).
    """

    name: str
    hosting: Hosting
    provider: str
    base_url: str


# Local Ollama, auto-detected at localhost:11434 — the spike's working setup.
LOCAL_OLLAMA = HostingProfile(
    name="Local Ollama",
    hosting=Hosting.LOCAL,
    provider="local-ollama",
    base_url="http://localhost:11434/v1",
)


@dataclass(frozen=True)
class CandidateModel:
    """A model under test.

    ``context_length`` is the value forced via ``model.context_length`` in
    ``config.yaml`` (the spike's override mechanism), not necessarily the model's
    native window. ``price_per_1m_input`` / ``price_per_1m_output`` are the
    dollar assumptions used to normalize cost: for API models they should match
    the provider's real card; for local models they express the operator's
    self-hosting cost assumption (default 0 — overridden per RunConfig).
    """

    id: str  # model name as the provider knows it, e.g. "qwen3.6:latest"
    hosting_profile: HostingProfile
    context_length: int
    label: str = ""  # human-friendly name for reports; defaults to id
    price_per_1m_input: float = 0.0
    price_per_1m_output: float = 0.0
    # Model family (e.g. "qwen", "gemma", "llama") — the judge must differ from it
    # (KTD4). Inferred from the id prefix when left blank.
    family: str = ""

    @property
    def family_name(self) -> str:
        if self.family:
            return self.family
        # Infer from the id: take the leading alphabetic run ("qwen3.6:latest" -> "qwen").
        head = self.id.split(":")[0].split("-")[0]
        alpha = "".join(c for c in head if c.isalpha())
        return alpha or self.id

    @property
    def hosting(self) -> Hosting:
        return self.hosting_profile.hosting

    @property
    def provider(self) -> str:
        return self.hosting_profile.provider

    @property
    def base_url(self) -> str:
        return self.hosting_profile.base_url

    @property
    def display_name(self) -> str:
        return self.label or self.id

    @property
    def meets_context_floor(self) -> bool:
        """Cheap pre-check; the authoritative gate is Hermes refusing the run."""
        return self.context_length >= MIN_CONTEXT_LENGTH


@dataclass(frozen=True)
class CompositeWeights:
    """Weights for the configurable weighted composite (R14).

    Per-dimension columns are always reported regardless of these weights; the
    weights only affect the single composite ranking. They need not sum to 1 —
    :meth:`normalized` rescales them — but keeping them on a 0..1 scale is clearest.
    """

    capability: float = 0.40
    memory: float = 0.25
    reliability: float = 0.20
    cost: float = 0.15

    def normalized(self) -> "CompositeWeights":
        total = self.capability + self.memory + self.reliability + self.cost
        if total <= 0:
            raise ValueError("composite weights must sum to a positive number")
        return CompositeWeights(
            capability=self.capability / total,
            memory=self.memory / total,
            reliability=self.reliability / total,
            cost=self.cost / total,
        )


@dataclass(frozen=True)
class RunConfig:
    """Run-wide parameters shared across the whole matrix."""

    candidates: tuple[CandidateModel, ...]
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)  # k=5 distinct tracks per (model x persona)
    k: int = 5  # reliability exponent for pass^k
    weights: CompositeWeights = field(default_factory=CompositeWeights)
    # Imputed self-hosting price for LOCAL models when state.db reports $0 (KTD7).
    # A rough blended rate; tune after first real runs (Open Question in plan).
    local_price_per_1m_input: float = 0.20
    local_price_per_1m_output: float = 0.20

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if len(self.seeds) < self.k:
            raise ValueError(
                f"need at least k={self.k} seeds to compute pass^k; got {len(self.seeds)}"
            )


# --- Default candidate field -------------------------------------------------
# Drawn from models present on this machine (spike). Ineligible/incompatible
# models are kept in the field on purpose: Stage 1 should *demonstrate* it drops
# them with a recorded reason, rather than them being silently absent.

DEFAULT_CANDIDATES: tuple[CandidateModel, ...] = (
    # Worked cleanly in the spike (tool-call format + recall).
    CandidateModel(
        id="qwen3.6:latest",
        hosting_profile=LOCAL_OLLAMA,
        context_length=65_536,
        label="Qwen3.6 (local)",
    ),
    CandidateModel(
        id="qwen3:32b",
        hosting_profile=LOCAL_OLLAMA,
        context_length=65_536,
        label="Qwen3 32B (local)",
    ),
    # Ran but emitted "no final response" — should fail the Stage-1 format gate.
    CandidateModel(
        id="gemma3:12b",
        hosting_profile=LOCAL_OLLAMA,
        context_length=65_536,
        label="Gemma3 12B (local)",
    ),
    # Newer Gemma — included to learn whether the family fixed the tool-call-format
    # problem that dropped gemma3 at Stage 1.
    CandidateModel(
        id="gemma4:latest",
        hosting_profile=LOCAL_OLLAMA,
        context_length=65_536,
        label="Gemma4 (local)",
    ),
    # Below the 64K floor — should be dropped at the eligibility gate.
    CandidateModel(
        id="qwen3:8b",
        hosting_profile=LOCAL_OLLAMA,
        context_length=40_960,
        label="Qwen3 8B (local)",
    ),
)


def default_run_config() -> RunConfig:
    """The out-of-the-box run: all default candidates, 5 seeds, default weights."""
    return RunConfig(candidates=DEFAULT_CANDIDATES)
