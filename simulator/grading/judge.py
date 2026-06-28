"""LLM-as-judge for the genuinely fuzzy dimensions.

The deterministic grader decides crisp success/fail; this judge scores only
qualitative behavior — tone, proactivity, whether the agent surfaces remembered
context (KTD4). It is a frontier API model from a **different family** than the
model under test, with the standard bias mitigations:

- **cross-family** — enforced: scoring a candidate with a same-family judge
  raises, since self-preference bias is strongest within a family.
- **rubric-anchored** — every dimension is scored 1-5 against explicit anchors
  included in the prompt, not a bare "rate this".
- **position-randomized** — :meth:`compare` presents two responses in a
  content-determined order and maps the verdict back, so ``compare(a, b)`` and
  ``compare(b, a)`` agree and identical responses tie regardless of order.
- **optional majority vote** — ``n_judges`` > 1 aggregates by median (scores) or
  majority (comparisons) to curb single-sample noise.

The model call is injectable (``chat_fn``) so tests never need a live frontier
key; the default talks to an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

ChatFn = Callable[..., str]

# Qualitative dimensions the judge scores (the deterministic grader owns the rest).
DEFAULT_RUBRIC: dict[str, str] = {
    "tone": "Warm, respectful, appropriate to a personal assistant. 1=curt/robotic, 5=natural and considerate.",
    "proactivity": "Anticipates needs and surfaces next steps without being asked. 1=purely reactive, 5=genuinely helpful initiative.",
    "memory_surfacing": "Brings up relevant remembered context (preferences, prior facts) when useful. 1=ignores known context, 5=weaves it in naturally.",
}


class JudgeError(RuntimeError):
    """The judge could not produce a usable verdict."""


class JudgeFamilyError(JudgeError):
    """The judge and the candidate share a model family — disallowed (self-preference)."""


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    family: str  # e.g. "anthropic", "openai", "google" — must differ from the candidate's
    base_url: str
    api_key: str = ""
    temperature: float = 0.0


@dataclass(frozen=True)
class Verdict:
    scores: dict[str, int]  # dimension -> 1..5
    rationale: str

    @property
    def mean(self) -> float:
        return statistics.mean(self.scores.values()) if self.scores else 0.0


def openai_chat(config: JudgeConfig) -> ChatFn:
    def _chat(messages: list[dict[str, str]], *, temperature: float) -> str:
        payload = {"model": config.model, "messages": messages, "temperature": temperature}
        req = urllib.request.Request(
            f"{config.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]

    return _chat


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise JudgeError(f"no JSON object in judge response: {text[:200]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"unparsable judge JSON: {exc}") from exc


class Judge:
    def __init__(
        self,
        config: JudgeConfig,
        *,
        chat_fn: Optional[ChatFn] = None,
        n_judges: int = 1,
        rubric: Optional[dict[str, str]] = None,
    ) -> None:
        if n_judges < 1 or n_judges % 2 == 0:
            raise ValueError("n_judges must be a positive odd number (for majority/median)")
        self.config = config
        self._chat = chat_fn or openai_chat(config)
        self.n_judges = n_judges
        self.rubric = rubric or DEFAULT_RUBRIC

    def _ensure_cross_family(self, candidate_family: str) -> None:
        if candidate_family.strip().lower() == self.config.family.strip().lower():
            raise JudgeFamilyError(
                f"judge family {self.config.family!r} matches candidate family "
                f"{candidate_family!r}; pick a judge from a different family"
            )

    def _rubric_text(self) -> str:
        return "\n".join(f"- {dim}: {desc}" for dim, desc in self.rubric.items())

    # --- single-response rubric scoring (what U9 uses) -----------------------

    def score(self, transcript: str, *, candidate_family: str) -> Verdict:
        """Score one agent transcript on the qualitative rubric (1-5 per dimension)."""
        self._ensure_cross_family(candidate_family)
        dims = list(self.rubric)
        system = (
            "You are an impartial evaluator of a personal-assistant AI. Score ONLY "
            "the rubric dimensions below, each from 1 to 5 using the anchors given. "
            "Judge the assistant's behavior, not its formatting. Respond with a JSON "
            'object: {"scores": {dim: int, ...}, "rationale": "one sentence"}.\n\n'
            f"Rubric:\n{self._rubric_text()}"
        )
        samples = [
            _extract_json(self._chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": f"Assistant transcript:\n{transcript}"}],
                temperature=self.config.temperature,
            ))
            for _ in range(self.n_judges)
        ]
        scores: dict[str, int] = {}
        for dim in dims:
            vals = [int(s.get("scores", {}).get(dim, 0)) for s in samples]
            scores[dim] = int(round(statistics.median(vals)))
        rationale = str(samples[0].get("rationale", ""))
        return Verdict(scores=scores, rationale=rationale)

    # --- position-bias-resistant pairwise comparison -------------------------

    def compare(self, response_a: str, response_b: str, *, candidate_family: str) -> str:
        """Return 'a', 'b', or 'tie'. Order-stable: compare(a,b) and compare(b,a) agree.

        Identical responses tie outright. Otherwise the two are presented in a
        content-determined order (so the physical prompt is the same whichever way
        the arguments come in), and the raw verdict is mapped back to a/b.
        """
        self._ensure_cross_family(candidate_family)
        if response_a == response_b:
            return "tie"

        # Deterministic, content-based ordering -> identical prompt for (a,b)/(b,a).
        ka, kb = _digest(response_a), _digest(response_b)
        first_is_a = ka < kb
        first, second = (response_a, response_b) if first_is_a else (response_b, response_a)

        votes = [self._compare_once(first, second) for _ in range(self.n_judges)]
        winner = _majority(votes)  # "first" | "second" | "tie"
        if winner == "tie":
            return "tie"
        first_wins = winner == "first"
        chose_a = first_wins == first_is_a
        return "a" if chose_a else "b"

    def _compare_once(self, first: str, second: str) -> str:
        system = (
            "You are an impartial evaluator. Two personal-assistant responses are "
            "labeled FIRST and SECOND. Decide which better satisfies the rubric, or "
            'TIE if equal. Respond JSON: {"winner": "FIRST"|"SECOND"|"TIE"}.\n\n'
            f"Rubric:\n{self._rubric_text()}"
        )
        user = f"FIRST:\n{first}\n\nSECOND:\n{second}"
        verdict = _extract_json(self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.config.temperature,
        ))
        return str(verdict.get("winner", "TIE")).strip().lower()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _majority(votes: list[str]) -> str:
    counts = {"first": 0, "second": 0, "tie": 0}
    for v in votes:
        counts[v if v in counts else "tie"] += 1
    return max(counts, key=counts.get)
