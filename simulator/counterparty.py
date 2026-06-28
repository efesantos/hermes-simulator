"""Simulated counterparties: spouse / school / coach stand-ins.

Multi-day coordination needs something to coordinate *with*. These implement the
:class:`~simulator.scenarios.types.Counterparty` protocol the runner calls between
agent turns: given one email the agent sent, return a reply email to seed into the
inbox (or ``None`` to stay silent).

Two design commitments from the plan:

- **Partial observability (tau-bench).** A counterparty sees only the agent's
  *message* — the email it received — never the agent's tool calls or internal
  state. That is structural here: ``reply`` is handed the email and nothing else.
- **One fixed model for everyone, deterministic where possible.** The same cheap
  model (``temperature=0`` + a fixed ``seed``) plays every counterparty across
  every candidate, so it adds equal noise to all of them. Key coordination beats
  can be pinned with :class:`ScriptedCounterparty` (or scripted overrides on
  :class:`LLMCounterparty`) for full determinism.

The LLM call is injectable (``chat_fn``) so tests never need a live model; the
default talks to an OpenAI-compatible endpoint (Ollama) over stdlib HTTP.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .scenarios.types import Persona

# A chat function: (messages, *, temperature, seed) -> assistant text.
ChatFn = Callable[..., str]


@dataclass(frozen=True)
class CounterpartyConfig:
    """How to reach the fixed counterparty model.

    Defaults to a small, fast local model — the counterparty only writes short
    in-character replies, so it needs neither a large context window nor
    tool-call competence (it is a plain chat model, not a Hermes agent).
    """

    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""  # set for hosted endpoints; unused for local Ollama
    temperature: float = 0.0
    seed: int = 7


def ollama_chat(
    config: CounterpartyConfig,
) -> ChatFn:
    """Build a :data:`ChatFn` that posts to ``{base_url}/chat/completions``."""

    def _chat(messages: list[dict[str, str]], *, temperature: float, seed: int) -> str:
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
        }
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


def _resolve_brief(persona: Persona, to_addr: str) -> Optional[tuple[str, dict]]:
    """Map an outbound recipient email to its ``(name, brief)``, or None if unknown.

    Joins the persona's seeded contacts (name+email) with the counterparty briefs
    (keyed by name). A counterparty with no brief stays silent — the agent
    emailing a stranger should not magically get a reply.
    """
    target = to_addr.strip().lower()
    for contact in persona.world_seed.get("contacts", []):
        if contact.get("email", "").strip().lower() == target:
            name = contact["name"]
            brief = persona.counterparty_brief.get(name)
            if brief is not None:
                return name, brief
    return None


def _build_messages(name: str, brief: dict, email: dict[str, Any]) -> list[dict[str, str]]:
    """Construct the chat prompt. Only the email is exposed (partial observability)."""
    voice = brief.get("voice", "natural and brief")
    knows = brief.get("knows", "your own context")
    system = (
        f"You are {name}. In one short, natural reply, answer the email below in "
        f"character. Your manner: {voice}. You know about: {knows}. Reply only to "
        f"what the message actually says — do not invent details or mention being an "
        f"AI. Keep it to a few sentences."
    )
    user = f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class ScriptedCounterparty:
    """Fully deterministic counterparty: canned replies keyed by recipient email.

    Use for pinned coordination beats, or in tests. A recipient with no script
    stays silent.
    """

    def __init__(self, replies_by_recipient: dict[str, str]) -> None:
        # normalize keys to lowercased emails
        self._replies = {k.strip().lower(): v for k, v in replies_by_recipient.items()}

    def reply(
        self, outbound_email: dict[str, Any], persona: Persona, *, sim_now: str
    ) -> Optional[dict[str, Any]]:
        to_addr = outbound_email.get("to_addr", "").strip().lower()
        body = self._replies.get(to_addr)
        if body is None:
            return None
        return _make_reply(outbound_email, body, sim_now)


class LLMCounterparty:
    """A fixed cheap LLM playing every briefed counterparty.

    ``scripted`` overrides (keyed by recipient email) pin specific beats; anything
    not pinned is generated by the model. Pass a custom ``chat_fn`` in tests to
    avoid a live model.
    """

    def __init__(
        self,
        config: Optional[CounterpartyConfig] = None,
        *,
        chat_fn: Optional[ChatFn] = None,
        scripted: Optional[dict[str, str]] = None,
    ) -> None:
        self.config = config or CounterpartyConfig()
        self._chat = chat_fn or ollama_chat(self.config)
        self._scripted = {k.strip().lower(): v for k, v in (scripted or {}).items()}

    def reply(
        self, outbound_email: dict[str, Any], persona: Persona, *, sim_now: str
    ) -> Optional[dict[str, Any]]:
        to_addr = outbound_email.get("to_addr", "").strip().lower()
        if to_addr in self._scripted:
            return _make_reply(outbound_email, self._scripted[to_addr], sim_now)

        resolved = _resolve_brief(persona, to_addr)
        if resolved is None:
            return None  # unknown recipient -> silence
        name, brief = resolved
        messages = _build_messages(name, brief, outbound_email)
        text = self._chat(messages, temperature=self.config.temperature, seed=self.config.seed)
        return _make_reply(outbound_email, text.strip(), sim_now)


def _make_reply(outbound_email: dict[str, Any], body: str, sim_now: str) -> dict[str, Any]:
    subject = outbound_email.get("subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    return {
        "from_addr": outbound_email.get("to_addr", ""),  # they reply from where mail was sent
        "to_addr": outbound_email.get("from_addr", "dana@home.test"),
        "subject": subject,
        "body": body,
        "timestamp": sim_now,
    }
