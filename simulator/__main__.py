"""CLI entry point: ``python -m simulator`` runs the default matrix and prints the report.

Defaults to the on-machine candidate field and the ``dana`` persona and no
Stage-1 pre-filter tasks. The LLM judge runs via the Claude Code subscription
(no API key) and is on by default for ``--candidates api``; toggle with
``--judge``/``--no-judge``. Pass ``--with-stage1`` to include the pre-filter
suite. Pass ``--candidates api`` to run the OpenRouter-hosted field (export
``OPENROUTER_API_KEY`` first). This is a thin driver over
:func:`simulator.pipeline.run_full`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

from .config import CANDIDATE_FIELDS, Hosting, run_config_for
from .counterparty import CounterpartyConfig, LLMCounterparty
from .env import load_project_env
from .grading.judge import subscription_judge
from .openrouter import tool_capable_ids
from .pipeline import run_full
from .scenarios.personas import ALL_PERSONAS
from .scenarios.stage1 import STAGE1_TASKS

# Cheap OpenRouter chat model that plays every simulated counterparty when running
# the API field (no local Ollama to fall back on). Verify availability in spike.
COUNTERPARTY_API_MODEL = "z-ai/glm-4.7-flash"


def main() -> None:
    # Load repo-local secrets/config from .env if present (without overriding real
    # exported env vars). Keeps API keys local and persistent across sessions.
    load_project_env()
    parser = argparse.ArgumentParser(prog="simulator", description=__doc__)
    parser.add_argument("--results-root", default="results", help="where to write trajectories/report")
    parser.add_argument("--with-stage1", action="store_true", help="include the Stage-1 pre-filter suite")
    parser.add_argument("--run-id", default=None, help="fixed run id (else timestamped)")
    parser.add_argument("--candidates", default="default", choices=sorted(CANDIDATE_FIELDS),
                        help="candidate field to run (default: on-machine; 'api' = OpenRouter)")
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=None,
                        help="run the Claude Code subscription judge (default: on for --candidates api)")
    parser.add_argument("--judge-model", default="sonnet",
                        help="model alias/id passed to `claude --model` for the judge")
    parser.add_argument("--seeds", type=int, default=None,
                        help="number of seeds (tracks per model); fewer = cheaper/faster (default 5)")
    parser.add_argument("--model", action="append", default=None,
                        help="restrict the field to these model id(s); repeatable. Lets one "
                             "model run as its own job (e.g. to fit a time cap), writing into "
                             "a shared --run-id for build_report.py to combine.")
    parser.add_argument("--persona", action="append", default=None,
                        help="persona name(s) to run; repeatable (default: all)")
    args = parser.parse_args()

    cfg = run_config_for(args.candidates)
    if args.model:
        wanted = set(args.model)
        kept = tuple(m for m in cfg.candidates if m.id in wanted)
        unknown = wanted - {m.id for m in kept}
        if unknown:
            sys.exit(f"--model id(s) not in the '{args.candidates}' field: {', '.join(sorted(unknown))}\n"
                     f"Available: {', '.join(m.id for m in cfg.candidates)}")
        cfg = replace(cfg, candidates=kept)
    if args.seeds is not None:
        if args.seeds < 1:
            sys.exit("--seeds must be >= 1")
        seeds = tuple(range(args.seeds))
        cfg = replace(cfg, seeds=seeds, k=min(cfg.k, len(seeds)))

    # Fail fast on a missing provider key rather than letting Hermes error opaquely
    # mid-run. API candidates name the env var holding their bearer key.
    missing = sorted({
        m.hosting_profile.key_env
        for m in cfg.candidates
        if m.hosting is Hosting.API and m.hosting_profile.key_env
        and not os.environ.get(m.hosting_profile.key_env)
    })
    if missing:
        sys.exit(
            "Missing required API key environment variable(s): "
            + ", ".join(missing)
            + f"\nThe '{args.candidates}' candidate field needs them set. "
            "Export the key(s) and re-run, e.g.:\n  export OPENROUTER_API_KEY=sk-or-..."
        )

    # Guard: an OpenRouter candidate that can't do tool use fails mid-run as a
    # confusing "did not call any tool" elimination — catch it up front. Network
    # failure is non-fatal (warn and proceed).
    or_ids = [m.id for m in cfg.candidates if m.hosting_profile.provider == "openrouter"]
    if or_ids:
        try:
            caps = tool_capable_ids(or_ids)
            no_tools = sorted(i for i, ok in caps.items() if not ok)
            if no_tools:
                sys.exit(
                    "OpenRouter candidate(s) without tool-use support (agentic runs "
                    "require it): " + ", ".join(no_tools)
                    + "\nPick models whose OpenRouter card lists 'tools' in supported_parameters."
                )
        except Exception as exc:  # network/parse error — don't block the run
            print(f"warning: could not verify OpenRouter tool support ({exc}); proceeding.",
                  file=sys.stderr)

    names = args.persona or list(ALL_PERSONAS)
    personas = [ALL_PERSONAS[n] for n in names]
    tasks = STAGE1_TASKS if args.with_stage1 else []

    has_api = any(m.hosting is Hosting.API for m in cfg.candidates)

    # Judge: subscription-backed (no API key). Default on whenever the field has an
    # API model — that is the run where qualitative capability scoring matters most.
    judge_on = args.judge if args.judge is not None else has_api
    judge = subscription_judge(model=args.judge_model) if judge_on else None

    # Counterparty: the local Ollama default has no fallback when running an API
    # field, so route the simulated counterparties through a cheap OpenRouter model.
    counterparty = None
    if has_api:
        counterparty = LLMCounterparty(CounterpartyConfig(
            model=COUNTERPARTY_API_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        ))

    _, rendered = run_full(
        cfg, personas, stage1_tasks=tasks,
        results_root=args.results_root, run_id=args.run_id,
        judge=judge, counterparty=counterparty,
    )
    print(rendered)


if __name__ == "__main__":
    main()
