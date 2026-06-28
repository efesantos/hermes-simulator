"""CLI entry point: ``python -m simulator`` runs the default matrix and prints the report.

Defaults to the on-machine candidate field and the ``dana`` persona, no Stage-1
pre-filter tasks, and no LLM judge (which needs a frontier API key). Pass
``--with-stage1`` to include the pre-filter suite. This is a thin driver over
:func:`simulator.pipeline.run_full`.
"""

from __future__ import annotations

import argparse

from .config import default_run_config
from .pipeline import run_full
from .scenarios.personas import ALL_PERSONAS
from .scenarios.stage1 import STAGE1_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(prog="simulator", description=__doc__)
    parser.add_argument("--results-root", default="results", help="where to write trajectories/report")
    parser.add_argument("--with-stage1", action="store_true", help="include the Stage-1 pre-filter suite")
    parser.add_argument("--run-id", default=None, help="fixed run id (else timestamped)")
    parser.add_argument("--persona", action="append", default=None,
                        help="persona name(s) to run; repeatable (default: all)")
    args = parser.parse_args()

    cfg = default_run_config()
    names = args.persona or list(ALL_PERSONAS)
    personas = [ALL_PERSONAS[n] for n in names]
    tasks = STAGE1_TASKS if args.with_stage1 else []

    _, rendered = run_full(
        cfg, personas, stage1_tasks=tasks,
        results_root=args.results_root, run_id=args.run_id,
    )
    print(rendered)


if __name__ == "__main__":
    main()
