# Running the API benchmark unattended on a VPS

The API field calls models over OpenRouter, so the machine that runs the
benchmark needs **no GPU** — it only runs the lightweight Python harness, the
`hermes` CLI, and the mock-world servers, all of which make API calls out. That
makes a cheap always-on Linux box (e.g. a **Hostinger KVM VPS**, ~$5/mo) the right
home for the long, definitive runs that exceed an interactive session's
background-task time limit.

> **Why a VPS and not your Mac?** Interactive Claude Code background tasks are
> killed at ~40 min, which caps a single run at ~6 tracks. A `tmux`/`nohup`
> process on a VPS has no such cap, so a full ≥5-seed, multi-model field runs to
> completion unattended. Local *Ollama* models still belong on your Mac (a cheap
> VPS has no GPU); only the **API** field moves here.

## What runs where

| Piece | Where it runs | Needs |
|---|---|---|
| Candidate models (GLM-5.2, Llama-3.3, …) | OpenRouter | `OPENROUTER_API_KEY` |
| Harness + mock-world MCP servers + `hermes` | The VPS | Python 3.11+, the `hermes` CLI |
| LLM judge | The VPS, via `claude -p` | Claude Code installed + logged in (subscription) |

## One-time setup

1. **Provision a Hostinger KVM VPS** (KVM1 is enough — models are remote). Note
   the IP and SSH in. Hostinger also offers a Claude-Code-preinstalled template and
   a browser terminal in hPanel.

2. **System deps + the repo:**
   ```bash
   # Python 3.11+, git, tmux
   sudo apt update && sudo apt install -y python3.11 python3.11-venv git tmux
   git clone <your-fork-url> hermes-simulator && cd hermes-simulator
   python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
   ```

3. **Install the `hermes` CLI** (Hermes Agent — the repo validates against
   **v0.16.0**) per Nous's instructions and confirm it's on `PATH`:
   ```bash
   hermes --version    # must succeed; the harness shells out to this
   ```
   The persistent-gateway fix means hermes no longer needs the
   `HERMES_MCP_DISCOVERY_WAIT` patch (the gateway removed the race) — applying
   `scripts/patch_hermes_mcp_wait.py` is harmless but optional now.

4. **The OpenRouter key** (gitignored `.env`):
   ```bash
   cp .env.example .env
   # edit .env: OPENROUTER_API_KEY=sk-or-...
   ```

5. **The judge (optional but recommended).** Install Claude Code and log in with
   your subscription so the judge's `claude -p` calls work:
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   claude   # then log in to your Pro/Max subscription
   ```
   If you'd rather not run the judge on the VPS, pass `--no-judge` (below) — the
   run still produces capability-from-tasks, memory, reliability, and cost.

6. **Smoke-check the wiring** at $0 before a paid run:
   ```bash
   .venv/bin/python -m simulator --candidates api-free --seeds 1 --persona dana
   ```
   This runs the free Llama-3.3 variant end-to-end. Confirm it reaches Stage-2 and
   prints a ranked report.

## The definitive run

Run inside `tmux` so it survives disconnects and isn't time-capped:

```bash
tmux new -s bench
.venv/bin/python -m simulator \
    --candidates api --seeds 5 --persona dana \
    --run-id api-field-5seed --results-root results \
    2>&1 | tee results/api-field-5seed.log
# detach with Ctrl-b d ; reattach later with: tmux attach -t bench
```

- `--candidates api` runs the full four-family field (GLM-5.2, Llama-3.3,
  Qwen2.5-72B, Mistral-Large).
- `--seeds 5` gives `pass^5` reliability and seed-to-seed reproducibility.
- Add `--no-judge` to skip the subscription judge; `--model <id>` (repeatable) to
  run a subset.

**Cost.** Rough order: a full Stage-2 track is ~60–300K tokens. At these models'
rates a 4-model × 5-seed field is plausibly **$5–15** of OpenRouter spend (Qwen
and Llama are cheap; GLM-5.2 and Mistral cost more). Start with `--seeds 3` or a
two-model `--model` subset if you want to bound it first.

## Recovering an interrupted run

Per-track results and `memory_exam.json` are written as the run proceeds, so a
report can be rebuilt from disk without re-running:

```bash
.venv/bin/python scripts/build_report.py api-field-5seed results
```

(The rebuild costs API candidates with their real prices but does not re-score the
judge — it reflects task/memory/reliability/cost from disk.)

## Scheduling (optional)

For a recurring run, a cron entry that invokes the same command works; keep the
output under `results/<dated-run-id>/` so reports don't overwrite each other.
```
