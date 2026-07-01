# Hermes model-simulator — API-field findings (2026-06-30)

First **reliable** API-model benchmark, run after the persistent MCP gateway
removed the cold-start race (PR #1). This is the companion to
`docs/benchmark-findings-2026-06-29.md` (the local field).

---

## Definitive update (5 seeds, 4 families) — the ranking changed

The larger run (all four API families — GLM-5.2, Llama-3.3, Qwen2.5-72B,
Mistral-Large — **5 seeds each**, judge-inclusive, run `api-5seed`, 20 tracks)
**supersedes the 3-seed result below. The 3-seed winner did not hold up.**

```
 #  Model              Cap   Mem   Rel*  Cost$   Composite
 1  Mistral-Large      0.73  0.70  1.00  0.089   0.753
 2  Qwen2.5-72B        0.78  0.45  1.00  0.103   0.673
 3  GLM-5.2            0.80  0.40  0.80  0.154   0.580
 4  Llama-3.3 70B      0.52  0.35  0.80  0.033   0.565
```
\* Rel here is the k=1 combined-rebuild pass-rate (`build_report` recovery mode).
The live per-model runs computed pass^5, which penalized the low-reliability models
*further* (GLM-5.2's live pass^5 was 0.33) — strengthening, not weakening, Mistral's
lead.

**What changed and why it matters:**

- **Mistral-Large is the real leader (0.753)** — the best memory (0.70) at perfect
  reliability. It is the only model to crack the *second* knowledge update (swim,
  1/5) on top of the first (soccer, 4/5).
- **GLM-5.2's 3-seed lead was largely noise.** At 5 seeds it falls to #3 (0.580):
  its reliability dropped (some seeds failed to clear the capability bar) and its
  memory regressed to the pack. A 3-seed run would have crowned the wrong model —
  this is the **ranking-reproducibility** metric doing exactly its job.
- **Memory is still the wall.** Soccer (the earlier update) is now handled by most
  (Mistral/GLM 4/5, Qwen 3/5); **swim (the later update) defeats everyone** (Mistral
  1/5, the rest 0/5). Adopting a *second* mid-run change is the open frontier.
- A grader bug surfaced and was fixed mid-run: a model wrote a non-ISO event time
  (`"this week 10:00 AM"`) that crashed the behavioral grader / `build_report`
  (PR #4).

The single most important lesson: **3 seeds was not enough to trust the winner.**
The 3-seed findings below are kept as the record of how the picture evolved.

---

## 3-seed run (superseded — kept for the record)

The numbers below come from run `api-final-3seed` (GLM-5.2 and Llama-3.3 70B,
dana persona, 3 seeds, LLM judge on).

> **Status of each claim:** results are facts with the run they came from; where
> I extrapolate it's marked. This run is 3 seeds — directional on absolute scores,
> solid on the qualitative ordering and the knowledge-update behavior. The
> definitive field is the larger VPS run (see Recommended next steps).

## Ranked result

```
 #  Model                 Cap   Mem   Rel    Cost$    Tokens   Composite
 1  GLM-5.2 (API)         0.91  0.58  1.00   0.1165    65,057   0.723
 2  Llama-3.3 70B (API)   0.59  0.33  1.00   0.0298   302,720   0.623
```

Both models ran the full Stage-2 funnel with **zero race-driven eliminations** —
the point of the gateway. Composite weights are the current 0.35/0.35/0.20/0.10
(capability/memory/reliability/cost), with memory up-weighted from the original
0.25 because knowledge-update is the discriminating dimension.

For comparison, the best local model (`docs/benchmark-findings-2026-06-29.md`)
was **Qwen3.6 at 0.717**. GLM-5.2 (0.723) edges it — and unlike every model
before it, GLM-5.2 actually handles a mid-run knowledge update.

## The headline: GLM-5.2 is the first model to adopt a mid-run knowledge update

The persona plants two facts that change mid-run (soccer Thu→Wed on day 2; swim
Tue→Mon on day 5). Every prior model — Qwen3.6, Owl Alpha, Llama-3.3 — failed
both. GLM-5.2 is the first to adopt one reliably:

```
                   soccer-update   swim-update   recall   abstention
GLM-5.2   seed0       OK            stale         OK        fabricated
          seed1       OK            stale         OK        OK
          seed2       OK            stale         OK        fabricated   -> mem mean 0.58
Llama-3.3 seed0       stale         missing       OK        fabricated
          seed1       stale         missing       OK        fabricated
          seed2       stale         missing       OK        OK           -> mem mean 0.33
```

- **GLM-5.2 gets soccer right in all 3 seeds** — a genuine capability step over
  the entire prior field. But it misses swim in all 3 (still asserts the stale
  Tuesday), and fabricates the never-scheduled dentist appointment on 2/3.
  *Inference:* it adopts the **first, most-reinforced** update but not the
  **second, later** one — partial memory, not solved memory.
- **Llama-3.3 fails both updates every seed** — the same liability as the local
  field. Its capability (0.59) and memory (0.33) trail GLM-5.2 by a clear margin.

The two-knowledge-update persona design earns its keep here: a single update
would have scored GLM-5.2 a perfect knowledge-update and hidden the gap. Two
updates separate "adopts one" (GLM) from "adopts none" (Llama).

## Cost and reliability notes

- **Cost is not the discriminator at this tier.** GLM-5.2 cost ~$0.12/3-seeds and
  Llama ~$0.03; both are cheap enough that capability/memory dominate the ranking.
  Llama burned ~5× the tokens (302K vs 65K) for a worse result.
- `Rel=1.00` for both means every track *completed* and cleared the 0.5 capability
  bar (pass^k=1). As with the local field, this measures completion-reliability,
  not quality consistency — the per-seed memory spread (GLM 0.50–0.75) is the
  real run-to-run signal.
- **The gate is now fair.** Pre-gateway, Llama-3.3 was eliminated 100% of the time
  at the smoke (a race artifact) and GLM flip-flopped. Post-gateway, both pass the
  smoke on the first attempt with mock-world tools present at turn 1. The ranking
  reflects the models, not eagerness.

## What changed since the local field

- **Owl Alpha is gone.** It was a free OpenRouter stealth model and was pulled from
  the catalog on 2026-06-30; the tool-support guard now blocks it. The `api-free`
  $0-validation slot moved to `meta-llama/llama-3.3-70b-instruct:free`.
- **Hermes-3 70B is not usable on OpenRouter** — its providers expose no tool-use
  endpoint (HTTP 404). Replaced with Llama-3.3 70B.
- The field broadened to four families: GLM-5.2, Llama-3.3, Qwen2.5-72B,
  Mistral-Large (the last two added but not yet run — they belong in the VPS run).

## Recommended next steps

1. **Definitive multi-seed run on a VPS** (`docs/guides/vps-unattended-run.md`).
   No 40-min background cap, fully unattended, ≥5 seeds across the four-family
   field. 3 seeds is directional; 5 seeds gives reproducibility evidence (does the
   GLM > Llama ordering hold, and does GLM's soccer-adoption survive more seeds?).
2. **Probe the swim-update miss.** GLM adopts soccer but not swim consistently. Is
   it recency (later update), reinforcement (soccer is also seeded on the
   calendar), or position in the day's inbox? A persona variant that swaps which
   update comes first would test the hypothesis.
3. **Tighten abstention.** Both models fabricate the dentist appointment on some
   seeds. The grader fix (a refusal that names its subject is still a refusal)
   lands the genuine declines; the remaining `fabricated` scores are real
   inventions worth a closer look at the transcripts.
