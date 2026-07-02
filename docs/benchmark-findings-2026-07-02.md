# Benchmark findings — 2026-07-02: cost-forward, latency-aware, multilingual persona

**Run:** `results/amsterdam-chunked` — 15 models × 5 seeds on the `amsterdam` persona
(multilingual EN/NL/PT family assistant), judged via the Claude Code subscription.
Per-model chunks stitched with `scripts/build_report.py`.

## Headline

**The user's own model, Qwen 3.7 Plus, is #1 — on both weightings.** It is the most
accurate (capability + memory), perfectly reliable across all 5 seeds, and cheap per task
($0.045). The user stopped using it believing it "could be better" and cost too much; on this
benchmark it is the best model in the field. The likely real drivers of that perception —
**per-call latency** (it is the slowest in real usage, ~15.9s/call) and **input-heavy cost
accumulation** — are real, but they are not accuracy problems, and the cheaper/faster
alternatives cost accuracy to fix them.

## Ranking (memory_heavy — the default weighting)

| # | Model | Cap | Mem | Rel | $/task | Speed(s) | Composite |
|---|---|---|---|---|---|---|---|
| 1 | **Qwen 3.7 Plus** *(user's model)* | 0.86 | 0.92 | 1.00 | 0.0447 | 253.7 | **0.923** |
| 2 | GLM-5.2 | 0.82 | 0.88 | 1.00 | 0.1542 | 379.8 | 0.886 |
| 3 | **Nemotron 3 Super 120B** | 0.79 | 0.88 | 1.00 | 0.0549 | 597.8 | 0.880 |
| 4 | GPT-5.4-mini | 0.85 | 0.76 | 1.00 | 0.1272 | 263.3 | 0.857 |
| 5 | DeepSeek V3.2 | 0.72 | 0.84 | 1.00 | 0.1775 | 562.3 | 0.836 |
| 6 | Mistral-Large | 0.83 | 0.60 | 1.00 | 0.0931 | 182.0 | 0.796 |
| 7 | Gemini 3.5 Flash | 0.91 | 0.76 | 0.80 | 1.3323 | 344.1 | 0.745 |
| 8 | Qwen 3.5 Flash | 0.67 | 0.60 | 0.80 | 0.0219 | 122.6 | 0.704 |
| 9 | Nemotron 3 Ultra | 0.86 | 0.40 | 0.80 | 0.1490 | 87.3 | 0.693 |
| 10 | GPT-OSS 120B | 0.66 | 0.44 | 0.80 | 0.0175 | 240.1 | 0.643 |
| 11 | Gemma 4 31B | 0.71 | 0.52 | 0.40 | 0.0285 | 898.4 | 0.611 |
| 12 | Llama-3.3 70B | 0.68 | 0.28 | 0.80 | 0.0268 | 73.2 | 0.595 |
| 13 | Qwen2.5 72B | 0.83 | 0.24 | 0.40 | 0.0524 | 214.0 | 0.554 |
| 14 | Sonnet 5 | 0.70 | 0.44 | 0.60 | 0.8793 | 644.7 | 0.553 |
| 15 | Qwen-Plus | 0.51 | 0.48 | 0.40 | 0.0437 | 128.6 | 0.525 |

Under the **cost_forward** weighting (cost + speed up-weighted, memory down) the order shifts
but the top stays: **Qwen 3.7 Plus #1** (0.912), then GPT-5.4-mini, GLM-5.2, Mistral-Large,
Nemotron 3 Super, Qwen 3.5 Flash. Sonnet 5 and Gemini fall to the bottom (14th, 13th) — their
cost sinks them once cost is weighted.

`$/task` = cost per completed track (tokens × live price); `Speed` = end-to-end sim task-time
(see latency note). 8 of 15 models clear the viability floor (cap ≥ 0.60, mem ≥ 0.50, rel ≥ 0.75).

## The three picks

- **Best accuracy → Qwen 3.7 Plus** — cap 0.86 / mem 0.92 / perfect reliability.
- **Best value → Qwen 3.5 Flash** — $0.0219/task, ~2× faster in-sim, cap 0.67 / mem 0.60.
- **Cheapest viable → Qwen 3.5 Flash** — same.

## The answer to the question

- **Don't switch for accuracy.** Qwen 3.7 Plus is the best model tested — most accurate,
  perfectly reliable, and cheap per task. Staying on it is the right call for quality.
- **If you want cheaper + faster and can accept lower accuracy:** **Qwen 3.5 Flash** (half the
  cost, ~2× faster in-sim; cap drops 0.86→0.67, memory 0.92→0.60) or, keeping accuracy high,
  **Nemotron 3 Super 120B** (cap 0.79 / mem 0.88 / perfect reliability at just $0.055/task —
  a standout cheap open-weight model, #3 overall).
- **The expensive frontier models are not worth it here.** Gemini 3.5 Flash ($1.33/task) and
  Sonnet 5 ($0.88/task) cost **20–30× more** than Qwen 3.7 Plus and ranked *below* it (7th and
  14th). Gemini is capable (cap 0.91) but failed a seed (rel 0.80) and is by far the priciest;
  Sonnet's score is understated (see caveats) but even generously it does not justify its cost.

## Latency: sim vs. real usage (they measure different things)

The `Speed(s)` column is **end-to-end simulator task-time** — the wall-clock for the agent to
complete a whole day's multi-turn task including tool round-trips. It is **not** comparable to a
per-API-call latency. Your real OpenRouter usage (`openrouter_activity_2026-07-01.csv`) gives the
per-call view, which is what you *feel* interactively:

| Model | Real per-call latency (your CSV) | Sim task-time (this run) |
|---|---|---|
| Qwen 3.7 Plus | **~15.9s/call** (slowest) | 253.7s/task |
| GLM-5.2 | ~13.0s/call | 379.8s/task |
| Gemini 3.5 Flash | **~2.5s/call** (fastest) | 344.1s/task |

**This is the crux of your original frustration:** Qwen 3.7 Plus is the *slowest per call* in
real use (~15.9s), which feels sluggish interactively, and your usage was input-token-heavy
(~77K input/call) so cost accumulated. Neither is an accuracy problem. If interactive speed is
the priority, Qwen 3.5 Flash is the fastest of the strong-accuracy Qwen options.

## Data-quality caveats (read before over-reading small gaps)

- **Sonnet 5's score is understated.** It shares the judge's family (Anthropic), so it was scored
  on the *deterministic* capability component only (no judge qualitative blend), and API flakiness
  left it 3/5 clean memory exams. Treat its 0.553 as a conservative floor, not a verdict. It is
  also the **most expensive** model ($0.88/task metered, ~$0.88/seed).
- **Partial memory exams from API flakiness** depressed several memory scores (empty exam ⇒ that
  seed's memory = 0): Gemini 4/5, GPT-OSS 3/5, Qwen2.5-72B 2/5. Their memory numbers are
  conservative. The clean-5/5 models (Qwen 3.7 Plus, GLM-5.2, Nemotron Super, GPT-5.4-mini,
  DeepSeek, Mistral, Qwen-Plus, Qwen 3.5 Flash) are unaffected.
- **Gemma 4 31B: rel 0.01** — it almost always failed the task and was very slow (898s/task); a
  poor fit for this workload despite being cheap.
- **MiniMax M2.5 dropped:** as a reasoning model on a saturated provider its calls hung 20min–
  1h44m; not benchmarkable here.
- **Cohere North Mini Code excluded:** free-only on OpenRouter (no paid variant), and its free
  tier rate-limits during the exam, so no reliable run was possible.
- **Formerly-free models run on cheap paid tiers:** GPT-OSS 120B, Gemma 4 31B, Nemotron 3
  Super/Ultra were first attempted on `:free` tiers but those rate-limited during the exam's rapid
  calls (empty exams); they were re-run on their (cheap) paid variants for clean data.

## Cost context

Per-task cost spans **$0.0175 (GPT-OSS 120B)** to **$1.33 (Gemini)** — a ~75× spread. Qwen 3.7
Plus at $0.045 sits near the cheap end while topping accuracy. The frontier models (Gemini,
Sonnet) are the only ones above $0.80/task and neither earns it. Total run cost ≈ $13 of credit
(Sonnet ~$4.4 and Gemini ~$1.3 dominated; the four formerly-free models cost ~$1 combined).

## Recommendation

**Stay on Qwen 3.7 Plus** for accuracy and reliability. If interactive latency or cost is the
pain point, trial **Qwen 3.5 Flash** (cheapest-viable, fastest strong-Qwen) or **Nemotron 3
Super 120B** (keeps near-top accuracy at a fraction of the cost) — but expect a real accuracy
drop with Qwen 3.5 Flash. Skip the frontier models for this workload.
