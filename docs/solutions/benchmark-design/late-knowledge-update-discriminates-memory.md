---
title: Late knowledge-updates discriminate memory — a change with little runway is the frontier
date: 2026-07-01
category: benchmark-design
module: simulator/scenarios/personas/dana.py
problem_type: best_practice
component: scenario-design
symptoms:
  - "Models pass the FIRST mid-run knowledge-update (soccer, day 2) but fail the SECOND (swim, day 5)"
  - "The later update is recalled WORSE despite being more recent (recency effect is absent)"
  - "Some models never process the late change email; others state the new fact in-context but revert to the stale one at exam time"
root_cause: memory_consolidation
resolution_type: finding
severity: medium
tags:
  - memory
  - knowledge-update
  - persona-design
  - consolidation
  - benchmark
  - discrimination
---

# Late knowledge-updates discriminate memory — a change with little runway is the frontier

## What we found

The `dana` persona plants two facts that change mid-run: soccer moves Thu→Wed on
**day 2**, swim moves Tue→Mon on **day 5**. In the definitive 5-seed × 4-family API
run (`api-5seed`; see `docs/benchmark-findings-2026-06-30.md`), every model handled
the **early** change far better than the **late** one:

```
              soccer-update (day 2)   swim-update (day 5)
Mistral-Large     4/5 correct            1/5 correct   (winner overall)
GLM-5.2           4/5                    0/5
Qwen2.5-72B       3/5                    0/5
Llama-3.3 70B     1/5                    0/5
```

The *second* mid-run change is the open frontier — no model reliably adopts it.

## Why — and why it is NOT recency

Recency would make the **later** change (swim, day 5) *easier* to recall at the
end-of-run exam. It is recalled **worse**, so recency is ruled out. Inspecting the
day-5/day-6 transcripts against the exam answer shows two distinct failure modes:

1. **Never processes the late email.** Llama (4/5) and Qwen (2/5) don't engage the
   day-5 swim change at all → the exam answer is "no info." An inbox-attention /
   task-completion failure that shows up late in the run.
2. **Processes it, but doesn't consolidate it.** Mistral and GLM *do* read it —
   Mistral even states "**Monday**" on day 5 — yet at the exam they revert to the
   stale **Tuesday**. They held the change in-context but never durably wrote it to
   memory.

The common cause is **timing / reinforcement, not the specific fact**:

- The day-2 soccer change gets **4 subsequent days** (3-6) to be re-encountered and
  consolidated (e.g. Sam asks about pickup day 3). The day-5 swim change gets **one**
  (day 6). Little runway → weak consolidation.
- **Structural anchor:** the mock calendar still holds the *original* seeded day
  (an email change does not auto-update the calendar event). A model consulting its
  calendar at exam time reads the stale day unless it proactively moved the event —
  which it has far more opportunity to do for the early change.

## Implications for scenario design

- **Put the discriminating update LATE.** A single, early knowledge-update
  over-credits models — most pass it. A change with minimal post-change runway is
  what actually separates models on memory consolidation. `dana`'s two-update design
  (early + late) is why the benchmark discriminates where a one-update persona would
  not (it flipped the 3-seed vs 5-seed ranking).
- **Mind the calendar-anchor confound.** Because the seeded event isn't auto-updated
  by the exogenous email, "recall the new day" partly measures "did the agent update
  its calendar," not pure memory. This is arguably realistic (a good assistant *would*
  update the calendar), but note it when interpreting the memory dimension.
- **Two failure modes need distinguishing.** "No info" (never processed) vs "stale"
  (processed, not consolidated) are different weaknesses; the memory exam already
  separates `missing` from `stale`, so keep that distinction visible in analysis.

## Open confirming experiment (not yet run)

To confirm "late, not swim," author a `dana` variant that **swaps the change days**
(soccer late / swim early). If *soccer* then fails and swim passes, timing is the
cause, not the fact. Cheap (~20 tracks). Deferred as confirmation — the transcript
evidence above (a model stating the new day in-context yet reverting at exam) already
establishes the consolidation mechanism.

## Related

- `docs/benchmark-findings-2026-06-30.md` — the run this came from (5-seed, 4 families).
- `simulator/scenarios/personas/dana.py` — the two-update persona.
- `simulator/grading/memory_exam.py` — scores `knowledge_update` as correct / stale /
  missing (forgetting-aware), which is what surfaced the distinction.
