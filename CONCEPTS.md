# Concepts

Shared domain vocabulary for this project — entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

## Simulation structure

### Track
One isolated run of a single model against one persona under one random seed, with its own memory profile so no two runs share state.
*Avoid:* run

### Persona
A multi-day curriculum for one synthetic person — a fixed schedule of incoming events plus a ground-truth answer key — authored to exercise memory (recall, a mid-run change of fact, an abstention trap), not just task completion.

### Counterparty
A simulated stand-in (spouse, school, coach) that replies to the agent's messages from a fixed cheap model with partial observability, so multi-day coordination has something consistent to coordinate with.

### Exogenous event stream
The fixed, timestamped sequence of events a persona injects into the world identically across every track, so variance between tracks reflects the model rather than the world.

## The funnel

### Two-stage funnel
The benchmark's shape: a cheap pre-filter (Stage 1) eliminates non-viable models before the expensive multi-day simulation (Stage 2) runs only on the survivors.

### Stage 1
The pre-filter — hard eligibility gates (context floor, tool-call format) plus single-shot cross-domain tasks that cheaply separate viable models from non-viable ones before any expensive simulation.

### Stage 2
The multi-day, memory-on persona simulation that survivors run, graded on capability, memory, reliability, and cost.

## World and grading

### Mock world
The fake email/calendar/contacts environment the agent operates through MCP tools, backed by a single store the grader can inspect.

### Out-of-band grading
Scoring that reads the world's backing store and persisted trajectories directly rather than through any agent-reachable tool, so an agent can neither influence the score nor leak the ground truth.

### Tool starvation
A run in which the agent executed without its mock-world tools loaded — because the MCP servers lost a startup race against the model load — recognizable by an input-token count far below a tool-loaded run together with zero tool calls; such runs are infrastructure artifacts, not model verdicts.
