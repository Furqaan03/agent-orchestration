# Agent Orchestration System with Tool Use, Memory, and Human-in-the-Loop

A multi-agent orchestration platform: a supervisor agent decomposes complex
tasks into a dependency-ordered plan, delegates subtasks to specialist tool-using
agents, maintains persistent memory across interactions, and escalates to a human
when confidence is low or the task is sensitive — with full observability into
every decision.

## Why this exists

Agents are the frontier of AI engineering, and most candidate projects are
single-agent toy demos. This is a multi-agent system with real tool use,
persistent memory, and graded human-in-the-loop escalation — the architecture
companies are actively building.

## Architecture

```
src/agents/plan.py            ExecutionPlan schema + validation: unique IDs, real
                               specialists, resolvable deps, ACYCLIC graph, topo-sort
src/agents/supervisor.py      LLM task decomposition -> validated ExecutionPlan
src/tools/registry.py         tool registry with per-agent permissions, rate limits,
                               and a full call log (inputs/outputs/latency/errors)
src/memory/working.py         short-term working memory, shared per-task then cleared
src/memory/longterm.py        long-term semantic memory: embed learnings, retrieve for
                               planning, importance scoring, decay, GDPR forget-user
src/hitl/escalation.py        escalation triggers -> 4 graded approval levels +
                               approval queue
src/observability/trace.py    execution trace tree + cost/token/latency aggregation
src/api/main.py               FastAPI: submit task, review + resolve approvals
```

## Design decisions

- **The plan is validated as a DAG before anything runs.** An LLM-produced plan
  can have duplicate IDs, reference nonexistent specialists, or contain dependency
  cycles. `validate_plan` catches all of those and `topological_order` produces a
  safe execution order — so a malformed plan fails fast instead of deadlocking mid-run.
- **Escalation is graded, not binary.** Not every human touchpoint needs the same
  depth: NOTIFY (proceed + inform), APPROVE_ACTION (confirm next step), APPROVE_PLAN
  (review the whole plan first), TAKE_OVER (human provides the output). Sensitive
  operations (financial, deletion, external comms) always force TAKE_OVER regardless
  of confidence — the highest-stakes actions get the strongest gate.
- **Two-tier memory with importance and decay.** Working memory is task-scoped and
  cleared on completion; long-term memory persists learnings, and frequently-accessed
  memories gain importance (boosting them in retrieval) while stale ones decay. A
  `forget_user` endpoint supports data-deletion requests.
- **Tools enforce permissions AND rate limits, and log everything.** A tool
  declares which agents may use it and a per-minute cap; every invocation — success
  or failure — is logged with inputs, output, latency, and error, which is what makes
  the trace explorer possible.
- **Everything except LLM calls is pure and tested.** Plan validation/topo-sort,
  the tool registry, both memory tiers, escalation routing, and trace aggregation are
  all deterministic and covered offline (injected embedder, injected clock) — only
  task decomposition itself calls a model.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements-core.txt   # or requirements.txt for ChromaDB/Redis
cp .env.example .env                    # OPENAI_API_KEY (+ ANTHROPIC_API_KEY)
uvicorn src.api.main:app --reload
```

## Example

```bash
curl -X POST localhost:8000/v1/tasks -H "Content-Type: application/json" \
  -d '{"task": "Research the top 3 vector databases, compare them, and write a summary."}'
# -> {"subtasks": [...], "execution_order": ["research-1", "analysis-1", "writing-1"]}
```

## Tests

```bash
pytest tests/ -v
```

22 tests covering plan validation (unknown specialist, duplicate IDs, unknown
dependency, cycle detection, topological ordering), the tool registry (success +
logging, permission denial, rate limiting, unknown tool, per-agent availability),
both memory tiers (working-memory clear, similarity retrieval, user isolation,
importance growth, forget-user), escalation routing (sensitive->take-over,
low-confidence->approve-plan, repeated-failure, no-escalation), the approval
queue, and trace aggregation — all offline, no API key required.

## Docker

```bash
docker compose up --build   # API + Redis (working memory) + Postgres (state)
```

## Status

Phases 1-4 complete (agent hierarchy + task decomposition, tool registry, two-tier
memory, graded HITL escalation, observability trace tree). The LangGraph state
machine is realized as an explicit validated-plan + topo-order executor; Phase 4's
trace-explorer UI and Phase 5's end-to-end showcase demo are not built — the API
exposes task submission and the approval queue.
