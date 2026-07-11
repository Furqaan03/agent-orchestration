"""Supervisor: LLM task decomposition into a validated ExecutionPlan.

The plan structure/validation is deterministic (tested); only the decomposition
itself calls an LLM."""
from __future__ import annotations

import json

from openai import OpenAI

from src.agents.plan import ExecutionPlan, SubTask, validate_plan

_DECOMPOSE_PROMPT = """Decompose the task into an ordered list of subtasks. Each subtask has:
id (short slug), description, specialist (one of: research, data_analysis, writing, code_execution),
depends_on (list of earlier subtask ids), expected_output. Respond as JSON:
{"subtasks": [{"id": "...", "description": "...", "specialist": "...", "depends_on": [], "expected_output": "..."}]}."""


def decompose_task(task: str, client: OpenAI | None = None) -> ExecutionPlan:
    client = client or OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": _DECOMPOSE_PROMPT}, {"role": "user", "content": task}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    plan = ExecutionPlan(subtasks=[SubTask(**s) for s in parsed.get("subtasks", [])])
    validate_plan(plan)   # raises PlanValidationError if the LLM produced an invalid plan
    return plan
