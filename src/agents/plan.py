"""Task decomposition: supervisor breaks a complex task into an ordered plan with
dependencies. Plan validation (DAG, valid specialists) is deterministic and tested."""
from __future__ import annotations

from pydantic import BaseModel, Field

SPECIALISTS = {"research", "data_analysis", "writing", "code_execution"}


class SubTask(BaseModel):
    id: str
    description: str
    specialist: str
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = ""


class ExecutionPlan(BaseModel):
    subtasks: list[SubTask] = Field(default_factory=list)


class PlanValidationError(Exception):
    pass


def validate_plan(plan: ExecutionPlan) -> None:
    """Ensures every specialist is real, IDs are unique, deps reference existing
    subtasks, and the dependency graph is acyclic."""
    ids = [s.id for s in plan.subtasks]
    if len(ids) != len(set(ids)):
        raise PlanValidationError("Duplicate subtask IDs.")

    id_set = set(ids)
    for s in plan.subtasks:
        if s.specialist not in SPECIALISTS:
            raise PlanValidationError(f"Unknown specialist '{s.specialist}' in {s.id}.")
        for dep in s.depends_on:
            if dep not in id_set:
                raise PlanValidationError(f"Subtask {s.id} depends on unknown {dep}.")

    _assert_acyclic(plan)


def _assert_acyclic(plan: ExecutionPlan) -> None:
    graph = {s.id: set(s.depends_on) for s in plan.subtasks}
    visited: set[str] = set()
    in_progress: set[str] = set()

    def visit(node: str) -> None:
        if node in in_progress:
            raise PlanValidationError(f"Cycle detected at {node}.")
        if node in visited:
            return
        in_progress.add(node)
        for dep in graph.get(node, ()):
            visit(dep)
        in_progress.discard(node)
        visited.add(node)

    for node in graph:
        visit(node)


def topological_order(plan: ExecutionPlan) -> list[str]:
    """Returns subtask IDs in a valid execution order (dependencies first)."""
    validate_plan(plan)
    graph = {s.id: set(s.depends_on) for s in plan.subtasks}
    order: list[str] = []
    done: set[str] = set()
    while len(order) < len(graph):
        ready = [n for n, deps in graph.items() if n not in done and deps <= done]
        if not ready:
            raise PlanValidationError("Unresolvable dependencies.")
        for n in sorted(ready):
            order.append(n)
            done.add(n)
    return order
