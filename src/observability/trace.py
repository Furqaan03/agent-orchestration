"""Execution trace tree + per-task cost/performance aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraceNode:
    agent: str
    decision: str
    tool_calls: list[str] = field(default_factory=list)
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"   # success | warning | failure | escalated
    children: list["TraceNode"] = field(default_factory=list)

    def add_child(self, node: "TraceNode") -> "TraceNode":
        self.children.append(node)
        return node


@dataclass
class TaskTrace:
    task_id: str
    root: TraceNode

    def total_tokens(self) -> int:
        return self._aggregate(lambda n: n.tokens)

    def total_cost(self) -> float:
        return round(self._aggregate(lambda n: n.cost_usd), 6)

    def total_latency(self) -> float:
        return self._aggregate(lambda n: n.latency_ms)

    def escalation_count(self) -> int:
        return self._count(lambda n: n.status == "escalated")

    def _aggregate(self, fn) -> float:
        def walk(node: TraceNode) -> float:
            return fn(node) + sum(walk(c) for c in node.children)
        return walk(self.root)

    def _count(self, pred) -> int:
        def walk(node: TraceNode) -> int:
            return (1 if pred(node) else 0) + sum(walk(c) for c in node.children)
        return walk(self.root)
