"""Tool registry: register tools with schemas + rate limits; every call is logged."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    tool_name: str
    inputs: dict
    output: Any
    latency_ms: float
    success: bool
    error: str | None = None


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    allowed_agents: list[str] = field(default_factory=list)
    rate_limit_per_min: int = 60
    sensitive: bool = False   # sensitive tools trigger HITL escalation


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._call_log: list[ToolCall] = []
        self._call_counts: dict[str, int] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def available_for(self, agent: str) -> list[str]:
        return [t.name for t in self._tools.values() if not t.allowed_agents or agent in t.allowed_agents]

    def invoke(self, name: str, agent: str, now: float, **inputs) -> ToolCall:
        import time

        tool = self._tools.get(name)
        if tool is None:
            call = ToolCall(name, inputs, None, 0.0, False, "unknown tool")
            self._call_log.append(call)
            return call
        if tool.allowed_agents and agent not in tool.allowed_agents:
            call = ToolCall(name, inputs, None, 0.0, False, f"agent '{agent}' not permitted")
            self._call_log.append(call)
            return call
        if self._call_counts.get(name, 0) >= tool.rate_limit_per_min:
            call = ToolCall(name, inputs, None, 0.0, False, "rate limit exceeded")
            self._call_log.append(call)
            return call

        start = time.perf_counter()
        try:
            output = tool.func(**inputs)
            call = ToolCall(name, inputs, output, (time.perf_counter() - start) * 1000, True)
        except Exception as exc:  # noqa: BLE001 — recorded on the call for observability
            call = ToolCall(name, inputs, None, (time.perf_counter() - start) * 1000, False, str(exc))

        self._call_counts[name] = self._call_counts.get(name, 0) + 1
        self._call_log.append(call)
        return call

    @property
    def call_log(self) -> list[ToolCall]:
        return self._call_log
