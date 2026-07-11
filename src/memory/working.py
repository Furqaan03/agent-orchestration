"""Short-term working memory — shared across agents during one task, then cleared.
(Redis in production; in-process dict here, same interface.)"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkingMemory:
    plan: dict = field(default_factory=dict)
    subtask_outputs: dict[str, object] = field(default_factory=dict)
    intermediate: dict[str, object] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def record_output(self, subtask_id: str, output: object) -> None:
        self.subtask_outputs[subtask_id] = output

    def get_output(self, subtask_id: str) -> object | None:
        return self.subtask_outputs.get(subtask_id)

    def record_error(self, message: str) -> None:
        self.errors.append(message)

    def clear(self) -> None:
        self.plan = {}
        self.subtask_outputs.clear()
        self.intermediate.clear()
        self.errors.clear()
