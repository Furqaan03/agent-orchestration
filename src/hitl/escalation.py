"""Human-in-the-loop escalation: triggers, approval levels, and the approval queue."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ApprovalLevel(str, Enum):
    NOTIFY = "notify"                 # proceed but inform the human
    APPROVE_ACTION = "approve_action" # human confirms the next step
    APPROVE_PLAN = "approve_plan"     # human reviews the full plan first
    TAKE_OVER = "take_over"           # human provides the output directly


@dataclass
class EscalationRequest:
    id: str
    task_id: str
    reason: str
    level: ApprovalLevel
    context: dict
    status: str = "pending"   # pending | approved | rejected | modified
    resolution: str = ""


def determine_escalation(
    plan_confidence: float,
    consecutive_failures: int,
    is_sensitive: bool,
    reviewer_quality: float | None,
    user_requested: bool,
    quality_threshold: float = 3.0,
) -> ApprovalLevel | None:
    """Maps triggers to the appropriate approval depth. Returns None if no
    escalation is needed."""
    if user_requested:
        return ApprovalLevel.APPROVE_ACTION
    if is_sensitive:
        return ApprovalLevel.TAKE_OVER          # sensitive ops (financial, deletion, external comms)
    if plan_confidence < 0.5:
        return ApprovalLevel.APPROVE_PLAN        # low plan confidence -> review the whole plan
    if consecutive_failures >= 2:
        return ApprovalLevel.APPROVE_ACTION
    if reviewer_quality is not None and reviewer_quality < quality_threshold:
        return ApprovalLevel.APPROVE_ACTION
    return None


@dataclass
class ApprovalQueue:
    pending: list[EscalationRequest] = field(default_factory=list)
    resolved: list[EscalationRequest] = field(default_factory=list)

    def enqueue(self, request: EscalationRequest) -> None:
        self.pending.append(request)

    def resolve(self, request_id: str, status: str, resolution: str = "") -> EscalationRequest | None:
        for i, req in enumerate(self.pending):
            if req.id == request_id:
                req.status = status
                req.resolution = resolution
                self.resolved.append(req)
                self.pending.pop(i)
                return req
        return None
