"""FastAPI: submit a task (decompose -> plan), review the approval queue."""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.plan import topological_order
from src.agents.supervisor import decompose_task
from src.hitl.escalation import ApprovalQueue

load_dotenv()

app = FastAPI(title="Agent Orchestration System")
_queue = ApprovalQueue()


class TaskRequest(BaseModel):
    task: str


@app.post("/v1/tasks")
def submit_task(req: TaskRequest) -> dict:
    plan = decompose_task(req.task)
    return {
        "subtasks": [s.model_dump() for s in plan.subtasks],
        "execution_order": topological_order(plan),
    }


@app.get("/v1/approvals")
def list_approvals() -> dict:
    return {"pending": [r.__dict__ for r in _queue.pending]}


class ResolveRequest(BaseModel):
    status: str
    resolution: str = ""


@app.post("/v1/approvals/{request_id}/resolve")
def resolve_approval(request_id: str, req: ResolveRequest) -> dict:
    resolved = _queue.resolve(request_id, req.status, req.resolution)
    if resolved is None:
        raise HTTPException(404, "Approval request not found")
    return {"status": resolved.status}
