from src.hitl.escalation import ApprovalLevel, ApprovalQueue, EscalationRequest, determine_escalation
from src.observability.trace import TaskTrace, TraceNode


def test_sensitive_op_takes_over():
    level = determine_escalation(plan_confidence=0.9, consecutive_failures=0, is_sensitive=True,
                                 reviewer_quality=5.0, user_requested=False)
    assert level == ApprovalLevel.TAKE_OVER


def test_low_confidence_reviews_plan():
    level = determine_escalation(plan_confidence=0.3, consecutive_failures=0, is_sensitive=False,
                                 reviewer_quality=5.0, user_requested=False)
    assert level == ApprovalLevel.APPROVE_PLAN


def test_repeated_failure_escalates():
    level = determine_escalation(plan_confidence=0.9, consecutive_failures=2, is_sensitive=False,
                                 reviewer_quality=5.0, user_requested=False)
    assert level == ApprovalLevel.APPROVE_ACTION


def test_no_escalation_when_all_good():
    level = determine_escalation(plan_confidence=0.9, consecutive_failures=0, is_sensitive=False,
                                 reviewer_quality=5.0, user_requested=False)
    assert level is None


def test_approval_queue_resolve():
    q = ApprovalQueue()
    q.enqueue(EscalationRequest(id="e1", task_id="t1", reason="sensitive", level=ApprovalLevel.TAKE_OVER, context={}))
    resolved = q.resolve("e1", "approved", "looks fine")
    assert resolved.status == "approved"
    assert not q.pending
    assert len(q.resolved) == 1


def test_trace_aggregates_cost_and_tokens():
    root = TraceNode(agent="supervisor", decision="plan", tokens=100, cost_usd=0.01)
    child = root.add_child(TraceNode(agent="research", decision="search", tokens=50, cost_usd=0.005, status="escalated"))
    child.add_child(TraceNode(agent="writing", decision="draft", tokens=30, cost_usd=0.003))
    trace = TaskTrace(task_id="t1", root=root)
    assert trace.total_tokens() == 180
    assert trace.total_cost() == 0.018
    assert trace.escalation_count() == 1
