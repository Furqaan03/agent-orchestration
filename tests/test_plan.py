import pytest

from src.agents.plan import ExecutionPlan, PlanValidationError, SubTask, topological_order, validate_plan


def _plan(subtasks):
    return ExecutionPlan(subtasks=subtasks)


def test_valid_plan_passes():
    plan = _plan([
        SubTask(id="a", description="research", specialist="research"),
        SubTask(id="b", description="write", specialist="writing", depends_on=["a"]),
    ])
    validate_plan(plan)  # no raise


def test_unknown_specialist_rejected():
    plan = _plan([SubTask(id="a", description="x", specialist="magic")])
    with pytest.raises(PlanValidationError):
        validate_plan(plan)


def test_duplicate_ids_rejected():
    plan = _plan([
        SubTask(id="a", description="x", specialist="research"),
        SubTask(id="a", description="y", specialist="writing"),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(plan)


def test_unknown_dependency_rejected():
    plan = _plan([SubTask(id="a", description="x", specialist="research", depends_on=["ghost"])])
    with pytest.raises(PlanValidationError):
        validate_plan(plan)


def test_cycle_detected():
    plan = _plan([
        SubTask(id="a", description="x", specialist="research", depends_on=["b"]),
        SubTask(id="b", description="y", specialist="writing", depends_on=["a"]),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(plan)


def test_topological_order_respects_dependencies():
    plan = _plan([
        SubTask(id="c", description="z", specialist="writing", depends_on=["a", "b"]),
        SubTask(id="a", description="x", specialist="research"),
        SubTask(id="b", description="y", specialist="data_analysis", depends_on=["a"]),
    ])
    order = topological_order(plan)
    assert order.index("a") < order.index("b") < order.index("c")
