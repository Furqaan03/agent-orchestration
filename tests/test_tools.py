from src.tools.registry import Tool, ToolRegistry


def _registry():
    reg = ToolRegistry()
    reg.register(Tool(name="search", description="web search", func=lambda query: f"results for {query}",
                      allowed_agents=["research"], rate_limit_per_min=2))
    return reg


def test_invoke_success_and_logged():
    reg = _registry()
    call = reg.invoke("search", agent="research", now=0.0, query="python")
    assert call.success is True
    assert call.output == "results for python"
    assert len(reg.call_log) == 1


def test_permission_denied_for_wrong_agent():
    reg = _registry()
    call = reg.invoke("search", agent="writing", now=0.0, query="x")
    assert call.success is False
    assert "not permitted" in call.error


def test_rate_limit_enforced():
    reg = _registry()
    reg.invoke("search", agent="research", now=0.0, query="1")
    reg.invoke("search", agent="research", now=0.0, query="2")
    call = reg.invoke("search", agent="research", now=0.0, query="3")  # 3rd exceeds limit of 2
    assert call.success is False
    assert "rate limit" in call.error


def test_unknown_tool():
    reg = _registry()
    call = reg.invoke("ghost", agent="research", now=0.0)
    assert call.success is False


def test_available_for_agent():
    reg = _registry()
    assert "search" in reg.available_for("research")
    assert "search" not in reg.available_for("writing")
