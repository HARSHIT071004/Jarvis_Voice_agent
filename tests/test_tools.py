"""Phase 4 unit tests: tool registry, validation, router, risk, results."""

from __future__ import annotations

import asyncio

import pytest

from app.memory.manager import MemoryManager
from app.tools import RiskLevel, Tool, ToolArgs, ToolError, ToolRegistry, ToolRouter, build_registry


@pytest.fixture
def manager(tmp_path):
    m = MemoryManager.open(tmp_path / "mem.db")
    yield m
    m.close()


@pytest.fixture
def registry(manager):
    return build_registry(manager)


@pytest.fixture
def router(registry):
    return ToolRouter(registry, execution_timeout=2.0)


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------- registry

EXPECTED_TOOLS = {
    "search_memory",
    "save_memory",
    "delete_memory",
    "get_contact",
    "save_contact",
    "update_contact",
    "create_task",
    "get_tasks",
    "complete_task",
}


def test_registry_registers_all_tools(registry):
    assert set(registry.names()) == EXPECTED_TOOLS
    assert len(registry) == 9


def test_registry_lookup_returns_correct_tool(registry):
    tool = registry.get("get_contact")
    assert tool is not None
    assert tool.name == "get_contact"
    assert tool.risk is RiskLevel.READ


def test_registry_unknown_tool_returns_none(registry):
    assert registry.get("foo_bar") is None
    assert "foo_bar" not in registry


def test_registry_duplicate_registration_rejected(registry):
    with pytest.raises(ValueError):
        registry.register(registry.get("get_contact"))


def test_declarations_have_schema(registry):
    decls = {d["name"]: d for d in registry.declarations()}
    assert set(decls) == EXPECTED_TOOLS
    task = decls["create_task"]
    assert task["description"]
    assert task["parameters"]["properties"]["title"]["type"] == "string"
    assert "title" in task["parameters"]["required"]


def test_declarations_reject_pydantic_only_keywords(registry):
    """Gemini 400s on additionalProperties/$schema — must be stripped."""
    import json

    blob = json.dumps(registry.declarations())
    assert '"additionalProperties"' not in blob
    assert '"$schema"' not in blob
    # schema-level title annotations gone, but property names survive
    task = next(d for d in registry.declarations() if d["name"] == "create_task")
    assert "title" in task["parameters"]["properties"]
    assert "title" not in task["parameters"]


# ------------------------------------------------------------------ risk

def test_risk_classification_matches_spec(registry):
    read = {t.name for t in registry.by_risk(RiskLevel.READ)}
    write = {t.name for t in registry.by_risk(RiskLevel.WRITE)}
    destructive = {t.name for t in registry.by_risk(RiskLevel.DESTRUCTIVE)}
    assert read == {"search_memory", "get_contact", "get_tasks"}
    assert write == {"save_memory", "save_contact", "update_contact", "create_task"}
    assert destructive == {"delete_memory", "complete_task"}
    assert read | write | destructive == EXPECTED_TOOLS


def test_blocked_risk_is_refused(manager, registry):
    router = ToolRouter(registry, blocked_risks={RiskLevel.DESTRUCTIVE})
    result = run(router.route("complete_task", {"title": "x"}))
    assert result.success is False
    assert result.error["code"] == "RISK_BLOCKED"


# ------------------------------------------------------- router / validation

def test_unknown_tool_fails_safely(router):
    result = run(router.route("foo_bar", {}))
    assert result.success is False
    assert result.error["code"] == "TOOL_NOT_FOUND"
    assert result.data is None


def test_invalid_arguments_rejected(router):
    result = run(router.route("create_task", {"title": 123}))
    assert result.success is False
    assert result.error["code"] == "INVALID_ARGUMENTS"


def test_extra_arguments_rejected(router):
    result = run(router.route("create_task", {"title": "ok", "evil": "drop table"}))
    assert result.success is False
    assert result.error["code"] == "INVALID_ARGUMENTS"


def test_non_dict_arguments_rejected(router):
    result = run(router.route("create_task", "just a string"))
    assert result.success is False
    assert result.error["code"] == "INVALID_ARGUMENTS"


def test_missing_required_argument_rejected(router):
    result = run(router.route("get_contact", {}))
    assert result.success is False
    assert result.error["code"] == "INVALID_ARGUMENTS"


def test_router_success_result_shape(router, manager):
    manager.save_contact(name="Rahul", company="ABC", role="Project Manager")
    result = run(router.route("get_contact", {"name": "Rahul"}))
    assert result.success is True
    d = result.as_dict()
    assert d["success"] and d["tool"] == "get_contact"
    assert d["error"] is None
    assert d["data"]["company"] == "ABC"
    assert result.risk == "READ"


def test_router_failure_result_shape(router):
    result = run(router.route("get_contact", {"name": "Nobody"}))
    assert result.success is False
    d = result.as_dict()
    assert d["data"] is None
    assert d["error"]["code"] == "CONTACT_NOT_FOUND"
    assert "Nobody" in d["error"]["message"]


def test_write_tool_classified_and_executes(router, manager):
    result = run(router.route("create_task", {"title": "Send proposal", "deadline": "tomorrow"}))
    assert result.success is True
    assert result.risk == "WRITE"
    assert manager.get_tasks()[0].title == "Send proposal"


def test_secret_content_rejected_by_policy(router):
    result = run(router.route("save_memory", {"content": "my password is hunter2"}))
    assert result.success is False
    assert result.error["code"] == "SECRET_REJECTED"


def test_nonexistent_memory_delete_fails_controlled(router):
    result = run(router.route("delete_memory", {"memory_id": 9999}))
    assert result.success is False
    assert result.error["code"] == "MEMORY_NOT_FOUND"


def test_execution_timeout_is_controlled(registry):
    class SlowTool(Tool):
        name = "slow_tool"
        description = "sleeps"
        risk = RiskLevel.READ
        args_model = ToolArgs

        async def execute(self, args):
            await asyncio.sleep(5)

    registry.register(SlowTool())
    fast = ToolRouter(registry, execution_timeout=0.2)
    result = run(fast.route("slow_tool", {}))
    assert result.success is False
    assert result.error["code"] == "TOOL_TIMEOUT"


def test_tool_crash_becomes_structured_error(registry):
    class BoomTool(Tool):
        name = "boom_tool"
        description = "crashes"
        risk = RiskLevel.READ
        args_model = ToolArgs

        async def execute(self, args):
            raise RuntimeError("boom: /secret/path")

    registry.register(BoomTool())
    result = run(ToolRouter(registry).route("boom_tool", {}))
    assert result.success is False
    assert result.error["code"] == "TOOL_INTERNAL_ERROR"
    assert "/secret/path" not in result.error["message"]  # no raw exception leakage


# ----------------------------------------------------------- business flows

def test_update_contact_requires_existing(manager, router):
    result = run(router.route("update_contact", {"name": "Ghost", "role": "x"}))
    assert result.success is False
    assert result.error["code"] == "CONTACT_NOT_FOUND"


def test_complete_task_by_title(manager, router):
    run(router.route("create_task", {"title": "Call Rahul"}))
    result = run(router.route("complete_task", {"title": "Call Rahul"}))
    assert result.success is True
    assert result.data["status"] == "done"


def test_complete_task_not_found(router):
    result = run(router.route("complete_task", {"title": "No such task"}))
    assert result.success is False
    assert result.error["code"] == "TASK_NOT_FOUND"


def test_get_tasks_defaults_to_pending(manager, router):
    run(router.route("create_task", {"title": "A"}))
    run(router.route("create_task", {"title": "B"}))
    result = run(router.route("get_tasks", {}))
    assert result.data["count"] == 2
    done = run(router.route("complete_task", {"title": "A"}))
    assert done.success
    result = run(router.route("get_tasks", {}))
    assert result.data["count"] == 1


def test_search_memory_returns_results(manager, router):
    manager.save_memory(content="I prefer Python for AI projects")
    result = run(router.route("search_memory", {"query": "python"}))
    assert result.success is True
    assert any("Python" in r["content"] for r in result.data["results"])
