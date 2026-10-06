"""Live Gemini agent integration tests (spec section 32 scenarios).

Skipped unless RUN_LIVE_TESTS=1 and GEMINI_API_KEY are set, so the main
suite never depends on the network. Uses the real GeminiAgentLLM
function-calling loop against the real ToolRouter.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="live API test; set RUN_LIVE_TESTS=1 to enable",
)


def make_runtime(manager):
    from app.agent.agent_llm import GeminiAgentLLM
    from app.agent.runtime import AGENT_PROMPT, AgentRuntime
    from app.config import load_settings
    from app.tools import ToolRouter, build_registry

    settings = load_settings(require_api_key=True)
    registry = build_registry(manager)
    router = ToolRouter(registry, execution_timeout=settings.agent_tool_timeout)
    llm = GeminiAgentLLM(
        api_key=settings.gemini_api_key,
        model=settings.intelligence_model,
        declarations=registry.declarations(),
        system_instruction=AGENT_PROMPT,
    )
    return AgentRuntime(
        llm, router,
        max_iterations=settings.agent_max_tool_iterations,
        max_tool_calls=settings.agent_max_tool_calls,
    )


@pytest.fixture
def manager(tmp_path):
    from app.memory.manager import MemoryManager

    m = MemoryManager.open(tmp_path / "agent.db")
    yield m
    m.close()


def ask(rt, text):
    return asyncio.run(rt.handle(text))


def test_scenario_1_contact_retrieval(manager):
    manager.save_contact(name="Rahul", company="ABC", role="Project Manager")
    resp = ask(make_runtime(manager), "Who is Rahul?")
    assert resp.error is None
    assert any(r.tool == "get_contact" and r.success for r in resp.tool_results)
    assert "ABC" in resp.text


def test_scenario_2_save_contact(manager):
    resp = ask(
        make_runtime(manager),
        "Remember that Rahul is the project manager at ABC.",
    )
    assert resp.error is None
    stored = manager.get_contact("Rahul")
    assert stored is not None
    assert stored.company == "ABC"
    assert "JSON" not in resp.text  # natural response, no raw tool output


def test_scenario_3_task_creation(manager):
    resp = ask(
        make_runtime(manager),
        "Create a task to send Rahul the proposal tomorrow.",
    )
    assert resp.error is None
    tasks = manager.get_tasks()
    assert any("proposal" in t.title.lower() for t in tasks)
    assert tasks[0].deadline  # deadline captured in some form


def test_scenario_4_memory_search(manager):
    manager.save_memory(content="I prefer Python for AI projects", importance=4)
    resp = ask(
        make_runtime(manager),
        "What do you remember about my Python preference?",
    )
    assert resp.error is None
    assert any(r.tool == "search_memory" and r.success for r in resp.tool_results)
    assert "python" in resp.text.lower()


def test_scenario_5_tool_failure_is_not_hallucinated(manager):
    resp = ask(make_runtime(manager), "Who is Zaphod Beeblebrox?")
    assert resp.error is None
    if any(r.tool == "get_contact" for r in resp.tool_results):
        # tool ran and failed -> answer must admit the contact is unknown
        assert "not found" in resp.text.lower() or "don't have" in resp.text.lower() or "no contact" in resp.text.lower()


def test_scenario_6_no_tool_for_general_question(manager):
    resp = ask(make_runtime(manager), "Explain RAG in two sentences.")
    assert resp.error is None
    assert resp.tool_results == []  # ordinary question → no tool call
    assert len(resp.text) > 40
