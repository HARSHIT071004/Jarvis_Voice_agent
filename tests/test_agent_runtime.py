"""Phase 4 unit tests: AgentRuntime loop (spec sections 16/29/31).

A FakeLLM scripts each round so tests need no network: round 1 can
request tools, later rounds see the real structured results.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.runtime import AgentResponse, AgentRuntime, LLMReply, LLMToolCall
from app.memory.manager import MemoryManager
from app.tools import ToolRouter, build_registry


@pytest.fixture
def manager(tmp_path):
    m = MemoryManager.open(tmp_path / "mem.db")
    yield m
    m.close()


@pytest.fixture
def router(manager):
    return ToolRouter(build_registry(manager), execution_timeout=2.0)


class FakeLLM:
    """Scripted LLM: pops one reply per round; records what it saw."""

    def __init__(self, replies: list[LLMReply]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, user_text, history, tool_results, transcript):
        self.calls.append({"tool_results": tool_results, "user_text": user_text})
        if not self.replies:
            raise AssertionError("FakeLLM script exhausted (extra loop round)")
        return self.replies.pop(0)


def make(*replies: LLMReply) -> FakeLLM:
    return FakeLLM(list(replies))


def run(coro):
    return asyncio.run(coro)


def text(s: str) -> LLMReply:
    return LLMReply(text=s)


def call(name: str, args: dict, call_id: str | None = None) -> LLMReply:
    return LLMReply(tool_calls=[LLMToolCall(name=name, arguments=args, call_id=call_id)])


# ------------------------------------------------------------ basic loop

def test_no_tool_request_gets_direct_answer(router):
    llm = make(text("Machine learning is a field of AI."))
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("What is machine learning?"))
    assert isinstance(resp, AgentResponse)
    assert resp.text == "Machine learning is a field of AI."
    assert resp.tool_results == []
    assert resp.error is None
    assert llm.calls[0]["tool_results"] == []  # no tool results on round 1


def test_greeting_triggers_no_tool(router):
    llm = make(text("Hello! How can I help?"))
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("Hello."))
    assert resp.tool_results == []


def test_single_tool_flow_grounded_in_result(manager, router):
    manager.save_contact(name="Rahul", company="ABC", role="Project Manager")
    llm = make(
        call("get_contact", {"name": "Rahul"}, call_id="c1"),
        LLMReply(text="Rahul is the project manager at ABC."),
    )
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("Who is Rahul?"))

    assert resp.iterations == 2
    assert len(resp.tool_results) == 1
    r = resp.tool_results[0]
    assert r.success and r.tool == "get_contact" and r.data["company"] == "ABC"
    # round 2 saw the REAL result
    assert llm.calls[1]["tool_results"][0]["data"]["company"] == "ABC"
    assert llm.calls[1]["tool_results"][0]["call_id"] == "c1"


def test_multi_tool_request(manager, router):
    manager.save_contact(name="Rahul", company="ABC")
    manager.save_task(title="Send proposal to Rahul")
    llm = make(
        LLMReply(
            tool_calls=[
                LLMToolCall("get_contact", {"name": "Rahul"}, "c1"),
                LLMToolCall("get_tasks", {}, "c2"),
            ]
        ),
        LLMReply(text="Rahul is at ABC and you have one pending task: send the proposal."),
    )
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("What is Rahul's company and my tasks related to him?"))

    assert len(resp.tool_results) == 2
    assert all(r.success for r in resp.tool_results)
    # both results fed back in one round
    seen = {r["tool"] for r in llm.calls[1]["tool_results"]}
    assert seen == {"get_contact", "get_tasks"}


def test_save_contact_scenario(manager, router):
    llm = make(
        call("save_contact", {"name": "Rahul", "company": "ABC", "role": "Project Manager"}),
        LLMReply(text="Got it. I'll remember that Rahul is the project manager at ABC."),
    )
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("Remember Rahul is the project manager at ABC."))
    assert resp.tool_results[0].success
    stored = manager.get_contact("Rahul")
    assert stored is not None and stored.company == "ABC"


def test_task_creation_scenario(manager, router):
    llm = make(
        call("create_task", {"title": "Send proposal to Rahul", "deadline": "tomorrow"}),
        LLMReply(text="Done. I've created the task to send Rahul the proposal tomorrow."),
    )
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("Create a task to send Rahul the proposal tomorrow."))
    assert resp.tool_results[0].success
    assert manager.get_tasks()[0].deadline == "tomorrow"


# ------------------------------------------------------------- failure paths

def test_failed_tool_reported_to_llm_not_hallucinated(manager, router):
    llm = make(
        call("get_contact", {"name": "Ghost"}),
        LLMReply(text="I couldn't find anyone named Ghost in memory."),
    )
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("Who is Ghost?"))
    assert resp.tool_results[0].success is False
    err = llm.calls[1]["tool_results"][0]
    assert err["success"] is False
    assert err["error"]["code"] == "CONTACT_NOT_FOUND"
    assert "couldn't find" in resp.text.lower()


def test_unknown_tool_call_returns_controlled_error_to_llm(router):
    llm = make(
        call("drop_database", {"x": 1}),
        LLMReply(text="Sorry, I can't do that."),
    )
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("do something naughty"))
    assert resp.tool_results[0].success is False
    assert resp.tool_results[0].error["code"] == "TOOL_NOT_FOUND"
    assert llm.calls[1]["tool_results"][0]["error"]["code"] == "TOOL_NOT_FOUND"


# ------------------------------------------------------------ loop safety

def test_max_iterations_enforced(router):
    # LLM always asks for another tool, never answers
    llm = FakeLLM([call("get_tasks", {}) for _ in range(10)])
    rt = AgentRuntime(llm, router, max_iterations=3)
    resp = run(rt.handle("loop forever"))
    assert resp.error == "MAX_ITERATIONS_EXCEEDED"
    assert resp.iterations == 3
    assert len(resp.tool_results) == 3  # stopped after budget, no more
    assert "simpler" in resp.text.lower()


def test_max_tool_calls_budget(router):
    many = LLMReply(
        tool_calls=[LLMToolCall("get_tasks", {}, f"c{i}") for i in range(5)]
    )
    llm = make(many, text("done"))
    rt = AgentRuntime(llm, router, max_iterations=5, max_tool_calls=2)
    resp = run(rt.handle("do many things"))
    assert resp.error == "MAX_TOOL_CALLS_EXCEEDED"
    assert len(resp.tool_results) == 2
    # budget hit during round 1 → loop never returned to the LLM
    assert len(llm.calls) == 1


def test_empty_answer_is_controlled(router):
    llm = make(LLMReply(text="   "))
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("hi"))
    assert resp.error == "EMPTY_ANSWER"
    assert resp.text  # still speaks something sane


# ----------------------------------------------------- repeatable async run

def test_runtime_is_reusable_across_requests(router):
    llm = make(text("first"), text("second"))
    rt = AgentRuntime(llm, router)
    assert run(rt.handle("q1")).text == "first"
    assert run(rt.handle("q2")).text == "second"
    # fresh transcript per request is the adapter's concern; FakeLLM just
    # records that round 1 of each request had no tool results
    assert llm.calls[0]["tool_results"] == []
    assert llm.calls[1]["tool_results"] == []
