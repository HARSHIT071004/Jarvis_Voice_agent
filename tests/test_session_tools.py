"""Phase 4 voice-path tests: Live tool_call translation + session routing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import types

from app.agent.provider import GeminiVoiceProvider, ToolCallEvent
from app.agent.session import VoiceSession
from app.config import load_settings
from app.memory.manager import MemoryManager
from app.state import StateMachine
from app.tools import ToolRouter, build_registry


class DummyMic:
    pass


class DummySpeaker:
    def clear(self):  # pragma: no cover - not used here
        pass


# ------------------------------------------------- provider translation

def _live_message(function_calls):
    return SimpleNamespace(
        tool_call=SimpleNamespace(function_calls=function_calls),
        server_content=None,
    )


def test_translate_tool_call_event():
    fc = types.FunctionCall(id="id1", name="get_contact", args={"name": "Rahul"})
    events = GeminiVoiceProvider._translate(_live_message([fc]))
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, ToolCallEvent)
    assert ev.calls == (("id1", "get_contact", {"name": "Rahul"}),)


def test_translate_tool_call_without_server_content_is_ok():
    fc = types.FunctionCall(id="id2", name="get_tasks", args={})
    events = GeminiVoiceProvider._translate(_live_message([fc]))
    assert isinstance(events[0], ToolCallEvent)


def test_translate_plain_message_has_no_tool_event():
    msg = SimpleNamespace(tool_call=None, server_content=None)
    assert GeminiVoiceProvider._translate(msg) == []


def test_translate_tool_call_multiple_calls():
    fcs = [
        types.FunctionCall(id="a", name="get_contact", args={"name": "Rahul"}),
        types.FunctionCall(id="b", name="get_tasks", args={}),
    ]
    ev = GeminiVoiceProvider._translate(_live_message(fcs))[0]
    assert [c[1] for c in ev.calls] == ["get_contact", "get_tasks"]


# ------------------------------------------------------- session routing

class FakeProvider:
    def __init__(self):
        self.tool_responses: list[list[dict]] = []

    async def send_tool_response(self, responses):
        self.tool_responses.append(responses)


@pytest.fixture
def router(tmp_path):
    m = MemoryManager.open(tmp_path / "mem.db")
    m.save_contact(name="Rahul", company="ABC", role="Project Manager")
    r = ToolRouter(build_registry(m), execution_timeout=2.0)
    yield r
    m.close()


def make_session(router):
    settings = load_settings(require_api_key=False)
    return VoiceSession(
        settings=settings,
        microphone=DummyMic(),  # type: ignore[arg-type]
        speaker=DummySpeaker(),  # type: ignore[arg-type]
        machine=StateMachine(),
        provider_factory=lambda: None,
        tool_router=router,
    )


def test_session_routes_tool_call_and_responds(router):
    session = make_session(router)
    provider = FakeProvider()
    ev = ToolCallEvent(calls=(("id1", "get_contact", {"name": "Rahul"}),))
    asyncio.run(session._handle_tool_calls(ev, provider))

    assert len(provider.tool_responses) == 1
    resp = provider.tool_responses[0][0]
    assert resp["success"] is True
    assert resp["data"]["company"] == "ABC"
    assert resp["call_id"] == "id1"


def test_session_tool_failure_is_structured(router):
    session = make_session(router)
    provider = FakeProvider()
    ev = ToolCallEvent(calls=(("id1", "get_contact", {"name": "Ghost"}),))
    asyncio.run(session._handle_tool_calls(ev, provider))
    resp = provider.tool_responses[0][0]
    assert resp["success"] is False
    assert resp["error"]["code"] == "CONTACT_NOT_FOUND"


def test_session_without_router_returns_tools_disabled(router):
    session = make_session(router=None)
    provider = FakeProvider()
    ev = ToolCallEvent(calls=(("id1", "get_contact", {"name": "Rahul"}),))
    asyncio.run(session._handle_tool_calls(ev, provider))
    resp = provider.tool_responses[0][0]
    assert resp["error"]["code"] == "TOOLS_DISABLED"


def test_session_tool_budget_is_enforced(router):
    settings = load_settings(require_api_key=False)
    session = make_session(router)
    session.settings = settings  # budget = agent_max_tool_calls (default 6)
    provider = FakeProvider()
    calls = tuple(
        ("id%d" % i, "get_tasks", {}) for i in range(settings.agent_max_tool_calls + 3)
    )
    asyncio.run(session._handle_tool_calls(ToolCallEvent(calls=calls), provider))
    responses = provider.tool_responses[0]
    ok = [r for r in responses if r["success"]]
    over = [r for r in responses if not r["success"] and r["error"]["code"] == "MAX_TOOL_CALLS"]
    assert len(ok) == settings.agent_max_tool_calls
    assert len(over) == 3
