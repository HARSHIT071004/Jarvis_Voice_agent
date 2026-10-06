"""Phase 8 §24 — Audit system tests: structured events, redaction,
credential material never stored."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.memory import MemoryManager
from app.security import build_control_plane
from app.security.audit import AuditLog, summarize_args, summarize_value
from app.security.models import AuditEvent
from app.tools import ToolRouter, build_registry


def run(coro):
    return asyncio.run(coro)


def make_settings(tmp_path, **overrides):
    base = dict(
        approval_required_for_high_risk=True,
        critical_action_mode="deny",
        security_denied_tools="",
        approval_timeout_seconds=120.0,
        audit_enabled=True,
        audit_path=tmp_path / "audit.jsonl",
        audit_redact_sensitive_data=True,
        security_enabled=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_setup(tmp_path, **overrides):
    manager = MemoryManager.open(tmp_path / "jarvis.db")
    plane = build_control_plane(make_settings(tmp_path, **overrides))
    router = ToolRouter(build_registry(manager), execution_timeout=5.0, security=plane)
    return manager, plane, router


# ------------------------------------------------------------ summarize/redact


def test_recipients_become_counts_not_addresses():
    summary = summarize_args({"to": "rahul@example.com, a@b.com", "subject": "Project Update"})
    assert summary["to"] == {"count": 2}
    assert summary["subject"] == "Project Update"


def test_attendee_lists_become_counts():
    summary = summarize_args({"attendees": ["a@example.com", "b@example.com"]})
    assert summary["attendees"] == {"count": 2}


def test_bodies_become_lengths():
    summary = summarize_args({"body": "x" * 5000})
    assert summary["body"] == {"length": 5000}


def test_sensitive_keys_are_redacted():
    summary = summarize_args(
        {
            "api_key": "AIzaSyDummyKeyValue123",
            "refresh_token": "tok_abc",
            "password": "hunter2",
            "authorization": "Bearer abc.def",
            "cookie": "session=1",
        }
    )
    for key in ("api_key", "refresh_token", "password", "authorization", "cookie"):
        assert summary[key] == "[REDACTED]"


def test_token_like_values_redacted_even_under_innocuous_keys():
    summary = summarize_args({"query": "ya29.a0AfB_bySeX"})
    assert summary["query"] == "[REDACTED]"
    summary = summarize_args({"note": "use Bearer abc123 now"})
    assert summary["note"] == "[REDACTED]"


def test_long_strings_become_length_markers():
    summary = summarize_args({"note": "y" * 400})
    assert summary["note"] == "<str len=400>"


def test_plain_scalars_survive():
    summary = summarize_args({"event_id": "ev1", "limit": 5, "flag": True})
    assert summary == {"event_id": "ev1", "limit": 5, "flag": True}


def test_nested_values_are_summarized():
    summary = summarize_args({"meta": {"body": "z" * 99, "token": "abc"}})
    assert summary["meta"]["body"] == {"length": 99}
    assert summary["meta"]["token"] == "[REDACTED]"


def test_non_dict_args_become_empty_summary():
    assert summarize_args(None) == {}
    assert summarize_args([1, 2]) == {}


# ---------------------------------------------------------------- audit log


def test_audit_log_disabled_records_nothing(tmp_path):
    log = AuditLog(path=tmp_path / "a.jsonl", enabled=False)
    log.record(
        AuditEvent(
            timestamp=1.0,
            event="policy",
            action_id="ACT-1",
            session_id="s",
            tool="search_memory",
        )
    )
    assert log.events() == []
    assert not (tmp_path / "a.jsonl").exists()


def test_audit_log_writes_json_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    log = AuditLog(path=path)
    log.record(
        AuditEvent(timestamp=1.0, event="policy", action_id="ACT-1", session_id="s", tool="t")
    )
    log.record(
        AuditEvent(timestamp=2.0, event="execution", action_id="ACT-1", session_id="s", tool="t")
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["event"] for p in parsed] == ["policy", "execution"]
    assert parsed[0]["timestamp_iso"].startswith("1970-01-01")


def test_audit_buffer_is_bounded(tmp_path):
    log = AuditLog(path=None, max_events=5)
    for i in range(20):
        log.record(
            AuditEvent(timestamp=float(i), event="policy", action_id=f"ACT-{i}", session_id="s", tool="t")
        )
    assert len(log.events()) == 5


def test_events_filter_by_type(tmp_path):
    log = AuditLog(path=None)
    log.record(AuditEvent(timestamp=1.0, event="policy", action_id="A", session_id="s", tool="t"))
    log.record(AuditEvent(timestamp=1.0, event="execution", action_id="A", session_id="s", tool="t"))
    assert len(log.events("policy")) == 1
    assert len(log.events("execution")) == 1
    assert len(log.events()) == 2


# ------------------------------------------------- end-to-end through router


def test_action_recorded_with_traceable_action_id(tmp_path):
    manager, plane, router = make_setup(tmp_path)
    result = run(router.route("search_memory", {"query": "chai"}))
    assert result.success
    events = plane.audit.events()
    kinds = [e["event"] for e in events]
    assert kinds == ["policy", "execution"]
    action_id = events[0]["action_id"]
    assert re.fullmatch(r"ACT-\d{8}-[0-9A-F]{6}", action_id)
    assert events[1]["action_id"] == action_id
    assert events[0]["decision"] == "ALLOW"
    assert events[0]["permission"] == "ALLOW"
    assert events[1]["success"] is True
    assert events[1]["verification"] == "passed"
    manager.close()


def test_approval_recorded(tmp_path):
    manager, plane, router = make_setup(tmp_path)
    run(router.route("delete_memory", {"memory_id": 1}))
    approval_events = plane.audit.events("approval")
    assert len(approval_events) == 1
    assert approval_events[0]["approval_status"] == "PENDING"
    assert approval_events[0]["approval_id"].startswith("APR-")

    plane.handle_user_utterance("yes")
    run(router.route("delete_memory", {"memory_id": 1}))
    statuses = [e["approval_status"] for e in plane.audit.events("approval")]
    assert statuses[-1] == "APPROVED"
    policy_events = plane.audit.events("policy")
    assert policy_events[-1]["approval_id"].startswith("APR-")
    manager.close()


def test_execution_failure_recorded(tmp_path):
    manager, plane, router = make_setup(tmp_path)
    # SECRET_REJECTED comes from tool execution (validation passes, gate allows)
    result = run(router.route("save_memory", {"content": "my password is hunter2"}))
    assert not result.success
    failures = [e for e in plane.audit.events("execution") if not e["success"]]
    assert failures
    assert failures[-1]["error_code"] == "SECRET_REJECTED"
    assert failures[-1]["verification"] == "not_run"
    manager.close()


def test_blocked_gate_recorded(tmp_path):
    manager, plane, router = make_setup(tmp_path, security_denied_tools="delete_memory")
    result = run(router.route("delete_memory", {"memory_id": 1}))
    assert result.error["code"] == "PERMISSION_DENIED"
    policies = plane.audit.events("policy")
    assert policies[-1]["permission"] == "DENY"
    assert policies[-1]["decision"] == "APPROVAL_REQUIRED"  # permission stops it first
    assert plane.audit.events("execution") == []
    manager.close()


def test_credentials_never_reach_audit_file(tmp_path):
    manager, plane, router = make_setup(tmp_path)
    token = "ya29.a0AfDummyTokenForTests123"
    run(router.route("search_memory", {"query": token}))
    run(router.route("save_memory", {"content": f"note {token}"}))
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert token not in raw
    assert "hunter" not in raw
    for event in plane.audit.events():
        assert token not in json.dumps(event)
    manager.close()


def test_audit_file_matches_memory_buffer(tmp_path):
    manager, plane, router = make_setup(tmp_path)
    run(router.route("search_memory", {"query": "x"}))
    run(router.route("save_memory", {"content": "hello"}))
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(plane.audit.events())
    manager.close()


def test_audit_can_be_disabled(tmp_path):
    manager, plane, router = make_setup(tmp_path, audit_enabled=False)
    run(router.route("search_memory", {"query": "x"}))
    assert plane.audit.events() == []
    assert not (tmp_path / "audit.jsonl").exists()
    manager.close()
