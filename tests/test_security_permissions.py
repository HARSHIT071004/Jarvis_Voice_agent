"""Phase 8 §24 — Permission engine tests (valid / missing / invalid /
expired / cross-session), plus fail-closed behavior."""

from __future__ import annotations

from app.security.models import ActionContext, Decision, new_action_id
from app.security.permissions import PermissionEngine, PermissionGrant


def ctx(tool: str = "send_email", session: str = "s1") -> ActionContext:
    return ActionContext(
        action_id=new_action_id(),
        session_id=session,
        tool=tool,
        args={},
    )


def test_default_allow_permits_actions():
    engine = PermissionEngine(mode="default_allow")
    decision, reason = engine.evaluate(ctx())
    assert decision is Decision.ALLOW


def test_denied_tool_is_denied():
    engine = PermissionEngine(mode="default_allow", denied_tools={"send_email"})
    decision, reason = engine.evaluate(ctx())
    assert decision is Decision.DENY
    assert "denied" in reason


def test_default_deny_with_valid_grant_allows():
    grant = PermissionGrant(
        grant_id="g1", session_id="s1", tools=frozenset({"send_email"})
    )
    engine = PermissionEngine(mode="default_deny", grants=[grant])
    decision, _ = engine.evaluate(ctx())
    assert decision is Decision.ALLOW


def test_default_deny_missing_grant_requires_permission():
    engine = PermissionEngine(mode="default_deny")
    decision, reason = engine.evaluate(ctx())
    assert decision is Decision.PERMISSION_REQUIRED
    assert "no permission" in reason


def test_invalid_grant_for_other_tool_requires_permission():
    grant = PermissionGrant(grant_id="g1", session_id="s1", tools=frozenset({"search_memory"}))
    engine = PermissionEngine(mode="default_deny", grants=[grant])
    decision, reason = engine.evaluate(ctx(tool="send_email"))
    assert decision is Decision.PERMISSION_REQUIRED


def test_expired_grant_requires_permission():
    now = [1000.0]
    grant = PermissionGrant(
        grant_id="g1",
        session_id="s1",
        tools=None,
        expires_at=1500.0,
    )
    engine = PermissionEngine(
        mode="default_deny", grants=[grant], clock=lambda: now[0]
    )
    assert engine.evaluate(ctx())[0] is Decision.ALLOW
    now[0] = 1500.0
    decision, reason = engine.evaluate(ctx())
    assert decision is Decision.PERMISSION_REQUIRED
    assert "expired" in reason


def test_cross_session_grant_does_not_transfer():
    grant = PermissionGrant(
        grant_id="g1", session_id="session-a", tools=frozenset({"send_email"})
    )
    engine = PermissionEngine(mode="default_deny", grants=[grant])
    decision, reason = engine.evaluate(ctx(session="session-b"))
    assert decision is Decision.PERMISSION_REQUIRED
    assert "another session" in reason


def test_auth_probe_reports_auth_required():
    engine = PermissionEngine(auth_probe=lambda c: "AUTH_REQUIRED")
    decision, reason = engine.evaluate(ctx(tool="search_emails"))
    assert decision is Decision.AUTH_REQUIRED
    assert reason == "AUTH_REQUIRED"


def test_auth_probe_passes_for_other_tools():
    def probe(action_ctx):
        return "AUTH_REQUIRED" if action_ctx.tool.startswith("send") else None

    engine = PermissionEngine(auth_probe=probe)
    assert engine.evaluate(ctx(tool="search_memory"))[0] is Decision.ALLOW
    assert engine.evaluate(ctx(tool="send_email"))[0] is Decision.AUTH_REQUIRED


def test_failing_auth_probe_denies():
    def probe(action_ctx):
        raise RuntimeError("probe exploded")

    engine = PermissionEngine(auth_probe=probe)
    decision, reason = engine.evaluate(ctx())
    assert decision is Decision.DENY
    assert reason == "permission evaluation failed"


def test_empty_tool_denied():
    engine = PermissionEngine()
    decision, _ = engine.evaluate(ctx(tool=""))
    assert decision is Decision.DENY


def test_invalid_mode_rejected():
    import pytest

    with pytest.raises(ValueError):
        PermissionEngine(mode="maybe")


def test_grant_covers_only_its_session():
    grant = PermissionGrant(grant_id="g", session_id="s1")
    assert grant.covers("s1", "anything", now=0.0)
    assert not grant.covers("s2", "anything", now=0.0)
