"""Phase 8 §24 — Policy and risk engine tests."""

from __future__ import annotations

import asyncio

import pytest

from app.security.models import ActionContext, Decision, SecurityRisk, new_action_id
from app.security.policy import PolicyEngine, PolicyRule, default_rules
from app.security.risk import RiskEngine, UnknownRiskError
from app.tools.base import RiskLevel


def run(coro):
    return asyncio.run(coro)


def ctx(tool: str, args: dict | None = None, session: str = "s1") -> ActionContext:
    return ActionContext(
        action_id=new_action_id(),
        session_id=session,
        tool=tool,
        args=dict(args or {}),
    )


# ------------------------------------------------------------------- risk


def test_low_risk_classification():
    engine = RiskEngine()
    assert engine.classify("search_memory", RiskLevel.READ) is SecurityRisk.LOW
    assert engine.classify("open_url", RiskLevel.READ) is SecurityRisk.LOW


def test_medium_risk_classification():
    engine = RiskEngine()
    assert engine.classify("create_task", RiskLevel.WRITE) is SecurityRisk.MEDIUM
    assert engine.classify("draft_email", RiskLevel.WRITE) is SecurityRisk.MEDIUM


def test_high_risk_overrides_declared_metadata():
    # update_calendar_event is declared WRITE but spec §6 makes it HIGH.
    engine = RiskEngine()
    assert engine.classify("update_calendar_event", RiskLevel.WRITE) is SecurityRisk.HIGH
    assert engine.classify("send_email", RiskLevel.HIGH_RISK) is SecurityRisk.HIGH
    assert engine.classify("delete_memory", RiskLevel.DESTRUCTIVE) is SecurityRisk.HIGH
    assert engine.classify("upload", RiskLevel.SENSITIVE) is SecurityRisk.HIGH


def test_fallback_by_declared_risk():
    engine = RiskEngine()
    assert engine.classify("wipe_record", RiskLevel.DESTRUCTIVE) is SecurityRisk.HIGH
    assert engine.classify("odd_tool", RiskLevel.SENSITIVE) is SecurityRisk.HIGH
    assert engine.classify("odd_tool", RiskLevel.WRITE) is SecurityRisk.MEDIUM
    assert engine.classify("odd_tool", RiskLevel.READ) is SecurityRisk.LOW


def test_critical_tool_names_are_critical():
    engine = RiskEngine()
    assert engine.classify("run_shell", RiskLevel.WRITE) is SecurityRisk.CRITICAL
    assert engine.classify("grant_permission", RiskLevel.WRITE) is SecurityRisk.CRITICAL


def test_unknown_risk_raises():
    engine = RiskEngine(overrides={"mystery": "BANANA"})
    with pytest.raises(UnknownRiskError):
        engine.classify("mystery", RiskLevel.READ)
    with pytest.raises(UnknownRiskError):
        RiskEngine().classify("no_such_tool", tool_risk=None)


def test_mass_recipients_escalate_to_critical():
    engine = RiskEngine()
    to = ",".join(f"user{i}@example.com" for i in range(6))
    assert engine.classify("send_email", RiskLevel.HIGH_RISK, {"to": to}) is SecurityRisk.CRITICAL
    assert (
        engine.classify("send_email", RiskLevel.HIGH_RISK, {"to": "a@example.com"})
        is SecurityRisk.HIGH
    )


# ----------------------------------------------------------------- policy


def test_low_risk_allow():
    engine = PolicyEngine()
    decision = engine.evaluate(ctx("search_memory", {"query": "x"}), RiskLevel.READ)
    assert decision.decision is Decision.ALLOW
    assert decision.risk is SecurityRisk.LOW
    assert decision.approval_required is False


def test_medium_risk_allow_without_approval():
    engine = PolicyEngine()
    decision = engine.evaluate(ctx("create_task", {"content": "t"}), RiskLevel.WRITE)
    assert decision.decision is Decision.ALLOW
    assert decision.risk is SecurityRisk.MEDIUM


def test_high_risk_requires_approval():
    engine = PolicyEngine()
    decision = engine.evaluate(
        ctx("send_email", {"to": "rahul@example.com", "subject": "Hi"}), RiskLevel.HIGH_RISK
    )
    assert decision.decision is Decision.APPROVAL_REQUIRED
    assert decision.approval_required is True
    assert decision.risk is SecurityRisk.HIGH
    assert "approval" in decision.reason


def test_critical_denied_by_default():
    engine = PolicyEngine()
    decision = engine.evaluate(ctx("run_shell", {"command": "ls"}), RiskLevel.WRITE)
    assert decision.decision is Decision.DENY
    assert decision.risk is SecurityRisk.CRITICAL


def test_critical_allow_mode_still_needs_approval():
    engine = PolicyEngine(critical_action_mode="allow")
    decision = engine.evaluate(ctx("run_shell", {"command": "ls"}), RiskLevel.WRITE)
    assert decision.decision is Decision.APPROVAL_REQUIRED
    assert decision.risk is SecurityRisk.CRITICAL


def test_invalid_critical_mode_rejected():
    with pytest.raises(ValueError):
        PolicyEngine(critical_action_mode="maybe")


def test_high_approval_can_be_configured_off():
    engine = PolicyEngine(approval_required_for_high_risk=False)
    decision = engine.evaluate(ctx("send_email", {"to": "a@example.com"}), RiskLevel.HIGH_RISK)
    assert decision.decision is Decision.ALLOW


def test_unknown_tool_with_unknown_metadata_denies():
    engine = PolicyEngine()
    decision = engine.evaluate(ctx("mystery_tool", {}), tool_risk=None)
    assert decision.decision is Decision.DENY
    assert "classify" in decision.reason


def test_malformed_policy_rule_fails_closed():
    class BoomRule(PolicyRule):
        name = "boom"

        def check(self, ctx_arg, risk):
            raise RuntimeError("rule exploded")

    engine = PolicyEngine(rules=[BoomRule()])
    decision = engine.evaluate(ctx("search_memory", {"query": "x"}), RiskLevel.READ)
    assert decision.decision is Decision.DENY
    assert decision.reason == "policy evaluation failed"


def test_unknown_risk_override_fails_closed():
    engine = PolicyEngine(risk_engine=RiskEngine(overrides={"search_memory": "BANANA"}))
    decision = engine.evaluate(ctx("search_memory", {"query": "x"}), RiskLevel.READ)
    assert decision.decision is Decision.DENY
    assert "classify" in decision.reason


def test_empty_tool_name_denied_by_general_rule():
    engine = PolicyEngine()
    decision = engine.evaluate(ctx(""), RiskLevel.READ)
    assert decision.decision is Decision.DENY


def test_default_rules_cover_all_domains():
    names = {r.name for r in default_rules()}
    assert names == {"general", "memory", "productivity", "browser", "research"}


def test_rule_overrides_beat_baseline():
    engine = PolicyEngine()
    decision = engine.evaluate(ctx("upload", {"target": "f", "file_id": "1"}), RiskLevel.SENSITIVE)
    assert decision.risk is SecurityRisk.HIGH
    assert decision.decision is Decision.APPROVAL_REQUIRED
