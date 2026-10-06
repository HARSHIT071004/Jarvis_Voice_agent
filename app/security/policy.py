"""PolicyEngine — the single policy evaluation point (Phase 8 §5/§14).

Composes pluggable rule classes (General / Memory / Productivity /
Browser / Research) over the RiskEngine and returns a structured
PolicyDecision. Evaluation is fail-closed: a raising rule, an unknown
risk level, or any internal failure yields DENY — never a silent ALLOW
(§9).
"""

from __future__ import annotations

import logging

from app.security.models import (
    ActionContext,
    Decision,
    PolicyDecision,
    SecurityRisk,
)
from app.security.risk import RiskEngine, UnknownRiskError

logger = logging.getLogger("jarvis.security")


class PolicyRule:
    """Base rule: contributes risk overrides and may veto an action."""

    name = "rule"
    overrides: dict[str, str] = {}

    def check(self, ctx: ActionContext, risk: SecurityRisk) -> PolicyDecision | None:
        return None


class GeneralToolPolicyRules(PolicyRule):
    name = "general"

    def check(self, ctx: ActionContext, risk: SecurityRisk) -> PolicyDecision | None:
        if not ctx.tool:
            return PolicyDecision(
                decision=Decision.DENY,
                risk=risk,
                reason="action has no tool",
                action_id=ctx.action_id,
            )
        return None


class MemoryPolicyRules(PolicyRule):
    name = "memory"
    overrides = {
        "delete_memory": SecurityRisk.HIGH.value,
        "save_memory": SecurityRisk.MEDIUM.value,
        "search_memory": SecurityRisk.LOW.value,
    }


class ProductivityPolicyRules(PolicyRule):
    name = "productivity"
    overrides = {
        "send_email": SecurityRisk.HIGH.value,
        "delete_calendar_event": SecurityRisk.HIGH.value,
        "update_calendar_event": SecurityRisk.HIGH.value,
        "draft_email": SecurityRisk.MEDIUM.value,
        "create_calendar_event": SecurityRisk.MEDIUM.value,
        "search_emails": SecurityRisk.LOW.value,
        "get_email": SecurityRisk.LOW.value,
        "list_calendar_events": SecurityRisk.LOW.value,
        "get_calendar_event": SecurityRisk.LOW.value,
        "find_calendar_availability": SecurityRisk.LOW.value,
    }


class BrowserPolicyRules(PolicyRule):
    name = "browser"
    overrides = {
        "upload": SecurityRisk.HIGH.value,
        "open_url": SecurityRisk.LOW.value,
        "read_page": SecurityRisk.LOW.value,
        "find_element": SecurityRisk.LOW.value,
    }


class ResearchPolicyRules(PolicyRule):
    name = "research"
    overrides = {
        "search_web": SecurityRisk.LOW.value,
        "open_page": SecurityRisk.LOW.value,
        "extract_content": SecurityRisk.LOW.value,
    }


def default_rules() -> list[PolicyRule]:
    return [
        GeneralToolPolicyRules(),
        MemoryPolicyRules(),
        ProductivityPolicyRules(),
        BrowserPolicyRules(),
        ResearchPolicyRules(),
    ]


class PolicyEngine:
    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
        *,
        risk_engine: RiskEngine | None = None,
        approval_required_for_high_risk: bool = True,
        critical_action_mode: str = "deny",
    ) -> None:
        self._rules = list(default_rules() if rules is None else rules)
        self._risk = risk_engine or RiskEngine()
        merged: dict[str, str] = {}
        for rule in self._rules:
            merged.update(rule.overrides)
        self._risk.add_defaults(merged)
        self.approval_required_for_high_risk = approval_required_for_high_risk
        mode = (critical_action_mode or "deny").strip().lower()
        if mode not in {"deny", "allow"}:
            raise ValueError(f"critical_action_mode must be deny or allow, got {critical_action_mode!r}")
        self.critical_action_mode = mode

    @property
    def risk_engine(self) -> RiskEngine:
        return self._risk

    def evaluate(self, ctx: ActionContext, tool_risk=None) -> PolicyDecision:
        """Policy -> risk, fail-closed (§9)."""
        try:
            risk = self._risk.classify(ctx.tool, tool_risk, ctx.args)
            ctx.risk = risk
            for rule in self._rules:
                veto = rule.check(ctx, risk)
                if veto is not None:
                    return veto
            if risk is SecurityRisk.CRITICAL:
                if self.critical_action_mode == "deny":
                    return PolicyDecision(
                        decision=Decision.DENY,
                        risk=risk,
                        reason="critical actions are blocked by policy",
                        action_id=ctx.action_id,
                    )
                return PolicyDecision(
                    decision=Decision.APPROVAL_REQUIRED,
                    risk=risk,
                    reason="critical action requires explicit approval",
                    approval_required=True,
                    action_id=ctx.action_id,
                )
            if risk is SecurityRisk.HIGH and self.approval_required_for_high_risk:
                return PolicyDecision(
                    decision=Decision.APPROVAL_REQUIRED,
                    risk=risk,
                    reason="high-risk action requires human approval",
                    approval_required=True,
                    action_id=ctx.action_id,
                )
            return PolicyDecision(
                decision=Decision.ALLOW,
                risk=risk,
                reason=f"{risk.value} risk action",
                action_id=ctx.action_id,
            )
        except UnknownRiskError as exc:
            logger.warning("[SECURITY] Unknown risk; denying action=%s", ctx.tool)
            return PolicyDecision(
                decision=Decision.DENY,
                risk=SecurityRisk.CRITICAL,
                reason=f"cannot classify risk: {exc}",
                action_id=ctx.action_id,
            )
        except Exception:
            logger.exception("[SECURITY] Policy evaluation failed; denying action=%s", ctx.tool)
            return PolicyDecision(
                decision=Decision.DENY,
                risk=SecurityRisk.CRITICAL,
                reason="policy evaluation failed",
                action_id=ctx.action_id,
            )
