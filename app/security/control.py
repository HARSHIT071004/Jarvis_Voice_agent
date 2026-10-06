"""SecurityControlPlane — the single enforcement point (Phase 8 §4/§13).

Flow inside :meth:`authorize`:

    action context -> policy (+risk) -> permission -> approval

The router calls ``create_context`` + ``authorize`` before executing any
tool, and ``audit_execution`` after. Approval outcomes are audited as
they happen; a consumed approval marks the execution scope via the
bridge so inner domain policies do not prompt twice.

Everything fails closed (§9): a raising engine becomes DENY, a missing
approval stays PENDING (APPROVAL_REQUIRED), and nothing executes on an
error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.security.approval import (
    ApprovalManager,
    approval_prompt,
    parse_utterance,
)
from app.security.audit import AuditLog
from app.security.bridge import (
    central_approval_active,
    mark_execution_approved,
    reset_execution_approved,
)
from app.security.errors import (
    APPROVAL_DENIED,
    APPROVAL_EXPIRED,
    APPROVAL_REQUIRED,
    AUTH_REQUIRED,
    PERMISSION_DENIED,
    PERMISSION_REQUIRED,
    POLICY_DENIED,
)
from app.security.models import (
    ActionContext,
    ApprovalRecord,
    AuditEvent,
    Decision,
    PolicyDecision,
    SecurityRisk,
    new_action_id,
)
from app.security.permissions import PermissionEngine
from app.security.policy import PolicyEngine

logger = logging.getLogger("jarvis.security")

_PERMISSION_CODES = {
    Decision.DENY: PERMISSION_DENIED,
    Decision.PERMISSION_REQUIRED: PERMISSION_REQUIRED,
    Decision.AUTH_REQUIRED: AUTH_REQUIRED,
}


@dataclass
class GateOutcome:
    """Result of the control-plane gate for one tool call."""

    allowed: bool
    action_id: str
    session_id: str
    tool: str
    risk: SecurityRisk
    policy_decision: Decision
    permission_decision: Decision
    reason: str
    code: str = ""
    message: str = ""
    approval: ApprovalRecord | None = None
    approval_verdict: str = ""
    args: dict | None = None


def _derive_target(tool: str, args: dict) -> str:
    for key in (
        "url",
        "to",
        "event_id",
        "draft_id",
        "file_id",
        "target",
        "query",
        "title",
        "contact_id",
        "task_id",
    ):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value[:160]
    return ""


def build_summary(ctx: ActionContext) -> dict:
    """Trusted structured approval summary (§16) — action metadata only."""
    summary: dict = {}
    if ctx.target:
        summary["target"] = ctx.target
    args = ctx.args or {}
    to = args.get("to") or args.get("attendees") or args.get("recipients")
    if isinstance(to, str) and to.strip():
        summary["recipient_count"] = len([p for p in to.split(",") if p.strip()])
    elif isinstance(to, (list, tuple)):
        summary["recipient_count"] = len(to)
    for key in ("subject", "title", "event_id", "url"):
        value = args.get(key)
        if isinstance(value, str) and value:
            summary[key] = value[:200]
    return summary


class SecurityControlPlane:
    def __init__(
        self,
        policy: PolicyEngine,
        permissions: PermissionEngine,
        approvals: ApprovalManager,
        audit: AuditLog,
        enabled: bool = True,
        session_id: str = "default",
    ) -> None:
        self.policy = policy
        self.permissions = permissions
        self.approvals = approvals
        self.audit = audit
        self.enabled = enabled
        self.session_id = session_id
        self.approvals.session_id = session_id

    # -------------------------------------------------------------- sessions

    def new_session(self) -> str:
        """Fresh session scope: prior pending approvals are cancelled (§21)."""
        self.approvals.cancel_session(self.session_id)
        self.session_id = f"s-{new_action_id().split('-')[-1].lower()}"
        self.approvals.session_id = self.session_id
        logger.info("[SECURITY] Session scope %s", self.session_id)
        return self.session_id

    # ----------------------------------------------------------------- gate

    def create_context(self, tool: str, args: dict) -> ActionContext:
        return ActionContext(
            action_id=new_action_id(),
            session_id=self.session_id,
            tool=tool,
            args=dict(args or {}),
            target=_derive_target(tool, args or {}),
        )

    async def authorize(self, ctx: ActionContext, tool_risk=None) -> GateOutcome:
        decision = self.policy.evaluate(ctx, tool_risk)
        permission_decision, permission_reason = self.permissions.evaluate(ctx)

        outcome = GateOutcome(
            allowed=False,
            action_id=ctx.action_id,
            session_id=ctx.session_id,
            tool=ctx.tool,
            risk=ctx.risk,
            policy_decision=decision.decision,
            permission_decision=permission_decision,
            reason=decision.reason,
            args=ctx.args,
        )

        blocked: tuple[Decision, str, str] | None = None
        if permission_decision is not Decision.ALLOW:
            blocked = (permission_decision, _PERMISSION_CODES.get(permission_decision, POLICY_DENIED), permission_reason)
        elif decision.decision is Decision.DENY:
            blocked = (Decision.DENY, POLICY_DENIED, decision.reason)
        elif decision.decision is Decision.APPROVAL_REQUIRED:
            summary = build_summary(ctx)
            verdict, record = await self.approvals.resolve_action(ctx, summary)
            outcome.approval = record
            outcome.approval_verdict = verdict
            self._audit_approval(record, verdict)
            if verdict == "APPROVED":
                outcome.allowed = True
                outcome.reason = decision.reason
                self._audit_policy(outcome, permission_reason)
                return outcome
            codes = {
                "PENDING": APPROVAL_REQUIRED,
                "DENIED": APPROVAL_DENIED,
                "EXPIRED": APPROVAL_EXPIRED,
            }
            messages = {
                "PENDING": (
                    approval_prompt(record)
                    + "\nTell the user exactly this and wait for their explicit yes. "
                    "After the user approves in their own words, call the same tool "
                    "again with the same arguments."
                ),
                "DENIED": (
                    "The user denied this action; it was NOT performed. "
                    "Tell the user politely and do not retry it in this conversation."
                ),
                "EXPIRED": (
                    "The approval expired before the action ran, so nothing happened. "
                    "Ask the user for approval again."
                ),
            }
            outcome.code = codes[verdict]
            outcome.message = messages[verdict]
            outcome.reason = record.status.value.lower() + " approval"
            self._audit_policy(outcome, permission_reason)
            return outcome

        if blocked is None and decision.decision is not Decision.ALLOW:
            blocked = (Decision.DENY, POLICY_DENIED, decision.reason)

        if blocked is not None:
            _, code, reason = blocked
            outcome.code = code
            outcome.message = self._deny_message(code, ctx, reason)
            outcome.reason = reason
            self._audit_policy(outcome, permission_reason)
            logger.info(
                "[SECURITY] Gate DENY tool=%s code=%s", ctx.tool, code
            )
            return outcome

        outcome.allowed = True
        self._audit_policy(outcome, permission_reason)
        return outcome

    @staticmethod
    def _deny_message(code: str, ctx: ActionContext, reason: str) -> str:
        if code == POLICY_DENIED:
            return (
                f"The action '{ctx.tool}' is blocked by security policy ({reason}). "
                "Do not attempt it another way; tell the user it is not possible."
            )
        if code == PERMISSION_DENIED:
            return (
                f"Permission for '{ctx.tool}' is denied ({reason}). "
                "Tell the user this action is not permitted right now."
            )
        if code == PERMISSION_REQUIRED:
            return (
                f"'{ctx.tool}' needs a permission that is missing or expired ({reason}). "
                "Tell the user the action was not performed."
            )
        if code == AUTH_REQUIRED:
            return (
                f"'{ctx.tool}' cannot run: authentication is missing ({reason}). "
                "Tell the user to connect the account first; never claim success."
            )
        return reason or "blocked by security policy"

    # ----------------------------------------------------------------- audit

    def _audit_policy(self, outcome: GateOutcome, permission_reason: str) -> None:
        self.audit.record(
            AuditEvent(
                timestamp=self.approvals.now(),
                event="policy",
                action_id=outcome.action_id,
                session_id=outcome.session_id,
                tool=outcome.tool,
                risk=outcome.risk.value,
                decision=outcome.policy_decision.value,
                permission=outcome.permission_decision.value,
                approval_id=outcome.approval.approval_id if outcome.approval else "",
                approval_status=outcome.approval_verdict,
                args_summary=dict(outcome.args or {}),
                reason=outcome.reason,
            )
        )

    def _audit_approval(self, record: ApprovalRecord, verdict: str) -> None:
        self.audit.record(
            AuditEvent(
                timestamp=self.approvals.now(),
                event="approval",
                action_id=record.action_id,
                session_id=record.session_id,
                tool=record.tool,
                risk=record.risk,
                approval_id=record.approval_id,
                approval_status=verdict,
                args_summary=record.summary,
                reason=record.decided_by or record.status.value,
            )
        )

    def audit_execution(
        self,
        outcome: GateOutcome,
        *,
        success: bool,
        error_code: str = "",
        duration_ms: int = 0,
        verification: str = "",
    ) -> None:
        self.audit.record(
            AuditEvent(
                timestamp=self.approvals.now(),
                event="execution",
                action_id=outcome.action_id,
                session_id=outcome.session_id,
                tool=outcome.tool,
                risk=outcome.risk.value,
                decision=outcome.policy_decision.value,
                permission=outcome.permission_decision.value,
                approval_id=outcome.approval.approval_id if outcome.approval else "",
                approval_status=outcome.approval_verdict,
                args_summary=dict(outcome.args or {}),
                success=success,
                error_code=error_code,
                verification=verification,
                duration_ms=duration_ms,
                reason=outcome.reason,
            )
        )

    # ----------------------------------------------------------------- voice

    def handle_user_utterance(self, text: str, decided_by: str = "user") -> str | None:
        """Trusted hook: a human's live answer to a pending approval (§23).

        Only the voice/text front door calls this with user-role input —
        there is no tool or model-facing API for approvals (§7/§18).
        """
        verdict = self.approvals.resolve_utterance(
            text, session_id=self.session_id, decided_by=decided_by
        )
        if verdict is None:
            return None
        decided = sorted(
            (
                r
                for r in self.approvals.all_records()
                if r.session_id == self.session_id
                and r.decided_by == decided_by
                and r.decided_at is not None
            ),
            key=lambda r: r.decided_at or 0.0,
        )
        if decided:
            self._audit_approval(decided[-1], verdict.upper())
        logger.info("[SECURITY] User %s pending approval", verdict)
        return verdict

    # --------------------------------------------------------------- bridging

    def mark_execution_approved(self):
        return mark_execution_approved()

    @staticmethod
    def reset_execution_approved(token) -> None:
        reset_execution_approved(token)

    def browser_approval_bridge(self):
        """ApprovalHandler for BrowserPolicy (Phase 6 inner seam).

        Central-approved executions pass through; otherwise the pending
        approval lives in the shared ApprovalManager so the voice flow
        can resolve it.
        """

        async def handler(request) -> bool:
            if central_approval_active():
                return True
            verdict, record = await self.approvals.resolve_request(
                self.session_id,
                operation=request.action,
                target=request.target,
                detail=request.detail or request.url,
                risk="MEDIUM",
                summary={"target": request.target, "url": request.url},
            )
            self._audit_approval(record, verdict)
            if verdict == "APPROVED":
                return True
            if verdict in ("PENDING", "DENIED", "EXPIRED"):
                logger.info(
                    "[SECURITY] Browser action %s needs approval (%s)",
                    request.action,
                    verdict,
                )
            return False

        return handler
