"""Security control plane (Phase 8 §4): policy, risk, permissions,
human approval and audit — the single authorization point every tool
call passes through before execution.

The invariant (§30):

    No high-risk external side effect may execute without passing the
    centralized security control plane and, when required, explicit
    human approval. The LLM may request an action; it may never
    authorize itself. Security failures fail closed.

This package initializer stays lightweight (models / errors / bridge
only); the engines that reach into app.tools are imported lazily so
``app.security.bridge`` can be used from any module without import
cycles.
"""

from __future__ import annotations

from pathlib import Path

from app.security.approval import (
    ApprovalManager,
    DecisionRequest,
    approval_prompt,
    parse_utterance,
)
from app.security.audit import AuditLog, summarize_args
from app.security.bridge import central_approval_active
from app.security.errors import (
    APPROVAL_DENIED,
    APPROVAL_EXPIRED,
    APPROVAL_REQUIRED,
    AUTH_REQUIRED,
    GATE_CODES,
    PERMISSION_DENIED,
    PERMISSION_REQUIRED,
    POLICY_DENIED,
    RISK_BLOCKED,
)
from app.security.models import (
    ActionContext,
    ApprovalRecord,
    ApprovalStatus,
    AuditEvent,
    Decision,
    PolicyDecision,
    SecurityRisk,
)
from app.security.permissions import PermissionEngine, PermissionGrant

__all__ = [
    "APPROVAL_DENIED",
    "APPROVAL_EXPIRED",
    "APPROVAL_REQUIRED",
    "AUTH_REQUIRED",
    "ActionContext",
    "ApprovalManager",
    "ApprovalRecord",
    "ApprovalStatus",
    "AuditEvent",
    "AuditLog",
    "BrowserPolicyRules",
    "Decision",
    "DecisionRequest",
    "GATE_CODES",
    "GateOutcome",
    "GeneralToolPolicyRules",
    "MemoryPolicyRules",
    "PERMISSION_DENIED",
    "PERMISSION_REQUIRED",
    "POLICY_DENIED",
    "PermissionEngine",
    "PermissionGrant",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyRule",
    "ProductivityPolicyRules",
    "RISK_BLOCKED",
    "ResearchPolicyRules",
    "RiskEngine",
    "SecurityControlPlane",
    "SecurityRisk",
    "UnknownRiskError",
    "approval_prompt",
    "build_control_plane",
    "build_summary",
    "central_approval_active",
    "default_rules",
    "parse_utterance",
    "summarize_args",
]

_LAZY = {
    "BrowserPolicyRules": ("app.security.policy", "BrowserPolicyRules"),
    "GeneralToolPolicyRules": ("app.security.policy", "GeneralToolPolicyRules"),
    "MemoryPolicyRules": ("app.security.policy", "MemoryPolicyRules"),
    "PolicyEngine": ("app.security.policy", "PolicyEngine"),
    "PolicyRule": ("app.security.policy", "PolicyRule"),
    "ProductivityPolicyRules": ("app.security.policy", "ProductivityPolicyRules"),
    "ResearchPolicyRules": ("app.security.policy", "ResearchPolicyRules"),
    "default_rules": ("app.security.policy", "default_rules"),
    "RiskEngine": ("app.security.risk", "RiskEngine"),
    "UnknownRiskError": ("app.security.risk", "UnknownRiskError"),
    "GateOutcome": ("app.security.control", "GateOutcome"),
    "SecurityControlPlane": ("app.security.control", "SecurityControlPlane"),
    "build_summary": ("app.security.control", "build_summary"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def build_control_plane(
    settings,
    *,
    auth_probe=None,
    audit_path: str | Path | None = None,
):
    """Wire the full control plane from application settings (§22).

    Defaults are secure: approval required for HIGH risk, critical
    actions denied, audit enabled with redaction on.
    """
    from app.security.control import SecurityControlPlane
    from app.security.policy import PolicyEngine
    from app.security.risk import RiskEngine  # noqa: F401  (ensures import order)

    policy = PolicyEngine(
        approval_required_for_high_risk=bool(
            getattr(settings, "approval_required_for_high_risk", True)
        ),
        critical_action_mode=str(getattr(settings, "critical_action_mode", "deny")),
    )
    denied = {
        t.strip()
        for t in str(getattr(settings, "security_denied_tools", "")).split(",")
        if t.strip()
    }
    permissions = PermissionEngine(
        mode="default_allow", denied_tools=denied, auth_probe=auth_probe
    )
    approvals = ApprovalManager(
        ttl_seconds=float(getattr(settings, "approval_timeout_seconds", 120.0))
    )
    audit_enabled = bool(getattr(settings, "audit_enabled", True))
    path = audit_path
    if path is None:
        path = getattr(settings, "audit_path", Path("data/audit.jsonl"))
    audit = AuditLog(
        path=path if audit_enabled else None,
        enabled=audit_enabled,
        redact=bool(getattr(settings, "audit_redact_sensitive_data", True)),
    )
    return SecurityControlPlane(
        policy,
        permissions,
        approvals,
        audit,
        enabled=bool(getattr(settings, "security_enabled", True)),
    )
