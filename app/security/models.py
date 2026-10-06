"""Central security models (Phase 8 §5–§11).

ActionContext / PolicyDecision / ApprovalRecord / AuditEvent are the
structured data every security decision is made from and recorded as.
Nothing here executes anything; the engines consume these models.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class SecurityRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    PERMISSION_REQUIRED = "PERMISSION_REQUIRED"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


def new_action_id(now: float | None = None) -> str:
    """Unique trace ID connecting request -> decision -> audit (§10)."""
    ts = time.strftime("%Y%m%d", time.localtime(now if now is not None else time.time()))
    return f"ACT-{ts}-{uuid.uuid4().hex[:6].upper()}"


def new_approval_id() -> str:
    return f"APR-{uuid.uuid4().hex[:10].upper()}"


@dataclass
class ActionContext:
    """One meaningful action request evaluated by the control plane."""

    action_id: str
    session_id: str
    tool: str
    args: dict
    target: str = ""
    risk: SecurityRisk = SecurityRisk.LOW
    requested_by: str = "llm"
    created_at: float = field(default_factory=time.time)


@dataclass
class PolicyDecision:
    """Structured result of policy + risk evaluation (§5)."""

    decision: Decision
    risk: SecurityRisk
    reason: str
    approval_required: bool = False
    action_id: str = ""


@dataclass
class ApprovalRecord:
    """One human-approval record (§8): explicit, bound, time-limited."""

    approval_id: str
    action_id: str
    session_id: str
    tool: str
    risk: str
    fingerprint: str
    requested_at: float
    expires_at: float
    status: ApprovalStatus = ApprovalStatus.PENDING
    summary: dict = field(default_factory=dict)
    requested_by: str = "llm"
    decided_by: str = ""
    decided_at: float | None = None
    consumed: bool = False

    def is_past_expiry(self, now: float) -> bool:
        return now >= self.expires_at


@dataclass
class AuditEvent:
    """One structured audit record (§11). Arguments arrive pre-summarized
    and pre-redacted — raw sensitive values never reach this object."""

    timestamp: float
    event: str  # "policy" | "approval" | "execution"
    action_id: str
    session_id: str
    tool: str
    risk: str = ""
    decision: str = ""
    permission: str = ""
    approval_id: str = ""
    approval_status: str = ""
    args_summary: dict = field(default_factory=dict)
    success: bool | None = None
    error_code: str = ""
    verification: str = ""
    reason: str = ""
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "action_id": self.action_id,
            "session_id": self.session_id,
            "tool": self.tool,
            "risk": self.risk,
            "decision": self.decision,
            "permission": self.permission,
            "approval_id": self.approval_id,
            "approval_status": self.approval_status,
            "args": self.args_summary,
            "success": self.success,
            "error_code": self.error_code,
            "verification": self.verification,
            "reason": self.reason,
            "duration_ms": self.duration_ms,
        }
