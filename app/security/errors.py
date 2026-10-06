"""Controlled error codes for security refusals (Phase 8 §9/§13).

Every gate refusal becomes one of these stable codes — raw internal
reasons never reach the model, and every code fails closed.
"""

from __future__ import annotations

POLICY_DENIED = "POLICY_DENIED"
PERMISSION_DENIED = "PERMISSION_DENIED"
PERMISSION_REQUIRED = "PERMISSION_REQUIRED"
AUTH_REQUIRED = "AUTH_REQUIRED"
APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
APPROVAL_DENIED = "APPROVAL_DENIED"
APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
RISK_BLOCKED = "RISK_BLOCKED"

GATE_CODES = frozenset(
    {
        POLICY_DENIED,
        PERMISSION_DENIED,
        PERMISSION_REQUIRED,
        AUTH_REQUIRED,
        APPROVAL_REQUIRED,
        APPROVAL_DENIED,
        APPROVAL_EXPIRED,
        RISK_BLOCKED,
    }
)
