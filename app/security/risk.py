"""RiskEngine — centralized risk classification (Phase 8 §6).

Classification order:
1. explicit per-tool overrides (config / policy rules)
2. the Phase 8 baseline table (spec §6 + §17 sensitive actions)
3. fallback from the tool's declared ``Tool.risk`` metadata
4. anything that cannot be classified raises — the caller fails closed

Risk depends on tool, operation and arguments; ``send_email`` escalates
to CRITICAL when it would fan out to many recipients (mass external
communication, §17).
"""

from __future__ import annotations

from typing import Any

from app.security.models import SecurityRisk
from app.tools.base import RiskLevel

MASS_RECIPIENT_LIMIT = 5

BASELINE: dict[str, SecurityRisk] = {
    # --- LOW: read-only ---
    "search_memory": SecurityRisk.LOW,
    "get_contact": SecurityRisk.LOW,
    "get_tasks": SecurityRisk.LOW,
    "search_web": SecurityRisk.LOW,
    "open_page": SecurityRisk.LOW,
    "read_page": SecurityRisk.LOW,
    "extract_content": SecurityRisk.LOW,
    "open_url": SecurityRisk.LOW,
    "find_element": SecurityRisk.LOW,
    "search_emails": SecurityRisk.LOW,
    "get_email": SecurityRisk.LOW,
    "list_calendar_events": SecurityRisk.LOW,
    "get_calendar_event": SecurityRisk.LOW,
    "find_calendar_availability": SecurityRisk.LOW,
    # --- MEDIUM: reversible writes / navigation ---
    "save_memory": SecurityRisk.MEDIUM,
    "save_contact": SecurityRisk.MEDIUM,
    "update_contact": SecurityRisk.MEDIUM,
    "create_task": SecurityRisk.MEDIUM,
    "draft_email": SecurityRisk.MEDIUM,
    "create_calendar_event": SecurityRisk.MEDIUM,
    "click": SecurityRisk.MEDIUM,
    "type": SecurityRisk.MEDIUM,
    "scroll": SecurityRisk.MEDIUM,
    "go_back": SecurityRisk.MEDIUM,
    "go_forward": SecurityRisk.MEDIUM,
    "reload": SecurityRisk.MEDIUM,
    "download": SecurityRisk.MEDIUM,
    # --- HIGH: external / destructive side effects (§17) ---
    "update_calendar_event": SecurityRisk.HIGH,
    "send_email": SecurityRisk.HIGH,
    "delete_calendar_event": SecurityRisk.HIGH,
    "delete_memory": SecurityRisk.HIGH,
    "complete_task": SecurityRisk.HIGH,
    "upload": SecurityRisk.HIGH,
    "external_message": SecurityRisk.HIGH,
    # --- CRITICAL: never runs without an explicitly safe mechanism (§17) ---
    "run_shell": SecurityRisk.CRITICAL,
    "shell": SecurityRisk.CRITICAL,
    "exec": SecurityRisk.CRITICAL,
    "execute_command": SecurityRisk.CRITICAL,
    "run_command": SecurityRisk.CRITICAL,
    "system_command": SecurityRisk.CRITICAL,
    "credential_access": SecurityRisk.CRITICAL,
    "read_credentials": SecurityRisk.CRITICAL,
    "bulk_delete": SecurityRisk.CRITICAL,
    "purge_all": SecurityRisk.CRITICAL,
    "grant_permission": SecurityRisk.CRITICAL,
    "set_permissions": SecurityRisk.CRITICAL,
    "modify_security_policy": SecurityRisk.CRITICAL,
}

_FALLBACK: dict[str, SecurityRisk] = {
    RiskLevel.READ.value: SecurityRisk.LOW,
    RiskLevel.WRITE.value: SecurityRisk.MEDIUM,
    RiskLevel.SENSITIVE.value: SecurityRisk.HIGH,
    RiskLevel.DESTRUCTIVE.value: SecurityRisk.HIGH,
    RiskLevel.HIGH_RISK.value: SecurityRisk.HIGH,
}


class UnknownRiskError(Exception):
    """A tool/risk combination the system cannot classify (§9: fail closed)."""


def _recipient_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        return len([p for p in value.split(",") if p.strip()])
    return 0


class RiskEngine:
    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self._overrides: dict[str, str] = dict(overrides or {})

    def add_overrides(self, overrides: dict[str, str]) -> None:
        self._overrides.update(overrides)

    def add_defaults(self, overrides: dict[str, str]) -> None:
        """Fill gaps only — explicit overrides always win."""
        for key, value in overrides.items():
            self._overrides.setdefault(key, value)

    def classify(
        self,
        tool: str,
        tool_risk: RiskLevel | str | None = None,
        args: dict | None = None,
    ) -> SecurityRisk:
        raw: Any = self._overrides.get(tool)
        if raw is None:
            raw = BASELINE.get(tool)
        if raw is None:
            declared = getattr(tool_risk, "value", tool_risk)
            raw = _FALLBACK.get(str(declared)) if declared is not None else None
        if raw is None:
            raise UnknownRiskError(f"no risk mapping for tool {tool!r}")
        try:
            risk = SecurityRisk(raw)
        except ValueError as exc:
            raise UnknownRiskError(f"invalid risk level {raw!r} for tool {tool!r}") from exc

        if risk is SecurityRisk.HIGH and tool in ("send_email", "external_message"):
            args = args or {}
            count = max(
                _recipient_count(args.get("to")),
                _recipient_count(args.get("recipients")),
                _recipient_count(args.get("attendees")),
            )
            if count > MASS_RECIPIENT_LIMIT:
                return SecurityRisk.CRITICAL
        return risk
