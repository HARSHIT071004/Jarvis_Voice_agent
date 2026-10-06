"""Tool Router — the security boundary for all tool execution.

Flow: identify tool → validate arguments → check risk → centralized
security gate (Phase 8: policy → risk → permission → approval) →
execute with timeout → validate result → audit → return structured
result.

The LLM is untrusted input (spec section 27). Raw exceptions never
reach the model; every failure becomes a structured {code, message}.

Risk policy: `blocked_risks` lets a caller refuse classes of tools.
When a SecurityControlPlane is attached, every call additionally passes
through the single authorization point (§13) — approvals, permissions
and audit apply uniformly to all tools.

Audit logging: [TOOL] lines for validation / risk / execution, plus
structured AuditEvents when a control plane is attached.
Nothing sensitive is ever logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.security.bridge import mark_execution_approved, reset_execution_approved
from app.tools.base import RiskLevel, Tool, ToolArgs, ToolError
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.security.control import GateOutcome, SecurityControlPlane

log = logging.getLogger("jarvis.tool")

DEFAULT_EXECUTION_TIMEOUT = 10.0


@dataclass
class ToolResult:
    """Structured tool result (spec section 13)."""

    tool: str
    success: bool
    data: Any = None
    error: dict | None = None
    risk: str | None = None
    duration_ms: int = 0
    call_id: str | None = None  # provider tool-call id, echoed on responses

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "tool": self.tool,
            "data": self.data,
            "error": self.error,
        }


@dataclass
class ToolRouter:
    registry: ToolRegistry
    execution_timeout: float = DEFAULT_EXECUTION_TIMEOUT
    blocked_risks: set[RiskLevel] = field(default_factory=set)
    security: "SecurityControlPlane | None" = None

    async def route(self, name: str, arguments: dict | None) -> ToolResult:
        """Validate, permission-check, and execute a tool call safely."""
        started = time.monotonic()
        arguments = arguments if isinstance(arguments, dict) else {}

        # 1. Tool exists?
        tool = self.registry.get(name)
        if tool is None:
            log.warning("[TOOL] Unknown tool: %s", name)
            return self._error(name, "TOOL_NOT_FOUND", f"No tool named '{name}' is available.", started)

        # 2. Validate arguments (LLM cannot bypass this).
        try:
            args: ToolArgs = tool.validate(arguments)
        except ToolError as exc:
            log.info("[TOOL] Validation: FAIL tool=%s error=%s", name, exc.code)
            return self._error(name, exc.code, exc.message, started, risk=tool.risk)
        log.info("[TOOL] Validation: PASS tool=%s", name)

        # 3. Risk / permission check.
        log.info("[TOOL] Risk: %s", tool.risk.value)
        if tool.risk in self.blocked_risks:
            log.info("[TOOL] Permission: DENY tool=%s risk=%s", name, tool.risk.value)
            return self._error(
                name,
                "RISK_BLOCKED",
                f"The '{name}' action (risk {tool.risk.value}) is not currently allowed.",
                started,
                risk=tool.risk,
            )

        # 4. Phase 8: centralized security control plane (§13).
        gate: GateOutcome | None = None
        if self.security is not None and self.security.enabled:
            ctx = self.security.create_context(name, args.model_dump())
            gate = await self.security.authorize(ctx, tool.risk)
            if not gate.allowed:
                log.info("[TOOL] Security: DENY tool=%s code=%s", name, gate.code)
                return self._error(name, gate.code, gate.message, started, risk=gate.risk.value)

        # 5. Mark the execution scope when a human approval was consumed,
        # then execute inside a timeout boundary.
        token = None
        if gate is not None and gate.approval is not None and gate.approval_verdict == "APPROVED":
            token = mark_execution_approved()

        def _finish(result: ToolResult, verification: str = "") -> ToolResult:
            if gate is not None and self.security is not None:
                code = "" if result.success else (result.error or {}).get("code", "")
                self.security.audit_execution(
                    gate,
                    success=result.success,
                    error_code=code,
                    duration_ms=result.duration_ms,
                    verification=verification,
                )
            return result

        try:
            try:
                data = await asyncio.wait_for(tool.execute(args), timeout=self.execution_timeout)
            except asyncio.TimeoutError:
                log.info("[TOOL] Execution: TIMEOUT tool=%s", name)
                return _finish(
                    self._error(
                        name,
                        "TOOL_TIMEOUT",
                        f"Tool '{name}' exceeded {self.execution_timeout:.0f}s and was stopped.",
                        started,
                        risk=tool.risk,
                    ),
                    "not_run",
                )
            except ToolError as exc:
                log.info("[TOOL] Execution: FAILED tool=%s code=%s", name, exc.code)
                return _finish(
                    self._error(name, exc.code, exc.message, started, risk=tool.risk),
                    "not_run",
                )
            except Exception:
                log.exception("[TOOL] Execution: CRASH tool=%s", name)
                return _finish(
                    self._error(
                        name, "TOOL_INTERNAL_ERROR", f"Tool '{name}' failed unexpectedly.", started, risk=tool.risk
                    ),
                    "not_run",
                )

            # 6. Result must be JSON-serializable for the model.
            safe_data = self._serialize(data)
            if safe_data is None and data is not None:
                log.info("[TOOL] Execution: FAILED tool=%s code=INVALID_RESULT", name)
                return _finish(
                    self._error(
                        name, "INVALID_RESULT", f"Tool '{name}' returned a non-serializable result.", started, risk=tool.risk
                    ),
                    "failed",
                )

            duration = int((time.monotonic() - started) * 1000)
            log.info("[TOOL] Execution: SUCCESS tool=%s duration_ms=%d", name, duration)
            return _finish(
                ToolResult(
                    tool=name, success=True, data=safe_data, risk=tool.risk.value, duration_ms=duration
                ),
                "passed",
            )
        finally:
            if token is not None:
                reset_execution_approved(token)

    def _error(self, name: str, code: str, message: str, started: float, risk: str | None = None) -> ToolResult:
        duration = int((time.monotonic() - started) * 1000)
        return ToolResult(
            tool=name,
            success=False,
            error={"code": code, "message": message},
            risk=risk,
            duration_ms=duration,
        )

    @staticmethod
    def _serialize(data: Any) -> Any:
        """Return a JSON-safe view of tool data, or None if impossible."""
        if data is None or isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, dict):
            out = {}
            for key, value in data.items():
                if not isinstance(key, str):
                    return None
                safe = ToolRouter._serialize(value)
                if safe is None and value is not None:
                    return None
                out[key] = safe
            return out
        if isinstance(data, (list, tuple)):
            out_list = []
            for item in data:
                safe = ToolRouter._serialize(item)
                if safe is None and item is not None:
                    return None
                out_list.append(safe)
            return out_list
        if hasattr(data, "model_dump"):  # Pydantic model
            return ToolRouter._serialize(data.model_dump())
        return None
