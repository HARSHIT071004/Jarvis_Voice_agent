"""PermissionEngine — explicit permission evaluation (Phase 8 §7).

Answers "is Jarvis allowed to perform this action?" for the current
session. Permission comes only from configured rules, grants, and
authenticated state — never from model-generated text.

Modes:
- ``default_allow`` (production): everything is allowed unless explicitly
  denied (deny list) or an auth probe reports missing authentication.
- ``default_deny``: a session-scoped, time-limited grant is required;
  missing, invalid, expired, or cross-session grants all fail closed
  with PERMISSION_REQUIRED.

Any internal failure yields DENY (§9).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from app.security.models import ActionContext, Decision

logger = logging.getLogger("jarvis.security")

AuthProbe = Callable[[ActionContext], str | None]


@dataclass
class PermissionGrant:
    """A session-scoped, expiring permission for specific tools."""

    grant_id: str
    session_id: str
    tools: frozenset[str] | None = None  # None = all tools in the session
    expires_at: float | None = None
    granted_by: str = "config"

    def covers(self, session_id: str, tool: str, now: float) -> bool:
        if session_id != self.session_id:
            return False
        if self.tools is not None and tool not in self.tools:
            return False
        if self.expires_at is not None and now >= self.expires_at:
            return False
        return True

    def touches(self, session_id: str, tool: str) -> bool:
        if session_id != self.session_id:
            return False
        return self.tools is None or tool in self.tools


class PermissionEngine:
    def __init__(
        self,
        mode: str = "default_allow",
        grants: Iterable[PermissionGrant] = (),
        denied_tools: Iterable[str] = (),
        auth_probe: AuthProbe | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        mode = (mode or "default_allow").strip().lower()
        if mode not in {"default_allow", "default_deny"}:
            raise ValueError(f"permission mode must be default_allow or default_deny, got {mode!r}")
        self.mode = mode
        self._grants = list(grants)
        self._denied = frozenset(denied_tools)
        self.auth_probe = auth_probe
        self._clock = clock

    @property
    def grants(self) -> list[PermissionGrant]:
        return list(self._grants)

    def add_grant(self, grant: PermissionGrant) -> None:
        self._grants.append(grant)

    def evaluate(self, ctx: ActionContext) -> tuple[Decision, str]:
        try:
            if not ctx.tool:
                return Decision.DENY, "action has no tool"
            if self.auth_probe is not None:
                code = self.auth_probe(ctx)
                if code:
                    return Decision.AUTH_REQUIRED, str(code)
            if ctx.tool in self._denied:
                return Decision.DENY, f"tool '{ctx.tool}' is denied by policy"
            if self.mode == "default_deny":
                now = self._clock()
                for grant in self._grants:
                    if grant.covers(ctx.session_id, ctx.tool, now):
                        return Decision.ALLOW, "permission granted"
                expired = any(
                    grant.touches(ctx.session_id, ctx.tool)
                    and grant.expires_at is not None
                    and now >= grant.expires_at
                    for grant in self._grants
                )
                if expired:
                    return Decision.PERMISSION_REQUIRED, "permission expired"
                covered_elsewhere = any(
                    grant.session_id != ctx.session_id
                    and (grant.tools is None or ctx.tool in grant.tools)
                    for grant in self._grants
                )
                if covered_elsewhere:
                    return Decision.PERMISSION_REQUIRED, "permission belongs to another session"
                return Decision.PERMISSION_REQUIRED, "no permission for this action"
            return Decision.ALLOW, "allowed by policy"
        except Exception:
            logger.exception("[SECURITY] Permission evaluation failed; denying action=%s", ctx.tool)
            return Decision.DENY, "permission evaluation failed"
