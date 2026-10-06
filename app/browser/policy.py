"""BrowserPolicy — the navigation/action boundary (§17/§19/§23).

Deliberately simple and extensible: domain/scheme checks plus a
sensitive-action classifier. Approval is a callable seam — Phase 8 will
wire real human-in-the-loop approval; with no handler configured,
sensitive actions are BLOCKED, never auto-executed (§19).

This policy never executes anything itself; the controller consults it
before acting and turns refusals into controlled BrowserErrors.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.security.bridge import central_approval_active

logger = logging.getLogger("jarvis.browser")

# Action semantics that must never run without a user decision (§19).
_SENSITIVE = re.compile(
    r"\b(buy|purchase|pay|checkout|place order|order now|donate|submit|"
    r"apply|send|delete|remove|erase|transfer|withdraw|deposit|"
    r"sign|agree|accept terms|confirm order|publish|post|upload|"
    r"change password|password|credential|logout|log out|close account|"
    r"cancel subscription|subscribe now)\b",
    re.I,
)

# Click targets that are obviously harmless navigation (never "sensitive"
# merely because a page hides a menu behind words like "Next").
_HARMLESS = re.compile(r"^\s*(next|back|continue|more|read more|show|hide|expand|collapse|menu|search|cancel|close)\b", re.I)

_DEFAULT_BLOCKED_DOMAINS = (
    "localhost", "metadata.google.internal",
)


def _host_is_public(host: str) -> bool:
    """True only for globally routable addresses (mirrors Phase 5)."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_global


def _resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return []
    return sorted({info[4][0] for info in infos})


@dataclass
class ApprovalRequest:
    """What a future approval flow must see before deciding (§18)."""

    action: str
    target: str = ""
    url: str = ""
    detail: str = ""


ApprovalHandler = Callable[[ApprovalRequest], Awaitable[bool]]


@dataclass
class BrowserPolicy:
    """can_* checks + requires_approval. All refusals carry a code."""

    allowed_schemes: tuple[str, ...] = ("http", "https")
    allowed_domains: tuple[str, ...] = ()  # empty = allow all (minus blocked)
    blocked_domains: tuple[str, ...] = _DEFAULT_BLOCKED_DOMAINS
    allow_private_hosts: bool = False  # tests with local fixtures flip this
    approval_handler: ApprovalHandler | None = None
    resolver: Callable[[str], list[str]] = field(default=_resolve)

    # ------------------------------------------------------------ scheme/url

    def can_navigate(self, url: str) -> str | None:
        """Return a refusal code, or None when navigation is allowed."""
        parsed = urlparse((url or "").strip())
        if parsed.scheme not in self.allowed_schemes:
            return "SCHEME_NOT_ALLOWED"
        if parsed.scheme in ("http", "https"):
            host = (parsed.hostname or "").lower()
            if not host:
                return "INVALID_URL"
            if any(host == b or host.endswith("." + b) for b in self.blocked_domains):
                return "BLOCKED_DOMAIN"
            if self._is_literal_host(host) and not self.allow_private_hosts and not _host_is_public(host):
                return "BLOCKED_DOMAIN"
            if not self._is_literal_host(host) and not self.allow_private_hosts:
                # DNS resolution check: refuse hosts that resolve to private IPs
                addresses = self.resolver(host)
                if not addresses:
                    return "HOST_UNRESOLVED"
                if not any(_host_is_public(a) for a in addresses):
                    return "BLOCKED_DOMAIN"
            if self.allowed_domains and not any(
                host == d or host.endswith("." + d) for d in self.allowed_domains
            ):
                return "DOMAIN_NOT_ALLOWED"
        return None

    @staticmethod
    def _is_literal_host(host: str) -> bool:
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False

    # ----------------------------------------------------------- sensitive

    @staticmethod
    def is_sensitive_target(target: str, kind: str = "click") -> bool:
        """Semantic context matters: 'Next' is harmless, 'Buy Now' is not."""
        text = (target or "").strip()
        if _HARMLESS.match(text):
            return False
        return bool(_SENSITIVE.search(text))

    @staticmethod
    def is_password_field(field_kind: str, name: str = "") -> bool:
        return field_kind == "password" or bool(
            re.search(r"password|passcode|otp", name or "", re.I)
        )

    async def requires_approval(self, request: ApprovalRequest) -> tuple[bool, str]:
        """(needs_approval, code). No handler -> always blocked (§19)."""
        if central_approval_active():
            # Phase 8: the control plane already consumed a human approval
            # for this exact execution — never prompt twice.
            return False, ""
        if self.approval_handler is None:
            return True, "APPROVAL_REQUIRED"
        try:
            approved = await self.approval_handler(request)
        except Exception:
            logger.warning("[BROWSER] approval handler failed; denying")
            return True, "APPROVAL_REQUIRED"
        if not approved:
            return True, "APPROVAL_REQUIRED"
        return False, ""

    async def approval_message(self, request: ApprovalRequest) -> str:
        return (
            f"The action '{request.action}' on '{request.target or request.url}' is sensitive "
            "(form submission / purchase / deletion / sending). Jarvis never does this "
            "automatically. Stop, tell the user what would happen, and ask for explicit "
            "permission before trying again."
        )
