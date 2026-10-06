"""Browser / computer-use capability (Phase 6).

Public API: BrowserController (all actions), BrowserPolicy (permission
boundary), BrowserSession (Playwright lifecycle), FileRegistry (upload
file_ids). Tools live in app/tools/browser.py and route through the
existing ToolRegistry / ToolRouter — no second agent runtime (§23).
"""

from __future__ import annotations

from app.browser.controller import BrowserController, FileRegistry
from app.browser.errors import BrowserError
from app.browser.models import ActionResult, ElementRef, PageSnapshot
from app.browser.policy import ApprovalHandler, ApprovalRequest, BrowserPolicy
from app.browser.session import BrowserSession

__all__ = [
    "ActionResult",
    "ApprovalHandler",
    "ApprovalRequest",
    "BrowserController",
    "BrowserError",
    "BrowserPolicy",
    "BrowserSession",
    "ElementRef",
    "FileRegistry",
    "PageSnapshot",
    "build_controller",
]


def build_controller(settings) -> BrowserController:
    """Wire a BrowserController from Settings (browser_enabled checked by caller)."""
    schemes = tuple(
        s.strip().lower()
        for s in str(getattr(settings, "browser_allowed_schemes", "http,https")).split(",")
        if s.strip()
    )
    blocked = tuple(
        d.strip().lower()
        for d in str(getattr(settings, "browser_blocked_domains", "")).split(",")
        if d.strip()
    )
    policy = BrowserPolicy(
        allowed_schemes=schemes,
        blocked_domains=("localhost", "metadata.google.internal", *blocked),
    )
    return BrowserController(
        policy,
        headless=bool(getattr(settings, "browser_headless", True)),
        navigation_timeout_ms=int(getattr(settings, "browser_timeout", 15000)),
        max_steps=int(getattr(settings, "browser_max_steps", 15)),
        download_dir=getattr(settings, "browser_download_dir", "data/downloads"),
        upload_dir=getattr(settings, "browser_upload_dir", "data/uploads"),
        max_download_bytes=int(getattr(settings, "browser_max_download_size", 10 * 1024 * 1024)),
    )
