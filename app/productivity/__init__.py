"""Phase 7 productivity stack: OAuth + Gmail + Calendar behind one seam.

build_productivity(settings) wires:
    CredentialStore -> OAuthManager -> GoogleAPI -> Gmail/Calendar providers
                   -> ProductivityPolicy (fail-closed approval)

Returns None when the Google OAuth client file is absent — the app boots
normally without Gmail/Calendar (tools are simply not registered).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from app.config import Settings
from app.productivity.calendar import CalendarProvider, GoogleCalendarProvider
from app.productivity.credentials import CredentialStore, FileCredentialStore
from app.productivity.errors import ProductivityError
from app.productivity.gmail import GmailProvider, GoogleGmailProvider
from app.productivity.google_api import GoogleAPI
from app.productivity.oauth import DEFAULT_SCOPES, OAuthManager, load_client_config
from app.productivity.policy import ProductivityPolicy

logger = logging.getLogger("jarvis.productivity")

__all__ = [
    "CalendarProvider",
    "GmailProvider",
    "ProductivityPolicy",
    "ProductivityStack",
    "UnconfiguredCalendar",
    "UnconfiguredGmail",
    "build_productivity",
]

_UNCONFIGURED_MSG = (
    "Google access is not configured yet — put the OAuth client JSON at "
    "data/google_client.json (README Phase 7) and run scripts/google_auth.py "
    "once to connect Gmail and Calendar."
)


class UnconfiguredGmail(GmailProvider):
    """Placeholder used when no OAuth client exists: every call fails with a
    controlled AUTH_REQUIRED, so voice/text can honestly say 'Gmail is not
    connected' instead of 'I don't have that tool' (§20/§24)."""

    async def search(self, query: str, max_results: int) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)

    async def get(self, message_id: str) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)

    async def create_draft(self, to, subject, body, thread_id=None) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)

    async def send(self, to, subject, body, thread_id=None) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)

    async def send_draft(self, draft_id: str) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)


class UnconfiguredCalendar(CalendarProvider):
    """Same contract as UnconfiguredGmail, for calendar operations."""

    async def list_events(self, start, end, max_results, timezone) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)

    async def get_event(self, event_id: str) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)

    async def availability(self, start, end, timezone) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)

    async def create_event(
        self, title, start, end, timezone, *,
        attendees=None, location="", description="", check_conflicts=True,
    ) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)

    async def update_event(self, event_id: str, fields: dict, timezone: str) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)

    async def delete_event(self, event_id: str) -> dict:
        raise ProductivityError("AUTH_REQUIRED", _UNCONFIGURED_MSG)


@dataclass
class ProductivityStack:
    oauth: OAuthManager
    api: GoogleAPI
    gmail: GmailProvider
    calendar: CalendarProvider
    policy: ProductivityPolicy
    store: CredentialStore
    timezone: str

    async def aclose(self) -> None:
        try:
            await self.api.aclose()
        finally:
            await self.oauth.aclose()


def build_productivity(
    settings: Settings,
    transport=None,
    clock=None,
) -> ProductivityStack | None:
    """Build the stack, or None when Google access is not configured yet."""
    client_path = settings.google_client_file
    if not client_path.exists():
        logger.info("[PRODUCTIVITY] not configured (missing %s)", client_path)
        return None

    client_id, client_secret = load_client_config(client_path)
    try:
        tz = ZoneInfo(settings.timezone)
    except Exception as exc:
        raise ProductivityError(
            "INVALID_TIMEZONE",
            f"Configured timezone '{settings.timezone}' is not a known IANA zone.",
        ) from exc

    store = FileCredentialStore(settings.oauth_token_path)
    oauth = OAuthManager(
        client_id,
        client_secret,
        store,
        scopes=DEFAULT_SCOPES,
        transport=transport,
        clock=clock,
    )
    api = GoogleAPI(oauth, transport=transport, timeout=settings.agent_tool_timeout + 10.0)
    gmail = GoogleGmailProvider(api, max_body_chars=settings.gmail_max_body_chars)
    calendar = GoogleCalendarProvider(api)
    policy = ProductivityPolicy(approval_handler=None)  # fail-closed (§18)
    return ProductivityStack(
        oauth=oauth,
        api=api,
        gmail=gmail,
        calendar=calendar,
        policy=policy,
        store=store,
        timezone=settings.timezone,
    )
