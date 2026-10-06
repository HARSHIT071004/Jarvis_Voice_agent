"""OAuth 2.0 (authorization-code + PKCE) for Google Gmail/Calendar (§4).

Conceptual flow:

    Jarvis -> OAuthManager -> Google authorization -> access/refresh token
           -> CredentialStore -> Gmail/Calendar clients (via GoogleAPI)

Least privilege: every tool maps to the minimum scope it needs
(SCOPE_FOR); get_access_token(required_scope) fails with AUTH_REQUIRED
when that scope was never granted. No token value is ever logged or
returned to callers outside this module (§5).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

from app.productivity.credentials import CredentialStore, TokenSet
from app.productivity.errors import ProductivityError

logger = logging.getLogger("jarvis.oauth")

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

# Least-privilege scopes — no Drive/Contacts/Sheets/Docs (§4).
GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_COMPOSE = "https://www.googleapis.com/auth/gmail.compose"
GMAIL_SEND = "https://www.googleapis.com/auth/gmail.send"
CALENDAR_READONLY = "https://www.googleapis.com/auth/calendar.readonly"
CALENDAR_EVENTS = "https://www.googleapis.com/auth/calendar.events"

DEFAULT_SCOPES: tuple[str, ...] = (
    GMAIL_READONLY,
    GMAIL_COMPOSE,
    GMAIL_SEND,
    CALENDAR_READONLY,
    CALENDAR_EVENTS,
)

# Minimum scope required by each operation (§4 "minimum scope per operation").
SCOPE_FOR: dict[str, str] = {
    "search_emails": GMAIL_READONLY,
    "get_email": GMAIL_READONLY,
    "draft_email": GMAIL_COMPOSE,
    "send_email": GMAIL_SEND,
    "send_draft": GMAIL_SEND,
    "list_calendar_events": CALENDAR_READONLY,
    "get_calendar_event": CALENDAR_READONLY,
    "find_calendar_availability": CALENDAR_READONLY,
    "create_calendar_event": CALENDAR_EVENTS,
    "update_calendar_event": CALENDAR_EVENTS,
    "delete_calendar_event": CALENDAR_EVENTS,
}

_REFRESH_SKEW = 60.0  # refresh a minute before expiry


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def load_client_config(path: str | Path) -> tuple[str, str]:
    """Read an OAuth client file (client_id + client_secret).

    Accepts Google's downloaded JSON ({"installed": {...}} or {"web": {...}})
    or a flat object. The values stay inside this process: they are never
    logged, never prompted on, never handed to the LLM (§5).
    """
    import json

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProductivityError(
            "AUTH_REQUIRED",
            f"Google OAuth client file not found: {path}. See README Phase 7 "
            "for Google Cloud setup instructions.",
        ) from exc
    except (OSError, ValueError) as exc:
        raise ProductivityError(
            "AUTH_REQUIRED", "Google OAuth client file is unreadable."
        ) from exc
    if not isinstance(data, dict):
        raise ProductivityError(
            "AUTH_REQUIRED", "Google OAuth client file has an unexpected shape."
        )
    for key in ("installed", "web"):
        if isinstance(data.get(key), dict):
            data = data[key]
            break
    client_id = str(data.get("client_id", "")).strip()
    client_secret = str(data.get("client_secret", "")).strip()
    if not client_id:
        raise ProductivityError(
            "AUTH_REQUIRED", "Google OAuth client file has no client_id."
        )
    return client_id, client_secret


class OAuthManager:
    """Authorization-code flow with PKCE, refresh and scope pre-checks.

    All HTTP goes through an injectable httpx transport so tests never
    touch the network.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        store: CredentialStore,
        scopes: tuple[str, ...] = DEFAULT_SCOPES,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: "callable[[], float] | None" = None,
        redirect_uri: str = "http://localhost",
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._store = store
        self._scopes = tuple(scopes)
        self._clock = clock or time.time
        self._redirect_uri = redirect_uri
        self._client = httpx.AsyncClient(transport=transport, timeout=15.0)
        self._code_verifier: str | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------- setup

    def authorization_url(self) -> str:
        """Build the URL the user opens once to grant access (PKCE)."""
        self._code_verifier = _b64url(secrets.token_bytes(48))
        challenge = _b64url(hashlib.sha256(self._code_verifier.encode("ascii")).digest())
        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "response_type": "code",
            "scope": " ".join(self._scopes),
            "access_type": "offline",
            "prompt": "consent",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "include_granted_scopes": "true",
        }
        return f"{AUTH_ENDPOINT}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> None:
        """Exchange the pasted authorization code for tokens; store them."""
        if not self._code_verifier:
            raise ProductivityError(
                "AUTH_REQUIRED",
                "No authorization in progress — run scripts/google_auth.py first.",
            )
        body = await self._post_token(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self._redirect_uri,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "code_verifier": self._code_verifier,
            }
        )
        self._store.save(self._tokens_from(body, fallback_refresh=""))
        self._code_verifier = None
        logger.info("[OAUTH] exchange ok scopes=%s", self._safe_scope_line(body))

    # ------------------------------------------------------------- tokens

    async def get_access_token(self, required_scope: str | None = None) -> str:
        """Return a valid access token, refreshing when needed.

        Raises ProductivityError AUTH_REQUIRED (nothing stored / scope not
        granted) or AUTH_EXPIRED (refresh rejected).
        """
        tokens = self._store.load()
        if tokens is None or not (tokens.access_token or tokens.refresh_token):
            raise ProductivityError(
                "AUTH_REQUIRED",
                "Google access is not authorized yet — run scripts/google_auth.py "
                "once to connect Gmail and Calendar.",
            )
        if required_scope and required_scope not in tokens.granted_scopes():
            raise ProductivityError(
                "AUTH_REQUIRED",
                "Google access for this action was never granted. Run "
                "scripts/google_auth.py again to add the missing permission.",
            )
        if self._needs_refresh(tokens):
            tokens = await self._refresh(tokens)
        if not tokens.access_token:
            raise ProductivityError(
                "AUTH_EXPIRED", "Google access expired and could not be refreshed."
            )
        return tokens.access_token

    def granted_scopes(self) -> set[str]:
        tokens = self._store.load()
        return tokens.granted_scopes() if tokens else set()

    def _needs_refresh(self, tokens: TokenSet) -> bool:
        if not tokens.refresh_token and not tokens.access_token:
            return False
        if tokens.expires_at <= 0:
            return bool(tokens.refresh_token) and not tokens.access_token
        return self._clock() >= tokens.expires_at - _REFRESH_SKEW

    async def _refresh(self, tokens: TokenSet) -> TokenSet:
        if not tokens.refresh_token:
            raise ProductivityError(
                "AUTH_EXPIRED",
                "Google access expired. Run scripts/google_auth.py to authorize again.",
            )
        try:
            body = await self._post_token(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": tokens.refresh_token,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                }
            )
        except ProductivityError:
            logger.warning("[OAUTH] refresh rejected; re-authorization needed")
            raise
        merged = self._tokens_from(body, fallback_refresh=tokens.refresh_token)
        self._store.save(merged)
        logger.info("[OAUTH] token refreshed")
        return merged

    async def _post_token(self, form: dict) -> dict:
        try:
            resp = await self._client.post(TOKEN_ENDPOINT, data=form)
        except httpx.HTTPError as exc:
            raise ProductivityError(
                "GOOGLE_API_ERROR", "Could not reach Google's token service."
            ) from exc
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if resp.status_code != 200:
            err = str(body.get("error", ""))
            desc = str(body.get("error_description", ""))
            if err in {"invalid_grant", "invalid_client"}:
                raise ProductivityError(
                    "AUTH_EXPIRED",
                    "Google rejected the saved authorization — run "
                    "scripts/google_auth.py to authorize again.",
                )
            if resp.status_code == 429:
                raise ProductivityError(
                    "API_RATE_LIMITED", "Google's authorization service is busy; try again shortly."
                )
            raise ProductivityError(
                "GOOGLE_API_ERROR",
                f"Authorization failed ({resp.status_code}): {desc or err or 'unknown error'}.",
            )
        if not isinstance(body, dict) or "access_token" not in body:
            raise ProductivityError("GOOGLE_API_ERROR", "Unexpected token response.")
        return body

    def _tokens_from(self, body: dict, fallback_refresh: str) -> TokenSet:
        expires_in = body.get("expires_in")
        try:
            expires_in_f = float(expires_in)
        except (TypeError, ValueError):
            expires_in_f = 0.0
        return TokenSet(
            access_token=str(body.get("access_token", "")),
            refresh_token=str(body.get("refresh_token") or fallback_refresh),
            expires_at=(self._clock() + expires_in_f) if expires_in_f else 0.0,
            scope=str(body.get("scope", " ".join(self._scopes))),
            token_type=str(body.get("token_type", "Bearer")),
        )

    @staticmethod
    def _safe_scope_line(body: dict) -> str:
        """Scope names only — never token values (§19)."""
        return str(body.get("scope", ""))[:300]
