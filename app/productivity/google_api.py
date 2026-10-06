"""Shared Google REST plumbing: auth header, bounded retries, error mapping (§20–21).

One choke point for Gmail + Calendar HTTP calls:
- attaches the OAuth access token (never logged, never returned)
- bounded retry for transient 429/5xx (no infinite loops)
- every failure becomes ProductivityError or the internal APIError
  that providers translate into controlled codes.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.productivity.errors import APIError, ProductivityError
from app.productivity.oauth import OAuthManager

logger = logging.getLogger("jarvis.googleapi")


class GoogleAPI:
    """Minimal async client over Google's REST APIs."""

    def __init__(
        self,
        oauth: OAuthManager,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
        max_retries: int = 2,
        retry_delay: float = 0.2,
    ) -> None:
        self._oauth = oauth
        self._max_retries = max(0, int(max_retries))
        self._retry_delay = max(0.0, float(retry_delay))
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(self, url: str, *, scope: str, params: dict | list | None = None) -> dict:
        return await self._request("GET", url, scope=scope, params=params)

    async def post(self, url: str, *, scope: str, json: dict) -> dict:
        return await self._request("POST", url, scope=scope, json=json)

    async def patch(self, url: str, *, scope: str, json: dict) -> dict:
        return await self._request("PATCH", url, scope=scope, json=json)

    async def delete(self, url: str, *, scope: str) -> dict:
        return await self._request("DELETE", url, scope=scope)

    # ------------------------------------------------------------- core

    async def _request(
        self,
        method: str,
        url: str,
        *,
        scope: str,
        params: dict | list | None = None,
        json: dict | None = None,
    ) -> dict:
        token = await self._oauth.get_access_token(required_scope=scope)
        headers = {"Authorization": f"Bearer {token}"}  # never logged
        attempt = 0
        while True:
            try:
                resp = await self._client.request(
                    method, url, params=params, json=json, headers=headers
                )
            except httpx.HTTPError as exc:
                if attempt < self._max_retries:
                    attempt += 1
                    await asyncio.sleep(self._retry_delay * 2 ** (attempt - 1))
                    continue
                raise ProductivityError(
                    "GOOGLE_API_ERROR", "Could not reach Google right now."
                ) from exc

            status = resp.status_code
            if 200 <= status < 300:
                if status == 204 or not resp.content:
                    return {}
                try:
                    body = resp.json()
                except ValueError as exc:
                    raise ProductivityError(
                        "GOOGLE_API_ERROR", "Google returned an unreadable response."
                    ) from exc
                if isinstance(body, dict):
                    return body
                return {"items": body}

            reason = self._reason(resp)
            if status == 401:
                raise ProductivityError(
                    "AUTH_EXPIRED",
                    "Google rejected the saved access. Run scripts/google_auth.py "
                    "to authorize again.",
                )
            if status == 403:
                lowered = reason.lower()
                if "permission" in lowered or "scope" in lowered:
                    raise ProductivityError(
                        "AUTH_REQUIRED",
                        "Google says this permission was not granted — run "
                        "scripts/google_auth.py to add it.",
                    )
                raise ProductivityError(
                    "PERMISSION_DENIED", "Google refused this operation (PERMISSION_DENIED)."
                )
            if status == 429:
                if attempt < self._max_retries:
                    attempt += 1
                    await asyncio.sleep(self._retry_delay * 2 ** (attempt - 1))
                    continue
                raise ProductivityError(
                    "API_RATE_LIMITED",
                    "Google is rate-limiting requests — try again in a minute.",
                )
            if status >= 500:
                if attempt < self._max_retries:
                    attempt += 1
                    await asyncio.sleep(self._retry_delay * 2 ** (attempt - 1))
                    continue
                raise APIError(status, reason or "server error")
            raise APIError(status, reason)

    @staticmethod
    def _reason(resp: httpx.Response) -> str:
        """Extract a short error message from a Google error body (no headers)."""
        try:
            body = resp.json()
        except ValueError:
            return ""
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            message = str(err.get("message", ""))[:300]
            return message
        if isinstance(err, str):
            return err[:300]
        if isinstance(body, dict) and body.get("error_description"):
            return str(body["error_description"])[:300]
        return ""
