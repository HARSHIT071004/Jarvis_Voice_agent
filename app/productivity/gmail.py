"""Gmail provider abstraction (§6): tools talk to GmailProvider, never to
Google internals. GoogleGmailProvider speaks the official Gmail REST API
over the shared GoogleAPI layer.

Bounds (§21): result count, body size, pagination depth are all capped.
Email bodies are plain text with HTML stripped and active content dropped;
everything returned is untrusted data (§8) — this module never acts on it.
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from email.message import EmailMessage
from html.parser import HTMLParser

from app.productivity.errors import APIError, ProductivityError
from app.productivity.google_api import GoogleAPI
from app.productivity.oauth import GMAIL_COMPOSE, GMAIL_READONLY, GMAIL_SEND

logger = logging.getLogger("jarvis.gmail")

_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_MAX_PAGES = 3
_MAX_SUBJECT = 200
_MAX_SNIPPET = 240
_MAX_SENDER = 200
_MAX_ATTACHMENTS = 10


def _b64url_decode(data: str) -> str:
    pad = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data + pad).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class _HTMLText(HTMLParser):
    """Strip HTML to plain text; drop script/style/active content entirely."""

    _SKIP = {"script", "style", "head", "noscript", "template"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        out = "".join(self._chunks)
        lines = [ln.strip() for ln in out.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def html_to_text(html: str) -> str:
    parser = _HTMLText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return ""
    return parser.text()


def build_mime(to: list[str], subject: str, body: str) -> str:
    """Build a plain-text RFC 822 message as base64url (no attachments)."""
    msg = EmailMessage()
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    msg.set_content(body)
    return _b64url_encode(msg.as_bytes())


class GmailProvider(ABC):
    """What the tool layer sees. No Google SDK types cross this boundary."""

    @abstractmethod
    async def search(self, query: str, max_results: int) -> dict: ...

    @abstractmethod
    async def get(self, message_id: str) -> dict: ...

    @abstractmethod
    async def create_draft(
        self, to: list[str], subject: str, body: str, thread_id: str | None = None
    ) -> dict: ...

    @abstractmethod
    async def send(
        self, to: list[str], subject: str, body: str, thread_id: str | None = None
    ) -> dict: ...

    @abstractmethod
    async def send_draft(self, draft_id: str) -> dict: ...


class GoogleGmailProvider(GmailProvider):
    def __init__(self, api: GoogleAPI, max_body_chars: int = 8000) -> None:
        self._api = api
        self._max_body_chars = max(500, int(max_body_chars))

    # ------------------------------------------------------------- reads

    async def search(self, query: str, max_results: int) -> dict:
        limit = max(1, min(int(max_results), 25))
        wanted: list[dict] = []
        page_token: str | None = None
        truncated = False
        for _ in range(_MAX_PAGES):
            params: list[tuple[str, str]] = [("q", query), ("maxResults", str(limit - len(wanted)))]
            if page_token:
                params.append(("pageToken", page_token))
            try:
                page = await self._api.get(
                    f"{_BASE}/messages", scope=GMAIL_READONLY, params=params
                )
            except APIError as exc:
                raise _map(exc, context="search") from exc
            for item in page.get("messages", []) or []:
                if len(wanted) >= limit:
                    truncated = True
                    break
                meta = await self._metadata(str(item.get("id", "")))
                if meta:
                    wanted.append(meta)
            page_token = page.get("nextPageToken")
            if not page_token or len(wanted) >= limit:
                break
            if page_token and len(wanted) >= limit:
                truncated = True
        logger.info("[GMAIL] search results=%d query_len=%d", len(wanted), len(query))
        return {"messages": wanted, "count": len(wanted), "query": query, "truncated": truncated}

    async def _metadata(self, message_id: str) -> dict | None:
        if not message_id:
            return None
        params: list[tuple[str, str]] = [
            ("format", "metadata"),
            ("metadataHeaders", "From"),
            ("metadataHeaders", "Subject"),
            ("metadataHeaders", "Date"),
        ]
        try:
            msg = await self._api.get(
                f"{_BASE}/messages/{message_id}", scope=GMAIL_READONLY, params=params
            )
        except APIError as exc:
            if exc.status == 404:
                return None
            raise _map(exc, context="search") from exc
        headers = _headers(msg)
        return {
            "id": str(msg.get("id", message_id)),
            "thread_id": str(msg.get("threadId", "")),
            "sender": headers.get("from", "")[:_MAX_SENDER],
            "subject": headers.get("subject", "")[:_MAX_SUBJECT],
            "date": headers.get("date", "")[:_MAX_SENDER],
            "snippet": str(msg.get("snippet", ""))[:_MAX_SNIPPET],
        }

    async def get(self, message_id: str) -> dict:
        try:
            msg = await self._api.get(
                f"{_BASE}/messages/{message_id}", scope=GMAIL_READONLY,
                params=[("format", "full")],
            )
        except APIError as exc:
            if exc.status == 404:
                raise ProductivityError(
                    "EMAIL_NOT_FOUND", f"No email with id '{message_id}'."
                ) from exc
            raise _map(exc, context="read") from exc
        headers = _headers(msg)
        body, is_html, truncated = _extract_body(msg.get("payload", {}) or {}, self._max_body_chars)
        attachments = _attachments(msg.get("payload", {}) or {})
        logger.info(
            "[GMAIL] read id=%s attachments=%d", str(msg.get("id", message_id))[:64], len(attachments)
        )
        return {
            "id": str(msg.get("id", message_id)),
            "thread_id": str(msg.get("threadId", "")),
            "sender": headers.get("from", "")[:_MAX_SENDER],
            "to": headers.get("to", "")[:500],
            "cc": headers.get("cc", "")[:300],
            "subject": headers.get("subject", "")[:_MAX_SUBJECT],
            "date": headers.get("date", "")[:_MAX_SENDER],
            "snippet": str(msg.get("snippet", ""))[:_MAX_SNIPPET],
            "body": body,
            "body_is_html_stripped": is_html,
            "truncated": truncated,
            "attachments": attachments,
        }

    # ------------------------------------------------------------ writes

    async def create_draft(
        self, to: list[str], subject: str, body: str, thread_id: str | None = None
    ) -> dict:
        message: dict = {"raw": build_mime(to, subject, body)}
        if thread_id:
            message["threadId"] = thread_id
        try:
            draft = await self._api.post(
                f"{_BASE}/drafts", scope=GMAIL_COMPOSE, json={"message": message}
            )
        except APIError as exc:
            raise _map(exc, context="draft") from exc
        draft_id = str(draft.get("id", ""))
        if not draft_id:
            raise ProductivityError("GOOGLE_API_ERROR", "Gmail did not return a draft id.")
        logger.info("[GMAIL] draft created id=%s", draft_id[:64])
        return {
            "draft_id": draft_id,
            "thread_id": str((draft.get("message") or {}).get("threadId", "")),
            "status": "created",
        }

    async def send(
        self, to: list[str], subject: str, body: str, thread_id: str | None = None
    ) -> dict:
        payload: dict = {"raw": build_mime(to, subject, body)}
        if thread_id:
            payload["threadId"] = thread_id
        try:
            sent = await self._api.post(
                f"{_BASE}/messages/send", scope=GMAIL_SEND, json=payload
            )
        except APIError as exc:
            raise _map(exc, context="send") from exc
        message_id = str(sent.get("id", ""))
        logger.info("[GMAIL] sent id=%s recipients=%d", message_id[:64], len(to))
        return {
            "message_id": message_id,
            "thread_id": str(sent.get("threadId", "")),
            "status": "sent",
            "to": to,
        }

    async def send_draft(self, draft_id: str) -> dict:
        try:
            sent = await self._api.post(
                f"{_BASE}/drafts/send", scope=GMAIL_SEND, json={"id": draft_id}
            )
        except APIError as exc:
            if exc.status == 404:
                raise ProductivityError(
                    "EMAIL_NOT_FOUND", f"No draft with id '{draft_id}'."
                ) from exc
            raise _map(exc, context="send") from exc
        logger.info("[GMAIL] draft sent id=%s", draft_id[:64])
        return {
            "message_id": str(sent.get("id", "")),
            "thread_id": str(sent.get("threadId", "")),
            "status": "sent",
            "draft_id": draft_id,
        }


def _headers(msg: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for header in (msg.get("payload", {}) or {}).get("headers", []) or []:
        name = str(header.get("name", "")).lower()
        if name and name not in out:
            out[name] = str(header.get("value", ""))
    return out


def _extract_body(payload: dict, max_chars: int) -> tuple[str, bool, bool]:
    """Best plain-text body, HTML stripped when that is all there is."""
    plain = _find_part(payload, "text/plain")
    html = _find_part(payload, "text/html")
    if plain:
        text = _b64url_decode(plain)
        is_html = False
    elif html:
        text = html_to_text(_b64url_decode(html))
        is_html = True
    else:
        text, is_html = "", False
    if len(text) > max_chars:
        return text[:max_chars], is_html, True
    return text, is_html, False


def _find_part(payload: dict, mime: str) -> str | None:
    if str(payload.get("mimeType", "")).startswith(mime):
        data = (payload.get("body") or {}).get("data")
        if data:
            return str(data)
    for part in payload.get("parts", []) or []:
        found = _find_part(part, mime)
        if found:
            return found
    return None


def _attachments(payload: dict, out: list[dict] | None = None) -> list[dict]:
    out = out if out is not None else []
    for part in payload.get("parts", []) or []:
        filename = str(part.get("filename", ""))
        if filename and len(out) < _MAX_ATTACHMENTS:
            body = part.get("body") or {}
            out.append(
                {
                    "filename": filename[:200],
                    "mime_type": str(part.get("mimeType", ""))[:120],
                    "size": int(body.get("size", 0) or 0),
                    "attachment_id": str(body.get("attachmentId", ""))[:128],
                }
            )
        out.extend(_attachments(part, [])[: max(0, _MAX_ATTACHMENTS - len(out))])
    return out


def _map(exc: APIError, context: str) -> ProductivityError:
    """Turn an HTTP status into a controlled Jarvis code (§20)."""
    if exc.status == 404:
        return ProductivityError("EMAIL_NOT_FOUND", "That email was not found.")
    if exc.status == 400:
        reason = exc.reason.lower()
        if context in {"send", "draft"} and any(
            k in reason for k in ("recipient", "address", "invalid to", "parse")
        ):
            return ProductivityError(
                "INVALID_RECIPIENT", "Gmail rejected the recipient address."
            )
        return ProductivityError(
            "GOOGLE_API_ERROR", f"Gmail rejected the request: {exc.reason or 'invalid query'}."
        )
    if exc.status == 429:
        return ProductivityError(
            "API_RATE_LIMITED", "Gmail is rate-limiting requests — try again shortly."
        )
    return ProductivityError(
        "GOOGLE_API_ERROR", f"Gmail request failed ({exc.status})."
    )
