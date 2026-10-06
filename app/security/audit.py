"""AuditLog — structured, redacted security audit records (Phase 8 §11).

Every gate decision, approval transition and execution outcome becomes an
AuditEvent. Events are kept in a bounded memory buffer and optionally
appended as JSON lines to a file. Redaction happens BEFORE an event is
stored: tokens, keys, passwords and other credential material never
reach the buffer or the file, and large/untrusted text is reduced to
metadata (lengths, counts).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from app.security.models import AuditEvent

logger = logging.getLogger("jarvis.security")

_SENSITIVE_KEY_RE = re.compile(
    r"(token|password|passwd|passcode|secret|api[_-]?key|authorization|"
    r"cookie|credential|private[_-]?key|bearer|otp)",
    re.I,
)

_VALUE_PATTERNS = (
    re.compile(r"ya29\.[A-Za-z0-9_\-]+"),
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
)

_RECIPIENT_KEYS = frozenset({"to", "cc", "bcc", "recipients", "attendees", "attendee"})
_TEXT_KEYS = frozenset(
    {"body", "content", "description", "notes", "details", "text", "page_text", "message"}
)
_KEEP_STRING_KEYS = frozenset({"subject", "title", "query", "q", "name", "label", "action"})
_STRING_LIMIT = 200

_REDACTED = "[REDACTED]"


def _scrub_string(value: str) -> str:
    for pattern in _VALUE_PATTERNS:
        if pattern.search(value):
            return _REDACTED
    if len(value) > _STRING_LIMIT:
        return f"<str len={len(value)}>"
    return value


def _count(value) -> dict:
    if isinstance(value, (list, tuple)):
        return {"count": len(value)}
    if isinstance(value, str):
        return {"count": len([p for p in value.split(",") if p.strip()])}
    if isinstance(value, dict):
        return {"count": len(value)}
    return {"count": 0}


def summarize_value(key: str, value, *, redact: bool = True):
    """Metadata-first view of one argument (§11: never store secrets)."""
    lowered = (key or "").lower()
    if redact and _SENSITIVE_KEY_RE.search(lowered):
        return _REDACTED
    if lowered in _RECIPIENT_KEYS or lowered.endswith("_count"):
        if lowered.endswith("_count") and isinstance(value, (int, float)):
            return int(value)
        return _count(value)
    if lowered in _TEXT_KEYS:
        if isinstance(value, str):
            return {"length": len(value)}
        return {"length": 0}
    if isinstance(value, dict):
        return {
            str(k): summarize_value(str(k), v, redact=redact)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) <= 5 and all(isinstance(v, (int, float, bool)) for v in value):
            return list(value)
        return {"count": len(value)}
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        if redact:
            return _scrub_string(value)
        if lowered in _KEEP_STRING_KEYS:
            return value[:_STRING_LIMIT]
        return value[:_STRING_LIMIT] if len(value) > _STRING_LIMIT else value
    return _scrub_string(str(value)) if redact else str(value)


def summarize_args(args: dict | None, *, redact: bool = True) -> dict:
    if not isinstance(args, dict):
        return {}
    return {str(k): summarize_value(str(k), v, redact=redact) for k, v in args.items()}


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


class AuditLog:
    def __init__(
        self,
        path: str | Path | None = None,
        enabled: bool = True,
        redact: bool = True,
        max_events: int = 500,
    ) -> None:
        self.enabled = enabled
        self.redact = redact
        self.path = Path(path) if path is not None else None
        self._events: deque[dict] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        if not self.enabled:
            return
        payload = event.as_dict()
        payload["args"] = summarize_args(payload.get("args"), redact=self.redact)
        payload["timestamp_iso"] = _iso(event.timestamp)
        with self._lock:
            self._events.append(payload)
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                except OSError:
                    logger.warning("[SECURITY] Audit file write failed; event kept in memory")

    def events(self, event_type: str | None = None) -> list[dict]:
        with self._lock:
            items = list(self._events)
        if event_type is None:
            return items
        return [e for e in items if e.get("event") == event_type]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
