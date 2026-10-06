"""Structured logging setup.

Rules:
- never log API keys, secrets, or raw audio content
- keep a consistent single-line format so the CLI stays readable
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(levelname)-7s %(name)s: %(message)s"
_sensitive_markers = ("api_key", "apikey", "authorization", "bearer", "secret")


class RedactingFilter(logging.Filter):
    """Best-effort scrubbing of secrets that might slip into a message."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        lowered = message.lower()
        for marker in _sensitive_markers:
            if marker in lowered:
                record.msg = "[redacted: message contained a sensitive marker]"
                record.args = ()
                break
        return True


def setup_logging(level: str = "INFO") -> logging.Logger:
    # Windows consoles often default to cp1252; token markers like U+2581 crash it.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    root = logging.getLogger()
    root.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler.addFilter(RedactingFilter())
        root.addHandler(handler)

    for noisy in ("google", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("jarvis")
