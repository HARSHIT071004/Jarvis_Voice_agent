"""Controlled browser failures with stable machine-readable codes."""

from __future__ import annotations


class BrowserError(Exception):
    """Controlled browser failure. Never leaks raw Playwright errors."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
