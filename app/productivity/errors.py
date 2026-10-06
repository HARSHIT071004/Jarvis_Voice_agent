"""Controlled errors for Phase 7 productivity providers.

Every Google/OAuth failure becomes a ProductivityError with a stable
machine-readable code — raw Google API exceptions never reach the user
or the LLM (§20).
"""

from __future__ import annotations


class ProductivityError(Exception):
    """Controlled productivity failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class APIError(Exception):
    """Internal HTTP status failure from a Google API call.

    Providers translate this into ProductivityError; it never escapes
    the provider layer.
    """

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"HTTP {status}: {reason}")
        self.status = status
        self.reason = reason
