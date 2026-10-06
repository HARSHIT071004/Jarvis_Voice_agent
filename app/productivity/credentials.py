"""OAuth token storage — the Phase 7 credential seam (§5).

Rules enforced here:
- tokens are NEVER logged, NEVER returned to tool callers, NEVER in prompts
- storage is outside source code, outside .env, outside database rows
- the file is written atomically with owner-only permissions where possible
- on Windows the blob is additionally protected with DPAPI (CryptProtectData)

Phase 8 can replace this by implementing CredentialStore — nothing else in
the codebase touches the storage format.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.productivity.errors import ProductivityError

logger = logging.getLogger("jarvis.oauth")

_PREFIX = b"Jarvis1:"


class TokenSet(BaseModel):
    """One stored Google OAuth grant. Never serialized into tool results."""

    model_config = ConfigDict(extra="forbid")

    access_token: str = Field(default="", repr=False)
    refresh_token: str = Field(default="", repr=False)
    expires_at: float = 0.0  # epoch seconds; 0 = unknown (no expiry)
    scope: str = ""  # space-separated granted scopes
    token_type: str = "Bearer"

    def granted_scopes(self) -> set[str]:
        return {s for s in self.scope.split() if s}


class CredentialStore(ABC):
    """Phase 8 seam: swap in DPAPI/keyring/OS credential manager."""

    @abstractmethod
    def load(self) -> TokenSet | None:
        """Return stored tokens, or None when nothing is authorized yet."""

    @abstractmethod
    def save(self, tokens: TokenSet) -> None:
        """Persist tokens (atomic replace)."""

    @abstractmethod
    def clear(self) -> None:
        """Drop stored tokens (revoke/reset)."""


def _dpapi_protect(data: bytes) -> bytes | None:
    try:
        import win32crypt  # noqa: PLC0415 — optional Windows-only dependency
    except Exception:
        return None
    try:
        return win32crypt.CryptProtectData(data, "jarvis-oauth", None, None, None, 0)
    except Exception:  # pragma: no cover - depends on Windows session state
        logger.warning("[OAUTH] dpapi protect failed; falling back to plain file")
        return None


def _dpapi_unprotect(blob: bytes) -> bytes | None:
    try:
        import win32crypt  # noqa: PLC0415
    except Exception:
        return None
    try:
        out = win32crypt.CryptUnprotectData(blob, None, None, None, 0)
        return bytes(out[1])
    except Exception:
        return None


class FileCredentialStore(CredentialStore):
    """Token file under data/ (git-ignored), DPAPI-protected on Windows."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> TokenSet | None:
        if not self._path.exists():
            return None
        try:
            raw = self._path.read_bytes()
        except OSError as exc:
            raise ProductivityError(
                "AUTH_REQUIRED",
                "Saved Google authorization could not be read — run "
                "scripts/google_auth.py to authorize again.",
            ) from exc
        if raw.startswith(_PREFIX):
            plain = _dpapi_unprotect(raw[len(_PREFIX):])
            if plain is None:
                raise ProductivityError(
                    "AUTH_REQUIRED",
                    "Saved Google authorization is locked to another Windows "
                    "account — run scripts/google_auth.py to authorize again.",
                )
            raw = plain
        try:
            data = json.loads(raw.decode("utf-8"))
            return TokenSet.model_validate(data)
        except Exception as exc:
            raise ProductivityError(
                "AUTH_REQUIRED",
                "Saved Google authorization is unreadable — run "
                "scripts/google_auth.py to authorize again.",
            ) from exc

    def save(self, tokens: TokenSet) -> None:
        plain = tokens.model_dump_json().encode("utf-8")
        blob = _dpapi_protect(plain)
        if blob is not None:
            plain = _PREFIX + blob
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # atomic replace so a crash never leaves a half-written token file
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), prefix=".tok")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(plain)
            try:
                os.chmod(tmp, 0o600)
            except OSError:  # pragma: no cover - best effort on Windows
                pass
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            pass
