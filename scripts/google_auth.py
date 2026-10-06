"""One-time Google authorization for Jarvis Phase 7 (Gmail + Calendar).

Usage:
    python scripts/google_auth.py            # authorize (opens browser, full scopes)
    python scripts/google_auth.py --readonly # authorize READ-ONLY (gmail.readonly only)
    python scripts/google_auth.py --status   # show granted scope NAMES only

--readonly is the safe first step: the token can only read mail, so sending,
drafting, or deleting is impossible no matter what the agent asks (Google
enforces the scope server-side).

Flow: prints the Google consent URL -> you approve -> Google redirects to
http://localhost/?code=... -> paste that whole URL (or just the code) here.
Tokens are stored via the CredentialStore (DPAPI-protected file); they are
never printed, never logged.
"""

from __future__ import annotations

import asyncio
import sys
from urllib.parse import parse_qs, urlparse

from app.config import load_settings
from app.productivity.credentials import FileCredentialStore
from app.productivity.oauth import (
    DEFAULT_SCOPES,
    GMAIL_READONLY,
    OAuthManager,
    load_client_config,
)


def _extract_code(text: str) -> str:
    text = text.strip()
    if text.startswith("http"):
        query = parse_qs(urlparse(text).query)
        code = (query.get("code") or [""])[0]
        if not code:
            print("No 'code' parameter found in that URL.")
            raise SystemExit(2)
        return code
    return text


async def authorize(readonly: bool = False) -> int:
    settings = load_settings(require_api_key=False)
    client_id, client_secret = load_client_config(settings.google_client_file)
    store = FileCredentialStore(settings.oauth_token_path)
    scopes = (GMAIL_READONLY,) if readonly else DEFAULT_SCOPES
    if readonly:
        print("\nREAD-ONLY mode: requesting gmail.readonly only.")
        print("This token can NOT send, draft, modify, or delete anything.")
    oauth = OAuthManager(client_id, client_secret, store, scopes=scopes)
    try:
        url = oauth.authorization_url()
        print("\nOpen this URL in your browser and approve access:\n")
        print(url)
        print("\nAfter approving, paste the full redirect URL (or just the code):")
        try:
            raw = await asyncio.to_thread(input, "> ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        code = _extract_code(raw)
        await oauth.exchange_code(code)
        scopes = sorted(store.load().granted_scopes()) if store.load() else []
        print("\nAuthorized. Granted scopes:")
        for scope in scopes:
            print(f"  - {scope.rsplit('/', 1)[-1]}")
        print(f"Tokens stored at {settings.oauth_token_path}")
        return 0
    finally:
        await oauth.aclose()


def status() -> int:
    settings = load_settings(require_api_key=False)
    if not settings.google_client_file.exists():
        print(f"Not configured: missing {settings.google_client_file}")
        return 1
    store = FileCredentialStore(settings.oauth_token_path)
    tokens = store.load()
    if tokens is None or not (tokens.access_token or tokens.refresh_token):
        print("Not authorized yet — run: python scripts/google_auth.py")
        return 1
    print("Authorized. Granted scopes (names only):")
    for scope in sorted(tokens.granted_scopes()):
        print(f"  - {scope.rsplit('/', 1)[-1]}")
    return 0


def main() -> int:
    if "--status" in sys.argv:
        return status()
    try:
        return asyncio.run(authorize(readonly="--readonly" in sys.argv))
    except Exception as exc:  # controlled message, never prints secrets
        print(f"Authorization failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
