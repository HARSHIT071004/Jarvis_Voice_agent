"""Phase 7 §17/§25: OAuth 2.0 + PKCE, scope gating, token refresh, no logging."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import httpx
import pytest
from conftest import run

from app.productivity.credentials import FileCredentialStore, TokenSet
from app.productivity.errors import ProductivityError
from app.productivity.oauth import (
    DEFAULT_SCOPES,
    GMAIL_READONLY,
    GMAIL_SEND,
    OAuthManager,
    load_client_config,
)

CLIENT = {"client_id": "cid123", "client_secret": "csecret456"}
ACCESS = "ya29-at-value-SECRET"
REFRESH = "refresh-value-SECRET"


def write_client(tmp_path: Path, data=None) -> Path:
    path = tmp_path / "google_client.json"
    path.write_text(json.dumps(data if data is not None else {"installed": CLIENT}), encoding="utf-8")
    return path


def make_handler(calls: list | None = None, refresh_error: str | None = None, expires_in=3600):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        body = dict(kv.split("=", 1) for kv in request.content.decode().split("&"))
        from urllib.parse import unquote_plus

        body = {k: unquote_plus(v) for k, v in body.items()}
        if body.get("grant_type") == "refresh_token":
            if refresh_error:
                return httpx.Response(400, json={"error": refresh_error})
            return httpx.Response(200, json={"access_token": "ya29-refreshed", "expires_in": 3600})
        if body.get("grant_type") == "authorization_code":
            return httpx.Response(
                200,
                json={
                    "access_token": ACCESS,
                    "refresh_token": REFRESH,
                    "expires_in": expires_in,
                    "token_type": "Bearer",
                    "scope": " ".join(DEFAULT_SCOPES),
                },
            )
        return httpx.Response(400, json={"error": "invalid_request"})

    return handler


class FakeClock:
    def __init__(self, t: float = 1_000_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def oauth_factory(tmp_path):
    """Build OAuthManagers against MockTransport; always close their clients."""
    made: list[OAuthManager] = []

    def _make(handler=None, clock=None, store=None, scopes=None) -> OAuthManager:
        m = OAuthManager(
            CLIENT["client_id"], CLIENT["client_secret"],
            store or FileCredentialStore(tmp_path / "tokens.json"),
            scopes=scopes or tuple(DEFAULT_SCOPES),
            transport=httpx.MockTransport(handler or make_handler()),
            clock=clock,
        )
        made.append(m)
        return m

    yield _make

    async def _close_all():
        for m in made:
            await m.aclose()

    run(_close_all())


async def authorize(oauth: OAuthManager) -> None:
    oauth.authorization_url()
    await oauth.exchange_code("authcode123")


# ------------------------------------------------------------ client config

def test_load_client_config_installed_wrapper(tmp_path):
    cid, secret = load_client_config(write_client(tmp_path))
    assert cid == "cid123" and secret == "csecret456"


def test_load_client_config_flat(tmp_path):
    cid, secret = load_client_config(write_client(tmp_path, dict(CLIENT)))
    assert cid == "cid123" and secret == "csecret456"


def test_missing_client_file_is_auth_required(tmp_path):
    with pytest.raises(ProductivityError) as exc:
        load_client_config(tmp_path / "nope.json")
    assert exc.value.code == "AUTH_REQUIRED"


def test_client_file_without_client_id_is_auth_required(tmp_path):
    with pytest.raises(ProductivityError) as exc:
        load_client_config(write_client(tmp_path, {"installed": {"client_secret": "x"}}))
    assert exc.value.code == "AUTH_REQUIRED"


# ------------------------------------------------------------------- PKCE

def test_authorization_url_uses_pkce_and_scopes(oauth_factory):
    oauth = oauth_factory()
    url = oauth.authorization_url()
    assert "code_challenge_method=S256" in url
    assert "code_challenge=" in url
    assert "client_id=cid123" in url
    assert "access_type=offline" in url
    assert "prompt=consent" in url
    assert "include_granted_scopes=true" in url
    assert "scope=" in url and "gmail.readonly" in url
    assert "gmail.send" in url  # least privilege: only the five declared scopes
    assert "drive" not in url and "contacts" not in url


def test_pkce_challenge_matches_internal_verifier(oauth_factory):
    oauth = oauth_factory()
    url = oauth.authorization_url()
    verifier = oauth._code_verifier  # never exposed through the public API
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
    assert expected.rstrip(b"=").decode("ascii") in url
    assert verifier not in url


def test_exchange_code_stores_tokens(oauth_factory, tmp_path):
    oauth = oauth_factory()
    run(authorize(oauth))
    loaded = oauth._store.load()
    assert loaded is not None
    assert loaded.access_token == ACCESS
    assert loaded.refresh_token == REFRESH
    assert GMAIL_SEND in loaded.granted_scopes()
    assert loaded.expires_at > 0
    raw = (tmp_path / "tokens.json").read_bytes()
    assert ACCESS.encode() not in raw  # DPAPI-protected at rest on Windows
    assert REFRESH.encode() not in raw
    assert raw.startswith(b"Jarvis1:")


def test_exchange_without_authorization_url_is_auth_required(oauth_factory):
    oauth = oauth_factory()
    with pytest.raises(ProductivityError) as exc:
        run(oauth.exchange_code("code"))
    assert exc.value.code == "AUTH_REQUIRED"


# ----------------------------------------------------------- token access

def test_get_access_token_without_grant_is_auth_required(oauth_factory):
    oauth = oauth_factory()
    with pytest.raises(ProductivityError) as exc:
        run(oauth.get_access_token(GMAIL_READONLY))
    assert exc.value.code == "AUTH_REQUIRED"


def test_valid_token_returned_without_refresh(oauth_factory):
    calls: list = []
    oauth = oauth_factory(handler=make_handler(calls))
    run(authorize(oauth))
    tok = run(oauth.get_access_token(GMAIL_SEND))
    assert tok == ACCESS  # refresh would have returned ya29-refreshed
    assert len(calls) == 1  # only the exchange


def test_expired_token_triggers_refresh_and_saves(oauth_factory):
    clock = FakeClock()
    oauth = oauth_factory(handler=make_handler(), clock=clock)
    run(authorize(oauth))
    clock.t += 7200  # well past expires_in=3600
    tok = run(oauth.get_access_token(GMAIL_READONLY))
    assert tok == "ya29-refreshed"
    assert oauth._store.load().access_token == "ya29-refreshed"


def test_refresh_inside_sixty_second_skew(oauth_factory):
    clock = FakeClock()
    oauth = oauth_factory(handler=make_handler(expires_in=30), clock=clock)
    run(authorize(oauth))
    clock.t += 1  # 29s left < 60s skew -> refresh now
    tok = run(oauth.get_access_token(GMAIL_READONLY))
    assert tok == "ya29-refreshed"


def test_invalid_grant_maps_to_auth_expired(oauth_factory):
    clock = FakeClock()
    oauth = oauth_factory(handler=make_handler(refresh_error="invalid_grant"), clock=clock)
    run(authorize(oauth))
    clock.t += 7200
    with pytest.raises(ProductivityError) as exc:
        run(oauth.get_access_token(GMAIL_READONLY))
    assert exc.value.code == "AUTH_EXPIRED"
    assert "google_auth.py" in exc.value.message


def test_ungranted_scope_fails_before_any_token_use(oauth_factory):
    store = FileCredentialStore("never-used.json")
    store.save(TokenSet(access_token=ACCESS, refresh_token=REFRESH,
                        expires_at=time.time() + 3600, scope=GMAIL_READONLY))
    oauth = oauth_factory(store=store)
    with pytest.raises(ProductivityError) as exc:
        run(oauth.get_access_token(GMAIL_SEND))
    assert exc.value.code == "AUTH_REQUIRED"
    assert "never granted" in exc.value.message


def test_refresh_error_http_is_not_confused_with_scope(oauth_factory):
    clock = FakeClock()
    oauth = oauth_factory(handler=make_handler(refresh_error="unauthorized_client"), clock=clock)
    run(authorize(oauth))
    clock.t += 7200
    with pytest.raises(ProductivityError) as exc:
        run(oauth.get_access_token(GMAIL_READONLY))
    assert exc.value.code == "GOOGLE_API_ERROR"


# ------------------------------------------------------------ credential store

def test_store_roundtrip_and_clear(tmp_path):
    store = FileCredentialStore(tmp_path / "t.json")
    assert store.load() is None  # fresh install: nothing authorized yet
    ts = TokenSet(access_token=ACCESS, refresh_token=REFRESH, expires_at=123456.0,
                  scope=GMAIL_READONLY)
    store.save(ts)
    loaded = store.load()
    assert loaded.access_token == ACCESS and loaded.refresh_token == REFRESH
    assert loaded.expires_at == 123456.0
    store.clear()
    assert store.load() is None


def test_corrupt_store_raises_auth_required_without_echoing_bytes(tmp_path):
    path = tmp_path / "t.json"
    path.write_bytes(b"broken-not-json")
    store = FileCredentialStore(path)
    with pytest.raises(ProductivityError) as exc:
        store.load()
    assert exc.value.code == "AUTH_REQUIRED"
    assert "broken-not-json" not in exc.value.message


def test_dpapi_locked_store_raises_auth_required(tmp_path):
    path = tmp_path / "t.json"
    path.write_bytes(b"Jarvis1:" + b"\xaa" * 32)  # prefix but undecryptable blob
    store = FileCredentialStore(path)
    with pytest.raises(ProductivityError) as exc:
        store.load()
    assert exc.value.code == "AUTH_REQUIRED"


# ------------------------------------------------------------------ logging

def test_tokens_never_appear_in_logs(oauth_factory, caplog):
    import logging

    clock = FakeClock()
    oauth = oauth_factory(handler=make_handler(), clock=clock)
    run(authorize(oauth))
    clock.t += 7200
    with caplog.at_level(logging.DEBUG):
        run(oauth.get_access_token(GMAIL_SEND))
    blob = " ".join(r.getMessage() for r in caplog.records)
    assert ACCESS not in blob
    assert REFRESH not in blob
    assert "ya29-refreshed" not in blob

