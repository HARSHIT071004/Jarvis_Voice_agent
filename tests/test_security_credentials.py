"""Phase 8 §24 — Credential material never reaches prompts, logs, or audit."""

from __future__ import annotations

import io
import logging

import httpx
from conftest import run
from test_productivity_oauth import ACCESS, CLIENT, REFRESH, FakeClock, authorize, make_handler

from app.agent.prompt import SECURITY_GUIDANCE
from app.agent.runtime import AGENT_PROMPT
from app.productivity.credentials import FileCredentialStore
from app.productivity.errors import ProductivityError
from app.productivity.oauth import DEFAULT_SCOPES, GMAIL_SEND, OAuthManager
from app.utils.logging import RedactingFilter

TOKEN_PATTERNS = ("ya29.", "AIza", "access_token", "refresh_token", "xoxb-", "ghp_")

PROMPT_SOURCES = {
    "AGENT_PROMPT": AGENT_PROMPT,
    "SECURITY_GUIDANCE": SECURITY_GUIDANCE,
}


def test_prompt_blocks_carry_no_credential_material():
    for name, text in PROMPT_SOURCES.items():
        for pattern in TOKEN_PATTERNS:
            assert pattern not in text, f"{name} leaked {pattern!r}"


def test_redacting_filter_replaces_sensitive_messages():
    filter_ = RedactingFilter()
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="connect with api_key=AIzaFakeKey123 now", args=(), exc_info=None,
    )
    assert filter_.filter(record) is True
    emitted = record.getMessage()
    assert "AIzaFakeKey123" not in emitted
    assert "redacted" in emitted


def test_redacting_filter_covers_marker_variants():
    filter_ = RedactingFilter()
    for marker in ("Authorization: Bearer abc.def", "the secret value", "apikey=1", "x ApiKey y"):
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg=marker, args=(), exc_info=None,
        )
        filter_.filter(record)
        assert marker not in record.getMessage()


def test_redacting_filter_leaves_normal_messages_alone():
    filter_ = RedactingFilter()
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="[GMAIL] sent id=abc recipients=1", args=(), exc_info=None,
    )
    assert filter_.filter(record) is True
    assert record.getMessage() == "[GMAIL] sent id=abc recipients=1"


def test_handler_with_filter_never_emits_token():
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.addFilter(RedactingFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("t.credentials.e2e")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    logger.info("api_key rotated to %s", ACCESS)
    logger.info("Authorization: Bearer %s", ACCESS)
    assert ACCESS not in buffer.getvalue()


def test_refresh_failure_message_contains_no_tokens(tmp_path):
    clock = FakeClock()
    oauth = OAuthManager(
        CLIENT["client_id"],
        CLIENT["client_secret"],
        FileCredentialStore(tmp_path / "tokens.json"),
        scopes=tuple(DEFAULT_SCOPES),
        transport=httpx.MockTransport(make_handler(refresh_error="invalid_grant")),
        clock=clock,
    )
    run(authorize(oauth))
    clock.t += 7200

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    capture = _Capture()
    capture.addFilter(RedactingFilter())
    logging.getLogger().addHandler(capture)
    try:
        try:
            run(oauth.get_access_token(GMAIL_SEND))
            raise AssertionError("expected AUTH_EXPIRED")
        except ProductivityError as exc:
            assert exc.code == "AUTH_EXPIRED"
            assert ACCESS not in exc.message
            assert REFRESH not in exc.message
    finally:
        logging.getLogger().removeHandler(capture)
        run(oauth.aclose())

    blob = " ".join(records)
    assert ACCESS not in blob
    assert "refresh-value" not in blob
    assert "csecret456" not in blob


def test_audit_summary_never_keeps_token_values(tmp_path):
    from app.security.audit import summarize_args

    summary = summarize_args(
        {
            "access_token": ACCESS,
            "refresh_token": "refresh-value-SECRET",
            "client_secret": "csecret456",
            "Authorization": "Bearer abc.def",
        }
    )
    assert ACCESS not in repr(summary)
    assert "refresh-value" not in repr(summary)
    assert "csecret456" not in repr(summary)
