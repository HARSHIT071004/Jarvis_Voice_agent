"""Live Google API test — READ-ONLY. §26.

Skipped unless RUN_LIVE_TESTS=1, and again when the Google OAuth client
(data/google_client.json) or the authorization (scripts/google_auth.py)
is missing. This file never sends mail, never creates/updates/deletes
calendar events — it only reads profile/search/list data.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="live network test; set RUN_LIVE_TESTS=1 to enable",
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def stack():
    from app.config import load_settings
    from app.productivity import build_productivity
    from app.productivity.errors import ProductivityError

    settings = load_settings(require_api_key=False)
    st = build_productivity(settings)
    if st is None:
        pytest.skip("Google OAuth client not configured (data/google_client.json missing)")
    try:
        tokens = st.store.load()
    except ProductivityError:
        pytest.skip("authorization unreadable — run scripts/google_auth.py")
    if tokens is None or not (tokens.access_token or tokens.refresh_token):
        pytest.skip("not authorized yet — run scripts/google_auth.py once")
    yield st
    run(st.aclose())


def test_live_granted_scopes_cover_reads(stack):
    granted = stack.oauth.granted_scopes()
    assert granted, "authorization exists but no scopes were granted"
    from app.productivity.oauth import CALENDAR_READONLY, GMAIL_READONLY

    assert GMAIL_READONLY in granted
    assert CALENDAR_READONLY in granted


def test_live_gmail_profile_and_search(stack):
    profile = run(stack.api.get(
        "https://gmail.googleapis.com/gmail/v1/users/me/profile",
        scope="https://www.googleapis.com/auth/gmail.readonly",
    ))
    assert profile.get("emailAddress"), "Gmail profile did not return an address"

    out = run(stack.gmail.search("newer_than:30d", 3))
    assert set(out) >= {"messages", "count", "query", "truncated"}
    assert out["count"] <= 3
    for msg in out["messages"]:
        assert set(msg) == {"id", "thread_id", "sender", "subject", "date", "snippet"}


def test_live_calendar_next_seven_days_readonly(stack):
    from app.productivity.oauth import GMAIL_READONLY  # noqa: F401 — scope sanity import

    now = datetime.now(ZoneInfo(stack.timezone))
    out = run(stack.calendar.list_events(now, now + timedelta(days=7), 10, stack.timezone))
    assert set(out) >= {"events", "count", "start", "end", "timezone"}
    assert out["timezone"] == stack.timezone
    for event in out["events"]:
        assert event["id"] and event["start"] and event["end"]


def test_live_token_never_appears_in_read_results(stack, caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        run(stack.gmail.search("in:inbox", 1))
    blob = " ".join(r.getMessage() for r in caplog.records)
    tokens = stack.store.load()
    if tokens and tokens.access_token:
        assert tokens.access_token not in blob
