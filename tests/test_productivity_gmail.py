"""Phase 7 §20-21: Gmail provider over mocked REST — bounds, mapping, retries."""

from __future__ import annotations

import base64
from email import message_from_string

import httpx
import pytest
from conftest import run

from app.productivity.errors import ProductivityError
from app.productivity.gmail import GoogleGmailProvider, html_to_text
from app.productivity.google_api import GoogleAPI
from app.productivity.oauth import GMAIL_COMPOSE, GMAIL_READONLY, GMAIL_SEND


class FakeOAuth:
    def __init__(self, error: ProductivityError | None = None) -> None:
        self.error = error
        self.scopes: list[str | None] = []

    async def get_access_token(self, required_scope: str | None = None) -> str:
        if self.error is not None:
            raise self.error
        self.scopes.append(required_scope)
        return "tok-SECRET"


def b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).rstrip(b"=").decode()


def ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body)


def err(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message}})


def full_message(
    mid="m1", thread="t1", frm="Rahul <rahul@example.com>", to="me@example.com",
    cc="", subject="Interview tomorrow", date="Mon, 5 Oct 2026 09:00:00 +0530",
    snippet="Can we meet?", plain=None, html=None, attachments=(),
) -> dict:
    headers = [
        {"name": "From", "value": frm},
        {"name": "To", "value": to},
        {"name": "Subject", "value": subject},
        {"name": "Date", "value": date},
    ]
    if cc:
        headers.append({"name": "Cc", "value": cc})
    if plain is not None and html is None and not attachments:
        payload = {"mimeType": "text/plain", "body": {"data": b64(plain)}, "headers": headers}
    elif html is not None and plain is None and not attachments:
        payload = {"mimeType": "text/html", "body": {"data": b64(html)}, "headers": headers}
    else:
        alt_parts = []
        if plain is not None:
            alt_parts.append({"mimeType": "text/plain", "body": {"data": b64(plain)}})
        if html is not None:
            alt_parts.append({"mimeType": "text/html", "body": {"data": b64(html)}})
        parts = [{"mimeType": "multipart/alternative", "parts": alt_parts}]
        for name, mime, size, aid in attachments:
            parts.append(
                {"filename": name, "mimeType": mime, "body": {"attachmentId": aid, "size": size}}
            )
        payload = {"mimeType": "multipart/mixed", "parts": parts, "headers": headers}
    return {"id": mid, "threadId": thread, "snippet": snippet, "payload": payload}


def route_handler(routes: dict, calls: list | None = None):
    """routes: (METHOD, path) -> Response | list[Response] | callable(request)."""
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        value = routes.get((request.method, request.url.path))
        if value is None:
            return err(404, "Not Found")
        if isinstance(value, list):
            item = value.pop(0) if len(value) > 1 else value[0]
            return item if isinstance(item, httpx.Response) else item(request)
        if isinstance(value, httpx.Response):
            return value
        return value(request)

    return handler


@pytest.fixture
def gmail_factory():
    """(provider, oauth, calls) per invocation; all clients closed after test."""
    made: list[GoogleAPI] = []

    def _make(routes: dict, max_body_chars: int = 8000, oauth: FakeOAuth | None = None):
        calls: list = []
        oauth = oauth or FakeOAuth()
        api = GoogleAPI(
            oauth, transport=httpx.MockTransport(route_handler(routes, calls)),
            retry_delay=0.0,
        )
        made.append(api)
        provider = GoogleGmailProvider(api, max_body_chars=max_body_chars)
        return provider, oauth, calls

    yield _make

    async def _close():
        for api in made:
            await api.aclose()

    run(_close())


def decode_raw(request: httpx.Request):
    import json

    data = json.loads(request.content)
    raw = data["message"]["raw"] if "message" in data else data["raw"]
    pad = "=" * (-len(raw) % 4)
    text = base64.urlsafe_b64decode(raw + pad).decode()
    return message_from_string(text)


BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
MESSAGES = "/gmail/v1/users/me/messages"
DRAFTS = "/gmail/v1/users/me/drafts"


# ------------------------------------------------------------------ search

def test_search_maps_metadata(gmail_factory):
    provider, oauth, calls = gmail_factory({
        ("GET", MESSAGES): ok({"messages": [{"id": "m1"}, {"id": "m2"}]}),
        ("GET", f"{MESSAGES}/m1"): ok(full_message(plain="x")),
        ("GET", f"{MESSAGES}/m2"): ok(full_message(mid="m2", subject="Lunch?", plain="y")),
    })
    out = run(provider.search("from:rahul", 5))
    assert out["count"] == 2 and out["query"] == "from:rahul"
    assert out["truncated"] is False
    first = out["messages"][0]
    assert set(first) == {"id", "thread_id", "sender", "subject", "date", "snippet"}
    assert first["sender"] == "Rahul <rahul@example.com>"
    assert first["subject"] == "Interview tomorrow"
    assert all(s == GMAIL_READONLY for s in oauth.scopes)
    assert calls[0].url.params["q"] == "from:rahul"
    assert calls[0].url.params["maxResults"] == "5"


def test_search_paginates_with_page_token(gmail_factory):
    def list_route(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("pageToken") == "p2":
            return ok({"messages": [{"id": "m3"}]})
        return ok({"messages": [{"id": "m1"}, {"id": "m2"}], "nextPageToken": "p2"})

    provider, _, calls = gmail_factory({
        ("GET", MESSAGES): list_route,
        ("GET", f"{MESSAGES}/m1"): ok(full_message(plain="a")),
        ("GET", f"{MESSAGES}/m2"): ok(full_message(mid="m2", plain="b")),
        ("GET", f"{MESSAGES}/m3"): ok(full_message(mid="m3", plain="c")),
    })
    out = run(provider.search("anything", 3))
    assert out["count"] == 3 and out["truncated"] is False
    assert any(c.url.params.get("pageToken") == "p2" for c in calls)


def test_search_caps_results_and_flags_truncation(gmail_factory):
    # Google ignored maxResults and returned too many ids -> cap + flag
    provider, _, _ = gmail_factory({
        ("GET", MESSAGES): ok({"messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]}),
        ("GET", f"{MESSAGES}/m1"): ok(full_message(plain="a")),
        ("GET", f"{MESSAGES}/m2"): ok(full_message(mid="m2", plain="b")),
        ("GET", f"{MESSAGES}/m3"): ok(full_message(mid="m3", plain="c")),
    })
    out = run(provider.search("q", 2))
    assert out["count"] == 2 and out["truncated"] is True


def test_search_empty_inbox(gmail_factory):
    provider, _, _ = gmail_factory({("GET", MESSAGES): ok({"messages": []})})
    out = run(provider.search("is:unread", 10))
    assert out == {"messages": [], "count": 0, "query": "is:unread", "truncated": False}


def test_search_malformed_query_is_controlled(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", MESSAGES): err(400, "Invalid argument: q"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.search("bad:{query", 5))
    assert exc.value.code == "GOOGLE_API_ERROR"


def test_search_skips_message_that_404s(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", MESSAGES): ok({"messages": [{"id": "m1"}, {"id": "gone"}]}),
        ("GET", f"{MESSAGES}/m1"): ok(full_message(plain="a")),
        ("GET", f"{MESSAGES}/gone"): err(404, "Not Found"),
    })
    out = run(provider.search("q", 5))
    assert out["count"] == 1


def test_search_result_fields_are_capped(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", MESSAGES): ok({"messages": [{"id": "m1"}]}),
        ("GET", f"{MESSAGES}/m1"): ok(full_message(
            subject="S" * 500, snippet="n" * 900, frm="F" * 500, plain="x"
        )),
    })
    meta = run(provider.search("q", 5))["messages"][0]
    assert len(meta["subject"]) <= 200
    assert len(meta["snippet"]) <= 240
    assert len(meta["sender"]) <= 200


# --------------------------------------------------------------------- get

def test_get_returns_plain_body(gmail_factory):
    provider, oauth, _ = gmail_factory({
        ("GET", f"{MESSAGES}/m1"): ok(full_message(plain="Meet at 10.\nThanks")),
    })
    out = run(provider.get("m1"))
    assert out["body"] == "Meet at 10.\nThanks"
    assert out["body_is_html_stripped"] is False
    assert out["truncated"] is False
    assert out["attachments"] == []
    assert out["thread_id"] == "t1"
    assert oauth.scopes == [GMAIL_READONLY]


def test_get_strips_html_and_drops_script(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", f"{MESSAGES}/m1"): ok(full_message(
            html="<p>Hello <b>there</b></p><script>alert('x')</script><style>.x{}</style>"
        )),
    })
    out = run(provider.get("m1"))
    assert out["body_is_html_stripped"] is True
    assert "alert" not in out["body"]
    assert "Hello there" in out["body"]


def test_get_prefers_plain_part_over_html(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", f"{MESSAGES}/m1"): ok(full_message(plain="PLAIN wins", html="<p>HTML</p>")),
    })
    out = run(provider.get("m1"))
    assert out["body"] == "PLAIN wins"
    assert out["body_is_html_stripped"] is False


def test_get_lists_attachment_metadata_without_content(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", f"{MESSAGES}/m1"): ok(full_message(
            plain="see attached",
            attachments=[("report.pdf", "application/pdf", 12345, "att-1"),
                         ("notes.txt", "text/plain", 99, "att-2")],
        )),
    })
    out = run(provider.get("m1"))
    assert len(out["attachments"]) == 2
    assert out["attachments"][0] == {
        "filename": "report.pdf", "mime_type": "application/pdf",
        "size": 12345, "attachment_id": "att-1",
    }
    assert "data" not in str(out["attachments"][0])  # metadata only


def test_get_truncates_huge_body(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", f"{MESSAGES}/m1"): ok(full_message(plain="Z" * 5000)),
    }, max_body_chars=500)
    out = run(provider.get("m1"))
    assert out["truncated"] is True
    assert len(out["body"]) == 500


def test_get_unknown_id_is_email_not_found(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", f"{MESSAGES}/nope"): err(404, "Not Found"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.get("nope"))
    assert exc.value.code == "EMAIL_NOT_FOUND"
    assert "nope" in exc.value.message


# ---------------------------------------------------------------- writes

def test_create_draft_builds_mime_and_scope(gmail_factory):
    provider, oauth, calls = gmail_factory({(POST := "POST", DRAFTS): ok({"id": "dr1", "message": {"threadId": "t9"}})})
    out = run(provider.create_draft(["a@example.com", "b@example.com"], "Hi", "Body text"))
    assert out == {"draft_id": "dr1", "thread_id": "t9", "status": "created"}
    assert oauth.scopes == [GMAIL_COMPOSE]
    msg = decode_raw(calls[0])
    assert msg["To"] == "a@example.com, b@example.com"
    assert msg["Subject"] == "Hi"
    assert msg.get_payload().rstrip("\n") == "Body text"


def test_send_builds_mime_and_uses_send_scope(gmail_factory):
    provider, oauth, calls = gmail_factory({
        ("POST", "/gmail/v1/users/me/messages/send"): ok({"id": "ms1", "threadId": "t9"}),
    })
    out = run(provider.send(["x@example.com"], "Subject", "Hello there"))
    assert out["status"] == "sent" and out["message_id"] == "ms1"
    assert out["to"] == ["x@example.com"]
    assert oauth.scopes == [GMAIL_SEND]
    msg = decode_raw(calls[0])
    assert msg["To"] == "x@example.com" and msg["Subject"] == "Subject"
    assert msg.get_payload().rstrip("\n") == "Hello there"


def test_send_draft_posts_existing_id(gmail_factory):
    provider, oauth, calls = gmail_factory({
        ("POST", "/gmail/v1/users/me/drafts/send"): ok({"id": "ms2", "threadId": ""}),
    })
    out = run(provider.send_draft("dr7"))
    assert out["draft_id"] == "dr7" and out["status"] == "sent"
    import json

    assert json.loads(calls[0].content) == {"id": "dr7"}
    assert oauth.scopes == [GMAIL_SEND]


def test_send_draft_missing_is_email_not_found(gmail_factory):
    provider, _, _ = gmail_factory({
        ("POST", "/gmail/v1/users/me/drafts/send"): err(404, "Not Found"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.send_draft("missing-1"))
    assert exc.value.code == "EMAIL_NOT_FOUND"


def test_send_rejecting_bad_recipient(gmail_factory):
    provider, _, _ = gmail_factory({
        ("POST", "/gmail/v1/users/me/messages/send"): err(400, "Invalid recipient address"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.send(["bad"], "S", "B"))
    assert exc.value.code == "INVALID_RECIPIENT"


def test_send_other_400_is_controlled_api_error(gmail_factory):
    provider, _, _ = gmail_factory({
        ("POST", "/gmail/v1/users/me/messages/send"): err(400, "Message is too large"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.send(["a@example.com"], "S", "B"))
    assert exc.value.code == "GOOGLE_API_ERROR"


# ---------------------------------------------------------- errors & retries

def test_rate_limit_retries_then_succeeds(gmail_factory):
    provider, _, calls = gmail_factory({
        ("GET", MESSAGES): [
            httpx.Response(429, json={"error": {"code": 429, "message": "Rate"}}),
            ok({"messages": [{"id": "m1"}]}),
        ],
        ("GET", f"{MESSAGES}/m1"): ok(full_message(plain="a")),
    })
    out = run(provider.search("q", 5))
    assert out["count"] == 1
    assert sum(1 for c in calls if c.url.path == MESSAGES) == 2  # retried


def test_persistent_rate_limit_maps_out(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", MESSAGES): httpx.Response(429, json={"error": {"code": 429, "message": "Rate"}}),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.search("q", 5))
    assert exc.value.code == "API_RATE_LIMITED"


def test_persistent_5xx_is_controlled(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", MESSAGES): err(500, "Backend error"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.search("q", 5))
    assert exc.value.code == "GOOGLE_API_ERROR"


def test_401_is_auth_expired(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", MESSAGES): err(401, "Invalid Credentials"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.search("q", 5))
    assert exc.value.code == "AUTH_EXPIRED"


def test_403_insufficient_permission_is_auth_required(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", MESSAGES): err(403, "Insufficient Permission"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.search("q", 5))
    assert exc.value.code == "AUTH_REQUIRED"


def test_403_other_is_permission_denied(gmail_factory):
    provider, _, _ = gmail_factory({
        ("GET", MESSAGES): err(403, "Domain policy forbids this action"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.search("q", 5))
    assert exc.value.code == "PERMISSION_DENIED"


def test_oauth_error_propagates_unchanged(gmail_factory):
    oauth = FakeOAuth(ProductivityError("AUTH_REQUIRED", "not authorized"))
    provider, _, calls = gmail_factory({("GET", MESSAGES): ok({})}, oauth=oauth)
    with pytest.raises(ProductivityError) as exc:
        run(provider.search("q", 5))
    assert exc.value.code == "AUTH_REQUIRED"
    assert not calls  # failed before any HTTP


def test_access_token_never_in_logs(gmail_factory, caplog):
    import logging

    provider, _, _ = gmail_factory({
        ("GET", f"{MESSAGES}/m1"): ok(full_message(plain="secret body")),
    })
    with caplog.at_level(logging.DEBUG):
        run(provider.get("m1"))
    blob = " ".join(r.getMessage() for r in caplog.records)
    assert "tok-SECRET" not in blob
    assert "Bearer" not in blob


def test_html_to_text_helper_drops_active_content():
    out = html_to_text("<div>Hi</div><script>evil()</script><style>a{}</style><p>Bye</p>")
    assert "evil" not in out and "Hi" in out and "Bye" in out
