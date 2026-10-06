"""Phase 5 tests: safe page fetching (spec sections 9/20/29) — fully mocked."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.research.fetcher import FetchError, PageFetcher, ip_is_public


def run(coro):
    return asyncio.run(coro)


PUBLIC = ["93.184.216.34"]


def make_fetcher(handler, max_size: int = 2 * 1024 * 1024, timeout: float = 5.0) -> PageFetcher:
    return PageFetcher(
        timeout=timeout,
        max_size=max_size,
        transport=httpx.MockTransport(handler),
        resolver=lambda host: PUBLIC,
    )


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        text="<html><head><title>Example Page</title></head><body><p>Hello world content here.</p></body></html>",
    )


def test_open_page_returns_structured_result():
    fetcher = make_fetcher(ok_handler)
    page = run(fetcher.fetch("https://example.com/page"))
    assert page.status_code == 200
    assert page.final_url.startswith("https://example.com/page")
    assert page.title == "Example Page"
    assert "Hello world" in page.content
    assert page.content_type == "text/html"
    assert page.truncated is False


def test_invalid_scheme_rejected():
    fetcher = make_fetcher(ok_handler)
    for url in ("file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)"):
        with pytest.raises(FetchError) as exc:
            run(fetcher.fetch(url))
        assert exc.value.code == "INVALID_URL"


def test_localhost_rejected_without_dns():
    fetcher = make_fetcher(ok_handler)
    with pytest.raises(FetchError) as exc:
        run(fetcher.fetch("http://localhost/admin"))
    assert exc.value.code == "FORBIDDEN_HOST"


def test_private_literal_ip_rejected():
    fetcher = make_fetcher(ok_handler)
    for host in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "[::1]"):
        with pytest.raises(FetchError) as exc:
            run(fetcher.fetch(f"http://{host}/latest/meta-data/"))
        assert exc.value.code == "FORBIDDEN_HOST"


def test_resolved_private_host_rejected():
    fetcher = PageFetcher(
        transport=httpx.MockTransport(ok_handler),
        resolver=lambda host: ["192.168.0.10"],
    )
    with pytest.raises(FetchError) as exc:
        run(fetcher.fetch("https://internal.example.com/"))
    assert exc.value.code == "FORBIDDEN_HOST"


def test_redirect_to_private_host_blocked():
    def redirect_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
        )

    fetcher = make_fetcher(redirect_handler)
    with pytest.raises(FetchError) as exc:
        run(fetcher.fetch("https://example.com/redirect"))
    assert exc.value.code == "FORBIDDEN_HOST"


def test_http_error_is_controlled():
    fetcher = make_fetcher(lambda request: httpx.Response(404, text="nope"))
    with pytest.raises(FetchError) as exc:
        run(fetcher.fetch("https://example.com/missing"))
    assert exc.value.code == "HTTP_404"


def test_timeout_is_controlled():
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    fetcher = make_fetcher(timeout_handler)
    with pytest.raises(FetchError) as exc:
        run(fetcher.fetch("https://slow.example.com/"))
    assert exc.value.code == "FETCH_TIMEOUT"


def test_unsupported_content_type_rejected():
    fetcher = make_fetcher(
        lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4"
        )
    )
    with pytest.raises(FetchError) as exc:
        run(fetcher.fetch("https://example.com/file.pdf"))
    assert exc.value.code == "UNSUPPORTED_CONTENT_TYPE"


def test_oversized_page_truncated_not_hung():
    big_html = "<html><body>" + ("x" * 10_000) + "</body></html>"
    fetcher = make_fetcher(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/html"}, text=big_html
        ),
        max_size=1000,
    )
    page = run(fetcher.fetch("https://example.com/big"))
    assert page.truncated is True
    assert len(page.content) <= 1000


def test_ip_is_public():
    assert ip_is_public("93.184.216.34") is True
    assert ip_is_public("8.8.8.8") is True
    assert ip_is_public("127.0.0.1") is False
    assert ip_is_public("10.1.2.3") is False
    assert ip_is_public("169.254.169.254") is False
    assert ip_is_public("0.0.0.0") is False
    assert ip_is_public("not-an-ip") is False
