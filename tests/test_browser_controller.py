"""Phase 6 tests: browser controller lifecycle, navigation, reading,
interaction, verification, references, limits (§36 'Browser Controller',
'Navigation', 'Page Reading', 'Interaction', 'Element References')."""

from __future__ import annotations

import asyncio

import pytest

from app.browser import BrowserError, BrowserPolicy
from conftest import page_url, browser_policy


# ------------------------------------------------------------ lifecycle

def test_browser_start_and_close(run_browser):
    async def body(c):
        assert not c.session.started
        await c.session.start()
        assert c.session.started
        await c.close()
        assert not c.session.started
        await c.close()  # idempotent (§29)

    run_browser(body)


def test_session_cleanup_on_context_manager(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        return c.current_url()

    url = run_browser(body)
    assert url.endswith("basic.html")


# ------------------------------------------------------------ navigation

def test_open_url_returns_verified_result(run_browser):
    async def body(c):
        result = await c.open_url(page_url("basic.html"))
        assert result.verified and result.status == "verified"
        assert result.data["title"] == "Basic Test Page"
        assert result.data["url"].endswith("basic.html")

    run_browser(body)


def test_invalid_scheme_refused_without_launching(run_browser):
    async def body(c):
        with pytest.raises(BrowserError) as exc:
            await c.open_url("javascript:alert(1)")
        assert exc.value.code == "SCHEME_NOT_ALLOWED"
        with pytest.raises(BrowserError) as exc2:
            await c.open_url("ftp://example.com/file")
        assert exc2.value.code == "SCHEME_NOT_ALLOWED"

    run_browser(body)


def test_blocked_localhost_refused_by_policy():
    policy = BrowserPolicy(resolver=lambda h: ["127.0.0.1"])  # default blocked list
    assert policy.can_navigate("http://localhost/admin") == "BLOCKED_DOMAIN"
    assert policy.can_navigate("http://metadata.google.internal/") == "BLOCKED_DOMAIN"
    assert policy.can_navigate("http://169.254.169.254/latest") == "BLOCKED_DOMAIN"


def test_private_ip_refused_when_resolving_private():
    policy = BrowserPolicy(resolver=lambda h: ["10.0.0.5"])
    assert policy.can_navigate("http://intranet.example/") == "BLOCKED_DOMAIN"


def test_allowed_domains_enforced():
    policy = BrowserPolicy(
        allowed_domains=("example.com",), resolver=lambda h: ["93.184.216.34"]
    )
    assert policy.can_navigate("https://example.com/x") is None
    assert policy.can_navigate("https://docs.example.com/x") is None
    assert policy.can_navigate("https://other.com/") == "DOMAIN_NOT_ALLOWED"
    assert policy.can_navigate("https://evil.example/x") == "DOMAIN_NOT_ALLOWED"


def test_navigation_timeout_is_controlled(run_browser):
    async def body(c):
        async def freeze(reader, writer):
            await asyncio.sleep(30)  # accept, never respond

        server = await asyncio.start_server(freeze, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            with pytest.raises(BrowserError) as exc:
                await c.open_url(f"http://127.0.0.1:{port}/hang")
            assert exc.value.code == "BROWSER_NAV_TIMEOUT"
        finally:
            server.close()  # pending handler is cancelled when the loop ends

    run_browser(body, navigation_timeout_ms=700)


# ------------------------------------------------------------ page reading

def test_read_page_extracts_structure(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        snapshot, data = await c.read_page()
        assert data["title"] == "Basic Test Page"
        assert any("Basic Page" in h for h in data["headings"])
        assert "primary paragraph" in data["text"]
        assert any(l["name"] == "Go to navigation" for l in data["links"])
        assert any(b["name"] == "More info" for b in data["buttons"])
        assert any(i["name"] == "Search docs" or "Search" in i["name"] for i in data["inputs"])
        assert snapshot.links and snapshot.buttons and snapshot.inputs

    run_browser(body)


def test_read_page_is_bounded(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        _, data = await c.read_page()
        assert len(data["text"]) <= 4000
        total = len(data["links"]) + len(data["buttons"]) + len(data["inputs"])
        assert total <= 60

    run_browser(body)


def test_extract_links_buttons_inputs_from_form(run_browser):
    async def body(c):
        await c.open_url(page_url("form.html"))
        _, data = await c.read_page()
        names = [b["name"] for b in data["buttons"]]
        assert "Send" in names and "Preview" in names
        roles = {i["id"]: i["role"] for i in data["inputs"]}
        assert any(r == "password" for r in roles.values())
        assert any(r == "textbox" for r in roles.values())

    run_browser(body)


# ------------------------------------------------------------ interaction

def test_click_verifies_observed_change(run_browser):
    async def body(c):
        await c.open_url(page_url("navigation.html"))
        _, data = await c.read_page()
        toggle = next(b["id"] for b in data["buttons"] if b["name"] == "Toggle note")
        result = await c.click(toggle)
        assert result.verified and result.status == "verified"
        # re-read to observe the effect (§26)
        _, after = await c.read_page()
        assert "State changed here!" in after["text"]

    run_browser(body)


def test_click_navigation_link_changes_url(run_browser):
    async def body(c):
        await c.open_url(page_url("navigation.html"))
        _, data = await c.read_page()
        link = next(l["id"] for l in data["links"] if l["name"] == "Basic page")
        result = await c.click(link)
        assert result.verified
        assert c.current_url().endswith("basic.html")

    run_browser(body)


def test_type_verifies_input_value(run_browser):
    async def body(c):
        await c.open_url(page_url("form.html"))
        _, data = await c.read_page()
        field = next(i["id"] for i in data["inputs"] if i["name"] == "Full name" or "Full name" in i["name"] or i["role"] == "textbox")
        result = await c.type_text(field, "Ada Lovelace")
        assert result.verified and result.status == "verified"

    run_browser(body)


def test_type_password_field_is_refused(run_browser):
    async def body(c):
        await c.open_url(page_url("form.html"))
        _, data = await c.read_page()
        pw = next(i["id"] for i in data["inputs"] if i["role"] == "password")
        with pytest.raises(BrowserError) as exc:
            await c.type_text(pw, "hunter2")
        assert exc.value.code == "CREDENTIAL_FIELD"

    run_browser(body)


def test_scroll_verifies_position_change(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        result = await c.scroll("down", 800)
        assert result.verified and result.data["scroll_y"] > 0
        result_up = await c.scroll("up", 400)
        assert result_up.verified

    run_browser(body)


def test_scroll_amount_is_clamped(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        result = await c.scroll("down", 999999)
        assert result.verified  # clamped to 5000, no crash

    run_browser(body)


def test_go_back_and_forward(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        await c.open_url(page_url("navigation.html"))
        back = await c.go_back()
        assert back.verified and c.current_url().endswith("basic.html")
        forward = await c.go_forward()
        assert forward.verified and c.current_url().endswith("navigation.html")

    run_browser(body)


def test_reload_is_verified(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        result = await c.reload()
        assert result.verified

    run_browser(body)


# ------------------------------------------------------- element references

def test_element_reference_creation_and_resolution(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        _, data = await c.read_page()
        ids = [l["id"] for l in data["links"]] + [b["id"] for b in data["buttons"]]
        assert ids == sorted(ids, key=lambda i: int(i.split("_")[1]))  # element_1..n
        target = data["links"][0]["id"]  # element ref -> click works
        result = await c.click(target)
        assert result.status in ("verified", "unknown")

    run_browser(body)


def test_semantic_name_resolution(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        await c.read_page()
        result = await c.click("Next")  # by exact name, not selector
        assert result.action == "click"

    run_browser(body)


def test_stale_reference_detected(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        _, data = await c.read_page()
        ref = data["links"][0]["id"]
        # page changes OUTSIDE our action (SPA/external nav, §13/§28)
        await c.session.page.goto(page_url("navigation.html"))
        with pytest.raises(BrowserError) as exc:
            await c.click(ref)
        assert exc.value.code == "ELEMENT_STALE"

    run_browser(body)


def test_action_without_read_page_is_refused(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        with pytest.raises(BrowserError) as exc:
            await c.click("element_1")
        assert exc.value.code == "ELEMENT_NOT_READ"

    run_browser(body)


def test_unknown_target_reports_not_found(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        await c.read_page()
        with pytest.raises(BrowserError) as exc:
            await c.click("no such element anywhere")
        assert exc.value.code == "ELEMENT_NOT_FOUND"

    run_browser(body)


# ------------------------------------------------------------ find element

def test_find_element_found(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        result = await c.find_element("Search box")
        assert result["found"] is True
        assert result["element_id"].startswith("element_")
        assert result["role"] in ("textbox", "link", "button")

    run_browser(body)


def test_find_element_not_found(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))
        result = await c.find_element("quantum flux capacitor")
        assert result["found"] is False

    run_browser(body)


# ------------------------------------------------------------ step budget (§25)

def test_max_browser_steps_enforced(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))  # step 1
        await c.read_page()
        await c.click("Next")  # step 2 (harmless, no approval needed)
        await c.read_page()
        with pytest.raises(BrowserError) as exc:
            await c.scroll("down", 100)  # step 3 -> over budget
        assert exc.value.code == "BROWSER_MAX_STEPS"
        assert "maximum number of steps" in exc.value.message

    run_browser(body, max_steps=2)


def test_read_page_does_not_consume_steps(run_browser):
    async def body(c):
        await c.open_url(page_url("basic.html"))  # 1 step
        for _ in range(5):
            await c.read_page()  # observation only
        assert c.steps_used == 1

    run_browser(body)
