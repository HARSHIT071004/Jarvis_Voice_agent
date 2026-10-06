"""Phase 6 tests: browser tools through the standard registry/router and
the Phase 4 AgentRuntime loop (§36 'Agent Loop', §23/§24 integration)."""

from __future__ import annotations

import asyncio

import pytest

from app.agent.runtime import AgentResponse, AgentRuntime, LLMReply, LLMToolCall
from app.browser import BrowserController
from app.tools import RiskLevel, ToolRouter, build_registry, build_web_tools
from app.tools.browser import BROWSER_TOOL_NAMES, build_browser_tools
from app.research.providers import WebSearchProvider
from conftest import make_controller, page_url, browser_policy


def make_router(controller: BrowserController, timeout: float = 25.0) -> ToolRouter:
    registry = build_registry(None, browser_tools=build_browser_tools(controller))
    return ToolRouter(registry, execution_timeout=timeout)


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ registry (§23)

def test_browser_tools_registered_with_risk_levels():
    controller = BrowserController(browser_policy())
    registry = build_registry(None, browser_tools=build_browser_tools(controller))
    assert BROWSER_TOOL_NAMES <= set(registry.names())
    expected_risks = {
        "open_url": RiskLevel.READ,
        "read_page": RiskLevel.READ,
        "find_element": RiskLevel.READ,
        "click": RiskLevel.WRITE,
        "type": RiskLevel.WRITE,
        "scroll": RiskLevel.WRITE,
        "go_back": RiskLevel.WRITE,
        "go_forward": RiskLevel.WRITE,
        "reload": RiskLevel.WRITE,
        "download": RiskLevel.WRITE,
        "upload": RiskLevel.SENSITIVE,
    }
    for name, risk in expected_risks.items():
        assert registry.get(name).risk is risk, name


def test_browser_and_web_tools_coexist():
    """§32: research + browser in ONE registry, one router.

    Both specs mandate a tool named `read_page` (P5: read_page(url),
    P6: read_page() on the live page) — build_registry merges them into
    one optional-url tool instead of crashing or dropping either.
    """

    class DummySearch(WebSearchProvider):
        async def search(self, query, max_results=8):
            return []

    controller = BrowserController(browser_policy())
    registry = build_registry(
        None,
        web_tools=build_web_tools(DummySearch(), fetcher=None),  # type: ignore[arg-type]
        browser_tools=build_browser_tools(controller),
    )
    assert "search_web" in registry.names()
    assert "open_url" in registry.names()
    assert len(registry) >= 13
    merged = registry.get("read_page")
    merged.validate({})  # Phase 6 form: current browser page
    merged.validate({"url": "https://example.com/docs"})  # Phase 5 form
    assert "url" in merged.args_model.model_fields


# ------------------------------------------------------------ tool execution

def test_open_url_tool_via_router(run_browser):
    async def body(c):
        router = make_router(c)
        result = await router.route("open_url", {"url": page_url("basic.html")})
        assert result.success and result.risk == "READ"
        assert result.data["status"] == "verified"
        page = await router.route("read_page", {})
        assert page.success
        assert any(l["name"] == "Go to navigation" for l in page.data["links"])

    run_browser(body)


def test_open_url_invalid_arguments_rejected(run_browser):
    async def body(c):
        router = make_router(c)
        result = await router.route("open_url", {})  # missing url
        assert result.success is False and result.error["code"] == "INVALID_ARGUMENTS"
        result2 = await router.route("open_url", {"url": "javascript:alert(1)"})
        assert result2.success is False
        assert result2.error["code"] in ("INVALID_ARGUMENTS", "SCHEME_NOT_ALLOWED")

    run_browser(body)


def test_type_and_click_via_router(run_browser):
    async def body(c):
        router = make_router(c)
        await router.route("open_url", {"url": page_url("form.html")})
        page = await router.route("read_page", {})
        field = next(i["id"] for i in page.data["inputs"] if i["role"] == "textbox")
        typed = await router.route("type", {"target": field, "text": "Grace Hopper"})
        assert typed.success and typed.data["verified"] is True
        preview = next(b["id"] for b in page.data["buttons"] if b["name"] == "Preview")
        clicked = await router.route("click", {"target": preview})
        assert clicked.success and clicked.data["status"] == "verified"

    run_browser(body)


def test_sensitive_tool_error_flows_through_router(run_browser):
    async def body(c):
        router = make_router(c)
        await router.route("open_url", {"url": page_url("form.html")})
        page = await router.route("read_page", {})
        send = next(b["id"] for b in page.data["buttons"] if b["name"] == "Send")
        result = await router.route("click", {"target": send})
        assert result.success is False
        assert result.error["code"] == "APPROVAL_REQUIRED"
        assert "permission" in result.error["message"].lower()

    run_browser(body)


def test_scroll_direction_validation(run_browser):
    async def body(c):
        router = make_router(c)
        await router.route("open_url", {"url": page_url("basic.html")})
        result = await router.route("scroll", {"direction": "sideways", "amount": 100})
        assert result.error["code"] == "INVALID_ARGUMENTS"

    run_browser(body)


# --------------------------------------------------------- agent loop (§24)

class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def complete(self, user_text, history, tool_results, transcript):
        self.calls.append({"tool_results": tool_results})
        if not self.replies:
            raise AssertionError("script exhausted")
        return self.replies.pop(0)


def call(name, args, call_id=None):
    return LLMReply(tool_calls=[LLMToolCall(name=name, arguments=args, call_id=call_id)])


def test_multi_step_browser_task(run_browser):
    async def body(c):
        router = make_router(c)
        llm = ScriptedLLM(
            [
                call("open_url", {"url": page_url("navigation.html")}, "c1"),
                call("read_page", {}, "c2"),
                call("click", {"target": "element_1"}, "c3"),  # 'Basic page' link
                call("read_page", {}, "c4"),
                LLMReply(text="Opened the navigation page and clicked through to the basic page."),
            ]
        )
        rt = AgentRuntime(llm, router, max_iterations=6, max_tool_calls=8)
        resp = await rt.handle("Open the navigation page and go to the basic page.")
        assert isinstance(resp, AgentResponse)
        assert resp.error is None
        assert len(resp.tool_results) == 4
        assert all(r.success for r in resp.tool_results)
        # each round saw the real structured browser state (observe -> replan)
        click_seen = llm.calls[3]["tool_results"][0]
        assert click_seen["data"]["status"] in ("verified", "unknown")
        assert "basic" in resp.text.lower()

    run_browser(body)


def test_failed_action_is_recoverable(run_browser):
    async def body(c):
        router = make_router(c)
        llm = ScriptedLLM(
            [
                call("open_url", {"url": page_url("basic.html")}, "c1"),
                call("read_page", {}, "c2"),
                call("click", {"target": "no-such-thing"}, "c3"),  # fails
                call("read_page", {}, "c4"),  # replan: observe the page
                LLMReply(text="I could not find that element on the page."),
            ]
        )
        rt = AgentRuntime(llm, router, max_iterations=5, max_tool_calls=6)
        resp = await rt.handle("click the thing")
        assert resp.error is None
        failed = resp.tool_results[2]
        assert failed.success is False
        assert failed.error["code"] == "ELEMENT_NOT_FOUND"
        # the failure was fed back to the LLM (no hallucinated success)
        assert llm.calls[3]["tool_results"][0]["success"] is False
        assert "could not find" in resp.text.lower()

    run_browser(body)


def test_max_browser_steps_stops_task(run_browser):
    async def body(c):
        router = make_router(c)
        llm = ScriptedLLM(
            [
                call("open_url", {"url": page_url("basic.html")}, "c1"),
                call("read_page", {}, "c2"),
                call("click", {"target": "element_3"}, "c3"),  # 'Next' button = step 2
                call("scroll", {"direction": "down", "amount": 100}, "c4"),  # step 3 -> over
                LLMReply(text="Browser task stopped because the maximum number of steps was reached."),
            ]
        )
        rt = AgentRuntime(llm, router, max_iterations=5, max_tool_calls=6)
        resp = await rt.handle("do many browser things")
        # results: open(c1), read(c2), click(c3), scroll(c4)
        assert len(resp.tool_results) == 4
        overflow = resp.tool_results[3]
        assert overflow.success is False
        assert overflow.error["code"] == "BROWSER_MAX_STEPS"
        assert "maximum number of steps" in overflow.error["message"]

    run_browser(body, max_steps=2)


def test_action_verification_visible_to_llm(run_browser):
    async def body(c):
        router = make_router(c)
        llm = ScriptedLLM(
            [
                call("open_url", {"url": page_url("navigation.html")}, "c1"),
                call("read_page", {}, "c2"),
                call("click", {"target": "Toggle note"}, "c3"),  # JS state change, no nav
                LLMReply(text="The note toggled."),
            ]
        )
        rt = AgentRuntime(llm, router, max_iterations=5, max_tool_calls=6)
        resp = await rt.handle("toggle the note")
        click_result = resp.tool_results[2]
        assert click_result.success is True
        assert click_result.data["status"] == "verified"
        assert click_result.data["verified"] is True

    run_browser(body)


def test_navigation_to_blocked_site_is_controlled():
    from app.browser import BrowserPolicy

    async def main():
        controller = BrowserController(
            BrowserPolicy(resolver=lambda h: ["127.0.0.1"]), headless=True
        )
        try:
            router = make_router(controller)
            result = await router.route("open_url", {"url": "http://localhost:8080/admin"})
            assert result.success is False
            assert result.error["code"] == "BLOCKED_DOMAIN"
        finally:
            await controller.close()

    run(main())
