"""Phase 5 tests: web tools + registry integration (spec sections 3/22/29)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.research.fetcher import PageFetcher
from app.research.models import SearchResult
from app.research.providers import SearchProviderError, WebSearchProvider
from app.tools import RiskLevel, ToolError, ToolRouter, build_registry, build_web_tools
from app.tools.web import PageCache, SearchWebTool


def run(coro):
    return asyncio.run(coro)


class FakeProvider(WebSearchProvider):
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error

    async def search(self, query, max_results=8):
        if self.error:
            raise SearchProviderError(self.error)
        return self.results


def make_fetcher(handler) -> PageFetcher:
    return PageFetcher(
        transport=httpx.MockTransport(handler),
        resolver=lambda host: ["93.184.216.34"],
    )


HTML_PAGE = """<html><head><title>Docs</title></head><body>
<nav>Menu Home</nav><main>
<h1>Getting Started</h1>
<p>The Gemini Live API supports function calling when tools are declared in the session configuration.</p>
</main></body></html>"""


def page_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html"}, text=HTML_PAGE)


# ------------------------------------------------------------- registry

def test_web_tools_registered_with_read_risk():
    tools = build_web_tools(FakeProvider(), make_fetcher(page_handler))
    registry = build_registry(None, web_tools=tools)
    assert {"search_web", "open_page", "read_page", "extract_content"} <= set(registry.names())
    for name in ("search_web", "open_page", "read_page", "extract_content"):
        assert registry.get(name).risk is RiskLevel.READ


def test_web_tools_route_through_standard_router():
    registry = build_registry(None, web_tools=build_web_tools(
        FakeProvider([SearchResult(title="t", url="https://x.example.com/", source="x.example.com")]),
        make_fetcher(page_handler),
    ))
    router = ToolRouter(registry)
    result = run(router.route("search_web", {"query": "gemini live"}))
    assert result.success is True
    assert result.risk == "READ"
    assert result.data["results"][0]["url"] == "https://x.example.com/"


# -------------------------------------------------------- search tool

def test_search_web_returns_structured_results():
    tool = SearchWebTool(FakeProvider([
        SearchResult(title="Docs", url="https://a.example.com/", source="a.example.com", snippet=None, published_at=None)
    ]))
    data = run(tool.execute(tool.validate({"query": "latest model"})))
    assert data["results"][0]["title"] == "Docs"
    assert data["results"][0]["published_at"] is None  # null when unknown


def test_search_web_provider_error_becomes_tool_error():
    tool = SearchWebTool(FakeProvider(error="SEARCH_TIMEOUT"))
    with pytest.raises(ToolError) as exc:
        run(tool.execute(tool.validate({"query": "gemini live"})))
    assert exc.value.code == "SEARCH_TIMEOUT"


def test_search_web_invalid_arguments_rejected():
    tool = SearchWebTool(FakeProvider())
    with pytest.raises(ToolError):
        tool.validate({"query": "x"})  # too short
    with pytest.raises(ToolError):
        tool.validate({"query": "valid query", "max_results": 99})


# ---------------------------------------------------------- page tools

def test_open_page_fetches_and_previews():
    registry = build_registry(None, web_tools=build_web_tools(FakeProvider(), make_fetcher(page_handler)))
    router = ToolRouter(registry)
    result = run(router.route("open_page", {"url": "https://example.com/docs"}))
    assert result.success
    assert result.data["status_code"] == 200
    assert result.data["title"] == "Docs"
    assert "function calling" in result.data["content"]


def test_page_cache_prevents_refetch():
    calls = {"n": 0}

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return page_handler(request)

    tools = build_web_tools(FakeProvider(), make_fetcher(counting_handler))
    open_tool = next(t for t in tools if t.name == "open_page")
    read_tool = next(t for t in tools if t.name == "read_page")
    url = "https://example.com/docs"
    run(open_tool.execute(open_tool.validate({"url": url})))
    run(read_tool.execute(read_tool.validate({"url": url})))
    assert calls["n"] == 1  # second call served from PageCache


def test_read_page_removes_navigation():
    registry = build_registry(None, web_tools=build_web_tools(FakeProvider(), make_fetcher(page_handler)))
    router = ToolRouter(registry)
    result = run(router.route("read_page", {"url": "https://example.com/docs"}))
    assert result.success
    assert "Menu" not in result.data["text"]
    assert "function calling" in result.data["text"]


def test_extract_content_returns_structure():
    registry = build_registry(None, web_tools=build_web_tools(FakeProvider(), make_fetcher(page_handler)))
    router = ToolRouter(registry)
    result = run(router.route("extract_content", {"url": "https://example.com/docs"}))
    assert result.success
    assert "Getting Started" in result.data["headings"]
    assert any("function calling" in p for p in result.data["paragraphs"])


def test_non_http_url_rejected_by_validation():
    tools = build_web_tools(FakeProvider(), make_fetcher(page_handler))
    open_tool = next(t for t in tools if t.name == "open_page")
    with pytest.raises(ToolError) as exc:
        open_tool.validate({"url": "javascript:alert(1)"})
    assert exc.value.code == "INVALID_ARGUMENTS"


def test_page_cache_eviction_is_bounded():
    from app.research.models import RawPage

    cache = PageCache(max_entries=2)
    for i in range(3):
        cache.put(
            f"https://e{i}.example.com/",
            RawPage(url=f"u{i}", final_url=f"u{i}", status_code=200, content="x"),
        )
    assert len(cache._pages) == 2
    assert cache.get("https://e0.example.com/") is None
    assert cache.get("https://e2.example.com/") is not None


# ------------------------------------------- Phase 4 runtime integration (§23)

def test_agent_runtime_calls_web_tool_in_standard_loop():
    """The Phase 4 AgentRuntime drives search_web via the same registry."""
    from app.agent.runtime import AgentRuntime, LLMReply, LLMToolCall

    provider = FakeProvider([
        SearchResult(title="Live docs", url="https://ai.google.dev/live", source="ai.google.dev")
    ])
    registry = build_registry(None, web_tools=build_web_tools(provider, make_fetcher(page_handler)))
    router = ToolRouter(registry, execution_timeout=2.0)

    class ScriptedLLM:
        def __init__(self):
            self.replies = [
                LLMReply(tool_calls=[LLMToolCall("search_web", {"query": "gemini live docs"}, "c1")]),
                LLMReply(text="The docs are at https://ai.google.dev/live [1]."),
            ]
            self.calls = []

        async def complete(self, user_text, history, tool_results, transcript):
            self.calls.append({"tool_results": tool_results})
            return self.replies.pop(0)

    llm = ScriptedLLM()
    rt = AgentRuntime(llm, router)
    resp = run(rt.handle("Search the web for gemini live docs"))

    assert resp.error is None
    assert resp.iterations == 2
    assert resp.tool_results[0].success is True
    assert resp.tool_results[0].risk == RiskLevel.READ
    seen = llm.calls[1]["tool_results"][0]["data"]["results"][0]["url"]
    assert seen == "https://ai.google.dev/live"  # round 2 saw the real search results
