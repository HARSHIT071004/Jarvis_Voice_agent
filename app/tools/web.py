"""Web tools — the Phase 5 surface exposed to the agent (all READ).

Thin adapters over the research layer: search provider + safe fetcher +
extractor. A short-lived in-process PageCache keeps open→read→extract
from refetching the same URL within one session (temp state, not memory).
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from pydantic import Field

from app.research.extractor import extract, summarize
from app.research.fetcher import FetchError, PageFetcher
from app.research.models import ExtractedContent, RawPage
from app.research.providers import SearchProviderError, WebSearchProvider
from app.tools.base import RiskLevel, Tool, ToolArgs, ToolError

logger = logging.getLogger("jarvis.tool")

MAX_RESULTS_LIMIT = 10
OPEN_PREVIEW_CHARS = 4000
READ_CHARS = 8000


class SearchWebArgs(ToolArgs):
    query: str = Field(min_length=2, max_length=300)
    max_results: int = Field(default=6, ge=1, le=MAX_RESULTS_LIMIT)


class PageArgs(ToolArgs):
    url: str = Field(min_length=10, max_length=2048, pattern=r"^https?://")


class PageCache:
    """Bounded per-process cache of fetched pages (never persisted)."""

    def __init__(self, max_entries: int = 16) -> None:
        self._max = max_entries
        self._pages: dict[str, RawPage] = {}

    def get(self, url: str) -> RawPage | None:
        page = self._pages.get(url)
        if page is not None:
            # refresh recency
            self._pages[url] = self._pages.pop(url)
        return page

    def put(self, url: str, page: RawPage) -> None:
        self._pages.pop(url, None)
        self._pages[url] = page
        while len(self._pages) > self._max:
            self._pages.pop(next(iter(self._pages)))


def _domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


class SearchWebTool(Tool):
    name = "search_web"
    description = (
        "Search the live web and return real results (title, url, snippet). "
        "Use for current/latest information: prices, versions, releases, news, "
        "docs of fast-moving products, or when the user asks to search/research. "
        "Do NOT use for stable general knowledge."
    )
    risk = RiskLevel.READ
    args_model = SearchWebArgs

    def __init__(self, provider: WebSearchProvider) -> None:
        self._provider = provider

    async def execute(self, args: SearchWebArgs) -> dict:
        try:
            results = await self._provider.search(args.query, max_results=args.max_results)
        except SearchProviderError as exc:
            code = str(exc) or "SEARCH_FAILED"
            raise ToolError(code, "Web search is unavailable right now.") from exc
        return {
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "published_at": r.published_at,
                }
                for r in results
            ]
        }


class _PageTool(Tool):
    risk = RiskLevel.READ
    args_model = PageArgs

    def __init__(self, fetcher: PageFetcher, cache: PageCache) -> None:
        self._fetcher = fetcher
        self._cache = cache

    async def _page(self, url: str) -> RawPage:
        cached = self._cache.get(url)
        if cached is not None:
            return cached
        try:
            page = await self._fetcher.fetch(url)
        except FetchError as exc:
            raise ToolError(exc.code, f"Could not open that page: {exc.message}") from exc
        self._cache.put(url, page)
        return page


class OpenPageTool(_PageTool):
    name = "open_page"
    description = (
        "Fetch a web page by URL and return its metadata plus a bounded raw "
        "content preview (status, final url after redirects, title). Use after "
        "search_web to open a promising result."
    )

    async def execute(self, args: PageArgs) -> dict:
        page = await self._page(args.url)
        return {
            "url": page.url,
            "final_url": page.final_url,
            "status_code": page.status_code,
            "content_type": page.content_type,
            "title": page.title,
            "content": page.content[:OPEN_PREVIEW_CHARS],
            "truncated": page.truncated or len(page.content) > OPEN_PREVIEW_CHARS,
        }


class ReadPageTool(_PageTool):
    name = "read_page"
    description = (
        "Get the readable text of a web page (navigation/scripts/ads removed). "
        "Use to actually read a page's content before answering."
    )

    async def execute(self, args: PageArgs) -> dict:
        page = await self._page(args.url)
        content: ExtractedContent = extract(page.content, url=page.final_url)
        text = summarize(content, max_chars=READ_CHARS)
        if not text.strip():
            raise ToolError("EMPTY_PAGE", "That page has no readable text content.")
        return {
            "url": page.final_url,
            "title": content.title or page.title,
            "text": text,
        }


class ExtractContentTool(_PageTool):
    name = "extract_content"
    description = (
        "Extract structured content from a web page: title, headings and "
        "paragraphs (boilerplate removed). Use when you need the relevant "
        "sections of a page rather than raw HTML."
    )

    async def execute(self, args: PageArgs) -> dict:
        page = await self._page(args.url)
        content = extract(page.content, url=page.final_url)
        if not content.text.strip():
            raise ToolError("EMPTY_PAGE", "That page has no readable text content.")
        return {
            "url": page.final_url,
            "title": content.title or page.title,
            "headings": content.headings[:30],
            "paragraphs": [
                p for p in content.paragraphs if len(p) >= 60
            ][:20],
            "text_preview": summarize(content, max_chars=3000),
        }


def build_web_tools(
    provider: WebSearchProvider, fetcher: PageFetcher, cache: PageCache | None = None
) -> list[Tool]:
    """The Phase 5 web toolset (spec section 3), all READ risk."""
    cache = cache or PageCache()
    return [
        SearchWebTool(provider),
        OpenPageTool(fetcher, cache),
        ReadPageTool(fetcher, cache),
        ExtractContentTool(fetcher, cache),
    ]
