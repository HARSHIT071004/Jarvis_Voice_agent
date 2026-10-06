"""Search provider abstraction (spec section 8).

The research layer depends only on WebSearchProvider, so swapping or
adding providers never touches the researcher or tools. Providers must
return results exactly as observed — no fabricated URLs/titles/dates
(missing fields stay None).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.research.models import SearchResult

logger = logging.getLogger("jarvis.research")


class SearchProviderError(Exception):
    """Controlled search failure (timeout, HTTP error, parse failure)."""


class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        """Return structured results or raise SearchProviderError."""


class DuckDuckGoHTMLProvider(WebSearchProvider):
    """Keyless provider: parses DuckDuckGo's HTML endpoint.

    DDG result links are redirects (/l/?uddg=<url>) — unwrapped here so
    downstream code sees the real URL. Any field DDG does not provide
    (published_at) stays None.
    """

    ENDPOINT = "https://html.duckduckgo.com/html/"
    _TITLE_RE = re.compile(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
    _SNIPPET_RE = re.compile(r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', re.S)
    _TAG_RE = re.compile(r"<[^>]+>")

    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout

    async def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        query = (query or "").strip()
        if not query:
            raise SearchProviderError("EMPTY_QUERY")
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (Jarvis research)"},
            ) as client:
                response = await client.post(self.ENDPOINT, data={"q": query})
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise SearchProviderError("SEARCH_TIMEOUT") from exc
        except httpx.HTTPError as exc:
            raise SearchProviderError("SEARCH_HTTP_ERROR") from exc

        results = self._parse(response.text, max_results)
        logger.info("[RESEARCH] search_completed query=%r results=%d", query, len(results))
        return results

    def _parse(self, html: str, max_results: int) -> list[SearchResult]:
        titles = self._TITLE_RE.findall(html)
        snippets = self._SNIPPET_RE.findall(html)
        results: list[SearchResult] = []
        for index, (href, raw_title) in enumerate(titles[:max_results]):
            url = self._unwrap(href)
            if not url:
                continue
            title = self._TAG_RE.sub("", raw_title).strip()
            snippet = (
                self._TAG_RE.sub("", snippets[index]).strip()
                if index < len(snippets)
                else None
            )
            domain = urlparse(url).netloc.removeprefix("www.") or None
            results.append(
                SearchResult(
                    title=title or url,
                    url=url,
                    snippet=snippet or None,
                    source=domain,
                    published_at=None,  # DDG HTML gives no date — never invent one
                )
            )
        return results

    @staticmethod
    def _unwrap(href: str) -> str:
        if href.startswith("//"):
            href = "https:" + href
        if "duckduckgo.com/l/" in href:
            target = parse_qs(urlparse(href).query).get("uddg", [""])[0]
            return unquote(target)
        if href.startswith("http://") or href.startswith("https://"):
            return href
        return ""
