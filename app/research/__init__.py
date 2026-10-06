"""Web research capability (Phase 5).

Public API: build_researcher(settings) -> Researcher, or assemble the
pieces directly (providers/fetcher/planner are independent).
"""

from __future__ import annotations

import logging

from app.research.evidence import build_evidence, detect_conflicts
from app.research.extractor import extract, summarize
from app.research.fetcher import FetchError, PageFetcher
from app.research.models import (
    Evidence,
    ExtractedContent,
    RawPage,
    ResearchAnswer,
    ResearchDecision,
    ResearchPlan,
    ResearchSession,
    SearchResult,
    Source,
)
from app.research.planner import ResearchPlanner, TextLLM, heuristic_decision
from app.research.providers import (
    DuckDuckGoHTMLProvider,
    SearchProviderError,
    WebSearchProvider,
)
from app.research.researcher import Researcher, sanitize_citations
from app.research.sources import classify_source, dedupe, rank, to_sources

logger = logging.getLogger("jarvis.research")

__all__ = [
    "DuckDuckGoHTMLProvider",
    "Evidence",
    "ExtractedContent",
    "FetchError",
    "PageFetcher",
    "RawPage",
    "ResearchAnswer",
    "ResearchDecision",
    "ResearchPlanner",
    "ResearchPlan",
    "ResearchSession",
    "Researcher",
    "SearchProviderError",
    "SearchResult",
    "Source",
    "TextLLM",
    "WebSearchProvider",
    "build_evidence",
    "build_researcher",
    "classify_source",
    "dedupe",
    "detect_conflicts",
    "extract",
    "heuristic_decision",
    "rank",
    "sanitize_citations",
    "summarize",
    "to_sources",
]


def build_researcher(settings) -> Researcher | None:
    """Wire a Researcher from settings. None when research is disabled
    or no API key is available for the synthesis/planner LLM.

    Search itself is keyless (DuckDuckGo HTML), so even without an LLM
    the fetch/extract tools still work for the agent tool loop.
    """
    if not getattr(settings, "research_enabled", True):
        return None

    provider_name = (getattr(settings, "web_search_provider", "duckduckgo_html") or "").lower()
    if provider_name not in {"duckduckgo_html", "duckduckgo"}:
        logger.warning(
            "[RESEARCH] Unknown provider %r, falling back to duckduckgo_html", provider_name
        )
    search: WebSearchProvider = DuckDuckGoHTMLProvider(
        timeout=getattr(settings, "web_search_timeout", 8.0)
    )
    fetcher = PageFetcher(
        timeout=getattr(settings, "request_timeout", 10.0),
        max_size=getattr(settings, "max_page_size", 2 * 1024 * 1024),
    )

    llm: TextLLM | None = None
    if settings.gemini_api_key:
        from app.intelligence.extractor import GeminiTextLLM

        llm = GeminiTextLLM(
            api_key=settings.gemini_api_key,
            model=getattr(settings, "intelligence_model", "gemini-3.8-flash"),
        )

    planner = ResearchPlanner(llm=llm, max_sources=getattr(settings, "max_research_sources", 6))
    researcher = Researcher(
        search=search,
        fetcher=fetcher,
        planner=planner,
        synthesize_llm=llm,
        max_sources=getattr(settings, "max_research_sources", 6),
        max_open=getattr(settings, "max_research_open", 6),
    )
    logger.info(
        "[RESEARCH] Researcher ready provider=%s llm=%s",
        provider_name,
        "yes" if llm else "no",
    )
    return researcher
