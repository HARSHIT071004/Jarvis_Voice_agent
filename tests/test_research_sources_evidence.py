"""Phase 5 tests: search schema, sources, evidence (spec sections 7/12–15/29)."""

from __future__ import annotations

import asyncio

import pytest

from app.research.evidence import build_evidence, detect_conflicts
from app.research.models import SearchResult, Source
from app.research.providers import DuckDuckGoHTMLProvider, SearchProviderError, WebSearchProvider
from app.research.sources import classify_source, dedupe, rank, to_sources

SAMPLE_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="https://ai.google.dev/gemini/live">Gemini Live API Documentation</a>
  <a class="result__snippet">Build real-time multimodal voice apps with the Gemini Live API.</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fopenai.com%2Frealtime%2F">OpenAI Realtime API</a>
  <a class="result__snippet">Realtime speech-to-speech API.</a>
</div>
<div class="result">
  <a class="result__a" href="https://news.example.com/story">Some News Story</a>
</div>
</body></html>
"""


# --------------------------------------------------------------- search

def test_search_result_schema_from_provider():
    provider = DuckDuckGoHTMLProvider()
    results = provider._parse(SAMPLE_HTML, max_results=5)
    assert len(results) == 3
    first = results[0]
    assert first.title == "Gemini Live API Documentation"
    assert first.url == "https://ai.google.dev/gemini/live"
    assert first.source == "ai.google.dev"
    assert first.published_at is None  # DDG HTML has no dates — never fabricated


def test_search_redirect_urls_unwrapped():
    provider = DuckDuckGoHTMLProvider()
    results = provider._parse(SAMPLE_HTML, max_results=5)
    assert results[1].url == "https://openai.com/realtime/"


def test_search_missing_snippet_stays_none():
    provider = DuckDuckGoHTMLProvider()
    results = provider._parse(SAMPLE_HTML, max_results=5)
    assert results[2].snippet is None


def test_empty_query_raises_controlled():
    provider = DuckDuckGoHTMLProvider()
    with pytest.raises(SearchProviderError):
        asyncio.run(provider.search("   "))


class FailingProvider(WebSearchProvider):
    async def search(self, query: str, max_results: int = 8):
        raise SearchProviderError("SEARCH_TIMEOUT")


def test_search_provider_error_is_controlled():
    with pytest.raises(SearchProviderError):
        asyncio.run(FailingProvider().search("q"))


def test_no_results_returns_empty_list():
    provider = DuckDuckGoHTMLProvider()
    assert provider._parse("<html><body>no results</body></html>", 5) == []


# --------------------------------------------------------------- sources

def _results() -> list[SearchResult]:
    return [
        SearchResult(title="Forum take", url="https://reddit.com/r/x/comments/1", source="reddit.com", snippet="a"),
        SearchResult(title="Gemini docs", url="https://ai.google.dev/gemini/live", source="ai.google.dev", snippet="b"),
        SearchResult(title="Gemini docs again", url="https://ai.google.dev/gemini/live?utm_source=x", source="ai.google.dev"),
        SearchResult(title="Arxiv paper", url="https://arxiv.org/abs/2501.00001", source="arxiv.org", snippet="c"),
    ]


def test_source_deduplication():
    unique = dedupe(_results())
    assert len(unique) == 3  # utm variant removed


def test_source_ranking_prefers_primary_sources():
    ranked = rank(dedupe(_results()), max_sources=3)
    assert ranked[0].source == "ai.google.dev"  # documentation beats community
    assert ranked[-1].source == "reddit.com"  # community last


def test_ranking_respects_max_sources():
    assert len(rank(dedupe(_results()), max_sources=2)) == 2


def test_source_type_classification_is_evidence_based():
    assert classify_source("ai.google.dev") == "documentation"
    assert classify_source("arxiv.org") == "academic"
    assert classify_source("reuters.com") == "news"
    assert classify_source("stackoverflow.com") == "community"
    assert classify_source("whitehouse.gov") == "government"
    assert classify_source("random-blog.example") == "unknown"  # never blindly "official"


def test_to_sources_assigns_stable_ids():
    sources = to_sources(rank(dedupe(_results())), retrieved_at="2026-10-05T00:00:00Z")
    assert [s.source_id for s in sources] == ["source_1", "source_2", "source_3"]
    assert all(s.retrieved_at for s in sources)


# -------------------------------------------------------------- evidence

def _source(sid: str, content: str) -> Source:
    return Source(
        source_id=sid,
        title=f"Title {sid}",
        url=f"https://example.com/{sid}",
        domain="example.com",
        content=content,
    )


def test_evidence_source_mapping():
    sources = [
        _source("source_1", "The Gemini Live API supports function calling with tools declared at connect time."),
        _source("source_2", "Unrelated cooking content about pasta with garlic and olive oil recipes."),
    ]
    evidence = build_evidence("Does Gemini Live support function calling?", sources)
    assert evidence
    assert all(e.source_id == "source_1" for e in evidence)  # only relevant source maps
    assert all(0.0 < e.relevance <= 1.0 for e in evidence)
    assert evidence[0].url.endswith("/source_1")


def test_evidence_empty_when_no_overlap():
    sources = [_source("source_1", "Cooking pasta requires salted water and patience for al dente texture.")]
    assert build_evidence("quantum entanglement experiments", sources) == []


def test_unsupported_claim_detection_via_missing_evidence():
    # no evidence => researcher must declare insufficient (see researcher tests)
    assert build_evidence("anything specific", [_source("source_1", "tiny")]) == []


def test_conflict_detection_same_fact_different_values():
    evidence = build_evidence(
        "What is the price?",
        [
            _source("source_1", "The price of the Pro plan is $20 per month for teams using the API today."),
            _source("source_2", "The price of the Pro plan is $30 per month for teams using the API today."),
        ],
    )
    conflicts = detect_conflicts(evidence)
    assert conflicts, "expected a price conflict to be flagged"
    assert "$20" in conflicts[0] and "$30" in conflicts[0]


def test_no_conflict_for_agreeing_sources():
    evidence = build_evidence(
        "What is the price?",
        [
            _source("source_1", "The price of the Pro plan is $20 per month for teams using the API today."),
            _source("source_2", "The price of the Pro plan is $20 per month for teams using the API today."),
        ],
    )
    assert detect_conflicts(evidence) == []
