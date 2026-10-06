"""Source management: dedupe, ranking, cautious type classification.

Preference order (spec section 14): official docs > official org >
academic > reputable technical > secondary > community. Classification
is evidence-based on the domain — never blindly "official".
"""

from __future__ import annotations

from urllib.parse import urlparse

from app.research.models import SearchResult, Source, SourceType

_MAX_TITLE = 200

_DOC_HINTS = ("docs.", "developer.", "documentation", "readthedocs", "api.", "google.dev", "platform.")
_GOV_TLDS = (".gov", ".gov.uk", ".gc.ca", ".gob.", ".go.jp")
_ACADEMIC_HINTS = ("arxiv.org", "acm.org", "ieee.org", ".edu", ".ac.")
_NEWS_HINTS = (
    "reuters.com", "apnews.com", "bbc.", "theguardian.com", "bloomberg.com",
    "nytimes.com", "wsj.com", "theverge.com", "techcrunch.com", "arstechnica.com",
)
_COMMUNITY_HINTS = ("reddit.com", "stackoverflow.com", "stackexchange.com", "news.ycombinator.com", "forum.", "discuss.")
_OFFICIAL_DOMAINS = ("ai.google.dev", "openai.com", "platform.openai.com", "developers.google.com", "github.com")


def classify_source(domain: str, url: str = "") -> str:
    """Best-effort type from domain evidence only (unknown when unsure)."""
    domain = (domain or "").lower().removeprefix("www.")
    if not domain:
        return SourceType.UNKNOWN
    if any(hint in domain for hint in _DOC_HINTS):
        return SourceType.DOCUMENTATION
    if domain.endswith(_GOV_TLDS) or domain.endswith(".mil"):
        return SourceType.GOVERNMENT
    if any(hint in domain for hint in _ACADEMIC_HINTS):
        return SourceType.ACADEMIC
    if any(domain == d or domain.endswith("." + d) for d in _OFFICIAL_DOMAINS):
        return SourceType.COMPANY
    if any(hint in domain for hint in _NEWS_HINTS):
        return SourceType.NEWS
    if any(hint in domain for hint in _COMMUNITY_HINTS):
        return SourceType.COMMUNITY
    return SourceType.UNKNOWN


_TYPE_RANK = {
    SourceType.DOCUMENTATION: 100,
    SourceType.OFFICIAL: 95,
    SourceType.GOVERNMENT: 90,
    SourceType.ACADEMIC: 85,
    SourceType.COMPANY: 75,
    SourceType.NEWS: 60,
    SourceType.UNKNOWN: 40,
    SourceType.COMMUNITY: 30,
}


def normalize_url(url: str) -> str:
    """Dedupe key: lowercase host, drop fragment, trailing slash, tracking params."""
    parsed = urlparse((url or "").strip())
    query = "&".join(
        p for p in (parsed.query or "").split("&")
        if p and not p.lower().startswith(("utm_", "fbclid=", "gclid="))
    )
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}" + (f"?{query}" if query else "")


def dedupe(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate URLs, keeping first occurrence (provider order)."""
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for result in results:
        key = normalize_url(result.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def rank(results: list[SearchResult], max_sources: int = 6) -> list[SearchResult]:
    """Rank: source-type quality + freshness + snippet presence + original order."""
    scored: list[tuple[float, int, SearchResult]] = []
    for index, result in enumerate(results):
        domain = result.source or urlparse(result.url).netloc
        score = float(_TYPE_RANK.get(classify_source(domain, result.url), 40))
        if result.published_at:
            score += 25.0  # freshness signal when the provider supplies a date
        if result.snippet:
            score += 5.0
        # primary-source-looking domains for common AI/API topics
        if any(h in domain for h in _DOC_HINTS):
            score += 15.0
        scored.append((-score, index, result))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored[:max_sources]]


def to_sources(results: list[SearchResult], retrieved_at: str) -> list[Source]:
    """Convert ranked results into Source records with stable ids."""
    sources: list[Source] = []
    for number, result in enumerate(results, start=1):
        domain = result.source or urlparse(result.url).netloc.removeprefix("www.")
        sources.append(
            Source(
                source_id=f"source_{number}",
                title=(result.title or domain or result.url)[:_MAX_TITLE],
                url=result.url,
                domain=domain,
                source_type=classify_source(domain, result.url),
                published_at=result.published_at,
                retrieved_at=retrieved_at,
                snippet=result.snippet,
            )
        )
    return sources
