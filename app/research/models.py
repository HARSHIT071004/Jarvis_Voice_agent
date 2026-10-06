"""Research domain models (Phase 5).

All models are Pydantic with strict typing so search/fetch/extract
pipelines can never silently pass malformed data. Fields the provider
cannot know are explicitly Optional and stay None — nothing is ever
fabricated (spec section 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """One search hit exactly as the provider returned it."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str | None = None
    source: str | None = None  # domain
    published_at: str | None = None


class RawPage(BaseModel):
    """Result of open_page(): fetch metadata + raw (possibly truncated) body."""

    model_config = ConfigDict(extra="forbid")

    url: str
    final_url: str
    status_code: int
    content_type: str | None = None
    title: str | None = None
    content: str = ""  # raw text/html, truncated to MAX_PAGE_SIZE
    truncated: bool = False


class ExtractedContent(BaseModel):
    """Readable content extracted from a page (clean text, no scripts/nav)."""

    model_config = ConfigDict(extra="forbid")

    url: str
    title: str | None = None
    text: str = ""
    headings: list[str] = Field(default_factory=list)
    paragraphs: list[str] = Field(default_factory=list)


class SourceType:
    """Possible source types (spec section 13). Never assigned blindly."""

    OFFICIAL = "official"
    DOCUMENTATION = "documentation"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    NEWS = "news"
    COMPANY = "company"
    COMMUNITY = "community"
    UNKNOWN = "unknown"


class Source(BaseModel):
    """A retrieved source with provenance (spec section 13)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str  # "source_1", ...
    title: str
    url: str
    domain: str
    source_type: str = SourceType.UNKNOWN
    published_at: str | None = None
    retrieved_at: str | None = None  # UTC ISO-8601
    snippet: str | None = None
    content: str | None = None  # extracted text (bounded)


class Evidence(BaseModel):
    """A passage supporting an answer, traceable to one source (§12)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str
    url: str
    claim: str  # what this passage supports (query aspect)
    supporting_text: str  # actual retrieved text
    relevance: float = Field(ge=0.0, le=1.0)


class ResearchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requires_research: bool
    reason: str


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_required: bool = True
    queries: list[str] = Field(default_factory=list)
    source_preferences: list[str] = Field(default_factory=list)
    max_sources: int = Field(default=6, ge=1, le=20)


class ResearchAnswer(BaseModel):
    """Final synthesized answer + the sources actually retrieved."""

    model_config = ConfigDict(extra="forbid")

    text: str
    sources: list[Source] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)  # source_ids cited in text
    conflicts: list[str] = Field(default_factory=list)  # surfaced, never hidden (§18)
    insufficient_evidence: bool = False


@dataclass
class ResearchSession:
    """Temporary state for ONE research operation (spec section 24).

    Never persisted to Jarvis memory — web content is transient by design.
    """

    user_question: str
    research_required: bool = False
    plan: ResearchPlan | None = None
    search_results: list[SearchResult] = field(default_factory=list)
    selected_sources: list[Source] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    final_answer: ResearchAnswer | None = None
