"""Evidence building: passage selection, source mapping, conflict signals.

Deterministic and explainable — no embeddings (the seam for future
semantic search is `build_evidence`, keep its signature). Relevance is
query-term overlap with a phrase bonus, normalized to [0, 1].
"""

from __future__ import annotations

import logging
import re

from app.research.models import Evidence, Source

logger = logging.getLogger("jarvis.research")

MIN_RELEVANCE = 0.12
PASSAGE_SENTENCES = 3
MAX_EVIDENCE = 12
_PER_SOURCE = 4

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "what", "which", "how", "does", "do", "does",
    "about", "please", "tell", "me", "with", "from", "that", "this",
}
_FACT_RE = re.compile(
    r"(?P<ctx>(?:[a-z][a-z0-9]*\s+){0,3})(?P<val>\$?\d+(?:\.\d+)?\s*(?:%|percent|usd|dollars?|rs|inr|kB|MB|GB|kbps|mbps|v\d+(?:\.\d+)*|\d{4})?)",
    re.I,
)


def _terms(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall((text or "").lower()) if len(w) >= 3 and w not in _STOP]


def _score_passage(query: str, passage: str) -> float:
    """0..1: coverage of query terms + full-phrase bonus."""
    terms = _terms(query)
    if not terms:
        return 0.0
    passage_l = passage.lower()
    # word-start match: "live" must not hit "olive" (substring false positives)
    hits = sum(
        1
        for t in terms
        if re.search(r"(?:^|[^a-z0-9])" + re.escape(t), passage_l)
    )
    score = hits / len(terms)
    if (query.lower().strip() and query.lower().strip() in passage_l) or all(
        t in passage_l for t in terms
    ):
        score = min(1.0, score + 0.25)
    return min(1.0, score)


def _split_passages(text: str) -> list[str]:
    """Paragraphs first; long paragraphs split into sentence windows."""
    paragraphs = [p.strip() for p in (text or "").split("\n") if len(p.strip()) >= 40]
    passages: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= 700:
            passages.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for start in range(0, len(sentences), PASSAGE_SENTENCES):
            chunk = " ".join(sentences[start : start + PASSAGE_SENTENCES]).strip()
            if len(chunk) >= 40:
                passages.append(chunk)
    return passages


def build_evidence(
    question: str, sources: list[Source], max_evidence: int = MAX_EVIDENCE
) -> list[Evidence]:
    """Select the passages that best support the question, per source."""
    evidence: list[Evidence] = []
    for source in sources:
        if not source.content:
            continue
        scored: list[tuple[float, str]] = []
        for passage in _split_passages(source.content):
            relevance = _score_passage(question, passage)
            if relevance >= MIN_RELEVANCE:
                scored.append((relevance, passage))
        scored.sort(key=lambda item: -item[0])
        for relevance, passage in scored[:_PER_SOURCE]:
            evidence.append(
                Evidence(
                    source_id=source.source_id,
                    title=source.title,
                    url=source.url,
                    claim=question,
                    supporting_text=passage[:1200],
                    relevance=round(relevance, 3),
                )
            )
    evidence.sort(key=lambda e: -e.relevance)
    return evidence[:max_evidence]


def detect_conflicts(evidence: list[Evidence]) -> list[str]:
    """Flag contradictions: same fact-context, different value, two sources.

    Best-effort heuristic (price/version/date style facts). Conflicts are
    surfaced to the synthesizer — never silently resolved (spec §18).
    """
    seen: dict[str, tuple[str, str]] = {}  # context skeleton -> (value, source_id)
    conflicts: list[str] = []
    for item in evidence:
        for match in _FACT_RE.finditer(item.supporting_text):
            context = re.sub(r"\s+", " ", match.group("ctx").lower()).strip()
            value = re.sub(r"\s+", " ", match.group("val").lower()).strip()
            if len(context) < 4 or not value:
                continue
            key = context
            if key in seen:
                prev_value, prev_source = seen[key]
                if prev_value != value and prev_source != item.source_id:
                    conflicts.append(
                        f"'{context.strip()}' is reported as {prev_value} by "
                        f"{prev_source} but {value} by {item.source_id}"
                    )
            else:
                seen[key] = (value, item.source_id)
    unique = list(dict.fromkeys(conflicts))
    if unique:
        logger.info("[RESEARCH] conflicts_detected count=%d", len(unique))
    return unique
