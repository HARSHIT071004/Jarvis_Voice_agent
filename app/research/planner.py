"""Research decision + planning (spec sections 5/6).

`decide()` asks the LLM (freshness, explicit search requests, current
events, pricing, versions...) and falls back to a heuristic when no LLM
is configured or it fails — never keyword-only when an LLM is available.

`plan()` turns the question into search queries and source preferences.
Both are single, cheap calls; the planner refuses to research when the
question is stable general knowledge (avoid wasted searches, §27).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from app.research.models import ResearchDecision, ResearchPlan

logger = logging.getLogger("jarvis.research")

DEFAULT_MAX_SOURCES = 6

DECISION_PROMPT = """\
You are a research planner for a voice assistant. Decide whether answering
the user's question requires fresh web research.

Return ONLY compact JSON: {{"requires_research": true|false, "reason": "..."}}.

Research IS required for: latest/current info, prices, versions, releases,
news, policies, docs of fast-moving products, explicit "search/research/
look up" requests, comparisons of current products, anything time-sensitive.

Research is NOT required for: stable general knowledge (what is Python,
explain binary search, definitions, math, how established concepts work).

Question: {question}
"""

PLAN_PROMPT = """\
You are a research planner. Given the user's question, produce a compact
JSON plan ONLY in this exact shape:
{{"queries": ["q1", "q2"], "max_sources": 6}}

Rules: 1-3 short, precise search queries (first = best). Prefer official
documentation queries for product/API questions. Question: {question}
"""

_JSON_RE = re.compile(r"\{.*\}", re.S)

# Heuristic signals (fallback only — LLM path is primary).
# "search" must look like a web-search request ("search for/the/web..."),
# not a noun like "binary search".
_EXPLICIT = re.compile(
    r"\b(research(?:\s+the)?|look\s?up|google|"
    r"search\s+(?:for|the|web|online|internet|up)|"
    r"find\s+(?:me\s+)?(?:the\s+)?latest|"
    r"check\s+(?:the\s+)?(?:latest|current))\b", re.I
)
_FRESHNESS = re.compile(
    r"\b(latest|current|today|now|right now|newest|recent|recently|this year|"
    r"\b20\d{2}\b|price|pricing|cost|release|released|version|update|updated|"
    r"news|changelog|roadmap|vs\.?|versus|compare|comparison|who won|stock|"
    r"weather|score|deadline)\b", re.I
)


class TextLLM(Protocol):
    async def complete(self, prompt: str) -> str: ...


def heuristic_decision(question: str) -> ResearchDecision:
    q = (question or "").strip()
    if not q:
        return ResearchDecision(requires_research=False, reason="Empty question.")
    if _EXPLICIT.search(q):
        return ResearchDecision(
            requires_research=True,
            reason="The user explicitly asked to search/research.",
        )
    if _FRESHNESS.search(q):
        return ResearchDecision(
            requires_research=True,
            reason="The question involves current/fast-changing information.",
        )
    return ResearchDecision(
        requires_research=False, reason="The question is stable general knowledge."
    )


def _parse_json(text: str) -> dict:
    match = _JSON_RE.search(text or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class ResearchPlanner:
    def __init__(self, llm: TextLLM | None = None, max_sources: int = DEFAULT_MAX_SOURCES) -> None:
        self.llm = llm
        self.max_sources = max_sources

    async def decide(self, question: str) -> ResearchDecision:
        if self.llm is not None:
            try:
                raw = await self.llm.complete(DECISION_PROMPT.format(question=question))
                data = _parse_json(raw)
                if "requires_research" in data:
                    decision = ResearchDecision(
                        requires_research=bool(data["requires_research"]),
                        reason=str(data.get("reason") or "decided by planner")[:300],
                    )
                    logger.info(
                        "[RESEARCH] research_decision=%s reason=%s",
                        decision.requires_research,
                        decision.reason,
                    )
                    return decision
            except Exception:
                logger.warning("[RESEARCH] planner LLM failed, using heuristic")
        decision = heuristic_decision(question)
        logger.info(
            "[RESEARCH] research_decision=%s (heuristic) reason=%s",
            decision.requires_research,
            decision.reason,
        )
        return decision

    async def plan(self, question: str) -> ResearchPlan:
        queries = [question.strip()]
        source_preferences = ["official documentation", "primary sources"]
        if self.llm is not None:
            try:
                raw = await self.llm.complete(PLAN_PROMPT.format(question=question))
                data = _parse_json(raw)
                llm_queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
                if llm_queries:
                    queries = llm_queries[:3]
                if isinstance(data.get("max_sources"), int):
                    self.max_sources = max(1, min(20, data["max_sources"]))
            except Exception:
                logger.warning("[RESEARCH] plan LLM failed, using defaults")
        else:
            queries.append(f"{question.strip()} official documentation")
        # dedupe, keep order, cap at 3 (§27 efficiency)
        queries = list(dict.fromkeys(q for q in queries if q))[:3]
        plan = ResearchPlan(
            research_required=True,
            queries=queries,
            source_preferences=source_preferences,
            max_sources=self.max_sources,
        )
        logger.info("[RESEARCH] research_query plan=%s", plan.queries)
        return plan
