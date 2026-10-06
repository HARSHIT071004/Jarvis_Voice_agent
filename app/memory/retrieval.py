"""Memory retrieval: deterministic keyword search + ranking + context brief.

Ranking (spec section 17, kept simple and explainable):
  exact match > keyword coverage > recency > importance > pending-task bonus
No embeddings, no FTS5 yet — the seam for future semantic search is
`search()` returning ranked `Retrieved` rows (swap scorer, keep API).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from app.memory.manager import MemoryManager, _keywords

logger = logging.getLogger("jarvis.memory")

_TS = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class Retrieved:
    kind: str  # contact | task | memory | conversation
    text: str
    score: float
    detail: str | None = None


def _age_days(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        when = datetime.strptime(stamp, _TS).replace(tzinfo=UTC)
    except ValueError:
        return None
    return max(0.0, (datetime.now(UTC) - when).total_seconds() / 86400.0)


def _recency_bonus(stamp: str | None) -> float:
    age = _age_days(stamp)
    if age is None:
        return 0.0
    if age <= 7:
        return 5.0
    if age <= 30:
        return 3.0
    return 1.0


def _score(query: str, haystack: str, stamp: str | None, extra: float = 0.0) -> float | None:
    """None when the item does not match the query at all."""
    query_l = (query or "").strip().lower()
    hay_l = (haystack or "").lower()
    if not query_l:
        return None
    if hay_l == query_l:
        return 100.0
    words = _keywords(query)
    if not words:
        return None
    matched = [w for w in words if w in hay_l]
    if not matched:
        return None
    score = 30.0 * (len(matched) / len(words))
    if re.search(rf"\b{re.escape(words[0])}\b", hay_l):
        score += 10.0
    return score + _recency_bonus(stamp) + extra


class MemoryRetrieval:
    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    def search(self, query: str, limit: int = 8) -> list[Retrieved]:
        logger.info("[MEMORY] Retrieval query: %s", query)
        found: list[Retrieved] = []

        for c in self._m.contacts.all():
            hay = " ".join(
                x for x in (c.name, c.company, c.role, c.notes) if x
            )
            score = _score(query, hay, c.updated_at)
            if score is not None:
                found.append(
                    Retrieved(
                        "contact",
                        c.name,
                        score + (5.0 if (c.company or c.role) else 0.0),
                        detail=", ".join(x for x in (c.company, c.role) if x) or None,
                    )
                )

        for t in self._m.tasks.list(status=None, limit=100):
            score = _score(
                query,
                " ".join(x for x in (t.title, t.description) if x),
                t.updated_at,
                extra=2.0 if t.status == "pending" else 0.0,
            )
            if score is not None:
                found.append(
                    Retrieved(
                        "task",
                        t.title,
                        score,
                        detail=f"{t.status}"
                        + (f", due {t.deadline}" if t.deadline else ""),
                    )
                )

        for m in self._m.memories.recent(limit=200):
            score = _score(query, m.content, m.updated_at, extra=float(m.importance))
            if score is not None:
                found.append(Retrieved("memory", m.content, score))

        for r in self._m.conversations.recent(limit=50):
            hay = " ".join(
                x for x in (r.purpose, r.requirement, r.summary) if x
            )
            score = _score(query, hay, r.created_at)
            if score is not None:
                found.append(
                    Retrieved("conversation", r.requirement or r.summary or hay, score)
                )

        found.sort(key=lambda r: (-r.score, r.text))
        results = found[:limit]
        logger.info("[MEMORY] Results: %d", len(results))
        return results

    def brief(
        self,
        max_chars: int = 1500,
        max_contacts: int = 20,
        max_tasks: int = 15,
        max_notes: int = 15,
    ) -> str:
        """Bounded digest of stored memory for the model's context.

        Deliberately capped (spec rule 10: never dump the database).
        """
        parts: list[str] = []

        contacts = self._m.contacts.all()[:max_contacts]
        if contacts:
            lines = [
                "- "
                + c.name
                + "".join(f" — {x}" for x in (c.company, c.role) if x)
                for c in contacts
            ]
            parts.append("Known contacts:\n" + "\n".join(lines))

        tasks = self._m.tasks.list(status="pending", limit=max_tasks)
        if tasks:
            lines = [
                "- "
                + t.title
                + (f" (deadline: {t.deadline})" if t.deadline else "")
                for t in tasks
            ]
            parts.append("Pending tasks:\n" + "\n".join(lines))

        notes = self._m.memories.recent(limit=max_notes)
        if notes:
            lines = [f"- {m.content}" for m in notes]
            parts.append("Remembered facts:\n" + "\n".join(lines))

        if not parts:
            return ""
        brief = "\n\n".join(parts)
        if len(brief) > max_chars:
            brief = brief[: max_chars - 3].rstrip() + "..."
        return brief
