"""High-level memory operations (the only API the rest of Jarvis uses).

Conflict policy (documented, spec section 19):
- Contacts are identified by case-insensitive exact name match.
- A new non-null company/role overwrites the old value (most recent
  statement wins) AND the superseded value is appended to `notes`
  with a date stamp — history is never silently destroyed.
- Re-stating identical facts is a no-op (no duplicate rows).
- Tasks dedupe by title (pending), memories dedupe by exact content.
- Forgetting only deletes memories whose content matches the query by
  majority of keywords — contacts need an explicit id-based delete.

All decisions are logged as [MEMORY] lines for auditability.
"""

from __future__ import annotations

import logging
import re

from app.memory.database import Database, utcnow
from app.memory.migrations import migrate
from app.memory.models import Contact, ConversationRecord, MemoryItem, Task
from app.memory.policy import MemoryDecision, MemoryPolicy, looks_like_secret
from app.memory.repository import (
    ContactRepository,
    ConversationRepository,
    MemoryRepository,
    TaskRepository,
)

logger = logging.getLogger("jarvis.memory")

_STOPWORDS = {
    "the", "that", "this", "fact", "about", "please", "and", "for",
    "with", "from", "was", "are", "has", "have", "not", "dont",
}


def _keywords(text: str) -> list[str]:
    return [
        w
        for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) >= 3 and w not in _STOPWORDS
    ]


class MemoryManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.contacts = ContactRepository(db)
        self.conversations = ConversationRepository(db)
        self.tasks = TaskRepository(db)
        self.memories = MemoryRepository(db)
        self.policy = MemoryPolicy()

    @classmethod
    def open(cls, path) -> MemoryManager:
        db = Database(path)
        migrate(db)
        logger.info("[MEMORY] Database ready at %s", path)
        return cls(db)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------- saves

    def save_contact(
        self,
        name: str,
        company: str | None = None,
        role: str | None = None,
        notes: str | None = None,
        source: str | None = None,
    ) -> Contact:
        existing = self.contacts.find_by_name(name)
        if not existing:
            logger.info("[MEMORY] Contact: %s (new)", name)
            return self.contacts.add(
                Contact(name=name, company=company, role=role, notes=notes)
            )

        current = existing[0]
        changes: list[str] = []
        stamp = utcnow()[:10]
        if company and current.company and company != current.company:
            changes.append(f"company: '{current.company}' -> '{company}'")
        if role and current.role and role != current.role:
            changes.append(f"role: '{current.role}' -> '{role}'")

        if not changes and not (company and not current.company) and not (
            role and not current.role
        ):
            logger.debug("[MEMORY] Contact %s unchanged, skipping", name)
            return current  # identical re-statement: no duplicate, no churn

        merged_notes = current.notes or ""
        if changes:
            addition = "; ".join(f"[{stamp}] {c}" for c in changes)
            merged_notes = f"{merged_notes}\n{addition}" if merged_notes else addition
            logger.info("[MEMORY] Contact %s updated: %s", name, addition)
        elif notes:
            merged_notes = f"{merged_notes}\n{notes}" if merged_notes else notes

        return self.contacts.update(
            current.model_copy(
                update={
                    "company": company or current.company,
                    "role": role or current.role,
                    "notes": merged_notes or None,
                }
            )
        )

    def save_conversation(
        self,
        purpose: str | None = None,
        requirement: str | None = None,
        urgency: str | None = None,
        deadline: str | None = None,
        summary: str | None = None,
        person_id: int | None = None,
    ) -> ConversationRecord:
        record = ConversationRecord(
            person_id=person_id,
            purpose=purpose,
            requirement=requirement,
            urgency=urgency,
            deadline=deadline,
            summary=summary,
        )
        return self.conversations.add(record)

    def save_task(
        self,
        title: str,
        deadline: str | None = None,
        description: str | None = None,
        person_id: int | None = None,
    ) -> Task:
        existing = self.tasks.find_by_title(title)
        if existing is not None and existing.status == "pending":
            logger.info("[MEMORY] Task already pending, skipping: %s", title)
            return existing
        return self.tasks.add(
            Task.new(
                title=title,
                deadline=deadline,
                description=description,
                person_id=person_id,
            )
        )

    def save_memory(
        self,
        content: str,
        importance: int = 3,
        category: str = "general",
        source: str | None = None,
    ) -> MemoryItem | None:
        if looks_like_secret(content):
            logger.warning("[MEMORY] Policy decision: REJECT (secret)")
            return None
        existing = self.memories.find_by_content(content)
        if existing is not None:
            logger.info("[MEMORY] Memory already stored, skipping")
            return existing
        return self.memories.add(
            MemoryItem(
                content=content, importance=importance, category=category, source=source
            )
        )

    # -------------------------------------------------------------- gets

    def get_contact(self, name: str) -> Contact | None:
        found = self.contacts.find_by_name(name)
        return found[0] if found else None

    def search_contacts(self, query: str) -> list[Contact]:
        logger.info("[MEMORY] Retrieval query: %s", query)
        return self.contacts.search(query)

    def get_tasks(self, status: str | None = "pending") -> list[Task]:
        return self.tasks.list(status=status)

    def search_memory(self, query: str) -> list[MemoryItem]:
        logger.info("[MEMORY] Retrieval query: %s", query)
        return self.memories.search(query)

    def get_conversation(self, record_id: int) -> ConversationRecord | None:
        return self.conversations.get(record_id)

    def delete_memory(self, memory_id: int) -> bool:
        return self.memories.delete(memory_id)

    # ------------------------------------------------------------ ingest

    def ingest(self, state, turn_text: str) -> list[MemoryDecision]:
        """Phase 2 -> Phase 3 pipeline: policy decides, manager stores."""
        directive, decisions = self.policy.decide(state, turn_text)

        if directive.kind == "forget":
            deleted = self.forget(directive.content or "")
            logger.info(
                "[MEMORY] Forget directive: deleted %d memory(ies)", len(deleted)
            )
            return decisions

        for decision in decisions:
            logger.info(
                "[MEMORY] Policy decision: %s (%s) - %s",
                decision.action.upper(),
                decision.kind,
                decision.reason,
            )
            if decision.action == "reject":
                logger.warning("[MEMORY] Rejected %s: secret-like content", decision.kind)
                continue
            if not decision.stores:
                continue
            self._apply(decision, state)
        return decisions

    def _apply(self, decision: MemoryDecision, state) -> None:
        if decision.kind == "contact":
            payload = decision.payload or {}
            self.save_contact(
                name=payload["name"],
                company=payload.get("company"),
                role=payload.get("role"),
                source="extraction",
            )
        elif decision.kind == "conversation":
            payload = decision.payload or {}
            person_id = None
            name = getattr(getattr(state, "person", None), "name", None)
            if name:
                contact = self.get_contact(name)
                person_id = contact.id if contact else None
            self.save_conversation(
                purpose=payload.get("purpose")
                or getattr(getattr(state, "conversation", None), "purpose", None),
                requirement=payload.get("requirement")
                or getattr(getattr(state, "conversation", None), "requirement", None),
                urgency=payload.get("urgency")
                or getattr(getattr(state, "conversation", None), "urgency", None),
                deadline=payload.get("deadline")
                or getattr(getattr(state, "conversation", None), "deadline", None),
                summary=payload.get("summary"),
                person_id=person_id,
            )
        elif decision.kind == "task":
            payload = decision.payload or {}
            person_id = None
            name = getattr(getattr(state, "person", None), "name", None)
            if name:
                contact = self.get_contact(name)
                person_id = contact.id if contact else None
            self.save_task(
                title=payload["title"],
                deadline=payload.get("deadline"),
                person_id=person_id,
            )
        elif decision.kind == "memory":
            payload = decision.payload or {}
            content = payload.get("content", "")
            if content:
                self.save_memory(
                    content=content,
                    importance=int(payload.get("importance", 4)),
                    source="explicit",
                )

    def forget(self, query: str) -> list[int]:
        """Delete memories matching the query by keyword majority.

        Conservative: only the memories table can be affected by a
        spoken forget command; contacts require delete_memory-style
        explicit id operations (spec section 23).
        """
        words = _keywords(query)
        if not words:
            return []
        candidates: dict[int, MemoryItem] = {}
        for word in words:
            for item in self.memories.search(word):
                candidates[item.id] = item
        deleted: list[int] = []
        for item in candidates.values():
            content_words = set(_keywords(item.content))
            matched = sum(1 for w in words if w in content_words or w in item.content.lower())
            if matched / len(words) >= 0.6:
                if self.memories.delete(item.id):
                    deleted.append(item.id)
        logger.info("[MEMORY] Retrieval query: %s -> deleted %s", query, deleted)
        return deleted
