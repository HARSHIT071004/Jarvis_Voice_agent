"""Repository layer: all SQL lives here.

The rest of the application talks to these classes (via MemoryManager)
and never writes SQL itself — this is the seam where PostgreSQL can
replace SQLite later without touching policy/manager/retrieval.
"""

from __future__ import annotations

import logging

from app.memory.database import Database, utcnow
from app.memory.models import Contact, ConversationRecord, MemoryItem, Task

logger = logging.getLogger("jarvis.memory")


def _row_to_contact(row) -> Contact:
    return Contact(**dict(row))


class ContactRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, contact: Contact) -> Contact:
        now = utcnow()
        cur = self._db.conn.execute(
            """
            INSERT INTO contacts (name, company, role, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (contact.name, contact.company, contact.role, contact.notes, now, now),
        )
        self._db.conn.commit()
        logger.info("[MEMORY] Database operation: INSERT contact %s", contact.name)
        return contact.model_copy(update={"id": cur.lastrowid, "created_at": now, "updated_at": now})

    def update(self, contact: Contact) -> Contact:
        assert contact.id is not None
        now = utcnow()
        self._db.conn.execute(
            """
            UPDATE contacts
               SET name = ?, company = ?, role = ?, notes = ?, updated_at = ?
             WHERE id = ?
            """,
            (contact.name, contact.company, contact.role, contact.notes, now, contact.id),
        )
        self._db.conn.commit()
        logger.info("[MEMORY] Database operation: UPDATE contact #%s", contact.id)
        return contact.model_copy(update={"updated_at": now})

    def find_by_name(self, name: str) -> list[Contact]:
        rows = self._db.conn.execute(
            "SELECT * FROM contacts WHERE name = ? COLLATE NOCASE ORDER BY id",
            (name,),
        ).fetchall()
        return [_row_to_contact(r) for r in rows]

    def get(self, contact_id: int) -> Contact | None:
        row = self._db.conn.execute(
            "SELECT * FROM contacts WHERE id = ?", (contact_id,)
        ).fetchone()
        return _row_to_contact(row) if row else None

    def search(self, query: str) -> list[Contact]:
        pattern = f"%{query}%"
        rows = self._db.conn.execute(
            """
            SELECT * FROM contacts
             WHERE name     LIKE ? COLLATE NOCASE
                OR company  LIKE ? COLLATE NOCASE
                OR role     LIKE ? COLLATE NOCASE
                OR notes    LIKE ? COLLATE NOCASE
             ORDER BY updated_at DESC
            """,
            (pattern, pattern, pattern, pattern),
        ).fetchall()
        return [_row_to_contact(r) for r in rows]

    def all(self) -> list[Contact]:
        rows = self._db.conn.execute(
            "SELECT * FROM contacts ORDER BY updated_at DESC"
        ).fetchall()
        return [_row_to_contact(r) for r in rows]

    def delete(self, contact_id: int) -> bool:
        cur = self._db.conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        self._db.conn.commit()
        return cur.rowcount > 0


class ConversationRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, record: ConversationRecord) -> ConversationRecord:
        now = utcnow()
        cur = self._db.conn.execute(
            """
            INSERT INTO conversations
                (person_id, purpose, requirement, urgency, deadline, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.person_id,
                record.purpose,
                record.requirement,
                record.urgency,
                record.deadline,
                record.summary,
                now,
            ),
        )
        self._db.conn.commit()
        logger.info("[MEMORY] Database operation: INSERT conversation")
        return record.model_copy(update={"id": cur.lastrowid, "created_at": now})

    def get(self, record_id: int) -> ConversationRecord | None:
        row = self._db.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (record_id,)
        ).fetchone()
        return ConversationRecord(**dict(row)) if row else None

    def search(self, query: str) -> list[ConversationRecord]:
        pattern = f"%{query}%"
        rows = self._db.conn.execute(
            """
            SELECT * FROM conversations
             WHERE purpose     LIKE ? COLLATE NOCASE
                OR requirement LIKE ? COLLATE NOCASE
                OR summary     LIKE ? COLLATE NOCASE
             ORDER BY created_at DESC
            """,
            (pattern, pattern, pattern),
        ).fetchall()
        return [ConversationRecord(**dict(r)) for r in rows]

    def recent(self, limit: int = 10) -> list[ConversationRecord]:
        rows = self._db.conn.execute(
            "SELECT * FROM conversations ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [ConversationRecord(**dict(r)) for r in rows]


class TaskRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, task: Task) -> Task:
        now = utcnow()
        cur = self._db.conn.execute(
            """
            INSERT INTO tasks (title, description, deadline, status, person_id,
                               created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.title,
                task.description,
                task.deadline,
                task.status,
                task.person_id,
                now,
                now,
            ),
        )
        self._db.conn.commit()
        logger.info("[MEMORY] Database operation: INSERT task %s", task.title)
        return task.model_copy(update={"id": cur.lastrowid, "created_at": now, "updated_at": now})

    def update(self, task: Task) -> Task:
        assert task.id is not None
        now = utcnow()
        self._db.conn.execute(
            """
            UPDATE tasks
               SET title = ?, description = ?, deadline = ?, status = ?,
                   person_id = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                task.title,
                task.description,
                task.deadline,
                task.status,
                task.person_id,
                now,
                task.id,
            ),
        )
        self._db.conn.commit()
        return task.model_copy(update={"updated_at": now})

    def get(self, task_id: int) -> Task | None:
        row = self._db.conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return Task(**dict(row)) if row else None

    def find_by_title(self, title: str) -> Task | None:
        row = self._db.conn.execute(
            "SELECT * FROM tasks WHERE title = ? COLLATE NOCASE ORDER BY id",
            (title,),
        ).fetchone()
        return Task(**dict(row)) if row else None

    def list(
        self, status: str | None = "pending", limit: int = 50
    ) -> list[Task]:
        if status is None:
            rows = self._db.conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._db.conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        return [Task(**dict(r)) for r in rows]

    def search(self, query: str) -> list[Task]:
        pattern = f"%{query}%"
        rows = self._db.conn.execute(
            """
            SELECT * FROM tasks
             WHERE title       LIKE ? COLLATE NOCASE
                OR description LIKE ? COLLATE NOCASE
             ORDER BY created_at DESC
            """,
            (pattern, pattern),
        ).fetchall()
        return [Task(**dict(r)) for r in rows]


class MemoryRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, item: MemoryItem) -> MemoryItem:
        now = utcnow()
        cur = self._db.conn.execute(
            """
            INSERT INTO memories (category, content, importance, source,
                                  created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (item.category, item.content, item.importance, item.source, now, now),
        )
        self._db.conn.commit()
        logger.info("[MEMORY] Database operation: INSERT memory #%s", cur.lastrowid)
        return item.model_copy(update={"id": cur.lastrowid, "created_at": now, "updated_at": now})

    def get(self, memory_id: int) -> MemoryItem | None:
        row = self._db.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return MemoryItem(**dict(row)) if row else None

    def find_by_content(self, content: str) -> MemoryItem | None:
        row = self._db.conn.execute(
            "SELECT * FROM memories WHERE content = ? COLLATE NOCASE",
            (content,),
        ).fetchone()
        return MemoryItem(**dict(row)) if row else None

    def search(self, query: str) -> list[MemoryItem]:
        pattern = f"%{query}%"
        rows = self._db.conn.execute(
            """
            SELECT * FROM memories
             WHERE content LIKE ? COLLATE NOCASE
                OR category LIKE ? COLLATE NOCASE
                OR source   LIKE ? COLLATE NOCASE
             ORDER BY importance DESC, updated_at DESC
            """,
            (pattern, pattern, pattern),
        ).fetchall()
        return [MemoryItem(**dict(r)) for r in rows]

    def recent(self, limit: int = 15) -> list[MemoryItem]:
        rows = self._db.conn.execute(
            "SELECT * FROM memories ORDER BY updated_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [MemoryItem(**dict(r)) for r in rows]

    def delete(self, memory_id: int) -> bool:
        cur = self._db.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._db.conn.commit()
        logger.info("[MEMORY] Database operation: DELETE memory #%s", memory_id)
        return cur.rowcount > 0

    def count(self) -> int:
        return int(
            self._db.conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
        )
