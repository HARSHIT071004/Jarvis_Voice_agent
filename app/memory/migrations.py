"""Schema creation and versioning.

Rules:
- SCHEMA_VERSION tracks the highest known migration.
- Migrations run once each (recorded in schema_migrations) and are
  append-only: never DROP/ALTER destructively, so future versions can
  migrate v1 -> v2 without destroying stored memories.
- The database is NOT recreated at startup (spec section 30).
"""

from __future__ import annotations

import logging

from app.memory.database import Database, utcnow

logger = logging.getLogger("jarvis.memory")

SCHEMA_VERSION = 1

# version -> list of SQL statements (append-only history)
MIGRATIONS: dict[int, list[str]] = {
    1: [
        """
        CREATE TABLE IF NOT EXISTS contacts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            company    TEXT,
            role       TEXT,
            notes      TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_contacts_name ON contacts(name COLLATE NOCASE)",
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER REFERENCES contacts(id) ON DELETE SET NULL,
            purpose     TEXT,
            requirement TEXT,
            urgency     TEXT,
            deadline    TEXT,
            summary     TEXT,
            created_at  TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_conversations_person ON conversations(person_id)",
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            description TEXT,
            deadline    TEXT,
            status      TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'done', 'cancelled')),
            person_id   INTEGER REFERENCES contacts(id) ON DELETE SET NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
        """
        CREATE TABLE IF NOT EXISTS memories (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            category   TEXT NOT NULL DEFAULT 'general',
            content    TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 3
                       CHECK (importance BETWEEN 1 AND 5),
            source     TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_memories_content ON memories(content COLLATE NOCASE)",
    ],
}


def get_version(db: Database) -> int:
    row = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if row is None:
        return 0
    row = db.conn.execute(
        "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations"
    ).fetchone()
    return int(row["v"])


def migrate(db: Database) -> int:
    """Apply pending migrations. Returns the resulting schema version."""
    db.conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    current = get_version(db)
    for version in sorted(MIGRATIONS):
        if version <= current:
            continue
        logger.info("[MEMORY] Applying schema migration v%d", version)
        try:
            for statement in MIGRATIONS[version]:
                db.conn.execute(statement)
            db.conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, utcnow()),
            )
            db.conn.commit()
        except Exception:
            db.conn.rollback()
            raise
        current = version
    return current
