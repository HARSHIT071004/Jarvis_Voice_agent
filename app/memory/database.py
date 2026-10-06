"""SQLite connection management.

One connection per Database instance, used from the event-loop thread
(all operations are small and synchronous). Foreign keys are enforced;
WAL mode keeps reads cheap while a session is writing.

Timestamps (whole system): UTC ISO-8601, "YYYY-MM-DDTHH:MM:SSZ".
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def utcnow() -> str:
    """Current UTC time in the canonical stored format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Database:
    """Thin SQLite wrapper; all SQL lives in repository.py / migrations.py."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.Error:
            pass  # WAL not critical (e.g. network shares)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()
