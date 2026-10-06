"""Domain models for persistent memory (Phase 3).

Timestamps: UTC ISO-8601 "YYYY-MM-DDTHH:MM:SSZ" (see database.utcnow).
IDs: stable SQLite INTEGER PRIMARY KEY.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

TASK_STATUSES = ("pending", "done", "cancelled")


class Contact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    name: str
    company: str | None = None
    role: str | None = None
    notes: str | None = None
    created_at: str = ""
    updated_at: str = ""


class ConversationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    person_id: int | None = None
    purpose: str | None = None
    requirement: str | None = None
    urgency: str | None = None
    deadline: str | None = None
    summary: str | None = None
    created_at: str = ""


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    title: str
    description: str | None = None
    deadline: str | None = None
    status: str = "pending"
    person_id: int | None = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def new(
        cls,
        title: str,
        deadline: str | None = None,
        person_id: int | None = None,
        description: str | None = None,
        status: str = "pending",
    ) -> Task:
        return cls(
            title=title,
            deadline=deadline,
            person_id=person_id,
            description=description,
            status=status,
        )


class MemoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    category: str = "general"
    content: str
    importance: int = Field(default=3, ge=1, le=5)
    source: str | None = None
    created_at: str = ""
    updated_at: str = ""
