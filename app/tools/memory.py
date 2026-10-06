"""Memory tools — thin adapters over the Phase 3 MemoryManager.

The LLM only ever sees these safe interfaces; it never touches SQLite
directly (spec section 7). Secret-policy rejection from Phase 3 is
surfaced as a controlled ToolError so the model can tell the truth.
"""

from __future__ import annotations

from pydantic import Field

from app.memory.manager import MemoryManager
from app.tools.base import RiskLevel, Tool, ToolArgs, ToolError


class SearchMemoryArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=500)


class SaveMemoryArgs(ToolArgs):
    content: str = Field(min_length=1, max_length=2000)
    importance: int = Field(default=3, ge=1, le=5)


class DeleteMemoryArgs(ToolArgs):
    memory_id: int = Field(ge=1)


class SearchMemoryTool(Tool):
    name = "search_memory"
    description = (
        "Search persistent Jarvis memory (remembered facts, contacts, tasks, "
        "prior conversation summaries) for information relevant to a query. "
        "Use when the user asks what Jarvis remembers or when stored context "
        "is needed to answer."
    )
    risk = RiskLevel.READ
    args_model = SearchMemoryArgs

    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    async def execute(self, args: SearchMemoryArgs) -> dict:
        hits = self._m.search_memory(args.query)
        if not hits:
            return {"results": [], "note": "no matching memories"}
        return {
            "results": [
                {"id": h.id, "content": h.content, "importance": h.importance}
                for h in hits[:8]
            ]
        }


class SaveMemoryTool(Tool):
    name = "save_memory"
    description = (
        "Persist a durable piece of information that the user explicitly asked "
        "Jarvis to remember or that conversation established as worth keeping. "
        "Secrets (passwords, API keys, payment details) are rejected by policy."
    )
    risk = RiskLevel.WRITE
    args_model = SaveMemoryArgs

    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    async def execute(self, args: SaveMemoryArgs) -> dict:
        item = self._m.save_memory(content=args.content, importance=args.importance, source="tool")
        if item is None:
            raise ToolError(
                "SECRET_REJECTED",
                "Policy: secret-like content (passwords, keys, tokens) must not be stored.",
            )
        return {"id": item.id, "content": item.content, "importance": item.importance}


class DeleteMemoryTool(Tool):
    name = "delete_memory"
    description = (
        "Delete a single remembered fact by its numeric id after the user "
        "explicitly asked to forget it. Only affects general memories — "
        "contacts and tasks are never deleted by this tool."
    )
    risk = RiskLevel.DESTRUCTIVE
    args_model = DeleteMemoryArgs

    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    async def execute(self, args: DeleteMemoryArgs) -> dict:
        if not self._m.delete_memory(args.memory_id):
            raise ToolError(
                "MEMORY_NOT_FOUND", f"No memory with id {args.memory_id} exists."
            )
        return {"deleted": True, "memory_id": args.memory_id}
