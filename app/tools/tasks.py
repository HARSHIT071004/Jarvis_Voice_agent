"""Task tools — thin adapters over the Phase 3 MemoryManager.

Jarvis records tasks only; it never executes them (spec section 19).
"""

from __future__ import annotations

from pydantic import Field

from app.memory.manager import MemoryManager
from app.tools.base import RiskLevel, Tool, ToolArgs, ToolError


class CreateTaskArgs(ToolArgs):
    title: str = Field(min_length=1, max_length=500)
    deadline: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=2000)


class GetTasksArgs(ToolArgs):
    status: str = Field(default="pending", pattern="^(pending|done|cancelled|all)$")


class CompleteTaskArgs(ToolArgs):
    task_id: int | None = Field(default=None, ge=1)
    title: str | None = Field(default=None, max_length=500)


def _dump(task) -> dict:
    return {
        "id": task.id,
        "title": task.title,
        "deadline": task.deadline,
        "status": task.status,
        "description": task.description,
    }


class CreateTaskTool(Tool):
    name = "create_task"
    description = (
        "Record a to-do for the user with a title and optional deadline "
        "(free text such as 'tomorrow' or '2026-10-10'). Deduplicates by "
        "title while the same task is still pending. Does NOT perform the task."
    )
    risk = RiskLevel.WRITE
    args_model = CreateTaskArgs

    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    async def execute(self, args: CreateTaskArgs) -> dict:
        task = self._m.save_task(
            title=args.title, deadline=args.deadline, description=args.description
        )
        return _dump(task)


class GetTasksTool(Tool):
    name = "get_tasks"
    description = (
        "List the user's stored tasks. Defaults to pending tasks; "
        "pass status='all' for everything or 'done' for completed ones."
    )
    risk = RiskLevel.READ
    args_model = GetTasksArgs

    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    async def execute(self, args: GetTasksArgs) -> dict:
        status = None if args.status == "all" else args.status
        tasks = self._m.get_tasks(status=status)
        return {"tasks": [_dump(t) for t in tasks[:50]], "count": len(tasks)}


class CompleteTaskTool(Tool):
    name = "complete_task"
    description = (
        "Mark an existing task as done, by task id or exact title. "
        "Fails with TASK_NOT_FOUND when nothing matches."
    )
    risk = RiskLevel.DESTRUCTIVE
    args_model = CompleteTaskArgs

    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    async def execute(self, args: CompleteTaskArgs) -> dict:
        task = None
        if args.task_id is not None:
            task = self._m.tasks.get(args.task_id)
            if task is None:
                raise ToolError("TASK_NOT_FOUND", f"No task with id {args.task_id}.")
        elif args.title:
            task = self._m.tasks.find_by_title(args.title)
            if task is None:
                # fall back to substring match
                matches = self._m.tasks.search(args.title)
                pending = [t for t in matches if t.status == "pending"]
                if len(pending) == 1:
                    task = pending[0]
                elif not matches:
                    raise ToolError("TASK_NOT_FOUND", f"No task matching '{args.title}'.")
                else:
                    raise ToolError(
                        "TASK_AMBIGUOUS",
                        "Multiple tasks matched: " + ", ".join(t.title for t in matches[:5]),
                    )
            if task.status == "done":
                return {**_dump(task), "already_done": True}
        else:
            raise ToolError("INVALID_ARGUMENTS", "Provide task_id or title.")

        updated = self._m.tasks.update(
            task.model_copy(update={"status": "done"})
        )
        return _dump(updated)
