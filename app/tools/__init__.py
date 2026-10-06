"""Controlled tool layer for the Jarvis agent (Phase 4).

build_registry(manager) wires the nine Phase-4 tools (memory, contacts,
tasks) around the Phase 3 MemoryManager — no duplicate storage code.
Optional Phase 5 web research tools and Phase 6 browser tools ride the
same registry/router (§23: one runtime, one permission boundary).
"""

from __future__ import annotations

from pydantic import Field

from app.memory.manager import MemoryManager
from app.tools.base import RiskLevel, Tool, ToolArgs, ToolError
from app.tools.browser import BROWSER_TOOL_NAMES, build_browser_tools
from app.tools.contacts import GetContactTool, SaveContactTool, UpdateContactTool
from app.tools.memory import DeleteMemoryTool, SaveMemoryTool, SearchMemoryTool
from app.tools.productivity import PRODUCTIVITY_TOOL_NAMES, build_productivity_tools
from app.tools.registry import ToolRegistry
from app.tools.router import ToolResult, ToolRouter
from app.tools.tasks import CompleteTaskTool, CreateTaskTool, GetTasksTool
from app.tools.web import PageCache, build_web_tools

__all__ = [
    "BROWSER_TOOL_NAMES",
    "PageCache",
    "PRODUCTIVITY_TOOL_NAMES",
    "RiskLevel",
    "Tool",
    "ToolArgs",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolRouter",
    "build_browser_tools",
    "build_productivity_tools",
    "build_registry",
    "build_web_tools",
]


class _MergedReadPageArgs(ToolArgs):
    url: str | None = Field(default=None, max_length=2048)


def _merge_read_page(web_read: Tool, browser_read: Tool) -> Tool:
    """One `read_page` for both specs: Phase 5 mandates read_page(url), Phase 6
    mandates read_page() on the live browser page — merged, never dropped."""

    class MergedReadPageTool(Tool):
        name = "read_page"
        description = (
            "With url: fetch that web page and return its readable text (research). "
            "With no arguments: read the CURRENT browser page as a compact structure "
            "(url, title, text, element references) — call after navigation or a click."
        )
        risk = RiskLevel.READ
        args_model = _MergedReadPageArgs

        async def execute(self, args: _MergedReadPageArgs):
            if args.url:
                return await web_read.execute(web_read.args_model(url=args.url))
            return await browser_read.execute(browser_read.args_model())

    return MergedReadPageTool()


def build_registry(
    manager: MemoryManager | None = None,
    web_tools: list[Tool] | None = None,
    browser_tools: list[Tool] | None = None,
    productivity_tools: list[Tool] | None = None,
) -> ToolRegistry:
    """Register the controlled toolset: Phase 4 memory/contact/task tools
    plus optional Phase 5 web research, Phase 6 browser and Phase 7
    productivity tools — one registry, one permission boundary (§2)."""
    registry = ToolRegistry()
    if manager is not None:
        registry.register_many(
            [
                # memory (READ / WRITE / DESTRUCTIVE)
                SearchMemoryTool(manager),
                SaveMemoryTool(manager),
                DeleteMemoryTool(manager),
                # contacts
                GetContactTool(manager),
                SaveContactTool(manager),
                UpdateContactTool(manager),
                # tasks
                CreateTaskTool(manager),
                GetTasksTool(manager),
                CompleteTaskTool(manager),
            ]
        )
    web = list(web_tools or [])
    browser = list(browser_tools or [])
    merged: list[Tool] = []
    web_read = next((t for t in web if t.name == "read_page"), None)
    browser_read = next((t for t in browser if t.name == "read_page"), None)
    if web_read is not None and browser_read is not None:
        web = [t for t in web if t.name != "read_page"]
        browser = [t for t in browser if t.name != "read_page"]
        merged.append(_merge_read_page(web_read, browser_read))
    if web:
        registry.register_many(web)
    if browser:
        registry.register_many(browser)
    if merged:
        registry.register_many(merged)
    if productivity_tools:
        registry.register_many(list(productivity_tools))
    return registry
