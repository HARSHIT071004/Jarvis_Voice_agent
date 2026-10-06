"""Central tool registry.

The agent receives tool definitions from here; tool-selection logic is
NEVER hard-coded inside the Gemini provider (spec section 11).
"""

from __future__ import annotations

from app.tools.base import RiskLevel, Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("tool must have a name")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def declarations(self) -> list[dict]:
        """All tools in the format Gemini's function-calling expects."""
        return [t.declaration() for t in self._tools.values()]

    def by_risk(self, risk: RiskLevel) -> list[Tool]:
        return [t for t in self._tools.values() if t.risk is risk]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
