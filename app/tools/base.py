"""Tool abstraction: one consistent interface for every tool.

Each tool declares name / description / input schema / risk level and
exposes an async execute(). Arguments are validated with a per-tool
Pydantic model BEFORE execution — the LLM cannot bypass validation.

Risk levels exist so the router (and later Phase 8 approval flows)
can make permission decisions without knowing tool internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class RiskLevel(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    SENSITIVE = "SENSITIVE"      # Phase 6: needs approval (upload, form submit...)
    DESTRUCTIVE = "DESTRUCTIVE"
    HIGH_RISK = "HIGH_RISK"      # Phase 7/8: purchases, sends, deletions


class ToolError(Exception):
    """Controlled tool failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ToolArgs(BaseModel):
    """Base for tool argument models (extra args are rejected)."""

    model_config = ConfigDict(extra="forbid")


class Tool(ABC):
    """A single controlled capability exposed to the agent."""

    name: str = ""
    description: str = ""
    risk: RiskLevel = RiskLevel.READ
    args_model: type[ToolArgs] = ToolArgs

    @abstractmethod
    async def execute(self, args: ToolArgs) -> Any:
        """Run the tool with validated args. Raise ToolError on failure."""

    def validate(self, arguments: dict) -> ToolArgs:
        """Validate raw LLM arguments. Raises ToolError when invalid."""
        if not isinstance(arguments, dict):
            raise ToolError(
                "INVALID_ARGUMENTS", f"arguments must be an object, got {type(arguments).__name__}"
            )
        try:
            return self.args_model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolError("INVALID_ARGUMENTS", problems) from exc

    def declaration(self) -> dict:
        """Gemini function-declaration format (JSON schema).

        Pydantic emits keywords Gemini's OpenAPI subset rejects
        (additionalProperties, $schema, titles) — strip them recursively.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _sanitize_schema(self.args_model.model_json_schema()),
        }


_PYDANTIC_ONLY_KEYS = {"title", "additionalProperties", "$schema", "default"}


def _sanitize_schema(schema):
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                # property NAMES may be "title"/"default" — keep them
                out[key] = {k: _sanitize_schema(v) for k, v in value.items()}
            elif key in _PYDANTIC_ONLY_KEYS:
                continue
            else:
                out[key] = _sanitize_schema(value)
        return out
    if isinstance(schema, list):
        return [_sanitize_schema(v) for v in schema]
    return schema
