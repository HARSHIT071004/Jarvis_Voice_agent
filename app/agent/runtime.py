"""Agent Runtime — the tool-use loop (Phase 4, spec sections 16/29).

Loop: user text → LLM → (tool calls? → ToolRouter → results → LLM)
until the LLM produces a plain final answer or max_iterations is hit.

Design notes:
- The LLM is pluggable (AgentLLM protocol) so unit tests can drive the
  loop with a FakeLLM and no network.
- Every tool call passes through ToolRouter — the runtime never touches
  tools directly, so validation / risk / timeout apply uniformly.
- Loop safety: max_iterations + total tool-call budget; exceeding either
  returns a controlled error, never an infinite loop.
- Grounding: the final answer must follow actual tool results; the
  system instruction (AGENT_PROMPT) states this explicitly and only
  results that really executed are ever fed back to the model.
- `transcript` is an opaque per-request list the LLM adapter uses to
  keep multi-round function-calling state (Content objects for Gemini).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from app.agent.prompt import (
    BROWSER_GUIDANCE,
    PRODUCTIVITY_GUIDANCE,
    RESEARCH_GUIDANCE,
    SECURITY_GUIDANCE,
)
from app.tools.router import ToolResult, ToolRouter

logger = logging.getLogger("jarvis.agent")

AGENT_PROMPT = """\
You are Jarvis, a voice assistant. You have access to controlled tools for
memory, contacts, tasks, web research, browser control, and (when enabled)
email and calendar.

Rules:
- Use a tool ONLY when the request actually needs stored data, a stored
  change, (for current/latest information) live web research, or operating
  a website. General knowledge questions and greetings need no tool.
- Never claim a tool succeeded unless its result said success=true.
- When a tool fails, tell the user the truth briefly; never invent data.
- Never expose raw JSON or tool-call details in your final answer.
- Keep the final answer concise and natural — it will be spoken aloud.
- Multiple tool calls in one turn are fine when genuinely needed.
- Hinglish/Hindi/English are all fine; match the user's language.
""" + RESEARCH_GUIDANCE + BROWSER_GUIDANCE + PRODUCTIVITY_GUIDANCE + SECURITY_GUIDANCE


class LLMToolCall:
    """A tool invocation requested by the LLM."""

    def __init__(self, name: str, arguments: dict, call_id: str | None = None) -> None:
        self.name = name
        self.arguments = arguments or {}
        self.call_id = call_id


class LLMReply:
    """One LLM round-trip: either a final text or tool calls to run."""

    def __init__(self, text: str | None = None, tool_calls: list[LLMToolCall] | None = None) -> None:
        self.text = text
        self.tool_calls = tool_calls or []


class AgentLLM(Protocol):
    """Text-model round-trip used by the runtime loop."""

    async def complete(
        self,
        user_text: str,
        history: list[dict],
        tool_results: list[dict],
        transcript: list,
    ) -> LLMReply:
        """Advance the conversation.

        user_text: the current user request (first round only matters).
        history: prior turns as {"role","text"} dicts.
        tool_results: structured results from tools run since the last
        round (empty on the first round).
        transcript: opaque per-request state owned by the adapter.
        Returns final text OR the next tool calls.
        """
        ...


@dataclass
class AgentResponse:
    text: str
    tool_results: list[ToolResult] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None  # controlled loop-level error, if any


class AgentRuntime:
    def __init__(
        self,
        llm: AgentLLM,
        router: ToolRouter,
        max_iterations: int = 5,
        max_tool_calls: int = 8,
    ) -> None:
        self.llm = llm
        self.router = router
        self.max_iterations = max(1, max_iterations)
        self.max_tool_calls = max(1, max_tool_calls)

    async def handle(self, user_text: str, history: list[dict] | None = None) -> AgentResponse:
        history = history or []
        results: list[ToolResult] = []
        pending: list[dict] = []  # results since the last LLM round
        transcript: list = []  # opaque adapter state for this request
        tool_calls_made = 0
        logger.info("[AGENT] Request: %s", user_text[:120])

        for iteration in range(1, self.max_iterations + 1):
            reply = await self.llm.complete(user_text, history, pending, transcript)
            pending = []

            if not reply.tool_calls:
                text = (reply.text or "").strip()
                if not text:
                    logger.warning("[AGENT] Empty final answer; returning controlled fallback")
                    return AgentResponse(
                        text="Sorry, I couldn't form a response just now.",
                        tool_results=results,
                        iterations=iteration,
                        error="EMPTY_ANSWER",
                    )
                logger.info(
                    "[AGENT] Final answer after %d iteration(s), %d tool call(s)",
                    iteration,
                    len(results),
                )
                return AgentResponse(text=text, tool_results=results, iterations=iteration)

            for call in reply.tool_calls:
                if tool_calls_made >= self.max_tool_calls:
                    logger.warning("[AGENT] Tool-call budget exceeded; stopping safely")
                    return AgentResponse(
                        text=(
                            "I hit my safety limit for tool calls in one request. "
                            "Please try again with a simpler ask."
                        ),
                        tool_results=results,
                        iterations=iteration,
                        error="MAX_TOOL_CALLS_EXCEEDED",
                    )
                tool_calls_made += 1
                logger.info("[AGENT] Tool requested: %s", call.name)
                result = await self.router.route(call.name, call.arguments)
                result.call_id = call.call_id  # echoed back for providers that need it
                results.append(result)
                if not result.success:
                    logger.info(
                        "[TOOL] Error code: %s", result.error["code"] if result.error else "?"
                    )
                pending.append(result.as_dict() | {"call_id": call.call_id})

        logger.warning("[AGENT] Max iterations (%d) exceeded; stopping safely", self.max_iterations)
        return AgentResponse(
            text=(
                "I ran out of steps while working on that. "
                "Could you ask it in a simpler way?"
            ),
            tool_results=results,
            iterations=self.max_iterations,
            error="MAX_ITERATIONS_EXCEEDED",
        )
