"""Gemini AgentLLM — text function-calling for the Agent Runtime loop.

Uses google-genai `generate_content` with function declarations (the
INTELLIGENCE_MODEL text model — the Live native-audio model rejects
generateContent, see FINDINGS.md). The voice path does NOT use this
class: Gemini Live runs its own function-calling loop server-side and
we route its tool_call events through the same ToolRouter.

Transcript ownership: the runtime hands us one opaque list per request;
we append google-genai Content objects so multi-round function calling
keeps ids/names intact across rounds.
"""

from __future__ import annotations

import asyncio
import logging

from google import genai
from google.genai import errors, types

from app.agent.runtime import LLMReply, LLMToolCall

logger = logging.getLogger("jarvis.agent")

# Free-tier generate_content quota is ~5 req/min; retry politely instead
# of failing the agent loop (spec section 30: controlled error handling).
_429_RETRIES = 5
_429_DELAY_SECONDS = 7.0


class GeminiAgentLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        declarations: list[dict],
        system_instruction: str,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._declarations = declarations
        self._system = system_instruction

    async def complete(
        self,
        user_text: str,
        history: list[dict],
        tool_results: list[dict],
        transcript: list,
    ) -> LLMReply:
        if not transcript:
            # New request: seed transcript with prior turns + user text.
            for turn in history[-12:]:
                transcript.append(
                    types.Content(
                        role="user" if turn.get("role") == "user" else "model",
                        parts=[types.Part(text=turn.get("text", ""))],
                    )
                )
            transcript.append(
                types.Content(role="user", parts=[types.Part(text=user_text)])
            )
        else:
            # Round 2+: append function responses for the calls we just ran.
            responses = []
            for result in tool_results:
                responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=result.get("tool"),
                            id=result.get("call_id"),
                            response=(
                                result.get("data")
                                if result.get("success")
                                else {"error": result.get("error")}
                            ),
                        )
                    )
                )
            if responses:
                transcript.append(types.Content(role="user", parts=responses))

        response = None
        for attempt in range(_429_RETRIES):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=transcript,
                    config=types.GenerateContentConfig(
                        system_instruction=self._system,
                        tools=[types.Tool(function_declarations=self._declarations)],
                        temperature=0.3,
                    ),
                )
                break
            except (errors.ClientError, errors.ServerError) as exc:
                transient = "429" in str(exc) or "503" in str(exc) or "500" in str(exc)
                if not transient or attempt == _429_RETRIES - 1:
                    raise
                logger.info(
                    "Transient API error (attempt %d/%d), waiting %.0fs: %.80s",
                    attempt + 1,
                    _429_RETRIES,
                    _429_DELAY_SECONDS,
                    exc,
                )
                await asyncio.sleep(_429_DELAY_SECONDS)

        # Record the model's turn so the next round sees its function calls.
        if response.candidates and response.candidates[0].content:
            transcript.append(response.candidates[0].content)

        fn_calls = getattr(response, "function_calls", None) or []
        if fn_calls:
            calls = [
                LLMToolCall(
                    name=fc.name,
                    arguments=dict(fc.args or {}),
                    call_id=fc.id,
                )
                for fc in fn_calls
            ]
            logger.info("[AGENT] LLM requested tools: %s", [c.name for c in calls])
            return LLMReply(tool_calls=calls)
        return LLMReply(text=response.text or "")
