"""Voice provider abstraction.

The rest of the application talks to VoiceProvider only, so Gemini can be
swapped for another real-time model later without touching session or audio
code (see phase1.md, "PROVIDER ABSTRACTION").
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from google import genai
from google.genai import types

from app.agent.prompt import SYSTEM_PROMPT

logger = logging.getLogger("jarvis.provider")


class ProviderError(Exception):
    """Base class for voice-provider failures."""


class ProviderConnectionError(ProviderError):
    """Connection to the real-time API failed or dropped."""


@dataclass(frozen=True)
class AudioData:
    data: bytes


@dataclass(frozen=True)
class TranscriptEvent:
    role: str  # "user" or "jarvis"
    text: str


@dataclass(frozen=True)
class InterruptedEvent:
    """The user barged in; queued playback must be flushed."""


@dataclass(frozen=True)
class TurnCompleteEvent:
    """The model finished its turn."""


@dataclass(frozen=True)
class ToolCallEvent:
    """The model requested one or more tool calls (Phase 4).

    Each call is (call_id, name, arguments); the session must route them
    through the ToolRouter and answer with send_tool_response().
    """

    calls: tuple[tuple[str | None, str, dict], ...]


ProviderEvent = (
    AudioData | TranscriptEvent | InterruptedEvent | TurnCompleteEvent | ToolCallEvent
)


class VoiceProvider(ABC):
    """Minimal real-time voice interface."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None: ...

    @abstractmethod
    async def send_text(self, text: str) -> None: ...

    @abstractmethod
    def events(self) -> AsyncIterator[ProviderEvent]: ...

    async def send_context(self, text: str) -> None:
        """Add background context without triggering a reply (optional).

        Default is a no-op so Phase 1 providers/fakes stay valid;
        Gemini implements this by appending content with turn_complete=False.
        """
        return None

    async def send_tool_response(self, responses: list[dict]) -> None:
        """Answer a ToolCallEvent (Phase 4). Default is a no-op for fakes."""
        return None

    @abstractmethod
    async def close(self) -> None: ...


class GeminiVoiceProvider(VoiceProvider):
    """Gemini Live API provider (official google-genai SDK, WebSocket)."""

    INPUT_RATE = 16000
    OUTPUT_RATE = 24000

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str = SYSTEM_PROMPT,
        memory_context: str = "",
        tool_declarations: list[dict] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt
        self._tool_declarations = tool_declarations or []
        # Phase 5/6/7: teach the voice model when to research / drive the
        # browser / handle mail & calendar.
        from app.agent.prompt import (
            BROWSER_GUIDANCE,
            PRODUCTIVITY_GUIDANCE,
            RESEARCH_GUIDANCE,
            SECURITY_GUIDANCE,
        )
        from app.tools.browser import BROWSER_TOOL_NAMES
        from app.tools.productivity import PRODUCTIVITY_TOOL_NAMES

        if any(d.get("name") == "search_web" for d in self._tool_declarations):
            self._system_prompt += RESEARCH_GUIDANCE
        if any(d.get("name") in BROWSER_TOOL_NAMES for d in self._tool_declarations):
            self._system_prompt += BROWSER_GUIDANCE
        if any(d.get("name") in PRODUCTIVITY_TOOL_NAMES for d in self._tool_declarations):
            self._system_prompt += PRODUCTIVITY_GUIDANCE
        if self._tool_declarations:
            # Phase 8: approval flow rules travel with every toolset.
            self._system_prompt += SECURITY_GUIDANCE
        if memory_context:
            self._system_prompt = (
                f"{self._system_prompt}\n\nPersistent memory (context only — "
                "never invent facts beyond these):\n" + memory_context
            )
        self._client: genai.Client | None = None
        self._session: object | None = None
        self._connect_cm = None
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            self._client = genai.Client(api_key=self._api_key)
            config_kwargs: dict = {
                "response_modalities": ["AUDIO"],
                "system_instruction": self._system_prompt,
                "input_audio_transcription": types.AudioTranscriptionConfig(),
                "output_audio_transcription": types.AudioTranscriptionConfig(),
            }
            if self._tool_declarations:
                # Phase 4: expose the controlled toolset to the Live model.
                config_kwargs["tools"] = [
                    types.Tool(function_declarations=self._tool_declarations)
                ]
                logger.info(
                    "Tools exposed to Live model: %d", len(self._tool_declarations)
                )
            config = types.LiveConnectConfig(**config_kwargs)
            self._connect_cm = self._client.aio.live.connect(
                model=self._model, config=config
            )
            self._session = await self._connect_cm.__aenter__()
            self._connected = True
            logger.info("Connected to %s", self._model)
        except Exception as exc:
            await self.close()
            raise ProviderConnectionError(f"Could not connect to {self._model}: {exc}") from exc

    async def send_audio(self, pcm: bytes) -> None:
        if not self._connected or self._session is None or not pcm:
            return
        try:
            await self._session.send_realtime_input(
                audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={self.INPUT_RATE}")
            )
        except Exception as exc:
            raise ProviderConnectionError(f"send_audio failed: {exc}") from exc

    async def send_text(self, text: str) -> None:
        if not self._connected or self._session is None:
            return
        try:
            await self._session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True,
            )
        except Exception as exc:
            raise ProviderConnectionError(f"send_text failed: {exc}") from exc

    async def send_context(self, text: str) -> None:
        """Prime memory context for upcoming turns without replying."""
        if not self._connected or self._session is None or not text:
            return
        try:
            await self._session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=False,
            )
            logger.debug("Context primed (%d chars)", len(text))
        except Exception as exc:
            raise ProviderConnectionError(f"send_context failed: {exc}") from exc

    async def send_tool_response(self, responses: list[dict]) -> None:
        """Answer the model's tool calls with structured results.

        Each response: {"tool", "success", "data", "error", "call_id"} —
        exactly the ToolRouter's output shape, so nothing unvalidated
        ever reaches the model.
        """
        if not self._connected or self._session is None or not responses:
            return
        function_responses = [
            types.FunctionResponse(
                name=r.get("tool") or "unknown",
                id=r.get("call_id"),
                response=(
                    {"success": True, "data": r.get("data")}
                    if r.get("success")
                    else {"success": False, "error": r.get("error")}
                ),
            )
            for r in responses
        ]
        try:
            await self._session.send_tool_response(
                function_responses=function_responses
            )
            logger.info(
                "Tool responses sent: %s",
                [r.get("tool") for r in responses],
            )
        except Exception as exc:
            raise ProviderConnectionError(f"send_tool_response failed: {exc}") from exc

    async def events(self) -> AsyncIterator[ProviderEvent]:
        if not self._connected or self._session is None:
            raise ProviderConnectionError("Not connected")
        try:
            while True:
                async for message in self._session.receive():
                    for event in self._translate(message):
                        yield event
                # receive() stops at turn completion; loop for the next turn
        except ProviderConnectionError:
            raise
        except Exception as exc:
            raise ProviderConnectionError(f"Connection lost: {exc}") from exc

    @staticmethod
    def _translate(message: types.LiveServerMessage) -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        # Phase 4: the model wants tools executed (may arrive without
        # any server_content in the same message).
        if message.tool_call and message.tool_call.function_calls:
            calls = tuple(
                (fc.id, fc.name, dict(fc.args or {}))
                for fc in message.tool_call.function_calls
            )
            logger.info(
                "[AGENT] Tool requested: %s", [c[1] for c in calls]
            )
            events.append(ToolCallEvent(calls=calls))
        sc = message.server_content
        if sc is None:
            return events
        if sc.interrupted:
            events.append(InterruptedEvent())
        if sc.input_transcription and sc.input_transcription.text:
            events.append(TranscriptEvent("user", sc.input_transcription.text))
        if sc.output_transcription and sc.output_transcription.text:
            events.append(TranscriptEvent("jarvis", sc.output_transcription.text))
        if sc.model_turn and sc.model_turn.parts:
            for part in sc.model_turn.parts:
                if part.inline_data and isinstance(part.inline_data.data, bytes):
                    events.append(AudioData(part.inline_data.data))
        if sc.turn_complete:
            events.append(TurnCompleteEvent())
        return events

    async def close(self) -> None:
        cm, self._connect_cm = self._connect_cm, None
        self._session = None
        if self._connected or cm is not None:
            self._connected = False
            try:
                if cm is not None:
                    await cm.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error while closing Live session", exc_info=True)
            logger.info("Live session closed")
