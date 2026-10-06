"""Voice session lifecycle.

One session = wake word detected -> conversation -> goodbye/timeout -> close.
Coordinates microphone, speaker, provider and the state machine; owns
reconnection, barge-in and the session timeouts from configuration.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from app.agent.provider import (
    AudioData,
    InterruptedEvent,
    ProviderConnectionError,
    ProviderEvent,
    ToolCallEvent,
    TranscriptEvent,
    TurnCompleteEvent,
    VoiceProvider,
)
from app.audio.input import Microphone
from app.audio.output import Speaker
from app.config import Settings
from app.state import AppState, StateMachine

if TYPE_CHECKING:
    from app.intelligence.extractor import ConversationIntelligence
    from app.memory.manager import MemoryManager
    from app.tools.router import ToolRouter

logger = logging.getLogger("jarvis.session")

EXIT_PHRASES = re.compile(
    r"\b(goodbye|good bye|bye bye|bye|see you|stop listening|sleep now|that's all)\b",
    re.IGNORECASE,
)


def is_exit_request(text: str) -> bool:
    """True when the user is asking to end the session."""
    return bool(text and EXIT_PHRASES.search(text))


class VoiceSession:
    def __init__(
        self,
        settings: Settings,
        microphone: Microphone,
        speaker: Speaker,
        machine: StateMachine,
        provider_factory: Callable[[], VoiceProvider],
        intelligence: ConversationIntelligence | None = None,
        memory: MemoryManager | None = None,
        tool_router: ToolRouter | None = None,
    ) -> None:
        self.settings = settings
        self.mic = microphone
        self.speaker = speaker
        self.machine = machine
        self.provider_factory = provider_factory
        self.intelligence = intelligence
        self.memory = memory
        self.tool_router = tool_router

        self._stop = asyncio.Event()
        self._exit_requested = False
        self._metrics: dict[str, float] = {}
        self._turn_parts: list[str] = []
        self._bg_tasks: set[asyncio.Task] = set()
        self._provider: VoiceProvider | None = None
        self._tool_calls_this_turn = 0

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        """Drive one full conversation until goodbye/timeout/error."""
        started = time.monotonic()
        self._metrics["wake"] = started

        self.machine.transition(AppState.AWAKENING)
        logger.info("Voice session started")
        await self.speaker.start()

        attempt = 0
        while True:
            provider = self.provider_factory()
            try:
                await self._run_with_provider(provider, started)
                break
            except ProviderConnectionError as exc:
                attempt += 1
                if attempt > self.settings.reconnect_attempts or self._stop.is_set():
                    logger.error("Session failed after %d attempt(s): %s", attempt, exc)
                    raise
                logger.warning(
                    "Connection lost (%s). Reconnecting (%d/%d)...",
                    exc,
                    attempt,
                    self.settings.reconnect_attempts,
                )
                self.speaker.clear()
            finally:
                self._provider = None
                await provider.close()

        if not self._metrics.get("first_audio"):
            logger.info(
                "Latency: connect=%dms (no model response before session end)",
                self._ms("connect"),
            )

    async def _run_with_provider(self, provider: VoiceProvider, started: float) -> None:
        await provider.connect()
        self._provider = provider
        self._metrics["connect"] = time.monotonic()
        logger.info("Session connection: %d ms", self._ms("connect"))

        # Phase 8: each voice conversation gets a fresh approval scope;
        # approvals from a previous conversation cannot be replayed (§21).
        security = self._security()
        if security is not None:
            security.new_session()

        self._transition_quietly(AppState.LISTENING)
        await provider.send_text(self.settings.wake_word)

        tasks = [
            asyncio.create_task(self._mic_loop(provider), name="mic-to-provider"),
            asyncio.create_task(self._event_loop(provider), name="provider-events"),
            asyncio.create_task(self._timeout_loop(), name="session-timeouts"),
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ----------------------------------------------------------------- loops

    async def _mic_loop(self, provider: VoiceProvider) -> None:
        """Microphone -> provider (only while the session is alive)."""
        async for frame in self.mic.frames():
            if self._stop.is_set():
                break
            await provider.send_audio(frame)
        self._stop.set()

    async def _event_loop(self, provider: VoiceProvider) -> None:
        """Provider -> speaker / transcripts / state machine."""
        async for event in provider.events():
            if isinstance(event, ToolCallEvent):
                # Phase 4: tool execution is async, handled inline so the
                # model waits for our response before continuing.
                await self._handle_tool_calls(event, provider)
            else:
                self._handle_event(event)
            if self._stop.is_set():
                break
        self._stop.set()

    async def _timeout_loop(self) -> None:
        max_duration = self.settings.voice_session_timeout
        inactivity = self.settings.voice_inactivity_timeout
        start = time.monotonic()
        last_activity = start
        while not self._stop.is_set():
            await asyncio.sleep(0.5)
            now = time.monotonic()
            if now - start >= max_duration:
                logger.info("Maximum session duration reached (%.0f s)", max_duration)
                self._stop.set()
                return
            if self._exit_requested:
                # give Jarvis time to say the farewell, then stop
                if await self.speaker.wait_idle(timeout=2.0):
                    self._stop.set()
                    return
                last_activity = now
                continue
            if not self.speaker.idle.is_set():
                last_activity = now  # she is speaking; not idle
                continue
            if now - last_activity >= inactivity:
                logger.info("Inactivity timeout after %.0f s", inactivity)
                self._stop.set()
                return

    # --------------------------------------------------------------- events

    def _handle_event(self, event: ProviderEvent) -> None:
        now = time.monotonic()
        if isinstance(event, AudioData):
            if "first_audio" not in self._metrics:
                self._metrics["first_audio"] = now
                logger.info(
                    "First response: %d ms after wake", self._ms("first_audio")
                )
            if self.machine.state in (AppState.LISTENING, AppState.THINKING):
                self.machine.transition(AppState.SPEAKING)
            self.speaker.play(event.data)
        elif isinstance(event, InterruptedEvent):
            logger.info("Barge-in detected, flushing playback")
            self.speaker.clear()
            self._transition_quietly(AppState.LISTENING)
        elif isinstance(event, TranscriptEvent):
            if event.role == "user":
                logger.info("You: %s", event.text)
                self._turn_parts.append(event.text)
                # Phase 8: a live user answer resolves a pending approval —
                # only user-role transcripts are trusted for this (§23).
                security = self._security()
                if security is not None:
                    verdict = security.handle_user_utterance(event.text)
                    if verdict:
                        logger.info("[SECURITY] User %s pending approval", verdict)
                if is_exit_request(event.text):
                    self._exit_requested = True
                if self.machine.state == AppState.LISTENING:
                    self.machine.transition(AppState.THINKING)
            else:
                logger.info("Jarvis: %s", event.text)
        elif isinstance(event, TurnCompleteEvent):
            self._tool_calls_this_turn = 0  # new turn: fresh tool budget
            if self._exit_requested:
                logger.info("Goodbye detected, ending session")
                self._stop.set()
            else:
                self._start_intelligence()
                self._transition_quietly(AppState.LISTENING)

    # -------------------------------------------------------------- tools

    async def _handle_tool_calls(
        self, event: ToolCallEvent, provider: VoiceProvider
    ) -> None:
        """Route Live tool_call requests through the ToolRouter (Phase 4).

        Every call is validated/risk-checked/timeout-bounded by the
        router; results (success or structured error) go back to the
        model so its answer stays grounded in real execution.
        """
        results: list[dict] = []
        budget = self.settings.agent_max_tool_calls
        for call_id, name, args in event.calls:
            if self.tool_router is None:
                logger.error("[AGENT] Tool requested but no router configured: %s", name)
                results.append(
                    {
                        "tool": name,
                        "success": False,
                        "error": {"code": "TOOLS_DISABLED", "message": "Tools are not available."},
                        "call_id": call_id,
                    }
                )
                continue
            self._tool_calls_this_turn += 1
            if self._tool_calls_this_turn > budget:
                logger.warning("[AGENT] Per-turn tool budget exceeded (%d)", budget)
                results.append(
                    {
                        "tool": name,
                        "success": False,
                        "error": {
                            "code": "MAX_TOOL_CALLS",
                            "message": "Tool-call limit for this turn reached; answer from what you already know.",
                        },
                        "call_id": call_id,
                    }
                )
                continue
            logger.info("[AGENT] Tool requested: %s", name)
            result = await self.tool_router.route(name, args)
            result.call_id = call_id
            results.append(result.as_dict() | {"call_id": call_id})
        if results:
            await provider.send_tool_response(results)

    def _transition_quietly(self, target: AppState) -> None:
        try:
            if self.machine.state != target:
                self.machine.transition(target)
        except Exception:
            logger.debug("Ignored transition %s -> %s", self.machine.state, target)

    def _security(self):
        """The attached Phase 8 control plane, or None (§23)."""
        security = getattr(self.tool_router, "security", None)
        if security is not None and getattr(security, "enabled", False):
            return security
        return None

    # ----------------------------------------------------------- intelligence

    def _start_intelligence(self) -> None:
        """Kick off background extraction for the completed turn (Phase 2).

        Voice response must never wait for extraction, so this is a
        fire-and-forget task; failures are logged, never raised.
        """
        text = " ".join(p for p in self._turn_parts if p).strip()
        self._turn_parts.clear()
        if not text or (self.intelligence is None and self.memory is None):
            return
        task = asyncio.create_task(
            self._run_intelligence(text), name="intelligence-extract"
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _run_intelligence(self, text: str) -> None:
        # Phase 2: extract conversation state (never raises out of here)
        state = None
        if self.intelligence is not None:
            try:
                state = await self.intelligence.process_turn(text)
            except Exception:
                logger.exception(
                    "Background intelligence extraction crashed; "
                    "memory will only see the raw turn"
                )

        # Phase 3: policy -> store, then prime the model with relevant memory
        if self.memory is None:
            return
        try:
            self.memory.ingest(state, text)
        except Exception:
            logger.exception("Memory ingest failed")
            return
        await self._send_memory_note(text)

    async def _send_memory_note(self, text: str) -> None:
        """Per-request retrieval (spec section 16): only relevant items."""
        provider = self._provider
        if provider is None:
            return
        try:
            from app.memory.retrieval import MemoryRetrieval

            results = MemoryRetrieval(self.memory).search(text, limit=5)  # type: ignore[arg-type]
            if not results:
                return
            lines = []
            for r in results:
                line = f"- [{r.kind}] {r.text}"
                if r.detail:
                    line += f" ({r.detail})"
                lines.append(line)
            await provider.send_context(
                "Relevant stored memory for the conversation:\n" + "\n".join(lines)
            )
        except Exception:
            logger.debug("Could not send memory context", exc_info=True)

    # -------------------------------------------------------------- helpers

    def _ms(self, key: str) -> int:
        if "wake" not in self._metrics or key not in self._metrics:
            return 0
        return int((self._metrics[key] - self._metrics["wake"]) * 1000)

    def request_stop(self) -> None:
        self._stop.set()
