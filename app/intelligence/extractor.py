"""Conversation extraction engine.

Converts conversation text into a validated, normalized ConversationState.

Provider-specific logic is isolated behind the tiny `LLM` interface
(`async complete(prompt) -> str`); unit tests inject fakes and never
touch the network. Failures are raised as ExtractionError so callers
can keep their previous state instead of corrupting it.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from app.intelligence.normalizer import normalize_state
from app.intelligence.prompt import build_extraction_prompt
from app.intelligence.schema import ConversationState

logger = logging.getLogger("jarvis.intelligence")

MAX_HISTORY_TURNS = 12


class ExtractionError(Exception):
    """The LLM returned unusable output; existing state must be kept."""


class LLM(Protocol):
    async def complete(self, prompt: str) -> str: ...


class GeminiTextLLM:
    """Gemini text adapter (google-genai), JSON mode, isolated here."""

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def complete(self, prompt: str) -> str:
        from google.genai import types

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        text = response.text or ""
        if not text.strip():
            raise ExtractionError("empty response from Gemini")
        return text


def _parse_json(raw: str) -> dict:
    """Parse the model response, tolerating markdown code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ExtractionError(f"no JSON object in response: {raw[:200]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ExtractionError("response JSON is not an object")
    return data


class ExtractionEngine:
    """Builds the prompt, calls the LLM, validates and normalizes."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def extract(
        self,
        latest: str,
        existing: ConversationState | None = None,
        history: list[str] | None = None,
    ) -> ConversationState:
        """Extract state from `latest` (context from existing/history).

        Returns the *new* candidate state only — merging is the caller's
        job (see ConversationIntelligence / ConversationState.merge).
        """
        if not latest or not latest.strip():
            return existing or ConversationState()

        prompt = build_extraction_prompt(latest, existing=existing, history=history)
        try:
            raw = await self._llm.complete(prompt)
            data = _parse_json(raw)
            candidate = ConversationState.model_validate(data)
        except ExtractionError:
            raise
        except Exception as exc:  # validation / transport
            raise ExtractionError(str(exc)) from exc

        return normalize_state(candidate)


class ConversationIntelligence:
    """Owns the in-memory conversation state for ONE voice session.

    State is intentionally per-session: Phase 2 forbids persistence.
    """

    def __init__(self, engine: ExtractionEngine) -> None:
        self.engine = engine
        self.state: ConversationState | None = None
        self.history: list[str] = []

    async def process_turn(self, text: str) -> ConversationState | None:
        """Extract + merge one user turn. On failure, keeps old state."""
        text = (text or "").strip()
        if not text:
            return self.state

        logger.info("[INTELLIGENCE] Conversation received: %s", text)
        self.history.append(text)
        if len(self.history) > MAX_HISTORY_TURNS:
            del self.history[:-MAX_HISTORY_TURNS]

        try:
            candidate = await self.engine.extract(
                text, existing=self.state, history=list(self.history)
            )
        except ExtractionError as exc:
            logger.warning("[INTELLIGENCE] Extraction failed, keeping state: %s", exc)
            return self.state

        self.state = (
            candidate if self.state is None else self.state.merge(candidate)
        )
        self._log_state()
        return self.state

    def _log_state(self) -> None:
        assert self.state is not None
        p, c, a = self.state.person, self.state.conversation, self.state.action
        logger.info("[INTELLIGENCE] Extraction completed")
        logger.info("[INTELLIGENCE] Person: %s", p.name)
        logger.info("[INTELLIGENCE] Company: %s", p.company)
        logger.info("[INTELLIGENCE] Requirement: %s", c.requirement)
        logger.info("[INTELLIGENCE] Deadline: %s", c.deadline)
        logger.info("[INTELLIGENCE] Follow-up: %s", a.follow_up)


def build_intelligence(settings) -> ConversationIntelligence | None:
    """Factory for main.py. Returns None when intelligence is off/unusable."""
    if not getattr(settings, "intelligence_enabled", False):
        return None
    if not settings.gemini_api_key:
        logger.warning("[INTELLIGENCE] Disabled: no API key")
        return None
    model = settings.intelligence_model or settings.gemini_model
    llm = GeminiTextLLM(api_key=settings.gemini_api_key, model=model)
    logger.info("[INTELLIGENCE] Enabled (model=%s)", model)
    return ConversationIntelligence(ExtractionEngine(llm))
