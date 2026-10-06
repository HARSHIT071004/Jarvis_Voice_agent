"""Conversation Intelligence — Phase 2.

Turns conversation text into a structured ConversationState.
Strictly in-memory and per-session: no persistence (that is Phase 3).
"""

from app.intelligence.extractor import (
    ConversationIntelligence,
    ExtractionEngine,
    ExtractionError,
    GeminiTextLLM,
    build_intelligence,
)
from app.intelligence.normalizer import normalize_state, normalize_text
from app.intelligence.schema import Action, Conversation, ConversationState, Person

__all__ = [
    "Action",
    "Conversation",
    "ConversationIntelligence",
    "ConversationState",
    "ExtractionEngine",
    "ExtractionError",
    "GeminiTextLLM",
    "Person",
    "build_intelligence",
    "normalize_state",
    "normalize_text",
]
