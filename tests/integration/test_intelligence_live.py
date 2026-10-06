"""Live Gemini extraction integration test.

Skipped unless RUN_LIVE_TESTS=1 and GEMINI_API_KEY is set, so the main
suite never depends on the network (per phase2.md §16).
"""

import asyncio
import os

import pytest

from app.intelligence.extractor import ConversationIntelligence, ExtractionEngine
from app.intelligence.schema import ConversationState

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="live API test; set RUN_LIVE_TESTS=1 to enable",
)


def make_intel() -> ConversationIntelligence:
    from app.config import load_settings

    settings = load_settings(require_api_key=True)
    from app.intelligence.extractor import GeminiTextLLM

    llm = GeminiTextLLM(
        api_key=settings.gemini_api_key, model=settings.intelligence_model
    )
    return ConversationIntelligence(ExtractionEngine(llm))


def process(intel, text):
    return asyncio.run(intel.process_turn(text))


def test_live_basic_extraction():
    intel = make_intel()
    state = process(intel, "Rahul from ABC called about the project. He wants the proposal by tomorrow.")
    assert isinstance(state, ConversationState)
    assert state.person.name == "Rahul"
    assert state.person.company == "ABC"
    assert state.conversation.deadline is not None
    assert state.action.follow_up is True


def test_live_no_hallucination():
    intel = make_intel()
    state = process(intel, "Someone called.")
    assert state.person.name is None
    assert state.person.company is None


def test_live_hinglish_and_multi_turn():
    intel = make_intel()
    process(intel, "Rahul ka phone aaya tha.")
    state = process(intel, "Wo ABC se hai, unko proposal kal chahiye.")
    assert state.person.name == "Rahul"
    assert state.person.company == "ABC"
    assert state.conversation.deadline == "tomorrow"
    assert state.action.follow_up is True
