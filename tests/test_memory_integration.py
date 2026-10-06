"""Phase 2 -> Phase 3 integration flow (spec section 26).

Conversation -> extraction (mocked LLM, deterministic) -> policy ->
SQLite -> "application restart" (reopen DB) -> retrieval.
"""

import asyncio

import pytest

from app.intelligence.extractor import ConversationIntelligence, ExtractionEngine
from app.memory.database import Database
from app.memory.manager import MemoryManager
from app.memory.migrations import migrate
from app.memory.retrieval import MemoryRetrieval


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)

    async def complete(self, prompt):
        return self.responses.pop(0)


STATE_JSON = """
{
  "person": {"name": "Rahul", "company": "ABC", "role": "Project Manager"},
  "conversation": {"purpose": null, "requirement": "send proposal",
                   "urgency": null, "deadline": "tomorrow"},
  "action": {"callback_required": false, "follow_up": true,
             "follow_up_reason": "Send project proposal to Rahul"}
}
"""


def test_full_pipeline_survives_restart(tmp_path):
    db_path = tmp_path / "jarvis.db"

    # --- session 1: conversation -> extraction -> memory -------------
    m1 = MemoryManager.open(db_path)
    intel = ConversationIntelligence(
        ExtractionEngine(
            ScriptedLLM(STATE_JSON)
        )
    )
    state = asyncio.run(
        intel.process_turn(
            "Rahul from ABC is the project manager. "
            "He needs the proposal tomorrow. Remember that."
        )
    )
    assert state.person.name == "Rahul"
    m1.ingest(state, "Rahul from ABC is the project manager. Remember that.")
    m1.close()

    # --- "restart": reopen the database ------------------------------
    db2 = Database(db_path)
    migrate(db2)
    m2 = MemoryManager(db2)

    contact = m2.get_contact("Rahul")
    assert contact is not None
    assert contact.company == "ABC"
    assert contact.role == "Project Manager"

    tasks = m2.get_tasks(status="pending")
    assert any("proposal" in t.title for t in tasks)

    # --- retrieval answers "Who is Rahul?" ---------------------------
    results = MemoryRetrieval(m2).search("Who is Rahul")
    assert any(r.kind == "contact" and r.text == "Rahul" for r in results)

    m2.close()


def test_multiple_turns_accumulate_then_store(tmp_path):
    db_path = tmp_path / "multi.db"
    m = MemoryManager.open(db_path)

    intel = ConversationIntelligence(
        ExtractionEngine(
            ScriptedLLM(
                '{"person":{"name":"Rahul","company":null,"role":null},'
                '"conversation":{},"action":{}}',
                '{"person":{"name":null,"company":"ABC","role":null},'
                '"conversation":{},"action":{}}',
                '{"person":{"name":null,"company":null,"role":null},'
                '"conversation":{"requirement":"send proposal",'
                '"deadline":"tomorrow"},'
                '"action":{"follow_up":true,'
                '"follow_up_reason":"Send proposal to Rahul"}}',
            )
        )
    )
    asyncio.run(intel.process_turn("Rahul called."))
    asyncio.run(intel.process_turn("He is from ABC."))
    final = asyncio.run(intel.process_turn("He needs the proposal tomorrow."))
    assert final.person.name == "Rahul"
    assert final.person.company == "ABC"

    m.ingest(final, "He needs the proposal tomorrow.")
    assert m.get_contact("Rahul").company == "ABC"
    assert m.get_tasks()[0].deadline == "tomorrow"
    m.close()


def test_brief_context_flows_into_provider_prompt(tmp_path):
    from app.agent.provider import GeminiVoiceProvider

    m = MemoryManager.open(tmp_path / "brief.db")
    m.save_contact("Rahul", company="ABC", role="Project Manager")
    brief = MemoryRetrieval(m).brief()
    assert "Rahul" in brief

    provider = GeminiVoiceProvider(
        api_key="fake", model="fake-model", memory_context=brief
    )
    assert "Persistent memory" in provider._system_prompt
    assert "Rahul" in provider._system_prompt
    m.close()


def test_memory_disabled_session_still_works():
    # memory=None path: intelligence runs, no ingest attempted
    from app.agent.session import VoiceSession
    from app.config import load_settings
    from app.state import StateMachine

    class _Intel:
        async def process_turn(self, text):
            return None

    class _Speaker:
        idle = None

        async def start(self):
            pass

        def play(self, pcm):
            pass

        def clear(self):
            pass

        async def wait_idle(self, timeout=None):
            return True

        async def stop(self):
            pass

    session = VoiceSession(
        settings=load_settings(require_api_key=False),
        microphone=None,  # type: ignore[arg-type]
        speaker=_Speaker(),  # type: ignore[arg-type]
        machine=StateMachine(),
        provider_factory=lambda: None,
        intelligence=_Intel(),
        memory=None,
    )
    assert session.memory is None

    async def drive():
        session._turn_parts.append("hello")
        session._start_intelligence()
        assert session._turn_parts == []
        await asyncio.gather(*session._bg_tasks)

    asyncio.run(drive())
