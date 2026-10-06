import asyncio

import pytest

from app.agent.provider import (
    AudioData,
    InterruptedEvent,
    TranscriptEvent,
    TurnCompleteEvent,
)
from app.agent.session import VoiceSession, is_exit_request
from app.config import load_settings
from app.state import AppState, StateMachine


class FakeSpeaker:
    def __init__(self):
        self.played = []
        self.clear_count = 0
        self.idle = asyncio.Event()
        self.idle.set()
        self.started = False

    async def start(self):
        self.started = True

    def play(self, pcm: bytes):
        self.played.append(pcm)

    def clear(self):
        self.clear_count += 1
        self.played.clear()

    async def wait_idle(self, timeout=None):
        return True

    async def stop(self):
        pass


class FakeMic:
    def frames(self):
        async def _gen():
            while True:
                yield b"\x00\x00" * 1600

        return _gen()

    def close(self):
        pass


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Goodbye Jarvis.", True),
        ("bye", True),
        ("okay goodbye", True),
        ("see you later", True),
        ("stop listening", True),
        ("What is RAG?", False),
        ("Byzantium is a country", False),
        ("", False),
    ],
)
def test_is_exit_request(text, expected):
    assert is_exit_request(text) is expected


def make_session(intelligence=None, memory=None):
    settings = load_settings(require_api_key=False)
    machine = StateMachine()
    speaker = FakeSpeaker()
    session = VoiceSession(
        settings=settings,
        microphone=FakeMic(),
        speaker=speaker,
        machine=machine,
        provider_factory=lambda: None,
        intelligence=intelligence,
        memory=memory,
    )
    return session, machine, speaker


def test_user_transcript_moves_to_thinking():
    session, machine, _ = make_session()
    machine.transition(AppState.AWAKENING)
    machine.transition(AppState.LISTENING)
    session._handle_event(TranscriptEvent("user", "what is RAG?"))
    assert machine.state is AppState.THINKING


def test_audio_plays_and_state_becomes_speaking():
    session, machine, speaker = make_session()
    machine.transition(AppState.AWAKENING)
    machine.transition(AppState.LISTENING)
    machine.transition(AppState.THINKING)
    session._handle_event(AudioData(b"\x01\x00" * 100))
    assert speaker.played == [b"\x01\x00" * 100]
    assert machine.state is AppState.SPEAKING


def test_barge_in_clears_speaker():
    session, machine, speaker = make_session()
    machine.transition(AppState.AWAKENING)
    machine.transition(AppState.LISTENING)
    machine.transition(AppState.THINKING)
    machine.transition(AppState.SPEAKING)
    session._handle_event(AudioData(b"\x02\x00" * 10))
    session._handle_event(InterruptedEvent())
    assert speaker.clear_count == 1
    assert speaker.played == []
    assert machine.state is AppState.LISTENING


def test_goodbye_transcript_sets_exit_request():
    session, _, _ = make_session()
    assert not session._exit_requested
    session._handle_event(TranscriptEvent("user", "goodbye Jarvis"))
    assert session._exit_requested


def test_turn_complete_stops_when_exit_requested():
    session, machine, _ = make_session()
    machine.transition(AppState.AWAKENING)
    machine.transition(AppState.LISTENING)
    session._handle_event(TranscriptEvent("user", "bye"))
    session._handle_event(TurnCompleteEvent())
    assert session._stop.is_set()


def test_turn_complete_without_exit_keeps_listening():
    session, machine, _ = make_session()
    machine.transition(AppState.AWAKENING)
    machine.transition(AppState.LISTENING)
    session._handle_event(TranscriptEvent("user", "hello"))
    session._handle_event(TurnCompleteEvent())
    assert not session._stop.is_set()
    assert machine.state is AppState.LISTENING


class FakeIntelligence:
    def __init__(self, fail=False):
        self.turns = []
        self.fail = fail

    async def process_turn(self, text):
        if self.fail:
            raise RuntimeError("extraction blew up")
        self.turns.append(text)
        return None


class FakeMemory:
    def __init__(self, fail=False):
        self.ingested = []
        self.fail = fail

    def ingest(self, state, text):
        if self.fail:
            raise RuntimeError("db exploded")
        self.ingested.append((state, text))


def test_memory_ingest_receives_completed_turn():
    async def scenario():
        intel = FakeIntelligence()
        mem = FakeMemory()
        session, machine, _ = make_session(intelligence=intel, memory=mem)
        _complete_turn(session, machine, "Rahul from ABC called")
        await asyncio.gather(*session._bg_tasks)
        return mem.ingested

    ingested = asyncio.run(scenario())
    assert len(ingested) == 1
    assert ingested[0][1] == "Rahul from ABC called"


def test_memory_ingest_runs_even_without_intelligence():
    async def scenario():
        mem = FakeMemory()
        session, machine, _ = make_session(intelligence=None, memory=mem)
        _complete_turn(session, machine, "Remember that I prefer Python")
        await asyncio.gather(*session._bg_tasks)
        return mem.ingested

    ingested = asyncio.run(scenario())
    assert ingested[0][0] is None  # state None, directive still processed
    assert "prefer Python" in ingested[0][1]


def test_memory_failure_does_not_break_session():
    async def scenario():
        mem = FakeMemory(fail=True)
        session, machine, _ = make_session(
            intelligence=FakeIntelligence(), memory=mem
        )
        _complete_turn(session, machine, "hello")
        results = await asyncio.gather(*session._bg_tasks, return_exceptions=True)
        return results, session._stop.is_set()

    results, stopped = asyncio.run(scenario())
    assert results == [None]
    assert not stopped


def test_intelligence_failure_still_allows_memory():
    async def scenario():
        intel = FakeIntelligence(fail=True)
        mem = FakeMemory()
        session, machine, _ = make_session(intelligence=intel, memory=mem)
        _complete_turn(session, machine, "Remember this")
        await asyncio.gather(*session._bg_tasks, return_exceptions=True)
        return mem.ingested

    # extraction failure must NOT block explicit memory commands
    ingested = asyncio.run(scenario())
    assert ingested[0][0] is None
    assert ingested[0][1] == "Remember this"


def _complete_turn(session, machine, *texts):
    machine.transition(AppState.AWAKENING)
    machine.transition(AppState.LISTENING)
    for t in texts:
        session._handle_event(TranscriptEvent("user", t))
    session._handle_event(TurnCompleteEvent())


def test_intelligence_receives_completed_turn():
    async def scenario():
        intel = FakeIntelligence()
        session, machine, _ = make_session(intelligence=intel)
        _complete_turn(
            session, machine, "Rahul from ABC", "called about the project"
        )
        await asyncio.gather(*session._bg_tasks)
        return intel.turns

    assert asyncio.run(scenario()) == ["Rahul from ABC called about the project"]


def test_no_intelligence_turn_still_completes():
    session, machine, _ = make_session(intelligence=None)
    _complete_turn(session, machine, "hello")
    assert not session._stop.is_set()
    assert session._turn_parts == []


def test_exit_turn_does_not_trigger_extraction():
    async def scenario():
        intel = FakeIntelligence()
        session, machine, _ = make_session(intelligence=intel)
        machine.transition(AppState.AWAKENING)
        machine.transition(AppState.LISTENING)
        session._handle_event(TranscriptEvent("user", "bye"))
        session._handle_event(TurnCompleteEvent())
        await asyncio.gather(*session._bg_tasks, return_exceptions=True)
        return intel.turns, session._stop.is_set()

    turns, stopped = asyncio.run(scenario())
    assert turns == []
    assert stopped


def test_intelligence_crash_does_not_break_session():
    async def scenario():
        intel = FakeIntelligence(fail=True)
        session, machine, _ = make_session(intelligence=intel)
        _complete_turn(session, machine, "hello")
        results = await asyncio.gather(
            *session._bg_tasks, return_exceptions=True
        )
        return results, session._stop.is_set()

    results, stopped = asyncio.run(scenario())
    assert results == [None]  # swallowed inside _run_intelligence
    assert not stopped


def test_turn_parts_accumulate_then_clear():
    async def scenario():
        intel = FakeIntelligence()
        session, machine, _ = make_session(intelligence=intel)
        session._handle_event(TranscriptEvent("user", "one"))
        assert session._turn_parts == ["one"]
        _complete_turn(session, machine, "two")
        await asyncio.gather(*session._bg_tasks)
        return intel.turns, session._turn_parts

    turns, parts = asyncio.run(scenario())
    assert turns == ["one two"]  # both parts joined into one extraction
    assert parts == []
