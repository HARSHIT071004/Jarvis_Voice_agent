"""Extraction engine tests with a mocked LLM (no live API calls).

Validates: schema validity, null handling, no-hallucination passthrough,
multi-turn context, state merging through ConversationIntelligence,
prompt contents, and failure isolation.
"""

import json

import pytest

from app.intelligence.extractor import (
    ConversationIntelligence,
    ExtractionEngine,
    ExtractionError,
)
from app.intelligence.prompt import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from app.intelligence.schema import ConversationState


def state_json(
    name=None,
    company=None,
    role=None,
    purpose=None,
    requirement=None,
    urgency=None,
    deadline=None,
    callback=False,
    follow_up=False,
    follow_up_reason=None,
):
    return json.dumps(
        {
            "person": {"name": name, "company": company, "role": role},
            "conversation": {
                "purpose": purpose,
                "requirement": requirement,
                "urgency": urgency,
                "deadline": deadline,
            },
            "action": {
                "callback_required": callback,
                "follow_up": follow_up,
                "follow_up_reason": follow_up_reason,
            },
        }
    )


class FakeLLM:
    """Returns scripted responses in order; records every prompt."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("FakeLLM exhausted")
        return self.responses.pop(0)


def run(engine, *args, **kwargs):
    import asyncio

    return asyncio.run(engine.extract(*args, **kwargs))


class TestBasicExtraction:
    def test_basic_extraction(self):
        llm = FakeLLM(
            state_json(name="Rahul", company="ABC", purpose="project")
        )
        engine = ExtractionEngine(llm)
        state = run(engine, "Rahul from ABC called about the project.")
        assert state.person.name == "Rahul"
        assert state.person.company == "ABC"
        assert state.conversation.purpose == "project"

    def test_missing_information_stays_null(self):
        llm = FakeLLM(state_json(purpose="project"))
        engine = ExtractionEngine(llm)
        state = run(engine, "Someone called about a project.")
        assert state.person.name is None
        assert state.person.company is None
        assert state.conversation.purpose == "project"

    def test_deadline_and_follow_up(self):
        llm = FakeLLM(
            state_json(
                requirement="send proposal",
                deadline="tomorrow",
                follow_up=True,
                follow_up_reason="Send proposal",
            )
        )
        engine = ExtractionEngine(llm)
        state = run(engine, "Send the proposal tomorrow.")
        assert state.conversation.requirement == "send proposal"
        assert state.conversation.deadline == "tomorrow"
        assert state.action.follow_up is True

    def test_hindi_input_accepted_by_pipeline(self):
        # Hindi/Hinglish understanding is the model's job; the engine must
        # accept and normalize whatever structured output comes back.
        llm = FakeLLM(
            state_json(
                name="Rahul", requirement="send proposal", deadline="tomorrow"
            )
        )
        engine = ExtractionEngine(llm)
        state = run(engine, "Rahul ka phone aaya tha, kal proposal bhejna hai.")
        assert state.person.name == "Rahul"
        assert state.conversation.deadline == "tomorrow"

    def test_hinglish_input_accepted_by_pipeline(self):
        llm = FakeLLM(state_json(name="Rahul", company="ABC", deadline="tomorrow"))
        engine = ExtractionEngine(llm)
        state = run(engine, "Rahul from ABC called, unko proposal kal chahiye.")
        assert state.person.name == "Rahul"
        assert state.person.company == "ABC"

    def test_no_hallucination_nulls_pass_through(self):
        llm = FakeLLM(state_json())
        engine = ExtractionEngine(llm)
        state = run(engine, "Someone called.")
        assert state.person.name is None
        assert state.person.company is None
        assert state.conversation.purpose is None
        assert state.action.follow_up is False

    def test_model_output_normalized(self):
        llm = FakeLLM(state_json(name=" Rahul ", company="unknown"))
        engine = ExtractionEngine(llm)
        state = run(engine, "Rahul called")
        assert state.person.name == "Rahul"
        assert state.person.company is None

    def test_empty_input_returns_existing_without_llm_call(self):
        llm = FakeLLM()
        engine = ExtractionEngine(llm)
        existing = ConversationState()
        state = run(engine, "   ", existing=existing)
        assert state is existing
        assert llm.prompts == []


class TestPrompt:
    def test_prompt_contains_no_hallucination_rules(self):
        llm = FakeLLM(state_json())
        engine = ExtractionEngine(llm)
        run(engine, "Someone called.")
        prompt = llm.prompts[0]
        assert "Do not invent" in prompt
        assert "null" in prompt.lower()
        assert EXTRACTION_SYSTEM_PROMPT.splitlines()[0] in prompt

    def test_prompt_includes_existing_state_for_context(self):
        existing = ConversationState.model_validate(
            {"person": {"name": "Rahul"}}
        )
        llm = FakeLLM(state_json(company="ABC"))
        engine = ExtractionEngine(llm)
        run(engine, "He is from ABC.", existing=existing)
        assert "Rahul" in llm.prompts[0]
        assert "State extracted so far" in llm.prompts[0]

    def test_prompt_includes_history_turns(self):
        llm = FakeLLM(state_json(name="Rahul"))
        engine = ExtractionEngine(llm)
        run(
            engine,
            "He needs the proposal",
            history=["Rahul called.", "He is from ABC.", "He needs the proposal"],
        )
        assert "- Rahul called." in llm.prompts[0]
        assert "- He is from ABC." in llm.prompts[0]

    def test_build_prompt_standalone_marks_latest(self):
        prompt = build_extraction_prompt("new message")
        assert "Current message to extract:\nnew message" in prompt


class TestFailures:
    def test_invalid_json_raises_extraction_error(self):
        llm = FakeLLM("this is not json at all")
        engine = ExtractionEngine(llm)
        with pytest.raises(ExtractionError):
            run(engine, "hello")

    def test_schema_violation_raises_extraction_error(self):
        llm = FakeLLM(json.dumps({"person": {"name": 123}}))
        engine = ExtractionEngine(llm)
        with pytest.raises(ExtractionError):
            run(engine, "hello")

    def test_code_fenced_json_accepted(self):
        llm = FakeLLM(f"```json\n{state_json(name='Rahul')}\n```")
        engine = ExtractionEngine(llm)
        state = run(engine, "Rahul called")
        assert state.person.name == "Rahul"

    def test_llm_transport_error_wrapped(self):
        class Boom:
            async def complete(self, prompt):
                raise ConnectionError("network down")

        engine = ExtractionEngine(Boom())
        with pytest.raises(ExtractionError):
            run(engine, "hello")


class TestMultiTurn:
    def make(self, *responses):
        return ConversationIntelligence(ExtractionEngine(FakeLLM(*responses)))

    def test_three_turn_accumulation(self):
        intel = self.make(
            state_json(name="Rahul"),
            state_json(company="ABC"),
            state_json(requirement="send proposal", deadline="tomorrow", follow_up=True),
        )
        import asyncio

        asyncio.run(intel.process_turn("Rahul called."))
        asyncio.run(intel.process_turn("He is from ABC."))
        asyncio.run(intel.process_turn("He needs the proposal tomorrow."))
        assert intel.state.person.name == "Rahul"
        assert intel.state.person.company == "ABC"
        assert intel.state.conversation.requirement == "send proposal"
        assert intel.state.conversation.deadline == "tomorrow"
        assert intel.state.action.follow_up is True

    def test_null_second_turn_keeps_first_turn_facts(self):
        intel = self.make(
            state_json(name="Rahul"),
            state_json(),  # model returns nothing new
        )
        import asyncio

        asyncio.run(intel.process_turn("Rahul called."))
        asyncio.run(intel.process_turn("okay thanks"))
        assert intel.state.person.name == "Rahul"

    def test_extraction_failure_keeps_previous_state(self):
        intel = self.make(
            state_json(name="Rahul"),
            "garbage not json",
        )
        import asyncio

        asyncio.run(intel.process_turn("Rahul called."))
        result = asyncio.run(intel.process_turn("and also blah"))
        assert result is not None
        assert result.person.name == "Rahul"
        assert intel.state.person.name == "Rahul"

    def test_history_is_recorded_and_bounded(self):
        intel = self.make(*[state_json() for _ in range(20)])
        import asyncio

        from app.intelligence.extractor import MAX_HISTORY_TURNS

        for i in range(15):
            asyncio.run(intel.process_turn(f"turn {i}"))
        assert len(intel.history) == MAX_HISTORY_TURNS
        assert intel.history[-1] == "turn 14"

    def test_second_turn_prompt_sees_existing_state(self):
        intel = self.make(state_json(name="Rahul"), state_json(company="ABC"))
        import asyncio

        asyncio.run(intel.process_turn("Rahul called."))
        asyncio.run(intel.process_turn("He is from ABC."))
        assert "State extracted so far" in intel.engine._llm.prompts[1]
        assert "Rahul" in intel.engine._llm.prompts[1]
