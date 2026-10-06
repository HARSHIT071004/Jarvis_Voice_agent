"""Schema validation and state-merging tests (no LLM involved)."""

from app.intelligence.schema import (
    Action,
    Conversation,
    ConversationState,
    Person,
)


def make_state(
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
    return ConversationState(
        person=Person(name=name, company=company, role=role),
        conversation=Conversation(
            purpose=purpose,
            requirement=requirement,
            urgency=urgency,
            deadline=deadline,
        ),
        action=Action(
            callback_required=callback,
            follow_up=follow_up,
            follow_up_reason=follow_up_reason,
        ),
    )


class TestDefaults:
    def test_everything_unknown_by_default(self):
        state = ConversationState()
        assert state.person.name is None
        assert state.person.company is None
        assert state.person.role is None
        assert state.conversation.purpose is None
        assert state.conversation.requirement is None
        assert state.conversation.urgency is None
        assert state.conversation.deadline is None
        assert state.action.callback_required is False
        assert state.action.follow_up is False
        assert state.action.follow_up_reason is None

    def test_partial_state_valid(self):
        state = ConversationState(person=Person(name="Rahul"))
        assert state.person.name == "Rahul"
        assert state.person.company is None

    def test_extra_fields_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ConversationState.model_validate({"person": {"name": "X"}, "bogus": 1})

    def test_wrong_types_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ConversationState.model_validate({"person": {"name": 123}})


class TestMerge:
    def test_null_in_newer_does_not_erase_existing(self):
        existing = make_state(name="Rahul", deadline="tomorrow")
        newer = make_state(company="ABC", name=None, deadline=None)
        merged = existing.merge(newer)
        assert merged.person.name == "Rahul"
        assert merged.person.company == "ABC"
        assert merged.conversation.deadline == "tomorrow"

    def test_fills_missing_from_newer(self):
        existing = make_state(name="Rahul")
        newer = make_state(
            company="ABC", requirement="send proposal", deadline="tomorrow", follow_up=True
        )
        merged = existing.merge(newer)
        assert merged.person.name == "Rahul"
        assert merged.person.company == "ABC"
        assert merged.conversation.requirement == "send proposal"
        assert merged.conversation.deadline == "tomorrow"
        assert merged.action.follow_up is True

    def test_contradiction_keeps_existing_value(self):
        existing = make_state(name="Rahul")
        newer = make_state(name="Rohit")
        merged = existing.merge(newer)
        assert merged.person.name == "Rahul"

    def test_bool_true_is_never_erased_by_false(self):
        existing = make_state(follow_up=True, follow_up_reason="send proposal")
        newer = make_state(follow_up=False, follow_up_reason=None)
        merged = existing.merge(newer)
        assert merged.action.follow_up is True
        assert merged.action.follow_up_reason == "send proposal"

    def test_bool_false_can_become_true(self):
        existing = make_state()
        newer = make_state(callback=True)
        merged = existing.merge(newer)
        assert merged.action.callback_required is True

    def test_merge_is_immutable_on_inputs(self):
        existing = make_state(name="Rahul")
        newer = make_state(company="ABC")
        existing.merge(newer)
        assert existing.person.company is None
        assert newer.person.name is None

    def test_multi_turn_accumulation(self):
        # "Rahul called." -> "He is from ABC." -> "He needs the proposal tomorrow."
        s1 = make_state(name="Rahul")
        s2 = make_state(company="ABC")
        s3 = make_state(requirement="send proposal", deadline="tomorrow", follow_up=True)
        merged = s1.merge(s2).merge(s3)
        assert merged.person.name == "Rahul"
        assert merged.person.company == "ABC"
        assert merged.conversation.requirement == "send proposal"
        assert merged.conversation.deadline == "tomorrow"
        assert merged.action.follow_up is True
