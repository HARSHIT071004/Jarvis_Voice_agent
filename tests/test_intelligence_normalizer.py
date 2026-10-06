"""Normalizer tests — conservative cleanup only, no meaning changes."""

from app.intelligence.normalizer import normalize_state, normalize_text
from app.intelligence.schema import Action, Conversation, ConversationState, Person


class TestNormalizeText:
    def test_none_stays_none(self):
        assert normalize_text(None) is None

    def test_whitespace_collapsed(self):
        assert normalize_text("  send   proposal\n") == "send proposal"

    def test_unknown_markers_become_none(self):
        for token in ("unknown", "UNKNOWN", "n/a", "none", "null", "", "  ? "):
            assert normalize_text(token) is None, token

    def test_asap_preserved(self):
        assert normalize_text("ASAP") == "ASAP"

    def test_urgent_preserved(self):
        assert normalize_text("urgent") == "urgent"

    def test_tomorrow_preserved(self):
        assert normalize_text("tomorrow") == "tomorrow"

    def test_kal_exact_word_maps_to_tomorrow(self):
        assert normalize_text("kal") == "tomorrow"
        assert normalize_text("Kal") == "tomorrow"

    def test_kal_inside_phrase_untouched(self):
        assert normalize_text("kal proposal") == "kal proposal"

    def test_deadline_phrase_untouched(self):
        assert normalize_text("tomorrow morning") == "tomorrow morning"

    def test_non_string_rejected(self):
        assert normalize_text(123) is None  # type: ignore[arg-type]

    def test_names_untouched(self):
        assert normalize_text("Rahul") == "Rahul"
        assert normalize_text("ABC Corp.") == "ABC Corp."


class TestNormalizeState:
    def test_all_string_fields_normalized(self):
        state = ConversationState(
            person=Person(name="  Rahul ", company="unknown", role=None),
            conversation=Conversation(
                purpose="project",
                requirement="  send   proposal ",
                urgency="n/a",
                deadline="kal",
            ),
            action=Action(follow_up=True, follow_up_reason=" unknown "),
        )
        out = normalize_state(state)
        assert out.person.name == "Rahul"
        assert out.person.company is None
        assert out.conversation.requirement == "send proposal"
        assert out.conversation.urgency is None
        assert out.conversation.deadline == "tomorrow"
        assert out.action.follow_up is True
        assert out.action.follow_up_reason is None

    def test_does_not_mutate_input(self):
        state = ConversationState(person=Person(name=" Rahul "))
        normalize_state(state)
        assert state.person.name == " Rahul "

    def test_defaults_survive(self):
        out = normalize_state(ConversationState())
        assert out == ConversationState()
