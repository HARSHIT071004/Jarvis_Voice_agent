"""Memory Policy tests — what gets stored, ignored, rejected."""

from types import SimpleNamespace

from app.intelligence.schema import ConversationState
from app.memory.policy import (
    MemoryPolicy,
    looks_like_secret,
    parse_directive,
)


def make_state(
    name=None,
    company=None,
    role=None,
    purpose=None,
    requirement=None,
    urgency=None,
    deadline=None,
    follow_up=False,
    follow_up_reason=None,
):
    return ConversationState.model_validate(
        {
            "person": {"name": name, "company": company, "role": role},
            "conversation": {
                "purpose": purpose,
                "requirement": requirement,
                "urgency": urgency,
                "deadline": deadline,
            },
            "action": {
                "follow_up": follow_up,
                "follow_up_reason": follow_up_reason,
            },
        }
    )


def stored_kinds(decisions):
    return sorted(d.kind for d in decisions if d.stores)


class TestSecrets:
    def test_obvious_credentials_detected(self):
        assert looks_like_secret("sk-abc123def456ghi789")
        assert looks_like_secret("my API key: AIzaSyA1234567890abcdefghij")
        assert looks_like_secret("password = hunter2secret")
        assert looks_like_secret("Bearer abcdefghijklmnop")
        assert looks_like_secret("ghp_abcdefghijklmnop123456")

    def test_normal_text_not_flagged(self):
        assert not looks_like_secret("Rahul from ABC called about the project")
        assert not looks_like_secret("the password to success is patience")
        assert not looks_like_secret(None)


class TestDirectives:
    def test_remember_english(self):
        d = parse_directive("Remember that Rahul is the project manager at ABC.")
        assert d.kind == "remember"
        assert d.content.startswith("Rahul is the project manager")

    def test_remember_mid_sentence(self):
        d = parse_directive("ok note this: remember that I prefer Python.")
        assert d.kind == "remember"
        assert "prefer Python" in d.content

    def test_dont_remember_suppresses(self):
        assert parse_directive("Don't remember that.").kind == "suppress"
        assert parse_directive("do not remember this").kind == "suppress"

    def test_forget(self):
        d = parse_directive("Forget the fact that Rahul works at ABC.")
        assert d.kind == "forget"
        assert "Rahul works at ABC" in d.content

    def test_forget_anchored_not_matched_inside_sentence(self):
        # "don't forget" is not a delete command
        assert parse_directive("don't forget about the meeting").kind == "none"

    def test_plain_turn_is_none(self):
        assert parse_directive("Rahul from ABC called.").kind == "none"
        assert parse_directive("").kind == "none"


class TestAutomaticPolicy:
    def setUp(self):
        self.policy = MemoryPolicy()

    def test_contact_stored_when_name_present(self):
        _, d = MemoryPolicy().decide(
            make_state(name="Rahul", company="ABC"), "Rahul from ABC called"
        )
        contact = [x for x in d if x.kind == "contact" and x.stores]
        assert len(contact) == 1
        assert contact[0].payload == {"name": "Rahul", "company": "ABC", "role": None}

    def test_actionable_conversation_stored(self):
        _, d = MemoryPolicy().decide(
            make_state(
                name="Rahul", requirement="send proposal", deadline="tomorrow", follow_up=True
            ),
            "Rahul needs the proposal tomorrow",
        )
        assert "conversation" in stored_kinds(d)
        assert "task" in stored_kinds(d)

    def test_task_title_from_follow_up_reason(self):
        _, d = MemoryPolicy().decide(
            make_state(follow_up=True, follow_up_reason="Send proposal to Rahul"),
            "anything",
        )
        task = next(x for x in d if x.kind == "task")
        assert task.payload["title"] == "Send proposal to Rahul"
        assert task.payload["deadline"] is None

    def test_temporary_conversation_ignored(self):
        _, d = MemoryPolicy().decide(
            make_state(purpose="project"), "Someone called about a project"
        )
        assert stored_kinds(d) == []
        reasons = [x.reason for x in d if x.kind == "conversation"]
        assert any("temporary" in r for r in reasons)

    def test_small_talk_stores_nothing(self):
        _, d = MemoryPolicy().decide(
            make_state(), "hello how are you doing today"
        )
        assert stored_kinds(d) == []

    def test_no_general_memory_without_explicit_request(self):
        _, d = MemoryPolicy().decide(
            make_state(name="Rahul"), "I like Python by the way"
        )
        assert "memory" not in stored_kinds(d)


class TestExplicitFlows:
    def test_remember_stores_general_memory(self):
        directive, d = MemoryPolicy().decide(None, "Remember that I prefer Python.")
        assert directive.kind == "remember"
        memory = next(x for x in d if x.kind == "memory" and x.stores)
        assert "prefer Python" in memory.payload["content"]

    def test_remember_alongside_state_stores_contact_too(self):
        _, d = MemoryPolicy().decide(
            make_state(name="Rahul", company="ABC"),
            "Remember that Rahul is the project manager at ABC.",
        )
        assert "contact" in stored_kinds(d)
        assert "memory" in stored_kinds(d)

    def test_suppress_prevents_all_stores(self):
        _, d = MemoryPolicy().decide(
            make_state(name="Rahul", requirement="x", follow_up=True),
            "Don't remember that.",
        )
        assert stored_kinds(d) == []

    def test_forget_creates_no_stores(self):
        directive, d = MemoryPolicy().decide(
            make_state(name="Rahul"), "Forget Rahul."
        )
        assert directive.kind == "forget"
        assert stored_kinds(d) == []

    def test_remember_with_secret_rejected(self):
        _, d = MemoryPolicy().decide(None, "Remember my password is hunter2secret")
        memory = next(x for x in d if x.kind == "memory")
        assert memory.action == "reject"

    def test_state_none_with_remember(self):
        _, d = MemoryPolicy().decide(None, "Remember Jarvis is my assistant")
        assert stored_kinds(d) == ["memory"]
