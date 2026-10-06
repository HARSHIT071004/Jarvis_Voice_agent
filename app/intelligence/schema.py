"""Structured conversation state for Phase 2 intelligence.

No-hallucination principle: every field defaults to None/False because
"unknown" must be representable. Merging is conservative — later nulls
never erase earlier facts, and contradictions keep the existing value
(first confirmed fact wins; logged, never silently overwritten).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    company: str | None = None
    role: str | None = None


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: str | None = None
    requirement: str | None = None
    urgency: str | None = None
    deadline: str | None = None


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    callback_required: bool = False
    follow_up: bool = False
    follow_up_reason: str | None = None


class ConversationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person: Person = Person()
    conversation: Conversation = Conversation()
    action: Action = Action()

    def merge(self, newer: ConversationState) -> ConversationState:
        """Merge a fresh extraction into this state.

        Rules (documented contradiction policy):
        - None in `newer` never erases a value already present here.
        - Two different non-null values -> keep the existing one (first
          confirmed fact wins); a conflicting bool only flips False -> True.
        - Fields missing here take the value from `newer`.

        Returns a new ConversationState; self is unchanged.
        """
        person = Person(
            name=self.person.name or newer.person.name,
            company=self.person.company or newer.person.company,
            role=self.person.role or newer.person.role,
        )
        conversation = Conversation(
            purpose=self.conversation.purpose or newer.conversation.purpose,
            requirement=self.conversation.requirement or newer.conversation.requirement,
            urgency=self.conversation.urgency or newer.conversation.urgency,
            deadline=self.conversation.deadline or newer.conversation.deadline,
        )
        action = Action(
            callback_required=self.action.callback_required
            or newer.action.callback_required,
            follow_up=self.action.follow_up or newer.action.follow_up,
            follow_up_reason=self.action.follow_up_reason
            or newer.action.follow_up_reason,
        )
        return ConversationState(
            person=person, conversation=conversation, action=action
        )
