"""Memory Policy: decides what becomes persistent memory.

Conservative by default (spec section 7):
- contact details / actionable follow-ups / deadlines -> STORE
- temporary conversation, small talk, assumptions       -> IGNORE
- explicit "remember ..."                                -> STORE (auditable)
- explicit "don't remember ..."                          -> suppress this turn
- explicit "forget ..."                                  -> delete flow (manager)
- credentials/secrets                                    -> REJECT, always

Never stores every conversation automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

DecisionKind = Literal["contact", "conversation", "task", "memory"]
DecisionAction = Literal["store", "ignore", "reject"]
DirectiveKind = Literal["none", "remember", "suppress", "forget"]


class StructuredState(Protocol):
    """Duck-typed view of Phase 2's ConversationState (no hard import)."""

    person: object
    conversation: object
    action: object


# ---------------------------------------------------------------- secrets

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+\S{10,}", re.IGNORECASE),
    re.compile(
        r"\b(?:password|passwd|pwd|api[_ ]?key|access[_ ]?token|auth[_ ]?token"
        r"|secret\s+key|client\s+secret)\b\s*(?:is|:|=)\s*\S{4,}",
        re.IGNORECASE,
    ),
]


def looks_like_secret(text: str | None) -> bool:
    """True when the text obviously carries a credential."""
    if not text:
        return False
    return any(p.search(text) for p in SECRET_PATTERNS)


# ------------------------------------------------------------ directives

REMEMBER_RE = re.compile(
    r"\b(?:please\s+)?remember\s+(?:that\s+)?(.+)", re.IGNORECASE | re.DOTALL
)
SUPPRESS_RE = re.compile(
    r"\b(?:don'?t\s+remember|do\s+not\s+remember|don'?t\s+store|don'?t\s+save)\b",
    re.IGNORECASE,
)
FORGET_RE = re.compile(
    r"^\s*(?:please\s+)?forget\s+(?:that\s+)?(.+)", re.IGNORECASE | re.DOTALL
)


@dataclass(frozen=True)
class MemoryDirective:
    kind: DirectiveKind
    content: str | None = None


def parse_directive(turn_text: str) -> MemoryDirective:
    """Classify explicit memory commands in a user turn."""
    text = (turn_text or "").strip()
    if not text:
        return MemoryDirective("none")
    match = FORGET_RE.match(text)
    if match:
        return MemoryDirective("forget", match.group(1).strip())
    if SUPPRESS_RE.search(text):
        return MemoryDirective("suppress")
    match = REMEMBER_RE.search(text)
    if match:
        content = match.group(1).strip().rstrip(".,;:!?").strip()
        if content.lower() in {"", "that", "this"}:
            # "Remember that." refers to the conversation itself;
            # the structured state carries the facts (no free-text content)
            return MemoryDirective("remember", None)
        return MemoryDirective("remember", content)
    return MemoryDirective("none")


# ------------------------------------------------------------- decisions


@dataclass(frozen=True)
class MemoryDecision:
    action: DecisionAction
    kind: DecisionKind | Literal["none"]
    reason: str
    payload: dict | None = None

    @property
    def stores(self) -> bool:
        return self.action == "store"


class MemoryPolicy:
    """Pure function of (state, turn_text) -> decisions. No I/O."""

    def decide(
        self, state: StructuredState | None, turn_text: str
    ) -> tuple[MemoryDirective, list[MemoryDecision]]:
        directive = parse_directive(turn_text)

        if directive.kind == "forget":
            # deletion is the manager's job; nothing new should be stored
            return directive, [
                MemoryDecision("ignore", "none", "forget directive: no stores")
            ]
        if directive.kind == "suppress":
            return directive, [
                MemoryDecision("ignore", "none", "user asked not to remember")
            ]

        decisions: list[MemoryDecision] = []
        forced = directive.kind == "remember"

        if state is None:
            if forced and directive.content:
                decisions.extend(self._general(directive.content, forced=True))
            return directive, decisions

        person, conv, action = state.person, state.conversation, state.action
        name = _s(getattr(person, "name", None))
        company = _s(getattr(person, "company", None))
        role = _s(getattr(person, "role", None))
        purpose = _s(getattr(conv, "purpose", None))
        requirement = _s(getattr(conv, "requirement", None))
        urgency = _s(getattr(conv, "urgency", None))
        deadline = _s(getattr(conv, "deadline", None))
        follow_up = bool(getattr(action, "follow_up", False))
        follow_reason = _s(getattr(action, "follow_up_reason", None))

        # -- contact -----------------------------------------------------
        if name:
            if looks_like_secret(" ".join(x for x in (name, company, role) if x)):
                decisions.append(
                    MemoryDecision("reject", "contact", "looks like a secret")
                )
            else:
                decisions.append(
                    MemoryDecision(
                        "store",
                        "contact",
                        "contact detail explicitly mentioned",
                        {"name": name, "company": company, "role": role},
                    )
                )
        elif forced and (company or role):
            decisions.append(
                MemoryDecision(
                    "ignore",
                    "contact",
                    "explicit remember without a person name is not a contact",
                )
            )

        # -- conversation ------------------------------------------------
        actionable = any([requirement, deadline, urgency, follow_up])
        if actionable:
            decisions.append(
                MemoryDecision(
                    "store",
                    "conversation",
                    "actionable: requirement/deadline/urgency/follow-up present",
                    {
                        "purpose": purpose,
                        "requirement": requirement,
                        "urgency": urgency,
                        "deadline": deadline,
                        "summary": requirement or purpose,
                    },
                )
            )
        elif forced and any([purpose, requirement, deadline, urgency]):
            decisions.append(
                MemoryDecision(
                    "store", "conversation", "explicit remember of conversation facts"
                )
            )
        else:
            decisions.append(
                MemoryDecision(
                    "ignore", "conversation", "no actionable content (temporary chat)"
                )
            )

        # -- task --------------------------------------------------------
        if follow_up:
            title = follow_reason or requirement or "Follow up"
            if looks_like_secret(title):
                decisions.append(
                    MemoryDecision("reject", "task", "looks like a secret")
                )
            else:
                decisions.append(
                    MemoryDecision(
                        "store",
                        "task",
                        "follow-up explicitly required",
                        {"title": title, "deadline": deadline},
                    )
                )

        # -- general (explicit remember only) ----------------------------
        if forced and directive.content:
            decisions.extend(self._general(directive.content, forced=True))
        elif not forced:
            decisions.append(
                MemoryDecision(
                    "ignore",
                    "memory",
                    "general facts stored only on explicit request",
                )
            )

        return directive, decisions

    @staticmethod
    def _general(content: str, forced: bool) -> list[MemoryDecision]:
        if looks_like_secret(content):
            return [
                MemoryDecision(
                    "reject", "memory", "refusing to store a credential", {"content": content}
                )
            ]
        return [
            MemoryDecision(
                "store",
                "memory",
                "explicit remember request" if forced else "",
                {"content": content, "importance": 4 if forced else 3},
            )
        ]


def _s(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
