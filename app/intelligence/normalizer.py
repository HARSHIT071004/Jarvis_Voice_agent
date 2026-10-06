"""Conservative normalization of extracted values.

Cleans model output without changing meaning:
- strips/collapses whitespace
- maps explicit "unknown" markers to None (prefer null over garbage)
- maps whole-value Hindi time words to English ONLY when the entire value
  is that word ("kal" -> "tomorrow"); never rewrites phrases
- never resolves dates to calendar timestamps (Phase 3 territory)
"""

from __future__ import annotations

import re

from app.intelligence.schema import ConversationState

UNKNOWN_TOKENS = {"", "unknown", "unk", "n/a", "na", "none", "null", "nil", "?"}

# whole-value-only Hindi -> English time words (conservative, no phrases)
WHOLE_VALUE_TIME_WORDS = {
    "kal": "tomorrow",
    "parso": "day after tomorrow",
    "parson": "day after tomorrow",
    "aaj": "today",
}

_WS = re.compile(r"\s+")


def normalize_text(value: str | None) -> str | None:
    """Normalize a single extracted string. None stays None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = _WS.sub(" ", value).strip()
    if cleaned.lower() in UNKNOWN_TOKENS:
        return None
    if cleaned.lower() in WHOLE_VALUE_TIME_WORDS:
        return WHOLE_VALUE_TIME_WORDS[cleaned.lower()]
    return cleaned


def normalize_state(state: ConversationState) -> ConversationState:
    """Return a copy of `state` with every string field normalized."""
    return ConversationState(
        person={
            "name": normalize_text(state.person.name),
            "company": normalize_text(state.person.company),
            "role": normalize_text(state.person.role),
        },
        conversation={
            "purpose": normalize_text(state.conversation.purpose),
            "requirement": normalize_text(state.conversation.requirement),
            "urgency": normalize_text(state.conversation.urgency),
            "deadline": normalize_text(state.conversation.deadline),
        },
        action={
            "callback_required": state.action.callback_required,
            "follow_up": state.action.follow_up,
            "follow_up_reason": normalize_text(state.action.follow_up_reason),
        },
    )
