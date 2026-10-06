"""Extraction prompt for the Conversation Intelligence component.

Language coverage: English, Hindi, Hinglish (few-shot examples for each).
The no-hallucination rule is stated repeatedly and operationally:
unknown -> null, never guess.
"""

from __future__ import annotations

import json

from app.intelligence.schema import ConversationState

EXTRACTION_SYSTEM_PROMPT = """\
You are the Conversation Intelligence component of Jarvis.

Extract only information explicitly supported by the conversation.

Do not invent names, companies, roles, deadlines, urgency,
requirements, or actions.

Unknown information must be null.

Determine whether a follow-up is explicitly required or clearly
requested by the conversation.

Return only structured data matching the provided schema.
"""

SCHEMA_DESCRIPTION = """\
Output EXACTLY one JSON object with this shape and nothing else:
{
  "person": {
    "name": string | null,
    "company": string | null,
    "role": string | null
  },
  "conversation": {
    "purpose": string | null,
    "requirement": string | null,
    "urgency": string | null,
    "deadline": string | null
  },
  "action": {
    "callback_required": boolean,
    "follow_up": boolean,
    "follow_up_reason": string | null
  }
}

Rules:
- Only include facts explicitly stated in the conversation.
- If a person or company is not named, use null. Never guess.
- Deadlines stay as spoken phrases ("tomorrow", "Friday", "tomorrow morning").
- "urgency" only when urgency is actually stated ("urgent", "ASAP").
- follow_up_reason is a short imperative phrase summarizing what must
  be done next, or null when no follow-up is required.
- The conversation may be English, Hindi, or Hinglish. Understand all
  three; extract meaning, keep names/companies as spoken.
"""

FEW_SHOT_EXAMPLES = """\
Examples:

Input: "Rahul from ABC called about the project. He needs the proposal tomorrow."
Output: {"person":{"name":"Rahul","company":"ABC","role":null},"conversation":{"purpose":"project","requirement":"send proposal","urgency":null,"deadline":"tomorrow"},"action":{"callback_required":false,"follow_up":true,"follow_up_reason":"Send proposal to Rahul"}}

Input: "Someone called about a project."
Output: {"person":{"name":null,"company":null,"role":null},"conversation":{"purpose":"project","requirement":null,"urgency":null,"deadline":null},"action":{"callback_required":false,"follow_up":false,"follow_up_reason":null}}

Input: "Rahul ka phone aaya tha, kal proposal bhejna hai."
Output: {"person":{"name":"Rahul","company":null,"role":null},"conversation":{"purpose":null,"requirement":"send proposal","urgency":null,"deadline":"tomorrow"},"action":{"callback_required":false,"follow_up":true,"follow_up_reason":"Send proposal to Rahul"}}

Input: "Rahul from ABC called, unko proposal kal chahiye."
Output: {"person":{"name":"Rahul","company":"ABC","role":null},"conversation":{"purpose":null,"requirement":"send proposal","urgency":null,"deadline":"tomorrow"},"action":{"callback_required":false,"follow_up":true,"follow_up_reason":"Send proposal to Rahul"}}

Input: "Priya said she will call back, tell her it is urgent."
Output: {"person":{"name":"Priya","company":null,"role":null},"conversation":{"purpose":null,"requirement":"call back Priya","urgency":"urgent","deadline":null},"action":{"callback_required":true,"follow_up":true,"follow_up_reason":"Call Priya back urgently"}}
"""


def build_extraction_prompt(
    latest: str,
    existing: ConversationState | None = None,
    history: list[str] | None = None,
) -> str:
    """Assemble the full extraction prompt.

    Args:
        latest: the newest user utterance.
        existing: state accumulated so far (lets the model resolve
            "he"/"it" against previously known facts).
        history: earlier user turns (may include `latest`).
    """
    sections = [EXTRACTION_SYSTEM_PROMPT, SCHEMA_DESCRIPTION]

    if existing is not None:
        sections.append(
            "State extracted so far from earlier turns:\n"
            + json.dumps(existing.model_dump(), ensure_ascii=False)
        )

    prior = [t for t in (history or []) if t and t.strip() and t != latest]
    if prior:
        sections.append(
            "Earlier turns in this conversation:\n"
            + "\n".join(f"- {t}" for t in prior)
        )

    sections.append(FEW_SHOT_EXAMPLES)
    sections.append(f"Current message to extract:\n{latest}")
    sections.append("Return only the JSON object.")
    return "\n\n".join(sections)
