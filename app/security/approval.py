"""ApprovalManager — human approval with binding, expiry and one-shot
consumption (Phase 8 §8/§16/§19/§20/§21).

Approvals are explicit, time-limited, session-scoped, fingerprint-bound
to the exact action (tool + arguments), and consumed on use — a second
run needs a fresh approval. Decisions come only from trusted sources: an
injected decision handler, or ``record_decision`` called by trusted code
after the human actually answered. Model text and external content can
never create or resolve an approval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.security.models import (
    ActionContext,
    ApprovalRecord,
    ApprovalStatus,
    new_action_id,
    new_approval_id,
)

logger = logging.getLogger("jarvis.security")


class Verdict(str):
    """Result of an approval resolution (plain str for easy comparison)."""

    APPROVED = "APPROVED"
    DENIED = "DENIED"
    PENDING = "PENDING"
    EXPIRED = "EXPIRED"


@dataclass
class DecisionRequest:
    """Structured prompt handed to a trusted decision source (§16)."""

    action_id: str
    session_id: str
    tool: str
    target: str
    risk: str
    summary: dict


DecisionSource = Callable[[DecisionRequest], Awaitable[bool | None]]

# A clean, unambiguous yes in English / Hinglish. Anything else is NOT
# an approval (the LLM must never be able to grant itself permission).
_APPROVE_RE = re.compile(
    r"^\W*(?:yes|yeah|yep|yup|haan|han|ji|ok|okay|sure|"
    r"yes please|sure thing|go ahead|do it|please do|send it|send|"
    r"confirm|confirmed|approve|approved|allow|allowed|"
    r"karo|bhej do|bhej dijiye|haan bhej do|"
    r"yes[, ]+(?:please|go ahead|do it|send it|send|confirm(?:ed)?|approve(?:d)?))"
    r"\W*[.!?]*$",
    re.I,
)

_DENY_RE = re.compile(
    r"^\W*(?:no|nope|nah|nahi|nhi|mat karo|nahi karo|don'?t|do not|never|"
    r"stop|cancel|deny|denied|refuse|refused|not now|not today|no thanks|"
    r"nahi chahiye)\b",
    re.I,
)


def parse_utterance(text: str) -> bool | None:
    """Trusted voice/text answer -> True (approve), False (deny), None.

    Only a clean yes counts as approval; a leading no counts as denial;
    everything else leaves the approval untouched (fail closed).
    """
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    if _APPROVE_RE.match(cleaned):
        return True
    if _DENY_RE.match(cleaned):
        return False
    return None


def _canon(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))


class ApprovalManager:
    def __init__(
        self,
        ttl_seconds: float = 120.0,
        decision_source: DecisionSource | None = None,
        session_id: str = "default",
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("approval ttl must be positive")
        self.ttl = float(ttl_seconds)
        self.decision_source = decision_source
        self.session_id = session_id
        self._clock = clock
        self._records: dict[str, ApprovalRecord] = {}

    # ------------------------------------------------------------- identity

    @staticmethod
    def fingerprint(session_id: str, *parts: object) -> str:
        payload = json.dumps(
            [session_id, *parts], sort_keys=True, default=str, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------- resolve

    async def resolve_action(
        self, ctx: ActionContext, summary: dict | None = None
    ) -> tuple[str, ApprovalRecord]:
        fingerprint = self.fingerprint(
            ctx.session_id, "tool", ctx.tool, _canon(ctx.args)
        )
        return await self._resolve(
            session_id=ctx.session_id,
            fingerprint=fingerprint,
            tool=ctx.tool,
            risk=getattr(ctx.risk, "value", str(ctx.risk)),
            summary=summary or {},
            action_id=ctx.action_id,
        )

    async def resolve_request(
        self,
        session_id: str,
        operation: str,
        target: str = "",
        detail: str = "",
        risk: str = "MEDIUM",
        summary: dict | None = None,
        action_id: str = "",
    ) -> tuple[str, ApprovalRecord]:
        """Side channel for inner domain policies (browser sensitive actions)."""
        fingerprint = self.fingerprint(session_id, "op", operation, target, detail)
        return await self._resolve(
            session_id=session_id,
            fingerprint=fingerprint,
            tool=operation,
            risk=risk,
            summary=summary if summary is not None else {"target": target},
            action_id=action_id or new_action_id(self._clock()),
        )

    async def _resolve(
        self,
        *,
        session_id: str,
        fingerprint: str,
        tool: str,
        risk: str,
        summary: dict,
        action_id: str,
    ) -> tuple[str, ApprovalRecord]:
        now = self._clock()
        records = sorted(
            (
                r
                for r in self._records.values()
                if r.session_id == session_id and r.fingerprint == fingerprint
            ),
            key=lambda r: r.requested_at,
        )
        latest = records[-1] if records else None
        if latest is not None:
            if latest.status is ApprovalStatus.APPROVED and not latest.consumed:
                if latest.is_past_expiry(now):
                    latest.status = ApprovalStatus.EXPIRED
                    logger.info("[SECURITY] Approval expired id=%s", latest.approval_id)
                    return Verdict.EXPIRED, latest
                latest.consumed = True
                return Verdict.APPROVED, latest
            if latest.status is ApprovalStatus.DENIED and not latest.is_past_expiry(now):
                return Verdict.DENIED, latest
            if latest.status is ApprovalStatus.PENDING and not latest.is_past_expiry(now):
                return Verdict.PENDING, latest

        decision: bool | None = None
        if self.decision_source is not None:
            try:
                decision = await self.decision_source(
                    DecisionRequest(
                        action_id=action_id,
                        session_id=session_id,
                        tool=tool,
                        target=str(summary.get("target", "")),
                        risk=risk,
                        summary=dict(summary),
                    )
                )
            except Exception:
                logger.warning("[SECURITY] Decision source failed; approval stays closed")
                decision = None

        if decision is True:
            record = self._create(
                session_id=session_id,
                fingerprint=fingerprint,
                tool=tool,
                risk=risk,
                summary=summary,
                action_id=action_id,
                now=now,
                status=ApprovalStatus.APPROVED,
                decided_by="decision_source",
                consumed=True,
            )
            return Verdict.APPROVED, record
        if decision is False:
            record = self._create(
                session_id=session_id,
                fingerprint=fingerprint,
                tool=tool,
                risk=risk,
                summary=summary,
                action_id=action_id,
                now=now,
                status=ApprovalStatus.DENIED,
                decided_by="decision_source",
            )
            return Verdict.DENIED, record
        record = self._create(
            session_id=session_id,
            fingerprint=fingerprint,
            tool=tool,
            risk=risk,
            summary=summary,
            action_id=action_id,
            now=now,
            status=ApprovalStatus.PENDING,
        )
        return Verdict.PENDING, record

    def _create(
        self,
        *,
        session_id: str,
        fingerprint: str,
        tool: str,
        risk: str,
        summary: dict,
        action_id: str,
        now: float,
        status: ApprovalStatus,
        decided_by: str = "",
        consumed: bool = False,
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=new_approval_id(),
            action_id=action_id,
            session_id=session_id,
            tool=tool,
            risk=risk,
            fingerprint=fingerprint,
            requested_at=now,
            expires_at=now + self.ttl,
            status=status,
            summary=dict(summary),
            decided_by=decided_by,
            decided_at=now if decided_by else None,
            consumed=consumed,
        )
        self._records[record.approval_id] = record
        return record

    # ------------------------------------------------------ trusted decisions

    def record_decision(
        self,
        approval_id: str | None = None,
        *,
        approved: bool,
        decided_by: str = "user",
        session_id: str | None = None,
    ) -> ApprovalRecord | None:
        """Record a human decision on the newest pending approval.

        Called only by trusted code (voice/text front door, UI, tests) —
        never from model-generated text.
        """
        session = session_id or self.session_id
        now = self._clock()
        pending = sorted(
            (
                r
                for r in self._records.values()
                if r.session_id == session
                and r.status is ApprovalStatus.PENDING
                and not r.is_past_expiry(now)
                and (approval_id is None or r.approval_id == approval_id)
            ),
            key=lambda r: r.requested_at,
        )
        if not pending:
            return None
        record = pending[-1]
        record.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        record.decided_by = decided_by
        record.decided_at = now
        return record

    def resolve_utterance(
        self, text: str, session_id: str | None = None, decided_by: str = "user"
    ) -> str | None:
        """Map a trusted human utterance onto the newest pending approval."""
        session = session_id or self.session_id
        now = self._clock()
        has_pending = any(
            r.session_id == session
            and r.status is ApprovalStatus.PENDING
            and not r.is_past_expiry(now)
            for r in self._records.values()
        )
        if not has_pending:
            return None
        decision = parse_utterance(text)
        if decision is None:
            return None
        record = self.record_decision(
            approved=decision, decided_by=decided_by, session_id=session
        )
        if record is None:
            return None
        return "approved" if decision else "denied"

    def cancel_session(self, session_id: str) -> int:
        count = 0
        for record in self._records.values():
            if record.session_id == session_id and record.status is ApprovalStatus.PENDING:
                record.status = ApprovalStatus.CANCELLED
                count += 1
        return count

    # --------------------------------------------------------------- queries

    def now(self) -> float:
        return self._clock()

    def pending(self, session_id: str | None = None) -> list[ApprovalRecord]:
        session = session_id or self.session_id
        now = self._clock()
        return sorted(
            (
                r
                for r in self._records.values()
                if r.session_id == session
                and r.status is ApprovalStatus.PENDING
                and not r.is_past_expiry(now)
            ),
            key=lambda r: r.requested_at,
        )

    def all_records(self) -> list[ApprovalRecord]:
        return sorted(self._records.values(), key=lambda r: r.requested_at)


def approval_prompt(record: ApprovalRecord) -> str:
    """Trusted, structured approval summary shown to the model/user (§16).

    Built only from action metadata — untrusted email/webpage text can
    never shape it.
    """
    lines = ["Jarvis wants to perform:", f"Action: {record.tool}"]
    for key, label in (
        ("target", "Target"),
        ("recipient_count", "Recipients"),
        ("subject", "Subject"),
        ("url", "URL"),
        ("event_id", "Event"),
    ):
        value = record.summary.get(key)
        if value not in (None, "", 0):
            lines.append(f"{label}: {value}")
    lines.append(f"Risk: {record.risk}")
    lines.append("Approve? (the user must answer)")
    return "\n".join(lines)
