"""Calendar provider abstraction (§11): tools talk to CalendarProvider.
GoogleCalendarProvider speaks the official Google Calendar REST API over
the shared GoogleAPI layer.

All datetimes are explicit-timezone (§16): callers pass aware datetimes,
we serialize them with their offset and echo the IANA timezone to Google.
Availability comes from Google's freebusy data — never invented (§12).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

from app.productivity.errors import APIError, ProductivityError
from app.productivity.google_api import GoogleAPI
from app.productivity.oauth import CALENDAR_EVENTS, CALENDAR_READONLY
from app.productivity.timeutil import iso

logger = logging.getLogger("jarvis.calendar")

_BASE = "https://calendar.googleapis.com/calendar/v3"
_MAX_DESCRIPTION = 500
_MAX_ATTENDEES_IN_RESULT = 10


class CalendarProvider(ABC):
    @abstractmethod
    async def list_events(
        self, start: datetime, end: datetime, max_results: int, timezone: str
    ) -> dict: ...

    @abstractmethod
    async def get_event(self, event_id: str) -> dict: ...

    @abstractmethod
    async def availability(self, start: datetime, end: datetime, timezone: str) -> dict: ...

    @abstractmethod
    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        timezone: str,
        *,
        attendees: list[str] | None = None,
        location: str = "",
        description: str = "",
        check_conflicts: bool = True,
    ) -> dict: ...

    @abstractmethod
    async def update_event(self, event_id: str, fields: dict, timezone: str) -> dict: ...

    @abstractmethod
    async def delete_event(self, event_id: str) -> dict: ...


def _event_summary(raw: dict) -> dict:
    start = raw.get("start") or {}
    end = raw.get("end") or {}
    attendees = [
        {
            "email": str(a.get("email", ""))[:200],
            "response_status": str(a.get("responseStatus", ""))[:40],
        }
        for a in (raw.get("attendees") or [])[:_MAX_ATTENDEES_IN_RESULT]
    ]
    return {
        "id": str(raw.get("id", ""))[:128],
        "title": str(raw.get("summary", ""))[:300],
        "start": str(start.get("dateTime") or start.get("date") or ""),
        "end": str(end.get("dateTime") or end.get("date") or ""),
        "location": str(raw.get("location", ""))[:200],
        "description": str(raw.get("description", ""))[:_MAX_DESCRIPTION],
        "status": str(raw.get("status", ""))[:40],
        "attendees": attendees,
        "attendee_count": len(raw.get("attendees") or []),
    }


def _when_payload(dt: datetime, timezone: str) -> dict:
    """Google dateTime carries the offset; timeZone disambiguates DST (§16)."""
    return {"dateTime": iso(dt), "timeZone": timezone}


def _map(exc: APIError, not_found_code: str, not_found_msg: str) -> ProductivityError:
    if exc.status == 404:
        return ProductivityError(not_found_code, not_found_msg)
    if exc.status == 410:  # deleted event still queryable
        return ProductivityError(not_found_code, not_found_msg)
    if exc.status == 400:
        return ProductivityError(
            "GOOGLE_API_ERROR", f"Calendar rejected the request: {exc.reason or 'invalid input'}."
        )
    if exc.status == 409:
        return ProductivityError(
            "GOOGLE_API_ERROR", "Calendar reported a conflicting update; re-read the event."
        )
    if exc.status == 429:
        return ProductivityError(
            "API_RATE_LIMITED", "Calendar is rate-limiting requests — try again shortly."
        )
    return ProductivityError("GOOGLE_API_ERROR", f"Calendar request failed ({exc.status}).")


class GoogleCalendarProvider(CalendarProvider):
    def __init__(self, api: GoogleAPI) -> None:
        self._api = api

    async def list_events(
        self, start: datetime, end: datetime, max_results: int, timezone: str
    ) -> dict:
        limit = max(1, min(int(max_results), 100))
        params = [
            ("timeMin", iso(start)),
            ("timeMax", iso(end)),
            ("singleEvents", "true"),
            ("orderBy", "startTime"),
            ("maxResults", str(limit)),
            ("timeZone", timezone),
            ("showDeleted", "false"),
        ]
        try:
            page = await self._api.get(
                f"{_BASE}/calendars/primary/events", scope=CALENDAR_READONLY, params=params
            )
        except APIError as exc:
            raise _map(exc, "CALENDAR_EVENT_NOT_FOUND", "Calendar events were not found.") from exc
        items = [
            _event_summary(e)
            for e in (page.get("items") or [])
            if str(e.get("status", "")) != "cancelled"
        ][:limit]
        logger.info("[CALENDAR] list count=%d", len(items))
        return {
            "events": items,
            "count": len(items),
            "start": iso(start),
            "end": iso(end),
            "timezone": timezone,
        }

    async def get_event(self, event_id: str) -> dict:
        try:
            raw = await self._api.get(
                f"{_BASE}/calendars/primary/events/{event_id}", scope=CALENDAR_READONLY
            )
        except APIError as exc:
            raise _map(exc, "CALENDAR_EVENT_NOT_FOUND", f"No event with id '{event_id}'.") from exc
        logger.info("[CALENDAR] get id=%s", event_id[:64])
        return _event_summary(raw)

    async def availability(self, start: datetime, end: datetime, timezone: str) -> dict:
        try:
            body = await self._api.post(
                f"{_BASE}/freebusy",
                scope=CALENDAR_READONLY,
                json={
                    "timeMin": iso(start),
                    "timeMax": iso(end),
                    "timeZone": timezone,
                    "items": [{"id": "primary"}],
                },
            )
        except APIError as exc:
            raise _map(exc, "CALENDAR_EVENT_NOT_FOUND", "Availability could not be read.") from exc
        busy_raw = ((body.get("calendars") or {}).get("primary") or {}).get("busy") or []
        busy = [
            {"start": str(b.get("start", "")), "end": str(b.get("end", ""))}
            for b in busy_raw
            if b.get("start") and b.get("end")
        ]
        free = _complement(start, end, busy)
        logger.info("[CALENDAR] availability busy=%d free=%d", len(busy), len(free))
        return {
            "start": iso(start),
            "end": iso(end),
            "timezone": timezone,
            "busy": busy,
            "free": free,
        }

    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        timezone: str,
        *,
        attendees: list[str] | None = None,
        location: str = "",
        description: str = "",
        check_conflicts: bool = True,
    ) -> dict:
        conflicts: list[dict] = []
        if check_conflicts:
            try:
                state = await self.availability(start, end, timezone)
                conflicts = state["busy"]
            except ProductivityError:
                conflicts = []  # conflict check is advisory; never blocks the create
        payload: dict = {
            "summary": title,
            "start": _when_payload(start, timezone),
            "end": _when_payload(end, timezone),
        }
        if location:
            payload["location"] = location
        if description:
            payload["description"] = description[:2000]
        if attendees:
            payload["attendees"] = [{"email": a} for a in attendees]
        try:
            raw = await self._api.post(
                f"{_BASE}/calendars/primary/events",
                scope=CALENDAR_EVENTS,
                json=payload,
            )
        except APIError as exc:
            if exc.status == 400 and any(
                k in exc.reason.lower() for k in ("attendee", "recipient")
            ):
                raise ProductivityError(
                    "INVALID_RECIPIENT", "Calendar rejected an attendee address."
                ) from exc
            raise _map(exc, "CALENDAR_EVENT_NOT_FOUND", "The event was not found.") from exc
        logger.info("[CALENDAR] created id=%s conflicts=%d", str(raw.get("id", ""))[:64], len(conflicts))
        summary = _event_summary(raw)
        summary["conflicts"] = conflicts
        summary["status"] = "created"
        return summary

    async def update_event(self, event_id: str, fields: dict, timezone: str) -> dict:
        payload: dict = {}
        if "title" in fields:
            payload["summary"] = fields["title"]
        if "location" in fields:
            payload["location"] = fields["location"]
        if "description" in fields:
            payload["description"] = fields["description"]
        if "start" in fields:
            payload["start"] = _when_payload(fields["start"], timezone)
        if "end" in fields:
            payload["end"] = _when_payload(fields["end"], timezone)
        if "attendees" in fields:
            payload["attendees"] = [{"email": a} for a in fields["attendees"]]
        try:
            raw = await self._api.patch(
                f"{_BASE}/calendars/primary/events/{event_id}",
                scope=CALENDAR_EVENTS,
                json=payload,
            )
        except APIError as exc:
            raise _map(exc, "CALENDAR_EVENT_NOT_FOUND", f"No event with id '{event_id}'.") from exc
        logger.info("[CALENDAR] updated id=%s fields=%d", event_id[:64], len(payload))
        summary = _event_summary(raw)
        summary["status"] = "updated"
        return summary

    async def delete_event(self, event_id: str) -> dict:
        try:
            await self._api.delete(
                f"{_BASE}/calendars/primary/events/{event_id}", scope=CALENDAR_EVENTS
            )
        except APIError as exc:
            raise _map(exc, "CALENDAR_EVENT_NOT_FOUND", f"No event with id '{event_id}'.") from exc
        logger.info("[CALENDAR] deleted id=%s", event_id[:64])
        return {"event_id": event_id, "status": "deleted"}


def _complement(start: datetime, end: datetime, busy: list[dict]) -> list[dict]:
    """Free windows = [start, end] minus merged busy intervals."""
    from datetime import timezone as _tz

    def parse(value: str) -> datetime:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz.utc)
        return dt

    intervals: list[tuple[datetime, datetime]] = []
    for b in busy:
        try:
            s, e = parse(b["start"]), parse(b["end"])
        except (KeyError, ValueError):
            continue
        if e > s:
            intervals.append((s, e))
    intervals.sort()
    merged: list[tuple[datetime, datetime]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    free: list[dict] = []
    cursor = start
    for s, e in merged:
        if s > cursor:
            free.append({"start": iso(cursor), "end": iso(min(s, end))})
        cursor = max(cursor, e)
        if cursor >= end:
            break
    if cursor < end:
        free.append({"start": iso(cursor), "end": iso(end)})
    return [
        w for w in free
        if datetime.fromisoformat(w["end"]) > datetime.fromisoformat(w["start"])
    ]
