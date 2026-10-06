"""Date & time safety (§16): natural phrases -> timezone-aware datetimes.

The user's timezone comes from configuration (Settings.timezone), never is
it hardcoded here. Every returned datetime is timezone-aware; naive inputs
to parse_bounds are rejected via parse_when's contract (now must be aware).

Accepted forms (examples):
    today, tomorrow, tonight, next monday, this friday, monday, next week,
    in two hours / in 30 minutes / in 3 days,
    3 PM, 3:30 pm, 15:30, tomorrow 3 PM, next monday 10:00, friday morning,
    2026-01-05, 2026-01-05 15:00, 2026-01-05T15:00:00+05:30 (ISO 8601)

Failures raise ProductivityError("INVALID_TIME") — never a guess (§12).
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.productivity.errors import ProductivityError

_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_DAYPARTS = {"morning": (9, 0), "afternoon": (15, 0), "evening": (18, 0), "night": (21, 0)}
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
    "sixty": 60,
}
_UNITS = {
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours",
    "day": "days", "days": "days",
    "week": "weeks", "weeks": "weeks",
}

_ERR = (
    "I could not understand that date/time. Use forms like 'tomorrow 3 PM', "
    "'next Monday 10:00', 'in two hours', '3:30 PM', or an ISO datetime "
    "like '2026-01-05 15:00'."
)

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([Tt ]\d{2}:\d{2}(:\d{2})?(\.\d+)?([Zz]|[+-]\d{2}:?\d{2})?)?$")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_TIME_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")
_IN_RE = re.compile(
    r"^in\s+(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    + r"|\d+)\s+([a-z]+)$"
)


def _fail() -> "ProductivityError":
    return ProductivityError("INVALID_TIME", _ERR)


def _parse_time_token(token: str) -> time | None:
    m = _TIME_RE.match(token.strip())
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ap = m.group(3)
    if ap:
        if not 1 <= hour <= 12 or minute > 59:
            return None
        hour = hour % 12
        if ap == "pm":
            hour += 12
    else:
        if hour > 23 or minute > 59:
            return None
    return time(hour, minute)


def parse_when(
    text: str,
    tz: ZoneInfo,
    now: datetime,
    *,
    default_hour: int = 9,
    default_minute: int = 0,
) -> datetime:
    """Parse a natural/ISO when-expression into an aware datetime in `tz`."""
    raw = " ".join(str(text or "").split())
    if not raw:
        raise _fail()
    if now.tzinfo is None:
        raise ProductivityError(
            "INVALID_TIME", "Internal clock has no timezone; refusing to guess (§16)."
        )
    now = now.astimezone(tz)
    low = raw.lower()
    default = time(default_hour, default_minute)

    # --- ISO 8601 (with or without offset) -------------------------------
    if _ISO_RE.match(low):
        try:
            if len(low) == 10:  # bare date -> default_hour in the configured tz
                from datetime import date as _date

                d = _date.fromisoformat(low)
                return datetime(d.year, d.month, d.day, default.hour, default.minute, tzinfo=tz)
            text = low
            if text[10:11] == "t":  # fromisoformat wants an uppercase separator
                text = text[:10] + "T" + text[11:]
            if text.endswith("z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
        except ValueError:
            raise _fail() from None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(tz)

    # --- "in <n> <unit>" -------------------------------------------------
    m = _IN_RE.match(low)
    if m:
        amount_raw, unit_raw = m.group(1), m.group(2)
        amount = (
            _NUMBER_WORDS.get(amount_raw)
            if not amount_raw.isdigit()
            else int(amount_raw)
        )
        unit = _UNITS.get(unit_raw.rstrip("s") + "s") or _UNITS.get(unit_raw)
        if amount and unit:
            return now + timedelta(**{unit: amount})
        raise _fail()

    # --- "next week" -----------------------------------------------------
    if low in {"next week", "a week later", "in a week"}:
        return now + timedelta(weeks=1)

    # --- split off a trailing time token ---------------------------------
    #    "tomorrow 3 PM" | "at 15:00" | "10:30" | "next monday 10"
    time_token: str | None = None
    body = low
    for pattern in (
        r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))$",
        r"\bat\s+(\d{1,2}(?::\d{2})?)$",
        r"\b(\d{1,2}:\d{2})$",
        r"\b(\d{1,2})$",
    ):
        m = re.search(pattern, low)
        if m:
            time_token = m.group(1)
            body = (low[: m.start()] + " " + low[m.end():]).strip()
            break

    parsed_time: time | None = None
    if time_token is not None:
        parsed_time = _parse_time_token(time_token)
        if parsed_time is None:
            raise _fail()
    body = re.sub(r"\s+\bat\b", " ", body).strip()  # "tomorrow at 3 PM"

    # --- daypart (morning/afternoon/evening/night) -----------------------
    #    skipped for complete date phrases: 'tonight' contains 'night'.
    daypart: tuple[int, int] | None = None
    _DATE_PHRASES = {
        "tonight", "tonite", "today", "tomorrow", "tmr", "tmrw",
        "day after tomorrow", "next week",
    }
    if body not in _DATE_PHRASES:
        for part, hm in _DAYPARTS.items():
            if part in body:
                daypart = hm
                body = body.replace(part, " ").strip()
                break

    # --- date expressions ------------------------------------------------
    date: datetime | None = None
    if body in {"tonight", "tonite"}:
        date = datetime.combine(now.date(), time(0, 0), tzinfo=tz)
        if parsed_time is None and daypart is None:
            parsed_time = time(20, 0)
    elif body in {"today"}:
        date = datetime.combine(now.date(), time(0, 0), tzinfo=tz)
    elif body in {"tomorrow", "tmr", "tmrw"}:
        date = datetime.combine(now.date() + timedelta(days=1), time(0, 0), tzinfo=tz)
    elif body in {"day after tomorrow"}:
        date = datetime.combine(now.date() + timedelta(days=2), time(0, 0), tzinfo=tz)
    else:
        m = re.match(r"^(?P<qual>this|next)\s+(?P<dow>[a-z]+)$", body)
        bare = re.match(r"^(?P<dow>[a-z]+)$", body)
        if (m and m.group("dow") in _WEEKDAYS) or (bare and bare.group("dow") in _WEEKDAYS):
            if m:
                dow = _WEEKDAYS[m.group("dow")]
                qual = m.group("qual")
            else:
                dow = _WEEKDAYS[bare.group("dow")]  # type: ignore[union-attr]
                qual = "this"
            delta = (dow - now.weekday()) % 7
            if qual == "next" and delta == 0:
                delta = 7
            date = datetime.combine(
                now.date() + timedelta(days=delta), time(0, 0), tzinfo=tz
            )

    if date is None and parsed_time is None:
        raise _fail()

    if date is None:
        # time-only: today if still ahead of us, else tomorrow
        assert parsed_time is not None
        candidate = datetime.combine(now.date(), parsed_time, tzinfo=tz)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    clock = parsed_time or (time(*daypart) if daypart else default)
    return date.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)


def parse_bounds(
    start_text: str | None,
    end_text: str | None,
    tz: ZoneInfo,
    now: datetime,
    *,
    default_days: float = 7.0,
    max_days: float = 31.0,
) -> tuple[datetime, datetime]:
    """Parse an optional [start, end] range with defaults and caps (§21)."""
    if start_text:
        start = parse_when(start_text, tz, now, default_hour=0, default_minute=0)
    else:
        start = now.astimezone(tz)
    if end_text:
        end = parse_when(end_text, tz, now, default_hour=23, default_minute=59)
    else:
        end = start + timedelta(days=default_days)
    if end <= start:
        raise ProductivityError("INVALID_TIME_RANGE", "The end of the range must be after its start.")
    if (end - start) > timedelta(days=max_days):
        raise ProductivityError(
            "INVALID_TIME_RANGE",
            f"That range is too wide — keep it within {int(max_days)} days.",
        )
    return start, end


def iso(dt: datetime) -> str:
    """RFC 3339-ish rendering for Google APIs (keeps the offset)."""
    return dt.isoformat(timespec="seconds")
