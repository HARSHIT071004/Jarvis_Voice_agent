"""Phase 7 §16: date & time safety — natural phrases, boundaries, timezones."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from conftest import FIXED_NOW

from app.productivity.errors import ProductivityError
from app.productivity.timeutil import parse_bounds, parse_when

IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")


def parse(text, now=FIXED_NOW, tz=IST, **kw):
    return parse_when(text, tz, now, **kw)


# ------------------------------------------------------------ basics

def test_tomorrow_with_time():
    dt = parse("tomorrow 3 PM")
    assert dt == datetime(2026, 10, 6, 15, 0, tzinfo=IST)
    assert dt.utcoffset() == timedelta(hours=5, minutes=30)


def test_today_defaults_to_default_hour():
    assert parse("today") == datetime(2026, 10, 5, 9, 0, tzinfo=IST)


def test_tonight_is_evening():
    assert parse("tonight") == datetime(2026, 10, 5, 20, 0, tzinfo=IST)


def test_tonight_with_explicit_time_wins():
    assert parse("tonight 9 PM") == datetime(2026, 10, 5, 21, 0, tzinfo=IST)


def test_next_monday_same_day_means_next_week():
    # now is Monday 2026-10-05 -> "next Monday" must not mean today
    dt = parse("next monday 10:00")
    assert dt == datetime(2026, 10, 12, 10, 0, tzinfo=IST)


def test_this_friday():
    dt = parse("this friday")
    assert dt == datetime(2026, 10, 9, 9, 0, tzinfo=IST)
    assert dt.weekday() == 4


def test_bare_weekday():
    dt = parse("wednesday 2 PM")
    assert dt == datetime(2026, 10, 7, 14, 0, tzinfo=IST)


def test_next_week_keeps_time():
    dt = parse("next week")
    assert dt == FIXED_NOW + timedelta(weeks=1)


def test_in_two_hours():
    assert parse("in two hours") == datetime(2026, 10, 5, 12, 0, tzinfo=IST)


def test_in_thirty_minutes_word_number():
    assert parse("in 30 minutes") == datetime(2026, 10, 5, 10, 30, tzinfo=IST)


def test_in_three_days_crosses_month_safely():
    dt = parse("in 3 days")
    assert dt.date().isoformat() == "2026-10-08"


def test_time_only_future_is_today():
    assert parse("3 PM") == datetime(2026, 10, 5, 15, 0, tzinfo=IST)


def test_time_only_past_rolls_to_tomorrow():
    dt = parse("8:30 AM")  # now is 10:00
    assert dt == datetime(2026, 10, 6, 8, 30, tzinfo=IST)


def test_24h_time():
    assert parse("15:30") == datetime(2026, 10, 5, 15, 30, tzinfo=IST)


def test_at_prefixed_time():
    assert parse("tomorrow at 3 pm") == datetime(2026, 10, 6, 15, 0, tzinfo=IST)


def test_daypart_afternoon():
    assert parse("tomorrow afternoon") == datetime(2026, 10, 6, 15, 0, tzinfo=IST)


# ------------------------------------------------------- ISO & explicit tz

def test_iso_date_only_uses_default_hour():
    assert parse("2026-01-05") == datetime(2026, 1, 5, 9, 0, tzinfo=IST)


def test_iso_datetime_without_offset_assumes_configured_tz():
    dt = parse("2026-01-05 15:00")
    assert dt.tzinfo is not None and dt.hour == 15
    assert dt.utcoffset() == timedelta(hours=5, minutes=30)


def test_iso_datetime_with_offset_converts_to_configured_tz():
    dt = parse("2026-01-05T15:00:00+08:00")
    assert dt.utcoffset() == timedelta(hours=5, minutes=30)
    assert dt.hour == 12  # 15:00 +08:00 == 12:00 +05:30


def test_explicit_timezone_override():
    dt = parse("2026-06-15 09:00", tz=NY)
    assert dt.tzinfo.key == "America/New_York"
    assert dt.utcoffset() == timedelta(hours=-4)  # EDT in June


# -------------------------------------------------------- DST / boundaries

def test_dst_spring_forward_day_is_correct():
    now = datetime(2025, 3, 8, 12, 0, tzinfo=NY)
    dt = parse_when("tomorrow 9 AM", NY, now)
    assert dt.date().isoformat() == "2025-03-09"
    assert dt.hour == 9
    assert dt.utcoffset() == timedelta(hours=-4)  # EDT after the transition


def test_dst_fall_back_day_is_correct():
    now = datetime(2025, 11, 1, 12, 0, tzinfo=NY)
    dt = parse_when("tomorrow 9 AM", NY, now)
    assert dt.date().isoformat() == "2025-11-02"
    assert dt.utcoffset() == timedelta(hours=-5)  # EST after the transition


def test_midnight_rollover_in_two_hours():
    now = datetime(2026, 10, 5, 23, 30, tzinfo=IST)
    dt = parse_when("in two hours", IST, now)
    assert dt == datetime(2026, 10, 6, 1, 30, tzinfo=IST)


def test_month_boundary_tomorrow():
    now = datetime(2026, 12, 31, 10, 0, tzinfo=IST)
    dt = parse_when("tomorrow 3 PM", IST, now)
    assert dt == datetime(2027, 1, 1, 15, 0, tzinfo=IST)


def test_year_boundary_in_two_days():
    now = datetime(2026, 12, 31, 22, 0, tzinfo=IST)
    dt = parse_when("in 2 days", IST, now)
    assert dt.year == 2027 and dt.month == 1 and dt.day == 2


def test_month_boundary_next_monday():
    now = datetime(2026, 10, 30, 10, 0, tzinfo=IST)  # Saturday
    dt = parse_when("next monday", IST, now)
    assert dt.date().isoformat() == "2026-11-02"


# ------------------------------------------------------------- failures

@pytest.mark.parametrize("text", ["banana", "sometime soon", "3:99 PM", "next spoofday", ""])
def test_unparseable_raises_invalid_time(text):
    with pytest.raises(ProductivityError) as exc:
        parse(text)
    assert exc.value.code == "INVALID_TIME"
    assert "tomorrow" in exc.value.message  # helpful, names accepted forms


def test_naive_now_is_refused_not_guessed():
    with pytest.raises(ProductivityError) as exc:
        parse_when("tomorrow", IST, datetime(2026, 10, 5, 10, 0))
    assert exc.value.code == "INVALID_TIME"


# ------------------------------------------------------------ ranges

def test_range_defaults():
    start, end = parse_bounds(None, None, IST, FIXED_NOW, default_days=7.0)
    assert start == FIXED_NOW
    assert end == FIXED_NOW + timedelta(days=7)


def test_range_with_phrases():
    start, end = parse_bounds("tomorrow", "next monday", IST, FIXED_NOW)
    assert start == datetime(2026, 10, 6, 0, 0, tzinfo=IST)
    assert end == datetime(2026, 10, 12, 23, 59, tzinfo=IST)


def test_range_end_before_start_rejected():
    with pytest.raises(ProductivityError) as exc:
        parse_bounds("next monday", "tomorrow", IST, FIXED_NOW)
    assert exc.value.code == "INVALID_TIME_RANGE"


def test_range_too_wide_rejected():
    with pytest.raises(ProductivityError) as exc:
        parse_bounds("2026-10-01", "2027-06-01", IST, FIXED_NOW, max_days=31)
    assert exc.value.code == "INVALID_TIME_RANGE"
