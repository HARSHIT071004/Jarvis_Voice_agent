"""Phase 7 §11-16: Calendar provider over mocked REST — freebusy, tz, mapping."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
from conftest import FIXED_NOW, run

from app.productivity.calendar import GoogleCalendarProvider, _complement
from app.productivity.errors import ProductivityError
from app.productivity.google_api import GoogleAPI
from app.productivity.oauth import CALENDAR_EVENTS, CALENDAR_READONLY
from app.productivity.timeutil import iso

IST = ZoneInfo("Asia/Kolkata")

BASE = "/calendar/v3"
EVENTS = f"{BASE}/calendars/primary/events"
FREEBUSY = f"{BASE}/freebusy"


class FakeOAuth:
    def __init__(self) -> None:
        self.scopes: list[str | None] = []

    async def get_access_token(self, required_scope: str | None = None) -> str:
        self.scopes.append(required_scope)
        return "tok-SECRET"


def ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body)


def err(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message}})


def gcal_event(eid="ev1", summary="Team sync",
               start="2026-10-06T11:00:00+05:30", end="2026-10-06T12:00:00+05:30",
               status="confirmed", attendees=(), location="", description="") -> dict:
    return {
        "id": eid, "summary": summary,
        "start": {"dateTime": start, "timeZone": "Asia/Kolkata"},
        "end": {"dateTime": end, "timeZone": "Asia/Kolkata"},
        "status": status, "location": location, "description": description,
        "attendees": list(attendees),
    }


def route_handler(routes: dict, calls: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        value = routes.get((request.method, request.url.path))
        if value is None:
            return err(404, "Not Found")
        if isinstance(value, list):
            item = value.pop(0) if len(value) > 1 else value[0]
            return item if isinstance(item, httpx.Response) else item(request)
        if isinstance(value, httpx.Response):
            return value
        return value(request)

    return handler


@pytest.fixture
def cal_factory():
    made: list[GoogleAPI] = []

    def _make(routes: dict):
        calls: list = []
        oauth = FakeOAuth()
        api = GoogleAPI(
            oauth, transport=httpx.MockTransport(route_handler(routes, calls)),
            retry_delay=0.0,
        )
        made.append(api)
        return GoogleCalendarProvider(api), oauth, calls

    yield _make

    async def _close():
        for api in made:
            await api.aclose()

    run(_close())


def busy_response(*windows: tuple[str, str]) -> httpx.Response:
    return ok({"calendars": {"primary": {"busy": [{"start": s, "end": e} for s, e in windows]}}})


START = datetime(2026, 10, 6, 9, 0, tzinfo=IST)
END = datetime(2026, 10, 6, 17, 0, tzinfo=IST)


# ------------------------------------------------------------------- list

def test_list_maps_events_and_skips_cancelled(cal_factory):
    provider, oauth, calls = cal_factory({
        ("GET", EVENTS): ok({"items": [
            gcal_event(),
            gcal_event(eid="ev2", summary="Old", status="cancelled"),
            gcal_event(eid="ev3", summary="1:1", start="2026-10-06T14:00:00+05:30",
                       end="2026-10-06T14:30:00+05:30"),
        ]}),
    })
    out = run(provider.list_events(START, END, 25, "Asia/Kolkata"))
    assert out["count"] == 2
    assert [e["title"] for e in out["events"]] == ["Team sync", "1:1"]
    assert out["events"][0]["start"] == "2026-10-06T11:00:00+05:30"
    assert out["timezone"] == "Asia/Kolkata"
    assert out["start"] == iso(START) and out["end"] == iso(END)
    assert oauth.scopes == [CALENDAR_READONLY]
    params = {k: v for k, v in calls[0].url.params.items()}
    assert params["timeMin"] == iso(START) and params["timeMax"] == iso(END)
    assert params["timeZone"] == "Asia/Kolkata"
    assert params["singleEvents"] == "true"
    assert params["showDeleted"] == "false"


def test_list_empty_calendar(cal_factory):
    provider, _, _ = cal_factory({("GET", EVENTS): ok({"items": []})})
    out = run(provider.list_events(START, END, 25, "Asia/Kolkata"))
    assert out["count"] == 0 and out["events"] == []


def test_list_caps_max_results(cal_factory):
    provider, _, calls = cal_factory({
        ("GET", EVENTS): ok({"items": [gcal_event(eid=f"e{i}") for i in range(5)]}),
    })
    out = run(provider.list_events(START, END, 3, "Asia/Kolkata"))
    assert out["count"] == 3
    assert calls[0].url.params["maxResults"] == "3"


def test_list_persistent_5xx_is_controlled(cal_factory):
    provider, _, _ = cal_factory({("GET", EVENTS): err(500, "Backend")})
    with pytest.raises(ProductivityError) as exc:
        run(provider.list_events(START, END, 10, "Asia/Kolkata"))
    assert exc.value.code == "GOOGLE_API_ERROR"


def test_list_rate_limited(cal_factory):
    provider, _, _ = cal_factory({
        ("GET", EVENTS): httpx.Response(429, json={"error": {"code": 429, "message": "Rate"}}),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.list_events(START, END, 10, "Asia/Kolkata"))
    assert exc.value.code == "API_RATE_LIMITED"


# -------------------------------------------------------------------- get

def test_get_maps_event_and_caps_displayed_attendees(cal_factory):
    attendees = [{"email": f"p{i}@example.com", "responseStatus": "accepted"} for i in range(12)]
    provider, _, _ = cal_factory({
        ("GET", f"{EVENTS}/ev1"): ok(gcal_event(attendees=attendees)),
    })
    out = run(provider.get_event("ev1"))
    assert out["title"] == "Team sync"
    assert out["status"] == "confirmed"
    assert len(out["attendees"]) == 10  # display cap
    assert out["attendee_count"] == 12  # truth kept


def test_get_missing_event(cal_factory):
    provider, _, _ = cal_factory({("GET", f"{EVENTS}/gone"): err(404, "Not Found")})
    with pytest.raises(ProductivityError) as exc:
        run(provider.get_event("gone"))
    assert exc.value.code == "CALENDAR_EVENT_NOT_FOUND"
    assert "gone" in exc.value.message


# ---------------------------------------------------------- availability

def test_availability_computes_free_windows(cal_factory):
    provider, oauth, calls = cal_factory({
        ("POST", FREEBUSY): busy_response(
            ("2026-10-06T10:00:00+05:30", "2026-10-06T11:00:00+05:30"),
            ("2026-10-06T13:00:00+05:30", "2026-10-06T14:00:00+05:30"),
        ),
    })
    out = run(provider.availability(START, END, "Asia/Kolkata"))
    free = [(datetime.fromisoformat(w["start"]), datetime.fromisoformat(w["end"]))
            for w in out["free"]]
    assert free == [
        (datetime(2026, 10, 6, 9, 0, tzinfo=IST), datetime(2026, 10, 6, 10, 0, tzinfo=IST)),
        (datetime(2026, 10, 6, 11, 0, tzinfo=IST), datetime(2026, 10, 6, 13, 0, tzinfo=IST)),
        (datetime(2026, 10, 6, 14, 0, tzinfo=IST), datetime(2026, 10, 6, 17, 0, tzinfo=IST)),
    ]
    assert len(out["busy"]) == 2
    assert out["timezone"] == "Asia/Kolkata"
    assert oauth.scopes == [CALENDAR_READONLY]
    body = json.loads(calls[0].content)
    assert body["timeMin"] == iso(START) and body["timeZone"] == "Asia/Kolkata"


def test_availability_all_free(cal_factory):
    provider, _, _ = cal_factory({("POST", FREEBUSY): busy_response()})
    out = run(provider.availability(START, END, "Asia/Kolkata"))
    assert out["busy"] == []
    assert len(out["free"]) == 1
    assert datetime.fromisoformat(out["free"][0]["start"]) == START


def test_availability_overlapping_busy_is_merged(cal_factory):
    provider, _, _ = cal_factory({
        ("POST", FREEBUSY): busy_response(
            ("2026-10-06T09:30:00+05:30", "2026-10-06T11:00:00+05:30"),
            ("2026-10-06T10:30:00+05:30", "2026-10-06T12:00:00+05:30"),
        ),
    })
    out = run(provider.availability(START, END, "Asia/Kolkata"))
    assert len(out["free"]) == 2  # 9:00-9:30 and 12:00-17:00


def test_availability_bad_response_is_controlled(cal_factory):
    provider, _, _ = cal_factory({("POST", FREEBUSY): err(400, "Bad request")})
    with pytest.raises(ProductivityError) as exc:
        run(provider.availability(START, END, "Asia/Kolkata"))
    assert exc.value.code == "GOOGLE_API_ERROR"


# ----------------------------------------------------------------- create

def test_create_posts_payload_with_offset_and_timezone(cal_factory):
    provider, oauth, calls = cal_factory({
        ("POST", FREEBUSY): busy_response(),
        ("POST", EVENTS): ok(gcal_event(eid="new1")),
    })
    start = datetime(2026, 10, 7, 15, 0, tzinfo=IST)
    end = datetime(2026, 10, 7, 15, 30, tzinfo=IST)
    out = run(provider.create_event(
        "Dentist", start, end, "Asia/Kolkata",
        attendees=["a@example.com"], location="Clinic", description="Checkup",
    ))
    assert out["status"] == "created" and out["id"] == "new1"
    payload = json.loads(calls[-1].content)
    assert payload["summary"] == "Dentist"
    assert payload["start"] == {"dateTime": "2026-10-07T15:00:00+05:30", "timeZone": "Asia/Kolkata"}
    assert payload["end"] == {"dateTime": "2026-10-07T15:30:00+05:30", "timeZone": "Asia/Kolkata"}
    assert payload["attendees"] == [{"email": "a@example.com"}]
    assert payload["location"] == "Clinic"
    assert oauth.scopes == [CALENDAR_READONLY, CALENDAR_EVENTS]  # freebusy + insert


def test_create_reports_advisory_conflicts(cal_factory):
    provider, _, calls = cal_factory({
        ("POST", FREEBUSY): busy_response(
            ("2026-10-06T15:00:00+05:30", "2026-10-06T16:00:00+05:30")
        ),
        ("POST", EVENTS): ok(gcal_event(eid="new2")),
    })
    start = datetime(2026, 10, 6, 15, 0, tzinfo=IST)
    end = datetime(2026, 10, 6, 15, 30, tzinfo=IST)
    out = run(provider.create_event("Overlap", start, end, "Asia/Kolkata"))
    assert len(out["conflicts"]) == 1
    assert out["status"] == "created"  # advisory only, create still happened
    assert sum(1 for c in calls if c.method == "POST" and c.url.path == EVENTS) == 1


def test_create_without_conflict_check_skips_freebusy(cal_factory):
    provider, _, calls = cal_factory({("POST", EVENTS): ok(gcal_event(eid="new3"))})
    run(provider.create_event(
        "Quiet", datetime(2026, 10, 6, 15, 0, tzinfo=IST),
        datetime(2026, 10, 6, 15, 30, tzinfo=IST), "Asia/Kolkata",
        check_conflicts=False,
    ))
    assert not any(c.url.path == FREEBUSY for c in calls)


def test_create_freebusy_failure_still_creates(cal_factory):
    provider, _, calls = cal_factory({
        ("POST", FREEBUSY): err(500, "Backend"),
        ("POST", EVENTS): ok(gcal_event(eid="new4")),
    })
    out = run(provider.create_event(
        "Resilient", datetime(2026, 10, 6, 15, 0, tzinfo=IST),
        datetime(2026, 10, 6, 15, 30, tzinfo=IST), "Asia/Kolkata",
    ))
    assert out["status"] == "created"
    assert out["conflicts"] == []


def test_create_bad_attendee_is_invalid_recipient(cal_factory):
    provider, _, _ = cal_factory({
        ("POST", FREEBUSY): busy_response(),
        ("POST", EVENTS): err(400, "Invalid attendee address"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.create_event(
            "Bad", datetime(2026, 10, 6, 15, 0, tzinfo=IST),
            datetime(2026, 10, 6, 15, 30, tzinfo=IST), "Asia/Kolkata",
            attendees=["@@@@"],
        ))
    assert exc.value.code == "INVALID_RECIPIENT"


def test_create_other_400_is_controlled(cal_factory):
    provider, _, _ = cal_factory({
        ("POST", FREEBUSY): busy_response(),
        ("POST", EVENTS): err(400, "Start time is invalid"),
    })
    with pytest.raises(ProductivityError) as exc:
        run(provider.create_event(
            "Bad", datetime(2026, 10, 6, 15, 0, tzinfo=IST),
            datetime(2026, 10, 6, 15, 30, tzinfo=IST), "Asia/Kolkata",
        ))
    assert exc.value.code == "GOOGLE_API_ERROR"


def test_create_caps_description(cal_factory):
    provider, _, calls = cal_factory({
        ("POST", FREEBUSY): busy_response(),
        ("POST", EVENTS): ok(gcal_event(eid="new5")),
    })
    run(provider.create_event(
        "Long", datetime(2026, 10, 6, 15, 0, tzinfo=IST),
        datetime(2026, 10, 6, 15, 30, tzinfo=IST), "Asia/Kolkata",
        description="D" * 5000,
    ))
    assert len(json.loads(calls[-1].content)["description"]) <= 2000


# ---------------------------------------------------------------- update

def test_update_patches_only_given_fields(cal_factory):
    provider, oauth, calls = cal_factory({
        ("PATCH", f"{EVENTS}/ev1"): ok(gcal_event(summary="Renamed")),
    })
    new_start = datetime(2026, 10, 6, 16, 0, tzinfo=IST)
    out = run(provider.update_event(
        "ev1", {"title": "Renamed", "start": new_start}, "Asia/Kolkata"
    ))
    assert out["status"] == "updated" and out["title"] == "Renamed"
    payload = json.loads(calls[0].content)
    assert set(payload) == {"summary", "start"}
    assert payload["start"] == {"dateTime": "2026-10-06T16:00:00+05:30",
                                "timeZone": "Asia/Kolkata"}
    assert oauth.scopes == [CALENDAR_EVENTS]


def test_update_missing_event(cal_factory):
    provider, _, _ = cal_factory({("PATCH", f"{EVENTS}/gone"): err(404, "Not Found")})
    with pytest.raises(ProductivityError) as exc:
        run(provider.update_event("gone", {"title": "x"}, "Asia/Kolkata"))
    assert exc.value.code == "CALENDAR_EVENT_NOT_FOUND"


# ---------------------------------------------------------------- delete

def test_delete_success(cal_factory):
    provider, oauth, calls = cal_factory({("DELETE", f"{EVENTS}/ev1"): ok({})})
    out = run(provider.delete_event("ev1"))
    assert out == {"event_id": "ev1", "status": "deleted"}
    assert calls[0].method == "DELETE"
    assert oauth.scopes == [CALENDAR_EVENTS]


def test_delete_missing_event(cal_factory):
    provider, _, _ = cal_factory({("DELETE", f"{EVENTS}/gone"): err(404, "Not Found")})
    with pytest.raises(ProductivityError) as exc:
        run(provider.delete_event("gone"))
    assert exc.value.code == "CALENDAR_EVENT_NOT_FOUND"


def test_delete_gone_410_is_not_found(cal_factory):
    provider, _, _ = cal_factory({("DELETE", f"{EVENTS}/old"): err(410, "Gone")})
    with pytest.raises(ProductivityError) as exc:
        run(provider.delete_event("old"))
    assert exc.value.code == "CALENDAR_EVENT_NOT_FOUND"


# ------------------------------------------------------------- complement

def test_complement_busy_outside_range_is_ignored():
    busy = [{"start": "2020-01-01T00:00:00+00:00", "end": "2020-01-01T01:00:00+00:00"}]
    free = _complement(START, END, busy)
    assert len(free) == 1
    assert datetime.fromisoformat(free[0]["start"]) == START


def test_complement_skips_malformed_windows():
    free = _complement(START, END, [{"start": "junk", "end": "2026-10-06T10:00:00+05:30"}])
    assert len(free) == 1
