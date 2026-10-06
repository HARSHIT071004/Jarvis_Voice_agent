"""Live voice tool-calling test (Phase 4 voice integration).

Connects to Gemini Live with the tool declarations, sends text prompts,
and verifies: tool_call event -> ToolRouter -> send_tool_response ->
grounded spoken (transcribed) final answer at turn_complete.

No microphone needed — exercises the same path a spoken utterance takes
after the model decides to call a tool.

Usage:
    .venv\\Scripts\\python.exe scripts\\test_live_tools.py [a|b|c|all]
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from app.agent.provider import (  # noqa: E402
    GeminiVoiceProvider,
    ToolCallEvent,
    TranscriptEvent,
    TurnCompleteEvent,
)
from app.browser import BrowserController  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.memory.manager import MemoryManager  # noqa: E402
from app.productivity import build_productivity  # noqa: E402
from app.research import DuckDuckGoHTMLProvider, PageFetcher  # noqa: E402
from app.tools import ToolRouter, build_registry, build_web_tools  # noqa: E402
from app.tools.browser import build_browser_tools  # noqa: E402
from app.tools.productivity import (  # noqa: E402
    build_productivity_tools,
    build_unconfigured_productivity_tools,
)

TIMEOUT = 60.0

# (label, prompt, expect_tool, expected_tool_name|None)
SCENARIOS = {
    "a": (
        "Scenario A: 'Who is Rahul?' (expect get_contact)",
        "Who is Rahul?",
        True,  # expect at least one tool call
        "get_contact",
    ),
    "b": (
        "Scenario B: 'Create a task...' (expect create_task)",
        "Create a task to send Rahul the proposal tomorrow.",
        True,
        "create_task",
    ),
    "c": (
        "Scenario C: general question (expect NO tool)",
        "What is machine learning? One sentence.",
        False,
        None,
    ),
    "d": (
        "Scenario D: memory search (expect search_memory)",
        "What do you remember about my proposal for Rahul?",
        True,
        "search_memory",
    ),
    "e": (
        "Scenario E: research request (expect search_web)",
        "Search the web for what the Gemini Live API is. One sentence.",
        True,
        "search_web",
    ),
    "f": (
        "Scenario F: browser navigation (expect open_url + read_page)",
        "Open https://example.com in the browser and tell me the title of the page.",
        True,
        "open_url",
    ),
    "g": (
        "Scenario G: 'Check my email' (expect search_emails)",
        "Check my email. One sentence — is there anything new?",
        True,
        "search_emails",
    ),
    "h": (
        "Scenario H: 'What's on my calendar tomorrow?' (expect list_calendar_events)",
        "What's on my calendar tomorrow? One sentence.",
        True,
        "list_calendar_events",
    ),
    "i": (
        "Scenario I: 'Send an email...' (expect send_email/draft_email -> controlled code)",
        "Send an email to a@example.com with subject 'Meeting Confirmation' "
        "saying the meeting is confirmed.",
        True,
        ("send_email", "draft_email"),
    ),
}

# Codes tolerated for a scenario when the capability is legitimately not
# configured yet (no Google OAuth client / not authorized) or when the
# action is correctly held for approval — reported as PASS-with-note,
# never as a silent success.
ALLOWED_ERROR_CODES = {
    "g": {"AUTH_REQUIRED", "TOOL_NOT_FOUND"},
    "h": {"AUTH_REQUIRED", "TOOL_NOT_FOUND"},
    "i": {"APPROVAL_REQUIRED", "AUTH_REQUIRED", "TOOL_NOT_FOUND"},
}


async def run_scenario(settings, manager, router, declarations, prompt: str, on_finish=None) -> dict:
    provider = GeminiVoiceProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        tool_declarations=declarations,
    )
    outcome = {"tool_calls": [], "responses": [], "transcript": [], "error": None}
    try:
        await provider.connect()
    except Exception as exc:
        outcome["error"] = f"connect failed: {exc}"
        return outcome

    async def worker():
        await provider.send_text(prompt)
        async for event in provider.events():
            if isinstance(event, ToolCallEvent):
                results = []
                for call_id, name, args in event.calls:
                    outcome["tool_calls"].append((name, args))
                    result = await router.route(name, args)
                    result.call_id = call_id
                    results.append(result.as_dict() | {"call_id": call_id})
                    outcome["responses"].append(
                        (name, result.success, result.error["code"] if result.error else None)
                    )
                await provider.send_tool_response(results)
            elif isinstance(event, TranscriptEvent):
                outcome["transcript"].append((event.role, event.text))
            elif isinstance(event, TurnCompleteEvent):
                # wait for the real end of the turn (tool loop included)
                break

    try:
        await asyncio.wait_for(worker(), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        outcome["error"] = "timed out waiting for turn_complete"
    except Exception as exc:
        outcome["error"] = f"session error: {exc}"
    finally:
        await provider.close()
        if on_finish is not None:
            # same event loop as the browser session (Playwright is loop-bound)
            await on_finish()
    return outcome


def show(key: str, manager, router, settings, declarations, browser=None) -> bool:
    label, prompt, expect_tool, expected_name = SCENARIOS[key]
    outcome = asyncio.run(
        run_scenario(
            settings, manager, router, declarations, prompt,
            on_finish=browser.close if browser is not None else None,
        )
    )
    print(f"\n=== {label} ===")
    if outcome["error"]:
        print(f"  ERROR: {outcome['error']}")
        return False
    print(f"  tool calls: {outcome['tool_calls']}")
    print(f"  results:    {outcome['responses']}")
    answer = " ".join(t for r, t in outcome["transcript"] if r == "jarvis")
    print(f"  answer: {answer[:300]}")
    got_tool = bool(outcome["tool_calls"])
    has_answer = bool(answer)
    ok = has_answer and (got_tool == expect_tool)
    tolerated: list[str] = []
    if expect_tool and got_tool:
        allowed = ALLOWED_ERROR_CODES.get(key, set())
        ok = ok and all(
            success or (code in allowed)
            for _, success, code in outcome["responses"]
        )
        tolerated = sorted({code for _, success, code in outcome["responses"]
                            if not success and code in allowed})
        if expected_name:
            wanted = (
                set(expected_name)
                if isinstance(expected_name, (list, tuple, set))
                else {expected_name}
            )
            ok = ok and any(name in wanted for name, _ in outcome["tool_calls"])
    note = f" tolerated={tolerated}" if tolerated else ""
    print(f"  -> {'PASS' if ok else 'FAIL'} (expected tool={expect_tool}, got={got_tool}){note}")
    return ok


def main() -> int:
    settings = load_settings(require_api_key=True)
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    keys = list(SCENARIOS) if which == "all" else which.split(",")
    results: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as tmp:
        manager = MemoryManager.open(Path(tmp) / "voice_tools.db")
        web_tools = build_web_tools(DuckDuckGoHTMLProvider(), PageFetcher())
        browser = BrowserController(headless=True)  # default policy: http/https
        prod = build_productivity(settings)
        prod_tools = (
            build_productivity_tools(
                prod.gmail, prod.calendar, prod.policy, prod.timezone
            )
            if prod is not None
            else build_unconfigured_productivity_tools(settings.timezone)
        )
        registry = build_registry(
            manager, web_tools=web_tools, browser_tools=build_browser_tools(browser),
            productivity_tools=prod_tools,
        )
        router = ToolRouter(registry, execution_timeout=25.0)
        manager.save_contact(name="Rahul", company="ABC", role="Project Manager")
        declarations = registry.declarations()
        try:
            for key in keys:
                results[key] = show(
                    key, manager, router, settings, declarations,
                    browser=browser if key == "f" else None,
                )
        finally:
            manager.close()
            if prod is not None:
                asyncio.run(prod.aclose())

    print("\nSummary:", {k: "PASS" if v else "FAIL" for k, v in results.items()})
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
