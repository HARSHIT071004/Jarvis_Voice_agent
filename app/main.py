"""Jarvis application entry point.

Flow:  standby (local wake word) -> voice session (Gemini Live) -> standby
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from app.agent.provider import GeminiVoiceProvider, ProviderConnectionError
from app.agent.session import VoiceSession
from app.audio.input import AudioDeviceError as MicError
from app.audio.input import Microphone
from app.audio.output import AudioDeviceError as SpeakerError
from app.audio.output import Speaker
from app.browser import BrowserController, build_controller
from app.config import ConfigError, Settings, load_settings
from app.intelligence import build_intelligence
from app.memory import MemoryManager, MemoryRetrieval
from app.productivity import ProductivityStack, build_productivity
from app.research import DuckDuckGoHTMLProvider, PageFetcher
from app.research import build_researcher  # noqa: F401  (available for scripts)
from app.security import build_control_plane
from app.state import AppState, StateMachine
from app.tools import (
    ToolRouter,
    build_browser_tools,
    build_productivity_tools,
    build_unconfigured_productivity_tools,
    build_registry,
    build_web_tools,
)
from app.utils.logging import setup_logging
from app.wake.detector import WakeDetector
from app.wake.model_setup import WakeModelError

logger = logging.getLogger("jarvis")

BANNER = """====================================
              J A R V I S
===================================="""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis voice assistant (Phase 1)")
    parser.add_argument(
        "--wake-only",
        action="store_true",
        help="test wake-word detection without connecting to the AI",
    )
    return parser.parse_args()


def _status(label: str, ok: bool, detail: str = "") -> None:
    state = "OK" if ok else f"FAILED ({detail})"
    print(f"{label}: {state}")


def _productivity_auth_probe(productivity_stack, settings: Settings):
    """Gate-level AUTH_REQUIRED when Gmail/Calendar is not configured (§7).

    Keeps the voice flow honest: an unconfigured account is refused before
    an approval is ever asked for.
    """
    if not settings.productivity_enabled or productivity_stack is not None:
        return None
    from app.tools.productivity import PRODUCTIVITY_TOOL_NAMES

    def probe(ctx):
        if ctx.tool in PRODUCTIVITY_TOOL_NAMES:
            return "AUTH_REQUIRED"
        return None

    return probe


async def run_app(settings: Settings, wake_only: bool) -> None:
    if settings.browser_enabled:
        # §25: a browser task may take up to BROWSER_MAX_STEPS tool calls;
        # lift the Phase 4 loop budgets so browser tasks are not cut early.
        settings = settings.model_copy(
            update={
                "agent_max_tool_calls": min(
                    50, max(settings.agent_max_tool_calls, settings.browser_max_steps)
                ),
                "agent_max_tool_iterations": min(
                    20, max(settings.agent_max_tool_iterations, settings.browser_max_steps)
                ),
            }
        )
    machine = StateMachine()
    mic = Microphone(device=settings.audio_input_device)
    speaker = Speaker(device=settings.audio_output_device)
    detector = WakeDetector(settings)

    print(BANNER)
    print("\nStatus: Initializing...\n")

    # ---- health checks -------------------------------------------------
    failures: list[str] = []
    try:
        mic.probe()
        _status("Microphone", True)
    except MicError as exc:
        _status("Microphone", False, str(exc))
        failures.append(str(exc))

    if wake_only:
        _status("Speaker", True, "not required in wake-only mode")
    else:
        try:
            speaker.probe()
            _status("Speaker", True)
        except SpeakerError as exc:
            _status("Speaker", False, str(exc))
            failures.append(str(exc))

    try:
        detector.initialize()
        _status("Wake engine", True)
    except WakeModelError as exc:
        _status("Wake engine", False, str(exc))
        failures.append(str(exc))

    if wake_only:
        _status("AI service", True, "skipped in wake-only mode")
    elif settings.gemini_api_key:
        _status("AI service", True)
    else:
        _status("AI service", False, "GEMINI_API_KEY missing")
        failures.append("GEMINI_API_KEY missing")

    if settings.intelligence_enabled and not wake_only:
        model = settings.intelligence_model or settings.gemini_model
        _status("Intelligence", True, f"extraction via {model}")
    else:
        _status("Intelligence", True, "disabled")

    memory: MemoryManager | None = None
    retrieval: MemoryRetrieval | None = None
    tool_router: ToolRouter | None = None
    if wake_only:
        _status("Memory", True, "skipped in wake-only mode")
    elif settings.memory_enabled:
        try:
            memory = MemoryManager.open(settings.memory_db_path)
            retrieval = MemoryRetrieval(memory)
            _status("Memory", True, str(settings.memory_db_path))
        except Exception as exc:
            _status("Memory", False, str(exc))
            failures.append(f"memory database: {exc}")
    else:
        _status("Memory", True, "disabled")

    # Phase 4: tool layer (wraps the Phase 3 manager; no SQL from tools)
    # Phase 5: web research tools ride in the same registry/router.
    # Phase 6: browser tools ride in the same registry/router too (§23).
    # Phase 7: productivity tools ride the same registry/router (§2).
    tool_registry = None
    browser_controller: BrowserController | None = None
    productivity_stack: ProductivityStack | None = None
    if wake_only or not settings.agent_enabled:
        detail = "skipped in wake-only mode" if wake_only else "disabled"
        _status("Research", True, detail)
        _status("Browser", True, detail)
        _status("Productivity", True, detail)
        _status("Tools", True, detail)
        _status("Security", True, detail)
    else:
        web_tools = []
        if settings.research_enabled:
            try:
                search = DuckDuckGoHTMLProvider(timeout=settings.web_search_timeout)
                fetcher = PageFetcher(
                    timeout=settings.request_timeout, max_size=settings.max_page_size
                )
                web_tools = build_web_tools(search, fetcher)
                _status(
                    "Research", True,
                    f"{settings.web_search_provider}, {len(web_tools)} tools",
                )
            except Exception as exc:
                _status("Research", False, str(exc))
                failures.append(f"research tools: {exc}")
        else:
            _status("Research", True, "disabled")

        browser_tools = []
        if settings.browser_enabled:
            try:
                browser_controller = build_controller(settings)
                browser_tools = build_browser_tools(browser_controller)
                _status(
                    "Browser", True,
                    f"headless={settings.browser_headless}, {len(browser_tools)} tools, "
                    f"max_steps={settings.browser_max_steps}",
                )
            except Exception as exc:
                _status("Browser", False, str(exc))
                failures.append(f"browser: {exc}")
                browser_controller = None
        else:
            _status("Browser", True, "disabled")

        productivity_tools = []
        if settings.productivity_enabled:
            try:
                productivity_stack = build_productivity(settings)
                if productivity_stack is None:
                    # not configured yet — app must boot normally without Google;
                    # tools still register so the model can honestly answer
                    # "connect Gmail first" (AUTH_REQUIRED) instead of nothing
                    productivity_tools = build_unconfigured_productivity_tools(
                        settings.timezone, manager=memory
                    )
                    _status(
                        "Productivity", True,
                        f"not configured (missing {settings.google_client_file}); "
                        f"{len(productivity_tools)} tools return AUTH_REQUIRED",
                    )
                else:
                    productivity_tools = build_productivity_tools(
                        productivity_stack.gmail,
                        productivity_stack.calendar,
                        productivity_stack.policy,
                        productivity_stack.timezone,
                        manager=memory,
                        gmail_max_results=settings.gmail_max_results,
                        calendar_max_events=settings.calendar_max_events,
                        calendar_max_range_days=settings.calendar_max_range_days,
                    )
                    _status(
                        "Productivity", True,
                        f"{len(productivity_tools)} tools, tz={settings.timezone}",
                    )
            except Exception as exc:
                _status("Productivity", False, str(exc))
                failures.append(f"productivity: {exc}")
                if productivity_stack is not None:
                    await productivity_stack.aclose()
                productivity_stack = None
                productivity_tools = []
        else:
            _status("Productivity", True, "disabled")

        if memory is not None:
            tool_registry = build_registry(
                memory, web_tools=web_tools, browser_tools=browser_tools,
                productivity_tools=productivity_tools,
            )
        elif web_tools or browser_tools or productivity_tools:
            tool_registry = build_registry(
                None, web_tools=web_tools, browser_tools=browser_tools,
                productivity_tools=productivity_tools,
            )
        if tool_registry is not None:
            # Browser navigation may take BROWSER_TIMEOUT ms and Google API
            # calls may retry — the router must not cut them short (§5).
            timeout = settings.agent_tool_timeout
            if settings.browser_enabled:
                timeout = max(timeout, settings.browser_timeout / 1000.0 + 5.0)
            if settings.productivity_enabled and productivity_tools:
                timeout = max(timeout, 20.0)
            # Phase 8: every tool call passes the centralized control plane.
            security = None
            if settings.security_enabled:
                security = build_control_plane(
                    settings, auth_probe=_productivity_auth_probe(productivity_stack, settings)
                )
                if browser_controller is not None:
                    # Browser sensitive actions resolve through the shared
                    # ApprovalManager so voice can approve them (§23).
                    browser_controller.policy.approval_handler = security.browser_approval_bridge()
            tool_router = ToolRouter(tool_registry, execution_timeout=timeout, security=security)
            _status(
                "Tools", True,
                f"{len(tool_registry)} registered, timeout {timeout:.0f}s",
            )
            if security is not None:
                audit_detail = str(settings.audit_path) if settings.audit_enabled else "off"
                _status(
                    "Security", True,
                    f"control plane, approval ttl {settings.approval_timeout_seconds:.0f}s, "
                    f"audit {audit_detail}",
                )
            else:
                _status("Security", True, "disabled by configuration")
        else:
            _status("Tools", True, "disabled (memory off, no web)")
            _status("Security", True, "no tools to guard")

    if failures:
        print("\nStartup failed:")
        for f in failures:
            print(f"  - {f}")
        if memory is not None:
            memory.close()
        return

    print("\n------------------------------------")
    print(f'Say "{settings.wake_word}" to begin.')
    print("------------------------------------")

    try:
        await _standby_loop(
            settings, machine, mic, speaker, detector, wake_only,
            memory, retrieval, tool_registry, tool_router,
        )
    finally:
        if browser_controller is not None:
            await browser_controller.close()
        if productivity_stack is not None:
            await productivity_stack.aclose()
        mic.close()
        detector.close()
        await speaker.stop()
        if memory is not None:
            memory.close()


async def _standby_loop(
    settings: Settings,
    machine: StateMachine,
    mic: Microphone,
    speaker: Speaker,
    detector: WakeDetector,
    wake_only: bool,
    memory: MemoryManager | None = None,
    retrieval: MemoryRetrieval | None = None,
    tool_registry=None,
    tool_router: ToolRouter | None = None,
) -> None:
    mic.open()
    frames = mic.frames()

    while True:
        if machine.state != AppState.SLEEPING:
            machine.sleep()
        logger.info('Waiting for "%s"...', settings.wake_word)

        detected = ""
        async for frame in frames:
            detected = detector.process(frame)
            if detected:
                break
        if not detected:
            continue

        print(f"\n[WAKE] {settings.wake_word} detected.")

        if wake_only:
            logger.info("Wake OK (wake-only mode, no AI session)")
            await asyncio.sleep(detector.refractory_seconds)
            continue

        started = time.monotonic()

        def make_provider() -> GeminiVoiceProvider:
            # fresh memory brief per session (recomputed on reconnect too)
            brief = retrieval.brief() if retrieval is not None else ""
            declarations = (
                tool_registry.declarations() if tool_registry is not None else None
            )
            return GeminiVoiceProvider(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                memory_context=brief,
                tool_declarations=declarations,
            )

        session = VoiceSession(
            settings=settings,
            microphone=mic,
            speaker=speaker,
            machine=machine,
            provider_factory=make_provider,
            intelligence=build_intelligence(settings),
            memory=memory,
            tool_router=tool_router,
        )
        try:
            await session.run()
        except ProviderConnectionError as exc:
            logger.error("Session error: %s", exc)
            print("\n[SESSION] Could not reach the AI service.")
        except Exception:
            logger.exception("Unexpected session failure")
            print("\n[SESSION] Unexpected error; recovering.")
        finally:
            machine.sleep()
            speaker.clear()
            logger.info("Voice session ended in %.1fs", time.monotonic() - started)

        print("\n[SESSION] Conversation ended.")
        print("\nReturning to standby...\n")
        print(f'Say "{settings.wake_word}" to begin.')
        print()


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings(require_api_key=not args.wake_only)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2

    setup_logging(settings.log_level)

    try:
        asyncio.run(run_app(settings, wake_only=args.wake_only))
    except KeyboardInterrupt:
        print("\nShutting down. Goodbye.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
