r"""Full voice test: TTS-generated user speech -> VoiceSession -> captured replies.

- User side: edge-tts (fallback pyttsx3) generates 16 kHz PCM utterances that
  are streamed into the session exactly like microphone frames.
- Agent side: every spoken reply is captured to data/voice_test/<name>_reply.wav
  (24 kHz PCM) plus a full text transcript, tool-call log, and audit trail.
- Phase 8: browser clicks ask for approval over voice; a live "yes" (generated
  by TTS too) satisfies the pending approval.

Usage:
    $env:PYTHONPATH='.'; .venv\Scripts\python.exe scripts\test_voice_full.py [email|browser|all]
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

from app.agent.provider import (
    GeminiVoiceProvider,
    TranscriptEvent,
    TurnCompleteEvent,
)
from app.agent.session import VoiceSession
from app.browser import BrowserController
from app.config import load_settings
from app.memory.manager import MemoryManager
from app.productivity import build_productivity
from app.security import build_control_plane
from app.state import StateMachine
from app.tools import ToolRouter, build_registry
from app.tools.browser import build_browser_tools
from app.tools.productivity import build_productivity_tools

OUT_DIR = Path("data") / "voice_test"
IN_RATE = 16000
OUT_RATE = 24000
CHUNK = 3200  # 100 ms @ 16 kHz, matches the real microphone

PROMPTS = {
    "email": [
        "Hey Jarvis, check my new emails and tell me which job opportunity "
        "emails are important and which ones I can ignore."
    ],
    "browser": [
        "Hey Jarvis, open en dot wikipedia dot org in the browser.",
        "Search for Albert Einstein and tell me who he was from the article."
    ],
    "complete": [
        "Hey Jarvis, open wikipedia dot org in the browser and search for Albert Einstein.",
        "Tell me what the result says.",
        "Now check my new emails and tell me which job opportunity emails are important and which ones I can ignore."
    ],
}

_tts_n = 0


async def tts(text: str) -> np.ndarray:
    """Generate user speech -> int16 mono @ 16 kHz."""
    global _tts_n
    _tts_n += 1
    tmp = Path(tempfile.gettempdir())
    try:
        import edge_tts

        mp3 = tmp / f"jarvis_utt_{_tts_n}.mp3"
        await edge_tts.Communicate(text, "en-US-JennyNeural").save(str(mp3))
        data, sr = sf.read(str(mp3), dtype="int16", always_2d=True)
        mp3.unlink(missing_ok=True)
    except Exception:
        import pyttsx3

        wav = tmp / f"jarvis_utt_{_tts_n}.wav"
        engine = pyttsx3.init()
        engine.save_to_file(text, str(wav))
        engine.runAndWait()
        data, sr = sf.read(str(wav), dtype="int16", always_2d=True)
        wav.unlink(missing_ok=True)
    mono = data.mean(axis=1).astype("<i2")
    if sr != IN_RATE:
        mono = np.interp(
            np.arange(int(len(mono) * IN_RATE / sr)), np.arange(len(mono)), mono
        ).astype("<i2")
    return mono


class FileMic:
    """Replays TTS utterances like a microphone, turn-synchronised."""

    def __init__(self, prompts: list[str], pending_fn):
        self.prompts = prompts
        self.pending_fn = pending_fn
        self._turn = asyncio.Event()

    def signal_turn(self) -> None:
        self._turn.set()

    async def _wait_turn(self) -> None:
        await asyncio.wait_for(self._turn.wait(), timeout=240.0)
        self._turn.clear()

    async def _stream(self, pcm: np.ndarray):
        raw = pcm.tobytes()
        for i in range(0, len(raw), CHUNK):
            yield raw[i : i + CHUNK]
            await asyncio.sleep(0.1)

    async def frames(self):
        print(f"[MIC] Waiting for wake-word greeting turn...", flush=True)
        silence = b"\x00" * CHUNK
        while not self._turn.is_set():
            yield silence
            await asyncio.sleep(0.1)
        self._turn.clear()

        for prompt in self.prompts:
            print(f"[MIC] Generating TTS for: '{prompt[:60]}...' ...", flush=True)
            pcm = await tts(prompt)
            print(f"[MIC] Streaming {len(pcm)} audio samples to Jarvis...", flush=True)
            async for chunk in self._stream(pcm):
                yield chunk
            print(f"[MIC] Speech sent. Streaming silence while waiting for Jarvis response...", flush=True)
            while not self._turn.is_set():
                yield silence
                await asyncio.sleep(0.1)
            self._turn.clear()

            # Phase 8: if the agent is asking for approval, answer it
            for _ in range(4):
                if not self.pending_fn():
                    break
                print(f"[MIC] Agent requested approval. Speaking 'yes' via TTS...", flush=True)
                async for chunk in self._stream(await tts("yes")):
                    yield chunk
                while not self._turn.is_set():
                    yield silence
                    await asyncio.sleep(0.1)
                self._turn.clear()
        print("[MIC] All dialog turns completed.", flush=True)
        # all turns done -> end the session

    def close(self) -> None:
        pass


class CaptureSpeaker:
    """Records everything the agent says to data/voice_test/<name>_reply.wav."""

    def __init__(self):
        self.idle = asyncio.Event()
        self.idle.set()
        self.chunks: list[bytes] = []
        self._drain: asyncio.Task | None = None

    async def start(self) -> None:
        pass

    def play(self, pcm: bytes) -> None:
        self.chunks.append(pcm)
        self.idle.clear()
        if self._drain is None or self._drain.done():
            self._drain = asyncio.create_task(self._drain_later())

    async def _drain_later(self) -> None:
        total = sum(len(c) for c in self.chunks)
        await asyncio.sleep(total / (2 * OUT_RATE) + 0.3)
        self.idle.set()

    def clear(self) -> None:
        self.idle.set()

    async def wait_idle(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self.idle.wait(), timeout)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            return False

    async def stop(self) -> None:
        pass

    def save(self, path: Path) -> float:
        data = b"".join(self.chunks)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(OUT_RATE)
            wf.writeframes(data)
        return len(data) / (2 * OUT_RATE)


class HarnessSession(VoiceSession):
    transcripts: list[tuple[str, str]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transcripts = []
        self.on_turn = None

    def _handle_event(self, event) -> None:
        super()._handle_event(event)
        if isinstance(event, TranscriptEvent):
            print(f"  [{event.role.upper()}]: {event.text}", flush=True)
            self.transcripts.append((event.role, event.text))
        elif isinstance(event, TurnCompleteEvent):
            print("  [TURN_COMPLETE]", flush=True)
            if self.on_turn is not None:
                self.on_turn()


async def run_scenario(name: str) -> dict:
    settings = load_settings(require_api_key=True)
    settings.voice_inactivity_timeout = 180  # browser turns can think a while
    settings.voice_session_timeout = 900
    plane = build_control_plane(settings)
    manager = MemoryManager.open(Path("data") / "jarvis.db")
    stack = build_productivity(settings)
    browser = None
    session = None
    tools_log: list[tuple] = []
    report: dict = {"name": name, "transcripts": [], "tools": [], "audio": None,
                    "audit": [], "error": None}
    try:
        browser_tools = []
        if name in ("browser", "complete"):
            browser = BrowserController(headless=True)
            browser_tools = build_browser_tools(browser)
        prod_tools = []
        if stack is not None:
            prod_tools = build_productivity_tools(
                stack.gmail, stack.calendar, stack.policy, stack.timezone,
                manager=manager,
                gmail_max_results=settings.gmail_max_results,
            )
        registry = build_registry(
            manager, browser_tools=browser_tools, productivity_tools=prod_tools
        )
        router = ToolRouter(registry, execution_timeout=30.0, security=plane)
        if browser is not None:
            browser.policy.approval_handler = plane.browser_approval_bridge()

        orig_route = router.route

        async def spy(nm, args):
            print(f"  [TOOL CALL] {nm}({repr(dict(args))[:100]})", flush=True)
            r = await orig_route(nm, args)
            print(f"  [TOOL RESULT] {nm} success={r.success} err={r.error}", flush=True)
            tools_log.append(
                (nm, repr(dict(args))[:90], r.success,
                 (r.error or {}).get("code") if r.error else None)
            )
            return r

        router.route = spy  # type: ignore[method-assign]

        mic = FileMic(PROMPTS[name], lambda: bool(plane.approvals.pending()))
        speaker = CaptureSpeaker()
        session = HarnessSession(
            settings=settings,
            microphone=mic,
            speaker=speaker,
            machine=StateMachine(),
            provider_factory=lambda: GeminiVoiceProvider(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                tool_declarations=registry.declarations(),
            ),
            memory=manager,
            tool_router=router,
        )
        session.on_turn = mic.signal_turn

        await asyncio.wait_for(session.run(), timeout=700.0)
        wav_path = OUT_DIR / f"{name}_reply.wav"
        report["audio"] = str(wav_path)
        report["seconds"] = round(speaker.save(wav_path), 1)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["transcripts"] = session.transcripts if session is not None else []
        report["tools"] = tools_log
        report["audit"] = [
            (e.get("event"), e.get("tool"), e.get("decision") or e.get("success"),
             e.get("approval_status") or "")
            for e in plane.audit.events()
        ]
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass
        if stack is not None:
            await stack.aclose()
        manager.close()
    return report


def show(r: dict) -> None:
    print(f"\n{'='*70}\n=== SCENARIO {r['name'].upper()} ===")
    if r["error"]:
        print(f"ERROR: {r['error']}")
    for role, text in r["transcripts"]:
        who = "YOU    " if role == "user" else "Jarvis  "
        print(f"  {who}: {text}")
    print("  tools:")
    for nm, args, ok, code in r["tools"]:
        print(f"    - {nm} {'OK' if ok else 'ERR ' + str(code)}  {args}")
    if r.get("audio"):
        print(f"  reply audio: {r['audio']}  ({r.get('seconds')}s)")
    print("  audit trail:")
    for ev in r["audit"][-30:]:
        print(f"    {ev}")


async def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = ["email", "browser"] if which == "all" else [which]
    ok = True
    for key in keys:
        report = await run_scenario(key)
        show(report)
        ok = ok and not report["error"] and bool(report["transcripts"])
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
