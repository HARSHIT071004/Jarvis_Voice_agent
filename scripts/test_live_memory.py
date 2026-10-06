"""Live Phase 3 verification: memory brief + per-turn context injection.

Run: RUN_LIVE_TESTS=1 python scripts/test_live_memory.py
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.provider import (  # noqa: E402
    AudioData,
    GeminiVoiceProvider,
    TranscriptEvent,
    TurnCompleteEvent,
)
from app.config import load_settings  # noqa: E402
from app.memory import MemoryManager, MemoryRetrieval  # noqa: E402


async def collect_turn(provider, timeout=30):
    texts, audio = [], 0
    deadline = time.monotonic() + timeout
    async for event in provider.events():
        if isinstance(event, TranscriptEvent):
            texts.append(event.text)
        elif isinstance(event, AudioData):
            audio += len(event.data)
        elif isinstance(event, TurnCompleteEvent):
            break
        if time.monotonic() > deadline:
            break
    return " ".join(texts), audio


async def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()

    with tempfile.TemporaryDirectory() as tmp:
        memory = MemoryManager.open(Path(tmp) / "live.db")
        memory.save_contact("Rahul", company="ABC", role="Project Manager")
        memory.save_memory("I prefer Python for AI projects", importance=5)
        memory.save_task("Send proposal to Rahul", deadline="tomorrow")
        brief = MemoryRetrieval(memory).brief()
        print(f"== memory brief ({len(brief)} chars):\n{brief}\n")

        provider = GeminiVoiceProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            memory_context=brief,
        )
        t = time.monotonic()
        await provider.connect()
        print(f"connected in {time.monotonic() - t:.1f}s")
        await provider.send_text(settings.wake_word)
        reply1, audio1 = await collect_turn(provider)
        print(f"wake reply: {reply1!r} audio={audio1}")

        # per-turn context priming (turn_complete=False -> no reply expected)
        await provider.send_context(
            "Relevant stored memory for the conversation:\n"
            "- [contact] Rahul (ABC, Project Manager)"
        )
        await asyncio.sleep(1.0)
        print("context primed (no reply expected)")

        await provider.send_text("Who is Rahul? What company is he from?")
        reply2, audio2 = await collect_turn(provider)
        print(f"memory answer: {reply2!r} audio={audio2}")

        await provider.send_context(
            "Relevant stored memory for the conversation:\n"
            "- [memory] I prefer Python for AI projects"
        )
        await asyncio.sleep(1.0)
        await provider.send_text("What programming language do I prefer?")
        reply3, audio3 = await collect_turn(provider)
        print(f"preference answer: {reply3!r} audio={audio3}")

        await provider.close()
        memory.close()

    ok1 = "rahul" in reply2.lower()
    ok2 = "python" in reply3.lower()
    print(f"\nRESULT: contact_retrieval={'OK' if ok1 else 'FAIL'}, "
          f"fact_retrieval={'OK' if ok2 else 'FAIL'}")
    return 0 if (ok1 and ok2 and audio1 and audio2) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
