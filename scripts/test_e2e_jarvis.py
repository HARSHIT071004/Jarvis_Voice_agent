"""End-to-end JARVIS test: Jenny TTS wake -> Gemini Live reply (audio out loud)."""

import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from app.agent.provider import (  # noqa: E402
    AudioData,
    GeminiVoiceProvider,
    InterruptedEvent,
    TranscriptEvent,
    TurnCompleteEvent,
)
from app.config import load_settings  # noqa: E402
from app.wake.detector import WakeDetector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("asyncio").setLevel(logging.WARNING)


def load16k(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    data = data.mean(axis=1).astype("<i2")
    if sr != 16000:
        data = np.interp(
            np.arange(int(len(data) * 16000 / sr)), np.arange(len(data)), data
        ).astype("<i2")
    return data


async def play_pcm(pcm: bytes, rate: int = 24000) -> None:
    import sounddevice as sd

    audio = np.frombuffer(pcm, dtype="<i2")
    sd.play(audio, samplerate=rate)
    sd.wait()


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    print(f"== wake word: {settings.wake_word!r}  model: {settings.gemini_model}")

    # --- STAGE 1: wake detection on Jenny's "Jarvis." -----------------
    det = WakeDetector(settings)
    det.initialize()
    audio = load16k("jv_jenny.mp3")
    detected = None
    t0 = time.monotonic()
    for i in range(0, len(audio) - 1600 + 1, 1600):
        if det.process(audio[i : i + 1600].tobytes()):
            detected = time.monotonic() - t0
            break
    if detected is None:
        print("STAGE 1 FAIL: wake word NOT detected -> stopping")
        return
    print(f"STAGE 1 PASS: wake detected after {detected:.2f}s of audio")

    # --- STAGE 2: connect to Gemini and ask --------------------------
    provider = GeminiVoiceProvider(
        api_key=settings.gemini_api_key, model=settings.gemini_model
    )
    t = time.monotonic()
    await provider.connect()
    print(f"STAGE 2: connected in {time.monotonic() - t:.2f}s")
    await provider.send_text(settings.wake_word)

    # --- STAGE 3: collect reply (audio + transcripts), play it back ---
    audio_chunks: list[bytes] = []
    user_tx, pri_tx = [], []
    turn_done = False
    deadline = time.monotonic() + 30
    async for event in provider.events():
        if isinstance(event, AudioData):
            audio_chunks.append(event.data)
        elif isinstance(event, TranscriptEvent):
            if event.role == "user":
                user_tx.append(event.text)
            else:
                pri_tx.append(event.text)
                print(f"  Jarvis: {event.text}")
        elif isinstance(event, InterruptedEvent):
            print("  (barge-in)")
        elif isinstance(event, TurnCompleteEvent):
            turn_done = True
            break
        if time.monotonic() > deadline:
            break
    await provider.close()

    pcm = b"".join(audio_chunks)
    print(f"STAGE 3: turn_complete={turn_done} audio={len(pcm)} bytes "
          f"({len(pcm) // (24000 * 2)}s) user_said={user_tx} reply={pri_tx}")
    if pcm:
        print("playing reply through speaker...")
        await play_pcm(pcm)
    print("RESULT:", "WAKE+REPLY OK" if turn_done and pcm else "PARTIAL/FAIL")


if __name__ == "__main__":
    asyncio.run(main())
