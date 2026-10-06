"""Wake-word live self-test: records the microphone, runs the detector,
prints hits, and saves the recording for offline tuning.

Usage:  python -m app.wake.selftest [--seconds 60]
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np

from app.config import load_settings
from app.wake.detector import WakeDetector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()

    import sounddevice as sd

    settings = load_settings(require_api_key=False)
    detector = WakeDetector(settings)
    detector.initialize()

    rate = 16000
    chunk = 1600  # 100 ms
    frames: list[bytes] = []
    hits: list[tuple[float, str]] = []
    start = time.monotonic()

    print(f"Recording {args.seconds:.0f}s - say the wake phrase now...", flush=True)

    with sd.InputStream(
        samplerate=rate, channels=1, dtype="int16", blocksize=chunk
    ) as stream:
        deadline = start + args.seconds
        while time.monotonic() < deadline:
            data, _ = stream.read(chunk)
            raw = bytes(data)
            frames.append(raw)
            result = detector.process(raw)
            if result:
                t = time.monotonic() - start
                hits.append((t, result))
                print(f"  [{t:6.2f}s] DETECTED: {result}", flush=True)

    path = "wakeword_live.wav"
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(frames))

    total = len(frames) * chunk / rate
    print(f"\nRecorded {total:.1f}s -> {path}")
    print(f"Detections: {len(hits)}")
    for t, r in hits:
        print(f"  {t:.2f}s  {r}")
    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
