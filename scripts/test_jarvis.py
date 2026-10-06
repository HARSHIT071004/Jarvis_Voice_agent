"""Test the wake detector against JARVIS TTS samples (dev tool)."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from probe_decoding import transcribe  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.wake.detector import WakeDetector  # noqa: E402


def load16k(path):
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    data = data.mean(axis=1).astype("<i2")
    if sr != 16000:
        data = np.interp(
            np.arange(int(len(data) * 16000 / sr)), np.arange(len(data)), data
        ).astype("<i2")
    return data


def count_hits(det, audio):
    hits = 0
    for i in range(0, len(audio) - 1600 + 1, 1600):
        if det.process(audio[i : i + 1600].tobytes()):
            hits += 1
    return hits


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pos = ["jv_neerja.mp3", "jv_prabhat.mp3", "jv_jenny.mp3", "jv_sonia.mp3"]
    neg = [
        "jvn_sonia.mp3",
        "jvn_jenny.mp3",
        "jvn_neerja.mp3",
        "jvn_prabhat.mp3",
        "tts_sonia.mp3",
        "tts_jenny.mp3",
    ]

    settings = load_settings(require_api_key=False)
    print("wake word:", repr(settings.wake_word))
    det = WakeDetector(settings)
    det.initialize()

    print("\n--- what the KWS model HEARS ---")
    for f in pos:
        print(f"  {f:18s} -> {transcribe(f)!r}")

    print("\n--- detection ---")
    total = 0
    for f in pos:
        hits = count_hits(det, load16k(f))
        total += hits
        status = "PASS" if hits else "MISS"
        print(f"  [POS] {f:18s} hits={hits} {status}")

    falses = 0
    for f in neg:
        hits = count_hits(det, load16k(f))
        falses += hits
        status = "FALSE-ACCEPT!" if hits else "ok"
        print(f"  [NEG] {f:18s} hits={hits} {status}")

    silence = np.zeros(16000 * 20, dtype="<i2")
    sh = count_hits(det, silence)
    print(f"  [SIL] 20s silence    hits={sh}")

    print(f"\nRESULT: total wake hits on positives={total}, false accepts={falses + sh}")


if __name__ == "__main__":
    main()
