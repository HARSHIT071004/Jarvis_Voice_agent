"""Parameter sweep for the wake detector (development tool)."""

import numpy as np
import soundfile as sf

from app.config import load_settings
from app.wake.detector import WakeDetector


def load16k(path):
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    data = data.mean(axis=1).astype("<i2")
    if sr != 16000:
        data = np.interp(
            np.arange(int(len(data) * 16000 / sr)), np.arange(len(data)), data
        ).astype("<i2")
    return data


POS = ["tts_neerja.mp3", "tts_prabhat.mp3", "tts_jenny.mp3", "tts_sonia.mp3"]
NEG = ["neg_sonia1.mp3", "neg_jenny1.mp3", "neg_neerja1.mp3", "neg_prabhat1.mp3"]


def run(det, audio):
    hits = 0
    for i in range(0, len(audio) - 1600 + 1, 1600):
        if det.process(audio[i : i + 1600].tobytes()):
            hits += 1
    return hits


def main():
    pos = {f: load16k(f) for f in POS}
    neg = {f: load16k(f) for f in NEG}
    silence = np.zeros(16000 * 30, dtype="<i2")
    settings = load_settings(require_api_key=False)

    header = f"{'score':>5} {'thr':>5} | positives | false-accepts"
    print(header)
    for score in (1.5, 2.5, 3.5, 5.0):
        for thr in (0.25, 0.15, 0.10, 0.05):
            settings.wake_keyword_score = score
            settings.wake_threshold = thr
            det = WakeDetector(settings)
            det.initialize()
            ph = {f: run(det, a) for f, a in pos.items()}
            det = WakeDetector(settings)
            det.initialize()
            nh = {f: run(det, a) for f, a in neg.items()}
            det = WakeDetector(settings)
            det.initialize()
            sh = run(det, silence)
            npass = sum(1 for v in ph.values() if v)
            nfalse = sum(nh.values()) + sh
            caught = [k[4:11] for k, v in ph.items() if v]
            neg_hits = {k[4:11]: v for k, v in nh.items() if v}
            print(
                f"{score:5.1f} {thr:5.2f} | {npass}/4 {caught} | "
                f"false={nfalse} {neg_hits} sil={sh}"
            )


if __name__ == "__main__":
    main()
