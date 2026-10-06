"""Fresh-spotter isolation test: does each file trigger JARVIS on its own?"""

import sys

import numpy as np

sys.path.insert(0, "scripts")
from debug_kws import load16k, make_spotter, run  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    files = [
        "jv_jenny.mp3",
        "jv_sonia.mp3",
        "jv_neerja.mp3",
        "jv_prabhat.mp3",
        "jvn_jenny.mp3",
        "jvn_sonia.mp3",
        "jvn_neerja.mp3",
        "jvn_prabhat.mp3",
        "tts_jenny.mp3",
        "tts_sonia.mp3",
    ]
    for f in files:
        sp = make_spotter(1.5, 0.25)
        r = run(sp, load16k(f))
        print(f"{f:20s} -> {r}")

    sp = make_spotter(1.5, 0.25)
    print(f"{'20s_silence':20s} -> {run(sp, np.zeros(16000 * 20, dtype='<i2'))}")

    sp = make_spotter(1.5, 0.25)
    noise = (np.random.default_rng(0).standard_normal(16000 * 10) * 500).astype("<i2")
    print(f"{'10s_white_noise':20s} -> {run(sp, noise)}")


if __name__ == "__main__":
    main()
