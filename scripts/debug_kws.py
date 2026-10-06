"""Debug KeywordSpotter behaviour on JARVIS samples across settings."""

import sys
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

MD = Path("models/kws/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")


def load16k(path):
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    data = data.mean(axis=1).astype("<i2")
    if sr != 16000:
        data = np.interp(
            np.arange(int(len(data) * 16000 / sr)), np.arange(len(data)), data
        ).astype("<i2")
    return data


def make_spotter(score, thr):
    return sherpa_onnx.KeywordSpotter(
        tokens=str(MD / "tokens.txt"),
        encoder=str(MD / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        decoder=str(MD / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        joiner=str(MD / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        keywords_file=str(MD / "keywords.txt"),
        keywords_score=score,
        keywords_threshold=thr,
        num_trailing_blanks=1,
        num_threads=2,
        sample_rate=16000,
        provider="cpu",
    )


def run(spotter, audio):
    stream = spotter.create_stream()
    results = []
    for i in range(0, len(audio) - 1600 + 1, 1600):
        stream.accept_waveform(
            16000, (audio[i : i + 1600].astype(np.float32) / 32768).tolist()
        )
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
            r = spotter.get_result(stream)
            if r:
                results.append(r.strip())
    return results


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pos = ["jv_sonia.mp3", "jv_jenny.mp3", "jv_neerja.mp3", "jv_prabhat.mp3"]
    neg = ["jvn_jenny.mp3"]
    audios = {f: load16k(f) for f in pos + neg}

    for score in (1.5, 3.0, 6.0, 10.0):
        for thr in (0.25, 0.15, 0.08):
            sp = make_spotter(score, thr)
            out = []
            for f in pos:
                r = run(sp, audios[f])
                out.append(f"{f[3:8]}={r}")
            fn = run(sp, audios["jvn_jenny.mp3"])
            print(f"score={score:<5} thr={thr:<5} {' '.join(out)} java={fn}")


if __name__ == "__main__":
    main()
