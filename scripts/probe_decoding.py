"""Transcribe audio with the KWS model itself (it is a tiny streaming ASR)
to see what token sequence the model actually produces for the wake phrase."""

import numpy as np
import soundfile as sf
import sherpa_onnx
from pathlib import Path

MD = Path("models/kws/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")


def load16k(path):
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    data = data.mean(axis=1).astype("<i2")
    if sr != 16000:
        data = np.interp(
            np.arange(int(len(data) * 16000 / sr)), np.arange(len(data)), data
        ).astype("<i2")
    return data


def transcribe(path):
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(MD / "tokens.txt"),
        encoder=str(MD / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        decoder=str(MD / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        joiner=str(MD / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        provider="cpu",
    )
    stream = rec.create_stream()
    audio = load16k(path)
    stream.accept_waveform(16000, (audio.astype(np.float32) / 32768).tolist())
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    return rec.get_result(stream).strip()


if __name__ == "__main__":
    files = [
        "tts_neerja.mp3",
        "tts_prabhat.mp3",
        "tts_jenny.mp3",
        "tts_sonia.mp3",
        "neg_sonia1.mp3",
        "neg_jenny1.mp3",
        "neg_neerja1.mp3",
        "neg_prabhat1.mp3",
        "wakeword_test.wav",
    ]
    for f in files:
        if Path(f).exists():
            print(f"{f:22s} -> {transcribe(f)!r}")
