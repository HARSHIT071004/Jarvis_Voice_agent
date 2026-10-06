"""Test hotword-biased streaming ASR as a wake-word engine.

The idea: a real ASR model + context-graph biasing for the phrase "hey jarvis".
Biasing injects arbitrary (out-of-vocabulary) phrases into the decoder, which
is exactly what tiny KWS models cannot do.
"""

import sys
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

MD = Path("models/kws/asr-en-2023-06-26")
PHRASE = "hey jarvis"


def load16k(path):
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    data = data.mean(axis=1).astype("<i2")
    if sr != 16000:
        data = np.interp(
            np.arange(int(len(data) * 16000 / sr)), np.arange(len(data)), data
        ).astype("<i2")
    return data


def make_recognizer(hotwords_score):
    hw = MD / "hotwords.txt"
    hw.write_text(PHRASE + "\n", encoding="utf-8")
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(MD / "tokens.txt"),
        encoder=str(MD / "encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx"),
        decoder=str(MD / "decoder-epoch-99-avg-1-chunk-16-left-64.onnx"),
        joiner=str(MD / "joiner-epoch-99-avg-1-chunk-16-left-64.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        provider="cpu",
        decoding_method="modified_beam_search",
        hotwords_file=str(hw),
        hotwords_score=hotwords_score,
        modeling_unit="bpe",
        bpe_vocab=str(MD / "bpe_vocab.txt"),
    )


def decode_incremental(rec, audio, wake_word="jarvis", chunk=1600):
    """Feed audio in 100 ms chunks; return (first_match_time_s, final_text)."""
    stream = rec.create_stream()
    first_match = None
    t = 0.0
    for i in range(0, len(audio) - chunk + 1, chunk):
        stream.accept_waveform(
            16000, (audio[i : i + chunk].astype(np.float32) / 32768).tolist()
        )
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        text = rec.get_result(stream).lower()
        if wake_word in text and first_match is None:
            first_match = t
        t += chunk / 16000
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    return first_match, rec.get_result(stream).strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pos = ["tts_neerja.mp3", "tts_prabhat.mp3", "tts_jenny.mp3", "tts_sonia.mp3"]
    neg = ["neg_sonia1.mp3", "neg_jenny1.mp3", "neg_neerja1.mp3", "neg_prabhat1.mp3"]

    for score in (5.0, 10.0, 20.0):
        print(f"\n=== hotwords_score={score} ===")
        rec = make_recognizer(score)
        for f in pos + neg:
            audio = load16k(f)
            match_t, text = decode_incremental(rec, audio)
            kind = "POS" if f.startswith("tts") else "NEG"
            hit = "HIT " if match_t is not None else "miss"
            flag = ""
            if kind == "NEG" and match_t is not None:
                flag = "  <-- FALSE ACCEPT"
            print(f"  [{kind}] {hit} {f:20s} at={match_t}  text={text!r}{flag}")


if __name__ == "__main__":
    main()
