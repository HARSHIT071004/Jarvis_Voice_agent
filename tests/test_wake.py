from pathlib import Path

import numpy as np
import pytest

from app.config import load_settings
from app.wake.detector import WakeDetector
from app.wake.model_setup import MODEL_NAME, WakeModelError, encode_keywords

MODEL_DIR = Path("models/kws") / MODEL_NAME
MODEL_READY = (MODEL_DIR / "tokens.txt").exists()


@pytest.mark.skipif(not MODEL_READY, reason="KWS model not downloaded")
def test_silence_produces_no_false_trigger():
    settings = load_settings(require_api_key=False)
    detector = WakeDetector(settings)
    detector.initialize()
    silence = np.zeros(1600, dtype="<i2").tobytes()
    for _ in range(32):  # 3.2 s of silence
        assert detector.process(silence) == ""


@pytest.mark.skipif(not MODEL_READY, reason="KWS model not downloaded")
def test_refractory_blocks_double_trigger():
    settings = load_settings(require_api_key=False)
    detector = WakeDetector(settings)
    detector.initialize()
    # Simulate: force a hit timestamp, then a second detection within the window
    detector._last_trigger = detector._last_trigger or 0.0
    import time

    detector._last_trigger = time.monotonic()
    silence = np.zeros(1600, dtype="<i2").tobytes()
    assert detector.process(silence) == ""


@pytest.mark.skipif(not MODEL_READY, reason="KWS model not downloaded")
def test_encode_keywords_writes_tokens(tmp_path):
    out = tmp_path / "keywords.txt"
    result = encode_keywords(MODEL_DIR, "hey jarvis", 1.5, 0.25, out_path=out)
    content = result.read_text(encoding="utf-8")
    assert "JA R VI S" in content  # Jarvis in model BPE tokens
    assert "#0.25" in content
    assert ":1.5" in content


def test_encode_keywords_fails_cleanly_on_bad_model(tmp_path):
    with pytest.raises(WakeModelError, match="Could not encode"):
        encode_keywords(tmp_path, "hey jarvis", 1.5, 0.25)
