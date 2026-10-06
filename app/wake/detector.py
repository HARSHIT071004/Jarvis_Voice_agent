"""Local wake-word detection.

Runs continuously in standby and never sends audio anywhere: everything is
on-device ONNX inference. Only after a detection does the voice session (and
therefore the network) get involved.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import sherpa_onnx

from app.config import Settings
from app.wake.model_setup import ensure_model

logger = logging.getLogger("jarvis.wake")

_SAMPLE_RATE = 16000


class WakeDetector:
    """Feeds PCM frames to sherpa-onnx and reports wake-phrase hits."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._spotter: sherpa_onnx.KeywordSpotter | None = None
        self._stream: sherpa_onnx.OnlineStream | None = None
        self._model_dir = None
        self._last_trigger = 0.0
        self.refractory_seconds = 2.0

    @property
    def initialized(self) -> bool:
        return self._spotter is not None

    def initialize(self) -> None:
        model_dir, keywords_file = ensure_model(
            self.settings.kws_model_dir,
            self.settings.wake_word,
            self.settings.wake_keyword_score,
            self.settings.wake_threshold,
        )
        self._model_dir = model_dir
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(model_dir / "tokens.txt"),
            encoder=str(model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            decoder=str(model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            joiner=str(model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
            keywords_file=str(keywords_file),
            keywords_score=self.settings.wake_keyword_score,
            keywords_threshold=self.settings.wake_threshold,
            num_threads=2,
            sample_rate=_SAMPLE_RATE,
            provider="cpu",
        )
        self._stream = self._spotter.create_stream()
        logger.info("Wake-word detector initialized for %r", self.settings.wake_word)

    def process(self, frame: bytes) -> str:
        """Feed one PCM frame; return the detected keyword or an empty string."""
        if self._spotter is None or self._stream is None:
            raise RuntimeError("WakeDetector.initialize() must be called first")
        if not frame:
            return ""

        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        self._stream.accept_waveform(_SAMPLE_RATE, samples.tolist())

        detected = ""
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                detected = result.strip()

        if detected:
            now = time.monotonic()
            if now - self._last_trigger < self.refractory_seconds:
                return ""
            self._last_trigger = now
            self._spotter.reset_stream(self._stream)
        return detected

    def close(self) -> None:
        self._stream = None
        self._spotter = None
