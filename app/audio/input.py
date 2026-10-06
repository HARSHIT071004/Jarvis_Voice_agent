"""Microphone input abstraction.

One persistent 16 kHz mono int16 stream is opened for the lifetime of the
process. The wake-word detector and the voice session both consume from the
same async iterator, so we never reopen the device mid-conversation (a common
source of clicks, glitches and lost audio on Windows).

Hardware access is isolated here so tests can substitute a fake microphone.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import sounddevice as sd

logger = logging.getLogger("jarvis.audio.input")


class AudioDeviceError(Exception):
    """Raised when the microphone cannot be opened or read."""


class Microphone:
    def __init__(
        self,
        sample_rate: int = 16000,
        device: int | None = None,
        chunk_ms: int = 100,
        queue_size: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.samples_per_chunk = int(sample_rate * chunk_ms / 1000)
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dropped = 0

    def probe(self) -> None:
        """Verify a microphone exists before starting the app."""
        try:
            devices = sd.query_devices()
        except Exception as exc:
            raise AudioDeviceError(f"Audio system unavailable: {exc}") from exc
        if self.device is not None:
            info = sd.query_devices(self.device)
            if info["max_input_channels"] < 1:
                raise AudioDeviceError(f"Device {self.device} is not an input device")
            return
        if not any(d["max_input_channels"] > 0 for d in devices):
            raise AudioDeviceError("No microphone found")

    def open(self) -> None:
        if self._stream is not None:
            return
        self._loop = asyncio.get_running_loop()
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.samples_per_chunk,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioDeviceError(f"Could not open microphone: {exc}") from exc
        logger.info("Microphone opened at %d Hz", self.sample_rate)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.warning("Microphone status: %s", status)
        data = bytes(indata)
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._enqueue, data)
        except RuntimeError:
            pass  # event loop is closing

    def _enqueue(self, data: bytes) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped += 1
                if self._dropped % 50 == 1:
                    logger.warning("Microphone queue full, dropped %d frames", self._dropped)
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(data)

    async def frames(self) -> AsyncIterator[bytes]:
        """Yield 16-bit PCM chunks. Must be consumed by exactly one task."""
        if self._stream is None:
            self.open()
        while True:
            yield await self._queue.get()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.warning("Error while closing microphone", exc_info=True)
            logger.info("Microphone closed")

    async def __aenter__(self) -> Microphone:
        self.open()
        return self

    async def __aexit__(self, *exc) -> None:
        self.close()
