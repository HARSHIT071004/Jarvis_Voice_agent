"""Speaker output abstraction.

Plays raw PCM from an asyncio queue on a dedicated task. Exposes clear() so a
barge-in (user interrupts Jarvis) can flush undelivered audio immediately.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger("jarvis.audio.output")


class AudioDeviceError(Exception):
    """Raised when the speaker cannot be opened or written."""


class Speaker:
    def __init__(
        self,
        sample_rate: int = 24000,
        device: int | None = None,
        queue_size: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task | None = None
        self._running = False
        self._writing = asyncio.Event()
        self.idle = asyncio.Event()
        self.idle.set()

    def probe(self) -> None:
        try:
            devices = sd.query_devices()
        except Exception as exc:
            raise AudioDeviceError(f"Audio system unavailable: {exc}") from exc
        if self.device is not None:
            info = sd.query_devices(self.device)
            if info["max_output_channels"] < 1:
                raise AudioDeviceError(f"Device {self.device} is not an output device")
            return
        if not any(d["max_output_channels"] > 0 for d in devices):
            raise AudioDeviceError("No speaker found")

    async def start(self) -> None:
        if self._task is not None:
            return
        self.probe()
        self._running = True
        self._task = asyncio.create_task(self._run(), name="speaker-writer")
        logger.info("Speaker opened at %d Hz", self.sample_rate)

    async def _run(self) -> None:
        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=self.device,
                blocksize=0,
            )
            stream.start()
        except Exception as exc:
            self._running = False
            logger.error("Could not open speaker: %s", exc)
            return
        try:
            while self._running:
                chunk = await self._queue.get()
                self.idle.clear()
                self._writing.set()
                try:
                    audio = np.frombuffer(chunk, dtype="<i2")
                    await asyncio.to_thread(stream.write, audio)
                except Exception:
                    logger.warning("Speaker write failed", exc_info=True)
                finally:
                    self._writing.clear()
                if self._queue.empty():
                    self.idle.set()
        except asyncio.CancelledError:
            pass
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.warning("Error while closing speaker", exc_info=True)

    def play(self, pcm: bytes) -> None:
        """Queue a PCM chunk; drops it silently if the queue is saturated."""
        if not self._running or not pcm:
            return
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            logger.warning("Speaker queue full, dropping audio chunk")

    def clear(self) -> None:
        """Flush queued audio (used for interruption/barge-in)."""
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        if cleared:
            logger.debug("Cleared %d queued audio chunks", cleared)
        if not self._writing.is_set():
            self.idle.set()

    async def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until everything queued has been played."""
        try:
            await asyncio.wait_for(self.idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def stop(self) -> None:
        self._running = False
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.clear()
        logger.info("Speaker closed")
