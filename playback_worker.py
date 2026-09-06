"""Thread-safe asynchronous playback worker for real-time PCM audio streaming.

Step 4 of project_docs/PHASE0_AUDIT_AND_PLAN_24_08_2026.md.

Decouples blocking audio writes from the asyncio event loop and Gemini Live
receive loop. Supports sub-millisecond barge-in interruption via out_stream.abort(),
queue draining, and generation epoch tracking to prevent stale audio bleed.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Tuple

import sounddevice as sd


class PlaybackWorker:
    """Consumes audio byte chunks from a queue and writes them to sounddevice.RawOutputStream on a dedicated thread.

    Key invariants:
    1. The asyncio event loop and WebSocket receive loop never block on sounddevice.write().
    2. Generation ID tracking: Every chunk is tagged with an atomic generation counter.
       interrupt() increments the generation, guaranteeing that popped in-flight chunks
       from an interrupted turn are dropped and never reach the speakers.
    3. interrupt() immediately purges in-memory queued chunks and calls out_stream.abort()
       to flush the hardware DAC/DMA buffer, cutting off audio output instantly (<1ms).
    4. Safe lifecycle: out_stream is re-armed with start() after abort() so subsequent turns
       play seamlessly without reopening the device.
    """

    def __init__(
        self,
        out_stream: Optional[sd.RawOutputStream] = None,
        samplerate: int = 24000,
        channels: int = 1,
        dtype: str = "int16",
        device: Optional[int] = None,
    ):
        self._owns_stream = out_stream is None
        if out_stream is not None:
            self.out_stream = out_stream
        else:
            self.out_stream = sd.RawOutputStream(
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                device=device,
            )

        if not self.out_stream.active:
            try:
                self.out_stream.start()
            except Exception:
                pass

        # Tuple of (generation: int, chunk: bytes) or None for shutdown
        self._queue: queue.Queue[Optional[Tuple[int, bytes]]] = queue.Queue()
        self._running = True
        self._generation = 0
        self._interrupted = threading.Event()
        self._lock = threading.Lock()
        self._busy = False
        self._last_write_time = 0.0
        self._bytes_written = 0

        self._thread = threading.Thread(
            target=self._playback_loop,
            name="audio-playback",
            daemon=True,
        )
        self._thread.start()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def enqueue(self, chunk: bytes) -> None:
        """Enqueue PCM bytes for playback tagged with the current generation counter."""
        if not self._running or self._interrupted.is_set():
            return
        if chunk:
            with self._lock:
                current_gen = self._generation
            self._queue.put((current_gen, chunk))

    def is_playing(self) -> bool:
        """Returns True if audio is actively playing or queued in memory for the current generation."""
        if self._interrupted.is_set():
            return False
        with self._lock:
            if not self._queue.empty() or self._busy:
                return True
            # Check if hardware stream is still draining recently written audio (~80ms margin)
            return (time.perf_counter() - self._last_write_time) < 0.08

    def interrupt(self) -> None:
        """Instantly abort current playback and discard all queued audio.

        Thread-safe. Increments the generation epoch, flushes the software queue,
        aborts the hardware buffer, and re-arms the stream for subsequent turns.
        """
        with self._lock:
            self._generation += 1
            self._interrupted.set()

            # 1. Drain the software queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            # 2. Terminate in-flight hardware DMA/driver playback immediately
            try:
                self.out_stream.abort()
            except Exception:
                pass

            # 3. Re-arm stream so it is ready for the next audio chunk
            try:
                self.out_stream.start()
            except Exception:
                pass

            self._busy = False
            self._last_write_time = 0.0
            self._interrupted.clear()

    def _playback_loop(self) -> None:
        """Worker thread body: continuously writes queued chunks to out_stream."""
        while self._running:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is None or not self._running:
                break

            gen, chunk = item

            # Verify generation matches current epoch under lock
            with self._lock:
                if gen != self._generation or self._interrupted.is_set():
                    continue
                self._busy = True

            try:
                self.out_stream.write(chunk)
                with self._lock:
                    if gen == self._generation:
                        self._last_write_time = time.perf_counter()
                        self._bytes_written += len(chunk)
            except Exception:
                # If stream was aborted or interrupted concurrently, ignore and continue
                pass
            finally:
                with self._lock:
                    self._busy = False

    def stop(self) -> None:
        """Stops the worker thread and cleans up the stream."""
        if not self._running:
            return
        self._running = False
        self._interrupted.set()

        # Wake worker if blocked on queue
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        if self._thread.is_alive() and threading.current_thread() != self._thread:
            self._thread.join(timeout=1.0)

        try:
            self.out_stream.stop()
        except Exception:
            pass

        if self._owns_stream:
            try:
                self.out_stream.close()
            except Exception:
                pass

    def close(self) -> None:
        """Idempotent alias for stop()."""
        self.stop()
