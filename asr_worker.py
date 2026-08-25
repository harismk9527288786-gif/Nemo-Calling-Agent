"""ASR decode on a dedicated worker thread.

Step 3 of project_docs/PHASE0_AUDIT_AND_PLAN_24_08_2026.md.

The problem this solves: streaming ASR runs at roughly 1.03x real time on the
target CPU, and it was executing on the same asyncio thread that services the
Gemini Live websocket. Every decode chunk therefore delayed websocket reads,
and every websocket read delayed decode. Nothing could overlap.

The audit's key detail, easy to get wrong: ``push_audio()`` only *buffers*
(src/asr/recognizer.h: "Buffer mono float32 audio (no decode); next() drives
decoding"). The expensive work happens in ``poll_results()`` via stream_next ->
step(). Offloading the push alone would buy nothing at all.

Two invariants govern this design:

1. **Thread affinity.** Every call touching the stream handle -- start_stream,
   push_audio, poll_results, finish_stream -- happens on the one worker thread.
   The C++ engine models a stream as a single-owner object and the ctypes layer
   does no locking, so serializing all access on one thread is both the safe
   and the simple option. The recognizer itself is still created on the main
   thread (that is just a factory for streams).

2. **Ordering.** A flush must be processed *after* all audio submitted before
   it. That is why audio and control messages share one queue instead of the
   obvious two-queue design: with separate queues the flush would overtake
   pending audio and drop the tail of the utterance -- exactly the bug that
   the finish_stream fix just removed. The drop policy below is careful to
   never discard a control message.

Backpressure: at ~1.03x real time there is no headroom, so a long turn can
outrun the decoder. An unbounded queue would convert that into silently
growing latency and eventually memory pressure on an 8GB machine. Instead the
audio queue is bounded; on overflow the OLDEST audio is dropped, because this
is a latency-critical phone call where being current matters more than being
complete, and the drop is counted and reported rather than hidden.

The bound is expressed in **seconds of queued audio**, not in chunk count.
Chunk size is set by the caller's capture blocksize (agent.py's
CHUNK_DURATION), so a count-based bound silently changes meaning if that
constant moves -- 50 chunks is 4s at 80ms but 1s at 20ms. Seconds is what the
policy actually cares about, and it makes the overflow warning report a
backlog a human can reason about. A generous item cap is kept alongside it
purely to bound memory if a caller ever submits degenerate zero-length chunks.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np

# Default bound: 4 seconds of queued audio. Generous on purpose -- normal turns
# are 2-5s and the queue drains while the agent is speaking, so drops should
# indicate genuine overload, not routine operation.
DEFAULT_MAX_AUDIO_SECONDS = 4.0
# Hard item cap, only relevant for degenerate (near-empty) chunks that would
# never move the seconds-based bound.
DEFAULT_MAX_AUDIO_ITEMS = 400
DEFAULT_MAX_RESULTS = 256


class _Flush:
    """Control message: flush the stream, resolve ``future``, then restart."""

    __slots__ = ("future", "loop")

    def __init__(self, future, loop):
        self.future = future
        self.loop = loop


class _Stop:
    """Control message: leave the worker loop."""

    __slots__ = ()


class AsrWorker:
    """Runs a NeMoStreamingASR on its own thread behind bounded queues.

    The asyncio side only ever does non-blocking submits and drains, plus one
    awaitable flush per turn. It never blocks on decode.
    """

    def __init__(
        self,
        asr,
        sample_rate: int = 16000,
        language_code: str = "hi",
        interim_results: bool = True,
        max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
        max_audio_items: int = DEFAULT_MAX_AUDIO_ITEMS,
        max_results: int = DEFAULT_MAX_RESULTS,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self._asr = asr
        self._sample_rate = sample_rate
        self._language_code = language_code
        self._interim_results = interim_results
        self._max_audio_seconds = max_audio_seconds
        self._max_audio_items = max_audio_items
        self._max_samples = int(max_audio_seconds * sample_rate)
        self._max_results = max_results
        self._on_error = on_error or (lambda msg: print(f"[AsrWorker] {msg}"))

        self._q: deque = deque()
        self._cv = threading.Condition()
        self._results: deque = deque()
        self._results_lock = threading.Lock()

        # Queue accounting, maintained O(1) under _cv rather than rescanning the
        # deque on every submit.
        self._queued_items = 0
        self._queued_samples = 0

        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._started = False

        # Counters. Plain ints written only by one thread each, or under a
        # lock; read for reporting, so exactness under races is not critical.
        self.submitted_chunks = 0
        self.dropped_chunks = 0
        self.decoded_chunks = 0
        self.dropped_results = 0
        self.max_queue_depth = 0
        self.max_backlog_seconds = 0.0
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.decode_seconds = 0.0

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Spawn the worker thread. The initial stream is opened *on* it."""
        if self._started:
            return
        self._started = True
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="asr-decode", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Ask the worker to finish and join it. Safe to call more than once."""
        if not self._started:
            return
        with self._cv:
            self._stopping = True
            self._q.append(_Stop())
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Daemon thread, so it will not hold up interpreter exit. Say
                # so rather than pretending the shutdown was clean.
                self._on_error(
                    "decode thread did not exit within "
                    f"{timeout}s; abandoning it (daemon)"
                )
            self._thread = None
        self._started = False

    # ------------------------------------------------------------- asyncio side

    def submit_audio(self, samples: np.ndarray) -> bool:
        """Queue audio for decode. Returns False if a chunk had to be dropped.

        Never blocks and never raises: this is called from the event loop.
        """
        n = len(samples)
        with self._cv:
            dropped = False
            # Make room first, so the queue never exceeds either bound. Loop
            # because one dropped chunk may be smaller than the incoming one.
            while self._queued_items > 0 and (
                self._queued_samples + n > self._max_samples
                or self._queued_items >= self._max_audio_items
            ):
                if not self._drop_oldest_audio_locked():
                    break
                dropped = True

            self._q.append(samples)
            self._queued_items += 1
            self._queued_samples += n
            self.submitted_chunks += 1

            if self._queued_items > self.max_queue_depth:
                self.max_queue_depth = self._queued_items
            backlog = self._queued_samples / float(self._sample_rate)
            if backlog > self.max_backlog_seconds:
                self.max_backlog_seconds = backlog
            self._cv.notify()
        return not dropped

    def drain_results(self) -> List[Tuple[bool, str]]:
        """Pop everything decoded so far. Non-blocking."""
        with self._results_lock:
            if not self._results:
                return []
            out = list(self._results)
            self._results.clear()
        return out

    async def flush_and_restart(self, timeout: float = 3.0) -> str:
        """Flush the stream tail and return the final transcript.

        Ordered behind any audio already queued, so the tail of the utterance
        is included. Awaits rather than blocking the event loop. On timeout the
        caller gets "" and should fall back to the interim text; the worker is
        still instructed to restart, so one slow flush cannot kill the call.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._cv:
            self._q.append(_Flush(future, loop))
            self._cv.notify()

        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self.error_count += 1
            self.last_error = f"flush exceeded {timeout}s"
            self._on_error(
                f"flush did not complete within {timeout}s -- falling back to "
                "interim text for this turn"
            )
            return ""
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._on_error(f"flush failed: {self.last_error}")
            return ""

    def stats(self) -> dict:
        with self._cv:
            queued_items = self._queued_items
            queued_seconds = self._queued_samples / float(self._sample_rate)
        return {
            "submitted_chunks": self.submitted_chunks,
            "decoded_chunks": self.decoded_chunks,
            "dropped_chunks": self.dropped_chunks,
            "dropped_results": self.dropped_results,
            "max_queue_depth": self.max_queue_depth,
            "max_backlog_seconds": round(self.max_backlog_seconds, 3),
            "max_backlog_bound_seconds": self._max_audio_seconds,
            "queued_items_at_exit": queued_items,
            "queued_seconds_at_exit": round(queued_seconds, 3),
            "decode_seconds": round(self.decode_seconds, 3),
            "error_count": self.error_count,
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------- worker side

    def _drop_oldest_audio_locked(self) -> bool:
        """Remove the oldest audio item, preserving control messages.

        Control messages must survive: dropping a _Flush would hang the turn
        waiting on its future, and dropping a _Stop would hang shutdown.
        """
        for i, item in enumerate(self._q):
            if isinstance(item, np.ndarray):
                del self._q[i]
                self._queued_items -= 1
                self._queued_samples -= len(item)
                self.dropped_chunks += 1
                if self.dropped_chunks in (1, 10, 100, 1000):
                    self._on_error(
                        f"audio queue full ({self._max_audio_seconds}s bound); "
                        f"dropped {self.dropped_chunks} chunk(s) so far. Decode "
                        "cannot keep up with capture."
                    )
                return True
        return False

    def _publish(self, results) -> None:
        with self._results_lock:
            for item in results:
                if len(self._results) >= self._max_results:
                    # Oldest interims are superseded by newer ones, so dropping
                    # from the front is the least destructive choice.
                    self._results.popleft()
                    self.dropped_results += 1
                self._results.append(item)

    def _run(self) -> None:
        # Open the stream on this thread so the handle is only ever touched here.
        try:
            self._asr.start_stream(
                interim_results=self._interim_results,
                language_code=self._language_code,
            )
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"initial start_stream: {type(exc).__name__}: {exc}"
            self._on_error(self.last_error)
            return

        while True:
            with self._cv:
                while not self._q:
                    if self._stopping:
                        return
                    # Timed wait so a stop that races the wait is still seen.
                    self._cv.wait(0.1)
                item = self._q.popleft()
                # Keep the accounting in step with the pop, under the same lock
                # the submit side uses.
                if isinstance(item, np.ndarray):
                    self._queued_items -= 1
                    self._queued_samples -= len(item)

            if isinstance(item, _Stop):
                return

            if isinstance(item, _Flush):
                self._handle_flush(item)
                continue

            self._handle_audio(item)

    def _handle_audio(self, samples: np.ndarray) -> None:
        started = time.perf_counter()
        try:
            self._asr.push_audio(samples, sample_rate=self._sample_rate)
            # This is the expensive call -- the mel front-end and encoder run
            # here, not in push_audio.
            results = [(bool(is_final), text) for is_final, text in self._asr.poll_results()]
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"decode: {type(exc).__name__}: {exc}"
            if self.error_count in (1, 10, 100):
                self._on_error(self.last_error)
            return
        finally:
            self.decode_seconds += time.perf_counter() - started

        self.decoded_chunks += 1
        if results:
            self._publish(results)

    def _handle_flush(self, flush: _Flush) -> None:
        """Flush, hand the text back, then reopen the stream.

        The future is resolved *before* the restart so the caller can get on
        with the LLM request while this thread reopens the stream. Because
        items are processed in order, any audio submitted after the flush is
        still handled after the restart completes.
        """
        text = ""
        error: Optional[str] = None
        try:
            text = self._asr.final_transcript()
        except Exception as exc:
            error = f"final_transcript: {type(exc).__name__}: {exc}"
            self.error_count += 1
            self.last_error = error

        self._resolve(flush, text, error)

        # Restart unconditionally. If the flush failed we still need a live
        # stream for the next turn -- otherwise one bad flush ends the call.
        try:
            self._asr.start_stream(
                interim_results=self._interim_results,
                language_code=self._language_code,
            )
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"restart start_stream: {type(exc).__name__}: {exc}"
            self._on_error(self.last_error)

        # A fresh stream means older results describe the previous utterance.
        with self._results_lock:
            self._results.clear()

    @staticmethod
    def _resolve(flush: _Flush, text: str, error: Optional[str]) -> None:
        """Complete the caller's future from this thread, safely."""

        def _set():
            if flush.future.done():
                return  # caller already timed out
            if error is not None:
                flush.future.set_exception(RuntimeError(error))
            else:
                flush.future.set_result(text)

        try:
            flush.loop.call_soon_threadsafe(_set)
        except RuntimeError:
            # Loop already closed (shutdown race). Nobody is waiting.
            pass
