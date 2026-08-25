"""Concurrency tests for AsrWorker against a stub recognizer.

Runs without the DLL, so it works anywhere. The properties tested here are the
ones that fail *silently* in production: wrong ordering loses the tail of an
utterance, a dropped control message hangs a turn, and a missed restart kills
every turn after the first.

    python test_asr_worker.py
"""

import asyncio
import sys
import threading
import time

import numpy as np

from asr_worker import AsrWorker


class StubAsr:
    """Records the thread and order of every call, and can be made to fail."""

    def __init__(self, decode_delay=0.0, fail_flush=False, fail_restart=False,
                 hang_flush=0.0):
        self.calls = []
        self.threads = set()
        self.decode_delay = decode_delay
        self.fail_flush = fail_flush
        self.fail_restart = fail_restart
        self.hang_flush = hang_flush
        self.pushed_samples = 0
        self.stream_open = False
        self.starts = 0
        self._lock = threading.Lock()

    def _log(self, name):
        with self._lock:
            self.calls.append(name)
            self.threads.add(threading.current_thread().name)

    def start_stream(self, interim_results=True, language_code=None):
        self._log("start_stream")
        self.starts += 1
        if self.fail_restart and self.starts > 1:
            raise RuntimeError("simulated restart failure")
        self.stream_open = True

    def push_audio(self, samples, sample_rate=16000):
        self._log("push")
        if not self.stream_open:
            raise RuntimeError("Stream is not started")
        self.pushed_samples += len(samples)

    def poll_results(self):
        self._log("poll")
        if self.decode_delay:
            time.sleep(self.decode_delay)
        yield False, f"interim {self.pushed_samples}"

    def final_transcript(self):
        self._log("final")
        if self.hang_flush:
            time.sleep(self.hang_flush)
        if self.fail_flush:
            raise RuntimeError("simulated flush failure")
        return f"FINAL after {self.pushed_samples} samples"

    def close(self):
        self._log("close")


def chunk(n=1280):
    return np.zeros(n, dtype=np.float32)


results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))


async def test_ordering_flush_after_audio():
    """The flush must see every sample submitted before it.

    This is the invariant that a two-queue design would break, silently
    truncating the end of each utterance.
    """
    asr = StubAsr(decode_delay=0.01)
    w = AsrWorker(asr)
    w.start()
    for _ in range(10):
        w.submit_audio(chunk(1280))
    text = await w.flush_and_restart(timeout=5.0)
    w.stop()

    check("flush returns final text", text.startswith("FINAL"), text)
    check("flush saw all 12800 samples", "12800" in text, text)
    order = [c for c in asr.calls if c in ("push", "final")]
    check("final comes after every push", order[-1] == "final" and order.count("push") == 10,
          f"{order.count('push')} pushes, last call {order[-1]}")


async def test_decode_off_event_loop():
    """The event loop must keep ticking while decode is busy."""
    asr = StubAsr(decode_delay=0.05)   # 50ms per chunk
    w = AsrWorker(asr)
    w.start()

    ticks = 0
    stop = False

    async def ticker():
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.005)

    t = asyncio.create_task(ticker())
    for _ in range(10):          # 10 chunks x 50ms = ~500ms of decode
        w.submit_audio(chunk())
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.4)
    stop = True
    await t
    w.stop()

    # If decode were inline, the loop would have been blocked ~500ms and
    # managed only a handful of ticks.
    check("event loop kept running during decode", ticks > 40, f"{ticks} ticks")
    check("decode ran on the worker thread", asr.threads == {"asr-decode"}, str(asr.threads))


async def test_drop_policy_preserves_control():
    """Overflow drops audio but never the flush, and is counted."""
    asr = StubAsr(decode_delay=0.05)
    # 0.4s bound == 5 chunks of 80ms, so 40 chunks will overflow hard.
    w = AsrWorker(asr, max_audio_seconds=0.4)
    w.start()
    for _ in range(40):
        w.submit_audio(chunk())
    text = await w.flush_and_restart(timeout=5.0)
    w.stop()
    s = w.stats()

    check("dropped chunks were counted", s["dropped_chunks"] > 0, str(s["dropped_chunks"]))
    check("backlog never exceeded its bound",
          s["max_backlog_seconds"] <= 0.4 + 1e-9, str(s["max_backlog_seconds"]))
    check("flush still completed despite drops", text.startswith("FINAL"), text)
    check("submitted == decoded + dropped",
          s["submitted_chunks"] == s["decoded_chunks"] + s["dropped_chunks"],
          f"{s['submitted_chunks']} vs {s['decoded_chunks']}+{s['dropped_chunks']}")
    check("queue accounting returned to zero",
          s["queued_items_at_exit"] == 0 and s["queued_seconds_at_exit"] == 0.0,
          f"{s['queued_items_at_exit']} items, {s['queued_seconds_at_exit']}s")


async def test_bound_is_seconds_not_chunks():
    """The bound must mean the same thing at any capture chunk size.

    This is the regression guard for coupling the bound to agent.py's
    CHUNK_DURATION: at 20ms chunks a count-based bound of 50 would hold only
    1s, not the 4s it claims.
    """
    for chunk_ms, n_samples in ((80, 1280), (20, 320)):
        asr = StubAsr(decode_delay=0.05)
        w = AsrWorker(asr, max_audio_seconds=0.4)
        w.start()
        for _ in range(60):
            w.submit_audio(chunk(n_samples))
        s_live = w.stats()
        await w.flush_and_restart(timeout=5.0)
        w.stop()
        check(f"{chunk_ms}ms chunks respect the 0.4s bound",
              s_live["max_backlog_seconds"] <= 0.4 + 1e-9,
              f"peak {s_live['max_backlog_seconds']}s")


async def test_flush_failure_still_restarts():
    """A failed flush must not leave the recognizer dead for later turns."""
    asr = StubAsr(fail_flush=True)
    w = AsrWorker(asr, on_error=lambda m: None)
    w.start()
    w.submit_audio(chunk())
    text = await w.flush_and_restart(timeout=5.0)

    check("failed flush returns empty (caller falls back to interim)", text == "", repr(text))
    await asyncio.sleep(0.15)
    check("stream was reopened after the failure", asr.starts >= 2, f"starts={asr.starts}")

    # And the next turn still works.
    w.submit_audio(chunk())
    await asyncio.sleep(0.15)
    w.stop()
    check("decode still works after a failed flush", w.stats()["decoded_chunks"] >= 2,
          str(w.stats()["decoded_chunks"]))


async def test_flush_timeout_does_not_wedge():
    """A slow flush times out, and a late result does not blow up."""
    asr = StubAsr(hang_flush=0.6)
    w = AsrWorker(asr, on_error=lambda m: None)
    w.start()
    w.submit_audio(chunk())
    start = time.perf_counter()
    text = await w.flush_and_restart(timeout=0.2)
    elapsed = time.perf_counter() - start

    check("flush timed out promptly", elapsed < 0.45, f"{elapsed*1000:.0f}ms")
    check("timeout yields empty string", text == "", repr(text))
    # The worker resolves the future late; must not raise into the loop.
    await asyncio.sleep(0.7)
    w.stop()
    check("late resolution did not raise", True)
    check("stream reopened after timeout", asr.starts >= 2, f"starts={asr.starts}")


async def test_results_cleared_on_restart():
    """Interims from the previous utterance must not leak into the next turn."""
    asr = StubAsr()
    w = AsrWorker(asr)
    w.start()
    w.submit_audio(chunk())
    await asyncio.sleep(0.15)
    await w.flush_and_restart(timeout=5.0)
    await asyncio.sleep(0.15)
    leftover = w.drain_results()
    w.stop()
    check("no stale interims after restart", leftover == [], str(leftover))


async def test_stop_is_idempotent():
    asr = StubAsr()
    w = AsrWorker(asr)
    w.start()
    w.submit_audio(chunk())
    await asyncio.sleep(0.1)
    w.stop()
    w.stop()
    check("double stop is safe", True)
    check("worker thread exited", not any(
        t.name == "asr-decode" for t in threading.enumerate()))


async def main():
    tests = [
        test_ordering_flush_after_audio,
        test_decode_off_event_loop,
        test_drop_policy_preserves_control,
        test_bound_is_seconds_not_chunks,
        test_flush_failure_still_restarts,
        test_flush_timeout_does_not_wedge,
        test_results_cleared_on_restart,
        test_stop_is_idempotent,
    ]
    for t in tests:
        print(f"\n{t.__name__}:")
        try:
            await t()
        except Exception as exc:
            check(f"{t.__name__} raised", False, f"{type(exc).__name__}: {exc}")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'=' * 50}\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
