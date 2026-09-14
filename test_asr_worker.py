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
    # managed only a handful of ticks (< 5). On Windows with 15.6ms timer resolution,
    # 500ms yields ~30-36 ticks.
    check("event loop kept running during decode", ticks >= 25, f"{ticks} ticks")
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

    check("failed flush returns empty (no interim guess)", text == "", repr(text))
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


async def test_typed_finalization_and_late_interims():
    class TypedAsr(StubAsr):
        def finish_stream(self):
            time.sleep(0.03)
            return [(False, "sofa ka"), (True, "सोफा का starting price क्या है?")]

    w = AsrWorker(TypedAsr(decode_delay=0.04))
    w.start()
    for _ in range(3):
        w.submit_audio(chunk())
    result = await w.finalize_turn(timeout=2)
    w.stop()
    check("late final is retained instead of early empty snapshot", result.status == "final" and "starting price" in result.text)
    check("queue wait is measured separately from native flush", result.queue_wait_ms >= 80 and result.flush_ms >= 20)
    check("restart barrier leaves no published stale results", w.drain_results() == [])

    class InterimOnly(StubAsr):
        def finish_stream(self):
            return [(False, "sofa ka")]

    w = AsrWorker(InterimOnly())
    w.start()
    w.submit_audio(chunk())
    result = await w.finalize_turn(timeout=2)
    w.stop()
    check("promoted interim flush is labelled final", result.status == "final" and "sofa ka" in result.text and result.interim == "sofa ka")


async def test_late_publication_is_fenced_before_restart():
    polled = threading.Event()
    release = threading.Event()

    class DelayedAsr(StubAsr):
        def poll_results(self):
            polled.set()
            release.wait(2)
            yield False, "previous caller text"

    w = AsrWorker(DelayedAsr(hang_flush=0.15), on_error=lambda msg: None)
    w.start()
    w.submit_audio(chunk())
    await asyncio.to_thread(polled.wait, 1)
    result = await w.finalize_turn(timeout=0.02)
    check("hard timeout is explicit and contains no interim guess", result.status == "timeout" and not result.text)
    release.set()
    await asyncio.sleep(0.05)  # old poll completed, but restart is still pending
    check("old interims cannot leak during timed-out flush", w.drain_results() == [])
    await asyncio.sleep(0.2)
    w.submit_audio(chunk())
    next_result = await w.finalize_turn(timeout=2)
    w.stop()
    check("next generation still finalizes after timeout", next_result.status == "final")


async def test_decode_error_and_restart_error_reject_text():
    class BrokenDecode(StubAsr):
        def poll_results(self):
            raise RuntimeError("native encoder failed")

    for recognizer in (BrokenDecode(), StubAsr(fail_restart=True)):
        w = AsrWorker(recognizer, on_error=lambda msg: None)
        w.start()
        w.submit_audio(chunk())
        result = await w.finalize_turn(timeout=2)
        w.stop()
        check("decode/restart error rejects otherwise plausible final", result.status == "error" and not result.text and bool(result.error))


async def test_oversized_audio_and_worker_liveness():
    w = AsrWorker(StubAsr(), max_audio_seconds=0.1)
    check("oversized block explicitly rejected", not w.submit_audio(chunk(3200)))
    check("oversized block cannot bypass backlog bound", w.stats()["max_backlog_seconds"] == 0 and w.dropped_chunks == 1)
    result = await w.finalize_turn(timeout=0.01)
    check("unstarted worker fails without a timeout wait", result.status == "error")


def load_agent_without_hardware():
    """Import the real call loop with device/cloud/native boundaries stubbed."""
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    fake_types = MagicMock()
    fake_types.Content.side_effect = lambda **kw: SimpleNamespace(**kw)
    fake_types.Part.side_effect = lambda **kw: SimpleNamespace(**kw)
    genai = SimpleNamespace(types=fake_types, Client=MagicMock())
    modules = {
        "sounddevice": MagicMock(),
        "nemo_asr": SimpleNamespace(NeMoStreamingASR=MagicMock()),
        "google": SimpleNamespace(genai=genai),
        "google.genai": genai,
        "google.genai.types": fake_types,
    }
    spec = importlib.util.spec_from_file_location("agent_under_test", Path(__file__).with_name("agent.py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


async def test_real_call_loop_with_buffered_turns():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch
    from pipeline_metrics import TurnMetrics

    module = load_agent_without_hardware()
    # Includes greeting-only, usable finals, empty and interim-only turns.
    transcripts = ["hello", "सोफा का starting price क्या है?", "Compact wala", "", "interim only"]
    completed = []

    class MemoryMetrics:
        def __init__(self, **kw):
            self.turns = []

        def start_turn(self, kind):
            turn = TurnMetrics(len(self.turns) + 1, "test", kind=kind)
            self.turns.append(turn)
            return turn

        def finish_turn(self, turn):
            completed.append(turn)

        def close(self):
            pass

    class TypedAsr(StubAsr):
        def __init__(self, **kw):
            super().__init__(decode_delay=0.01)
            self.turn_samples = []
            self.utterances = []

        def start_stream(self, **kw):
            super().start_stream(**kw)
            self.turn_samples = []

        def push_audio(self, samples, **kw):
            super().push_audio(samples, **kw)
            self.turn_samples.extend(samples.tolist())

        def finish_stream(self):
            self.utterances.append(self.turn_samples[:])
            text = transcripts[self.starts - 1]
            return [(self.starts != 5, text)]

    class FakeVAD:
        using_fallback = False
        silence_threshold = 0.35

        def __init__(self, **kw):
            self.index = 0

        def reset(self):
            pass

        def process_chunk(self, samples, **kw):
            # We now process audio during playback to track VAD state properly.
            self.index += 1
            speech_start = self.index % 2 == 1
            barge_in = kw.get("agent_active", False) and speech_start
            return SimpleNamespace(
                is_speech=True, speech_start=speech_start,
                speech_probability=0.9, barge_in=barge_in,
                endpoint_detected=self.index % 2 == 0, audio_for_asr=samples,
            )

    sent = []
    session = MagicMock()

    async def send(**kw):
        sent.append(kw["turns"].parts[0].text)
        # Reproduce the Windows scheduling race on every platform: the old
        # loop dequeued and discarded caller blocks during this send window.
        agent._audio_callback(np.full(1280, 0.9, np.float32), 1280, None, None)
        await asyncio.sleep(0.05)
        playback.is_playing.return_value = True
        agent.is_speaking = False
        # Generation has ended, but the speaker still has queued PCM. A second
        # capture must be rejected while earlier caller blocks stay queued.
        agent._audio_callback(np.full(1280, 0.9, np.float32), 1280, None, None)
        await asyncio.sleep(0.05)
        playback.is_playing.return_value = False

    session.send_client_content = AsyncMock(side_effect=send)
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=session)
    connection.__aexit__ = AsyncMock(return_value=False)
    module.genai.Client.return_value.aio.live.connect.return_value = connection
    module.sd.query_devices.return_value = {"name": "fake mono headset", "default_samplerate": 16000}
    playback = MagicMock()
    playback.is_playing.return_value = False
    module.PlaybackWorker = MagicMock(return_value=playback)

    with patch.object(module, "NeMoStreamingASR", TypedAsr), \
         patch.object(module, "SileroVADEndpointer", FakeVAD), \
         patch.object(module, "MetricsLogger", MemoryMetrics), \
         patch.object(module.CallingAgent, "_load_secret", return_value="unit-test-not-a-key"):
        agent = module.CallingAgent()
    agent._session_resumption_handle = "unit-test-resumed-session"
    agent.disable_barge_in = False


    def input_enter():
        # Entire multi-turn backlog arrives at once: each VAD boundary MUST
        # reach ASR before the next pair of blocks is submitted.
        for i in range(len(transcripts)):
            for _ in range(2):
                agent._audio_callback(np.full(1280, (i + 1) / 10, np.float32), 1280, None, None)

    module.sd.InputStream.return_value.__enter__.side_effect = input_enter

    async def receive_forever(_):
        await asyncio.Future()

    agent._receive_loop = receive_forever

    async def stop_after_processing():
        for _ in range(200):
            if len(agent.asr.utterances) >= len(transcripts):
                await asyncio.sleep(0.03)
                agent.is_running = False
                return
            await asyncio.sleep(0.01)
        raise AssertionError("call loop did not consume all simulated turns")

    stopper = asyncio.create_task(stop_after_processing())
    try:
        await asyncio.wait_for(agent._run_once(), 3)
        await stopper
    finally:
        stopper.cancel()
        await asyncio.gather(stopper, return_exceptions=True)
    check("real loop sends only usable finals, ignoring hello/empty", sent == transcripts[1:3] + [transcripts[4]], repr(sent))
    check("one persistent Gemini connection for all caller turns", module.genai.Client.return_value.aio.live.connect.call_count == 1)
    check("buffered VAD turns preserve all 10 blocks in order", (
        len(agent.asr.utterances) == 5
        and agent.asr_worker.submitted_chunks == 10
        and agent.asr_worker.dropped_chunks == 0
        and all(len(x) == 2560 and np.allclose(x, (i + 1) / 10)
                for i, x in enumerate(agent.asr.utterances))
    ))

    check("diagnostics serialized for rejected and accepted turns", len(completed) == 5 and all(t.to_dict()["asr_diagnostics"] for t in completed))


async def test_capture_echo_overflow_and_response_filtering():
    import queue
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    module = load_agent_without_hardware()
    agent = module.CallingAgent.__new__(module.CallingAgent)
    agent.audio_q = queue.Queue(maxsize=1)
    agent._input_dropped = agent._input_overflows = agent._echo_suppressed = 0
    agent._mic_resume_after = 0
    agent.disable_barge_in = True
    agent.is_speaking = True
    agent.playback = None
    agent._audio_callback(chunk(), 1280, None, None)
    agent._mic_resume_after = 0

    agent._emit_turn_event = MagicMock()
    session = MagicMock()
    await agent.generate_gemini_native_live_audio(session, "  ")
    check("empty transcript never invokes Gemini", not session.send_client_content.called)

    def part(data=None, mime="audio/pcm;rate=24000", thought=False, text=None):
        return SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type=mime), thought=thought, text=text)

    messages = [SimpleNamespace(server_content=SimpleNamespace(
        interrupted=False, turn_complete=True,
        model_turn=SimpleNamespace(parts=[
            part(text="Refining the Welcome"),
            part(b"hidden", thought=True), part(b"not audio", mime="text/plain"), part(b"safe pcm"),
        ]),
    ))]
    receives = 0

    async def receive():
        nonlocal receives
        receives += 1
        if receives == 1:
            for msg in messages:
                yield msg

    agent.current_turn_id = "test"
    agent._active_turn_metrics = lambda: None
    agent._gemini_event_logged = agent._gemini_audio_logged = False
    agent._audio_chunks_received = 0
    agent._turn_interrupted = False
    agent._mark_turn = MagicMock()
    agent._mark_playback_started = MagicMock()
    agent._finish_turn = MagicMock()
    agent._interruption_boundary_event = None
    agent.is_running = False
    agent.playback = MagicMock()
    await agent._receive_loop(SimpleNamespace(receive=receive))
    check("only caller PCM plays, never text/thought/non-audio parts", agent.playback.enqueue.call_args_list == [((b"safe pcm",), {})])


async def test_soft_deadline_retains_final_and_native_errors_surface():
    import importlib.util
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    module = load_agent_without_hardware()
    module.ASR_SLOW_SECONDS = 0.01
    module.ASR_FLUSH_TIMEOUT = 1.0
    agent = module.CallingAgent.__new__(module.CallingAgent)
    agent._emit_turn_event = MagicMock()
    agent.asr_worker = AsrWorker(StubAsr(hang_flush=0.06))
    agent.asr_worker.start()
    agent.asr_worker.submit_audio(chunk())
    result = await agent._finalize_asr_turn("slow-turn")
    agent.asr_worker.stop()
    check("soft latency deadline logs slow but retains eventual final", result.status == "final" and agent._emit_turn_event.call_args.args[1] == "ASR_SLOW")

    spec = importlib.util.spec_from_file_location("native_asr_under_test", Path(__file__).with_name("nemo_asr.py"))
    native = importlib.util.module_from_spec(spec)
    lib = MagicMock()
    lib.nemo_speech_asr_stream_next.return_value = 1
    lib.nemo_speech_asr_last_error.return_value = b"encoder failure"
    with patch("ctypes.CDLL", return_value=lib):
        spec.loader.exec_module(native)
    recognizer = native.NeMoStreamingASR.__new__(native.NeMoStreamingASR)
    recognizer._stream = 1
    recognizer._recognizer = None
    try:
        list(recognizer.poll_results())
        check("native decode error is not disguised as empty ASR", False)
    except RuntimeError as exc:
        check("native decode error is not disguised as empty ASR", "encoder failure" in str(exc))
    finally:
        recognizer.close()


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
        test_typed_finalization_and_late_interims,
        test_late_publication_is_fenced_before_restart,
        test_decode_error_and_restart_error_reject_text,
        test_oversized_audio_and_worker_liveness,
        test_real_call_loop_with_buffered_turns,
        test_capture_echo_overflow_and_response_filtering,
        test_soft_deadline_retains_final_and_native_errors_surface,
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
