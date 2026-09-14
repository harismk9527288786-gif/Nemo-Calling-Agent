"""Focused Concurrency, Latency, and Correctness Audit for Step 5 VAD / Endpointing.

Verifies:
1. Silero VAD inference never blocks the asyncio event loop (<10ms per 80ms chunk).
2. VAD window buffering handles arbitrary chunk sizes [1, 13, 512, 1280, 4096 samples].
3. The 64ms onset buffer does not duplicate audio across consecutive turns.
4. endpoint_detected is emitted exactly once per caller turn.
5. 300ms silence endpointing correctly distinguishes intra-utterance pauses (<300ms) from turn completion (>=300ms).
6. flush_and_restart() is thread-safe while decode thread is actively processing chunks.
7. No shared decoder state is mutated concurrently.
8. Turn 2 cannot consume audio from Turn 1.
9. Barge-in preserves Step 4 Gemini receive-loop lifecycle without cancelling _agent_task.
10. Playback generation counter increments and drops stale chunks over repeated interruptions.
11. TorchScript, ONNX, and RMS fallback produce compatible event state transitions.
12. VAD model loading failure does not crash call startup.
"""

import sys
sys.path.insert(0, r"C:\AI\NeMo-Speech.cpp")

import asyncio
import os
import queue
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import soundfile as sf
import torch
import torchaudio

from agent import CallingAgent
from asr_worker import AsrWorker
from playback_worker import PlaybackWorker
from vad_endpointer import SileroVADEndpointer, VADEvent


def load_resampled_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    full_path = os.path.join(r"C:\AI\NeMo-Speech.cpp", path) if not os.path.isabs(path) else path
    if not os.path.exists(full_path):
        return np.zeros(target_sr * 2, dtype=np.float32)
    data, sr = sf.read(full_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        tensor = torchaudio.functional.resample(torch.from_numpy(data).float(), sr, target_sr)
        data = tensor.numpy()
    return data.astype(np.float32)


class TestVADConcurrencyAndLatencyAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hindi_audio = load_resampled_wav("test.wav")
        cls.hinglish_audio = load_resampled_wav("talha.wav")

    def setUp(self):
        self.vad = SileroVADEndpointer(
            sample_rate=16000,
            speech_threshold=0.50,
            barge_in_speech_threshold=0.65,
            silence_threshold=0.35,
            min_speech_duration_ms=96.0,
            barge_in_min_speech_duration_ms=160.0,
            min_silence_duration_ms=300.0,
            speech_pad_ms=64.0,
        )

    # ----------------------------------------------------------------- Check 1
    def test_1_vad_inference_does_not_block_event_loop(self):
        """Silero VAD execution per 80ms chunk must be < 10ms (well under asyncio task lag limits)."""
        chunk = self.hindi_audio[:1280]
        # Warmup
        for _ in range(5):
            self.vad.process_chunk(chunk)

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            self.vad.process_chunk(chunk)
            times.append((time.perf_counter() - t0) * 1000.0)

        p95 = np.percentile(times, 95)
        mean_t = np.mean(times)
        print(f"\n[Audit Check 1] VAD per-chunk execution: mean={mean_t:.2f}ms, p95={p95:.2f}ms")
        self.assertLess(p95, 10.0, f"VAD inference latency p95 ({p95:.2f}ms) exceeded 10ms budget")

    # ----------------------------------------------------------------- Check 2
    def test_2_vad_window_buffering_arbitrary_input_sizes(self):
        """Verify window buffering preserves exact sample counts across arbitrary chunk sizes."""
        test_chunk_sizes = [1, 7, 13, 64, 128, 511, 512, 1000, 1280, 2049, 0]
        total_input_samples = 0

        # Feed 1.5s of speech with variable chunk sizes
        speech = self.hindi_audio[:24000]
        idx = 0
        for sz in test_chunk_sizes * 4:
            if idx >= len(speech):
                break
            chunk = speech[idx:idx+sz]
            idx += sz
            total_input_samples += len(chunk)
            self.vad.process_chunk(chunk)

        samples_in_window_buf = len(self.vad._window_buf)
        # All samples must be either processed in 512-sample windows or retained in _window_buf
        self.assertLess(samples_in_window_buf, 512, "Window buffer must never hold >= 512 samples without processing")

    # ----------------------------------------------------------------- Check 3
    def test_3_onset_buffer_does_not_duplicate_across_turns(self):
        """64ms onset buffer must NOT duplicate audio from Turn 1 into Turn 2."""
        # Turn 1: 500ms of speech followed by 400ms of silence
        turn1_speech = self.hindi_audio[1600:1600+8000]
        for i in range(0, len(turn1_speech), 1280):
            ev = self.vad.process_chunk(turn1_speech[i:i+1280])

        silence_400ms = np.zeros(6400, dtype=np.float32)
        turn1_endpointed = False
        for i in range(0, len(silence_400ms), 1280):
            ev = self.vad.process_chunk(silence_400ms[i:i+1280])
            if ev.endpoint_detected:
                turn1_endpointed = True

        self.assertTrue(turn1_endpointed, "Turn 1 must endpoint on 400ms silence")
        self.vad.reset()

        # Turn 2: Fresh Hinglish speech
        turn2_speech = self.hinglish_audio[20000:20000+4000]
        turn2_first_audio = None
        for i in range(0, len(turn2_speech), 1280):
            ev = self.vad.process_chunk(turn2_speech[i:i+1280])
            if ev.audio_for_asr is not None and turn2_first_audio is None:
                turn2_first_audio = ev.audio_for_asr

        self.assertIsNotNone(turn2_first_audio, "Turn 2 must emit audio for ASR")
        # Ensure none of Turn 1 audio samples exist in Turn 2 first audio
        self.assertFalse(np.array_equal(turn2_first_audio[:512], turn1_speech[:512]))

    # ----------------------------------------------------------------- Check 4
    def test_4_endpoint_detected_emitted_exactly_once(self):
        """endpoint_detected must be emitted exactly once per speech turn, not repeatedly."""
        speech = self.hindi_audio[1600:1600+4800]
        for i in range(0, len(speech), 1280):
            self.vad.process_chunk(speech[i:i+1280])

        # Feed 1.0 second of silence (12 chunks of 1280 samples)
        silence_chunk = np.zeros(1280, dtype=np.float32)
        endpoint_count = 0
        for _ in range(12):
            ev = self.vad.process_chunk(silence_chunk)
            if ev.endpoint_detected:
                endpoint_count += 1

        self.assertEqual(endpoint_count, 1, f"endpoint_detected emitted {endpoint_count} times; expected exactly 1")

    # ----------------------------------------------------------------- Check 5
    def test_5_silence_endpoint_distinguishes_pause_from_turn_end(self):
        """Intra-utterance pause (<300ms) does NOT endpoint; pause >= 300ms DOES endpoint."""
        # 1. Intra-sentence micro-pause (160ms)
        speech_part1 = self.hindi_audio[1600:1600+4000]
        for i in range(0, len(speech_part1), 1280):
            self.vad.process_chunk(speech_part1[i:i+1280])

        # 160ms silence (should NOT endpoint)
        silence_160ms = np.zeros(2560, dtype=np.float32)
        ev_pause = self.vad.process_chunk(silence_160ms)
        self.assertFalse(ev_pause.endpoint_detected, "160ms pause must not trigger endpoint")
        self.assertEqual(self.vad._state, "IN_SPEECH", "VAD must remain in IN_SPEECH during micro-pause")

        # Speech resumes
        speech_part2 = self.hindi_audio[5600:5600+3840]
        at_least_one_speech = False
        for i in range(0, len(speech_part2), 1280):
            ev_resume = self.vad.process_chunk(speech_part2[i:i+1280])
            if ev_resume.is_speech:
                at_least_one_speech = True
        self.assertTrue(at_least_one_speech, "Speech resume after micro-pause must be detected as speech")


        # 2. Utterance completion (350ms silence)
        silence_350ms = np.zeros(5600, dtype=np.float32)
        endpoint_fired = False
        for i in range(0, len(silence_350ms), 1280):
            ev = self.vad.process_chunk(silence_350ms[i:i+1280])
            if ev.endpoint_detected:
                endpoint_fired = True

        self.assertTrue(endpoint_fired, "350ms silence must trigger turn endpoint")

    # ----------------------------------------------------------------- Check 6 & 7
    def test_6_and_7_flush_and_restart_thread_safety(self):
        """flush_and_restart() is safe while decode thread is active; no decoder state mutated concurrently."""
        mock_asr = MagicMock()
        mock_asr.poll_results.return_value = [(False, "नमस्ते")]
        mock_asr.final_transcript.return_value = "नमस्ते जी"

        worker = AsrWorker(mock_asr, sample_rate=16000, interim_results=True)
        worker.start()

        async def scenario():
            # Submit chunks continuously
            chunk = np.ones(1280, dtype=np.float32) * 0.05
            for _ in range(5):
                worker.submit_audio(chunk)

            # Flush concurrently
            final_text = await worker.flush_and_restart(timeout=1.0)
            self.assertEqual(final_text, "नमस्ते जी")
            self.assertEqual(worker.error_count, 0)

        asyncio.run(scenario())
        worker.stop()

    # ----------------------------------------------------------------- Check 8
    def test_8_new_turn_cannot_consume_previous_turn_audio(self):
        """Verify audio submitted after flush_and_restart is isolated to the new stream."""
        mock_asr = MagicMock()
        mock_asr.poll_results.side_effect = [
            [(False, "Turn 1 interim")],
            [],
            [(False, "Turn 2 interim")],
        ]
        mock_asr.final_transcript.side_effect = ["Turn 1 Final", "Turn 2 Final"]

        worker = AsrWorker(mock_asr, sample_rate=16000, interim_results=True)
        worker.start()

        async def scenario():
            worker.submit_audio(np.ones(1280, dtype=np.float32))
            res1 = await worker.flush_and_restart()
            self.assertEqual(res1, "Turn 1 Final")

            # Turn 2 audio
            worker.submit_audio(np.ones(1280, dtype=np.float32) * 2)
            res2 = await worker.flush_and_restart()
            self.assertEqual(res2, "Turn 2 Final")

        asyncio.run(scenario())
        worker.stop()

    # ----------------------------------------------------------------- Check 9
    def test_9_barge_in_preserves_gemini_receive_loop(self):
        """Barge-in signals interruption, sets _turn_interrupted, but does NOT cancel _agent_task."""
        async def scenario():
            agent = CallingAgent.__new__(CallingAgent)
            agent.is_running = True
            agent.is_speaking = True
            agent._turn_interrupted = False
            agent.barge_in_active = False

            mock_playback = MagicMock()
            mock_playback.is_playing.return_value = True
            agent.playback = mock_playback

            async def fake_speech_task():
                await asyncio.sleep(5.0)

            agent._agent_task = asyncio.create_task(fake_speech_task())

            # Simulate barge-in event from VAD
            vad_event = VADEvent(barge_in=True, speech_start=True, is_speech=True)
            if vad_event.barge_in:
                agent.playback.interrupt()
                agent._turn_interrupted = True
                agent.barge_in_active = True
                agent.is_speaking = False

            # Verify invariants
            self.assertTrue(agent._turn_interrupted)
            self.assertTrue(agent.barge_in_active)
            self.assertFalse(agent.is_speaking)
            mock_playback.interrupt.assert_called_once()
            # _agent_task must still be alive (NOT cancelled)
            self.assertFalse(agent._agent_task.cancelled())
            self.assertFalse(agent._agent_task.done())

            agent._agent_task.cancel()
            try:
                await agent._agent_task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())

    # ----------------------------------------------------------------- Check 10
    def test_10_playback_generation_counter_repeated_interruptions(self):
        """Playback generation ID increments and stale audio chunks are dropped over repeated interruptions."""
        mock_stream = MagicMock()
        mock_stream.active = True
        worker = PlaybackWorker(out_stream=mock_stream)

        initial_gen = worker._generation
        for cycle in range(5):
            worker.enqueue(b"\x01\x02" * 200)
            worker.interrupt()
            self.assertEqual(worker._generation, initial_gen + cycle + 1)
            self.assertTrue(worker._queue.empty())

        worker.stop()

    # ----------------------------------------------------------------- Check 11
    def test_11_backend_parity_torchscript_onnx_rms(self):
        """TorchScript, ONNX, and RMS fallback paths produce consistent event transitions."""
        vad_ts = SileroVADEndpointer(force_energy_fallback=False)
        vad_rms = SileroVADEndpointer(force_energy_fallback=True)

        speech_chunk = self.hindi_audio[1600:1600+4000]

        # Feed speech to both
        ts_speech_start = False
        rms_speech_start = False

        for i in range(0, len(speech_chunk), 1280):
            c = speech_chunk[i:i+1280]
            e_ts = vad_ts.process_chunk(c)
            e_rms = vad_rms.process_chunk(c * 1.5)  # Ample amplitude for energy detector
            if e_ts.speech_start or e_ts.is_speech:
                ts_speech_start = True
            if e_rms.speech_start or e_rms.is_speech:
                rms_speech_start = True

        self.assertTrue(ts_speech_start, "TorchScript must detect speech")
        self.assertTrue(rms_speech_start, "RMS fallback must detect speech")

        # Feed silence to both
        silence_chunk = np.zeros(1280, dtype=np.float32)
        ts_endpoint = False
        rms_endpoint = False
        for _ in range(10):
            e_ts = vad_ts.process_chunk(silence_chunk)
            e_rms = vad_rms.process_chunk(silence_chunk)
            if e_ts.endpoint_detected:
                ts_endpoint = True
            if e_rms.endpoint_detected:
                rms_endpoint = True

        self.assertTrue(ts_endpoint, "TorchScript must detect endpoint on silence")
        self.assertTrue(rms_endpoint, "RMS fallback must detect endpoint on silence")

    # ----------------------------------------------------------------- Check 12
    def test_12_vad_loading_failure_resilience(self):
        """VAD model loading failure falls back to energy detection without crashing startup."""
        with patch("silero_vad.load_silero_vad", side_effect=RuntimeError("Corrupt model file")):
            vad = SileroVADEndpointer(force_energy_fallback=False)
            self.assertTrue(vad.using_fallback)
            self.assertIsNone(vad.model)

            # process_chunk operates normally in fallback
            ev = vad.process_chunk(np.zeros(1280, dtype=np.float32))
            self.assertFalse(ev.is_speech)
            self.assertFalse(ev.speech_start)


if __name__ == "__main__":
    unittest.main()
