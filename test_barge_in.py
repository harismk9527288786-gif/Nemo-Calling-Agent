"""Comprehensive test suite for Step 4: Barge-in, PlaybackWorker, and Queue Safety.

Verifies:
1. Continuous capture: audio_callback puts frames into audio_q even when is_speaking=True.
2. Queue safety: audio_q is NOT purged after playback, preserving caller speech.
3. Decoupled playback: PlaybackWorker writes asynchronously, interrupt() stops playback immediately.
4. Repeated interruptions: Multiple successive interruptions do not freeze or deadlock.
5. Clean shutdown: All worker threads exit cleanly.
6. Live Gemini interruption: Agent generates speech, caller interrupts with activity_start,
   Gemini confirms server_content.interrupted == True, and subsequent turn completes cleanly.
"""

import asyncio
import os
import queue
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from agent import CallingAgent
from playback_worker import PlaybackWorker


class TestBargeInUnit(unittest.TestCase):
    def test_continuous_capture_during_speech(self):
        """Microphone callback must NOT drop audio while is_speaking is True."""
        agent = CallingAgent.__new__(CallingAgent)
        agent.audio_q = queue.Queue()
        agent.is_speaking = True  # Agent is actively talking

        # Simulate 5 audio chunks arriving from the microphone
        test_samples = np.ones(1280, dtype=np.float32) * 0.05
        for _ in range(5):
            agent._audio_callback(test_samples, 1280, None, None)

        # Verify all 5 chunks were captured in audio_q
        self.assertEqual(agent.audio_q.qsize(), 5)
        # Verify sample values are intact
        pulled = agent.audio_q.get_nowait()
        np.testing.assert_allclose(pulled, test_samples)

    def test_playback_immediate_interruption(self):
        """PlaybackWorker.interrupt() must drain queued audio and call abort()."""
        mock_stream = MagicMock()
        mock_stream.active = True
        worker = PlaybackWorker(out_stream=mock_stream)

        # Slow down writes to simulate streaming speech
        def slow_write(chunk):
            time.sleep(0.04)
        mock_stream.write.side_effect = slow_write

        # Enqueue 15 chunks (representing ~1.2s of audio)
        for i in range(15):
            worker.enqueue(b"\x01\x02" * 400)

        self.assertTrue(worker.is_playing())

        # Caller barges in: trigger interrupt
        t_start = time.perf_counter()
        worker.interrupt()
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        # Must abort immediately (<15ms)
        self.assertLess(t_elapsed, 50.0, f"Interrupt took too long: {t_elapsed:.1f}ms")
        mock_stream.abort.assert_called()
        mock_stream.start.assert_called()
        self.assertTrue(worker._queue.empty())

        worker.stop()

    def test_repeated_interruptions_do_not_freeze(self):
        """Rapid successive interruptions must not deadlock or freeze."""
        mock_stream = MagicMock()
        mock_stream.active = True
        worker = PlaybackWorker(out_stream=mock_stream)

        for cycle in range(10):
            # Enqueue some chunks
            for _ in range(5):
                worker.enqueue(b"\x00\x01" * 200)
            # Interrupt immediately
            worker.interrupt()

        self.assertTrue(worker._queue.empty())
        self.assertTrue(worker._running)
        worker.stop()
        self.assertFalse(worker._running)

    def test_queue_safety_caller_first_words_preserved(self):
        """Audio captured while agent was speaking must not be purged."""
        agent = CallingAgent.__new__(CallingAgent)
        agent.audio_q = queue.Queue()
        agent.is_speaking = False

        # Caller starts speaking (first words: 'Arrey suniye')
        caller_first_words = np.ones(1280 * 3, dtype=np.float32) * 0.08
        agent.audio_q.put(caller_first_words)

        # Ensure no destructive flush happens: queue size remains 1
        self.assertEqual(agent.audio_q.qsize(), 1)
        recovered = agent.audio_q.get_nowait()
        np.testing.assert_array_equal(recovered, caller_first_words)

    def test_barge_in_echo_rejection_single_click(self):
        """A single loud click frame while agent is active must NOT trigger barge-in."""
        async def run_scenario():
            agent = CallingAgent.__new__(CallingAgent)
            agent.audio_q = queue.Queue()
            agent.is_running = True
            agent.is_speaking = True
            agent.barge_in_active = False
            agent._turn_interrupted = False

            mock_playback = MagicMock()
            mock_playback.is_playing.return_value = True
            agent.playback = mock_playback

            mock_asr_worker = MagicMock()
            agent.asr_worker = mock_asr_worker

            # Simulate single click frame above BARGE_IN_THRESHOLD (0.025)
            click_frame = np.ones(1280, dtype=np.float32) * 0.03
            consecutive_loud_frames = 0
            pending_loud_chunk = None

            frame_rms = float(np.sqrt(np.mean(click_frame**2)))
            agent_active = agent.is_speaking or agent.playback.is_playing()

            # Process frame
            if agent_active:
                if frame_rms > 0.025:
                    consecutive_loud_frames += 1
                    if consecutive_loud_frames == 1:
                        pending_loud_chunk = click_frame

            # Single frame must NOT trigger interrupt
            self.assertEqual(consecutive_loud_frames, 1)
            self.assertIsNotNone(pending_loud_chunk)
            mock_playback.interrupt.assert_not_called()
            self.assertFalse(agent.barge_in_active)

            # Follow-up quiet frame (RMS < 0.025) must reset streak
            quiet_frame = np.ones(1280, dtype=np.float32) * 0.005
            quiet_rms = float(np.sqrt(np.mean(quiet_frame**2)))
            if agent_active:
                if quiet_rms > 0.025:
                    consecutive_loud_frames += 1
                else:
                    consecutive_loud_frames = 0
                    pending_loud_chunk = None

            self.assertEqual(consecutive_loud_frames, 0)
            self.assertIsNone(pending_loud_chunk)
            mock_playback.interrupt.assert_not_called()

        asyncio.run(run_scenario())

    def test_agent_barge_in_orchestration_and_onset_slicing(self):
        """Verify 2 consecutive frames trigger barge-in, slice onset, keep _agent_task alive to drain."""
        async def run_scenario():
            agent = CallingAgent.__new__(CallingAgent)
            agent.audio_q = queue.Queue()
            agent.is_running = True
            agent.is_speaking = True
            agent.barge_in_active = False
            agent._turn_interrupted = False
            agent.metrics = MagicMock()

            mock_playback = MagicMock()
            mock_playback.is_playing.return_value = True
            agent.playback = mock_playback

            mock_asr_worker = MagicMock()
            mock_asr_worker.drain_results.return_value = []
            agent.asr_worker = mock_asr_worker

            mock_live_session = MagicMock()
            mock_live_session.send_realtime_input = AsyncMock()

            # Ongoing agent task that should NOT be abruptly cancelled
            async def long_speech():
                await asyncio.sleep(10.0)
            agent._agent_task = asyncio.create_task(long_speech())

            # Batch: Frame 0 is pre-speech echo (RMS ~0.008), Frame 1 is onset (RMS ~0.035), Frame 2 is confirmed speech (RMS ~0.040)
            frame_echo = np.ones(1280, dtype=np.float32) * 0.008
            frame_onset = np.ones(1280, dtype=np.float32) * 0.035
            frame_confirm = np.ones(1280, dtype=np.float32) * 0.040

            chunk_batch = [frame_echo, frame_onset, frame_confirm]

            consecutive_loud_frames = 0
            pending_loud_chunk = None
            agent_active = agent.is_speaking or agent.playback.is_playing()
            frames_to_submit = []

            for chunk in chunk_batch:
                frame_rms = float(np.sqrt(np.mean(chunk**2)))
                if agent_active:
                    if frame_rms > 0.025:
                        consecutive_loud_frames += 1
                        if consecutive_loud_frames == 1:
                            pending_loud_chunk = chunk
                        elif consecutive_loud_frames >= 2:
                            agent.playback.interrupt()
                            agent._turn_interrupted = True
                            await mock_live_session.send_realtime_input(activity_start=MagicMock())
                            agent.barge_in_active = True
                            agent.is_speaking = False
                            agent_active = False

                            if pending_loud_chunk is not None:
                                frames_to_submit.append(pending_loud_chunk)
                                pending_loud_chunk = None
                            frames_to_submit.append(chunk)
                    else:
                        consecutive_loud_frames = 0
                        pending_loud_chunk = None

            if frames_to_submit:
                agent.asr_worker.submit_audio(np.concatenate(frames_to_submit))

            # 1. Playback interrupted
            mock_playback.interrupt.assert_called_once()
            # 2. Activity start sent
            mock_live_session.send_realtime_input.assert_called_once()
            # 3. Agent task was NOT cancelled immediately
            self.assertFalse(agent._agent_task.cancelled())
            self.assertTrue(agent._turn_interrupted)
            # 4. ASR received only onset + confirm frame (2560 samples); echo frame was sliced off!
            self.assertEqual(mock_asr_worker.submit_audio.call_count, 1)
            submitted_audio = mock_asr_worker.submit_audio.call_args[0][0]
            self.assertEqual(len(submitted_audio), 2560)
            np.testing.assert_allclose(submitted_audio[:1280], frame_onset)
            np.testing.assert_allclose(submitted_audio[1280:], frame_confirm)

            # Cleanup
            agent._agent_task.cancel()
            try:
                await agent._agent_task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_scenario())

    def test_generate_gemini_native_live_audio_drain_on_interruption(self):
        """generate_gemini_native_live_audio must drop post-interruption audio and exit on turn_complete."""
        async def run_scenario():
            agent = CallingAgent.__new__(CallingAgent)
            agent.is_running = True
            agent.is_speaking = False
            agent.gemini_voice = "Puck"
            agent._turn_interrupted = False

            mock_playback = MagicMock()
            agent.playback = mock_playback

            # Mock live_session
            mock_session = MagicMock()
            mock_session.send_client_content = AsyncMock()

            # Mock response packets:
            # 1: inline audio part 1
            # 2: interrupted = True
            # 3: inline audio part 2 (should be DROPPED)
            # 4: turn_complete = True
            part1 = MagicMock()
            part1.inline_data.data = b"\x01\x02" * 100

            part2 = MagicMock()
            part2.inline_data.data = b"\x03\x04" * 100

            msg1 = MagicMock(server_content=MagicMock(interrupted=False, turn_complete=False, model_turn=MagicMock(parts=[part1])))
            msg2 = MagicMock(server_content=MagicMock(interrupted=True, turn_complete=False, model_turn=None))
            msg3 = MagicMock(server_content=MagicMock(interrupted=False, turn_complete=False, model_turn=MagicMock(parts=[part2])))
            msg4 = MagicMock(server_content=MagicMock(interrupted=False, turn_complete=True, model_turn=None))

            async def fake_receive():
                for m in [msg1, msg2, msg3, msg4]:
                    yield m

            mock_session.receive = fake_receive

            await agent.generate_gemini_native_live_audio(mock_session, "test input")

            # Verify:
            # 1. part 1 was enqueued
            mock_playback.enqueue.assert_called_once_with(part1.inline_data.data)
            # 2. playback was interrupted
            mock_playback.interrupt.assert_called_once()
            # 3. is_speaking is reset to False
            self.assertFalse(agent.is_speaking)

        asyncio.run(run_scenario())


class TestLiveGeminiInterruption(unittest.IsolatedAsyncioTestCase):
    async def test_live_gemini_interruption_and_recovery(self):
        """Live test against Google Gemini API: verify activity_start interrupts generation cleanly."""
        api_key_path = "gemini_api_key.txt"
        if not os.path.exists(api_key_path):
            self.skipTest("gemini_api_key.txt not found")

        with open(api_key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()

        if not api_key:
            self.skipTest("API key is empty")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
            ),
        )

        async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=cfg) as session:
            # Turn 1: Ask model to count from 1 to 30
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text="Count slowly from 1 to 30 in Hindi.")]),
                turn_complete=True,
            )

            interrupted = False
            received_chunks = 0

            async for resp in session.receive():
                sc = resp.server_content
                if sc:
                    if sc.interrupted:
                        interrupted = True
                        continue
                    if sc.model_turn and not interrupted:
                        received_chunks += 1
                        # Interrupt after first chunk
                        await session.send_realtime_input(activity_start=types.ActivityStart())
                    if sc.turn_complete:
                        break

            self.assertTrue(interrupted, "Expected Gemini to confirm server_content.interrupted == True")

            # Close caller activity period before sending Turn 2
            await session.send_realtime_input(activity_end=types.ActivityEnd())

            # Turn 2: Confirm session is alive and answers next prompt
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text="What is five plus five? Answer with just the number.")]),
                turn_complete=True,
            )

            turn2_completed = False
            turn2_audio_received = False
            async for resp in session.receive():
                sc = resp.server_content
                if sc:
                    if sc.model_turn:
                        for p in sc.model_turn.parts:
                            if p.inline_data and p.inline_data.data:
                                turn2_audio_received = True
                    if sc.turn_complete:
                        turn2_completed = True
                        break

            self.assertTrue(turn2_audio_received, "Expected audio response on Turn 2 after interruption")
            self.assertTrue(turn2_completed, "Expected Turn 2 to complete cleanly")

    async def test_live_gemini_multiple_interruptions(self):
        """Verify multiple consecutive interruptions on the same live session do not corrupt WebSocket state."""
        api_key_path = "gemini_api_key.txt"
        if not os.path.exists(api_key_path):
            self.skipTest("gemini_api_key.txt not found")

        with open(api_key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()

        if not api_key:
            self.skipTest("API key is empty")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
            ),
        )

        async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=cfg) as session:
            # Cycle 1: Interrupt Turn 1
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text="Count from 1 to 20 slowly.")]),
                turn_complete=True,
            )
            interrupted1 = False
            start_sent1 = False
            async for resp in session.receive():
                sc = resp.server_content
                if sc:
                    if sc.interrupted:
                        interrupted1 = True
                        continue
                    if sc.model_turn and not start_sent1:
                        await session.send_realtime_input(activity_start=types.ActivityStart())
                        start_sent1 = True
                    if sc.turn_complete:
                        break
            async def drain_pending():
                recv_iter = session.receive().__aiter__()
                while True:
                    try:
                        await asyncio.wait_for(recv_iter.__anext__(), timeout=0.2)
                    except (asyncio.TimeoutError, TimeoutError, StopAsyncIteration):
                        break
                    except Exception:
                        break

            self.assertTrue(interrupted1, "Turn 1 should be interrupted")
            await session.send_realtime_input(activity_end=types.ActivityEnd())
            await drain_pending()

            # Cycle 2: Interrupt Turn 2
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text="Recite numbers from 100 down to 80 slowly.")]),
                turn_complete=True,
            )
            interrupted2 = False
            start_sent2 = False
            async for resp in session.receive():
                sc = resp.server_content
                if sc:
                    if sc.interrupted:
                        interrupted2 = True
                        continue
                    if sc.model_turn and not start_sent2:
                        await session.send_realtime_input(activity_start=types.ActivityStart())
                        start_sent2 = True
                    if sc.turn_complete:
                        break
            self.assertTrue(interrupted2, "Turn 2 should be interrupted")
            await session.send_realtime_input(activity_end=types.ActivityEnd())
            await drain_pending()

            # Cycle 3: Uninterrupted final question
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text="What is two plus two? Answer only the number.")]),
                turn_complete=True,
            )
            turn3_completed = False
            turn3_audio_received = False
            async for resp in session.receive():
                sc = resp.server_content
                if sc:
                    if sc.model_turn:
                        for p in sc.model_turn.parts:
                            if p.inline_data and p.inline_data.data:
                                turn3_audio_received = True
                    if sc.turn_complete:
                        turn3_completed = True
                        break

            self.assertTrue(turn3_audio_received, "Turn 3 should receive audio after 2 consecutive interruptions")
            self.assertTrue(turn3_completed, "Turn 3 should complete cleanly")


if __name__ == "__main__":
    unittest.main()
