"""Comprehensive test suite for Step 5: Silero VAD and Turn Endpointing.

Verifies:
1. Short noise bursts: Brief noise (<96ms) is rejected; does not trigger speech_start.
2. Keyboard clicks: Transient impulse clicks do not trigger speech_start.
3. Speaker echo: Echo frames while agent is active do not trigger barge-in.
4. Hindi speech: Voice frames from Hindi speech trigger speech_start and stream audio to ASR.
5. Hinglish speech: Voice frames from Hinglish speech trigger speech_start and maintain speech state.
6. Short caller answers: Short response (e.g., 'हाँ', 'yes' ~250ms) followed by silence endpoints cleanly.
7. Long caller answers: Multi-second speech with short inter-word pauses does not prematurely endpoint.
8. Interruption during agent playback: Speech while agent_active=True triggers barge-in and slices onset.
9. Silence-based turn completion: Trailing silence >= min_silence_duration_ms triggers endpoint_detected.
10. Clean shutdown & Fallback: reset() clears all internal buffers; energy fallback works when Silero is disabled.
"""

import os
import unittest
import numpy as np
import soundfile as sf
import torch
import torchaudio

from vad_endpointer import SileroVADEndpointer, VADEvent


def load_audio_slice(file_path: str, start_sec: float, duration_sec: float, target_sr: int = 16000) -> np.ndarray:
    """Loads a slice from a wav file and resamples to target_sr."""
    if not os.path.exists(file_path):
        # Fallback to noise if file not found
        return np.zeros(int(target_sr * duration_sec), dtype=np.float32)
    data, sr = sf.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        tensor_data = torchaudio.functional.resample(torch.from_numpy(data).float(), sr, target_sr)
        data = tensor_data.numpy()
    start_idx = int(start_sec * target_sr)
    end_idx = start_idx + int(duration_sec * target_sr)
    sliced = data[start_idx:end_idx]
    if len(sliced) < int(duration_sec * target_sr):
        pad_len = int(duration_sec * target_sr) - len(sliced)
        sliced = np.pad(sliced, (0, pad_len))
    return sliced.astype(np.float32)


def generate_click(duration_samples: int = 1280, amplitude: float = 0.3) -> np.ndarray:
    """Generates a transient click / impulse (e.g. mouse/keyboard click)."""
    click = np.zeros(duration_samples, dtype=np.float32)
    spike_len = min(duration_samples, 64)
    click[:spike_len] = (np.random.rand(spike_len) * 2 - 1) * amplitude
    return click


def generate_noise(duration_seconds: float, sample_rate: int = 16000, amplitude: float = 0.005) -> np.ndarray:
    """Generates ambient room noise."""
    samples = int(sample_rate * duration_seconds)
    return ((np.random.rand(samples) * 2 - 1) * amplitude).astype(np.float32)


class TestSileroVADEndpointer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Pre-load Hindi speech from test.wav (active speech between 0.10s and 1.50s)
        cls.hindi_speech = load_audio_slice("test.wav", start_sec=0.10, duration_sec=1.20)
        # Pre-load Hinglish speech from talha.wav (active speech between 1.20s and 2.50s)
        cls.hinglish_speech = load_audio_slice("talha.wav", start_sec=1.20, duration_sec=1.20)

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

    def test_1_short_noise_burst_rejected(self):
        """A short 40ms noise burst must NOT trigger speech_start."""
        noise_burst = generate_noise(duration_seconds=0.04, amplitude=0.08)
        event = self.vad.process_chunk(noise_burst, agent_active=False)
        self.assertFalse(event.speech_start)
        self.assertFalse(event.is_speech)
        self.assertIsNone(event.audio_for_asr)

    def test_2_keyboard_clicks_rejected(self):
        """Impulsive keyboard clicks must NOT trigger speech_start."""
        click = generate_click(duration_samples=1280, amplitude=0.4)
        event = self.vad.process_chunk(click, agent_active=False)
        self.assertFalse(event.speech_start)
        self.assertIsNone(event.audio_for_asr)

    def test_3_speaker_echo_rejected_during_agent_active(self):
        """Low-to-moderate speaker echo (low volume or short duration) during agent playback does not trigger barge-in."""
        # Low volume speech slice (RMS ~0.01)
        low_echo = self.hindi_speech[:1280] * 0.04
        event = self.vad.process_chunk(low_echo, agent_active=True)
        self.assertFalse(event.barge_in)
        self.assertFalse(event.speech_start)

    def test_4_hindi_speech_detection(self):
        """Hindi speech harmonic input triggers speech_start and streams audio to ASR."""
        # Feed 300ms of Hindi speech in 80ms chunks
        chunk_len = 1280
        speech_slice = self.hindi_speech[:int(16000 * 0.35)]
        speech_started = False
        audio_submitted = False

        for i in range(0, len(speech_slice), chunk_len):
            chunk = speech_slice[i:i+chunk_len]
            event = self.vad.process_chunk(chunk, agent_active=False)
            if event.speech_start or event.is_speech:
                speech_started = True
            if event.audio_for_asr is not None:
                audio_submitted = True

        self.assertTrue(speech_started, "Expected Hindi speech to be detected")
        self.assertTrue(audio_submitted, "Expected audio with onset padding to be produced for ASR")

    def test_5_hinglish_speech_detection(self):
        """Hinglish speech with natural bilingual inflection triggers speech_start."""
        chunk_len = 1280
        speech_slice = self.hinglish_speech[:int(16000 * 0.35)]
        speech_started = False

        for i in range(0, len(speech_slice), chunk_len):
            chunk = speech_slice[i:i+chunk_len]
            event = self.vad.process_chunk(chunk, agent_active=False)
            if event.speech_start or event.is_speech:
                speech_started = True

        self.assertTrue(speech_started, "Expected Hinglish speech to be detected")

    def test_6_short_caller_answer_endpoints_cleanly(self):
        """Short caller answer (e.g. 'हाँ', 'yes' ~250ms) followed by silence triggers endpoint_detected."""
        chunk_len = 1280
        short_speech = self.hindi_speech[:int(16000 * 0.25)]
        silence = np.zeros(int(16000 * 0.40), dtype=np.float32)

        # Feed speech
        for i in range(0, len(short_speech), chunk_len):
            self.vad.process_chunk(short_speech[i:i+chunk_len], agent_active=False)

        # Feed silence
        endpoint_fired = False
        for i in range(0, len(silence), chunk_len):
            event = self.vad.process_chunk(silence[i:i+chunk_len], agent_active=False)
            if event.endpoint_detected:
                endpoint_fired = True
                break

        self.assertTrue(endpoint_fired, "Expected endpoint_detected to fire after 400ms of silence")

    def test_7_long_caller_answer_with_brief_pauses_maintains_speech(self):
        """Multi-second speech with brief (100ms) inter-word pauses does NOT prematurely endpoint."""
        chunk_len = 1280
        word1 = self.hindi_speech[:int(16000 * 0.40)]
        brief_pause = np.zeros(int(16000 * 0.10), dtype=np.float32)  # 100ms pause < 300ms min silence
        word2 = self.hindi_speech[int(16000 * 0.40):int(16000 * 0.80)]

        # Feed Word 1
        for i in range(0, len(word1), chunk_len):
            self.vad.process_chunk(word1[i:i+chunk_len], agent_active=False)

        # Feed 100ms pause
        e_pause = self.vad.process_chunk(brief_pause, agent_active=False)
        self.assertFalse(e_pause.endpoint_detected, "100ms pause should NOT trigger endpoint")

        # Feed Word 2
        e2_active = False
        for i in range(0, len(word2), chunk_len):
            e = self.vad.process_chunk(word2[i:i+chunk_len], agent_active=False)
            if e.is_speech:
                e2_active = True
            self.assertFalse(e.endpoint_detected, "Speech should not trigger endpoint")

        self.assertTrue(e2_active)

    def test_8_interruption_during_agent_playback(self):
        """Loud caller speech while agent_active=True triggers barge_in=True and slices onset."""
        chunk_len = 1280
        loud_speech = self.hinglish_speech[:int(16000 * 0.35)] * 1.2
        barge_in_fired = False
        audio_produced = False

        for i in range(0, len(loud_speech), chunk_len):
            event = self.vad.process_chunk(loud_speech[i:i+chunk_len], agent_active=True)
            if event.barge_in:
                barge_in_fired = True
            if event.audio_for_asr is not None:
                audio_produced = True

        self.assertTrue(barge_in_fired, "Expected barge_in to fire during agent playback")
        self.assertTrue(audio_produced, "Expected onset sliced audio to be emitted on barge-in")

    def test_9_silence_based_turn_completion_exact_timing(self):
        """Verify endpoint_detected fires once silence reaches min_silence_duration_ms."""
        chunk_len = 1280
        speech = self.hindi_speech[:int(16000 * 0.30)]
        for i in range(0, len(speech), chunk_len):
            self.vad.process_chunk(speech[i:i+chunk_len], agent_active=False)

        # Exactly 200ms silence (should NOT endpoint yet)
        silence_200ms = np.zeros(int(16000 * 0.20), dtype=np.float32)
        e_200 = self.vad.process_chunk(silence_200ms, agent_active=False)
        self.assertFalse(e_200.endpoint_detected)

        # Additional 250ms silence (total 450ms > 300ms, triggers endpoint)
        silence_250ms = np.zeros(int(16000 * 0.25), dtype=np.float32)
        e_450 = self.vad.process_chunk(silence_250ms, agent_active=False)
        self.assertTrue(e_450.endpoint_detected)

    def test_10_clean_shutdown_and_energy_fallback(self):
        """Verify reset() clears internal state and energy fallback operates seamlessly."""
        self.vad.process_chunk(self.hindi_speech[:int(16000 * 0.3)], agent_active=False)
        self.vad.reset()
        self.assertEqual(self.vad._state, "LISTENING")
        self.assertEqual(self.vad._speech_samples_accum, 0)
        self.assertEqual(len(self.vad._window_buf), 0)

        # Test energy fallback mode
        fallback_vad = SileroVADEndpointer(force_energy_fallback=True)
        self.assertTrue(fallback_vad.using_fallback)
        self.assertIsNone(fallback_vad.model)

        # Feed speech to fallback (must exceed min_speech_duration_ms = 96ms)
        chunk1 = self.hindi_speech[:1280] * 1.5
        chunk2 = self.hindi_speech[1280:2560] * 1.5
        fb_event1 = fallback_vad.process_chunk(chunk1, agent_active=False)
        fb_event2 = fallback_vad.process_chunk(chunk2, agent_active=False)
        self.assertTrue(fb_event2.speech_start or fb_event2.is_speech)
        self.assertIsNotNone(fb_event2.audio_for_asr)



if __name__ == "__main__":
    unittest.main()
