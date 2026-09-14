"""Silero VAD and Turn Endpointing module for real-time speech streaming.

Step 5 of project_docs/PHASE0_AUDIT_AND_PLAN_24_08_2026.md.

Replaces naive energy thresholds with neural Silero VAD (v6) running on 512-sample (32ms)
windows. Decouples turn completion from ASR partial latency. Provides onset padding,
hysteresis, echo protection, and graceful fallback to energy detection.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

logger = logging.getLogger("vad_endpointer")


@dataclass
class VADEvent:
    """Represents the output of processing an audio chunk through VAD and endpointing."""
    is_speech: bool = False
    speech_probability: float = 0.0
    speech_start: bool = False
    barge_in: bool = False
    endpoint_detected: bool = False
    audio_for_asr: Optional[np.ndarray] = None


class SileroVADEndpointer:
    """Manages streaming VAD and turn-taking endpoint detection with Silero VAD and energy fallback."""

    WINDOW_SIZE = 512  # Fixed 512-sample (32ms @ 16kHz) window required by Silero VAD v6

    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.50,
        barge_in_speech_threshold: float = 0.65,
        silence_threshold: float = 0.35,
        min_speech_duration_ms: float = 96.0,
        barge_in_min_speech_duration_ms: float = 160.0,
        min_silence_duration_ms: float = 300.0,
        speech_pad_ms: float = 64.0,
        energy_fallback_threshold: float = 0.015,
        energy_fallback_barge_in_threshold: float = 0.025,
        force_energy_fallback: bool = False,
    ):
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.barge_in_speech_threshold = barge_in_speech_threshold
        self.silence_threshold = silence_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.barge_in_min_speech_duration_ms = barge_in_min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.energy_fallback_threshold = energy_fallback_threshold
        self.energy_fallback_barge_in_threshold = energy_fallback_barge_in_threshold

        # Samples calculations
        self.min_speech_samples = int(self.sample_rate * (self.min_speech_duration_ms / 1000.0))
        self.barge_in_min_speech_samples = int(self.sample_rate * (self.barge_in_min_speech_duration_ms / 1000.0))
        self.min_silence_samples = int(self.sample_rate * (self.min_silence_duration_ms / 1000.0))
        self.speech_pad_samples = int(self.sample_rate * (self.speech_pad_ms / 1000.0))

        # Model initialization
        self.using_fallback = force_energy_fallback
        self.model = None
        self._torch = None

        if not force_energy_fallback:
            self._init_model()

        # Streaming audio window buffer
        self._window_buf = np.empty(0, dtype=np.float32)

        # Circular buffer for pre-speech onset padding
        self._pre_speech_pad: Deque[np.ndarray] = deque()
        self._pre_speech_samples = 0

        # State machine: "LISTENING", "IN_SPEECH"
        self._state = "LISTENING"
        self._speech_samples_accum = 0
        self._silence_samples_accum = 0
        self._speech_chunks_in_flight: List[np.ndarray] = []

    def _init_model(self):
        """Initializes Silero VAD, preferring TorchScript with ONNX fallback."""
        try:
            import silero_vad
            import torch
            self._torch = torch

            # 1. Try TorchScript (fastest: ~1.59ms)
            try:
                self.model = silero_vad.load_silero_vad(onnx=False)
                self.using_fallback = False
                print("[VAD] Silero VAD (TorchScript) loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"TorchScript Silero VAD load failed: {e}. Trying ONNX...")

            # 2. Try ONNX wrapper
            try:
                self.model = silero_vad.load_silero_vad(onnx=True)
                self.using_fallback = False
                print("[VAD] Silero VAD (ONNX) loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"ONNX Silero VAD load failed: {e}.")

        except Exception as exc:
            logger.warning(f"Could not initialize Silero VAD: {exc}. Falling back to energy detection.")

        # 3. Fallback to calibrated energy VAD
        self.using_fallback = True
        self.model = None
        print("[VAD] Using energy-based VAD fallback.")

    def reset_states(self):
        """Resets the recurrent neural network states and internal counters."""
        if self.model is not None and hasattr(self.model, "reset_states"):
            try:
                self.model.reset_states()
            except Exception:
                pass

    def reset(self):
        """Resets the endpointer state machine for a new turn."""
        self.reset_states()
        self._state = "LISTENING"
        self._speech_samples_accum = 0
        self._silence_samples_accum = 0
        self._speech_chunks_in_flight = []
        self._pre_speech_pad.clear()
        self._pre_speech_samples = 0
        self._window_buf = np.empty(0, dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray, agent_active: bool = False) -> VADEvent:
        """Processes an audio chunk (arbitrary sample count) and returns a VADEvent.

        Args:
            chunk: 1D float32 numpy array of PCM audio at self.sample_rate.
            agent_active: True if the agent is actively speaking or playing audio.
        """
        if chunk.ndim > 1:
            chunk = chunk.flatten()
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        # Append to window buffer
        if len(self._window_buf) == 0:
            self._window_buf = chunk.copy()
        else:
            self._window_buf = np.concatenate([self._window_buf, chunk])

        event = VADEvent()
        output_audio_pieces: List[np.ndarray] = []
        latest_prob = 0.0

        # Process all available 512-sample (32ms) windows
        while len(self._window_buf) >= self.WINDOW_SIZE:
            window = self._window_buf[:self.WINDOW_SIZE]
            self._window_buf = self._window_buf[self.WINDOW_SIZE:]

            # Compute speech probability
            if not self.using_fallback and self.model is not None and self._torch is not None:
                tensor_w = self._torch.from_numpy(window)
                prob = float(self.model(tensor_w, self.sample_rate).item())
            else:
                # Energy fallback: compute RMS
                rms = float(np.sqrt(np.mean(window**2)))
                thresh = self.energy_fallback_barge_in_threshold if agent_active else self.energy_fallback_threshold
                prob = 1.0 if rms > thresh else 0.0

            latest_prob = prob

            # Threshold selection
            active_speech_thresh = self.barge_in_speech_threshold if agent_active else self.speech_threshold
            required_speech_samples = self.barge_in_min_speech_samples if agent_active else self.min_speech_samples

            if self._state == "LISTENING":
                if prob >= active_speech_thresh:
                    self._speech_samples_accum += self.WINDOW_SIZE
                    self._speech_chunks_in_flight.append(window)

                    # Check if speech duration meets minimum required threshold
                    if self._speech_samples_accum >= required_speech_samples:
                        # Confirmed speech start!
                        self._state = "IN_SPEECH"
                        self._silence_samples_accum = 0
                        event.is_speech = True
                        event.speech_start = True
                        if agent_active:
                            event.barge_in = True

                        # Sliced onset: pre-speech pad (leading consonants) + in-flight speech windows
                        onset_pad = list(self._pre_speech_pad)
                        self._pre_speech_pad.clear()
                        self._pre_speech_samples = 0

                        output_audio_pieces.extend(onset_pad)
                        output_audio_pieces.extend(self._speech_chunks_in_flight)
                        self._speech_chunks_in_flight = []
                else:
                    # Non-speech window: transient noise, click, or room silence
                    self._speech_samples_accum = 0
                    self._speech_chunks_in_flight = []

                    # Maintain circular buffer of pre-speech padding
                    self._pre_speech_pad.append(window)
                    self._pre_speech_samples += self.WINDOW_SIZE
                    while self._pre_speech_samples > self.speech_pad_samples:
                        popped = self._pre_speech_pad.popleft()
                        self._pre_speech_samples -= len(popped)

            elif self._state == "IN_SPEECH":
                event.is_speech = True
                output_audio_pieces.append(window)

                if prob >= self.silence_threshold:
                    # Speech continues: reset trailing silence counter
                    self._silence_samples_accum = 0
                else:
                    # Silence detected during utterance
                    self._silence_samples_accum += self.WINDOW_SIZE
                    if self._silence_samples_accum >= self.min_silence_samples:
                        # Turn complete! Endpoint detected!
                        event.endpoint_detected = True
                        self._state = "LISTENING"
                        self._speech_samples_accum = 0
                        self._silence_samples_accum = 0
                        self._speech_chunks_in_flight = []
                        self._pre_speech_pad.clear()
                        self._pre_speech_samples = 0
                        self.reset_states()

        event.speech_probability = latest_prob
        if output_audio_pieces:
            event.audio_for_asr = np.concatenate(output_audio_pieces)

        return event
