from __future__ import annotations

# Rebuilt simple call-loop implementation.

import asyncio
import base64
import json
import os
import queue
import re
import sys
import time
import uuid
from typing import Any, Dict, Optional

import httpx
import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types

from asr_worker import AsrWorker
from nemo_asr import NeMoStreamingASR
from pipeline_metrics import MetricsLogger
from playback_worker import PlaybackWorker
from vad_endpointer import SileroVADEndpointer

pygame = None


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Keep the project's established audio and model settings.
MODEL_PATH = r"models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
KB_PATH = "knowledge_base.json"

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.08
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_TIMEOUT = 0.80
ENERGY_THRESHOLD = 0.015
BARGE_IN_THRESHOLD = 0.06
BARGE_IN_CONSECUTIVE_FRAMES = 2

# A slow barrier is diagnostic, not permission to send an interim to Gemini.
ASR_SLOW_SECONDS = 1.50
ASR_FLUSH_TIMEOUT = 8.0
ECHO_GUARD_SECONDS = 0.25
GEMINI_SEND_TIMEOUT = 10.0
INTERRUPTION_BOUNDARY_TIMEOUT = 0.75

GEMINI_RECONNECT_LIMIT = 3
GEMINI_RECONNECT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
GEMINI_MODEL = "gemini-2.5-flash-native-audio-latest"
GEMINI_OUTPUT_SAMPLE_RATE = 24000

DEFAULT_MODE = "gemini-live"
DEFAULT_GEMINI_VOICE = "Puck"
DEFAULT_SARVAM_SPEAKER = "rahul"


def _compact_text(value: Any, limit: int = 400) -> str:
    text = " ".join(str(value).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _safe_error_text(exc: BaseException, secrets: tuple[str, ...] = ()) -> str:
    """Bound and redact diagnostic text before it reaches the console/log."""
    text = _compact_text(f"{type(exc).__name__}: {exc}")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    text = re.sub(
        r"(?i)(api[-_ ]?key|authorization|bearer|access[_-]?token|token)"
        r"\s*[:=]\s*[^\s,;]+",
        r"\1=<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)([?&](?:key|token|access_token)=)[^&\s]+",
        r"\1<redacted>",
        text,
    )
    return text


def load_system_instruction(kb_path: str = KB_PATH) -> str:
    """Build the concise phone-agent instruction from the project KB."""
    kb: Dict[str, Any] = {}
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as handle:
                kb = json.load(handle)
        except Exception as exc:
            print(f"[Warning] Failed to parse {_compact_text(kb_path)}: {_compact_text(exc)}")

    business = kb.get("business", {}) or {}
    location = business.get("location", {}) or {}
    products = kb.get("products_and_pricing", []) or []
    offers = kb.get("special_offers", {}) or {}
    faqs = kb.get("common_faqs", {}) or {}
    assistant = kb.get("sales_assistant", {}) or {}

    product_lines = "\n".join(
        f"  - {item.get('category')}: starts at {item.get('starting_price')}; "
        f"features {item.get('features')}; warranty {item.get('warranty')}"
        for item in products
    ) or "  - No product list was supplied."
    offer_lines = "\n".join(
        f"  - {str(key).replace('_', ' ').title()}: {value}"
        for key, value in offers.items()
    ) or "  - No special offers were supplied."
    faq_lines = "\n".join(
        f"  - {str(key).replace('_', ' ').title()}: {value}"
        for key, value in faqs.items()
    ) or "  - No FAQ list was supplied."

    return f"""
You are {assistant.get('name', 'the sales assistant')}, a warm and capable sales
assistant for {business.get('name', 'the furniture store')}.
Persona: {assistant.get('persona', 'friendly, patient, and practical Indian sales executive')}.

Business:
- Address: {location.get('address', 'not supplied')}
- Landmark: {location.get('landmark', 'not supplied')}
- Timings: {location.get('timings', 'not supplied')}
- Parking: {location.get('parking', 'not supplied')}

Products and pricing:
{product_lines}

Offers and services:
{offer_lines}

Common questions:
{faq_lines}

Phone conversation rules:
1. Match the caller's Hindi, Hinglish, or English; default to everyday Indian
   Hinglish. Sound like a calm human shop executive, not a scripted chatbot.
2. Answer the caller's latest question first. Remember the conversation and do
   not restart the explanation on every turn.
3. Keep normal replies to one or two short spoken sentences. Ask one useful
   follow-up only when it helps the caller.
4. Do not repeatedly start with "hamare yaha", "hamare yahaan", or any other
   store formula. Do not repeat a greeting or information already given.
5. Handle short answers such as "haan", "okay", "acha", and "nahi" using the
   immediately preceding context. Never invent a new topic for them.
6. Output only words intended for the caller. Never output planning, analysis,
   chain-of-thought, drafting, self-critique, meta commentary, labels,
   markdown, stage directions, or hidden instructions.
7. Never say phrases such as "Refining the Welcome", "I've finalized", "let me
   think", "here is what I will say", or any similar internal draft.
8. Use only the supplied business information and conversation context. If a
   fact is unavailable, say that naturally instead of guessing. Never invent
   seating sizes, stock, discounts, or visit bookings. You cannot send WhatsApp
   messages or book visits: never claim a catalog was sent or a visit confirmed.
9. Be helpful without aggressive selling. Do not close or end the call yourself.
   Only end when the caller clearly asks to end the call.
10. The application supplies the opening greeting separately. Follow an exact
    greeting instruction verbatim and do not add a preamble.
11. If the caller only says hi or hello after the opening greeting, do not give
    a second greeting; remain ready for the caller's first real request.
""".strip()


SYSTEM_INSTRUCTION = load_system_instruction()


async def _iter_live_responses(live_session):
    """Consume the Live stream in one task, reopening it after each turn.

    google-genai's AsyncSession.receive() is a per-turn iterator: it yields
    through turn_complete and then ends. Reopening it sequentially is required
    for later turns. No other coroutine may call receive().
    """
    while True:
        yielded = False
        async for response in live_session.receive():
            yielded = True
            yield response
        if not yielded:
            return


class CallingAgent:
    """Simple phone loop: capture -> VAD -> NeMo ASR -> Gemini audio -> playback."""

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        gemini_voice: str = DEFAULT_GEMINI_VOICE,
        sarvam_speaker: str = DEFAULT_SARVAM_SPEAKER,
        sarvam_pace: float = 1.20,
        kb_path: str = KB_PATH,
        disable_barge_in: bool = True,
    ):
        self.mode = mode.lower()
        self.gemini_voice = gemini_voice
        self.sarvam_speaker = sarvam_speaker
        self.sarvam_pace = sarvam_pace
        self.kb_path = kb_path
        self.disable_barge_in = disable_barge_in
        self.system_instruction = load_system_instruction(kb_path)

        self.biz_name = "our store"
        try:
            with open(kb_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.biz_name = str(data.get("business", {}).get("name", self.biz_name))
        except Exception:
            pass

        self.audio_q: queue.Queue = queue.Queue(maxsize=200)
        self.is_running = False
        self.is_speaking = False
        self.barge_in_active = False
        self._turn_interrupted = False
        self._stop_requested = False

        self.playback: Optional[PlaybackWorker] = None
        self._out_stream = None
        self._agent_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._interruption_boundary_event: Optional[asyncio.Event] = None
        self._session_failure: Optional[BaseException] = None
        self._send_lock: Optional[asyncio.Lock] = None
        self._session_resumption_handle: Optional[str] = None

        self.current_turn_id: Optional[str] = None
        self._turn_metrics: Dict[str, Any] = {}
        self._finished_turn_ids: set[str] = set()
        self._turn_sequence = 0
        self._gemini_event_logged = False
        self._gemini_audio_logged = False
        self._playback_started_logged = False
        self._audio_chunks_received = 0
        self._awaiting_first_meaningful_input = True
        self._input_dropped = 0
        self._input_overflows = 0
        self._echo_suppressed = 0
        self._mic_resume_after = 0.0

        self.api_key = self._load_secret("GEMINI_API_KEY", "gemini_api_key.txt")
        if not self.api_key:
            print("\n" + "=" * 55)
            print("  GEMINI API KEY NOT FOUND")
            print("  Please paste your Gemini API Key below:")
            print("=" * 55)
            try:
                entered = input("Enter Gemini API Key: ").strip()
            except (KeyboardInterrupt, EOFError):
                entered = ""
            if entered:
                self.api_key = entered
                try:
                    with open("gemini_api_key.txt", "w", encoding="utf-8") as handle:
                        handle.write(entered)
                    print("[Agent] API key saved to gemini_api_key.txt!\n")
                except Exception as exc:
                    self._emit_turn_error(None, "api_key_save", exc)

        self.sarvam_api_key = self._load_secret("SARVAM_API_KEY", "sarvam_api_key.txt")

        print("[Agent] Loading NeMo-Speech ASR model...")
        self.asr = NeMoStreamingASR(model_path=MODEL_PATH, gpu=-1)
        print("[Agent] ASR Model loaded successfully!")
        self.asr_worker: Optional[AsrWorker] = None
        self._worker_summary_printed = False

        self.client = genai.Client(api_key=self.api_key)
        self.chat = None
        self.http_client: Optional[httpx.AsyncClient] = httpx.AsyncClient(timeout=10.0)

        self.call_id = uuid.uuid4().hex[:8]
        self.metrics = MetricsLogger(
            call_id=self.call_id,
            silence_window_ms=SILENCE_TIMEOUT * 1000.0,
        )

        self.vad = SileroVADEndpointer(
            sample_rate=SAMPLE_RATE,
            speech_threshold=0.50,
            barge_in_speech_threshold=0.85,
            silence_threshold=0.35,
            min_speech_duration_ms=96.0,
            barge_in_min_speech_duration_ms=160.0,
            min_silence_duration_ms=SILENCE_TIMEOUT * 1000.0,
            speech_pad_ms=64.0,
            energy_fallback_threshold=ENERGY_THRESHOLD,
            energy_fallback_barge_in_threshold=BARGE_IN_THRESHOLD,
        )

    def _init_gemini_text_chat(self):
        """Compatibility hook for integrations that used the old text path.

        The real-time call loop intentionally uses Gemini native audio. This
        method remains available for older code that explicitly requests a
        legacy text-chat object.
        """
        for model_name in ("gemini-2.5-flash", "gemini-flash-latest"):
            try:
                self.chat = self.client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        temperature=0.7,
                    ),
                )
                return self.chat
            except Exception:
                continue
        return None

    @staticmethod
    def _load_secret(env_name: str, file_name: str) -> str:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
        try:
            if os.path.exists(file_name):
                with open(file_name, "r", encoding="utf-8") as handle:
                    return handle.read().strip()
        except Exception:
            pass
        return ""

    def _audio_callback(self, indata, frames, time_info, status):
        """Sounddevice callback: normalize only, never run inference or I/O."""
        del frames, time_info
        captured_at = time.perf_counter()
        if status and getattr(status, "input_overflow", False):
            self._input_overflows += 1
        # Gate at capture time is removed. We process all audio in the event loop
        # to maintain accurate VAD noise floors and allow barge-in.
        if indata.ndim > 1 and indata.shape[1] > 1:
            samples = np.mean(indata, axis=1, dtype=np.float32)
        else:
            samples = indata.copy().flatten().astype(np.float32)

        try:
            self.audio_q.put_nowait((captured_at, samples))
        except queue.Full:
            # Keep the newest audio. A blocked callback would create a much
            # worse failure mode than dropping one stale capture block.
            try:
                self.audio_q.get_nowait()
                self._input_dropped += 1
            except queue.Empty:
                pass
            try:
                self.audio_q.put_nowait((captured_at, samples))
            except queue.Full:
                self._input_dropped += 1

    async def _finalize_asr_turn(self, turn_id: str):
        task = asyncio.create_task(self.asr_worker.finalize_turn(timeout=ASR_FLUSH_TIMEOUT))
        try:
            try:
                return await asyncio.wait_for(asyncio.shield(task), ASR_SLOW_SECONDS)
            except asyncio.TimeoutError:
                self._emit_turn_event(turn_id, "ASR_SLOW", json.dumps(self.asr_worker.stats()))
                return await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def _next_turn_id(self, prefix: str = "TURN") -> str:
        self._turn_sequence += 1
        return f"{prefix}_{int(time.time() * 1000)}_{self._turn_sequence}"

    def _emit_turn_event(
        self, turn_id: Optional[str], event: str, detail: str = ""
    ) -> None:
        marker = turn_id or "TURN_UNKNOWN"
        suffix = f" {detail}" if detail else ""
        print(f"[{marker}] {time.time():.3f} {event}{suffix}", flush=True)

    def _emit_turn_error(
        self, turn_id: Optional[str], stage: str, exc: BaseException
    ) -> None:
        secrets = tuple(
            item
            for item in (
                getattr(self, "api_key", ""),
                getattr(self, "sarvam_api_key", ""),
            )
            if item
        )
        self._emit_turn_event(
            turn_id,
            "ERROR",
            f"stage={stage} detail={_safe_error_text(exc, secrets)}",
        )

    def _register_turn_context(self, turn_id: str, turn: Any) -> None:
        self.current_turn_id = turn_id
        if turn is not None:
            self._turn_metrics[turn_id] = turn
        self._gemini_event_logged = False
        self._gemini_audio_logged = False
        self._playback_started_logged = False
        self._audio_chunks_received = 0

    def _active_turn_metrics(self) -> Any:
        if self.current_turn_id is None:
            return None
        return self._turn_metrics.get(self.current_turn_id)

    def _finish_turn(
        self,
        turn_id: Optional[str],
        turn: Any = None,
        error: Optional[str] = None,
    ) -> None:
        if not turn_id or turn_id in self._finished_turn_ids:
            return
        record = self._turn_metrics.get(turn_id) or turn
        if record is None:
            return
        if error and not getattr(record, "error", None):
            record.error = _compact_text(error)
        self._finished_turn_ids.add(turn_id)
        try:
            self.metrics.finish_turn(record)
        except Exception as exc:
            self._emit_turn_error(turn_id, "metrics_finish", exc)

    def _mark_turn(self, turn: Any, mark: str, overwrite: bool = False) -> None:
        if turn is None:
            return
        try:
            turn.mark(mark, overwrite=overwrite)
        except Exception as exc:
            self._emit_turn_error(self.current_turn_id, f"metrics_{mark}", exc)

    def _mark_playback_started(self, turn_id: str, turn: Any) -> None:
        if self._playback_started_logged:
            return
        self._playback_started_logged = True
        self._mark_turn(turn, "playback_started")
        self._emit_turn_event(turn_id, "PLAYBACK_START")

    def _asr_worker_summary(self) -> str:
        if self.asr_worker is None or self._worker_summary_printed:
            return ""
        self._worker_summary_printed = True
        stats = self.asr_worker.stats()
        lines = [
            "",
            "=== ASR decode thread ===",
            f"chunks submitted {stats['submitted_chunks']}, decoded {stats['decoded_chunks']}, "
            f"dropped {stats['dropped_chunks']}",
            f"peak backlog {stats['max_backlog_seconds']}s of "
            f"{stats['max_backlog_bound_seconds']}s bound ({stats['max_queue_depth']} chunks)",
            f"decode thread busy {stats['decode_seconds']}s",
        ]
        if self._input_dropped:
            lines.append(f"input callback dropped {self._input_dropped} stale block(s)")
        if stats["dropped_chunks"]:
            lines.append(
                "WARNING: ASR queue overflow dropped audio; transcript may contain gaps."
            )
        if stats["queued_items_at_exit"]:
            lines.append(
                f"NOTE: {stats['queued_items_at_exit']} audio block(s) remained queued at exit."
            )
        if stats["dropped_results"]:
            lines.append(f"NOTE: {stats['dropped_results']} stale ASR result(s) discarded.")
        if stats["error_count"]:
            lines.append(
                f"errors {stats['error_count']}, last: {_compact_text(stats['last_error'])}"
            )
        return "\n".join(lines)

    async def play_audio_file(self, file_path: str):
        """Retain the legacy helper without using it in the native path."""
        global pygame
        if pygame is None:
            try:
                import pygame as pygame_module
                pygame = pygame_module
            except Exception:
                pygame = False
        if pygame is None:
            print("[Audio Playback Error] pygame is not installed.")
            return
        if pygame is False:
            print("[Audio Playback Error] pygame is not installed.")
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=GEMINI_OUTPUT_SAMPLE_RATE)
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self.is_running:
                await asyncio.sleep(0.01)
            pygame.mixer.music.unload()
        except Exception as exc:
            print(f"[Audio Playback Error] {_safe_error_text(exc)}")
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    async def _wait_for_interruption_boundary(self, turn_id: Optional[str]) -> None:
        event = self._interruption_boundary_event
        if event is None or event.is_set():
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=INTERRUPTION_BOUNDARY_TIMEOUT)
        except asyncio.TimeoutError as exc:
            self._emit_turn_error(turn_id, "interruption_boundary_timeout", exc)

    async def _start_barge_in(self, live_session, turn_id: Optional[str]) -> None:
        """Interrupt playback and mark the caller activity exactly once."""
        if self.barge_in_active:
            return

        self._emit_turn_event(turn_id, "BARGE_IN")
        print("\n[Agent] Caller barge-in detected! Halting playback...")
        if self.playback is not None:
            self.playback.interrupt()
        self._turn_interrupted = True
        self.is_speaking = False
        if self._interruption_boundary_event is not None:
            self._interruption_boundary_event.clear()

        try:
            await live_session.send_realtime_input(
                activity_start=types.ActivityStart()
            )
            self.barge_in_active = True
        except Exception as exc:
            self._emit_turn_error(turn_id, "activity_start", exc)
            if self._is_retryable_session_error(exc):
                self._session_failure = exc

    async def _end_barge_in(
        self,
        live_session,
        turn_id: Optional[str],
        wait_for_boundary: bool = True,
    ) -> None:
        """Always close a barge-in activity, including empty/noise turns."""
        if not self.barge_in_active:
            return

        try:
            await live_session.send_realtime_input(
                activity_end=types.ActivityEnd()
            )
        except Exception as exc:
            self._emit_turn_error(turn_id, "activity_end", exc)
            if self._is_retryable_session_error(exc):
                self._session_failure = exc
        finally:
            # Never leave the local state stuck just because the server call
            # failed. A reconnect, if needed, is handled by run().
            self.barge_in_active = False

        if wait_for_boundary:
            await self._wait_for_interruption_boundary(turn_id)

    async def _await_agent_task(self) -> None:
        task = self._agent_task
        if task is None or task.done():
            return
        try:
            await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit_turn_error(self.current_turn_id, "agent_task", exc)

    async def _receive_loop(self, live_session):
        """The only coroutine allowed to consume Gemini Live responses."""
        print("[Agent] Started dedicated Gemini receive loop.")
        try:
            async for response in _iter_live_responses(live_session):
                turn_id = self.current_turn_id
                turn = self._active_turn_metrics()

                if not self._gemini_event_logged:
                    self._gemini_event_logged = True
                    self._emit_turn_event(turn_id, "GEMINI_EVENT_RECEIVED")

                resumption = getattr(response, "session_resumption_update", None)
                new_handle = getattr(resumption, "new_handle", None)
                if new_handle:
                    self._session_resumption_handle = str(new_handle)

                server_content = getattr(response, "server_content", None)
                if server_content is None:
                    continue

                if bool(getattr(server_content, "interrupted", False)):
                    self._emit_turn_event(turn_id, "GEMINI_INTERRUPTED")
                    if self.playback is not None:
                        self.playback.interrupt()
                    self._turn_interrupted = True
                    self.is_speaking = False
                    if self._interruption_boundary_event is not None:
                        self._interruption_boundary_event.set()
                    self._finish_turn(
                        turn_id,
                        error="Gemini response interrupted by caller",
                    )
                    continue

                model_turn = getattr(server_content, "model_turn", None)
                parts = getattr(model_turn, "parts", None) if model_turn else None
                for part in parts or []:
                    if getattr(part, "thought", False):
                        continue
                    inline_data = getattr(part, "inline_data", None)
                    mime = getattr(inline_data, "mime_type", "") or ""
                    if not mime.lower().startswith("audio/pcm"):
                        continue
                    audio_data = getattr(inline_data, "data", None)
                    if not audio_data:
                        continue

                    if isinstance(audio_data, str):
                        try:
                            audio_data = base64.b64decode(audio_data)
                        except Exception as exc:
                            self._emit_turn_error(turn_id, "audio_decode", exc)
                            continue
                    elif isinstance(audio_data, memoryview):
                        audio_data = audio_data.tobytes()
                    elif isinstance(audio_data, bytearray):
                        audio_data = bytes(audio_data)

                    if not isinstance(audio_data, bytes):
                        self._emit_turn_error(
                            turn_id,
                            "audio_type",
                            TypeError(
                                "unsupported audio payload type "
                                f"{type(audio_data).__name__}"
                            ),
                        )
                        continue

                    self._audio_chunks_received += 1
                    if not self._gemini_audio_logged:
                        self._gemini_audio_logged = True
                        self._mark_turn(turn, "first_audio")
                        self._emit_turn_event(
                            turn_id,
                            "GEMINI_AUDIO_RECEIVED",
                            f"bytes={len(audio_data)}",
                        )
                    if turn is not None:
                        turn.audio_bytes += len(audio_data)

                    # Consume interrupted trailing frames but never play them.
                    if self._turn_interrupted:
                        continue
                    if self.playback is not None:
                        self._mark_playback_started(turn_id or "TURN_UNKNOWN", turn)
                        self.playback.enqueue(audio_data)

                if bool(getattr(server_content, "turn_complete", False)):
                    self._mark_turn(turn, "turn_complete")
                    self._emit_turn_event(
                        turn_id,
                        "GEMINI_TURN_COMPLETE",
                        f"audio_chunks={self._audio_chunks_received}",
                    )
                    self.is_speaking = False
                    if self._interruption_boundary_event is not None:
                        self._interruption_boundary_event.set()
                    self._finish_turn(turn_id)

            if self.is_running and not self._stop_requested:
                print("\n[Agent] Call pending end. Waiting up to 5 seconds for customer response...")
                self._agent_task = asyncio.create_task(asyncio.sleep(0.0))  # clear task
                # Wait for 5 seconds. If the customer speaks, `capture["turn"]` will become not None.
                for _ in range(50):
                    if not self.is_running or self._stop_requested:
                        break
                    
                    # If customer spoke, trigger reconnect
                    if capture_obj := getattr(self, "_grace_capture", None):
                        if capture_obj["turn"] is not None:
                            print("\n[Agent] Speech detected during grace period! Resuming...")
                            failure = ConnectionError("Reconnect required for barge-in after goodbye")
                            self._emit_turn_error(self.current_turn_id, "gemini_receive", failure)
                            self._session_failure = failure
                            return
                    
                    await asyncio.sleep(0.1)

                if self.is_running and not self._stop_requested:
                    failure = ConnectionError("Grace period expired, stream closed")
                    self._emit_turn_error(self.current_turn_id, "gemini_receive", failure)
                    self._session_failure = failure
        except asyncio.CancelledError:
            print("[Agent] Receive loop cancelled.")
            if self.playback is not None:
                self.playback.interrupt()
            raise
        except Exception as exc:
            self._emit_turn_error(self.current_turn_id, "gemini_receive", exc)
            self._session_failure = exc
            if self.playback is not None:
                self.playback.interrupt()
        finally:
            self.is_speaking = False

    async def generate_gemini_native_live_audio(
        self,
        live_session,
        user_transcript: str,
        turn=None,
        turn_id: Optional[str] = None,
    ):
        """Send one completed caller turn to the persistent native-audio model."""
        if not user_transcript or not user_transcript.strip():
            self._emit_turn_event(turn_id, "SEND_SKIPPED", "reason=empty_transcript")
            return
        if turn_id is None:
            turn_id = self._next_turn_id()
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()

        async with self._send_lock:
            self._register_turn_context(turn_id, turn)
            self._turn_interrupted = False
            self.is_speaking = True

            print(f"\n[{turn_id}] [User]: {_compact_text(user_transcript, 1000)}")
            print(
                f"[{turn_id}] [Agent ({_compact_text(self.gemini_voice, 40)})]: ",
                end="",
                flush=True,
            )
            self._emit_turn_event(turn_id, "SEND_START")

            try:
                await asyncio.wait_for(
                    live_session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=user_transcript)],
                        ),
                        turn_complete=True,
                    ),
                    timeout=GEMINI_SEND_TIMEOUT,
                )
                self._emit_turn_event(turn_id, "SEND_OK")
                self._mark_turn(turn, "llm_sent")
            except asyncio.CancelledError:
                self.is_speaking = False
                raise
            except Exception as exc:
                self._emit_turn_error(turn_id, "gemini_send", exc)
                self.is_speaking = False
                if self._is_retryable_session_error(exc):
                    self._session_failure = exc
                if turn is not None:
                    turn.error = _safe_error_text(
                        exc,
                        tuple(item for item in (self.api_key, self.sarvam_api_key) if item),
                    )
                self._finish_turn(turn_id)

    async def _run_agent_turn(
        self,
        live_session,
        user_transcript: str,
        turn=None,
        turn_id: Optional[str] = None,
    ):
        if turn_id is None:
            turn_id = self._next_turn_id()
        try:
            await self.generate_gemini_native_live_audio(
                live_session,
                user_transcript,
                turn=turn,
                turn_id=turn_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit_turn_error(turn_id, "agent_turn", exc)
            self.is_speaking = False
            self._finish_turn(turn_id, error=_safe_error_text(exc))

    @staticmethod
    def _is_greeting_only(text: str) -> bool:
        normalized = re.sub(
            r"[^\w\s\u0900-\u097F]",
            " ",
            text.lower(),
            flags=re.UNICODE,
        )
        tokens = normalized.split()
        if not tokens or len(tokens) > 4:
            return False
        greetings = {
            "hi",
            "hii",
            "hiii",
            "hello",
            "helo",
            "helloo",
            "hey",
            "heyy",
            "namaste",
            "namaskar",
            "ji",
            "sir",
            "madam",
            "हेलो",
            "हैलो",
            "नमस्ते",
            "नमस्कार",
            "जी",
        }
        return all(token in greetings for token in tokens)

    def _deterministic_greeting_prompt(self) -> str:
        return 'Say exactly this greeting and nothing else, verbatim: "Hi!"'

    @staticmethod
    def _is_retryable_session_error(exc: BaseException) -> bool:
        name = type(exc).__name__.lower()
        detail = str(exc).lower()
        return (
            "connectionclosed" in name
            or "websocket" in name
            or "keepalive ping timeout" in detail
            or "grace period" in detail
            or "reconnect required" in detail
            or "no close frame" in detail
            or "1011" in detail
            or "connection reset" in detail
        )

    def _live_config(self) -> types.LiveConnectConfig:
        kwargs: Dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
            "enable_affective_dialog": True,
            "speech_config": types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.gemini_voice,
                    )
                )
            ),
            "system_instruction": self.system_instruction,
            "realtime_input_config": types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True
                )
            ),
        }
        if self._session_resumption_handle:
            kwargs["session_resumption"] = types.SessionResumptionConfig(
                handle=self._session_resumption_handle
            )
        return types.LiveConnectConfig(**kwargs)

    async def run(self, device: int = None):
        """Run the call; retry only transient Live-session failures."""
        settings = {
            "mode": self.mode,
            "gemini_voice": self.gemini_voice,
            "sarvam_speaker": self.sarvam_speaker,
            "sarvam_pace": self.sarvam_pace,
            "kb_path": self.kb_path,
            "disable_barge_in": self.disable_barge_in,
        }

        for attempt in range(GEMINI_RECONNECT_LIMIT + 1):
            try:
                await self._run_once(device)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = self._session_failure or exc
                if (
                    attempt >= GEMINI_RECONNECT_LIMIT
                    or not self._is_retryable_session_error(failure)
                ):
                    raise

                handle = self._session_resumption_handle
                delay = GEMINI_RECONNECT_BACKOFF_SECONDS[
                    min(attempt, len(GEMINI_RECONNECT_BACKOFF_SECONDS) - 1)
                ]
                print(
                    f"\n[Agent] Gemini Live connection lost; reconnecting in "
                    f"{delay:.0f}s (attempt {attempt + 1}/{GEMINI_RECONNECT_LIMIT})..."
                )
                await asyncio.sleep(delay)
                self.__init__(**settings)
                self._session_resumption_handle = handle

    async def _run_once(self, device: int = None):
        """Own one call lifecycle and one clear capture/finalize loop."""
        self.is_running = True
        self._stop_requested = False
        self._session_failure = None
        self._awaiting_first_meaningful_input = True
        self.audio_q = queue.Queue(maxsize=200)
        self._receive_task = None
        self._agent_task = None
        self.asr_worker = None
        self.playback = None
        self._out_stream = None

        in_stream = None
        try:
            if self.vad.using_fallback:
                raise RuntimeError("Silero VAD is unavailable; refusing noise-prone energy-only calling mode")
            dev_info = sd.query_devices(device, "input")
            # Request mono explicitly; averaging arbitrary stereo device channels
            # can attenuate speech or cancel opposite-phase channels.
            in_channels = 1
            self._emit_turn_event(None, "INPUT_DEVICE", json.dumps({
                "name": dev_info.get("name"), "sample_rate": SAMPLE_RATE,
                "channels": in_channels, "block_samples": CHUNK_SAMPLES,
                "default_sample_rate": dev_info.get("default_samplerate"),
                "barge_in_disabled": self.disable_barge_in,
            }))
            in_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=in_channels,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                device=device,
                callback=self._audio_callback,
            )
            self._out_stream = sd.RawOutputStream(
                samplerate=GEMINI_OUTPUT_SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            self.playback = PlaybackWorker(out_stream=self._out_stream)

            print("\n" + "=" * 60)
            print("  NeMo-Speech + Gemini Live Multimodal AI Calling Agent")
            print(f"  Mode: {self.mode.upper()} | Voice: {self.gemini_voice}")
            print("  Audio Output: Direct Low-Latency 24kHz PCM Stream")
            print(f"  Endpoint silence: {SILENCE_TIMEOUT}s | Press Ctrl+C to exit")
            print("=" * 60 + "\n")

            config = self._live_config()
            with in_stream:
                print(f"[Agent] Connecting to Gemini Live Session ({self.biz_name})...")
                async with self.client.aio.live.connect(
                    model=GEMINI_MODEL,
                    config=config,
                ) as live_session:
                    print("[Agent] Connected! Starting voice session...")
                    self._interruption_boundary_event = asyncio.Event()

                    # This is the sole receiver for the entire persistent
                    # session. It starts before the greeting.
                    self._receive_task = asyncio.create_task(
                        self._receive_loop(live_session)
                    )
                    await asyncio.sleep(0.05)

                    self.asr_worker = AsrWorker(
                        self.asr,
                        sample_rate=SAMPLE_RATE,
                        language_code="hi",
                        interim_results=True,
                    )
                    self.asr_worker.start()

                    if self._session_resumption_handle:
                        print("[Agent] Gemini conversation resumed; waiting for caller...")
                    else:
                        greeting = self.metrics.start_turn(kind="greeting")
                        greeting_id = self._next_turn_id("TURN_GREETING")
                        self._agent_task = asyncio.create_task(
                            self._run_agent_turn(
                                live_session,
                                self._deterministic_greeting_prompt(),
                                turn=greeting,
                                turn_id=greeting_id,
                            )
                        )

                    # All caller state is local to this one loop. VAD owns
                    # endpoint detection; this loop only finalizes one caller
                    # capture and sends one completed transcript.
                    capture: Dict[str, Any] = {
                        "turn": None,
                        "turn_id": None,
                        "audio_samples": 0,
                        "transcript": "",
                        "last_printed": "",
                        "open": False,
                        "input_dropped_start": 0,
                        "input_overflows_start": 0,
                    }
                    self._grace_capture = capture

                    def ensure_caller_turn() -> Any:
                        if capture["turn"] is None:
                            capture["turn"] = self.metrics.start_turn(kind="caller")
                            capture["turn_id"] = self._next_turn_id("TURN_CALLER")
                            capture["diagnostic_start"] = self.asr_worker.stats()
                            capture["max_capture_age_ms"] = 0.0
                            capture["energy_sum"] = 0.0
                            capture["peak"] = 0.0
                            self._emit_turn_event(capture["turn_id"], "VAD_START")
                        return capture["turn"]

                    def reset_capture() -> None:
                        capture.update(
                            {
                                "turn": None,
                                "turn_id": None,
                                "audio_samples": 0,
                                "transcript": "",
                                "last_printed": "",
                                "open": False,
                                "input_dropped_start": self._input_dropped,
                                "input_overflows_start": self._input_overflows,
                            }
                        )
                        self.vad.reset()

                    async def finalize_caller_turn() -> None:
                        caller_turn = capture["turn"]
                        caller_id = capture["turn_id"]
                        if caller_turn is None or caller_id is None:
                            reset_capture()
                            return

                        if capture["audio_samples"] <= 0:
                            self._emit_turn_event(
                                caller_id,
                                "ASR_EMPTY",
                                "reason=no_audio",
                            )
                            await self._end_barge_in(
                                live_session,
                                caller_id,
                                wait_for_boundary=True,
                            )
                            self._finish_turn(
                                caller_id,
                                caller_turn,
                                error="no captured speech audio",
                            )
                            reset_capture()
                            return

                        self._mark_turn(caller_turn, "turn_fired")
                        interim_text = str(capture["transcript"]).strip()
                        before = self.asr_worker.stats()
                        self._emit_turn_event(caller_id, "VAD_END", json.dumps({
                            "audio_seconds": capture["audio_samples"] / SAMPLE_RATE,
                            "asr": before,
                        }))
                        result = await self._finalize_asr_turn(caller_id)
                        after = self.asr_worker.stats()
                        lost_audio = (
                            after["dropped_chunks"] > capture["diagnostic_start"]["dropped_chunks"]
                            or self._input_dropped > capture["input_dropped_start"]
                            or self._input_overflows > capture["input_overflows_start"]
                        )
                        final_text = result.text.strip()
                        user_text = final_text if result.status == "final" and not lost_audio else ""
                        diagnostics = {
                            "status": "audio_loss" if lost_audio else result.status,
                            "queue_wait_ms": result.queue_wait_ms,
                            "flush_ms": result.flush_ms,
                            "audio_seconds": capture["audio_samples"] / SAMPLE_RATE,
                            "max_capture_age_ms": round(capture["max_capture_age_ms"], 1),
                            "rms": round((capture["energy_sum"] / capture["audio_samples"]) ** 0.5, 5),
                            "peak": round(capture["peak"], 5),
                            "input_dropped_total": self._input_dropped,
                            "input_overflows_total": self._input_overflows,
                            "echo_suppressed_total": self._echo_suppressed,
                            "before": before, "after": after,
                        }
                        self._emit_turn_event(caller_id, "ASR_DIAGNOSTICS", json.dumps(diagnostics))
                        caller_turn.asr_diagnostics = diagnostics
                        caller_turn.interim_transcript = result.interim or interim_text
                        caller_turn.final_transcript = final_text
                        caller_turn.used_final = bool(user_text)

                        if not user_text:
                            self._emit_turn_event(
                                caller_id,
                                "ASR_EMPTY",
                                f"reason={diagnostics['status']} audio_samples={capture['audio_samples']}",
                            )
                            await self._end_barge_in(
                                live_session,
                                caller_id,
                                wait_for_boundary=True,
                            )
                            self._finish_turn(
                                caller_id,
                                caller_turn,
                                error=f"ASR rejected turn: {diagnostics['status']}",
                            )
                            reset_capture()
                            return

                        self._mark_turn(caller_turn, "asr_final")
                        greeting_only = (
                            self._awaiting_first_meaningful_input
                            and self._is_greeting_only(user_text)
                        )
                        detail = (
                            f"chars={len(user_text)} "
                            f"used_final={bool(final_text)}"
                        )
                        if greeting_only:
                            detail += " ignored=greeting_only"
                        self._emit_turn_event(caller_id, "ASR_FINAL", detail)
                        print(
                            f"\r[Transcribing]: {_compact_text(user_text, 1000)} "
                            f"({'greeting ignored; listening' if greeting_only else 'Final'})"
                        )

                        # End the explicit activity before any next text turn.
                        await self._end_barge_in(
                            live_session,
                            caller_id,
                            wait_for_boundary=True,
                        )

                        if greeting_only:
                            self._finish_turn(caller_id, caller_turn)
                            print(
                                "\n[Agent] Greeting heard; listening for your request..."
                            )
                            reset_capture()
                            return

                        self._awaiting_first_meaningful_input = False
                        await self._await_agent_task()
                        self._agent_task = asyncio.create_task(
                            self._run_agent_turn(
                                live_session,
                                user_text,
                                turn=caller_turn,
                                turn_id=caller_id,
                            )
                        )
                        reset_capture()
                        print("\n[Agent] Listening for your voice...")

                    print("\n[Agent] Listening for your voice...")
                    while self.is_running:
                        if self._session_failure is not None:
                            raise self._session_failure

                        # Echo is rejected by _audio_callback at capture time.
                        # Audio already admitted to the queue is caller audio,
                        # even if a reply starts before we get around to it.
                        # Leave it queued until generation AND playback finish;
                        # never dequeue and discard it based on current state.
                        chunk_batch = []
                        capture_age_ms = 0.0
                        try:
                            captured_at, chunk = self.audio_q.get_nowait()
                            capture_age_ms = (time.perf_counter() - captured_at) * 1000
                            chunk_batch.append(chunk)
                        except queue.Empty:
                            pass

                        endpoint_detected = False
                        if chunk_batch:
                            to_submit = []

                            for chunk in chunk_batch:
                                # In half-duplex mode capture-time admission is
                                # authoritative. Only experimental barge-in uses
                                # current playback state to classify speech.
                                agent_active = self.is_speaking or (self.playback is not None and self.playback.is_playing())
                                if agent_active:
                                    self._agent_active_until = time.perf_counter() + ECHO_GUARD_SECONDS
                                vad_agent_active = agent_active or time.perf_counter() < getattr(self, "_agent_active_until", 0)
                                
                                if vad_agent_active and not getattr(self, "_was_agent_active", False):
                                    # Agent just started speaking! Force endpoint any open turn.
                                    if capture["turn"] is not None:
                                        await finalize_caller_turn()
                                self._was_agent_active = vad_agent_active

                                event = self.vad.process_chunk(
                                    chunk,
                                    agent_active=vad_agent_active,
                                    clear_pad=(self.playback is not None and self.playback.is_playing()),
                                )

                                # A normal caller turn starts when VAD confirms
                                # speech. During playback the same transition
                                # is an explicit barge-in.
                                if event.barge_in and not self.disable_barge_in:
                                    caller_turn = ensure_caller_turn()
                                    capture["open"] = True
                                    self._mark_turn(caller_turn, "speech_started")
                                    await self._start_barge_in(
                                        live_session,
                                        capture["turn_id"],
                                    )
                                    vad_agent_active = False
                                    self._agent_active_until = 0

                                if not (vad_agent_active and self.disable_barge_in):
                                    if event.speech_start or event.is_speech:
                                        caller_turn = ensure_caller_turn()
                                        capture["open"] = True
                                        self._mark_turn(
                                            caller_turn,
                                            "speech_started",
                                        )
                                        if event.speech_start or event.speech_probability >= self.vad.silence_threshold:
                                            self._mark_turn(caller_turn, "last_voice", overwrite=True)

                                if event.audio_for_asr is not None:
                                    # After a confirmed barge-in, accept the
                                    # same capture block's onset audio. Without
                                    # this, the first consonant can be lost.
                                    can_capture = (
                                        not vad_agent_active or capture["open"]
                                    )
                                    if can_capture and not (
                                        vad_agent_active and self.disable_barge_in
                                    ):
                                        to_submit.append(event.audio_for_asr)

                                endpoint_detected = (
                                    endpoint_detected or event.endpoint_detected
                                )

                            if to_submit and self.asr_worker is not None:
                                submitted = np.concatenate(to_submit).astype(
                                    np.float32,
                                    copy=False,
                                )
                                accepted_without_drop = self.asr_worker.submit_audio(
                                    submitted
                                )
                                # False means audio loss (an old block evicted,
                                # or an oversized new block rejected).
                                if capture["turn"] is not None:
                                    capture["audio_samples"] += len(submitted)
                                    capture["energy_sum"] += float(np.sum(submitted ** 2, dtype=np.float64))
                                    capture["peak"] = max(capture["peak"], float(np.max(np.abs(submitted))))
                                    capture["max_capture_age_ms"] = max(
                                        capture["max_capture_age_ms"], capture_age_ms
                                    )
                                if not accepted_without_drop:
                                    self._emit_turn_event(
                                        capture["turn_id"],
                                        "ASR_AUDIO_DROPPED",
                                        f"samples={len(submitted)}",
                                    )

                            if self.asr_worker is not None:
                                for is_final, transcript in self.asr_worker.drain_results():
                                    del is_final
                                    text = str(transcript or "").strip()
                                    if not text or capture["turn"] is None:
                                        continue
                                    caller_turn = capture["turn"]
                                    capture["transcript"] = text
                                    if text != capture["last_printed"]:
                                        print(
                                            f"\r[Transcribing]: {_compact_text(text, 1000)}",
                                            end="",
                                            flush=True,
                                        )
                                        capture["last_printed"] = text
                                    self._mark_turn(caller_turn, "first_partial")
                                    caller_turn.partial_count += 1

                        if endpoint_detected and capture["turn"] is not None:
                            await finalize_caller_turn()

                        await asyncio.sleep(0 if chunk_batch else 0.01)

        finally:
            self.is_running = False
            self._stop_requested = True

            if self._agent_task is not None and not self._agent_task.done():
                self._agent_task.cancel()
                try:
                    await self._agent_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    self._emit_turn_error(self.current_turn_id, "agent_shutdown", exc)

            if self._receive_task is not None and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    self._emit_turn_error(self.current_turn_id, "receive_shutdown", exc)

            if self.playback is not None:
                try:
                    self.playback.stop()
                except Exception as exc:
                    self._emit_turn_error(self.current_turn_id, "playback_shutdown", exc)

            if self.asr_worker is not None:
                worker_thread = getattr(self.asr_worker, "_thread", None)
                self.asr_worker.stop()
                summary = self._asr_worker_summary()
                if summary:
                    print(summary)
                # Do not destroy the native recognizer while a timed-out
                # worker thread could still be inside the DLL.
                if worker_thread is not None and worker_thread.is_alive():
                    self._emit_turn_error(
                        self.current_turn_id,
                        "asr_close",
                        RuntimeError(
                            "decode thread still active; recognizer left alive "
                            "for safe process exit"
                        ),
                    )
                else:
                    try:
                        self.asr.close()
                    except Exception as exc:
                        self._emit_turn_error(self.current_turn_id, "asr_close", exc)
            else:
                try:
                    self.asr.close()
                except Exception as exc:
                    self._emit_turn_error(self.current_turn_id, "asr_close", exc)

            if self._out_stream is not None:
                try:
                    self._out_stream.close()
                except Exception as exc:
                    self._emit_turn_error(
                        self.current_turn_id,
                        "output_close",
                        exc,
                    )
                self._out_stream = None

            if self.vad is not None:
                self.vad.reset()

            for turn_id in list(self._turn_metrics):
                if turn_id not in self._finished_turn_ids:
                    self._finish_turn(
                        turn_id,
                        error="call stopped before turn complete",
                    )

            if self.http_client is not None:
                try:
                    await self.http_client.aclose()
                except Exception as exc:
                    self._emit_turn_error(self.current_turn_id, "http_close", exc)
                self.http_client = None

            self.metrics.close()
            print("\n[Agent] Stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Live NeMo-Speech + Gemini Multimodal Calling Agent"
    )
    parser.add_argument("--device", type=int, default=None, help="Input audio device index")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["gemini-live", "sarvam"],
        help="Calling engine mode (native Gemini audio is the primary path)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=DEFAULT_GEMINI_VOICE,
        choices=["Puck", "Aoede", "Kore", "Fenrir", "Charon"],
        help="Gemini native voice",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=DEFAULT_SARVAM_SPEAKER,
        help="Legacy Sarvam speaker compatibility option",
    )
    parser.add_argument("--kb", type=str, default=KB_PATH, help="Knowledge base JSON path")
    barge = parser.add_mutually_exclusive_group()
    barge.add_argument(
        "--disable-barge-in", dest="disable_barge_in", action="store_true",
        help="Ignore speaker echo/interruptions (default)",
    )
    barge.add_argument(
        "--enable-barge-in", dest="disable_barge_in", action="store_false",
        help="Experimental: allow interruptions; use headphones to avoid echo",
    )
    parser.set_defaults(disable_barge_in=True)
    args = parser.parse_args()

    agent = CallingAgent(
        mode=args.mode,
        gemini_voice=args.voice,
        sarvam_speaker=args.speaker,
        kb_path=args.kb,
        disable_barge_in=args.disable_barge_in,
    )
    try:
        asyncio.run(agent.run(device=args.device))
    except KeyboardInterrupt:
        print("\n[Agent] Exited cleanly by user.")
    finally:
        if agent.playback is not None:
            agent.playback.stop()
        if agent.asr_worker is not None:
            agent.asr_worker.stop()
            summary = agent._asr_worker_summary()
            if summary:
                print(summary)
        agent.metrics.close()
