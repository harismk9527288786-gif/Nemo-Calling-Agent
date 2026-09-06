import json
import asyncio
import os
import sys
import time
import queue
import re
import uuid
import base64
import wave
import numpy as np
import sounddevice as sd
import httpx
import pygame
from google import genai
from google.genai import types

from nemo_asr import NeMoStreamingASR
from asr_worker import AsrWorker
from playback_worker import PlaybackWorker
from pipeline_metrics import MetricsLogger
from vad_endpointer import SileroVADEndpointer, VADEvent

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configuration
MODEL_PATH = r"models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
KB_PATH = "knowledge_base.json"

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.08  # 80ms chunk size for 2x faster audio processing
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_TIMEOUT = 0.30  # 300ms turn completion for snappy response
ENERGY_THRESHOLD = 0.015
BARGE_IN_THRESHOLD = 0.025  # Elevated threshold during agent speech to protect against acoustic echo
BARGE_IN_CONSECUTIVE_FRAMES = 2  # Require 2 consecutive frames (160ms) above threshold to trigger barge-in

# NOTE on Acoustic Echo:
# Without hardware or OS acoustic echo cancellation (AEC), the microphone will pick up
# audio from open speakers during agent playback. BARGE_IN_THRESHOLD and multi-frame
# confirmation prevent low-to-medium echo from falsely triggering barge-in.
# For local testing without headphones, keep speaker volume moderate. Headphone usage
# is recommended until full AEC / Silero VAD (Step 5) is connected.

DEFAULT_MODE = "gemini-live"  # "gemini-live" (Native Multimodal Audio) or "sarvam"
DEFAULT_GEMINI_VOICE = "Puck"  # Puck, Aoede, Kore, Fenrir, Charon
DEFAULT_SARVAM_SPEAKER = "rahul"

def load_system_instruction(kb_path: str = KB_PATH) -> str:
    """Dynamically loads and builds system prompt from knowledge_base.json."""
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                kb = json.load(f)
            
            biz = kb.get("business", {})
            loc = biz.get("location", {})
            products = kb.get("products_and_pricing", [])
            offers = kb.get("special_offers", {})
            faqs = kb.get("common_faqs", {})
            assistant = kb.get("sales_assistant", {})

            prod_text = "\n".join([f"   - {p.get('category')}: Starts at {p.get('starting_price')}. Features: {p.get('features')} (Warranty: {p.get('warranty')})" for p in products])
            offer_text = "\n".join([f"   - {k.replace('_', ' ').title()}: {v}" for k, v in offers.items()])
            faq_text = "\n".join([f"   - {k.replace('_', ' ').title()}: {v}" for k, v in faqs.items()])

            prompt = f"""
You are '{assistant.get('name', 'Rahul')}' (or 'Priya' if female voice), the sales assistant at '{biz.get('name', 'Royal Furniture')}'.
{assistant.get('persona', 'Energetic and polite Indian sales executive.')}

Business & Location:
- Address: {loc.get('address', 'Station Road')}, Landmark: {loc.get('landmark', 'Opp. Platform 1')}
- Timings: {loc.get('timings', '10 AM to 9:30 PM')}
- Parking: {loc.get('parking', 'Available')}

Products & Pricing:
{prod_text}

Special Offers & Services:
{offer_text}

Common Questions & Actions:
{faq_text}

Conversational Rules:
1. Language: Natural, everyday conversational Hinglish (Devanagari/English blend with friendly warmth like 'अरे बिल्कुल सर!', 'हाँ जी मैम!').
2. Tone: Enthusiastic, helpful, respectful, and encouraging caller to visit the showroom.
3. Turn Length: STRICTLY 1 TO 2 SENTENCES MAXIMUM per turn. Keep it fast and natural for telephone calls.
"""
            return prompt.strip()
        except Exception as e:
            print(f"[Warning] Failed to parse {kb_path}: {e}")

    return "You are an energetic Indian phone calling sales assistant. Speak in natural conversational Hinglish. Keep responses to 1-2 short sentences."

SYSTEM_INSTRUCTION = load_system_instruction()

class CallingAgent:
    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        gemini_voice: str = DEFAULT_GEMINI_VOICE,
        sarvam_speaker: str = DEFAULT_SARVAM_SPEAKER,
        sarvam_pace: float = 1.20,
        kb_path: str = KB_PATH
    ):
        self.mode = mode.lower()
        self.gemini_voice = gemini_voice
        self.sarvam_speaker = sarvam_speaker
        self.sarvam_pace = sarvam_pace
        self.kb_path = kb_path
        self.system_instruction = load_system_instruction(self.kb_path)
        
        # Load raw KB metadata for greetings
        self.biz_name = "our store"
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    _kb = json.load(f)
                    self.biz_name = _kb.get("business", {}).get("name", "our store")
            except Exception:
                pass

        self.audio_q = queue.Queue()
        self.is_running = False
        self.is_speaking = False
        self.barge_in_active = False
        self._turn_interrupted = False
        self.playback: Optional[PlaybackWorker] = None
        self._agent_task: Optional[asyncio.Task] = None
        self.client = None
        self.chat = None
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # 1. Load Gemini API Key
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.api_key and os.path.exists("gemini_api_key.txt"):
            try:
                with open("gemini_api_key.txt", "r", encoding="utf-8") as f:
                    self.api_key = f.read().strip()
            except Exception:
                pass

        if not self.api_key:
            print("\n" + "="*55)
            print("  GEMINI API KEY NOT FOUND")
            print("  Please paste your Gemini API Key below:")
            print("="*55)
            try:
                entered = input("Enter Gemini API Key: ").strip()
                if entered:
                    self.api_key = entered
                    with open("gemini_api_key.txt", "w", encoding="utf-8") as f:
                        f.write(entered)
                    print("[Agent] API key saved to gemini_api_key.txt!\n")
            except (KeyboardInterrupt, EOFError):
                pass

        # 2. Load Sarvam Key if in sarvam mode
        if self.mode == "sarvam":
            self.sarvam_api_key = os.environ.get("SARVAM_API_KEY", "").strip()
            if not self.sarvam_api_key and os.path.exists("sarvam_api_key.txt"):
                try:
                    with open("sarvam_api_key.txt", "r", encoding="utf-8") as f:
                        self.sarvam_api_key = f.read().strip()
                except Exception:
                    pass

        # 3. Load NeMo ASR
        print("[Agent] Loading NeMo-Speech ASR model...")
        self.asr = NeMoStreamingASR(model_path=MODEL_PATH, gpu=-1)
        print("[Agent] ASR Model loaded successfully!")

        # Decode runs on a dedicated thread (step 3). Constructed in run(),
        # where there is a running event loop for it to resolve futures against.
        self.asr_worker = None
        self._worker_summary_printed = False

        # 4. Initialize Gemini Client
        self.client = genai.Client(api_key=self.api_key)
        
        if self.mode == "sarvam":
            self._init_gemini_text_chat()

        pygame.mixer.init(frequency=24000)

        # Latency instrumentation (Phase 9). Disable with NEMO_AGENT_METRICS=0.
        # Records marks only -- turn-taking behaviour is unchanged.
        self.call_id = uuid.uuid4().hex[:8]
        self.metrics = MetricsLogger(
            call_id=self.call_id,
            silence_window_ms=SILENCE_TIMEOUT * 1000.0,
        )

        # 5. Silero VAD Endpointer with energy fallback (Step 5)
        self.vad = SileroVADEndpointer(
            sample_rate=SAMPLE_RATE,
            speech_threshold=0.50,
            barge_in_speech_threshold=0.65,
            silence_threshold=0.35,
            min_speech_duration_ms=96.0,
            barge_in_min_speech_duration_ms=160.0,
            min_silence_duration_ms=SILENCE_TIMEOUT * 1000.0,
            speech_pad_ms=64.0,
            energy_fallback_threshold=ENERGY_THRESHOLD,
            energy_fallback_barge_in_threshold=BARGE_IN_THRESHOLD,
        )

    def _init_gemini_text_chat(self):
        """Initializes standard Gemini text chat for Sarvam TTS mode."""
        candidate_models = ["gemini-3.7-flash", "gemini-2.5-flash", "gemini-flash-latest"]
        for model_name in candidate_models:
            try:
                self.chat = self.client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION + "\nAlways format response in Devanagari script for TTS.",
                        temperature=0.7,
                    )
                )
                print(f"[Agent] Connected to Gemini Text ({model_name})!")
                break
            except Exception:
                pass

    def _audio_callback(self, indata, frames, time_info, status):
        """SoundDevice capture callback (runs in audio thread). Unconditional capture."""
        if indata.ndim > 1 and indata.shape[1] > 1:
            samples = np.mean(indata, axis=1, dtype=np.float32)
        else:
            samples = indata.copy().flatten().astype(np.float32)
        self.audio_q.put(samples)

    def _asr_worker_summary(self) -> str:
        """Decode-thread health, printed on exit.

        These are the 'audio queue depth, max' and 'dropped chunks' rows of the
        benchmark table in the plan. A non-zero drop count means decode could
        not keep up with capture -- the number to watch when judging whether
        step 3 actually bought headroom.
        """
        if self.asr_worker is None:
            return "[asr] worker never started."
        if self._worker_summary_printed:
            return ""   # both the normal exit path and the Ctrl+C handler call this
        self._worker_summary_printed = True
        s = self.asr_worker.stats()
        lines = [
            "",
            "=== ASR decode thread ===",
            f"chunks submitted {s['submitted_chunks']}, decoded {s['decoded_chunks']}, "
            f"dropped {s['dropped_chunks']}",
            f"peak backlog {s['max_backlog_seconds']}s of {s['max_backlog_bound_seconds']}s "
            f"bound ({s['max_queue_depth']} chunks)",
            f"decode thread busy {s['decode_seconds']}s",
        ]
        if s["dropped_chunks"]:
            lines.append(
                f"WARNING: {s['dropped_chunks']} audio chunk(s) dropped -- decode is "
                "behind capture; transcripts will have gaps."
            )
        elif s["max_backlog_seconds"] >= 0.5 * s["max_backlog_bound_seconds"]:
            # No drops yet, but the margin is thin enough to be worth saying.
            lines.append(
                "NOTE: peak backlog used over half the bound -- little headroom "
                "before chunks start dropping."
            )
        if s["queued_items_at_exit"]:
            lines.append(
                f"NOTE: {s['queued_items_at_exit']} chunk(s) "
                f"({s['queued_seconds_at_exit']}s) still queued at exit."
            )
        if s["dropped_results"]:
            lines.append(f"NOTE: {s['dropped_results']} stale interim result(s) discarded.")
        if s["error_count"]:
            lines.append(f"errors {s['error_count']}, last: {s['last_error']}")
        return "\n".join(lines)

    async def play_audio_file(self, file_path: str):
        """Plays an audio file via pygame mixer."""
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self.is_running:
                await asyncio.sleep(0.01)
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"[Audio Playback Error] {e}")
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    async def _drain_trailing_live_frames(self, live_session, timeout: float = 0.2):
        """Drains any trailing server packets after barge-in/activity_end to ensure Turn N+1 receives a clean WebSocket."""
        recv_iter = live_session.receive().__aiter__()
        while True:
            try:
                await asyncio.wait_for(recv_iter.__anext__(), timeout=timeout)
            except (asyncio.TimeoutError, TimeoutError, StopAsyncIteration):
                break
            except Exception:
                break

    async def generate_gemini_native_live_audio(self, live_session, user_transcript: str, turn=None):
        """Streams real-time native speech chunks to PlaybackWorker with non-blocking queueing.

        ``turn`` is an optional TurnMetrics record; when supplied, the request
        and first-audio marks are recorded on it.
        """
        print(f"\n[User]: {user_transcript}")
        print(f"[Agent ({self.gemini_voice})]: ", end="", flush=True)
        self.is_speaking = True
        self._turn_interrupted = False

        interrupted = False
        try:
            await live_session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=user_transcript)]),
                turn_complete=True,
            )
            if turn is not None:
                turn.mark("llm_sent")

            recv_iter = live_session.receive().__aiter__()
            while self.is_running:
                if self._turn_interrupted and not interrupted:
                    interrupted = True

                try:
                    if interrupted:
                        # Once interrupted, wait at most 500ms to drain remaining turn frames (turn_complete)
                        response = await asyncio.wait_for(recv_iter.__anext__(), timeout=0.5)
                    else:
                        response = await recv_iter.__anext__()
                except StopAsyncIteration:
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    print("\n[Agent]: Turn drain timeout (500ms) reached after interruption.")
                    break

                server_content = response.server_content
                if server_content is not None:
                    if server_content.interrupted:
                        print("\n[Agent]: Gemini server acknowledged interruption.")
                        if self.playback is not None:
                            self.playback.interrupt()
                        interrupted = True
                        continue

                    model_turn = server_content.model_turn
                    if model_turn is not None and not interrupted:
                        for part in model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                if turn is not None:
                                    turn.mark("first_audio")
                                    turn.audio_bytes += len(part.inline_data.data)
                                # Non-blocking enqueue to background playback worker
                                if self.playback is not None:
                                    self.playback.enqueue(part.inline_data.data)
                                if turn is not None:
                                    turn.mark("playback_started")

                    if server_content.turn_complete:
                        if turn is not None and not interrupted:
                            turn.mark("turn_complete")
                        break

        except asyncio.CancelledError:
            if self.playback is not None:
                self.playback.interrupt()
            raise
        except Exception as e:
            print(f"\n[Gemini Live Audio Error]: {e}")
            if turn is not None:
                turn.error = f"{type(e).__name__}: {e}"
        finally:
            self.is_speaking = False

    async def _run_agent_turn(self, live_session, user_transcript: str, turn=None):
        """Wrapper to run speech generation as an asyncio task and record metrics cleanly."""
        try:
            await self.generate_gemini_native_live_audio(live_session, user_transcript, turn=turn)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"\n[Agent Turn Error]: {exc}")
        finally:
            self.is_speaking = False
            if turn is not None:
                self.metrics.finish_turn(turn)

    async def run(self, device: int = None):
        """Main real-time streaming calling loop with persistent Live session."""
        self.is_running = True
        
        dev_info = sd.query_devices(device, "input")
        in_channels = min(2, max(1, dev_info.get("max_input_channels", 1)))

        in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=in_channels,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=device,
            callback=self._audio_callback
        )

        out_stream = sd.RawOutputStream(
            samplerate=24000,
            channels=1,
            dtype='int16'
        )
        self.playback = PlaybackWorker(out_stream=out_stream)

        print("\n" + "="*60)
        print("  NeMo-Speech + Gemini Live Multimodal AI Calling Agent")
        print(f"  Mode: {self.mode.upper()} | Voice: {self.gemini_voice}")
        print(f"  Audio Output: Direct Low-Latency 24kHz PCM Stream (Decoupled)")
        print(f"  Turn Timeout: {SILENCE_TIMEOUT}s | Press Ctrl+C to exit")
        print("="*60 + "\n")

        config = types.LiveConnectConfig(
            response_modalities=['AUDIO'],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.gemini_voice
                    )
                )
            ),
            system_instruction=self.system_instruction,
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
            )
        )

        try:
            with in_stream:
                print(f"[Agent] Connecting to Gemini Live Session ({self.biz_name})...")
                async with self.client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=config) as live_session:
                    print("[Agent] Connected! Starting voice session...")

                    # Initial snappy greeting as a background task so audio capture runs immediately
                    greeting_turn = self.metrics.start_turn(kind="greeting")
                    self._agent_task = asyncio.create_task(
                        self._run_agent_turn(
                            live_session,
                            f"Give an enthusiastic, 1-sentence opening greeting welcoming the caller to {self.biz_name} and asking what they are looking for today.",
                            turn=greeting_turn,
                        )
                    )

                    # ASR decode runs on its dedicated thread
                    self.asr_worker = AsrWorker(
                        self.asr,
                        sample_rate=SAMPLE_RATE,
                        language_code="hi",
                        interim_results=True,
                    )
                    self.asr_worker.start()
                    last_speech_time = None
                    last_interaction_time = time.time()
                    inactivity_count = 0
                    INACTIVITY_TIMEOUT = 4.5  # 4.5s snappy silence nudge trigger
                    has_spoken = False
                    current_transcript = ""
                    last_printed = ""
                    turn = None  # TurnMetrics for the caller turn in progress

                    print("\n[Agent] Listening for your voice...")
                    while self.is_running:
                        agent_active = (
                            self.is_speaking
                            or (self.playback is not None and self.playback.is_playing())
                            or (self._agent_task is not None and not self._agent_task.done())
                        )

                        chunk_batch = []
                        while not self.audio_q.empty():
                            try:
                                chunk_batch.append(self.audio_q.get_nowait())
                            except queue.Empty:
                                break

                        endpoint_detected = False
                        if chunk_batch:
                            audio_pieces_to_submit = []

                            for chunk in chunk_batch:
                                vad_event = self.vad.process_chunk(chunk, agent_active=agent_active)

                                if vad_event.barge_in:
                                    print("\n[Agent] Caller barge-in detected! Halting playback...")
                                    if self.playback is not None:
                                        self.playback.interrupt()
                                    self._turn_interrupted = True

                                    # Signal Gemini to complete current turn without cancelling receive loop
                                    try:
                                        await live_session.send_realtime_input(activity_start=types.ActivityStart())
                                        self.barge_in_active = True
                                    except Exception as exc:
                                        print(f"[Agent] Failed to send activity_start: {exc}")

                                    self.is_speaking = False
                                    agent_active = False

                                    last_speech_time = time.time()
                                    last_interaction_time = time.time()
                                    has_spoken = True
                                    if turn is None:
                                        turn = self.metrics.start_turn(kind="caller")
                                    turn.mark("speech_started")
                                    turn.mark("last_voice", overwrite=True)

                                elif vad_event.speech_start:
                                    last_speech_time = time.time()
                                    last_interaction_time = time.time()
                                    has_spoken = True
                                    if turn is None:
                                        turn = self.metrics.start_turn(kind="caller")
                                    turn.mark("speech_started")
                                    turn.mark("last_voice", overwrite=True)

                                elif vad_event.is_speech:
                                    last_speech_time = time.time()
                                    last_interaction_time = time.time()
                                    has_spoken = True
                                    if turn is not None:
                                        turn.mark("last_voice", overwrite=True)

                                if vad_event.audio_for_asr is not None:
                                    audio_pieces_to_submit.append(vad_event.audio_for_asr)

                                if vad_event.endpoint_detected:
                                    endpoint_detected = True

                            # Submit sliced audio (with pre-speech onset padding) to ASR worker
                            if audio_pieces_to_submit:
                                audio_to_decode = np.concatenate(audio_pieces_to_submit)
                                self.asr_worker.submit_audio(audio_to_decode)

                            for is_final, transcript in self.asr_worker.drain_results():
                                if transcript:
                                    current_transcript = transcript
                                    if transcript != last_printed:
                                        print(f"\r[Transcribing]: {transcript}", end="", flush=True)
                                        last_printed = transcript
                                        last_interaction_time = time.time()
                                        if turn is None:
                                            turn = self.metrics.start_turn(kind="caller")
                                        turn.mark("first_partial")
                                        turn.partial_count += 1

                        now = time.time()

                        # 1. Caller finished speaking -> Generate response
                        if has_spoken and (endpoint_detected or (last_speech_time and (now - last_speech_time > SILENCE_TIMEOUT))):
                            interim_text = current_transcript.strip()
                            final_text = await self.asr_worker.flush_and_restart()
                            user_text = (final_text or interim_text).strip()

                            if user_text:
                                if turn is not None:
                                    turn.mark("turn_fired")
                                    turn.mark("asr_final")
                                    turn.interim_transcript = interim_text
                                    turn.final_transcript = final_text
                                    turn.used_final = bool(final_text)

                                print(f"\r[Transcribing]: {user_text} (Final)")

                                current_transcript = ""
                                last_printed = ""
                                has_spoken = False
                                last_speech_time = None
                                inactivity_count = 0
                                self.vad.reset()

                                # If caller barged in, close the activity period before sending new turn
                                if self.barge_in_active:
                                    try:
                                        await live_session.send_realtime_input(activity_end=types.ActivityEnd())
                                        # Drain any trailing server frames from the interrupted turn before Turn N+1 begins
                                        await self._drain_trailing_live_frames(live_session, timeout=0.2)
                                    except Exception as exc:
                                        print(f"[Agent] Failed to send activity_end: {exc}")
                                    self.barge_in_active = False

                                # Ensure previous agent turn task has finished draining before starting next turn
                                if self._agent_task is not None and not self._agent_task.done():
                                    try:
                                        await asyncio.wait_for(asyncio.shield(self._agent_task), timeout=0.5)
                                    except (asyncio.TimeoutError, TimeoutError, Exception):
                                        self._agent_task.cancel()
                                        try:
                                            await self._agent_task
                                        except (asyncio.CancelledError, Exception):
                                            pass

                                current_turn = turn
                                turn = None
                                self._agent_task = asyncio.create_task(
                                    self._run_agent_turn(live_session, user_text, turn=current_turn)
                                )
                                last_interaction_time = time.time()
                                print("\n[Agent] Listening for your voice...")
                            else:
                                has_spoken = False
                                turn = None
                                self.vad.reset()


                        # 2. Caller has been completely silent -> Proactive silence nudge
                        elif not has_spoken and not agent_active and (now - last_interaction_time > INACTIVITY_TIMEOUT):
                            if inactivity_count == 0:
                                print("\n[Agent] Caller silent for 4.5s. Sending check-in nudge...")
                                nudge_prompt = "The caller has been silent for a few seconds. Give a quick, natural, polite 1-sentence check in Hinglish asking if they can hear you (e.g. 'हेलो सर, क्या आपको मेरी आवाज़ आ रही है?')."
                                nudge_turn = self.metrics.start_turn(kind="nudge")
                                if self._agent_task is not None and not self._agent_task.done():
                                    try:
                                        await asyncio.wait_for(asyncio.shield(self._agent_task), timeout=0.5)
                                    except (asyncio.TimeoutError, TimeoutError, Exception):
                                        self._agent_task.cancel()
                                        try:
                                            await self._agent_task
                                        except (asyncio.CancelledError, Exception):
                                            pass
                                self._agent_task = asyncio.create_task(
                                    self._run_agent_turn(live_session, nudge_prompt, turn=nudge_turn)
                                )
                                inactivity_count += 1
                                last_interaction_time = time.time()
                                print("\n[Agent] Listening for your voice...")
                            elif inactivity_count == 1:
                                print("\n[Agent] Caller still silent. Offering WhatsApp catalog...")
                                nudge_prompt = "The caller is still silent. Give a polite 1-sentence offer in Hinglish asking if you should send the catalog and shop location to their WhatsApp."
                                nudge_turn = self.metrics.start_turn(kind="nudge")
                                if self._agent_task is not None and not self._agent_task.done():
                                    try:
                                        await asyncio.wait_for(asyncio.shield(self._agent_task), timeout=0.5)
                                    except (asyncio.TimeoutError, TimeoutError, Exception):
                                        self._agent_task.cancel()
                                        try:
                                            await self._agent_task
                                        except (asyncio.CancelledError, Exception):
                                            pass
                                self._agent_task = asyncio.create_task(
                                    self._run_agent_turn(live_session, nudge_prompt, turn=nudge_turn)
                                )
                                inactivity_count += 1
                                last_interaction_time = time.time()
                                print("\n[Agent] Listening for your voice...")

                        await asyncio.sleep(0.02)
        finally:
            if self._agent_task is not None and not self._agent_task.done():
                self._agent_task.cancel()
                try:
                    await self._agent_task
                except (asyncio.CancelledError, Exception):
                    pass

            if self.playback is not None:
                self.playback.stop()

            if self.asr_worker is not None:
                self.asr_worker.stop()
                summary = self._asr_worker_summary()
                if summary:
                    print(summary)
            if self.vad is not None:
                self.vad.reset()
            self.asr.close()
            await self.http_client.aclose()
            self.metrics.close()
            print("\n[Agent] Stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live NeMo-Speech + Gemini Multimodal Calling Agent")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["gemini-live", "sarvam"], help="Calling engine mode: 'gemini-live' (native audio) or 'sarvam'")
    parser.add_argument("--voice", type=str, default=DEFAULT_GEMINI_VOICE, choices=["Puck", "Aoede", "Kore", "Fenrir", "Charon"], help="Gemini native voice: Puck, Aoede, Kore, Fenrir, Charon")
    parser.add_argument("--speaker", type=str, default=DEFAULT_SARVAM_SPEAKER, help="Sarvam speaker (if mode is sarvam)")
    parser.add_argument("--kb", type=str, default=KB_PATH, help="Path to JSON knowledge base file (e.g. knowledge_bases/sk_furniture.json)")
    args = parser.parse_args()

    agent = CallingAgent(
        mode=args.mode,
        gemini_voice=args.voice,
        sarvam_speaker=args.speaker,
        kb_path=args.kb
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
