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
from pipeline_metrics import MetricsLogger

# Configuration
MODEL_PATH = r"models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
KB_PATH = "knowledge_base.json"

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.08  # 80ms chunk size for 2x faster audio processing
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_TIMEOUT = 0.30  # 300ms turn completion for snappy response
ENERGY_THRESHOLD = 0.015

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
        """SoundDevice capture callback (runs in audio thread)."""
        if not self.is_speaking:
            if indata.ndim > 1 and indata.shape[1] > 1:
                samples = np.mean(indata, axis=1, dtype=np.float32)
            else:
                samples = indata.copy().flatten().astype(np.float32)
            self.audio_q.put(samples)

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

    async def generate_gemini_native_live_audio(self, live_session, out_stream, user_transcript: str, turn=None):
        """Streams real-time native speech chunks directly to speakers with zero disk buffering.

        ``turn`` is an optional TurnMetrics record; when supplied, the request
        and first-audio marks are recorded on it. Instrumentation only -- it
        does not affect what is sent or played.
        """
        print(f"\n[User]: {user_transcript}")
        print(f"[Agent ({self.gemini_voice})]: ", end="", flush=True)
        self.is_speaking = True

        try:
            await live_session.send(input=user_transcript, end_of_turn=True)
            if turn is not None:
                turn.mark("llm_sent")

            async for response in live_session.receive():
                if not self.is_running:
                    break
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                # first_audio is marked before the write so it
                                # measures arrival, not device backpressure.
                                if turn is not None:
                                    turn.mark("first_audio")
                                    turn.audio_bytes += len(part.inline_data.data)
                                # Play raw 24kHz PCM chunk immediately through sounddevice
                                out_stream.write(part.inline_data.data)
                                if turn is not None:
                                    # First write only; blocking cost of later
                                    # writes is agent speech time, not latency.
                                    turn.mark("playback_started")
                    if server_content.turn_complete:
                        if turn is not None:
                            turn.mark("turn_complete")
                        break

        except Exception as e:
            print(f"\n[Gemini Live Audio Error]: {e}")
            if turn is not None:
                turn.error = f"{type(e).__name__}: {e}"
        finally:
            self.is_speaking = False
            # Clear any audio queued while the agent was speaking
            while not self.audio_q.empty():
                try:
                    self.audio_q.get_nowait()
                except queue.Empty:
                    break

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
        out_stream.start()

        print("\n" + "="*60)
        print("  NeMo-Speech + Gemini Live Multimodal AI Calling Agent")
        print(f"  Mode: {self.mode.upper()} | Voice: {self.gemini_voice}")
        print(f"  Audio Output: Direct Low-Latency 24kHz PCM Stream")
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
            system_instruction=self.system_instruction
        )

        print(f"[Agent] Connecting to Gemini Live Session ({self.biz_name})...")
        async with self.client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=config) as live_session:
            print("[Agent] Connected! Starting voice session...")

            # Initial snappy greeting
            greeting_turn = self.metrics.start_turn(kind="greeting")
            await self.generate_gemini_native_live_audio(
                live_session,
                out_stream,
                f"Give an enthusiastic, 1-sentence opening greeting welcoming the caller to {self.biz_name} and asking what they are looking for today.",
                turn=greeting_turn,
            )
            self.metrics.finish_turn(greeting_turn)

            self.asr.start_stream(interim_results=True, language_code="hi")
            last_speech_time = None
            last_interaction_time = time.time()
            inactivity_count = 0
            INACTIVITY_TIMEOUT = 4.5  # 4.5s snappy silence nudge trigger
            has_spoken = False
            current_transcript = ""
            last_printed = ""
            turn = None  # TurnMetrics for the caller turn in progress

            with in_stream:
                print("\n[Agent] Listening for your voice...")
                while self.is_running:
                    chunk_batch = []
                    while not self.audio_q.empty():
                        try:
                            chunk_batch.append(self.audio_q.get_nowait())
                        except queue.Empty:
                            break

                    if chunk_batch and not self.is_speaking:
                        audio_chunk = np.concatenate(chunk_batch)
                        rms = np.sqrt(np.mean(audio_chunk**2))

                        if rms > ENERGY_THRESHOLD:
                            last_speech_time = time.time()
                            last_interaction_time = time.time()
                            has_spoken = True
                            # Energy-only voice marks. Kept separate from the
                            # partial-driven bumping below so the summary can
                            # show how much ASR lag inflates the silence window.
                            if turn is None:
                                turn = self.metrics.start_turn(kind="caller")
                            turn.mark("speech_started")
                            turn.mark("last_voice", overwrite=True)

                        self.asr.push_audio(audio_chunk, sample_rate=SAMPLE_RATE)

                        for is_final, transcript in self.asr.poll_results():
                            if transcript:
                                current_transcript = transcript
                                if transcript != last_printed:
                                    print(f"\r[Transcribing]: {transcript}", end="", flush=True)
                                    last_printed = transcript
                                    last_speech_time = time.time()
                                    last_interaction_time = time.time()
                                    has_spoken = True
                                    if turn is None:
                                        turn = self.metrics.start_turn(kind="caller")
                                    turn.mark("first_partial")
                                    turn.partial_count += 1

                    now = time.time()

                    # 1. Caller finished speaking -> Generate response
                    if has_spoken and last_speech_time and (now - last_speech_time > SILENCE_TIMEOUT):
                        if current_transcript.strip():
                            if turn is not None:
                                turn.mark("turn_fired")
                            interim_text = current_transcript.strip()

                            # Flush the ASR tail. This now actually executes --
                            # finish_stream() used to be a generator that was
                            # never iterated, so the C-level flush never ran and
                            # we lost punctuation, ITN and the audio tail on
                            # every single turn. Prefer the flushed final; fall
                            # back to the interim if the flush yields nothing.
                            final_text = ""
                            try:
                                final_text = self.asr.final_transcript()
                            except Exception as exc:
                                print(f"\n[ASR flush failed, using interim]: {exc}")

                            user_text = final_text or interim_text
                            print(f"\r[Transcribing]: {user_text} (Final)")

                            if turn is not None:
                                turn.mark("asr_final")
                                turn.interim_transcript = interim_text
                                turn.final_transcript = final_text
                                turn.used_final = bool(final_text)

                            self.asr.start_stream(interim_results=True, language_code="hi")

                            current_transcript = ""
                            last_printed = ""
                            has_spoken = False
                            last_speech_time = None
                            inactivity_count = 0

                            await self.generate_gemini_native_live_audio(live_session, out_stream, user_text, turn=turn)
                            self.metrics.finish_turn(turn)
                            turn = None
                            last_interaction_time = time.time()
                            print("\n[Agent] Listening for your voice...")
                        else:
                            has_spoken = False
                            turn = None  # no speech content -- discard the record

                    # 2. Caller has been completely silent -> Proactive silence nudge
                    elif not has_spoken and not self.is_speaking and (now - last_interaction_time > INACTIVITY_TIMEOUT):
                        if inactivity_count == 0:
                            print("\n[Agent] Caller silent for 7s. Sending check-in nudge...")
                            nudge_prompt = "The caller has been silent for a few seconds. Give a quick, natural, polite 1-sentence check in Hinglish asking if they can hear you (e.g. 'हेलो सर, क्या आपको मेरी आवाज़ आ रही है?')."
                            nudge_turn = self.metrics.start_turn(kind="nudge")
                            await self.generate_gemini_native_live_audio(live_session, out_stream, nudge_prompt, turn=nudge_turn)
                            self.metrics.finish_turn(nudge_turn)
                            inactivity_count += 1
                            last_interaction_time = time.time()
                            print("\n[Agent] Listening for your voice...")
                        elif inactivity_count == 1:
                            print("\n[Agent] Caller still silent. Offering WhatsApp catalog...")
                            nudge_prompt = "The caller is still silent. Give a polite 1-sentence offer in Hinglish asking if you should send the catalog and shop location to their WhatsApp."
                            nudge_turn = self.metrics.start_turn(kind="nudge")
                            await self.generate_gemini_native_live_audio(live_session, out_stream, nudge_prompt, turn=nudge_turn)
                            self.metrics.finish_turn(nudge_turn)
                            inactivity_count += 1
                            last_interaction_time = time.time()
                            print("\n[Agent] Listening for your voice...")

                    await asyncio.sleep(0.02)

        out_stream.stop()
        out_stream.close()
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
        # Ctrl+C skips run()'s cleanup, and Ctrl+C is how most test calls end,
        # so print the latency summary here too. close() is idempotent.
        agent.metrics.close()
