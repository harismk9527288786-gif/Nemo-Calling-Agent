# 🎙️ PROJECT ARCHITECTURE: NeMo-Speech.cpp AI Calling System

> **Document Type**: Production System Architecture & Technical Specification  
> **Source of Truth**: Repository Codebase (`C:\AI\NeMo-Speech.cpp`)  
> **Primary Script**: `agent.py` | **ASR Binding**: `nemo_asr.py` | **Core Runtime**: `build-cpu/bin/nemo_speech_asr_c.dll`  
> **Hardware Profile**: Intel Core i7-8550U (4 Cores / 8 Threads @ 1.80GHz–4.0GHz), 8GB RAM, Windows 11 64-bit (100% CPU Inference)

---

## 1. 🏗️ OVERALL SYSTEM ARCHITECTURE

The system is a **hybrid edge-cloud, real-time autonomous AI phone calling agent** designed for high-concurrency, low-cost Indian business telephony (e.g., retail showrooms, clinics, local services). 

It combines **100% local, offline CPU speech recognition (ASR)** with **cloud-based persistent multimodal reasoning and native speech synthesis (LLM & TTS)** to achieve sub-400ms conversational turnaround with zero GPU costs.

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│                                    OVERALL SYSTEM ARCHITECTURE                                    │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                   │
│  [Caller / Mobile Phone]                                                                          │
│          │ (4G VoLTE Cellular Network - Jio SIM)                                                  │
│          ▼                                                                                        │
│  [1-Port 4G VoLTE VoIP Gateway] (Dinstar UC2000-VA-1G / Matrix SIMADO GFX11)                      │
│          │ (SIP Signaling: Port 5060 | RTP Audio Stream: UDP 10000-20000 / LAN RJ-45)            │
│          ▼                                                                                        │
│  [Audio Ingestion & Pre-processing] (`sounddevice.InputStream` / PortAudio - Device Index 1)      │
│          │ (Resampled to 16,000 Hz, 1-Ch Mono, float32 | 80ms Chunks / 1280 samples)              │
│          ▼                                                                                        │
│  [Local Streaming ASR Engine] (`nemo_asr.py` ──> `nemo_speech_asr_c.dll` C-ABI)                   │
│          │ (Model: `models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` | <100ms Latency on CPU)   │
│          ▼ (Live Hinglish Transcript | VAD Energy > 0.015 | Turn Silence: 300ms)                  │
│  [Calling Agent Orchestrator] (`agent.py` + `knowledge_base.json`)                                │
│          │ (Devanagari Script Hinglish | Strict 1-2 Sentences | 4.5s Inactivity Watchdog)         │
│          ▼ (Persistent WebSocket Session over Port 443 WSS)                                       │
│  [Cloud Multimodal Brain & Voice] (Google Gemini Live: `gemini-2.5-flash-native-audio-latest`)     │
│          │ (Direct 24,000 Hz HD PCM Chunks - Zero Disk I/O)                                       │
│          ▼                                                                                        │
│  [Audio Post-Processing & Playback] (`sounddevice.RawOutputStream` 24kHz int16)                   │
│          │ (Immediate Hardware Buffer DMA / RTP Return Stream)                                    │
│          ▼                                                                                        │
│  [1-Port 4G VoIP Gateway ──> 4G VoLTE ──> Caller Mobile] (<400ms Total Turnaround)                │
│                                                                                                   │
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. 🔄 COMPLETE END-TO-END CALL FLOW

```
 Caller
   │  (4G VoLTE Radio Frequency)
   ▼
 1-Port VoIP Gateway (Dinstar UC2000 / Matrix SIMADO)
   │  (SIP:5060 + RTP Audio Stream over LAN)
   ▼
 Audio Transport (`sounddevice.InputStream` / PortAudio)
   │  (Native rate resampled to 16 kHz Mono Float32)
   ▼
 Audio Pre-Processing (80ms chunking, RMS energy > 0.015, 300ms silence detection)
   │  (`audio_q` memory queue)
   ▼
 ASR Engine (`nemo_asr.py` + `nemo_speech_asr_c.dll` + `nemotron-3.5-0.6b.q8_0.gguf`)
   │  (Interim and final Devanagari/Hinglish transcripts)
   ▼
 Language & Hinglish Processing (`agent.py` + `knowledge_base.json`)
   │  (Dynamic catalog injection, Devanagari script formatting, 1-2 sentence constraint)
   ▼
 LLM / AI Agent (Google Gemini Live WebSocket Session over Port 443 WSS)
   │  (Real-time multimodal speech-to-speech reasoning)
   ▼
 TTS Engine (Gemini Live Native Audio: `Puck`/`Kore` / Alt: Sarvam AI `Bulbul`)
   │  (Raw 24kHz PCM chunks)
   ▼
 Audio Post-Processing (`sounddevice.RawOutputStream` 24kHz int16)
   │  (Direct memory-to-hardware DMA buffer write, zero disk I/O)
   ▼
 VoIP Gateway (RTP Audio packetization)
   │  (4G VoLTE Cellular Transmission)
   ▼
 Caller (Instant voice stream)
```

---

## 3. 📟 VOIP GATEWAY AND AUDIO FLOW

### 3.1 Hardware & Physical Connectivity
* **Hardware Unit**: 1-Port 4G VoLTE VoIP Gateway (e.g., *Dinstar UC2000-VA-1G 4G* or *Matrix SIMADO GFX11 4G*).
* **SIM Card**: Physical business Jio 4G SIM inserted into the gateway SIM slot (flat ₹299/month unlimited calling plan).
* **Physical Interconnect**: RJ-45 Ethernet cable connecting the VoIP Gateway to the local Wi-Fi router / Host PC LAN.
* **Development / Testing Audio Input**: Windows microphone capture via PortAudio device index 1 (`--device 1`).

### 3.2 Inbound Audio Flow
1. Customer calls the shop's 10-digit mobile number.
2. The 4G gateway answers or forwards the call, converting cellular VoLTE voice into standard SIP/RTP packets.
3. Audio stream is captured via `sounddevice.InputStream` at 16,000 Hz float32 mono in 80ms chunks (`CHUNK_SAMPLES = 1280`).
4. Chunks are pushed into an in-memory queue (`self.audio_q = queue.Queue()`).

### 3.3 Outbound Audio Flow
1. As Gemini emits raw 24kHz PCM audio frames over WebSockets, chunks are written directly to `sounddevice.RawOutputStream`.
2. Outbound audio feeds the RTP transmitter back to the VoIP Gateway.
3. The Gateway modulates the RTP audio into 4G VoLTE cellular radio signals received by the caller's mobile handset.
4. During agent speech, `self.is_speaking = True` blocks echo pickup, and `audio_q` is flushed upon turn completion.

---

## 4. 🎛️ AUDIO FORMATS AND SAMPLE RATES

| Pipeline Stage | Format / Encoding | Sample Rate | Channels | Bit Depth / Dtype | Frame / Chunk Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Inbound Capture** | Linear PCM | 16,000 Hz (16 kHz) | 1 (Mono) | `np.float32` | 80 ms (1,280 samples) |
| **ASR Ingestion** | Linear PCM | 16,000 Hz (16 kHz) | 1 (Mono) | `ctypes.c_float` | 80 ms – 160 ms blocks |
| **Gemini Live Output** | Raw Headerless PCM | 24,000 Hz (24 kHz) | 1 (Mono) | `int16` (Little Endian) | Variable chunk stream |
| **Hardware Playback** | `RawOutputStream` | 24,000 Hz (24 kHz) | 1 (Mono) | `int16` | Direct DMA buffer |
| **Sarvam AI (Alt Mode)**| WAV / Linear PCM | 24,000 Hz (24 kHz) | 1 (Mono) | `int16` | Sentence-pipelined chunks |
| **Standalone C++ TTS** | Nano-Codec Decoder | 22,050 Hz (22 kHz) | 1 (Mono) | `float32` / `int16` | 3 frames (codec queue: 4) |

### Key Audio Thresholds (`agent.py`)
* `SAMPLE_RATE = 16000`
* `CHUNK_DURATION = 0.08` (80ms)
* `CHUNK_SAMPLES = int(16000 * 0.08) = 1280`
* `ENERGY_THRESHOLD = 0.015` (RMS energy threshold for VAD voice activity)
* `SILENCE_TIMEOUT = 0.30` (300ms turn-completion silence detector)
* `INACTIVITY_TIMEOUT = 4.5` (4.5s silence trigger for proactive nudges)

---

## 5. 🧠 ASR PIPELINE AND NEMOTRON MODEL

### 5.1 Model Specifications
* **Model File**: `models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` (707 MB / 742 MB).
* **Architecture**: Quantized (Q8_0) Cache-Aware FastConformer-RNNT (`EncDecRNNTBPEModelWithPrompt`).
* **Features**: Language-ID prompt conditioning across 40+ languages, cache-aware streaming states, sub-100ms chunk latency.
* **Target Execution**: 100% Local CPU Inference (`gpu = -1`, Intel Core i7-8550U, 0% GPU required).

### 5.2 Implementation Architecture (`nemo_asr.py`)
Direct low-level C-ABI binding via Python `ctypes` communicating with `build-cpu\bin\nemo_speech_asr_c.dll`:

```python
# C-ABI Structures in nemo_asr.py:
BackendConfig      -> gpu: -1 (CPU Execution)
ModelConfig        -> path: models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
StreamingConfig    -> chunk_size: 0.16s, ctc_padding: 1.92s, rnnt_right_context: 1
RecognizerConfig   -> backend, model, streaming, decoder, vad, endpointing
RecognitionOptions -> interim_results: True, language_code: "hi"
```

### 5.3 Streaming ASR Execution Flow
1. `start_stream(interim_results=True, language_code="hi")` initializes an active recognizer stream.
2. `push_audio(samples_f32, sample_rate=16000)` streams float32 numpy buffers into `nemo_speech_asr_stream_push_f32`.
3. `poll_results()` yields `(is_final, transcript_text)` tuples in real-time.
4. `finish_stream()` closes the utterance window and resets the recognizer state for the next turn.

---

## 6. 🤖 LLM / AGENT PIPELINE

### 6.1 Core Modules & Files
* **Main Orchestrator**: `agent.py` (`CallingAgent` class).
* **Business Knowledge File**: `knowledge_base.json` (and `knowledge_bases/sk_furniture.json`).
* **SDK / API**: `google-genai` Python SDK (`from google import genai`, `from google.genai import types`).

### 6.2 Primary Engine: Gemini Live Multimodal WebSocket
* **Model ID**: `gemini-2.5-flash-native-audio-latest`
* **Session Type**: Single persistent bidirectional WebSocket connection (`client.aio.live.connect`).
* **Modality**: `response_modalities=['AUDIO']` with `speech_config.voice_config.prebuilt_voice_config`.
* **Zero Handshake Delay**: The WebSocket connection is established once at call start and preserved across all conversational turns.

### 6.3 Prompt Orchestration & G2P Solution
* **System Prompt Generator**: `load_system_instruction(kb_path)` dynamically converts `knowledge_base.json` into the agent's persona, store location, pricing catalog, discount rules, and FAQs.
* **Hinglish G2P Optimization**: LLM is instructed to speak in everyday conversational Hinglish formatted in **Devanagari script with English loanwords** (`"अरे बिल्कुल सर! स्टेशन से सिर्फ 2 मिनट की वॉकिंग डिस्टेंस है"`). This bypasses English phonetic mispronunciation.
* **Telephony Turn Limit**: Prompts enforce a strict **1 to 2 short sentences per turn** maximum.

### 6.4 Inactivity & Silence Nudge State Machine
To prevent dead air when callers are distracted or silent:
* **Timer Trigger**: `INACTIVITY_TIMEOUT = 4.5s` of continuous caller silence.
* **Nudge 1 (4.5s)**: Sends check-in prompt: *"हेलो सर, क्या आपको मेरी आवाज़ आ रही है?"*
* **Nudge 2 (Next 5.0s)**: Sends WhatsApp offer prompt: *"सर, अगर आप अभी बिजी हैं तो क्या मैं आपको सोफा और बेड का कैटलॉग वॉट्सऐप कर दूँ?"*

---

## 7. 🗣️ TTS (TEXT-TO-SPEECH) PIPELINE

The repository supports two TTS architectures:

### 7.1 Primary Champion: Gemini Live Native Audio (Multimodal S2S)
* **Mode**: `agent.py --mode gemini-live --voice <VoiceName>`
* **Supported Neural Voices**:
  - `Puck`: Energetic, friendly male (Default)
  - `Kore`: Empathetic, warm female
  - `Aoede`: Friendly conversational female
  - `Fenrir`: Deep, authoritative male
  - `Charon`: Calm, dignified male
* **Output Format**: Headerless 24,000 Hz, 1-Channel Mono, `int16` PCM.
* **Zero-Disk Streaming**: Raw PCM bytes from `part.inline_data.data` are written directly to `sounddevice.RawOutputStream` with 0ms disk buffering and sub-400ms turnaround.

### 7.2 Secondary / Alternative Mode: Sarvam AI Bulbul (Dedicated Indian TTS)
* **Mode**: `agent.py --mode sarvam --speaker rahul`
* **API Endpoint**: `https://api.sarvam.ai/text-to-speech` (Port 443 HTTPS REST)
* **Voice Speaker**: `rahul` (Pace: 1.20)
* **Asynchronous Pipelining**: Text chunks for sentence $N+1$ are downloaded asynchronously in the background while sentence $N$ plays through `pygame.mixer`.
* **Punctuation Smoothing**: Regex normalization (`हूँ!` $\rightarrow$ `हूँ,`) prevents trailing nasal exclamation pauses.

### 7.3 Standalone C++ Engine TTS (`nemo-speech serve`)
* **Magpie-TTS**: Multilingual 357M model (`magpie_tts_multilingual_357m.v2602.f16.gguf`).
* **Nano-Codec Decoder**: 22 kHz neural audio decoder (`nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf`).
* **Tokenizer**: Extracted SentencePiece tokenizer directory (`models/magpie-tts/extracted`).

---

## 8. 🌐 BACKEND / API SERVICES

The repository provides both Python application orchestration and compiled high-performance C++ server binaries:

```
C:\AI\NeMo-Speech.cpp\
│
├── agent.py                      # Main Python asyncio calling agent orchestrator
├── nemo_asr.py                   # Python C-ABI ctypes streaming bridge
│
├── build-cpu\bin\
│   ├── nemo-speech.exe           # Unified CLI & HTTP/WebSocket server
│   ├── riva_server.exe           # Riva-compatible gRPC speech server
│   ├── transcribe_live.exe       # C++ live microphone ASR binary
│   └── nemo_speech_asr_c.dll     # Compiled C-ABI shared library
│
└── server\http\
    ├── http_server.cpp           # C++ HTTP/1.1 & WebSocket server implementation
    └── http_server.h             # Server headers & routing tables
```

### 8.1 Standalone C++ HTTP / WebSocket Server (`nemo-speech serve`)
* **Command**: `nemo-speech serve --config config/server.example.yaml`
* **Port**: `8080` (Default host: `127.0.0.1`)
* **Endpoints**:
  - `GET /`: Embedded browser playground UI
  - `GET /health`, `GET /ready`: Liveness and readiness probes
  - `GET /version`: Build version
  - `GET /v1/models`: List loaded ASR, TTS, and NMT models
  - `POST /v1/audio/transcriptions`: OpenAI-compatible multipart audio transcription
  - `POST /v1/audio/speech`: OpenAI-compatible text-to-speech synthesis
  - `WebSocket /v1/realtime`: Real-time PCM16 streaming speech transcription

### 8.2 Standalone C++ Riva gRPC Server (`riva_server`)
* **Command**: `riva_server --asr.model.path models/asr.q8_0.gguf --bind 0.0.0.0:50051`
* **Port**: `50051` (gRPC)
* **Services**: NVIDIA Riva-compatible `ASR.StreamingRecognize`, `TTS.Synthesize`, `NMT.Translate`.

---

## 9. 💻 FRONTEND / DASHBOARD

* **Embedded Browser Playground**: Hosted by `nemo-speech serve` at `http://127.0.0.1:8080/`. Provides interactive microphone capture, file upload transcription, audio playback, and parameter tuning.
* **UI Development Workspace**: `llama.cpp/tools/ui/` contains a Svelte 5 / Vite / Storybook frontend submodule for building extended web client interfaces.

---

## 10. 🗄️ DATABASES AND STORAGE

The system uses a **zero-database, file-based decoupled architecture**:

| Storage Component | File / Path | Format | Purpose |
| :--- | :--- | :--- | :--- |
| **Business Catalog** | `knowledge_base.json` | JSON | Dynamic store catalog, pricing, location, discounts, FAQs |
| **Catalog Template** | `knowledge_bases/default_template.json` | JSON | Reusable business template for new retail/clinic clients |
| **Client Instance** | `knowledge_bases/sk_furniture.json` | JSON | SK Furniture Market (Thane West) production catalog |
| **ASR Model File** | `models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` | GGUF (Q8_0) | 707 MB local quantized speech recognition weights |
| **Gemini Credentials**| `gemini_api_key.txt` | Plaintext | Google AI Studio API authentication key |
| **Sarvam Credentials**| `sarvam_api_key.txt` | Plaintext | Sarvam AI API authentication key |

---

## 11. ☁️ EXTERNAL APIS AND SERVICES

| Service Name | Provider | Endpoint / Protocol | Authentication | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Google Gemini Live API** | Google AI Studio | `wss://generativelanguage.googleapis.com:443` | API Key via Header/Param | Real-time multimodal reasoning & 24kHz native voice generation |
| **Google Gemini REST API** | Google AI Studio | `https://generativelanguage.googleapis.com:443` | `x-goog-api-key` | Fallback text reasoning (`gemini-3.7-flash`, `gemini-2.5-flash`) |
| **Sarvam AI Audio API** | Sarvam AI | `https://api.sarvam.ai/text-to-speech` (Port 443) | `api-subscription-key` | Indian regional accent voice synthesis (Bulbul:v2/v3) |
| **Jio 4G VoLTE Cellular** | Reliance Jio | Cellular Radio Frequency / 4G SIM | Physical SIM Card | Voice call transport for inbound customer calls |

---

## 12. 🚢 DEPLOYMENT ARCHITECTURE

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│                                       DEPLOYMENT ARCHITECTURE                                     │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                   │
│  [ON-PREMISE / LOCAL EDGE]                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 💻 Host Machine (Intel Core i7-8550U 4C/8T @ 1.8GHz, 8GB RAM, Windows 11 64-bit)            │  │
│  │  ├─ Runtime: Python 3.10+ (`agent.py`, `nemo_asr.py`)                                       │  │
│  │  ├─ C-ABI Shared Libraries: `nemo_speech_asr_c.dll`, `ggml.dll`                             │  │
│  │  ├─ Local Weights: `models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` (707 MB)              │  │
│  │  └─ Audio Subsystem: PortAudio / SoundDevice (16kHz in, 24kHz out)                          │  │
│  └────────────────────────────────────────────────┬────────────────────────────────────────────┘  │
│                                                   │ RJ-45 Ethernet (LAN)                          │
│  ┌────────────────────────────────────────────────┴────────────────────────────────────────────┐  │
│  │ 📟 1-Port 4G VoLTE VoIP Gateway (Dinstar UC2000-VA-1G / Matrix SIMADO GFX11)                │  │
│  │  ├─ Physical Jio Business SIM Card (₹299/mo Unlimited Calling)                             │  │
│  │  └─ Cellular RF: Inbound / Outbound 4G VoLTE Mobile Calls                                  │  │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                   │ Port 443 WSS / HTTPS                          │
│  [CLOUD INFRASTRUCTURE]                           ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ ☁️ Google Cloud AI Studio (Gemini Multimodal Live Native Audio API)                          │  │
│  │  └─ Quota: 1,500 Requests/Day Free Tier (~560 requests used for 70 calls/day = 37% quota)   │  │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                   │
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 13. 🔌 PORTS AND PROTOCOLS

| Port Number | Transport Protocol | Application Protocol | Service / Component | Traffic Direction |
| :--- | :--- | :--- | :--- | :--- |
| **5060** | UDP / TCP | SIP | 4G VoIP Gateway ↔ Host PBX / Audio Stream | Inbound / Outbound (LAN) |
| **10000–20000** | UDP | RTP | Real-time audio media exchange with VoIP Gateway | Bidirectional (LAN) |
| **443** | TCP / TLS | WSS / HTTPS | Google Gemini Live WebSocket & Sarvam AI REST API | Outbound to Cloud |
| **8080** | TCP | HTTP / WebSocket | `nemo-speech serve` REST API & Browser Playground | Local Inbound / LAN |
| **50051** | TCP | gRPC / HTTP/2 | `riva_server` NVIDIA Riva-compatible API | Local Inbound / LAN |

---

## 14. ⚙️ ENVIRONMENT VARIABLES AND CONFIGURATION

| Variable / Parameter | Location / Source | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | Env / `gemini_api_key.txt` | None (User Prompted) | Google Gemini API authentication key |
| `SARVAM_API_KEY` | Env / `sarvam_api_key.txt` | None | Sarvam AI authentication key for Bulbul TTS |
| `NEMO_SPEECH_HTTP_API_KEY` | Environment Variable | None (Open Access) | Bearer token for `nemo-speech serve` HTTP endpoints |
| `--mode` | CLI argument (`agent.py`) | `gemini-live` | Engine mode (`gemini-live` or `sarvam`) |
| `--voice` | CLI argument (`agent.py`) | `Puck` | Gemini native voice (`Puck`, `Aoede`, `Kore`, `Fenrir`, `Charon`) |
| `--speaker` | CLI argument (`agent.py`) | `rahul` | Sarvam voice speaker |
| `--device` | CLI argument (`agent.py`) | `None` (Device 1) | PortAudio input audio device index |
| `--kb` | CLI argument (`agent.py`) | `knowledge_base.json` | Path to JSON business knowledge base |
| `thinking_budget` | Code (`test_gemini_3_7_streaming.py`) | `0` | Disables LLM chain-of-thought for 0ms TTFT |

---

## 15. 📦 DEPENDENCIES

### 15.1 Python Application Dependencies
* `google-genai` (>= 0.1.0): Official Google GenAI SDK for Gemini Live WebSockets.
* `sounddevice` (>= 0.4.6): PortAudio wrapper for real-time audio streams (`InputStream`, `RawOutputStream`).
* `numpy` (~ 1.26.4): Audio buffer manipulation, RMS energy calculation, and sample resampling.
* `httpx` (>= 0.27): Asynchronous HTTP client for Sarvam AI TTS requests.
* `pygame` (>= 2.5.0): Fallback audio playback engine for file-based audio testing.

### 15.2 Native C++ & SDK Runtime Dependencies
* `nemo_speech_asr_c.dll`: Core speech recognition engine compiled with MSVC 64-bit.
* `ggml.dll`: Fast tensor execution runtime for quantized GGUF models.
* `PortAudio`: Low-latency cross-platform audio I/O driver.
* `SentencePiece` & `Protobuf`: Text tokenization and gRPC protocol buffers (installed via `vcpkg`).

---

## 16. 📍 COMPONENT LOCATION MAPPING (LOCAL VS. REMOTE)

| Component | Location | Runtime / Execution Environment | Cost per Min |
| :--- | :--- | :--- | :--- |
| **Speech Recognition (ASR)** | **Local** | Intel Core i7-8550U CPU (`nemo_speech_asr_c.dll`) | **$0.00** |
| **Voice Activity Detection (VAD)** | **Local** | In-Memory RMS energy filter (`agent.py`) | **$0.00** |
| **Business Catalog (KB)** | **Local** | JSON file on local disk (`knowledge_base.json`) | **$0.00** |
| **Audio DMA Stream Output** | **Local** | Hardware buffer (`sounddevice.RawOutputStream`) | **$0.00** |
| **Telephony Gateway** | **Local** | Physical 1-Port Hardware on LAN | Flat ₹299/mo |
| **LLM Reasoning Brain** | **Cloud (Remote)** | Google AI Studio (Gemini 2.5 / 3.7 Flash) | **$0.00** (Free Tier) |
| **Voice Synthesis (TTS)** | **Cloud (Remote)** | Google Gemini Live Multimodal WebSocket | **$0.00** (Included) |
| **Sarvam TTS (Alt Mode)** | **Cloud (Remote)** | Sarvam AI Cloud API (`api.sarvam.ai`) | Per character |

---

## 17. 🛡️ FAILURE MODES AND RETRY PATHS

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│                                  FAILURE MODES & RESOLUTION MATRIX                                │
├──────────────────────────┬─────────────────────────────────────┬──────────────────────────────────┤
│ FAILURE SCENARIO         │ ROOT CAUSE                          │ ARCHITECTURAL RESOLUTION         │
├──────────────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
│ "Foreigner Accent" in    │ Latin English text passed to TTS    │ System prompt enforces           │
│ Hinglish responses       │ applied English G2P phonetics       │ Devanagari script with loanwords │
├──────────────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
│ 3-4 second awkward delay │ Writing .wav files to disk before   │ Zero-disk streaming to           │
│ between voice turns      │ playback; sequential blocking       │ `sounddevice.RawOutputStream`    │
├──────────────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
│ Gemini 3.7 4.3s delay &  │ Internal thinking token generation; │ Set `thinking_budget = 0`;       │
│ 503 UNAVAILABLE errors   │ free-tier demand spikes             │ fallback chain (3.7 ──> 2.5)     │
├──────────────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
│ Dead air on distracted / │ Caller thinking or pauses; agent    │ 4.5s Inactivity Watchdog:        │
│ silent callers           │ previously waited indefinitely      │ Nudge 1 (voice), Nudge 2 (WA)    │
├──────────────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
│ Windows PortAudio driver │ Device conflict on 44.1kHz capture  │ Resample to 16kHz float32;       │
│ "Invalid parameter"      │ with multi-channel endpoints        │ specify explicit `--device 1`    │
+──────────────────────────┴─────────────────────────────────────┴──────────────────────────────────┤
```

---

## 18. ⚠️ CURRENT BOTTLENECKS AND SINGLE POINTS OF FAILURE

1. **Cloud Internet Dependency**:
   - *Risk*: If the local broadband connection drops, the cloud LLM and TTS WebSocket sessions will disconnect.
   - *Impact*: Active calls will freeze.
2. **1-Port Gateway Hardware Concurrency**:
   - *Risk*: A 1-port 4G VoLTE gateway can process exactly **1 concurrent call** at any single moment.
   - *Mitigation*: Scale to 4-port / 8-port gateways (e.g. *Dinstar UC2000-VG-4G*) or connect a SIP Trunk provider for multi-line concurrency.
3. **Google AI Studio Free-Tier Rate Limits**:
   - *Risk*: Free tier is capped at 1,500 requests/day and 15 requests per minute (RPM).
   - *Status*: 70–80 calls/day utilizes ~560 requests (~37% quota). Sudden traffic bursts could trigger HTTP 429/503 rate limits.
4. **Single Host Process Architecture**:
   - *Risk*: `agent.py` runs as a single Python process without an external process supervisor (such as systemd or PM2).
   - *Mitigation*: Run with automated restart supervisors or deploy containerized instances via Docker.
