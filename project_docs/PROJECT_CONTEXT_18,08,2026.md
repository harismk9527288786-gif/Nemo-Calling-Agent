# NeMo-Speech AI Calling Agent — Project Context

## Current Hardware

- OS: Windows 64-bit
- CPU: Intel Core i7-8550U
- RAM: 8 GB
- GPU: AMD Radeon R7 M460 4 GB + Intel UHD Graphics 620
- CUDA is NOT being used.
- Current target: CPU inference.

## Project

Project path:

`C:\AI\NeMo-Speech.cpp`

vcpkg path:

`C:\vcpkg`

The project uses NVIDIA NeMo-Speech.cpp for local ASR/STT.

Eventual architecture:

```text
Phone / SIP
    ↓
Audio stream
    ↓
NeMo ASR
    ↓
LLM / Google ADK
    ↓
TTS
    ↓
Phone / SIP
    ↓
Customer
```

---

# Previous ASR Work — COMPLETED

NeMo-Speech.cpp 1.0.0 was successfully compiled previously.

Known executable:

`build-cpu\bin\nemo-speech.exe`

`nemo-speech.exe --help` worked successfully.

## Model

Current model:

`models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`

- Size: approximately 742 MB
- Architecture: ASR RNNT
- Sample rate: 16000 Hz
- `runtime_compatible: true`

## Dependencies Previously Solved

vcpkg was used.

Previously installed:

- SentencePiece
- Protobuf

An Abseil flags linking issue in:

`src\asr\CMakeLists.txt`

was fixed.

CPU configuration:

`NEMO_SPEECH_GGML_PATCHED=OFF`

IMPORTANT:

- Do NOT apply CUDA GGML patches.
- Do NOT switch the project to CUDA unless explicitly required later.

## Previous ASR Testing

English transcription worked well.

Hindi transcription worked, although rough audio caused some errors.

Hinglish worked reasonably well.

Example spoken audio:

> Thankyou Taha bhai for the review, accha hua bhai me nhi aaya allah ka boht boht shkar hai bhai

Example output:

> थैंक यू तहा बाय फॉर द रिव्यू अच्छा हुआ भाई, मैं नहीं आया अलवा का बहुत बहुत शुक्र है भाई।

## Previous Performance

For approximately 30 seconds of English audio on the previous Intel i5-9400F machine:

- Offline transcription: approximately 15.79 seconds
- Streaming transcription: approximately 19.24 seconds

The current machine is an Intel i7-8550U with 8 GB RAM, so performance must be benchmarked on the current machine rather than assumed.

---

# Existing Streaming Components

## `asr_stream_adapter.h`

Contains `AsrStreamAdapter`, a wrapper around the NeMo C ASR streaming API.

It supports:

- `push(float samples)`
- `finish()`
- interim results
- final results
- language selection
- audio processed time
- result callbacks

## `asr_stream_adapter.cpp`

Implements the adapter and creates a streaming NeMo recognizer with:

- CPU/GPU backend selection
- GGUF model path
- streaming configuration
- RNNT right context
- interim results
- language selection
- float32 audio input

## `transcribe_live.cpp`

The live microphone application uses:

```text
Microphone
    ↓
PortAudio
    ↓
16 kHz mono audio
    ↓
NeMo streaming ASR
    ↓
Interim/final transcripts
```

---

# LATEST CODEX UPDATE — IMPORTANT

The live-ASR microphone path was implemented and rebuilt.

The following work was completed:

- Added native-rate microphone capture with streaming resampling to 16 kHz.
- Added `--list-devices`.
- Added float32 → int16 capture fallback.
- Added detailed Windows host-error reporting.
- Copied CMake/Ninja into `build-tools` as requested.
- Rebuilt `transcribe_live.exe`.

The rebuilt executable exists at:

`build-cpu\bin\transcribe_live.exe`

## Current Testing Commands

From:

```text
C:\AI\NeMo-Speech.cpp
```

Device listing:

```powershell
.\build-cpu\bin\transcribe_live.exe --list-devices
```

Live ASR invocation is approximately:

```powershell
.\build-cpu\bin\transcribe_live.exe .\models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf --gpu -1
```

Use the actual options shown by `--help` if the current executable exposes additional/changed arguments.

---

# CURRENT STATUS

## ASR model loading

CONFIRMED WORKING.

The ASR model loads successfully in `transcribe_live.exe`.

## Current blocker: Windows microphone capture

The remaining blocker is NOT the ASR model.

Every tested Windows audio capture endpoint currently fails in the driver with:

`invalid parameter`

This remains true even after:

- native-rate capture
- streaming resampling to 16 kHz
- float32 capture
- int16 fallback

Therefore the current problem is Windows/PortAudio audio capture rather than NeMo model loading.

## Suggested immediate action

Before making further code changes:

1. Open Windows Settings.
2. Go to:

   `Privacy & security → Microphone`

3. Make sure microphone access is enabled.
4. Make sure:

   `Let desktop apps access your microphone`

   is enabled.
5. Close other applications that may currently be using the microphone.
6. Retry the microphone test, particularly the relevant device (previous testing referenced device 1).

Do not assume the ASR implementation is broken until Windows microphone capture permissions/device configuration have been ruled out.

---

# vcpkg STATUS / BUILD NOTE

The transferred vcpkg tree currently lacks some Abseil debug libraries.

This prevents a normal CMake reconfigure from completing cleanly.

However:

- The rebuilt `transcribe_live.exe` exists.
- The current executable can be tested without immediately rebuilding everything.
- Do NOT delete or reinstall the entire vcpkg tree just because of this.
- If a future rebuild is actually required, diagnose the exact missing Abseil/debug library first and install only what is necessary.

---

# IMPORTANT DEVELOPMENT RULES

This is an existing project.

Do NOT treat it as a new project.

Do NOT unnecessarily:

- reinstall NeMo-Speech.cpp
- reinstall CMake
- reinstall Ninja
- reinstall vcpkg
- reinstall all dependencies
- rebuild the whole project without a reason
- replace the working GGUF model
- apply CUDA patches
- switch to CUDA

If an actual error appears:

1. Identify the exact error.
2. Determine whether it is code, build configuration, dependency, Windows, PortAudio, or hardware related.
3. Make the smallest necessary change.
4. Rebuild only the affected target when possible.
5. Test again.

---

# IMMEDIATE PROJECT GOAL

Finish live microphone streaming ASR.

Target pipeline:

```text
Microphone
    ↓
PortAudio
    ↓
Native-rate capture
    ↓
Streaming resampling
    ↓
16 kHz mono float32
    ↓
NeMo-Speech streaming ASR
    ↓
Interim transcript
    ↓
Final transcript
```

The first priority is to get a real microphone stream working reliably on Windows.

Once that works:

1. Measure real-time latency on the current i7-8550U.
2. Verify interim transcript stability.
3. Verify final transcript accuracy.
4. Test English.
5. Test Hindi.
6. Test Hinglish.
7. Add clean audio-stream interfaces for the future phone/SIP input.

Only after live ASR is stable should we connect:

```text
ASR → LLM / Google ADK → TTS
```

and later:

```text
Phone / SIP → ASR → LLM → TTS → Phone / SIP
```

---

# TTS STATUS

Fish Speech was investigated as a possible TTS solution.

Do NOT integrate Fish Speech yet.

First finish and benchmark live ASR.

TTS will be selected after the latency and deployment requirements of the complete calling pipeline are clearer.

---

# CURRENT NEXT STEP

Do NOT start by rebuilding everything.

First test the existing rebuilt:

`build-cpu\bin\transcribe_live.exe`

and resolve the Windows microphone capture `invalid parameter` issue.

If microphone permissions and other applications are ruled out and the error remains, inspect the exact PortAudio/Windows host error and then make the smallest targeted fix.

