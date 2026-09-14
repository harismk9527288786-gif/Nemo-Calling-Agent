I'm continuing my NeMo-Speech.cpp setup from another chat.

PROJECT:
- Windows
- CPU inference
- CPU: Intel i5-9400F
- GPU: NVIDIA GT 710, but we're NOT using CUDA
- Project path: C:\AI\NeMo-Speech.cpp
- Build: CMake + Ninja + MSVC
- vcpkg: C:\vcpkg

BUILD STATUS:
- NeMo-Speech.cpp 1.0.0 compiled successfully
- Executable: build-cpu\bin\nemo-speech.exe
- `nemo-speech.exe --help` works
- GGUF model is compatible:
  models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
- Model size: ~742 MB
- Model architecture: ASR RNNT
- Sample rate: 16000 Hz
- runtime_compatible: true

IMPORTANT FIXES ALREADY DONE:
- Installed SentencePiece through vcpkg
- Installed Protobuf through vcpkg
- Fixed Abseil flags linking in src\asr\CMakeLists.txt
- CPU build uses NEMO_SPEECH_GGML_PATCHED=OFF
- Do NOT apply CUDA GGML patches
- Build currently succeeds

TESTING:
English transcription works well.
Hindi transcription works, but rough audio caused errors.
Hinglish works reasonably well, e.g.:
Spoken: "Thankyou Taha bhai for the review, accha hua bhai me nhi aaya allah ka boht boht shkar hai bhai"
Output: "थैंक यू तहा बाय फॉर द रिव्यू अच्छा हुआ भाई, मैं नहीं आया अलवा का बहुत बहुत शुक्र है भाई।"

Performance on ~30-second English audio:
- Offline transcription: ~15.79 sec
- Streaming mode: ~19.24 sec
- CPU: i5-9400F
- Streaming command:
build-cpu\bin\nemo-speech.exe transcribe conversation.wav --model models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf --device cpu --language en --stream

CURRENT GOAL:
I want to continue from here and eventually connect this working ASR to a live microphone/phone audio stream for an AI calling agent.

Do NOT make me rebuild/reinstall everything unless necessary.