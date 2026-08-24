import os
import sys
import ctypes
from ctypes import (
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_size_t,
    c_void_p,
    POINTER,
    Structure,
    byref,
)
import numpy as np

# Set DLL search paths
dll_dir = r"C:\AI\NeMo-speech.cpp\build-cpu\bin"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(dll_dir)
    if os.path.exists(r"C:\vcpkg\installed\x64-windows\bin"):
        os.add_dll_directory(r"C:\vcpkg\installed\x64-windows\bin")

dll_path = os.path.join(dll_dir, "nemo_speech_asr_c.dll")
_lib = ctypes.CDLL(dll_path)


# Structures matching nemo_speech/asr.h
class BackendConfig(Structure):
    _fields_ = [
        ("size", c_size_t),
        ("gpu", c_int32),
    ]


class ModelConfig(Structure):
    _fields_ = [
        ("size", c_size_t),
        ("path", c_char_p),
        ("name", c_char_p),
    ]


class StreamingConfig(Structure):
    _fields_ = [
        ("size", c_size_t),
        ("chunk_size", c_float),
        ("ctc_left_padding", c_float),
        ("ctc_right_padding", c_float),
        ("rnnt_right_context", c_int32),
    ]


class RecognizerConfig(Structure):
    _fields_ = [
        ("size", c_size_t),
        ("backend", POINTER(BackendConfig)),
        ("model", POINTER(ModelConfig)),
        ("streaming", POINTER(StreamingConfig)),
        ("decoder", c_void_p),
        ("vad", c_void_p),
        ("endpointing", c_void_p),
        ("postproc", c_void_p),
        ("diar", c_void_p),
        ("batching", c_void_p),
    ]


class SpeechContext(Structure):
    _fields_ = [
        ("size", c_size_t),
        ("phrases", POINTER(c_char_p)),
        ("phrase_count", c_size_t),
        ("boost", c_float),
    ]


class RecognitionOptions(Structure):
    _fields_ = [
        ("size", c_size_t),
        ("request_id", c_char_p),
        ("language_code", c_char_p),
        ("interim_results", c_bool),
        ("enable_word_time_offsets", c_bool),
        ("enable_automatic_punctuation", c_bool),
        ("verbatim_transcripts", c_bool),
        ("profanity_filter", c_bool),
        ("stop_history_eou_ms", c_int32),
        ("speech_contexts", POINTER(SpeechContext)),
        ("speech_context_count", c_size_t),
        ("max_alternatives", c_int32),
        ("enable_speaker_diarization", c_bool),
        ("max_speaker_count", c_int32),
    ]


# Function signatures
_lib.nemo_speech_asr_version.restype = c_char_p
_lib.nemo_speech_asr_last_error.restype = c_char_p

_lib.nemo_speech_asr_recognition_options_default.restype = RecognitionOptions

_lib.nemo_speech_asr_create.argtypes = [
    POINTER(RecognizerConfig),
    POINTER(c_void_p),
]
_lib.nemo_speech_asr_create.restype = c_int32

_lib.nemo_speech_asr_destroy.argtypes = [c_void_p]
_lib.nemo_speech_asr_destroy.restype = None

_lib.nemo_speech_asr_streaming_recognize.argtypes = [
    c_void_p,
    POINTER(RecognitionOptions),
    POINTER(c_void_p),
]
_lib.nemo_speech_asr_streaming_recognize.restype = c_int32

_lib.nemo_speech_asr_stream_push_f32.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_size_t,
    c_int32,
]
_lib.nemo_speech_asr_stream_push_f32.restype = c_int32

_lib.nemo_speech_asr_stream_finish.argtypes = [c_void_p]
_lib.nemo_speech_asr_stream_finish.restype = c_int32

_lib.nemo_speech_asr_stream_next.argtypes = [c_void_p, POINTER(c_void_p)]
_lib.nemo_speech_asr_stream_next.restype = c_int32

_lib.nemo_speech_asr_stream_close.argtypes = [c_void_p]
_lib.nemo_speech_asr_stream_close.restype = None

_lib.nemo_speech_asr_result_is_final.argtypes = [c_void_p]
_lib.nemo_speech_asr_result_is_final.restype = c_bool

_lib.nemo_speech_asr_result_transcript.argtypes = [c_void_p, c_size_t]
_lib.nemo_speech_asr_result_transcript.restype = c_char_p

_lib.nemo_speech_asr_result_destroy.argtypes = [c_void_p]
_lib.nemo_speech_asr_result_destroy.restype = None


class NeMoStreamingASR:
    def __init__(
        self,
        model_path: str = r"models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        gpu: int = -1,
        right_ctx: int = 1,
    ):
        self._stream = None
        self._recognizer = None
        self.model_path = os.path.abspath(model_path)
        self.gpu = gpu
        self.right_ctx = right_ctx

        self._backend = BackendConfig(size=ctypes.sizeof(BackendConfig), gpu=gpu)
        self._model = ModelConfig(
            size=ctypes.sizeof(ModelConfig),
            path=self.model_path.encode("utf-8"),
            name=None,
        )
        self._streaming = StreamingConfig(
            size=ctypes.sizeof(StreamingConfig),
            chunk_size=0.16,
            ctc_left_padding=1.92,
            ctc_right_padding=1.92,
            rnnt_right_context=right_ctx,
        )
        self._cfg = RecognizerConfig(
            size=ctypes.sizeof(RecognizerConfig),
            backend=ctypes.pointer(self._backend),
            model=ctypes.pointer(self._model),
            streaming=ctypes.pointer(self._streaming),
            decoder=None,
            vad=None,
            endpointing=None,
            postproc=None,
            diar=None,
            batching=None,
        )

        self._recognizer = c_void_p()
        status = _lib.nemo_speech_asr_create(
            byref(self._cfg), byref(self._recognizer)
        )
        if status != 0:
            err = _lib.nemo_speech_asr_last_error()
            raise RuntimeError(
                f"Failed to create NeMo ASR recognizer: {err.decode('utf-8') if err else status}"
            )

    def start_stream(
        self, interim_results: bool = True, language_code: str = None
    ):
        if self._stream is not None:
            self.close_stream()

        opts = _lib.nemo_speech_asr_recognition_options_default()
        opts.interim_results = interim_results
        self._lang_bytes = (
            language_code.encode("utf-8") if language_code else None
        )
        if self._lang_bytes:
            opts.language_code = self._lang_bytes

        stream_handle = c_void_p()
        status = _lib.nemo_speech_asr_streaming_recognize(
            self._recognizer, byref(opts), byref(stream_handle)
        )
        if status != 0:
            err = _lib.nemo_speech_asr_last_error()
            raise RuntimeError(
                f"Failed to start ASR stream: {err.decode('utf-8') if err else status}"
            )

        self._stream = stream_handle

    def push_audio(self, samples_f32: np.ndarray, sample_rate: int = 16000):
        if not self._stream:
            raise RuntimeError("Stream is not started")

        if samples_f32.dtype != np.float32:
            samples_f32 = samples_f32.astype(np.float32)

        c_arr = samples_f32.ctypes.data_as(POINTER(c_float))
        status = _lib.nemo_speech_asr_stream_push_f32(
            self._stream, c_arr, len(samples_f32), sample_rate
        )
        if status != 0:
            err = _lib.nemo_speech_asr_last_error()
            raise RuntimeError(
                f"Failed to push audio: {err.decode('utf-8') if err else status}"
            )

    def poll_results(self):
        """Yields (is_final, transcript_text) for all results currently ready."""
        if not self._stream:
            return

        while True:
            res_handle = c_void_p()
            status = _lib.nemo_speech_asr_stream_next(
                self._stream, byref(res_handle)
            )
            if status != 0 or not res_handle:
                break

            is_final = _lib.nemo_speech_asr_result_is_final(res_handle)
            text_ptr = _lib.nemo_speech_asr_result_transcript(res_handle, 0)
            text = text_ptr.decode("utf-8") if text_ptr else ""

            _lib.nemo_speech_asr_result_destroy(res_handle)
            yield is_final, text

    def finish_stream(self):
        """Flush the tail of the stream and return the drained results.

        This is deliberately NOT a generator. It used to contain a bare
        ``yield from self.poll_results()``, which made the whole function a
        generator function -- so calling ``asr.finish_stream()`` without
        iterating the result created a generator, discarded it, and never
        executed a single line of the body. The underlying
        ``nemo_speech_asr_stream_finish`` was therefore never invoked.

        That silently cost us, on every turn: the last <250ms of buffered
        audio, any tail tokens the greedy RNNT head was still holding,
        end-of-utterance punctuation, and ALL postprocessing (PnC/ITN) --
        interim results bypass postprocessing entirely. The agent was acting
        on raw interim text throughout.

        Returns a list of ``(is_final, transcript)`` tuples so callers cannot
        accidentally reintroduce the lazy-evaluation bug.
        """
        if not self._stream:
            return []

        status = _lib.nemo_speech_asr_stream_finish(self._stream)
        if status != 0:
            err = _lib.nemo_speech_asr_last_error()
            raise RuntimeError(
                f"Failed to finish ASR stream: {err.decode('utf-8') if err else status}"
            )

        return list(self.poll_results())

    def final_transcript(self):
        """Flush the stream and return the best final transcript, or "".

        Convenience wrapper for the common case. Prefers the last result
        flagged final; falls back to the last non-empty result so a turn is
        never lost if the flush yields only interims.
        """
        results = self.finish_stream()
        finals = [text for is_final, text in results if is_final and text.strip()]
        if finals:
            return finals[-1].strip()
        any_text = [text for _, text in results if text.strip()]
        return any_text[-1].strip() if any_text else ""

    def close_stream(self):
        if self._stream:
            _lib.nemo_speech_asr_stream_close(self._stream)
            self._stream = None

    def close(self):
        self.close_stream()
        if self._recognizer:
            _lib.nemo_speech_asr_destroy(self._recognizer)
            self._recognizer = None

    def __del__(self):
        self.close()
