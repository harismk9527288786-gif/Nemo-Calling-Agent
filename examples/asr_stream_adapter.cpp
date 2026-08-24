#include "asr_stream_adapter.h"

#include <cstdio>
#include <utility>

AsrStreamAdapter::AsrStreamAdapter(
    const std::string& model_path,
    const std::string& language,
    int gpu,
    int right_ctx)
    : language_(language) {

    nemo_speech_asr_backend_config backend = {};
    backend.size = sizeof(backend);
    backend.gpu = gpu;

    nemo_speech_asr_model_config model = {};
    model.size = sizeof(model);
    model.path = model_path.c_str();

    nemo_speech_asr_streaming_config streaming = {};
    streaming.size = sizeof(streaming);
    streaming.chunk_size = 0.16f;
    streaming.ctc_left_padding = 1.92f;
    streaming.ctc_right_padding = 1.92f;
    streaming.rnnt_right_context = right_ctx;

    nemo_speech_asr_recognizer_config cfg = {};
    cfg.size = sizeof(cfg);
    cfg.backend = &backend;
    cfg.model = &model;
    cfg.streaming = &streaming;

    if (nemo_speech_asr_create(&cfg, &recognizer_) != NEMO_SPEECH_ASR_OK) {
        std::fprintf(
            stderr,
            "[AsrStreamAdapter] create failed: %s\n",
            nemo_speech_asr_last_error());
        recognizer_ = nullptr;
        return;
    }

    nemo_speech_asr_recognition_options opts =
        nemo_speech_asr_recognition_options_default();

    opts.interim_results = true;

    if (!language_.empty()) {
        opts.language_code = language_.c_str();
    }

    if (nemo_speech_asr_streaming_recognize(
            recognizer_, &opts, &stream_) != NEMO_SPEECH_ASR_OK) {

        std::fprintf(
            stderr,
            "[AsrStreamAdapter] streaming_recognize failed: %s\n",
            nemo_speech_asr_last_error());

        nemo_speech_asr_destroy(recognizer_);
        recognizer_ = nullptr;
        return;
    }
}

AsrStreamAdapter::~AsrStreamAdapter() {
    if (stream_) {
        nemo_speech_asr_stream_close(stream_);
        stream_ = nullptr;
    }

    if (recognizer_) {
        nemo_speech_asr_destroy(recognizer_);
        recognizer_ = nullptr;
    }
}

void AsrStreamAdapter::set_result_callback(ResultCallback callback) {
    callback_ = std::move(callback);
}

bool AsrStreamAdapter::push(
    const float* samples,
    size_t sample_count,
    int sample_rate) {

    if (!stream_) {
        return false;
    }

    if (nemo_speech_asr_stream_push_f32(
            stream_,
            samples,
            sample_count,
            sample_rate) != NEMO_SPEECH_ASR_OK) {

        std::fprintf(
            stderr,
            "[AsrStreamAdapter] push failed: %s\n",
            nemo_speech_asr_last_error());

        return false;
    }

    return drain_results();
}

bool AsrStreamAdapter::finish() {
    if (!stream_) {
        return false;
    }

    if (nemo_speech_asr_stream_finish(stream_) != NEMO_SPEECH_ASR_OK) {
        std::fprintf(
            stderr,
            "[AsrStreamAdapter] finish failed: %s\n",
            nemo_speech_asr_last_error());

        return false;
    }

    return drain_results();
}

bool AsrStreamAdapter::drain_results() {
    nemo_speech_asr_result* result = nullptr;

    while (nemo_speech_asr_stream_next(stream_, &result) ==
               NEMO_SPEECH_ASR_OK &&
           result) {

        const char* text =
            nemo_speech_asr_result_transcript(result, 0);

        Result output;
        output.text = text ? text : "";
        output.is_final =
            nemo_speech_asr_result_is_final(result);
        output.audio_processed_sec =
            nemo_speech_asr_result_audio_processed(result);

        if (callback_) {
            callback_(output);
        }

        nemo_speech_asr_result_destroy(result);
        result = nullptr;
    }

    return true;
}