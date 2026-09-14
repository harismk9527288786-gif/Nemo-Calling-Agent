#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "nemo_speech/asr.h"

class AsrStreamAdapter {
public:
    struct Result {
        std::string text;
        bool is_final;
        float audio_processed_sec;
    };

    using ResultCallback = std::function<void(const Result&)>;

    AsrStreamAdapter(
        const std::string& model_path,
        const std::string& language,
        int gpu = -1,
        int right_ctx = 1);

    ~AsrStreamAdapter();

    AsrStreamAdapter(const AsrStreamAdapter&) = delete;
    AsrStreamAdapter& operator=(const AsrStreamAdapter&) = delete;

    bool push(
        const float* samples,
        size_t sample_count,
        int sample_rate = 16000);

    bool finish();

    void set_result_callback(ResultCallback callback);

private:
    bool drain_results();

    nemo_speech_asr_recognizer* recognizer_ = nullptr;
    nemo_speech_asr_stream* stream_ = nullptr;

    std::string language_;
    ResultCallback callback_;
};