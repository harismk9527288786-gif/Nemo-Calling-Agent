# Phase 0 — Pipeline Audit and Phased Plan

**Date:** 2026-08-24
**Baseline commit:** `1abed1f` (`chore: snapshot AI calling assistant work before pipeline tightening`)
**Scope:** read-only audit of the current calling pipeline against `AI_Calling_Assistant_Pipeline_Tightening.md`. No functional code was changed.

---

## Verdict up front

The headline conclusion is not the one the tightening plan expects. Adding up the latency contributions that are actually measured or derivable from the code, the current end-of-turn to first-audio path costs roughly **1.0–1.3 seconds** on a good network, which is already inside the 1.5–3 second target. The pipeline does not have a raw speed problem.

What it has instead are three structural problems that no amount of model tuning will fix: barge-in is impossible by construction rather than merely slow, the ASR final transcript is never actually produced because of a two-line Python bug, and ASR inference runs on the asyncio event loop at ~1.03x real-time so the whole process has effectively zero CPU headroom. Beyond that, the turn-detection logic relies on a fixed RMS energy threshold that will not survive contact with a real phone line, and there is no telephony layer at all yet, so none of this has been exercised on the audio conditions that matter.

The recommendation is therefore to reorder the plan: instrument first to replace estimates with real numbers, fix the two outright bugs, then make the barge-in and VAD changes, and only then consider the deeper architectural question described below. Phases 5 and 7 of the source plan should be skipped entirely, for reasons explained in the mapping section.

---

## What the pipeline actually is

The documented architecture and the running code differ, so this section describes the code.

Audio enters through `sounddevice.InputStream` at 16 kHz mono float32 in 80 ms blocks, pushed into a `queue.Queue` from the PortAudio callback thread (`agent.py:180-187`, `agent.py:247-254`). Turn detection is a fixed RMS energy threshold of 0.015 with a 300 ms silence window, implemented inline in the main loop (`agent.py:27-28`, `agent.py:314-336`). Speech-to-text is the local NeMo engine reached over ctypes, model `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`, CPU only (`nemo_asr.py:19-26`, `agent.py:152`).

The critical thing to understand is that **the LLM and the TTS are not separate components**. Both are Gemini Live native audio, `gemini-2.5-flash-native-audio-latest`, over a single persistent websocket session opened once per run (`agent.py:283`). The agent sends the ASR transcript as **text** and receives 24 kHz PCM audio back, which it writes straight to a `RawOutputStream` with no disk buffering (`agent.py:213-227`). Knowledge-base content is rendered into the system prompt once at startup (`agent.py:34-81`).

So the real shape is: local streaming ASR, then a cloud speech-out model driven by text. There is no local TTS in the path, no text LLM stage separate from synthesis, and no telephony — input is a microphone. The `--mode sarvam` branch is half-built: it creates a Gemini text chat (`agent.py:163-178`) but never calls Sarvam's synthesis endpoint anywhere in the file, and `play_audio_file` (`agent.py:189-204`) writes and deletes temp files, so if that path is ever completed it will reintroduce disk I/O into the real-time path.

The C++ side is considerably more capable than the Python client uses. It exposes an HTTP server with a genuine streaming websocket ASR endpoint at `/v1/realtime` (`server/http/http_server.cpp:1049-1050`), a Riva-compatible gRPC server whose `SynthesizeOnline` truly streams PCM out (`src/services/grpc_tts.cc:282-306`), models held resident in an `EngineRegistry` with a real warm-up pass (`app/serve.cpp:470-528`), and — most relevant below — a **Silero VAD implementation with a mid-stream endpointer that is currently switched off** (`src/asr/vad/silero_vad.h:51`, `src/asr/vad/vad_endpointer.h:34`, disabled by default at `:14`).

---

## Measured baseline, and the number nobody has

From `project_docs/PROJECT_CHECKPOINT_19_08_2026.md:37-54`, on this i7-8550U: ASR runs at 1.68x real-time offline and **1.03x streaming**, under 100 ms per chunk. Gemini Live native audio measured ~400 ms to first audio, which was the fastest of the six TTS options tried (Edge-TTS ~150 ms, Piper ~590 ms, Sarvam ~1200 ms, Kokoro ~18000 ms and rejected). Separately, `PROJECT_JOURNEY_AND_CASE_STUDY.md:90` records `gemini-3.7-flash` costing ~4.3 s to first speech until `thinking_budget=0` was set — keep that setting.

The ASR streaming geometry is worth stating precisely because it removes a common source of wasted effort. The model is RNNT with a cache-capable encoder, so `CacheStreamRunner` is selected (`src/asr/recognizer.cpp:132-143`). That runner **ignores `chunk_size`, `ctc_left_padding` and `ctc_right_padding` entirely** — the 1.92 s paddings hardcoded at `nemo_asr.py:168-171` are dead config for this model and are not costing latency. The only live knob is `rnnt_right_context = 1`, which yields 2 encoder frames, or **160 ms** of algorithmic latency, and that already matches the project's own low-latency default everywhere it appears (`config/asr.example.yaml:24`, `include/nemo_speech/asr.h:75`, `examples/transcribe_live.cpp:219`). The first partial requires 256 ms of pushed audio; steady state is one chunk per 160 ms.

Putting that together as an estimated budget:

| Stage | Estimated cost | Basis |
|---|---:|---|
| Silence window before end-of-turn fires | 300 ms | `agent.py:27` |
| ASR partial lag, added on top | 160–250 ms | `CacheStreamRunner` geometry; inflated further by the bug below |
| Gemini Live first audio | ~400 ms | measured, checkpoint doc |
| Network round trip, India to Google | 100–300 ms | not measured |
| **Total, end of speech to first audio** | **~1.0–1.3 s** | |

**No end-to-end mic-to-speaker number has ever been recorded**, and `agent.py` contains no latency instrumentation whatsoever. The "<400 ms total turnaround" figure in `PROJECT_ARCHITECTURE.md:14` and `PROJECT_IDEA_AND_BUSINESS_CASE.md:19` is a target, not a measurement, and the SK Furniture deployment narrative at `PROJECT_IDEA_AND_BUSINESS_CASE.md:106-118` cannot be literally true given no telephony exists. Everything in the table above is an estimate and should be treated as one until Phase 9 lands.

---

## The findings that matter

### The ASR final transcript is never produced

`nemo_asr.py:259-262` defines `finish_stream()` with a `yield from` in the body, which makes the whole function a **generator**. `agent.py:341` calls `self.asr.finish_stream()` and discards the returned generator without iterating it, so the function body never executes and `nemo_speech_asr_stream_finish` is never reached.

The C ABI is explicit that this call is required: "No more audio: flush the tail. The end-of-stream final is then returned by `nemo_speech_asr_stream_next`" (`include/nemo_speech/asr.h:230-233`), and `stream_close` performs no flush (`src/asr/c_api.cpp:393-395`). What is being silently lost on every single turn is the last under-250 ms of buffered audio, any tail tokens the greedy RNNT head is still holding, end-of-utterance punctuation, and **all postprocessing including PnC and inverse text normalization**, because interim results bypass postprocessing entirely (`src/asr/recognizer.cpp:318-325`).

So the agent has been feeding Gemini raw, unpunctuated interim text for every turn. To be fair to the current behaviour, the turn is not lost: `current_transcript` is populated from interims at `agent.py:323-325`, so most of the words do get through. What is lost is the tail that was still buffered, the punctuation, and the normalization. It remains a two-line fix with a real accuracy payoff and no latency cost, and it is the best value change in this document.

### Barge-in is not slow, it is impossible

`agent.py:182` gates the capture callback on `if not self.is_speaking`, so while the agent is talking, caller audio is **discarded at the microphone callback** and never reaches the queue. `agent.py:234-238` then drains whatever did accumulate. On top of that, `generate_gemini_native_live_audio` is awaited inline in the main loop at `agent.py:350`, so for the entire duration of the agent's turn the loop is not reading the queue or pushing to ASR at all, and `out_stream.write()` at `agent.py:225` is a blocking call that stalls the receive loop on playback.

Phase 8 of the source plan is therefore not a matter of tuning a detection threshold. It requires removing the `is_speaking` gate, decoupling playback from generation, and running detection concurrently with output. Two related details found while verifying this:

The gate also **under-covers at the tail**, which is a separate bug pointing the opposite way. `is_speaking` flips back to False at `agent.py:232` as soon as `turn_complete` breaks the receive loop, but `out_stream.write()` feeds a buffered `RawOutputStream`, so roughly a buffer's worth of agent audio is still playing after capture resumes. The agent can hear and transcribe its own voice. Any barge-in work needs to gate on playback actually draining, not on the generation loop exiting, or it will trade a missed-interruption bug for a self-trigger echo bug.

Also, no audio is captured at all during the opening greeting, because `in_stream` is not entered until `agent.py:302` while the greeting is generated at `agent.py:287`. A caller who starts talking over the greeting is inaudible. Conversely, the post-turn drain at `agent.py:234-238` discards very little, since the queue is already drained every loop iteration — that part of the plan's concern is not the problem here.

### ASR inference runs on the event loop — and it is not the call you would expect

ASR decode happens on the asyncio thread, so at ~1.03x real-time it saturates **the same thread that services the Gemini Live websocket**. The `await asyncio.sleep(0.02)` at `agent.py:373` is the only yield point, and with 4 cores, 8 GB of RAM and no CUDA there is no headroom to absorb it. This is the root cause of most jitter and the reason the pipeline cannot overlap work the way the plan's final success condition describes.

The important detail, which is easy to get wrong: **the expensive call is `poll_results()` at `agent.py:323`, not `push_audio()` at `agent.py:321`.** `push_audio` only buffers — `stream_push_f32` reaches `RecognitionStream::push` (`src/asr/recognizer.cpp:268-302`) which does a plain `audio_buf_.insert` (`src/asr/runner.cpp:234-238`), and the header states the contract outright: "Buffer mono float32 audio (no decode); next() drives decoding" (`src/asr/recognizer.h:62-64`). The mel front-end and encoder run in `step()` (`src/asr/runner.cpp:963+`), reached through `next()` (`src/asr/recognizer.cpp:414-438`), which the Python side calls from inside `poll_results` (`nemo_asr.py:239-257`).

So the worker thread must own the **drain**, not just the push. Offloading `push_audio` alone would buy nothing at all. Note also that ctypes releases the GIL during foreign calls, which is why microphone capture does not glitch — the capture thread keeps running — but that does not help the coroutine, which is blocked either way.

### End-of-turn detection is fragile in two distinct ways

First, `last_speech_time` is updated both by RMS crossings and by **transcript changes** (`agent.py:329-330`). Because partials arrive later than the audio they describe, a late partial pushes the deadline forward, so the effective end-of-turn delay is 300 ms measured from the last partial rather than from the last speech — ASR lag is silently added to the silence window.

Second, a fixed `ENERGY_THRESHOLD = 0.015` with no noise floor adaptation is a reasonable choice for a quiet room and a poor one for a phone line, where continuous line noise, comfort noise and codec artefacts will either hold the threshold permanently high or clip natural pauses. The plan's own warning about not cutting off callers who pause naturally applies directly.

The good news is that the fix already exists in the repo and is simply not wired up. `nemo_asr.py:179-181` passes `NULL` for both the `vad` and `endpointing` config pointers, and `RecognitionOptions.stop_history_eou_ms` (`nemo_asr.py:89`) is never set. Enabling the engine's Silero VAD and `VadEndpointer` gives model-based turn detection with per-stream recurrent state (`src/asr/recognizer.h:167-168`) instead of a hand-rolled energy heuristic, at the cost of exposing a few more config fields through the ctypes wrapper.

### Nothing survives a failure

The Live session is opened with `async with` (`agent.py:283`) and has no reconnect, no timeout handling and no fallback. A dropped websocket propagates out of `run()` and kills the process mid-call. There is no supervision, no per-stage timeout, and no fallback utterance. Against the plan's requirement that one bad call must not crash the service, the current answer is that one bad packet ends everything.

---

## The architectural question you should decide before Phase 1

Because Gemini Live is a **speech-to-speech** model that is currently being driven with text, the local ASR sits in the critical path doing work the cloud model can do itself. That opens a genuine fork, and it changes what the rest of the plan should look like.

**Option A — stream caller audio directly into Gemini Live.** The Live API accepts raw PCM input and performs its own server-side voice activity detection, turn-taking and interruption handling. Local ASR then moves off the critical path and runs in parallel purely for transcripts, logging and analytics. This deletes the 300 ms silence window, the ASR partial lag and the end-of-turn heuristic from the response path in one move, plausibly reaching 500–700 ms end to end, and it gets barge-in essentially for free because the server handles interruption. The costs are real: continuous audio streaming consumes far more quota than short text turns, which matters against a free-tier ceiling of 15 requests per minute and 1500 per day (`PROJECT_ARCHITECTURE.md:406-419`); the system becomes fully dependent on the network for every syllable rather than one request per turn; and the local ASR investment stops being load-bearing.

**Option B — keep local ASR in the critical path and fix the plumbing.** Move ASR to a worker thread, enable the engine's Silero VAD and endpointer, fix the finish bug, decouple playback, and add instrumentation. This keeps transcripts local, keeps quota consumption low and preserves the option of swapping the cloud component out later for a local text LLM plus local TTS. It will not get below roughly 900 ms end to end, because the silence window and ASR lag remain in the path.

I recommend **Option B first, then evaluate A with real numbers**, on the grounds that Option B's work is almost entirely prerequisite anyway and produces the measurements needed to judge whether A is worth its quota cost. But this is a product and cost decision as much as a technical one, so it is yours.

**One dependency to verify before either path:** I could not confirm the exact audio-input API surface of the `google-genai` version installed on your machine, because network access is blocked in this sandbox and the repo's `requirements.txt` covers only upstream model-conversion dependencies — the agent's real runtime dependencies are unpinned and undocumented. The three existing Live test scripts use only `session.send(input=..., end_of_turn=True)`, i.e. text (`test_persistent_live_session.py:55`, `test_streaming_live_playback.py:45`, `test_gemini_live.py:43`). Before Option A is costed seriously, a short probe script should confirm the realtime-audio-input method, the interruption event shape, and whether input transcription can be requested server-side. Pinning those dependencies is worth doing at the same time.

---

## Mapping the source plan onto this codebase

**Phase 1, async session architecture.** Applies, but scope it down. The plan lists six queues; this pipeline needs two, one between capture and ASR and one between generation and playback. A per-call session object is still the right container for state, and it is a prerequisite for concurrency later, but a 1-port gateway means one call at a time for the foreseeable future, so build the isolation without building a scheduler.

**Phase 2, streaming audio.** Largely already satisfied. Capture is already chunked at 80 ms, already float32, already in memory with no temp files in the Gemini path. The remaining work is bounding the queue and recording arrival intervals and queue depth so growth is visible. The one hazard to keep out is `play_audio_file` (`agent.py:189-204`), which writes to disk.

**Phase 3, VAD and turn detection.** Applies, and this is where I would deviate from the plan. Rather than writing new Python VAD, wire the ctypes layer through to the engine's existing Silero VAD and `VadEndpointer`, and make minimum speech duration, silence duration, sensitivity and maximum utterance length configurable as the plan asks. Also decouple the end-of-turn timer from transcript arrivals.

**Phase 4, streaming STT.** Mostly already correct, and NeMo should stay. Do not benchmark against Whisper — there is no evidence of an accuracy problem, and the plan itself forbids the swap without evidence. Two concrete changes: fix the finish flush, and stop recreating the stream per turn until the cost of `streaming_recognize` per turn has been measured. Drop the dead `ctc_*_padding` values or set them honestly so nobody tunes them again.

**Phase 5, Hinglish normalization. Skip.** This phase assumes a text LLM feeding a separate TTS engine. Gemini Live generates speech natively and the Devanagari-with-English-loanwords behaviour is already handled by the system prompt (`agent.py:71`), a fix that took real effort to find (`PROJECT_JOURNEY_AND_CASE_STUDY.md:69-72`). Inserting a normalization layer here would add latency and risk regressing the accent handling. Revisit only if you move to a split LLM-plus-TTS design.

**Phase 6, LLM latency.** Mostly done and worth protecting. Responses are already capped at one to two sentences (`agent.py:73`), the session is already persistent, and the knowledge base is already rendered once at startup rather than per turn. Keep `thinking_budget=0`. The one open item is that the knowledge base is read at startup only, so editing it requires a restart.

**Phase 7, sentence-level TTS. Skip as written.** Gemini Live already streams PCM chunks as it generates, and `agent.py:225` already plays the first chunk on arrival. There is no full-response wait to eliminate. The related real issue is that the blocking write couples playback to generation, which belongs to Phase 8's decoupling work. Note for later: if you ever move to local TTS, the C++ engine only streams via gRPC `SynthesizeOnline` — the HTTP `/v1/audio/speech` route concatenates the whole buffer before responding (`server/http/http_server.cpp:641-651`).

**Phase 8, barge-in.** Applies, and it is the biggest user-visible win after instrumentation. Remove the capture gate, move playback behind its own queue so it can be cleared on interruption, and handle the interruption signal from the Live session. Under Option A most of this is handled server-side.

**Phase 9, instrumentation. Do this first.** Every number in this document above is an estimate. The plan's timestamp list is the right one; the minimum useful subset is speech end, ASR final, request sent, first audio byte, and playback start, logged per turn as one structured line.

**Phase 10, model warm-up.** Already satisfied on the C++ side, which loads once into a resident registry and warms up before serving (`app/serve.cpp:470-528`), and `agent.py` loads the recognizer once in the constructor. Nothing to do.

**Phase 11, CPU optimization.** Defer. The model is already Q8 quantized and the audio path is already numpy-vectorized. The real CPU problem is scheduling, not arithmetic, and it is addressed by moving ASR off the event loop. Measure after that before touching quantization.

**Phase 12, failure handling.** Applies fully, and is currently absent. Reconnect with backoff on the Live session, a per-stage timeout, a canned fallback utterance when generation fails, and a supervisor so a single call cannot take down the process.

**Phase 13, concurrency.** Defer beyond session isolation. One gateway port means one call; the plan says as much itself. Worth knowing for later: each websocket session on the C++ server pins a thread pool worker for its lifetime and the pool defaults to four (`server/http/http_server.cpp:348-349`), and `Synthesizer` has no mutex (`src/tts/synthesizer.h:62-96`), so concurrent local TTS would need care.

**Phase 14, production architecture.** Blocked on hardware. No SIP, RTP or GSM code exists anywhere in the repo; the gateway is still a purchasing recommendation (`project_docs/COMPLETE_PROJECT_GUIDE.md:143-152`). Until it arrives, all tuning is against microphone audio, which is materially cleaner than an 8 kHz phone codec. Expect VAD thresholds and possibly ASR accuracy to need revisiting once real call audio is available.

---

## Proposed order of work

The sequence below is ordered by payoff against risk, and each step is independently revertible from `1abed1f`.

The first step is instrumentation and a repeatable benchmark, because it costs little and converts this document's estimates into facts. Second is the pair of outright bugs: the `finish_stream` generator and the resulting loss of postprocessing. Third is moving the ASR **decode drain** — `poll_results`, not merely `push_audio` — onto a worker thread with bounded queues either side, which is the change that makes overlap possible at all. Fourth is barge-in, which depends on the third and must gate on playback draining rather than on the generation loop exiting. Fifth is replacing energy VAD with the engine's Silero endpointer and decoupling the turn timer from partial arrivals. Sixth is failure handling and reconnection. Only then is it worth re-measuring and deciding whether Option A's quota cost buys enough latency to be worthwhile.

Steps one and two are safe to do together and I would expect them to be quick. Step three is the one that genuinely changes the shape of `agent.py` and deserves its own review before and after.

### Progress: steps 1 and 2 done (2026-08-24)

Steps one and two are implemented and byte-compile clean. Step one added `pipeline_metrics.py` (per-turn timeline instrumentation, gated by `NEMO_AGENT_METRICS`, writes JSONL to the already-ignored `client_logs/`) and `bench_pipeline.py` (a microphone-free benchmark that paces a wav through the ASR in real time and separately times Gemini Live first-audio). `agent.py` now records a `TurnMetrics` per turn without altering any turn-taking behaviour: the marks are recorded alongside the existing decisions, and the energy-only `last_voice` mark is kept deliberately separate from the partial-bumped `last_speech_time` so the summary can quantify how much ASR lag inflates the silence window.

Step two fixed the `finish_stream` generator bug in `nemo_asr.py` (it is now a plain function returning a list, with a new `final_transcript()` convenience wrapper) and wired it into `agent.py`, which now consumes the flushed, postprocessed final transcript and falls back to the interim only if the flush raises or yields nothing. Critically, `start_stream` is called on the path *after* the flush regardless of whether the flush raised, so a flush failure degrades to interim text for one turn rather than leaving the recognizer dead for the rest of the call.

The benchmark's real-time pacing sleeps to each chunk's **end** (a capture device only releases a fully-recorded chunk), and RTF now sums push and drain CPU while also reporting decode-only RTF, since `push_audio` merely buffers and `poll_results` does the compute. The "Before" column below must be filled on the target machine — the DLL, the model and the Gemini key only exist there; nothing in this sandbox can produce a real number.


---

## Benchmark to record before and after each step

| Metric | Before | After |
|---|---:|---:|
| End-of-speech to first audio byte (p50 / p95) | | |
| Silence window contribution | | |
| ASR final latency after speech end | | |
| Gemini first-audio latency | | |
| Playback start offset | | |
| Barge-in: caller speech to output stop | | |
| CPU during steady conversation | | |
| RAM resident | | |
| Audio queue depth, max | | |
| Dropped chunks | | |

The row that matters most is the first, and it currently has no value in either column.

### How to record it

Two commands, both on the target machine. The first needs no network and no API key:

```
python bench_pipeline.py --stage asr --wav test.wav --label "before step 3"
python bench_pipeline.py --stage gemini --runs 5
```

That fills the ASR and Gemini rows and prints a pasteable markdown block. The end-to-end row cannot come from the benchmark — it only exists during a real call, so place one call, talk for a few turns, then press Ctrl+C. `agent.py` prints a p50/p95 table per stage on exit and leaves the per-turn records in `client_logs/latency_<timestamp>_<call_id>.jsonl`. Set `NEMO_AGENT_METRICS=0` to turn all of it off.

Two things to check on that first instrumented call, because they are the audit's predictions and this is the run that confirms or refutes them. First, whether "Flushed ASR final used on N/N turns" reports every turn — if it does, the flush fix is live and the agent is finally acting on postprocessed text. Second, how far the median end-of-turn delay overshoots the configured 300 ms silence window; the summary prints that subtraction explicitly, and a large overshoot is the partial-driven timer bumping that step five removes.


---

## What I will not change without asking

The NeMo ASR engine and model choice stay as they are; there is no evidence of an accuracy problem and the plan forbids a Whisper swap without benchmark evidence. The Devanagari-with-loanwords prompt behaviour stays exactly as written. The C++ engine, its build and its API contracts stay untouched, since the work needed is all on the Python client side, with the single exception of exposing existing VAD and endpointing config fields through the ctypes wrapper. `git reset --hard 1abed1f` returns the tree to the state audited here.
