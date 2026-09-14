"""Repeatable latency benchmark for the calling pipeline.

Run this BEFORE and AFTER every optimization step, per rule 7 of
AI_Calling_Assistant_Pipeline_Tightening.md. It deliberately avoids the
microphone so results are reproducible: a wav file is fed to the ASR in
real time (as if it were arriving from a call), and the Gemini Live stage is
timed with a fixed text prompt.

    # ASR stage only, no network, no API key needed
    python bench_pipeline.py --stage asr --wav test.wav

    # Gemini Live first-audio latency, 5 samples
    python bench_pipeline.py --stage gemini --runs 5

    # Both, and write a markdown row you can paste into the plan's table
    python bench_pipeline.py --stage all --wav test.wav --label "before step 3"

What each number means:

  first_partial_ms   Audio pushed to first non-empty interim transcript. The
                     engine needs ~256ms of audio before the first chunk can
                     run, so anything near that is expected, not a problem.
  final_ms           Cost of the end-of-stream flush after the last audio was
                     pushed. This is what the finish_stream fix restored.
  rtf                Real-time factor: total decode CPU time / audio duration.
                     Must stay below 1.0 or the pipeline cannot keep up. On the
                     i7-8550U this has measured ~1.03, i.e. no headroom, which
                     is why the decode drain belongs on a worker thread.
  ttfa_ms            Gemini Live: request sent to first audio byte received.

Note the ASR stage measures decode cost in isolation, on a quiet thread. The
live agent will be slower because the same thread also services the
websocket -- comparing the two is itself a useful measurement.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import wave

import numpy as np


# ---------------------------------------------------------------- ASR stage

def load_wav_mono16k(path: str):
    """Read a wav as float32 mono. Returns (samples, sample_rate).

    Deliberately stdlib-only so the benchmark has no dependency the agent
    itself does not already have. Non-16kHz input is reported, not resampled,
    because silently resampling would distort the very numbers we are measuring.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return np.ascontiguousarray(data, dtype=np.float32), sample_rate


def resample_to_16k(samples: np.ndarray, sample_rate: int):
    """Linear-interpolation resample to 16kHz.

    Why bother: the agent feeds the engine 16kHz mic audio. If the benchmark
    pushes 22.05k or 48k instead, the engine's internal resampler runs on every
    chunk and inflates the RTF with work production never pays -- which would
    make before/after comparisons meaningless. Linear interpolation is crude
    (it will alias a little), but this measures *timing*, not accuracy, and
    keeping it dependency-free matters more than a perfect anti-alias filter.
    """
    if sample_rate == 16000:
        return samples, sample_rate
    n_out = int(round(len(samples) * 16000.0 / sample_rate))
    if n_out <= 1:
        return samples, sample_rate
    src_idx = np.linspace(0.0, len(samples) - 1, n_out, dtype=np.float64)
    out = np.interp(src_idx, np.arange(len(samples), dtype=np.float64), samples)
    return np.ascontiguousarray(out, dtype=np.float32), 16000


def bench_asr(
    wav_path: str,
    chunk_ms: float = 80.0,
    realtime: bool = True,
    model_path: str = None,
    resample: bool = True,
):
    """Feed a wav through the streaming ASR and time the stages.

    ``realtime=True`` sleeps between chunks so the arrival pattern matches a
    real call; that makes wall-clock latencies meaningful. ``realtime=False``
    pushes as fast as possible, which measures raw throughput / RTF only.
    """
    from nemo_asr import NeMoStreamingASR

    samples, sample_rate = load_wav_mono16k(wav_path)
    source_rate = sample_rate
    if resample:
        samples, sample_rate = resample_to_16k(samples, sample_rate)
    duration_s = len(samples) / float(sample_rate)
    chunk_samples = max(1, int(sample_rate * chunk_ms / 1000.0))

    print(f"  audio: {os.path.basename(wav_path)}  {duration_s:.2f}s @ {sample_rate}Hz"
          + (f" (resampled from {source_rate}Hz)" if sample_rate != source_rate else ""))
    if sample_rate != 16000:
        print(f"  NOTE: pushing {sample_rate}Hz -- the engine resamples internally, so this "
              f"RTF includes work the 16kHz production path does not do.")
    print(f"  chunks: {chunk_ms:.0f}ms  mode: {'real-time' if realtime else 'as-fast-as-possible'}")

    load_start = time.perf_counter()
    asr = NeMoStreamingASR(model_path=model_path) if model_path else NeMoStreamingASR()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    print(f"  model load: {load_ms:.0f}ms")

    try:
        asr.start_stream(interim_results=True, language_code="hi")

        decode_time = 0.0
        push_time = 0.0
        pacing_late_ms = 0.0   # worst-case overshoot of the real-time schedule
        first_partial_ms = None
        partial_count = 0
        last_text = ""
        t0 = time.perf_counter()

        for offset in range(0, len(samples), chunk_samples):
            chunk = samples[offset:offset + chunk_samples]

            if realtime:
                # Sleep until the chunk's END, not its start. A capture device
                # only hands over a chunk once it has been fully recorded, so
                # pacing on the start time would deliver every chunk one
                # chunk-length early and understate first-partial latency by
                # up to chunk_ms. Target is absolute (from t0), so this does
                # not accumulate drift.
                target = t0 + ((offset + len(chunk)) / float(sample_rate))
                sleep_for = target - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                # Track how late we actually woke up. Windows time.sleep() can
                # have ~15ms granularity on older Pythons, and a saturated CPU
                # delays wakeup too. If this is large, the latency figures carry
                # that much noise -- better to report it than to silently
                # attribute scheduler jitter to the ASR.
                late = (time.perf_counter() - target) * 1000.0
                if late > pacing_late_ms:
                    pacing_late_ms = late

            # Time push AND drain. push_audio only buffers, but in the live
            # agent both run on the asyncio thread, so both count against the
            # event loop. Reported separately below so the split is visible.
            push_start = time.perf_counter()
            asr.push_audio(chunk, sample_rate=sample_rate)
            push_time += time.perf_counter() - push_start

            # poll_results() is where the mel front-end and encoder actually
            # run -- push_audio only buffers. This is the expensive half.
            drain_start = time.perf_counter()
            for _is_final, text in asr.poll_results():
                if text:
                    last_text = text
                    partial_count += 1
                    if first_partial_ms is None:
                        first_partial_ms = (time.perf_counter() - t0) * 1000.0
            decode_time += time.perf_counter() - drain_start

        push_done = time.perf_counter()
        final_text = asr.final_transcript()
        final_ms = (time.perf_counter() - push_done) * 1000.0
    finally:
        asr.close()

    rtf = (decode_time + push_time) / duration_s if duration_s > 0 else float("nan")

    return {
        "audio_s": round(duration_s, 2),
        "source_rate": source_rate,
        "pushed_rate": sample_rate,
        "model_load_ms": round(load_ms, 1),
        "first_partial_ms": None if first_partial_ms is None else round(first_partial_ms, 1),
        "final_flush_ms": round(final_ms, 1),
        "decode_cpu_s": round(decode_time, 3),
        "push_cpu_s": round(push_time, 3),
        "rtf": round(rtf, 3),
        "rtf_decode_only": round(decode_time / duration_s, 3) if duration_s > 0 else None,
        "pacing_worst_late_ms": round(pacing_late_ms, 1),
        "partials": partial_count,
        "interim_text": last_text,
        "final_text": final_text,
        "final_differs": final_text.strip() != last_text.strip(),
    }


# ------------------------------------------------------------- Gemini stage

def load_production_system_instruction(kb_path: str):
    """Reuse agent.py's own system-instruction builder.

    This matters more than it looks: the system instruction is what the model
    prefills before it can emit a single audio token, and agent.py sends the
    full knowledge-base prompt. Benchmarking with a short stand-in would
    report a time-to-first-audio the production path never achieves.

    Importing agent pulls in the ASR DLL, pygame and sounddevice. That is fine
    on the target machine and it guarantees we measure the real prompt; if the
    import fails we say so loudly rather than quietly measuring the wrong thing.
    """
    try:
        from agent import load_system_instruction
        return load_system_instruction(kb_path), True
    except Exception as exc:
        print(f"  WARNING: could not import agent.load_system_instruction ({type(exc).__name__}: {exc}).")
        print("  Falling back to a short stand-in prompt -- ttfa will be OPTIMISTIC")
        print("  versus production, because the real system instruction is much longer.")
        return (
            "You are an energetic Indian phone calling sales assistant. "
            "Speak in natural conversational Hinglish. Keep responses to 1-2 short sentences."
        ), False


def bench_gemini(
    runs: int = 3,
    voice: str = "Puck",
    model: str = "gemini-2.5-flash-native-audio-latest",
    kb_path: str = "knowledge_base.json",
):
    """Time request-sent to first-audio-byte for Gemini Live, ``runs`` times.

    Uses one persistent session, matching how agent.py works -- per-request
    connections would measure handshake cost that the agent does not pay.
    Model, voice, config shape and receive loop all mirror
    ``generate_gemini_native_live_audio`` so the number is comparable.
    """
    import asyncio
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key and os.path.exists("gemini_api_key.txt"):
        with open("gemini_api_key.txt", "r", encoding="utf-8") as fh:
            api_key = fh.read().strip()
    if not api_key:
        raise SystemExit("No Gemini API key. Set GEMINI_API_KEY or create gemini_api_key.txt")

    prompt = (
        "Reply in one short Hinglish sentence, Devanagari script with English "
        "loanwords: tell the caller the showroom is open till 9:30 PM."
    )

    system_instruction, is_production_prompt = load_production_system_instruction(kb_path)
    print(f"  system instruction: {len(system_instruction)} chars"
          + ("" if is_production_prompt else "  (STAND-IN, not production)"))

    async def _run():
        client = genai.Client(api_key=api_key)
        config = types.LiveConnectConfig(
            response_modalities=['AUDIO'],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
            system_instruction=system_instruction,
        )

        results = []
        connect_start = time.perf_counter()
        async with client.aio.live.connect(model=model, config=config) as session:
            connect_ms = (time.perf_counter() - connect_start) * 1000.0
            print(f"  session connect: {connect_ms:.0f}ms")

            for i in range(runs):
                sent = time.perf_counter()
                await session.send(input=prompt, end_of_turn=True)
                ttfa_ms = None
                total_bytes = 0

                async for response in session.receive():
                    sc = response.server_content
                    if sc is None:
                        continue
                    if sc.model_turn is not None:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                if ttfa_ms is None:
                                    ttfa_ms = (time.perf_counter() - sent) * 1000.0
                                total_bytes += len(part.inline_data.data)
                    if sc.turn_complete:
                        break

                complete_ms = (time.perf_counter() - sent) * 1000.0
                # 24kHz mono int16 = 48000 bytes per second of speech
                audio_s = total_bytes / 48000.0
                print(f"  run {i + 1}/{runs}: ttfa {ttfa_ms:.0f}ms  "
                      f"complete {complete_ms:.0f}ms  audio {audio_s:.2f}s")
                results.append({"ttfa_ms": ttfa_ms, "complete_ms": complete_ms, "audio_s": audio_s})

        return connect_ms, results

    connect_ms, results = asyncio.run(_run())
    ttfas = [r["ttfa_ms"] for r in results if r["ttfa_ms"] is not None]
    return {
        "connect_ms": round(connect_ms, 1),
        "runs": len(results),
        "production_prompt": is_production_prompt,
        "system_instruction_chars": len(system_instruction),
        "ttfa_median_ms": round(statistics.median(ttfas), 1) if ttfas else None,
        "ttfa_min_ms": round(min(ttfas), 1) if ttfas else None,
        "ttfa_max_ms": round(max(ttfas), 1) if ttfas else None,
        "detail": results,
    }


# -------------------------------------------------------------------- main

def _clip(text: str, limit: int = 140) -> str:
    """Keep console output readable; a long turn's transcript can be huge."""
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."


def main():
    parser = argparse.ArgumentParser(
        description="Latency benchmark for the AI calling pipeline. Run before and after each change."
    )
    parser.add_argument("--stage", choices=["asr", "gemini", "all"], default="all")
    parser.add_argument("--wav", default="test.wav", help="Audio file for the ASR stage")
    parser.add_argument("--model", default=None, help="Override ASR model path")
    parser.add_argument("--chunk-ms", type=float, default=80.0, help="Chunk size, matching agent.py's 80ms")
    parser.add_argument("--no-realtime", action="store_true", help="Push audio as fast as possible (throughput only)")
    parser.add_argument("--native-rate", action="store_true",
                        help="Push the wav at its own sample rate instead of resampling to 16kHz. "
                             "Off by default because production feeds 16kHz.")
    parser.add_argument("--runs", type=int, default=3, help="Gemini Live samples")
    parser.add_argument("--voice", default="Puck")
    parser.add_argument("--kb", default="knowledge_base.json",
                        help="Knowledge base used to build the system instruction, "
                             "matching agent.py's --kb. Prompt length drives prefill, "
                             "so use the same one you run calls with.")
    parser.add_argument("--label", default="", help="Tag for this run, e.g. 'before step 3'")
    args = parser.parse_args()

    label = f" [{args.label}]" if args.label else ""
    print("=" * 64)
    print(f"  Pipeline latency benchmark{label}")
    print("=" * 64)

    asr_result = None
    gemini_result = None

    if args.stage in ("asr", "all"):
        print("\n--- ASR stage ---")
        if not os.path.exists(args.wav):
            print(f"  SKIPPED: {args.wav} not found. Pass --wav <file>.")
        else:
            asr_result = bench_asr(
                args.wav,
                chunk_ms=args.chunk_ms,
                realtime=not args.no_realtime,
                model_path=args.model,
                resample=not args.native_rate,
            )
            print(f"  first partial:   {asr_result['first_partial_ms']}ms")
            print(f"  final flush:     {asr_result['final_flush_ms']}ms")
            print(f"  blocking CPU:    {asr_result['decode_cpu_s']}s decode + "
                  f"{asr_result['push_cpu_s']}s push, for {asr_result['audio_s']}s audio")
            print(f"  RTF:             {asr_result['rtf']}"
                  f"  (decode only {asr_result['rtf_decode_only']})"
                  + ("  <-- at or above 1.0: no headroom" if asr_result["rtf"] >= 0.95 else ""))
            print(f"  partials:        {asr_result['partials']}")
            worst_late = asr_result.get("pacing_worst_late_ms")
            if worst_late is not None:
                note = ""
                if worst_late > args.chunk_ms:
                    note = ("  <-- exceeds one chunk: the feeder could not keep the "
                            "schedule, latency figures are noisy")
                elif worst_late > 20:
                    note = "  <-- sleep granularity / CPU contention; treat +/-this as noise"
                print(f"  pacing worst late: {worst_late}ms{note}")
            print(f"  interim text:    {_clip(asr_result['interim_text'])}")
            print(f"  final text:      {_clip(asr_result['final_text'])}")
            if asr_result["final_differs"]:
                print("  NOTE: final differs from interim -- the flush is contributing "
                      "punctuation/postprocessing, as intended.")
            else:
                print("  NOTE: final identical to interim. Expected on short clean audio; "
                      "if it never differs, check that the flush is running.")

    if args.stage in ("gemini", "all"):
        print("\n--- Gemini Live stage ---")
        try:
            gemini_result = bench_gemini(runs=args.runs, voice=args.voice, kb_path=args.kb)
            print(f"  ttfa median:     {gemini_result['ttfa_median_ms']}ms "
                  f"(min {gemini_result['ttfa_min_ms']}, max {gemini_result['ttfa_max_ms']})")
        except SystemExit as exc:
            print(f"  SKIPPED: {exc}")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    # A pasteable row for the plan's before/after table.
    if asr_result or gemini_result:
        print("\n--- Paste into the benchmark table ---")
        print("| Metric | Value |")
        print("|---|---:|")
        if asr_result:
            print(f"| ASR first partial | {asr_result['first_partial_ms']}ms |")
            print(f"| ASR final flush | {asr_result['final_flush_ms']}ms |")
            print(f"| ASR RTF | {asr_result['rtf']} |")
        if gemini_result and gemini_result.get("ttfa_median_ms") is not None:
            print(f"| Gemini first audio (median of {gemini_result['runs']}) | {gemini_result['ttfa_median_ms']}ms |")
        if asr_result and gemini_result and gemini_result.get("ttfa_median_ms") is not None:
            # Estimated, not measured: the live agent also pays the silence
            # window, and its decode competes with the websocket.
            est = 300.0 + asr_result["final_flush_ms"] + gemini_result["ttfa_median_ms"]
            print(f"| Estimated end-of-speech to first audio | ~{est:.0f}ms |")
            print("\nThat last row is an ESTIMATE (300ms silence window + flush + ttfa).")
            print("The real figure comes from agent.py's own per-turn log during a call.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
