"""Per-turn latency instrumentation for the AI calling pipeline.

Phase 9 of AI_Calling_Assistant_Pipeline_Tightening.md. The audit in
project_docs/PHASE0_AUDIT_AND_PLAN_24_08_2026.md found that every latency
figure for this pipeline was an estimate -- no end-to-end mic-to-speaker
number had ever been recorded. This module exists to replace those
estimates with measurements before any optimization work is done.

Design constraints, because this runs inside a real-time audio path:

  * Recording a mark is a single ``time.perf_counter()`` call and an
    attribute assignment. No formatting, no I/O, no locks on the hot path.
  * All disk I/O happens once per turn, after the turn's audio has already
    been played, never between chunks.
  * If anything in here raises, it must not take the call down -- every
    public entry point is defensive.

Usage:

    metrics = MetricsLogger(call_id="abc123")
    turn = metrics.start_turn(kind="caller")
    turn.mark("last_voice")
    ...
    turn.mark("first_audio")
    metrics.finish_turn(turn)     # logs one line + appends JSONL
    ...
    metrics.close()               # prints the session summary table

Output goes to ``client_logs/`` which is already in .gitignore, so captured
transcripts never end up in version control.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# Ordered timeline of a single conversational turn. Every mark is optional --
# a greeting turn has no ASR marks, and an errored turn may stop partway.
# Keep this list in sync with TurnMetrics fields; the summary table and the
# JSONL schema are both derived from it.
MARKS = (
    "speech_started",    # first mic frame over the energy threshold
    "last_voice",        # last mic frame over the threshold (RMS only -- see note)
    "first_partial",     # first non-empty interim transcript
    "turn_fired",        # end-of-turn decision made
    "asr_final",         # ASR flush drained; final transcript in hand
    "llm_sent",          # request handed to Gemini Live
    "first_audio",       # first audio byte received back
    "playback_started",  # first audio byte handed to the output device
    "turn_complete",     # model signalled end of its turn
)

# NOTE on last_voice: agent.py's turn-taking timer is bumped both by mic
# energy AND by arriving ASR partials, so it drifts later than actual speech.
# last_voice is recorded from mic energy ONLY. The gap between last_voice and
# turn_fired therefore exposes how much the partial-driven bumping inflates
# the configured silence window -- one of the audit's findings. Recording it
# separately changes no behaviour; it only makes the inflation visible.


def _ms(start: Optional[float], end: Optional[float]) -> Optional[float]:
    """Elapsed milliseconds between two perf_counter marks, or None."""
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 1)


def _fmt(value: Optional[float], unit: str = "ms") -> str:
    return "n/a" if value is None else f"{value:.0f}{unit}"


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile. Small samples make interpolation pointless."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return round(ordered[idx], 1)


@dataclass
class TurnMetrics:
    """Timestamps and derived latencies for one turn.

    Marks are ``time.perf_counter()`` values -- monotonic, unaffected by
    clock changes, and only meaningful relative to each other.
    """

    turn_index: int
    call_id: str
    kind: str = "caller"          # caller | greeting | nudge
    wall_clock: str = field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))

    speech_started: Optional[float] = None
    last_voice: Optional[float] = None
    first_partial: Optional[float] = None
    turn_fired: Optional[float] = None
    asr_final: Optional[float] = None
    llm_sent: Optional[float] = None
    first_audio: Optional[float] = None
    playback_started: Optional[float] = None
    turn_complete: Optional[float] = None

    interim_transcript: str = ""
    final_transcript: str = ""
    used_final: bool = False       # did the flushed final actually get used?
    partial_count: int = 0
    audio_bytes: int = 0
    error: Optional[str] = None

    def mark(self, name: str, value: Optional[float] = None, overwrite: bool = False) -> None:
        """Record a timeline mark. First write wins unless overwrite=True.

        First-write-wins matters for ``first_partial`` and ``first_audio``,
        which sit inside loops that fire many times per turn.
        """
        if name not in MARKS:
            raise KeyError(f"unknown mark {name!r}; expected one of {MARKS}")
        if not overwrite and getattr(self, name) is not None:
            return
        setattr(self, name, time.perf_counter() if value is None else value)

    # ---- derived latencies -------------------------------------------------
    # Each of these answers a specific question from the tightening plan's
    # benchmark table. They return None when the inputs weren't observed
    # rather than guessing.

    @property
    def speech_duration_ms(self) -> Optional[float]:
        return _ms(self.speech_started, self.last_voice)

    @property
    def first_partial_ms(self) -> Optional[float]:
        """Speech start to first interim transcript."""
        return _ms(self.speech_started, self.first_partial)

    @property
    def vad_delay_ms(self) -> Optional[float]:
        """Last real speech to end-of-turn decision.

        Should be close to the configured silence window. Anything much
        larger is the partial-driven timer bumping described above.
        """
        return _ms(self.last_voice, self.turn_fired)

    @property
    def asr_final_ms(self) -> Optional[float]:
        """Cost of the end-of-stream flush."""
        return _ms(self.turn_fired, self.asr_final)

    @property
    def llm_ttfa_ms(self) -> Optional[float]:
        """Request sent to first audio byte back (network + model)."""
        return _ms(self.llm_sent, self.first_audio)

    @property
    def playback_offset_ms(self) -> Optional[float]:
        """First audio byte received to first byte handed to the device."""
        return _ms(self.first_audio, self.playback_started)

    @property
    def total_response_ms(self) -> Optional[float]:
        """THE headline number: end of caller speech to first AI audio.

        This is the figure the 1.5-3s target refers to, and the one that
        had never been measured before this module existed.
        """
        return _ms(self.last_voice, self.first_audio)

    @property
    def agent_speech_ms(self) -> Optional[float]:
        return _ms(self.first_audio, self.turn_complete)

    def to_dict(self) -> Dict:
        """JSONL record: derived latencies plus enough raw context to debug."""
        return {
            "call_id": self.call_id,
            "turn": self.turn_index,
            "kind": self.kind,
            "wall_clock": self.wall_clock,
            "latency_ms": {
                "speech_duration": self.speech_duration_ms,
                "first_partial": self.first_partial_ms,
                "vad_delay": self.vad_delay_ms,
                "asr_final": self.asr_final_ms,
                "llm_ttfa": self.llm_ttfa_ms,
                "playback_offset": self.playback_offset_ms,
                "total_response": self.total_response_ms,
                "agent_speech": self.agent_speech_ms,
            },
            "partial_count": self.partial_count,
            "audio_bytes": self.audio_bytes,
            "used_final": self.used_final,
            "interim_transcript": self.interim_transcript,
            "final_transcript": self.final_transcript,
            "error": self.error,
        }

    def summary_line(self) -> str:
        """One compact console line. Deliberately fits in a terminal width."""
        if self.kind != "caller":
            return (
                f"[latency] {self.kind} turn {self.turn_index} | "
                f"gemini_ttfa {_fmt(self.llm_ttfa_ms)} | "
                f"playback +{_fmt(self.playback_offset_ms)}"
            )

        parts = [f"[latency] turn {self.turn_index}"]
        vad = self.vad_delay_ms
        if vad is not None:
            parts.append(f"vad {vad:.0f}ms")
        if self.asr_final_ms is not None:
            parts.append(f"asr_flush {self.asr_final_ms:.0f}ms")
        if self.llm_ttfa_ms is not None:
            parts.append(f"gemini_ttfa {self.llm_ttfa_ms:.0f}ms")
        total = self.total_response_ms
        parts.append(f"TOTAL {_fmt(total)}" if total is not None else "TOTAL n/a")
        if self.used_final:
            parts.append("final=yes")
        return " | ".join(parts)


class MetricsLogger:
    """Collects TurnMetrics, writes JSONL, prints per-turn and session summaries.

    Never raises into the caller: instrumentation must not be able to end a
    call. Set ``enabled=False`` (or NEMO_AGENT_METRICS=0) to make every
    method a cheap no-op.
    """

    def __init__(
        self,
        call_id: str,
        log_dir: str = "client_logs",
        enabled: Optional[bool] = None,
        echo: bool = True,
        silence_window_ms: Optional[float] = None,
    ):
        if enabled is None:
            enabled = os.environ.get("NEMO_AGENT_METRICS", "1").strip().lower() not in ("0", "false", "no")
        self.enabled = bool(enabled)
        self.echo = echo
        self.call_id = call_id
        self.silence_window_ms = silence_window_ms
        self.turns: List[TurnMetrics] = []
        self._turn_counter = 0
        self._fh = None
        self._closed = False
        self.log_path: Optional[str] = None

        if not self.enabled:
            return

        try:
            os.makedirs(log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(log_dir, f"latency_{stamp}_{call_id}.jsonl")
            self._fh = open(self.log_path, "a", encoding="utf-8")
        except Exception as exc:                        # pragma: no cover
            print(f"[metrics] disabled -- could not open log file: {exc}")
            self.enabled = False

    def start_turn(self, kind: str = "caller") -> TurnMetrics:
        """Allocate a turn record. Cheap enough to call unconditionally."""
        self._turn_counter += 1
        return TurnMetrics(turn_index=self._turn_counter, call_id=self.call_id, kind=kind)

    def finish_turn(self, turn: Optional[TurnMetrics]) -> None:
        """Persist and report one turn. Call after playback, not during."""
        if turn is None or not self.enabled:
            return
        try:
            self.turns.append(turn)
            if self._fh is not None:
                self._fh.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")
                self._fh.flush()
            if self.echo:
                print(turn.summary_line())
        except Exception as exc:                        # pragma: no cover
            print(f"[metrics] failed to record turn: {exc}")

    def _caller_turns(self) -> List[TurnMetrics]:
        return [t for t in self.turns if t.kind == "caller"]

    def summary_table(self) -> str:
        """Markdown table of p50/p95 per stage, ready to paste into the plan's
        before/after benchmark table."""
        caller = self._caller_turns()
        if not caller:
            return "[metrics] no caller turns recorded."

        rows = (
            ("End-of-speech to first audio (TOTAL)", "total_response_ms"),
            ("VAD / end-of-turn delay", "vad_delay_ms"),
            ("ASR flush (final)", "asr_final_ms"),
            ("Gemini first audio", "llm_ttfa_ms"),
            ("Playback offset", "playback_offset_ms"),
            ("First partial after speech start", "first_partial_ms"),
        )

        lines = [
            "",
            f"=== Latency summary: {len(caller)} caller turn(s), call {self.call_id} ===",
            "",
            "| Stage | p50 | p95 | min | max | n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for label, attr in rows:
            vals = [v for v in (getattr(t, attr) for t in caller) if v is not None]
            if not vals:
                lines.append(f"| {label} | n/a | n/a | n/a | n/a | 0 |")
                continue
            lines.append(
                f"| {label} | {_percentile(vals, 50):.0f}ms | {_percentile(vals, 95):.0f}ms "
                f"| {min(vals):.0f}ms | {max(vals):.0f}ms | {len(vals)} |"
            )

        # Surface the inflation finding explicitly rather than making the
        # reader subtract two numbers themselves.
        if self.silence_window_ms is not None:
            vad_vals = [t.vad_delay_ms for t in caller if t.vad_delay_ms is not None]
            if vad_vals:
                over = _percentile(vad_vals, 50) - self.silence_window_ms
                lines += [
                    "",
                    f"Configured silence window: {self.silence_window_ms:.0f}ms. "
                    f"Median end-of-turn delay ran {over:+.0f}ms against it"
                    + (
                        " -- that overshoot is ASR partials bumping the timer."
                        if over > 40
                        else "."
                    ),
                ]

        finals = sum(1 for t in caller if t.used_final)
        lines.append(f"Flushed ASR final used on {finals}/{len(caller)} turns.")
        if self.log_path:
            lines.append(f"Per-turn records: {self.log_path}")
        return "\n".join(lines)

    def close(self) -> None:
        """Print the session summary and close the log file. Safe to call twice.

        Idempotent because the caller closes it both on the normal exit path
        and from a KeyboardInterrupt handler -- Ctrl+C is the common way a
        test call ends, and that is precisely when the summary is wanted.
        """
        if not self.enabled or self._closed:
            return
        self._closed = True
        try:
            if self.echo:
                print(self.summary_table())
        finally:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None
