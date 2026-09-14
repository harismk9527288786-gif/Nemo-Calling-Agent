"""Compare latency between two instrumented calls.

Why this exists: ``bench_pipeline.py`` characterises *components* (the ASR
engine in isolation, Gemini Live in isolation). It drives ``nemo_asr``
directly and never touches ``AsrWorker``, so its numbers are by construction
IDENTICAL before and after step 3. The restructuring work only shows up in
end-to-end per-turn metrics from a real call, which ``pipeline_metrics``
writes to ``client_logs/*.jsonl``. This tool diffs two of those.

    python compare_runs.py --list
    python compare_runs.py before.jsonl after.jsonl
    python compare_runs.py --before client_logs/latency_A.jsonl \
                           --after  client_logs/latency_B.jsonl

Only latency fields are read. Transcripts in the JSONL are deliberately never
printed -- captured call audio content should not end up in a pasted table.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

# Reuse the logger's own percentile so "p50" means the same thing in the live
# summary and in this comparison. pipeline_metrics is stdlib-only, so importing
# it here costs nothing and cannot drag in numpy or the genai client.
try:
    from pipeline_metrics import _percentile
except Exception:  # pragma: no cover - keeps the tool usable standalone
    def _percentile(values, pct):
        if not values:
            return None
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
        return round(ordered[idx], 1)

# (label, key in the JSONL latency_ms object, lower_is_better)
METRICS = (
    ("End-of-speech to first audio (TOTAL)", "total_response", True),
    ("VAD / end-of-turn delay", "vad_delay", True),
    ("ASR flush (final)", "asr_final", True),
    ("Gemini first audio", "llm_ttfa", True),
    ("Playback offset", "playback_offset", True),
    ("First partial after speech start", "first_partial", True),
)

# Below this many caller turns, run-to-run noise swamps any real change.
MIN_USEFUL_TURNS = 5


def load_run(path: str) -> Dict:
    """Read one JSONL metrics file into caller-turn latency lists."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    values: Dict[str, List[float]] = {k: [] for _, k, _ in METRICS}
    caller_turns = 0
    used_final = 0
    partials: List[int] = []
    errors = 0
    malformed = 0
    call_ids = set()

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            # Greeting and nudge turns have no caller speech, so including them
            # would pollute every end-of-speech measurement.
            if rec.get("kind") != "caller":
                continue
            caller_turns += 1
            call_ids.add(rec.get("call_id", "?"))
            if rec.get("error"):
                errors += 1
            if rec.get("used_final"):
                used_final += 1
            if isinstance(rec.get("partial_count"), int):
                partials.append(rec["partial_count"])
            lat = rec.get("latency_ms") or {}
            for _, key, _ in METRICS:
                v = lat.get(key)
                if isinstance(v, (int, float)):
                    values[key].append(float(v))

    return {
        "path": path,
        "values": values,
        "caller_turns": caller_turns,
        "used_final": used_final,
        "partials": partials,
        "errors": errors,
        "malformed": malformed,
        "call_ids": sorted(call_ids),
    }


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.0f}ms"


def _delta(before: Optional[float], after: Optional[float], lower_is_better: bool) -> str:
    """Signed change plus a plain-language verdict."""
    if before is None or after is None:
        return "n/a"
    diff = after - before
    if abs(before) > 1e-9:
        pct = f" ({diff / before * 100.0:+.0f}%)"
    else:
        pct = ""
    if abs(diff) < 1.0:
        return f"{diff:+.0f}ms{pct} same"
    improved = (diff < 0) if lower_is_better else (diff > 0)
    return f"{diff:+.0f}ms{pct} {'better' if improved else 'WORSE'}"


def render(before: Dict, after: Dict) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=== Latency comparison ===")
    lines.append(f"  before: {before['path']}  ({before['caller_turns']} caller turns)")
    lines.append(f"  after:  {after['path']}  ({after['caller_turns']} caller turns)")
    lines.append("")
    lines.append("| Stage | before p50 | after p50 | change p50 | before p95 | after p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for label, key, lower_better in METRICS:
        b = before["values"][key]
        a = after["values"][key]
        b50, a50 = _percentile(b, 50), _percentile(a, 50)
        b95, a95 = _percentile(b, 95), _percentile(a, 95)
        lines.append(
            f"| {label} | {_fmt(b50)} | {_fmt(a50)} | {_delta(b50, a50, lower_better)} "
            f"| {_fmt(b95)} | {_fmt(a95)} |"
        )

    lines.append("")
    for tag, run in (("before", before), ("after", after)):
        n = run["caller_turns"]
        final_rate = f"{run['used_final']}/{n}" if n else "0/0"
        med_partials = _percentile(run["partials"], 50) if run["partials"] else None
        lines.append(
            f"  {tag}: flushed final used {final_rate} turns"
            + (f", median {med_partials:.0f} partials/turn" if med_partials is not None else "")
            + (f", {run['errors']} errored turn(s)" if run["errors"] else "")
            + (f", {run['malformed']} malformed line(s)" if run["malformed"] else "")
        )

    # Honesty guards: say when the comparison cannot support a conclusion.
    warnings: List[str] = []
    for tag, run in (("before", before), ("after", after)):
        if run["caller_turns"] == 0:
            warnings.append(
                f"{tag} run has NO caller turns -- nothing to compare. Did the call "
                "capture any speech?"
            )
        elif run["caller_turns"] < MIN_USEFUL_TURNS:
            warnings.append(
                f"{tag} run has only {run['caller_turns']} caller turn(s); under "
                f"{MIN_USEFUL_TURNS} the p50 is dominated by run-to-run noise. "
                "Treat any delta as indicative, not measured."
            )
    if warnings:
        lines.append("")
        for w in warnings:
            lines.append(f"  WARNING: {w}")

    lines.append("")
    lines.append(
        "  Note: Gemini first-audio depends on network and server load, so it "
        "varies between runs for reasons unrelated to local code."
    )
    return "\n".join(lines)


def list_runs(log_dir: str) -> int:
    paths = sorted(glob.glob(os.path.join(log_dir, "latency_*.jsonl")))
    if not paths:
        print(f"No metrics files in {log_dir}/. Place a call first -- the agent "
              "writes one file per run.")
        return 1
    print(f"Metrics files in {log_dir}/ (oldest first):\n")
    for p in paths:
        try:
            run = load_run(p)
            n = run["caller_turns"]
            p50 = _percentile(run["values"]["total_response"], 50)
            print(f"  {os.path.basename(p):<44} {n:>3} caller turns   "
                  f"TOTAL p50 {_fmt(p50)}")
        except Exception as exc:
            print(f"  {os.path.basename(p):<44} unreadable ({exc})")
    print("\nCompare two of them:\n  python compare_runs.py "
          "client_logs/<before>.jsonl client_logs/<after>.jsonl")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diff end-to-end latency between two instrumented calls."
    )
    ap.add_argument("positional", nargs="*", metavar="BEFORE AFTER",
                    help="Two JSONL metrics files")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--log-dir", default="client_logs")
    ap.add_argument("--list", action="store_true",
                    help="List available metrics files and exit")
    args = ap.parse_args()

    if args.list:
        return list_runs(args.log_dir)

    before = args.before
    after = args.after
    if not (before and after) and len(args.positional) == 2:
        before, after = args.positional
    if not (before and after):
        ap.error("need two files: BEFORE AFTER (or --before/--after, or --list)")

    try:
        b = load_run(before)
        a = load_run(after)
    except FileNotFoundError as exc:
        print(f"No such metrics file: {exc}. Try --list.")
        return 1

    print(render(b, a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
