# Baseline capture runbook

**Purpose:** fill in the before/after benchmark table in `PHASE0_AUDIT_AND_PLAN_24_08_2026.md` with real numbers from this machine. Nothing in a sandbox can produce these — the DLL, the GGUF model and the Gemini key exist only here.

**Read this first, because it corrects an instruction given earlier.** `bench_pipeline.py` characterises *components in isolation*: its ASR stage drives `nemo_asr` directly and never touches `AsrWorker`, and its Gemini stage opens its own Live session. Neither is affected by step 3, so running it with `--label "before step 3"` would produce a number identical to the "after" and imply a comparison it cannot make. Use the component benchmark **once**, to establish the per-stage budget. Step 3's actual effect appears only in the end-to-end per-turn metrics from a real call, which `pipeline_metrics` writes to `client_logs/*.jsonl` and `compare_runs.py` diffs.

So there are two separate exercises below, and they answer different questions.

## Part 1 — component budget (run once, on `main`)

This tells you where the ~1.0–1.3 s is actually spent, independent of any restructuring. From the repo root:

```
python bench_pipeline.py --stage asr --wav test.wav
python bench_pipeline.py --stage gemini --runs 5
```

Substitute any tracked wav for `test.wav` if that file is absent; `--stage asr` prints which file it used and its duration. Two things invalidate an ASR run and are worth checking in the output before trusting it. If `pacing worst late` exceeds one chunk (80 ms), the scheduler was too coarse to pace audio properly and the latency figures are noise — rerun with the machine otherwise idle. If the Gemini stage prints the `WARNING: could not import agent.load_system_instruction` fallback, it measured a short stand-in prompt rather than your real knowledge-base prompt, and its time-to-first-audio will be optimistic; fix the import or pass `--kb <your kb>.json` before recording the number.

Record `first partial`, `final flush`, `RTF` and the Gemini `ttfa median` in the plan's table. The RTF figure is the one to watch on this CPU: it sums push and drain, and anything at or above 1.0 means there is no headroom for decode to catch up after a stall.

## Part 2 — end-to-end before/after for step 3

This is the real comparison. It needs one call on the pre-step-3 code and one on current `main`, and the two must be made as similar as possible.

First make sure nothing is uncommitted, because you are about to move HEAD:

```
git status --short
```

If that prints anything you care about, commit or `git stash` it. Then capture the "before" call on the last commit that has instrumentation but not the worker thread:

```
git checkout d01f2e5
python agent.py --kb knowledge_bases/sk_furniture.json
```

Expect `asr_worker.py`, `test_asr_worker.py` and `compare_runs.py` to disappear from the folder while you are on that commit. That is normal — they are committed in `efde9e5` and `git checkout main` brings them back. Nothing is lost.

Hold a short conversation, then end it with Ctrl+C. The summary table prints on exit and the per-turn records land in `client_logs/`, which is gitignored and untracked, so the file survives the checkout back to `main`. Note the filename it prints on the last line.

Now return to current code and repeat:

```
git checkout main
python agent.py --kb knowledge_bases/sk_furniture.json
```

Say **the same utterances in the same order** as the first call. This is the main threat to the comparison: turn latency depends on how long you speak and how much silence you leave, so an unmatched pair of calls measures your speaking pattern rather than the code. Aim for at least six caller turns in each; below five, `compare_runs.py` will tell you the median is noise-dominated, and it means it.

Then diff them:

```
python compare_runs.py --list
python compare_runs.py client_logs/<before-file>.jsonl client_logs/<after-file>.jsonl
```

## Reading the result

The row that matters is **End-of-speech to first audio (TOTAL)**; that is the figure the 1.5–3 s target refers to. Step 3 should not move it much on a quiet machine, because on an idle system the event loop was not starved to begin with. What step 3 buys is headroom, and the place it shows is the `=== ASR decode thread ===` block on exit: a zero drop count with peak backlog well under the 4 s bound means decode is keeping up. If you see dropped chunks, or the "peak backlog used over half the bound" note, decode is losing the race at ~1.03× real time and the transcript has gaps.

Two other rows are diagnostic rather than performance. **VAD / end-of-turn delay** should sit near the configured 300 ms; the summary prints the overshoot explicitly, and a large positive overshoot is ASR partials bumping the turn timer. Step 3 does not fix that — it is step 5's job — so expect this row to stay high. **Gemini first audio** varies with network and server load between runs for reasons unrelated to local code, so treat a change there as weather, not signal, unless it is very large.

`Flushed ASR final used N/N turns` should read every turn. If it does not, the flush is failing and the agent is falling back to interim text, losing punctuation and inverse text normalisation.
