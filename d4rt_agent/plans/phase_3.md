# Phase 3 — run the agent over the complete DSI-Bench `std` split

## 1. Goal

Produce one recorded answer per DSI-Bench question for all **1769** questions of
the `std` split, on a single GPU, resumable across as many jobs as it takes.

This phase collects results. It does **not** score them. See §6.

## 2. What changed from the 25-question run

Phase 2 evaluated a 25-question Latin-rectangle sample (`d4rt_agent/results/dsi_bench`).
Three things had to change to run the whole split.

### 2.1 A census manifest

`build_manifest(census=True)` takes the split whole instead of drawing from it.
There is no seed and no ground-truth balance scan, because nothing is being
selected. Questions are ordered by `relative_path` then `csv_row_index`, which
makes the order independent of how the benchmark wrote its CSV and puts the
questions sharing a video next to each other — 1769 questions span only 943
videos.

The manifest records `sampling.design = "census"`. `dsi_bench_run` refuses to
resume a `--all-questions` run against a `latin_rectangle` manifest, because 25
answers against the wrong manifest look exactly like a finished run.

**Answer files are keyed by `question_id`**, and the sample's guarantee of
distinct videos does not hold here: uniqueness now rests entirely on `cate`
differing whenever two questions share a video. That holds for today's CSV (1769
unique ids, verified), and `build_manifest` raises rather than let two questions
overwrite each other if it ever stops holding.

### 2.2 One attempt per question, ungated

The Phase 2 runner made two attempts: a strict one requiring the final answer to
cite a `query_d4rt` call, then — if that failed — a complete re-run with the
requirement dropped. At census scale that second step budget is hours of GPU
time spent on questions that often answer neither way. In the 25-question run
the three questions that used it cost 226 s, 258 s and 746 s.

There is now exactly one attempt, and by default it does not require the
citation (`--require-d4rt` restores the gate, still without a second attempt).
Statuses are `complete`, `failed`, `error`; `complete_relaxed` and
`gated_attempt` are no longer produced.

Because the gate no longer decides it, tracker use is **measured** and split in
two:

| Field | Meaning |
| --- | --- |
| `d4rt_used` | the final answer cites at least one `d4rt_*` evidence id |
| `d4rt_queried` | the run produced any D4RT measurement at all |

They come apart in the way that matters: a question with `d4rt_queried=true` and
`d4rt_used=false` is one where the agent measured and then answered without
reference to the measurement — not distinguishable from the tool-free control.
**Filter on these before drawing any conclusion from this run.**

### 2.3 One GPU, many jobs

`scripts/run_dsi_bench_blackwell.slurm --all-questions` requests a single
Blackwell GPU and walks the questions sequentially. Each answer is written the
moment it finishes and finished questions are skipped, so the run is resumed by
resubmitting the identical command. A job that dies mid-question repeats only
that question.

Executed rather than submitted, the launcher submits itself: `--all-questions`
selects the 24 h partition, the `_full` results directory and its own job name
and log file, none of which the `#SBATCH` header can decide, because sbatch
reads that header before the script body runs.

Per-job identity goes to `run_metadata_<jobid>.json` rather than a single
`run_metadata.json`, which a resumed job would overwrite. Every answer embeds
its own copy regardless.

Every job must stay on the Blackwell nodes (`--constraint=EPYC_9654`): the
comparison tooling refuses a results directory containing more than one GPU
model.

## 3. Cost

Extrapolated from job 9086647 (25 questions, A100 80GB, mean 128 s/question),
weighted by the census's own category mix:

| | |
| --- | --- |
| Agent pass | **~45–50 GPU-hours** (~60 h before the retry was removed) |
| Baseline pass | ~2 GPU-hours |
| Answers on disk | ~270 MB across 1769 JSON files |

The estimate rests on 4 questions per category, so treat it as ±50%.

## 4. Running it

```bash
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python

# once, on the login node: build the manifest and check every clip decodes
"$D4RT_PYTHON" -m d4rt_agent.dsi_bench_run --all-questions --dry-run \
    --results-dir d4rt_agent/results/dsi_bench_full

# then, as many times as it takes
POINT_MODE=ensemble5 ./scripts/run_dsi_bench_blackwell.slurm --all-questions
```

Run the launcher, do not `sbatch` it: executed, it picks the census allocation
and submits itself. It prints the pending count before queueing anything, so a
resubmission states up front how much is left.

Set `SUBMIT_PARTITION=cuda13pr.120h SUBMIT_TIME=120:00:00` to run to completion
in one submission instead of resuming across 24 h jobs.

## 5. Where the output lives

`d4rt_agent/results/dsi_bench_full/`, on `/cluster/work`. **Not `/cluster/scratch`** —
this directory is the only copy of ~45 GPU-hours of work.

The answers are deliberately **left uncommitted** for now. They are neither
tracked nor gitignored, so `git status` will list them; that is the reminder that
they exist and are not yet backed up by anything but the filesystem.

## 6. TODO — evaluation, deferred

**Nothing in this phase scores anything.** The agent does not currently emit a
clean option letter as its final answer, so any accuracy computed today would
measure the answer format rather than the reasoning. The answers are recorded in
full so that scoring can be applied retrospectively, once the format is settled.

Still to build, once it is:

- [ ] **Settle the final-answer format** so an option letter can be extracted
      reliably. This is the blocker for everything below. `parse_choice_letter`
      already handles the control's `<answer>X</answer>` and its common
      abbreviations; the agent's prose answer is the open case.
- [ ] **Measure the extraction rate first**, per dataset and per task, and report
      it beside any accuracy. An accuracy over 1769 questions means nothing
      without knowing how many yielded a letter at all.
- [ ] **Grouped aggregate report** — the deliverable for this run:
  - by **dataset** (`CameraBench`, `SynFMC`, `internet`, `k700`, `llava178k`)
  - by **task** (the 6 `cate` values)
  - dataset × task cross-tab
  - each with: n, status breakdown, `d4rt_used` / `d4rt_queried` rates, steps
    used, wall time, extraction rate, and — once available — accuracy against
    `gt` versus the tool-free baseline.
- [ ] **Report groups, not just a total.** The census inherits the benchmark's
      skew: `internet` alone is 891 of 1769 questions, and the six tasks range
      from 85 to 582. Any single overall number is dominated by the largest
      cells. Attach an interval to per-group accuracies — the smallest task cell
      has 85 questions.
- [ ] **Decide how unanswered questions count.** ~12% of the 25-question run
      ended with no answer inside the step budget. Report strict accuracy (over
      all questions) and answered-only accuracy separately; do not silently pick
      one.
- [ ] **Do not reuse `dsi_bench_report.py` unchanged.** It renders per-question
      media — at 1769 questions that is ~2.5 GB of GIFs, ~650 MB of contact
      sheets and a ~25 MB markdown file. Its header also hardcodes the string
      "25 questions". Aggregate first; render media only for a chosen subset.
- [ ] **`dsi_bench_compare.py` does not apply** to this run. Its
      `--previous-results` / `--failed-results` arguments and its hardcoded
      headline check (`len(order) == 25`) are specific to the Phase 2 study.

## 7. Deliberately not done

- **Clip reuse across consecutive questions.** The manifest groups questions by
  video, so 826 of 1768 consecutive pairs share a clip and could skip the decode
  and re-encode. Worth roughly 2% of total runtime — not worth the risk of
  carrying backend state across questions on a 45-hour run.
- **Sharding across GPUs.** `--shard` / `--num-shards` still work and would cut
  wall time proportionally, but this run is constrained to one GPU.
