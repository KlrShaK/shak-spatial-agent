# 15 — Evaluation protocol

Theme 6 of the discussion. Status: **settled** on 2026-10-17 (decision log #69–76), except the
quick-check selection details at the end, which await your OK.

## What went wrong before

- Pilots were promoted on format success alone. A workflow whose repair step never ran became a
  subset campaign.
- Baseline artifacts drifted: three values were quoted for the same D7 flow.
- The share's file-count quota killed the compositional Dyn run. V5 wrote about 100 files per
  question.
- Bundled changes could not say which change helped or hurt.

## Comparisons (#69)

No new control runs. V6 is compared with published Qwen numbers and with our existing prediction
sets on identical question IDs:

| Benchmark | Published Qwen | Our existing predictions on the same questions |
|---|---|---|
| DSI | Qwen3-VL-8B + DynTrace 56.4 (sample-wise); our tool-free Qwen3-VL-8B control 41.5 on 200 × 4 | V4 (full census and 200 × 4), V5 legacy (800 std), V5 compositional (63 × 4) |
| Dyn | Qwen3-VL-8B 61.4 (Dyn-Bench paper, full 7,000); our official-protocol rerun 60.6 on the 2,800 split | V5 legacy Point4D and D4RT, D7 flow, frames-only and option log-probabilities (2,800) |
| VLM4D | **No Qwen3-VL-8B row in the paper.** Nearest: Qwen2.5-VL-7B 43.8, Qwen2.5-VL-72B 53.0; our video-blind text prior 47.2 | V5 legacy (all 1,816), V5 compositional (300); the paper's 22 stored models (all 1,816) |

## Question sets

**Quick-check set (#70).** 200–300 questions from all three benchmarks, drawn from questions that
previous systems have already answered. Every V6 iteration can then be compared answer by answer
with V4 and V5, including their failure cases. It also gives a first estimate of performance.

**Full runs, in priority order (#71).**

1. DSI: V4's 200-question draw × 4 renderings (800 records). V4 and the tool-free control exist on
   all of them.
2. Dyn: the 2,800-question split. V5, D7, frames-only and official-protocol Qwen exist on all of
   them.
3. VLM4D: a seeded random 1,000 of the 1,816. V5 legacy and the paper's 22 stored models exist on
   all of them.

Rough cost at V5's rates (28–36 s per question; V6 may be slower): about 6–8 GPU hours for DSI,
22–28 for Dyn and 8–10 for VLM4D.

## Gate from quick-check to full runs (#72)

All of these must hold on the quick-check set:

- V6 beats V5 and frames-only Qwen (where that exists) with a positive lower bound on the paired
  clip-bootstrap interval;
- no quantity group ([06](06_question_census.md)) falls more than 5 points below the best existing
  system on that group;
- on DSI, the reversed renderings are not below V4 on the same IDs.

With about 200 question IDs, a positive lower bound needs a gain of roughly 6–8 points; the workflow
pilot's interval on 168 questions was about ±5. That is a deliberately high bar.

## Ablations (#74)

One change at a time, on the quick-check set, cheapest first. **Each run needs your approval before
it is submitted.**

1. Answer turn: precedence rule, answer format, no added frames ([14](14_answering.md)).
2. Report format: V4-style headline and phases versus V5-style tables ([10](10_tool_reports.md)).
3. Plan versus no plan ([12](12_planning.md)).
4. Normaliser and doctor on/off ([13](13_tool_calls.md)).
5. Caption on/off.
6. Facing fallback on/off.
7. Tool visibility: all tools visible during act turns (#79) versus menus scoped by the current
   item's category. Run only if traces show frequent calls unrelated to the current plan item.

Items 1 and 2 can replay cached measurements and rerun only Qwen, so they need no geometry jobs.

## Reported with every run (#75)

- **Accuracy.** Per benchmark (DSI sample-wise and ≥3/4 group-wise), per quantity group (06), and
  per official breakdown (DSI category, Dyn level, VLM4D real/synthetic).
- **Process.** Unavailable rate; normaliser fixes, doctor fixes and calls sent back; plan adherence
  (items executed versus planned); flip rate (`because` against the chosen option); answer-letter
  distribution.
- **Cost.** Seconds per question, model calls, tokens, peak GPU memory.
- **Paired comparisons** against every existing prediction set on the same IDs, clustered by source
  clip, with DSI renderings of one question kept together.

## Run hygiene (#73, #76, #78)

- **Races are allowed for V6 evaluation runs (#73):** submit to A100 and Blackwell, keep whichever
  starts first, and cancel the other at once. This is a project exception to the "no partition
  races" rule in your global instructions. I keep at most one race pair queued at a time.
- **Jobs and progress (#78):** the 2-running + 1-pending limit applies to GPU jobs only; CPU-only
  SLURM jobs are not limited by it. Nothing computationally heavy runs on a login node. While a
  quick test, an evaluation or any job runs, I watch it with the Monitor tool and send you a progress
  update every 30 minutes.
- **Frozen source snapshot and config hash per run**, as V5's `research.snapshot` did. A results
  directory is never reused across variants.
- **File budget:** one answer JSON per question; caches packed per clip (one archive per clip and
  backbone instead of per-question frame links). Target: well under 10 files per question.

## To confirm: quick-check selection

My proposal for #70:

| Benchmark | Pool (questions with previous predictions) | Take |
|---|---|---|
| DSI | the 63 IDs of the compositional subset: V4, V5 compositional and the tool-free control exist in all 4 renderings | 40 IDs × 4 = 160 records |
| Dyn | the 2,800 split: V5 Point4D and D4RT, D7, frames-only, letters | 70 |
| VLM4D | the 300-question compositional subset: V5 legacy and V5 compositional | 70 |

That is 180 question IDs and 300 records. Within each benchmark, questions are drawn at random per
quantity group (06), **not by whether earlier systems got them wrong**. Picking known failures makes
any new system look better through regression to the mean. A random draw from these pools still
contains plenty of failures, since earlier systems missed 40–55% of them.
