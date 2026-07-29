# DSI-Bench forced-grounding experiment — findings

**Run:** the same fixed 25-question `std` subset, Qwen3-VL-8B-Instruct,
`point_mode=ensemble5`, seed 42, and at most 14 steps per attempt. The only
experimental change was the system prompt: before a `query_d4rt` action, Qwen was
asked to state a visible target, visual evidence, and a normalized 0–1000 box in
its thinking, then cite or reuse that grounding in the tool call.

Full answers, videos, refreshed grounding overlays, and complete saved traces:
[`DSI_BENCH_RESULTS.md`](DSI_BENCH_RESULTS.md).

## Headline

| System | Correct | Parsed / completed |
|---|---:|---:|
| Forced-grounding D4RT agent | **7 / 25 (28%)** | 21 / 25 |
| Previous committed D4RT agent | **10 / 25 (40%)** | 25 / 25 |
| Unchanged Qwen-only baseline | **13 / 25 (52%)** | 25 / 25 |

Failures are scored as incorrect. On only the 21 questions that returned an
answer, the forced-grounding agent scored 7/21 (33%).

This prompt-only intervention did **not** improve DSI-Bench performance. The new
agent gained four questions that the old agent missed, but lost seven that the
old agent got right; only three were right in both runs. The paired difference
is not statistically resolved at this sample size (exact McNemar p=0.549), but
the point estimate moved in the wrong direction. The new 7/25 result is also
consistent with four-way guessing (one-sided binomial p=0.439).

## What changed question by question

| Outcome versus previous agent | Questions |
|---|---|
| Newly correct | **1, 13, 19, 23** |
| Previously correct, now wrong or failed | **2, 4, 10, 12, 14, 22, 24** |
| Correct in both | **5, 9, 16** |
| Wrong in both | remaining 11 |

The most encouraging examples are #1 and #23. In both, explicit grounding
changed an old wrong answer into the correct forward-motion answer. For #1
(runner), the declared box `[447,350,475,440]` exactly matches the box from the
standalone Qwen grounding probe.

The counterweight is substantial: three of the four unanswered rows (#2, #14,
and #22) were correct in the previous agent run. The intervention therefore
removed three previously correct answers through orchestration failure alone.

## Per task

| Task | n | Forced grounding | Previous agent | Qwen-only |
|---|---:|---:|---:|---:|
| Obj:static cam | 4 | **3** | 0 | 2 |
| Obj:moving cam | 4 | 0 | **2** | **2** |
| Cam:static scene | 4 | 0 | 0 | **1** |
| Cam:dynamic scene | 4 | 1 | **3** | **3** |
| Obj-Cam distance | 5 | **3** | 2 | **3** |
| Obj-Cam orientation | 4 | 0 | **3** | 2 |

The prompt appears useful for the narrow `Obj:static cam` cell, which improved
from 0/4 to 3/4, but that gain was more than erased by `Obj-Cam orientation`,
`Obj:moving cam`, and `Cam:dynamic scene`. With only 4–5 examples per cell, this
is a diagnostic pattern rather than a task-level benchmark claim.

## Did Qwen follow the grounding protocol?

Across the strict attempts and the four automatic relaxed retries, the run saved
272 model actions and 105 `query_d4rt` actions.

- Every tool call's justification cited a grounding identifier: **105/105**.
- Only **63/105 (60%)** tool-call responses actually contained the requested
  `GROUNDING`/`REUSE GROUNDING` line, visual evidence, and a box matching the
  action coordinates. When the declaration was present, its coordinates matched
  the call exactly.
- Those 105 calls represented only **36 unique**
  `(question, source frame, label)` keys. Repeated keys almost always reused an
  identical box, but often without emitting the explicit reuse declaration.
- Qwen emitted multiple JSON actions in **102/272 (37.5%)** model responses even
  though the prompt requested one action. The host recorded **149 rejected
  actions**, including 95 invalid `python_math` actions. The previous run had
  only 24 rejected actions.

So a system-prompt instruction was not strong enough to make the protocol
reliable. More importantly, it affected behavior outside grounding: action
formatting and termination became much worse.

## Comparison with the standalone Qwen grounding probe

The standalone debug run used labels from the previous trace, while the new
agent sometimes chose different labels and source frames. There are 22 exact
`(question, source frame, label)` matches with non-null boxes between the two
runs:

- median IoU between new on-the-fly and standalone boxes: **0.263**;
- boxes with IoU at least 0.5: **2/22**;
- median normalized center distance: **137** on the 0–1000 coordinate scale.

This matched subset agrees more than the old trace did with the standalone
detector (the old global comparison had median IoU 0.013), but the denominators
differ and neither comparison has ground-truth boxes. It is evidence that
thinking visibly before the call can alter—and occasionally fix—the box, not
evidence that the boxes are generally correct. Most matched boxes still differ
substantially from the one-image detector.

## Why the run took almost three hours

The agent pass took **176.6 minutes**, versus 42.7 minutes reported for the
previous pass (about **4.1× slower**). Four failed questions consumed 110.0
minutes, or 63% of total per-question time.

Each failed question exhausted a strict 14-step attempt and then automatically
ran another 14-step relaxed attempt. Two failures repeated one identical box
14 times in each attempt (`person` in #2 and `police car` in #14); #22 repeated
the same `harvester` grounding six times per attempt. Reprocessing all 32 images
with an increasingly long context made these loops especially expensive.

The automatic relaxed retry is a poor fit for this failure mode. It was designed
to recover a valid visual-only answer when the D4RT requirement blocked an
otherwise finished response; here it simply repeated a non-terminating action
pattern and approximately doubled the cost of each failed row.

## Conclusion

The experiment refines the original diagnosis:

1. **On-the-fly grounding can be a real failure source.** Explicit visible
   grounding fixed compelling individual cases, especially the runner in #1.
2. **Prompt-only forced grounding is not a solution.** It improved box agreement
   on a limited matched subset but reduced end-task accuracy from 10/25 to 7/25.
3. **The larger failure is now orchestration control.** The model did not
   reliably obey declaration/reuse or single-action constraints, repeated
   identical queries, and failed to terminate on 4/25 questions.

The next clean experiment should enforce grounding in the host rather than in
free-form thinking: parse and validate exactly one declaration, cache it by
`(frame, label)`, reject duplicate identical queries without another model
generation, and do not launch the relaxed retry when the strict attempt failed
by step exhaustion. That would isolate box quality without introducing the
large action-format and runtime confound seen here.
