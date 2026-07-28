# DSI-Bench pilot — findings

**Run:** 25 questions, `std` split, Qwen3-VL-8B-Instruct, `point_mode=ensemble5`,
`benchmark_scale=1.0` (`identity_no_ground_truth_scale`), seed `20260725`,
max 14 steps/question. Agent wall time 42.7 min for the 25-question pass.

Per-question detail, videos and full traces: [`DSI_BENCH_RESULTS.md`](DSI_BENCH_RESULTS.md).
All letters below are re-derived from saved response text with `parse_choice_letter`;
0/25 agent and 0/25 baseline answers failed to parse.

## Headline

| | correct | vs. 25% chance |
|---|---|---|
| D4RT agent | **10 / 25** (40%) | p = 0.071 — **not** distinguishable from guessing |
| Qwen-only baseline | **13 / 25** (52%) | p = 0.003 — distinguishable |

> **The D4RT loop is a net negative on this sample.** It is also worth being precise
> about what "beats chance" means here: at n=25 and 4 options you need **≥11 correct**
> for a one-sided binomial test to clear p<0.05. The baseline clears that bar; the agent
> does not. So the honest reading is that the *baseline* beats chance, the *agent* does
> not, and the agent-vs-baseline gap itself is **not** statistically resolved.

**Why the gap isn't resolved.** The two systems answer the same 25 questions, so the
paired test is the right one. They agree on 12 questions and disagree on 13, but only
7 of those disagreements are decisive: the agent is right where the baseline is wrong on
**2**, the baseline right where the agent is wrong on **5**. McNemar's exact test on
(2, 5) gives **p = 0.45**. Eight questions both get right, ten both get wrong. A
3-question headline difference resting on a 5-vs-2 split is noise-consistent.

## Per task

| Task | n | Agent | Baseline |
|---|---|---|---|
| Cam:dynamic scene | 4 | 3 | 3 |
| Cam:static scene | 4 | **0** | 1 |
| Obj-Cam distance | 5 | 2 | 3 |
| Obj-Cam orientation | 4 | **3** | 2 |
| Obj:moving cam | 4 | 2 | 2 |
| Obj:static cam | 4 | **0** | 2 |

The agent's two 0/4 cells are `Cam:static scene` and `Obj:static cam`; its only win is
`Obj-Cam orientation` (3/4 vs 2/4) — the task predicted in the plan to be its **weakest**,
since orientation needs the object's own body frame and D4RT returns no orientation.

**That inversion is the thing to read traces for, not the headline** — but temper how
surprising the 0/4s are. Under pure guessing a single 4-question cell comes back 0/4 with
probability 0.75⁴ = **32%**, and across 6 tasks the chance that *at least one* cell is 0/4
is **~90%**. Two 0/4 cells out of six is only mildly unlucky. Per-task cells at n=4 cannot
support a claim about which task is hard; they can only tell you where to look.

## Per dataset

| Dataset | n | Agent | Baseline |
|---|---|---|---|
| CameraBench | 5 | 3 | 4 |
| SynFMC | 5 | 2 | 2 |
| internet | 5 | 2 | 1 |
| k700 | 5 | 1 | 2 |
| llava178k | 5 | 2 | 4 |

## Head-to-head: where they agreed and where they split

Twelve of 25 land on the same letter, thirteen split. `inv` is invisible/requested target
frames for that question; `st` is agent steps.

### Agent right, baseline wrong — 2

| Task | Dataset | Question | GT | Agent | Baseline | inv | st |
|---|---|---|---|---|---|---|---|
| Obj-Cam orientation | internet | observer's position vs. the car's own orientation | **D** from car's front to back | D ✓ | A from car's left to back | 4/5 | 6 |
| Obj:moving cam | llava178k | the man's location vs. his starting orientation | **A** Moving downward | A ✓ | B Moving upward | 0/2 | 12 |

The llava178k row is the cleanest win in the run: the baseline picked the **exact opposite**
direction, and the agent had 2/2 targets resolve. A sign flip on the vertical axis is
precisely what a metric depth measurement should fix and a language prior should not, so
this is the one row where the tool plausibly did the work the design intended.

### Baseline right, agent wrong — 5

| Task | Dataset | Question | GT | Agent | Baseline | inv | st |
|---|---|---|---|---|---|---|---|
| Obj:static cam | CameraBench | the runner's location vs. starting orientation | **B** Moving forward | A Moving backward | B ✓ | 0/4 | 5 |
| Obj:moving cam | k700 | the horse's location vs. starting orientation | **D** Moving forward | A Moving backward | D ✓ | 0/4 | 5 |
| Obj:static cam | llava178k | the car's location vs. starting orientation | **B** Moving forward | C Moving backward | B ✓ | 1/4 | 3 |
| Cam:static scene | llava178k | the observer's location vs. starting orientation | **B** Orbiting ccw | C Moving forward | B ✓ | 0/2 | 4 |
| Obj-Cam distance | llava178k | distance observer↔car | **D** Get closer | C *Cannot be determined* | D ✓ | 11/12 | 14 |

**Three of these five are the same error: forward/backward inverted.** Runner, horse and
car — GT forward, agent backward, baseline forward, and in all three *every or nearly every*
target frame resolved successfully. So this is not tracker dropout. It is the sign convention
on the facing axis: the agent builds a forward direction from two grounded body points
(rear→front) and projects displacement onto it, and if that axis comes out reversed the
answer flips deterministically. That is a **specific, checkable, fixable bug** and it is worth
more attention than the aggregate score — it accounts for 3 of the agent's 15 errors, and
fixing it alone would move the agent from 10/25 to 13/25, level with the baseline.

### Both right — 8

| Task | Dataset | GT | inv |
|---|---|---|---|
| Obj:moving cam | CameraBench | C Moving forward | 25/38 |
| Cam:dynamic scene | CameraBench | D Orbiting counterclockwise | 4/8 |
| Obj-Cam distance | CameraBench | A Get farther | 10/11 |
| Obj-Cam distance | SynFMC | B Get farther | 1/2 |
| Obj-Cam orientation | SynFMC | C from man's right to back | 0/4 |
| Cam:dynamic scene | internet | C Moving upward | 5/11 |
| Cam:dynamic scene | k700 | B Moving forward | 1/4 |
| Obj-Cam orientation | llava178k | B from harvester's left to front | 1/4 |

Note the invisibility column: the both-right rows carry a **57%** invisible-target rate
(47/82), the same as the run overall, and two of them (25/38 and 10/11) are the worst rows
in the entire study. The agent reached the right answer on questions where its measurements
largely failed — i.e. from the images, the same way the baseline did. **Agreement here is
weak evidence that the tool contributed anything**, and these 8 rows should not be counted
as tool successes without reading the traces.

### Both wrong, same letter — 4

| Task | Dataset | GT | Both chose | inv |
|---|---|---|---|---|
| Obj:moving cam | SynFMC | D Moving forward and turning left | C Moving backward and turning right | 0/4 |
| Cam:static scene | SynFMC | A Orbiting clockwise | B Moving backward | 10/20 |
| Cam:dynamic scene | SynFMC | A Moving upward | D Moving right | 10/18 |
| Obj:static cam | k700 | A Moving forward-right | C Moving backward-right | 0/4 |

**Three of the four are SynFMC** (the synthetic set), and the run's SynFMC score is 2/5 for
both systems. The first and last rows are again **forward↔backward inversions** with full
target visibility — the same failure as the section above, except here the baseline shares
it. Counting those, forward/backward sign errors appear in **5 of the 15 agent errors**.
These four rows are where the tool loop had a clean shot — targets resolved, budget spare —
and moved the answer nowhere.

### Both wrong, different letters — 6

| Task | Dataset | GT | Agent | Baseline | inv |
|---|---|---|---|---|---|
| Cam:static scene | CameraBench | C Moving downward | A Moving forward | B Moving upward | 10/11 |
| Obj:static cam | internet | B Orbiting ccw | C *Basically unchange* | A Orbiting cw | 7/8 |
| Cam:static scene | internet | B Moving upward | D Moving forward-right | C Moving downward | 1/2 |
| Obj-Cam distance | internet | D Get farther | A *Cannot be determined* | C Get closer | 1/2 |
| Obj-Cam distance | k700 | A Remain unchanged | C *Cannot be determined* | D Get closer | 1/2 |
| Obj-Cam orientation | k700 | C from skier's left to right | D from skier's front to right | B from skier's back to front | 0/4 |

### The pattern that explains the split: the agent hedges, the baseline never does

Across the 25 questions a hedge option — *"Cannot be determined"*, *"Remain unchanged"*,
*"Basically unchange"* — is offered in 7. **The agent picked one 4 times. The baseline picked
one 0 times. All 4 of the agent's were wrong.**

Narrowing to *"Cannot be determined"* specifically: offered in **5** questions, chosen by the
agent in **3**, by the baseline in **0**, and **it is never the ground truth in any of the 5**.

This is the single clearest mechanism behind the inversion, and it is not a 3D-reasoning
failure — it is an **answer-policy mismatch**. The tool loop hands the agent an honest signal
the baseline never receives ("my measurement failed / the displacement is within the noise"),
DSI-Bench's option lists offer somewhere to put that honesty, and the benchmark never scores
it as correct. The llava178k distance row is the archetype: 11 of 12 target frames came back
invisible, the agent burned all 14 steps, concluded *Cannot be determined*, and the baseline
— which never attempted a measurement and so had nothing to doubt — simply said *Get closer*
and was right.

Three of the agent's 15 errors are this. Combined with the 5 forward/backward sign errors,
**8 of 15 agent errors fall into two specific, named, fixable failure modes**, neither of
which is "the 3D tracker doesn't help with 3D reasoning."

> **Benchmark artifact, worth flagging upstream.** The internet `Obj:static cam` row above
> offers `C = "Basically unchange"` and `D = "Basically unchange"` — two identical options,
> with GT `B`. The agent picked C. That question is unanswerable as posed and one of the 25
> should be treated as void.

## What the traces actually show

Measured over all 135 `query_d4rt` calls and their 190 requested target frames:

- **54% of requested target frames came back invisible** (103/190). 17 of 25 questions hit
  at least one invisible target. This is the dominant failure mode of the tool loop — far
  more than tracker imprecision.
- **Depth precision, when a point *is* visible, is mostly fine.** Across 87 visible
  predictions the `ensemble5` spread `std_z/|z|` has median **0.02** (p25 0.01, p75 0.04).
  So the usual complaint — "the noise floor swamps the signal" — is **not** generally true.
  But the tail is heavy: **15%** of visible predictions have depth spread >20% of depth,
  and among the 7 queries where a start→end displacement was measurable against its own
  spread, 4 had **displacement smaller than the spread**. The tracker is usually precise
  and occasionally badly wrong, which is harder to reason around than uniform noise.
- **Occlusion is not what separates right from wrong.** Questions with ≥1 invisible target
  scored 8/17; questions where every target resolved scored 2/8. The clean ones are mostly
  the shallow 2-query rows, so this is confounded — but it does rule out "the agent fails
  because the tracker drops out" as the whole story.
- 24 actions were rejected by the host validator across the run; 194 model actions total.
  Mean 7.8 steps/question, 4 questions hit 12+ steps.

**Reading the `limitations` fields is the fastest route in.** The agent frequently
*announces* that its measurement failed and then commits to a letter anyway:

- *"The static background point was not visible in any of the queried frames, so the
  camera's motion was inferred from the changing perspective on the scene rather than
  direct measurement."* (Cam:static scene, CameraBench — 11 queries, 10 of 11 targets
  invisible, wrong)
- *"The camera's motion is primarily leftward and slightly backward, which does not match
  any of the options directly. 'Moving forward' is the closest option, but the motion is
  actually backward."* (Cam:static scene, llava178k — all targets visible, wrong)
- *"The car is not visible in frame 31, so the displacement could not be measured. The
  measurement was incomplete and the answer is based on the images showing the car moving
  forward."* (Obj:static cam, internet — wrong)

In those rows the agent has fallen back to the *same* visual evidence the baseline uses,
but after spending 3–12 steps on a failed measurement and with the failed measurement
still in context. That is a plausible mechanism for the loop being a net negative: it
doesn't add a channel so much as add a distractor. Confirming it means reading whether the
fallback answers are *worse* than the baseline's on the same rows, not just checking the
tally.

## Three worked traces

Reproduce any of these with:

```
python d4rt_agent/dsi_bench_show.py M0jmSsQ5ptw     # video path or question id substring
python d4rt_agent/dsi_bench_show.py --list          # all 25
```

### 1. `CameraBench/M0jmSsQ5ptw.3.12.mp4` — the most important trace in the run

Obj:static cam. GT **B: Moving forward**. Agent **A: Moving backward** ✗. Baseline B ✓.
5 steps, **every target frame visible (4/4)**.

This is the run's *methodologically perfect* trace. It is the only question where the agent
did everything the prompt asks: one query with `t_tgt=[0,31]` under fixed `t_cam=0` (not a
scan), a two-point rear→front facing axis, a `python_math` projection, all evidence cited.
It still gets the answer wrong, and the reason is not procedure.

| | value | 1σ noise | SNR |
|---|---|---|---|
| facing axis (chest − back, x/z) | `[+0.242, +0.544]`, ‖·‖ = **0.595** | 0.767 | **0.78** |
| displacement (frame 0→31, x/z) | `[-0.025, -0.244]`, ‖·‖ = **0.245** | 2.702 | **0.09** |

**Both inputs to the projection are smaller than their own error bars.** The displacement is
**11× smaller** than its uncertainty — driven by depth, where `dz = -0.244` against
`σ_z ≈ 2.7`. The facing axis is likewise unresolved: the "chest" and "back" boxes are
**9.6 px apart** on a 480×270 frame, on a runner whose entire bounding box is 86 px tall.

The agent computed `along = -0.2328`, reported it to four decimal places, and wrote
*"indicating backward motion."* It is a projection of noise onto noise; the sign is a coin
flip. Its `limitations` field worries about the *right* thing but the wrong magnitude:
*"the forward direction … may not perfectly align with the runner's actual facing direction,
but the projection is sufficient to determine the dominant motion component."*

> **This corrects an earlier claim in this document.** The head-to-head section above
> hypothesised, from the pattern of 5 forward↔backward errors, that the facing axis had a
> **reversed sign convention** — "a specific, checkable, fixable bug" worth 3 questions.
> This trace refutes that. The axis is not reversed, it is **unmeasured**. Fixing a sign
> convention would change nothing, because there is no signal whose sign to fix. Treat the
> "fix the facing-axis sign" recommendation as superseded by item 1 below.

**The prompt already told it to catch this.** The unscaled-units section instructs that a
displacement smaller than its own reported spread is tracker noise. Every result carried
`benchmark_aligned_xyz_std_m`. Across the entire run there were **10 `python_math` calls in
25 questions, and only 2 referenced a `_std_` field at all.** The noise floor `ensemble5` was
chosen to provide was computed, delivered, and then ignored in 8 of 10 calculations.

### 2. `CameraBench/u35WIs62R2M.1.4.mp4` — right method, then two compounding failures

Cam:static scene. GT **C: Moving downward**. Agent **A: Moving forward** ✗. 12 steps,
**10 of 11 targets invisible**.

Step 1 states the egomotion recipe correctly, inversion rule and all:

> "Ground a static background point … The camera's motion is the inverse of the static
> point's apparent motion. If the point appears to move upward (positive y), the camera moved
> downward, and vice versa."

Then calls 2–11 walk `t_tgt`/`t_cam` together through 31, 15, 10, 5, 2, 1, 3, 4, 6, 7 — every
one `visible: false`, `0/5` valid. Only frame 0 ever resolved, at 5/5.

**It lost track of its own success.** At step 3 it writes *"the point is not visible in frames
0, 15, and 31"* — but frame 0 **was** visible. That is `d4rt_1`, the evidence id it goes on to
cite in its final answer. It then repeats an ever-growing not-visible list that wrongly
includes 0 for nine more steps, holding a valid measurement it believed it did not have.

**The correct answer appears in its own final reasoning and is discarded.** Step 12:
*"The camera starts from a high vantage point and gradually moves downward… This suggests
that the camera is moving forward and possibly downward."* It then picks **Moving forward**.
GT is **Moving downward**.

### 3. `CameraBench/-2uIa-XMJC0.5.3.mp4` — correct answer, invalid reasoning

Obj-Cam distance. GT **A: Get farther**. Agent **A** ✓. 12 steps, **10 of 11 targets
invisible**.

Step 1 is sound and correctly invokes the scalar rule the prompt teaches: *"Since the
distance is frame-invariant, I can compute it in the camera's own frame at each point."*
Then the same collapse — calls 2–11 walk backwards 31→30→29→…→22, all invisible. And before
each one it **hallucinates the premise**: *"the character is visible in frame 30, and the
camera is still tracking them"* — ten times, each immediately falsified by the query it
then issues.

Its stated justification for the right answer:

> "The character is not visible in any frame after frame 0, so the distance is increasing.
> Therefore A: Get farther."

Tracker dropout implies nothing about distance — a target can vanish by leaving frame
laterally, by occlusion, or by tracker failure. One distance was measured and never compared
to anything. **This is a "both right" row, and it demonstrates concretely why that bucket is
not evidence of tool contribution.**

### The scanning in traces 2 and 3 is partly a harness artifact

`_refuse_hopeless_query` (`MAX_BLIND_QUERIES = 3`) was added after the job that produced most
of these answers. Checking every trace: the guard was active for only the **last 2 of the 25**
questions, and it fired correctly in both — query rejected at step 5, answer at step 6, no
scan. Traces 2 and 3 ran at 00:23 and 00:25 in the first job, before it existed, which is why
they scanned 10 times unchecked. Three questions show blind runs of 10, 10 and 11.

So the step counts in this study are **not comparable across questions**, and the scanning
pathology overstates what the current code would do. A re-run under the present guard is
cheap and would fix both.

### `t_tgt` is a list, and the agent almost never uses it as one

Trace 1 correctly requests `t_tgt=[0,31]` in a single call. Traces 2 and 3 issue 11
single-target queries each. One query with `t_tgt=[0,1,…,31]` under fixed `t_cam` would have
returned the whole visibility profile in one step and shown immediately that only frame 0
tracks. That is a prompt fix worth more than the refusal guard: it converts an 11-step dead
end into a 1-step one.

## Two confounds to rule out before believing the ordering

1. **Letter prior.** The GT histogram is A:6 **B:8** C:5 D:6. The baseline's answers are
   A:3 **B:10** C:7 D:5 — it leans B, and B is the modal GT. The agent leans C
   (A:6 B:4 **C:10** D:5), which is the *rarest* GT. Some of the baseline's 3-question
   edge may be prior/GT alignment on this particular 25-row draw rather than better
   reasoning. Checkable by re-scoring against a letter-permuted GT.
2. **Sampling design.** The Latin rectangle gives each task 4–5 questions, whereas real
   DSI-Bench is 33% `Obj:moving cam` and 31% `Cam:dynamic scene`. Neither number here
   estimates DSI-Bench accuracy, and the agent happens to do *fine* (3/4) on
   `Cam:dynamic scene`, one of the two blocks this design under-weights.

## What this pilot does and does not support

**Supported:** the harness works end-to-end — 25/25 complete, 25/25 citing at least one
D4RT measurement, no relaxed-gate retries needed, letters extractable from every answer.
The tool loop's biggest concrete problem is target visibility at 54%.

**Not supported:** any claim that the D4RT loop helps or hurts. n=25 with a 5-vs-2 paired
split cannot resolve a 3-question difference. A follow-up sized to detect a ~15-point
paired difference needs on the order of 150–200 questions, and would be better spent on
the two large real blocks than spread evenly across six tasks.

**But the per-question breakdown is actionable even at n=25**, because it names failure
modes rather than counting them. In descending order of expected return:

1. **Enforce the noise floor in the host, not the prompt.** Trace 1 is the whole problem in
   one question: a displacement 11× smaller than its own σ, projected onto a facing axis
   below *its* σ, reported to four decimals as a confident direction. The prompt already
   forbids this and was ignored — only 2 of 10 `python_math` calls in the entire run
   referenced a `_std_` field. Make it mechanical: when a bound displacement is below the
   summed spread of its endpoints, have the host say so in the tool result, the way
   `_diagnose_math` already does for nulls. This is the highest-value change in the list and
   it supersedes the "fix the facing-axis sign" hypothesis stated earlier.
2. **Decide the hedge policy.** 3 errors are *"Cannot be determined"* on questions where GT
   is never "Cannot be determined". Either instruct the agent that the hedge options are
   distractors and it must commit to a directional letter, or accept that measurement
   honesty costs accuracy on this benchmark and report both numbers. This is a prompt
   change, not a modelling one.
3. **Teach `t_tgt` as a list.** Traces 2 and 3 spent 11 steps each learning what one
   multi-target query would have shown in one. Pair this with the guard, which was active for
   only 2 of the 25 questions and worked both times.
4. **Re-run all 25 under the current harness.** The guard, the deadline notice and the math
   diagnosis landed mid-study, so step counts are not comparable across questions and the
   worst scans reflect code that no longer exists. It costs ~45 GPU-minutes.
5. **Reconsider `ensemble5` — but not for the reason expected.** Depth precision is already
   at a 2% median, so the noise floor was cheap to get; the problem is that the agent doesn't
   *use* it. `ensemble5` also requires 3/5 valid vs `centroid`'s 1/1, which may be part of the
   54% invisible rate. Worth one ablation, after item 1 — measuring whether the spread helps
   is meaningless while the spread is being ignored.
6. **Re-read the 8 both-right rows before crediting them.** They carry the run's average 57%
   invisibility, and trace 3 shows one of them reaching the right letter through a
   non-sequitur. The tool's true contribution on this sample is plausibly 1–2 questions, not
   10.
