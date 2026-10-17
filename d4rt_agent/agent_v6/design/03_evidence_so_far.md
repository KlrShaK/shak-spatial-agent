# 03 — What we already know

Findings from v1–v5 runs, the V5 workflow review (2026-10-13), the capability research
(2026-10-13), the pilots and the compositional failure analysis. Each theme states what was
observed and where the evidence is. "Verified" means reproduced in code or counted from saved
traces; "hypothesis" means no controlled test exists yet.

## 1. Loop design: who plans, and how much the 8B model is asked to do

Free-form tool loops lost to the tool-free control twice. On the 25-question DSI pilot, v2 scored
10 and v3 scored 9, against 13 for the control. Both failed on planning, not measurement:
frame-by-frame walks, re-querying invisible targets, budgets spent before answering. V4 moved the
plan to the host and scored 16/25, then 60.4% against 41.5% on 200 × 4. SpatialClaw, a free-form
code-writing agent, shows the same pattern: 41.3% on DSI 200 × 4 with an 8B backbone and 61.9% with
27B.

V5 gave planning back to the model inside kind-scoped menus. The model used that freedom sparingly:
2.3 measuring turns per question and 8 kind switches in 2,800 Dyn questions. Every attempt to add
richer model outputs failed operationally:

| Mechanism | What happened | Source |
|---|---|---|
| Rated answers (score every option) | 309 of 362 rejected attempts were invalid JSON envelopes; complete assessments on 52/168 | rated pilot `REPORT.md` |
| Evidence review step | 155 attempts used the wrong action name; +37% model calls, +53% latency | workflow pilot `MANUAL_REVIEW.md` |
| Repair after assessment | Executed 0 times in 552 compositional questions and 0 in both pilots | `failure_analysis.md`, rated pilot |
| Stronger planner (gpt-6-luna reading a caption) driving V5's phases | 56.7% on Dyn 2,800, below V5's 58.8; loses where pixels are needed | jev experiment, `results/archive/jev_eval2800_*` |

## 2. Routing and reference frames

**DSI subject motion is mostly a frame problem (verified from traces).** DSI subject questions ask
for motion "relative to its starting orientation". V5 legacy routed 290 of 366 to
`subject_direction`, which measures in the scene frame: 40.0% correct. The 74 routed to
`traj_subject_own_frame` scored 73.0%. Scene-x sign agreed with gold left/right at chance (29/54).
The tool note "scene matches the benchmark 9 in 10" is a Dyn calibration that is wrong for DSI.
V4's single bucket for this category avoided the choice entirely; its bucket choice agreed with the
taxonomy on 24/25.

**Camera own-frame motion is V5's largest DSI loss.** Cam: static scene is 45.2% in V5 against 78.3%
in V4's census. V5 splits these questions between `camera_translation` (56.5%) and `camera_path`
(42.9%). V4 used D4RT's camera grid, a gravity-levelled frame-0 heading, and a contract that spells
out the orbit sign rule. This is the strongest single target for V6 on DSI, and its cause is not
yet isolated: routing, backbone (V4 D4RT, V5 Point4D), camera smoothing and the reading contract all
differ.

**Kind names mix concepts.** Families and kinds combine who moves, operands, observer relation,
units, frame and reduction. Ambiguous pairs observed: `camera_viewpoint` versus
`cam_subject_elevation`, `subject_direction` versus `traj_subject_own_frame`, `camera_translation`
(playbook says own axes, tool returns gravity-levelled initial axes). Source: workflow review §11.

**When options are shown.** The workflow review argues options must be visible before choosing what
to measure, because compound options need several observables (414 of 1,769 DSI questions mention
rotation or orbit only in the options; that slice scores 31.7% in V5). V4 hid options until the
answer and still won on DSI, where each category's options are homogeneous. The compositional stack
showed options early and did not improve; that change was bundled with others, so its own effect is
unknown.

## 3. Measurement semantics and coverage

**Verified defects.** Twelve synthetic diagnostics replayed through the production measurement code
all reproduced. Representative ones:

| Diagnostic | What the code did | What is needed |
|---|---|---|
| Two entities swap left/right at constant distance | Identical range reports | A signed relative position; scalar range cannot decide left/right |
| Camera out and back; 78° pure roll | Labelled `static` | Motion over the interval, roll included |
| Round trip, 4 units in 6 s | "Average speed" 0 | Path speed and net displacement rate as separate quantities |
| Valid 2D masks, no depth | Image-frame track unavailable | 2D signals independent of 3D |
| Count in seconds 0–1.5 | Counts entities from the whole clip | Windowed, set-typed counts |
| One full 360° turn | Rendered turn 0° | Unwrapped, accumulated rotation |
| Fixed surface, changing visible part | Apparent motion 3.3 with noise 0.014 | Persistent correspondence, not visible-surface medians |

Source: `agent_v5/research/FINDINGS.md`. The compositional stack addressed most of these in code.
Its end-to-end accuracy did not improve, so fixing semantics was not sufficient on its own.

**Metric scale.** Against TUM, Bonn and KITTI ground truth, Point4D raw depth is off by 1.4× to 40×
and D4RT is not metric either (0.6–0.9 indoor, 7–13 on KITTI). Point4D × DA3-Nested is within ±10%
indoors and reads 7–25% short on KITTI. Metric speed scores 24.7% (nearest option correct on 19/83 with
net speed, 20/83 with path speed), so the estimator itself, not just the mode, is wrong.

**Backbone sensitivity.** Switching Point4D to D4RT kept the chosen kind on 2,774/2,800 questions
but moved many answers: zoom 91 → 72, shake 41 → 21, speed 25 → 16. Thresholds were calibrated on one
configuration and reused on another.

**Missing observables.** Signed relative position, group configuration, event transitions such as
stop and start, visual event localization, orbit about a defined reference, and predicate-filtered
counts. VLM4D adds six kinds V5 lacks (490 questions, 27%): repetition count, count with predicate,
rotation sense, relative position, subject turn, hand attribution.

## 4. From evidence to answer

**The answer turn is where several regressions happen (verified from traces).**

- V5 compositional gave the final answer no images on 136/252 DSI and 187/300 VLM4D records.
  The fallback for failed geometry became text-only.
- Missing evidence was read as proof. A failed detection became "no such object"; an unresolved
  orientation became "not moving".
- In five pilot regressions, the review endorsed the correct answer and the final answer
  contradicted it.
- On the 192 Dyn questions where V5 legacy made no tool call, it scored 140 against frames-only
  Qwen's 148. The agent's conversation and prompts cost accuracy even when no tool ran.

**Prompt contradictions (verified).** The system prompt says labels are summaries. The legacy answer
prompt said not to overrule them. The chronology rule ("describe in time order, never reorder by
size") conflicts with questions asking for the dominant or net motion, and its example had the
order wrong.

**A-bias (hypothesis).** V5 answers "A" 1,331 times against 1,052 true A on Dyn (267 against 210 on
DSI). The answer template's literal `"letter": "A"` is one suspect; numeric options are a confound,
since A is the smallest speed on 87 of 93 speed questions and speed estimates run low.

**V4's answer turn, for contrast.** All 32 frames stayed in context, one report arrived with its
reading contract, and the prompt said to answer from the frames when the measurement is
unavailable. V4's measured-versus-unmeasured split on 200 questions was 71% against 57%.

## 5. Time reversal is the sharpest test

The tool-free control falls to chance on reversed DSI video (25%), below chance on subject motion,
because it applies a motion prior. V4 holds 57% because its measurement inverts with the video.
Compositional V5 falls to 38% and 29% on the reversed splits of its 63 IDs. V4's own weak cell is
subject motion under reversal (c0/c1 at 37–39%). Whatever V6 does with subject motion should be
checked on reversed video first; a standard-split gain can hide a prior.

## 6. Where tool failures come from (tallied 2026-10-17)

"Tool failure" covers three different things. Counted from saved answers with
[analysis/failure_tally.py](analysis/failure_tally.py); question counts, with accuracy on those
questions in brackets.

| Run (questions) | Model output rejected | Host refused the arguments | A measurement came back unavailable | All measurements OK |
|---|---:|---:|---:|---:|
| V5 legacy Dyn (2,800) | 158 (57.6%) | 157 (56.1%) | 289 (55.4%) | 2,312 (58.1%) |
| V5 legacy DSI (800) | 78 (48.7%) | 15 (53.3%) | 112 (58.9%) | 687 (50.1%) |
| V5 legacy VLM4D (1,816) | 175 (37.1%) | 146 (39.7%) | 354 (50.6%) | 1,323 (55.7%) |
| V5 compositional DSI (252) | 85 (35.3%) | 100 (32.0%) | 104 (38.5%) | 100 (67.0%) |
| V5 compositional VLM4D (300) | 65 (33.8%) | 85 (38.8%) | 68 (35.3%) | 189 (49.7%) |
| V4 DSI census std (1,769) | 167 | — | 457 | — |

**Model output rejected.** In the legacy stack this is mostly the host's own word check refusing
reasonable entity names ("leopard" when the question says "animal": 238 of 280 Dyn rejections).
Broken JSON is rare during measurement and concentrated in the answer turn (83 on VLM4D, which
left 24 questions unanswered). The compositional stack's larger schemas raised broken JSON during
measurement to 129 rejections on 252 DSI questions.

**Host refused the arguments.** The model meant something sensible but used a form the host does
not accept. Legacy: instance handles (`c1#0` when only `c1` is registered, the count-display bug),
names instead of IDs (`snowboarder` for `e1`), `0` for `e1`. Compositional (486 and 458 refusals):
windows in generator-frame integers (178, "frames must be [start_generator_frame,
end_generator_frame]"), the camera not being a registered entity (`cam`, 108), and inspection or
visual IDs in the wrong form (about 180). These are interface failures, and nearly all are
mechanically recoverable from the history.

**Measurement unavailable.** Perception, not formatting: OAV2 finds no front (VLM4D 217, V4 206),
SAM3 finds nothing (VLM4D 183, Dyn 88, DSI 80), too few usable frames (V4 206), no metric scale
(Dyn 53). V4 had a measurement unavailable on **26%** of census questions, more than V5's 14% on
DSI, and still scored higher. Its fallback was to answer from all 32 frames.

**Two readings.** On legacy DSI, questions whose measurements all succeeded scored *lower* (50.1%)
than questions where one failed (58.9%), consistent with the wrong-frame routing in §2. On
compositional DSI the opposite holds: 67.0% when every measurement succeeded, 32–39% when anything
failed. That stack's measurements were usable; its failure handling (text-only answer packets,
missing evidence read as absence) is where it lost.

## 7. Evaluation hygiene

| Problem | Effect | Status |
|---|---|---|
| `report.py` clustered by question prefix | 1,200 bootstrap clusters instead of 400 clips | Fixed (`34ce23e`) |
| Evidence accuracy divided by all questions | 51% "evidence accuracy" was 59% on the 2,420 with evidence | Reporting rule noted |
| D7 baseline artifact changed | 63.7, 63.75 and 64.07 quoted for the same flow | Name the artifact with every number |
| Pilots reused audited clips | Pilot gains are progress on known failures, not generalization | Stated in every pilot report |
| Format success used as a pilot gate | A workflow whose repair never ran was promoted to the subset campaign | Gate must include a matched control |
| File-count quota on the share | Killed the compositional Dyn run; ~100 files per question | Open; affects run design |

Sources: `agent_v5/WORKFLOW_REVIEW.md`, `results/agent_v5_subset_20261014T021547Z_retry/`.
