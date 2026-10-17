# 01 — Where V4 and V5 stand

Snapshot on 2026-10-17. Every figure is question-weighted accuracy with unanswered questions
counted wrong. Intervals are clip-bootstrap 95% intervals where a run recorded one. Different
rows can be different samples; the matched comparisons in each section are the ones to trust.

## In one paragraph

V4 is still our best DSI-Bench system by a wide margin. Its lead is largest where its measurement
decides the answer: camera motion in the camera's own frame, and subject motion in the subject's
own frame. No V5 variant has beaten a tool-free Qwen3-VL-8B on Dyn-Bench; the only clear Dyn gain
over Qwen came from the deterministic D7 decision flow, where measurement rules decide camera
questions. The compositional V5 scored below legacy V5 on matched VLM4D questions and below V4 on
matched DSI questions. Its largest loss is on time-reversed DSI videos.

## DSI-Bench

DSI's own protocol averages four renderings of every question: standard, time-reversed, mirrored,
and both. Reversal inverts the true answer, so it separates measurement from motion priors.

| System | Sample | std | reverse | hflip | rev+hflip | 4-aug mean | ≥3/4 correct |
|---|---|---:|---:|---:|---:|---:|---:|
| **V4** (Qwen3-VL-8B + SAM3 + D4RT + OAV2) | full census, 1,769 × 4 | 71.6 | 57.0 | 69.7 | 56.8 | **63.8** | **52.5** |
| V4 | 200 × 4 | 66.5 | 54.5 | 68.0 | 52.5 | 60.4 | 51.0 |
| Tool-free Qwen3-VL-8B, 32 frames | same 200 × 4 | 58.5 | 25.0 | 56.5 | 26.0 | 41.5 | 19.0 |
| V5 legacy, Point4D | 800 random, std only | 51.4 | — | — | — | — | — |
| V5 compositional, Point4D | 63 IDs × 4 | 66.7 | 38.1 | 65.1 | 28.6 | 49.6 | — |

Published sample-wise means for reference: PolyV 58.7, Qwen3-VL-8B + DynTrace 56.4, Gemini-2.5-Pro
46.9, GPT-5 40.1. V4's census row is scored with DSI-Bench's unmodified `evaluate.py`.

**Matched comparisons.**

- 100 standard-split questions shared by the V5 legacy 800-run and the V4 200-run: V5 **50.0**,
  V4 **67.0**, tool-free control 59.0. V5 minus V4 is −17 points (interval −28 to −6).
- 63 question IDs × 4 renderings, compositional V5 versus V4 on the same IDs: 49.6 versus 56.0
  (−6.3, interval −17.7 to +5.5). Per rendering, V5 is level with V4 on std and hflip
  (66.7 / 63.5 and 65.1 / 65.1). It loses on reverse (38.1 / 47.6) and reverse+hflip
  (28.6 / 47.6, −19.0, interval −34.9 to −1.6).

**By category, standard split.** V5's 800-run and V4's census are both uniform draws from the
same split, so they are roughly comparable, though not matched.

| Category (DSI `cate`) | V4 census std | V5 legacy std (n) | V4 census, 4-aug mean |
|---|---:|---:|---:|
| c0 Obj: static camera | 63.2 | 50.0 (82) | 51.5 |
| c1 Obj: moving camera | 71.7 | 45.5 (257) | 53.2 |
| c2 Cam: static scene | 78.3 | 45.2 (104) | 80.1 |
| c3 Cam: dynamic scene | 72.9 | 57.1 (254) | 71.2 |
| c4 Obj–cam distance | 76.8 | 62.3 (61) | 74.6 |
| c5 Obj–cam orientation | 55.3 | 54.8 (42) | 53.2 |

V5 is lower in every category, by the most on camera own-frame motion (c2, −33) and subject
motion (c1, −26). V4's own weak spot is subject motion under reversal: c0/c1 fall to 37–39% on
both reversed splits, from 63–72% on std.

## Dyn-Bench

The working set is the decomposition study's 2,800-question eval split (400 clips, all
question types). The split is about 1.6 points easier than the full 7,000.

| System | Accuracy |
|---|---:|
| V5 legacy, Point4D | 58.8 (56.8–60.7) |
| V5 legacy, D4RT | 58.1 (56.2–59.9) |
| Qwen3-VL-8B, frames only (decomposition protocol) | 59.7 |
| Qwen3-VL-8B, official protocol (all frames, T = 0.7) | 60.6 |
| Qwen3-VL-8B, option log-probabilities | 61.4 |
| D7 decision flow (deterministic gates, Qwen fallback) | 63.7–64.1, depending on artifact snapshot |

Published: Qwen3-VL-8B 61.4 on the full benchmark (Dyn-Bench paper); Qwen3-VL-8B + DynTrace 65.8.
V4 was never run on Dyn; its four buckets cover about 17% of Dyn questions. The compositional V5
Dyn run never started (cluster file-count quota, jobs cancelled).

V5 legacy by Dyn level: Object–Scene (sceneqa) 67.2, Camera–Object (cameraqa) 55.9,
Inter-Object (qa) 54.6. Per kind, on the same question IDs, V5 Point4D against frames-only Qwen:

| V5 beats frames-only | V5 / frames | V5 loses to frames-only | V5 / frames |
|---|---:|---|---:|
| camera_path (282) | 59.9 / 46.5 | pair_relation (406) | 56.4 / 62.8 |
| camera_zoom (68) | 91.2 / 79.4 | camera_viewpoint (72) | 38.9 / 56.9 |
| camera_rhythm (127) | 56.7 / 45.7 | cam_subject_elevation (82) | 35.4 / 45.1 |
| scene_revisit (60) | 71.7 / 58.3 | metric_speed (93) | 24.7 / 36.6 |
| | | event_order (92) | 67.4 / 78.3 |
| | | instance_count (48) | 43.8 / 58.3 |

Camera kinds gain; entity, relation, metric and count kinds lose. That is the same split the D7
flow found: measurement rules win on camera kinds, entity kinds fall back to Qwen.

## VLM4D

| System | Sample | Accuracy |
|---|---|---:|
| V5 legacy, Point4D | all 1,816 | 55.1 (52.6–57.3) |
| Video-blind text prior (our rule-based estimate) | all 1,816 | 47.2 |
| V5 compositional | 300 stratified | 45.3 (older V5 on the same 300: 50.3) |

V5 legacy ranks 3rd of 23 in the paper's chain-of-thought table (Gemini-2.5-Pro 61.9, GPT-4o 57.5,
Qwen2.5-VL-72B 53.0). There is no tool-free Qwen3-VL-8B run on VLM4D, so the tools' contribution
is unknown. Weak areas: counting (37–40%), synthetic false-positive questions where the answer is
"no such object" (24.4%, n = 45), and rotation questions (125 of 317 had every measurement
unavailable). Synthetic directional questions are a strength (64.5%).

## Development pilots on Dyn

Same 24 clips and 168 questions, reused from the audited pool, so these measure progress on known
failures rather than generalization.

| Variant | Accuracy | Mean s/question | Note |
|---|---:|---:|---|
| V5 legacy baseline | 58.3 | 15.3 | |
| Workflow candidate: early options, one review, fresh answer context, `relative_position` | 61.3 | 23.5 | +3.0 (interval −2.4 to +7.7); not merged |
| Rated answers / rated + repair | 62.5 / 62.5 | 57.3 / 56.3 | repair ran 0 times; complete structured assessments on 52 and 57 of 168 |

## Cost

V5 legacy: DSI 800 questions in 6 h 45 min on one Blackwell GPU (median 28 s per question); VLM4D
1,816 questions in 18 h (about 100 per hour). In the pilot, the rated answer policy took 57 s per
question against the baseline's 15 s. One question writes about 100 files, which matters under the
share's file-count quota.

## Sources

| Figure | Where |
|---|---|
| V4 census and 200 × 4, published table | `d4rt_agent/results/official_export/PER_CATEGORY_VS_PUBLISHED.md` |
| V5 DSI 800 and the 100-question overlap | `d4rt_agent/results/agent_v5_dsi800_point4d_20261010T012544Z/RESULTS.md` |
| V5 Dyn runs, per-kind matched slices | `d4rt_agent/agent_v5/WORKFLOW_REVIEW.md`; run folders `agent_v5_eval2800_{point4d,d4rt}_*` |
| V5 VLM4D | `d4rt_agent/results/agent_v5_vlm4d_point4d_20261010T012544Z/RESULTS.md` |
| Compositional subset | `d4rt_agent/results/agent_v5_subset_20261014T021547Z_retry/{report,failure_analysis}.md` |
| Pooled per-kind/tool tables | `d4rt_agent/results/accuracy_breakdowns_20261015/report_sorted.md` |
| Pilots | `d4rt_agent/results/agent_v5_workflow_pilot_20261013/`, `agent_v5_rated_pilot_20261013/` |
| D7 flow, official Qwen | `dyn_dataset_exploration/logs/DECOMPOSITION.md`; `d4rt_agent/results/qwen3vl8b_dynbench_official_20261007T175618Z/` |
