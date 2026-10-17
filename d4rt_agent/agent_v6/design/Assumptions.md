# Assumptions

Things AgentV6 relies on but cannot check while answering. If one of them fails, the agent can do
nothing about it, and the error belongs to the component named here, not to the agent's reasoning.
Each entry says what the agent does because of the assumption, what we know about it, and how an
evaluation would show it failing.

## A1 — DA3-scaled depth gives correct metric scale

*Recorded 2026-10-17 (decision log #12).*

**What the agent does.** Reports distances in metres and speeds in metres per second whenever the
scale exists, **without reliability warnings**. The agent treats them like any other measurement
and picks the best-matching option.

**Basis.** The scale is Point4D geometry multiplied by the DA3-Nested depth ratio. Against TUM,
Bonn and KITTI ground truth it is within ±10% indoors and reads 7–25% short on KITTI
(`dyn_dataset_exploration/outputs/metric_gt_check/summary.md`). Point4D raw and D4RT raw are not
metric and are never used for metres.

**Known risks.** Metric answers have been poor so far. V5 scored 24.7% on metric speed and 33.3% on
metric distance; in the D7 study, answering with the measurement scored 18.5% (speed) and 30.1%
(distance), against 35.9% and 31.5% frames-only. Not all of this is scale. Speed also depends on
the trajectory estimator: V5 used a visible-surface centroid, and net and path speed differ.

**How failure would show.** Metric-question accuracy below frames-only Qwen on matched questions,
or a systematic ratio between our estimates and the option values (V5: median estimate/option
0.36). Then fix the estimator or the scale. The agent's prompt does not change.

## A2 — An unspecified direction means the world frame (the start view, gravity-levelled)

*Recorded 2026-10-17 (decision log #15).*

**What the agent does.** When a question asks which way something moves without naming a frame
("Which direction is the bird flying towards?"), it measures in the world frame. Forward is where
the camera first looked, and camera motion is removed.

**Basis.** On Dyn-Bench, left/right agreed with the reference answer about 90% of the time in
first-camera axes, against about 60% in moving-camera axes (`dyn_dataset_exploration/logs/DECOMPOSITION.md`).
DSI names its frame in every subject question. VLM4D: 1,196 direction questions give no frame
cue, 136 say "from the camera perspective" or "towards/away from the camera", and 82 say "from its
own perspective". VLM4D's convention for the unmarked 1,196 has not been checked.

**How failure would show.** On VLM4D, unmarked direction questions scoring clearly below the
camera-perspective ones, or below frames-only Qwen, especially on clips where the camera pans to
follow the subject. If so, the convention differs by benchmark source, and the frame description
needs a dataset-independent cue, not a benchmark rule.
