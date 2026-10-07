# Decomposing Dyn-Bench questions: the decision flow

Hand analysis, not generated. The successor of `AGENT_V5_GAPS.md`: that file said what AgentV4
could not do; this one is what was built instead, and what it measured.

**What it is.** A deterministic decision graph that turns one Dyn-Bench question into a typed
*Query Program*, solves each sub-problem with a tool (SAM3 for grounding, Point4D for 3D, DA3-Nested
for metric scale, Qwen3-VL-8B for reading language and looking at frames), and assembles the answer
from the measurements. Code: `dyn_dataset_exploration/decomposition/`. Notebook:
`dyn_dataset_exploration/testing_decomposition.ipynb` (the first cell is the graph).

```
S0 clip context   generator frames + time axis · Point4D scene pass (camera R,t,K · depth) · gravity · scale
S1 decompose      question + options -> Query Program  (kind first, then slots; rules, else Qwen)
   D0 option form  D1 route  D2 WHO  D3 WHEN  D4 WHAT+FRAME  D5 REDUCE  D6 OPTIONS
S2 ground         SAM3: entity | group | category | main subject | landmark   (role policies pick instances)
S3 measure        lift masks through Point4D depth -> signal in a frame and a window -> reduce
S4 interpret      score every option's claim -> D7 this kind's gate: clear margin, every claim checkable?
                    yes -> LEAF A  deterministic answer
                    no  -> LEAF B  the kind's fallback, fitted on dev: Qwen + frames + measured evidence,
                                   Qwen from the frames alone, or the SOFT FUSION of Qwen's option
                                   probabilities with the measurement's support (log p = a·log p_qwen + b·log(support+eps))
                  nothing measurable -> LEAF C Qwen from the frames
```

## What the exploration found (all verified on the data)

- **The generator's frames are not the files on disk.** `frame_sampling.json`'s 16 indices are
  `floor(linspace(0, L-1, 16))` over `L` = the GT mask-frame count. DynamicReplica and PointOdyssey
  are every 5th RGB frame. The shipped `total_frames` is wrong on 106 of 1000 clips.
- **Time is `gf_index / fps`, per source:** SA-V 6, YTVIS 6, DAVIS 30, DynamicReplica 6, DynPose ~12.
  No question timestamp passes its clip's end at those rates; TotalRecon, Spring and PointOdyssey
  have too little evidence and are flagged rather than guessed.
- **Question timestamps land on the 16 sampled frames** far above chance (DAVIS 71% vs 25%,
  SA-V 43% vs 21%): the generator reasoned over those frames.
- **Portrait video** (207/293 SA-V clips, all TotalRecon) must go through Point4D at 518x294; the
  default 294x518 squashes it 3x.
- **Answer distributions are very skewed** (zoom is "None" 76% of the time, "Fully visible" 75%,
  camera rhythm "steady" 71%). A detector that fires eagerly loses to a constant; thresholds have to
  be calibrated against that.
- **Qwen3-VL-8B picks the right kind 10/10 but paraphrases** ("child" for "people"), so the host
  checks every slot against the question's own words and sends a rejection back.
- **SAM3 splits one person into fragments** when they leave the frame and return; merging them (and
  how far apart in time) is a measured choice.

## What the calibration found about the benchmark's conventions (dev clips; test agrees where checked)

- **Left/right is physical, forward/backward is not.** For "how does X's position change?" the ground
  truth's left/right agrees with the subject's displacement in the *first camera's axes* on 94% of dev
  questions that name a side (30/32; camera-end axes 94%; the *moving* camera's own axes only 62%).
  Where the two frames disagree (camera motion dominates) the answer follows the world frame 11 times
  of 12, even when the question says "relative to the camera". Forward/backward agrees with nothing:
  46-71% in every frame and sign convention, and the benchmark uses "forward" both for moving away
  and for coming toward the camera. So the options are scored per axis with weights fitted on dev
  (`interpret.DECISION['dir_w_x/y/z']` = 1 / 0.5 / 0.3): one unreliable axis no longer vetoes an
  option, and two options that differ only in depth stay nearly tied and go to the fallback.
- **A range changes by a ratio.** Camera-to-subject distance is classified on log(range): "twice as
  far" and "half as far" are the same size of change. Each classification was calibrated on dev: the
  ratio-based one measured 57 of 90 right on dev and 53 of 86 on test, the unit-based one 52 and 48
  (108 + 98 questions before the catalog was extended).
- **Metric distance is limited by the answers as much as by the depth.** log(measured m) against
  log(answer) correlates 0.5 (0.69 for camera-to-subject pairs, 0.43 for object pairs); the
  measurement is low by a median factor 0.76 and typically off by a factor 1.8, while adjacent
  options differ by less than that. The evidence text tells Qwen so.
- **Camera-to-subject questions and "stationary or move?"** The latter, asked of the *camera*, is a
  question about how much and when the camera moves; the first rule set routed it to the
  subject's motion profile with "camera" as the subject and could never ground it. It is now left
  unread by the rules.

## How to run it

**Resource rule.** Anything that reads many clips (calibration, evidence writing, the policy fit, catalog
builds, frame materialisation) runs as a bounded Slurm job, never on a login node: a detached calibration
holding several GB of scenes crashed a login node once. `Context(out, max_clips=6)` keeps a one-pass job
flat at ~2 GB; `scripts/run_cpu_task.slurm` and `scripts/run_gates_cpu.slurm` carry the limits.

```bash
# catalogs (CPU job: sbatch scripts/run_cpu_task.slurm -m <module> <args>)
-m dyn_dataset_exploration.decomposition.build_catalog --materialize          # the 10-question catalog
-m dyn_dataset_exploration.decomposition.build_dev_catalog [--extend [--heldout]]; then --materialize --workers 4

# GPU (A100 launcher starts in minutes; Blackwell can queue for hours). Shard with SHARD=i/n NPROC=k
PART=point4d   PLATFORM=a100 OUT=<catalog> STAGES="scene metric" sbatch scripts/run_decomposition_prep_a100.slurm
PART=qwen_sam3 PLATFORM=a100 OUT=<catalog> STAGES="name ground answer letters" sbatch ...   # SAM3 + Qwen
PART=d4rt      PLATFORM=a100 OUT=<catalog> sbatch ...                                       # optional second metric scale
PART=census    PLATFORM=blackwell sbatch ...                                                # kind of all 7,000 questions

# fit (bounded CPU jobs)
run_calibration            # measurement constants on dev clips -> decomposition/calibrated.json
write_evidence --all       # the measured evidence text, then GPU STAGES=evidence answers it (skips unchanged text)
run_gates --dump-rows-only # per-question rows, once; policy_lab [--pooled] compares policies in seconds from them
run_gates --fit-all --method per_kind_vs_simple      # the shipped D7 policy;  run_gates --heldout  scores held-out clips

# the notebook (kernel `point4d`), one executed copy per branch
python -m dyn_dataset_exploration.decomposition.build_notebook
python -m dyn_dataset_exploration.decomposition.execute_notebook [--catalog dev --kinds camera_zoom]
```

## Pitfalls found while building it

- `torch.device(...)`-side initialisation matters: building DA3-Nested (1.4B parameters) on the
  CPU takes tens of minutes and every weight is overwritten by the checkpoint.
- The group is capped at two concurrent GPUs on the Blackwell partition; run several shard
  processes per job (`NPROC`) rather than many jobs.
- `envs/point4d` (torch cu124) cannot run on Blackwell; Point4D runs there from
  `envs/d4rt_bw_third_party` (torch cu128) with `addict` and `omegaconf` added, and agrees with the
  A100 output to ~1%.
- Per-frame pose steps turn estimator jitter into speed (0.1 deg/frame is 3 deg/s at 30 fps); every
  speed is a displacement over half a second.
- Savitzky-Golay rings around a step (Point4D's focal length jumps in a clip's last frames) and
  invents an opposite swing; step-like series use a running median.

## Representative evaluation (the headline number)

The curated catalogs below over-represent what the rules can read. This is the check on Dyn-Bench itself
(`build_eval_catalog.py`, `run_eval.py`): **400 clips drawn at random, stratified by source dataset, with all
7 questions of each clip: 2,800 questions, 40% of the benchmark**, task mix 1,200 camera / 800 object / 800
scene exactly as in the benchmark. The 254 clips used to fit anything were excluded from the pool, the
measurement constants are the dev-fitted ones, and the D7 policy was refitted on dev rows only
(`calibrated.eval.json`), so nothing is in-sample. Questions the rules cannot read (1,447 of 2,800) went
through the Qwen parse (kind, then slots) and the same measurement and policy code; kinds with no fitted
policy and the semantic questions are answered by Qwen's option probabilities. No question errored.

| answering | accuracy | 95% CI (clips resampled) |
|---|---|---|
| Qwen3-VL-8B from the frames, reasoning | 1671/2800 (59.7%) | 57.7-61.6 |
| Qwen option probabilities (direct answer) | 1720/2800 (61.4%) | 59.6-63.2 |
| Qwen with frames + measured evidence* | 1707/2800 (61.0%) | -- |
| **the decision flow** | **1785/2800 (63.8%)** | **61.9-65.6** |

\* evidence exists only for the 2,423 questions that were measured; the others use the frames-only answer.

The flow is **+4.1 points over Qwen from the frames (95% CI +2.5 to +5.8) and +2.3 over Qwen's direct answer
(+0.8 to +4.0)**. That is about half the gain seen on the curated catalogs (+8 to +10), and the table below
says why: the gain sits entirely in questions the measurement can reach.

| read by | n | decision flow | Qwen frames | Qwen letters | Qwen + evidence* |
|---|---|---|---|---|---|
| qwen | 1447 | 926/1447 (64.0%) | 923/1447 (63.8%) | 917/1447 (63.4%) | 883/1447 (61.0%) |
| rules | 1353 | 859/1353 (63.5%) | 748/1353 (55.3%) | 803/1353 (59.3%) | 824/1353 (60.9%) |

By task:

| task | n | decision flow | Qwen frames | Qwen letters | Qwen + evidence* |
|---|---|---|---|---|---|
| cameraqa | 1200 | 713/1200 (59.4%) | 621/1200 (51.8%) | 651/1200 (54.2%) | 708/1200 (59.0%) |
| qa | 800 | 474/800 (59.2%) | 461/800 (57.6%) | 463/800 (57.9%) | 424/800 (53.0%) |
| sceneqa | 800 | 598/800 (74.8%) | 589/800 (73.6%) | 606/800 (75.8%) | 575/800 (71.9%) |

By source dataset:

| dataset | n | decision flow | Qwen frames | Qwen letters | Qwen + evidence* |
|---|---|---|---|---|---|
| ytvis | 1428 | 884/1428 (61.9%) | 839/1428 (58.8%) | 870/1428 (60.9%) | 833/1428 (58.3%) |
| sav | 833 | 539/833 (64.7%) | 502/833 (60.3%) | 519/833 (62.3%) | 530/833 (63.6%) |
| dynpose | 217 | 154/217 (71.0%) | 140/217 (64.5%) | 141/217 (65.0%) | 142/217 (65.4%) |
| davis | 203 | 144/203 (70.9%) | 121/203 (59.6%) | 120/203 (59.1%) | 135/203 (66.5%) |
| totalrecon | 49 | 25/49 (51.0%) | 27/49 (55.1%) | 31/49 (63.3%) | 27/49 (55.1%) |
| dynamic_replica | 28 | 14/28 (50.0%) | 17/28 (60.7%) | 16/28 (57.1%) | 17/28 (60.7%) |
| point_odyssey | 21 | 13/21 (61.9%) | 17/21 (81.0%) | 15/21 (71.4%) | 12/21 (57.1%) |
| spring | 21 | 12/21 (57.1%) | 8/21 (38.1%) | 8/21 (38.1%) | 11/21 (52.4%) |

By kind (the 12 largest):

| kind | n | decision flow | Qwen frames | Qwen letters | Qwen + evidence* | answered by the measurement |
|---|---|---|---|---|---|---|
| semantic | 336 | 268/336 (79.8%) | 268/336 (79.8%) | 268/336 (79.8%) | 268/336 (79.8%) | 0 |
| pair_relation | 331 | 209/331 (63.1%) | 209/331 (63.1%) | 209/331 (63.1%) | 189/331 (57.1%) | 0 |
| camera_path | 267 | 156/267 (58.4%) | 121/267 (45.3%) | 134/267 (50.2%) | 164/267 (61.4%) | 171 |
| landmark_relation | 258 | 182/258 (70.5%) | 173/258 (67.1%) | 182/258 (70.5%) | 168/258 (65.1%) | 0 |
| cam_subject_range | 250 | 132/250 (52.8%) | 120/250 (48.0%) | 121/250 (48.4%) | 132/250 (52.8%) | 0 |
| subject_motion_profile | 174 | 128/174 (73.6%) | 126/174 (72.4%) | 123/174 (70.7%) | 128/174 (73.6%) | 0 |
| subject_direction | 132 | 77/132 (58.3%) | 74/132 (56.1%) | 74/132 (56.1%) | 72/132 (54.5%) | 43 |
| apparent_size | 130 | 87/130 (66.9%) | 88/130 (67.7%) | 89/130 (68.5%) | 87/130 (66.9%) | 0 |
| camera_rhythm | 128 | 81/128 (63.3%) | 60/128 (46.9%) | 59/128 (46.1%) | 74/128 (57.8%) | 96 |
| cam_subject_elevation | 101 | 58/101 (57.4%) | 50/101 (49.5%) | 61/101 (60.4%) | 49/101 (48.5%) | 0 |
| event_order | 97 | 74/97 (76.3%) | 74/97 (76.3%) | 73/97 (75.3%) | 71/97 (73.2%) | 0 |
| subject_visibility | 93 | 68/93 (73.1%) | 73/93 (78.5%) | 71/93 (76.3%) | 71/93 (76.3%) | 70 |
| metric_speed | 92 | 27/92 (29.3%) | 33/92 (35.9%) | 35/92 (38.0%) | 16/92 (17.4%) | 29 |
| camera_shake | 82 | 41/82 (50.0%) | 31/82 (37.8%) | 34/82 (41.5%) | 39/82 (47.6%) | 66 |
| camera_zoom | 74 | 66/74 (89.2%) | 57/74 (77.0%) | 57/74 (77.0%) | 66/74 (89.2%) | 0 |
| metric_distance | 73 | 27/73 (37.0%) | 23/73 (31.5%) | 27/73 (37.0%) | 21/73 (28.8%) | 0 |
| scene_revisit | 56 | 36/56 (64.3%) | 28/56 (50.0%) | 34/56 (60.7%) | 40/56 (71.4%) | 0 |
| camera_translation | 53 | 22/53 (41.5%) | 23/53 (43.4%) | 23/53 (43.4%) | 19/53 (35.8%) | 28 |
| tracked_objects_count | 41 | 20/41 (48.8%) | 19/41 (46.3%) | 20/41 (48.8%) | 19/41 (46.3%) | 0 |
| instance_count | 32 | 26/32 (81.2%) | 21/32 (65.6%) | 26/32 (81.2%) | 14/32 (43.8%) | 0 |

Reading it honestly:
- **Where it helps:** the camera questions (59.4% vs 51.8% frames, +7.6), and any kind the measurement can
  decide: zoom 89% vs 77%, rhythm 63% vs 47%, shake 50% vs 38%, camera path 58% vs 45%, instance count 81%
  vs 66%. On the 503 questions the measurement answered by itself it is right 59.6% of the time against 48.3% for Qwen
  on the same questions.
- **Where it does not:** the 1,447 questions the rules could not read (64.0% vs 63.8%) and the scene-level
  narrative questions (74.8% vs 73.6%; Qwen's direct answer alone is 75.8%). There is nothing to measure,
  and the flow correctly adds nothing; it does not hurt either.
- **Where it loses:** metric speed (29% vs 36-38%), subject visibility (73% vs 78%), camera translation
  (42% vs 43%). Metric speed is the clearest: the policy fitted on dev let the measurement answer where it
  was at chance. The cross-validated all-clips policy turned this off (see the Results section below); this
  dev-only policy is what a clean evaluation has to use.
- **Camera path** is 75% (vs 48% frames) on the 154 questions the rules read, as on the calibrated catalogs, but
  36% (vs 42%) on the 113 that Qwen had to read into the kind: those are worded differently from the
  templates and are harder for every method. Across all kinds, the questions the measurement answered by
  itself are right 64% of the time when a rule read them (Qwen's direct answer: 58%) and 40% when Qwen
  read them (Qwen: 37%). So the measurement is not the problem on them, but the gain is entirely on
  the templated questions.
- The small sources (TotalRecon, Spring, PointOdyssey, DynamicReplica: 119 questions together) are too
  few to read per dataset; time windows there are flagged (frame rate unverified), and the flow is not
  clearly ahead of Qwen on them.

## Calibration-set results

**Data.** 2,606 labelled questions on 719 clips, 18 kinds, split by clip: *dev* (509 clips, where every
measurement constant was chosen), *test* (the other half of those clips), and *held-out* (210 further
clips added later, chosen because they hold questions of the thinnest kinds; nothing was fitted on them
until the last step). Every question has the frames-only Qwen answer, Qwen's option probabilities, and the
Qwen answer given the measured evidence.

**How each way of answering does** (accuracy over the same questions):

| answering | fitted on dev, scored on test | ... and on the held-out clips | 5-fold CV over all clips |
|---|---|---|---|
| Qwen3-VL-8B from the frames, reason then answer | 54.8% (525/958) | 56.9% (400/703) | 54.3% |
| Qwen option probabilities (direct answer, no reasoning) | 59.2% (567/958) | 57.9% (407/703) | 57.7% |
| Qwen with the frames and the measured evidence | 60.4% (579/958) | 58.0% (408/703) | 60.1% |
| **the decision flow**, policy fitted on dev only | 63.5% (608/958) | 60.7% (427/703) | -- |
| **the decision flow**, policy fitted per kind on all clips | -- | -- | **64.4%** |

The first four rows are the honest held-out numbers (the policy had seen neither set): the flow is
+8.7 / +3.8 points over Qwen from the frames on test / held-out, and +3 over the best Qwen-only mode.
The last row is what to quote for the shipped policy: every question is scored by a policy fitted
without its clip. Which policy-fitting method to ship was chosen by that same cross-validation
(`policy_lab.py --pooled`): fitting each kind alone but making it beat the best untuned choice, 64.4%;
one policy for every kind, 62.7%; fitting each kind against a fixed baseline, 63.5%. Adding the
frames-only and evidence answers as extra soft votes did not help (65.7 vs 66.5 on dev CV) and is not in.

The held-out clips exposed what the small dev samples hid: a kind with 10 dev questions (camera translation)
or 38 (metric speed) let the measurement answer alone where it was at chance (23% and 29% on held-out).
Refitting on all 2,606 questions turned those gates off; they now answer with Qwen's option probabilities.
Per kind, with the shipped policy (*in-sample*, the CV above is the honest total):

**dev clips**

| kind | n | pipeline | frames-only | qwen letters | evidence | answers alone | alone acc |
|---|---|---|---|---|---|---|---|
| apparent_size | 44 | 30/44 (68%) | 24/44 (55%) | 25/44 (57%) | 31/44 (70%) | 44 | 30/44 (68%) |
| cam_subject_elevation | 39 | 23/39 (59%) | 16/39 (41%) | 18/39 (46%) | 23/39 (59%) | 39 | 23/39 (59%) |
| cam_subject_range | 135 | 84/135 (62%) | 54/135 (40%) | 67/135 (50%) | 84/135 (62%) | 0 | - |
| camera_path | 139 | 104/139 (75%) | 68/139 (49%) | 84/139 (60%) | 102/139 (73%) | 103 | 83/103 (81%) |
| camera_rhythm | 95 | 76/95 (80%) | 55/95 (58%) | 58/95 (61%) | 65/95 (68%) | 86 | 69/86 (80%) |
| camera_shake | 47 | 27/47 (57%) | 17/47 (36%) | 14/47 (30%) | 25/47 (53%) | 41 | 23/41 (56%) |
| camera_translation | 10 | 6/10 (60%) | 6/10 (60%) | 4/10 (40%) | 3/10 (30%) | 0 | - |
| camera_zoom | 37 | 35/37 (95%) | 30/37 (81%) | 30/37 (81%) | 35/37 (95%) | 0 | - |
| event_order | 52 | 38/52 (73%) | 38/52 (73%) | 36/52 (69%) | 31/52 (60%) | 0 | - |
| instance_count | 33 | 25/33 (76%) | 24/33 (73%) | 25/33 (76%) | 19/33 (58%) | 0 | - |
| landmark_relation | 34 | 25/34 (74%) | 23/34 (68%) | 25/34 (74%) | 24/34 (71%) | 0 | - |
| metric_distance | 74 | 31/74 (42%) | 21/74 (28%) | 31/74 (42%) | 20/74 (27%) | 0 | - |
| metric_speed | 38 | 14/38 (37%) | 16/38 (42%) | 14/38 (37%) | 8/38 (21%) | 0 | - |
| pair_relation | 28 | 18/28 (64%) | 18/28 (64%) | 13/28 (46%) | 15/28 (54%) | 0 | - |
| scene_revisit | 27 | 17/27 (63%) | 14/27 (52%) | 18/27 (67%) | 17/27 (63%) | 0 | - |
| subject_direction | 67 | 41/67 (61%) | 28/67 (42%) | 36/67 (54%) | 36/67 (54%) | 19 | 13/19 (68%) |
| subject_motion_profile | 18 | 14/18 (78%) | 14/18 (78%) | 10/18 (56%) | 16/18 (89%) | 0 | - |
| subject_visibility | 28 | 25/28 (89%) | 24/28 (86%) | 22/28 (79%) | 23/28 (82%) | 28 | 25/28 (89%) |
| **all** | 945 | 633/945 (67%) | 490/945 (52%) | 530/945 (56%) | 577/945 (61%) | 360 | 266/360 |

**test clips**

| kind | n | pipeline | frames-only | qwen letters | evidence | answers alone | alone acc |
|---|---|---|---|---|---|---|---|
| apparent_size | 39 | 24/39 (62%) | 17/39 (44%) | 19/39 (49%) | 22/39 (56%) | 38 | 23/38 (61%) |
| cam_subject_elevation | 36 | 23/36 (64%) | 21/36 (58%) | 22/36 (61%) | 22/36 (61%) | 34 | 21/34 (62%) |
| cam_subject_range | 135 | 91/135 (67%) | 71/135 (53%) | 74/135 (55%) | 91/135 (67%) | 0 | - |
| camera_path | 126 | 99/126 (79%) | 65/126 (52%) | 83/126 (66%) | 96/126 (76%) | 102 | 84/102 (82%) |
| camera_rhythm | 91 | 65/91 (71%) | 50/91 (55%) | 51/91 (56%) | 60/91 (66%) | 78 | 56/78 (72%) |
| camera_shake | 53 | 27/53 (51%) | 26/53 (49%) | 29/53 (55%) | 26/53 (49%) | 46 | 22/46 (48%) |
| camera_translation | 13 | 9/13 (69%) | 7/13 (54%) | 8/13 (62%) | 6/13 (46%) | 0 | - |
| camera_zoom | 36 | 35/36 (97%) | 28/36 (78%) | 28/36 (78%) | 35/36 (97%) | 0 | - |
| event_order | 43 | 33/43 (77%) | 33/43 (77%) | 33/43 (77%) | 32/43 (74%) | 0 | - |
| instance_count | 31 | 24/31 (77%) | 20/31 (65%) | 24/31 (77%) | 17/31 (55%) | 0 | - |
| landmark_relation | 43 | 29/43 (67%) | 26/43 (60%) | 29/43 (67%) | 25/43 (58%) | 0 | - |
| metric_distance | 79 | 33/79 (42%) | 26/79 (33%) | 33/79 (42%) | 22/79 (28%) | 0 | - |
| metric_speed | 38 | 20/38 (53%) | 20/38 (53%) | 20/38 (53%) | 9/38 (24%) | 0 | - |
| pair_relation | 42 | 27/42 (64%) | 27/42 (64%) | 27/42 (64%) | 17/42 (40%) | 0 | - |
| scene_revisit | 27 | 19/27 (70%) | 12/27 (44%) | 14/27 (52%) | 19/27 (70%) | 0 | - |
| subject_direction | 73 | 31/73 (42%) | 36/73 (49%) | 33/73 (45%) | 35/73 (48%) | 17 | 8/17 (47%) |
| subject_motion_profile | 14 | 11/14 (79%) | 11/14 (79%) | 11/14 (79%) | 10/14 (71%) | 0 | - |
| subject_visibility | 39 | 32/39 (82%) | 29/39 (74%) | 29/39 (74%) | 35/39 (90%) | 39 | 32/39 (82%) |
| **all** | 958 | 632/958 (66%) | 525/958 (55%) | 567/958 (59%) | 579/958 (60%) | 354 | 246/354 |

**held-out clips**

| kind | n | pipeline | frames-only | qwen letters | evidence | answers alone | alone acc |
|---|---|---|---|---|---|---|---|
| apparent_size | 24 | 19/24 (79%) | 19/24 (79%) | 21/24 (88%) | 19/24 (79%) | 23 | 19/23 (83%) |
| cam_subject_elevation | 22 | 16/22 (73%) | 15/22 (68%) | 14/22 (64%) | 16/22 (73%) | 20 | 15/20 (75%) |
| cam_subject_range | 97 | 58/97 (60%) | 52/97 (54%) | 55/97 (57%) | 58/97 (60%) | 0 | - |
| camera_path | 68 | 47/68 (69%) | 41/68 (60%) | 44/68 (65%) | 50/68 (74%) | 53 | 38/53 (72%) |
| camera_rhythm | 48 | 37/48 (77%) | 24/48 (50%) | 25/48 (52%) | 34/48 (71%) | 41 | 30/41 (73%) |
| camera_shake | 39 | 19/39 (49%) | 12/39 (31%) | 9/39 (23%) | 19/39 (49%) | 36 | 18/36 (50%) |
| camera_translation | 39 | 26/39 (67%) | 21/39 (54%) | 24/39 (62%) | 22/39 (56%) | 1 | 0/1 (0%) |
| camera_zoom | 37 | 31/37 (84%) | 26/37 (70%) | 26/37 (70%) | 31/37 (84%) | 0 | - |
| event_order | 33 | 25/33 (76%) | 25/33 (76%) | 22/33 (67%) | 20/33 (61%) | 0 | - |
| landmark_relation | 28 | 19/28 (68%) | 20/28 (71%) | 19/28 (68%) | 15/28 (54%) | 0 | - |
| metric_speed | 121 | 48/121 (40%) | 44/121 (36%) | 48/121 (40%) | 27/121 (22%) | 0 | - |
| pair_relation | 52 | 30/52 (58%) | 30/52 (58%) | 30/52 (58%) | 27/52 (52%) | 0 | - |
| scene_revisit | 22 | 13/22 (59%) | 10/22 (45%) | 13/22 (59%) | 13/22 (59%) | 0 | - |
| subject_motion_profile | 38 | 32/38 (84%) | 32/38 (84%) | 29/38 (76%) | 31/38 (82%) | 0 | - |
| subject_visibility | 35 | 29/35 (83%) | 29/35 (83%) | 28/35 (80%) | 26/35 (74%) | 34 | 28/34 (82%) |
| **all** | 703 | 449/703 (64%) | 400/703 (57%) | 407/703 (58%) | 408/703 (58%) | 208 | 148/208 |

Reading it. The measurement pays off most on the camera kinds (path, rhythm, shake, zoom) and on visibility,
size, revisit and range, ahead of every Qwen-only mode; for some of them (zoom, range, revisit) the fitted
policy hands the measurement to Qwen as evidence instead of letting it answer alone, which scored as well
or better. For metric distance and speed, landmark relation, pair relation, instance count and event
order the measurement does not beat Qwen's own answer; the policy leaves those to Qwen. There the weak spot
is the measurement itself, not the routing.

**Coverage of the whole benchmark** (`census.py`, 7,000 questions): rules read 3533; Qwen3-VL named the kind of
the rest from the text alone (agreeing with the rules on 99% of 884 rule-read questions). 91% of questions
are routed to a measurement kind, 9% to Qwen alone (semantic, tracked-object counts, unread). This is
the routing, not the accuracy: 12% of all questions land on `landmark_relation` and 10% on
`pair_relation`, the kinds with the weakest measurements.

**Metric scale.** D4RT was run on every clip as a second source of metres-per-unit (`prep_d4rt.py`: median of
D4RT depth over Point4D depth on the 16 sampled frames). It came out at 0.45x DA3-Nested's scale and agreed
*worse* with the benchmark's metric answers on dev (median log error 0.89 vs 0.60), so calibration kept
DA3 (`scene4d.SCALE_SOURCE`). A different D4RT protocol (the full video, native aspect) has not been tried.

## Known gaps (what the graph does not do, in the order they would pay off)

- **Action events.** "Which happens first: the person sits down or the dog turns?" has no geometric
  signal; only camera motions, appearing/leaving the frame and "starts moving" are measurable. The
  natural next tool is a per-frame yes/no detector run on the entity's crop (Qwen over the 16 frames,
  one event at a time), with the order compared deterministically.
- **Ordinal instances** ("the first camel", "the second camel") ground to the same best-tracked
  instance; choosing the k-th by first appearance or by left-to-right position is not implemented.
- **How much, and when, does the camera move?** ("stationary or move, only in the first half?") has a
  measurement (camera path per phase) but no kind yet; the camera orientation-change questions
  ("negligible / small / large yaw") likewise.
- **Grounding.** SAM3 is prompted on frame 0 only, so a subject that enters later is not found; a
  synonym SAM3 does not know ("kitesurfer", "go-kart") is not retried with another noun. A fallback
  *ladder* (drop the clause after the first preposition or participle, then the head noun) rescues
  a phrase that finds nothing, and the result then says it used a simpler prompt.
- **Metric scale** rests on one DA3-Nested ratio per clip and the answers are themselves rough; a
  metric-depth ensemble would only help if the options were farther apart.
- **Frame rates** are verified for five sources; TotalRecon, Spring and PointOdyssey are flagged,
  not guessed.

## Next tool (suggested by the user): Orient-Anything-V2 for object orientation

`third-party-tools/Orient-Anything-V2` gives an object's facing per frame, which SAM3 masks and
Point4D depth cannot. It targets three measured gaps at once: (1) **forward/backward** of a subject
("moves forward" is probably heading, not motion toward the camera, which is why it is at chance in
every frame), by comparing the facing direction with the displacement; (2) `side_filmed` (63 rows in
the census: which side of the object faces the camera); (3) `orbit_direction` (71 rows). Plan: crop
each entity by its SAM3 mask, run OAv2 on the sampled frames, lift the yaw into the first camera's
frame with Point4D's pose, and add it as a signal (`facing`) that `subject_direction` scores the
z-words against and that `side_filmed`/`orbit_direction` answer directly.
