# Dyn-Bench task-type taxonomy

Deterministic scans over all 7000 VQA rows; blind dual-axis LLM classification of 600 of 600 stratified-sampled questions. 1000 clips resolved to media, 0 unresolved.

**Headline:** 500 of 600 sampled questions (83%) fall outside the four measurement buckets `d4rt_agent/agent_v4` implements. 442 of those name a specific quantity AgentV4 does not produce, 52 need no measurement at all (`semantic_only`), and 6 fit no bucket in the list.

## 1. What the dataset ships

| coarse task | rows | distinct videos | distinct questions | answer histogram |
| --- | --- | --- | --- | --- |
| `cameraqa` | 3000 | 1000 | 870 | {'A': 1237, 'B': 738, 'C': 604, 'D': 421} |
| `qa` | 2000 | 1000 | 1698 | {'A': 759, 'B': 637, 'C': 499, 'D': 105} |
| `sceneqa` | 2000 | 1000 | 1953 | {'A': 728, 'B': 820, 'C': 354, 'D': 98} |

| source dataset | rows |
| --- | --- |
| `ytvis` | 3528 |
| `sav` | 2051 |
| `dynpose` | 595 |
| `davis` | 574 |
| `totalrecon` | 91 |
| `dynamic_replica` | 84 |
| `point_odyssey` | 42 |
| `spring` | 35 |

## 2. The template scan, and why it is not enough

The DSI-Bench pass settled its taxonomy with an exhaustive regex template scan: 1769 rows collapsed to 34 templates, so every question shape was a countable fact. Rerunning that scan here is the first thing this pass did, and the result is the reason the rest of it looks different.

| coarse task | rows | distinct templates | top-25 coverage |
| --- | --- | --- | --- |
| `cameraqa` | 3000 | 758 | 2098 (70%) |
| `qa` | 2000 | 1578 | 333 (17%) |
| `sceneqa` | 2000 | 1925 | 76 (4%) |

`cameraqa` is semi-templated and the scan works on it. `qa` and `sceneqa` are free-form generated text — 1578 and 1925 distinct templates for 2000 rows each — and no amount of regex tuning will collapse them. Hence the LLM open-coding pass below, and hence the regex *families* in `dyn_bench_data.FAMILIES`, which are a sampling stratum rather than a taxonomy.

## 3. Sampling

600 questions, 466 distinct videos, {'cameraqa': 200, 'qa': 200, 'sceneqa': 200} per coarse task, equal allocation across question families capped by availability, with 20% of each family's quota reserved for the four rare source datasets (dynamic_replica, point_odyssey, spring, totalrecon).

| source dataset | in sample | in population |
| --- | --- | --- |
| `ytvis` | 305 | 3528 |
| `sav` | 163 | 2051 |
| `dynpose` | 36 | 595 |
| `davis` | 36 | 574 |
| `totalrecon` | 22 | 91 |
| `dynamic_replica` | 20 | 84 |
| `spring` | 9 | 35 |
| `point_odyssey` | 9 | 42 |

## 4. Axis A — the paper's nine sub-tasks

The paper defines these; the released parquet does not ship them. Recovered here by blind classification.

**Dynamic Inter-Object Perception** — 238 of 600 sampled questions

| sub-task | n | share |
| --- | --- | --- |
| `Act. & Obj. Desc.` | 31 | 5% |
| `Move. & Temp. Dyn.` | 129 | 22% |
| `Spatial Rel. & Change` | 78 | 13% |

**Dynamic Object-Scene Tracking** — 123 of 600 sampled questions

| sub-task | n | share |
| --- | --- | --- |
| `Mov. Patterns & Traj.` | 54 | 9% |
| `Spatial Rel. & Comp.` | 50 | 8% |
| `Scene Focus & Dyn.` | 19 | 3% |

**Dynamic Camera-Object Reasoning** — 237 of 600 sampled questions

| sub-task | n | share |
| --- | --- | --- |
| `Cam. Motion & Orient.` | 58 | 10% |
| `Cam-Obj. Interaction` | 100 | 17% |
| `Temp. & Visual Change` | 79 | 13% |

`OTHER`: 2 questions fit none of the nine.

### Does the shipped coarse label predict the paper's level?

| coarse task | hypothesised level | n | agree | OTHER | agreement |
| --- | --- | --- | --- | --- | --- |
| `cameraqa` | Dynamic Camera-Object Reasoning | 200 | 178 | 0 | 89% |
| `qa` | Dynamic Inter-Object Perception | 200 | 147 | 0 | 74% |
| `sceneqa` | Dynamic Object-Scene Tracking | 200 | 95 | 2 | 48% |

The three shipped task files are not a clean partition of the paper's three levels. `cameraqa` is close to one, `qa` is mostly one, and `sceneqa` is not: fewer than half of its questions read as object-scene tracking, and most of the rest read as inter-object perception. Anyone reporting per-level Dyn-Bench numbers by grouping on the shipped `task` column is not reporting the paper's levels.

## 5. Axis B — measurement buckets

What physical quantity, in what reference frame, would have to be measured for the question to be decidable. The first four buckets are what AgentV4 implements today; the rest are candidates this pass was run to confirm or rule out.

*Method note.* The codebook was revised once. The first pass left 21 of 600 questions in `OTHER`, and reading them showed one missing distinction stated repeatedly: a scene object's trajectory expressed in the **camera's** frame, or in a frame fixed to the **static scene**, as opposed to the object's own axes. `traj_subject_camera_frame` and `traj_subject_scene_frame` were added and all 600 questions were reclassified under the extended list, so every count below comes from one codebook. `OTHER` fell from 21 to 6.

| bucket | n | share | AgentV4 coverage | quantity measured |
| --- | --- | --- | --- | --- |
| `traj_subject_own_frame` | 10 | 2% | `traj3d_buckets.traj_subject_own_frame` | Trajectory of a scene object, expressed in that object's own starting frame (its own forward/lateral/vertical axes) |
| `traj_camera_own_frame` | 18 | 3% | `traj3d_buckets.traj_camera_own_frame` | Trajectory of the camera, expressed in the camera's own starting frame (forward/lateral/vertical translation plus pan/tilt/roll) |
| `sub_obs_distance_change` | 47 | 8% | `traj3d_buckets.sub_obs_distance_change` | Whether the range between a scene object and the camera grew, shrank or held, as an ordinal judgement rather than a number |
| `orient_camera_subject_frame` | 25 | 4% | `traj3d_buckets.orient_camera_subject_frame` | Which side of a scene object the camera is on (front/back/left/right sector of the object's own facing), and how that sector changes |
| `traj_subject_camera_frame` | 30 | 5% | **new work** | Trajectory of a scene object expressed in the CAMERA's frame -- nearer/farther, left/right, up/down as the camera sees it, not in the object's own axes |
| `traj_subject_scene_frame` | 53 | 9% | **new work** | Trajectory of a scene object relative to the static scene around it -- a fixed landmark, a road, a court, the room -- rather than its own axes or the camera's |
| `subject_subject_relation` | 56 | 9% | **new work** | Distance or bearing between TWO scene objects, and how it changes -- neither of them being the camera |
| `metric_distance` | 17 | 3% | **new work** | Absolute distance in real-world units (metres) between two entities at a named moment -- a number, not an ordinal judgement |
| `metric_speed` | 17 | 3% | **new work** | Absolute speed in real-world units (m/s) of a scene object over a named interval |
| `instance_count` | 18 | 3% | **new work** | How many distinct instances of a category are present in the clip |
| `event_order` | 36 | 6% | **new work** | The onset times of two named events, compared -- which happened first |
| `camera_zoom_vs_dolly` | 22 | 4% | **new work** | Whether the camera's focal length changed, separated from whether the camera translated toward or away from the scene |
| `camera_motion_profile` | 41 | 7% | **new work** | The camera's speed as a function of time -- steady vs. accelerating vs. stop-start, and high-frequency jitter (handheld shake) |
| `subject_motion_profile` | 38 | 6% | **new work** | A scene object's speed as a function of time -- whether it moves at all, and whether that motion is continuous, intermittent or a single brief burst |
| `subject_orient_over_time` | 17 | 3% | **new work** | Change in which way a scene object faces, over time, independent of where it is and where the camera is |
| `subject_visibility` | 35 | 6% | **new work** | Whether a scene object is inside the image frame, where in the frame it sits, and at which times it enters or leaves |
| `apparent_size_change` | 21 | 4% | **new work** | The object's projected size in the image plane over time -- an image-plane quantity, which range and focal length both feed into |
| `camera_elevation_rel_subject` | 13 | 2% | **new work** | The camera's elevation relative to the subject -- shot from above, at eye level, or from below |
| `scene_revisit` | 28 | 5% | **new work** | Whether the view at the end of the clip shows the same place as the view at the start |
| `semantic_only` | 52 | 9% | **new routing** (no measurement) | No geometric measurement decides it -- the question is about what is happening, what something is, or how a scene reads |
| `OTHER` | 6 | 1% | n/a | none of the above fit |

### What each bucket looks like in the data

- **`traj_subject_own_frame`** (10)
  - *From the first frame to the frame at 4.67 seconds, how does the dog’s position change?*
  - *What is the main movement direction of the dog in the video?*
- **`traj_camera_own_frame`** (18)
  - *What is the primary trajectory movement of the camera in the video?*
  - *Is the camera stationary or does it move during the first 25% of the video (approx. 0-3s)?*
- **`sub_obs_distance_change`** (47)
  - *How does the distance between the camera and the main subject change?*
  - *Over the entire video, does the camera's distance to the orange cat change significantly?*
- **`orient_camera_subject_frame`** (25)
  - *Which side of the object is primarily filmed by the camera?*
  - *From the beginning (t≈0s) to the end (t≈11s), does the camera's position relative to the man standing with hands behind his back change from one lateral sector to another?*
- **`traj_subject_camera_frame`** (30)
  - *Between the start (0.0s) and the end (1.63s), how does the kitesurfer’s position relative to the camera change?*
  - *From the beginning to the end of the video, how does the position of the person in the blue shirt change?*
- **`traj_subject_scene_frame`** (53)
  - *Between 0.0s and 2.17s, how does the person’s position relative to the snowy slope change?*
  - *Between the start (0.0s) and the end (9.83s), how does the person's position relative to the front door change?*
- **`subject_subject_relation`** (56)
  - *Do the two people get closer or farther apart during the video?*
  - *Between the start and the end of the video, how does the person's position relative to the lamp change?*
- **`metric_distance`** (17)
  - *At the 5-second mark, what is the approximate distance between the person and the small animal?*
  - *At the 1.0-second mark, what is the approximate distance between the two cars?*
- **`metric_speed`** (17)
  - *What is the approximate average speed of the gray SUV from its first to last appearance?*
  - *What is the approximate average speed of the person’s movement from the start to the end of the video?*
- **`instance_count`** (18)
  - *How many distinct people appear in the video?*
  - *How many distinct large objects are tracked in the video?*
- **`event_order`** (36)
  - *Which happens first: the camera begins a noticeable rightward pan, or the person in the dark jacket enters the scene?*
  - *Which happens first: the camera starts a noticeable rightward pan or the man wearing a white shirt and brown vest begins to bend down to pick up shoes?*
- **`camera_zoom_vs_dolly`** (22)
  - *Was zooming (magnification or reduction) used during filming?*
  - *Was zooming (in or out) used during recording?*
- **`camera_motion_profile`** (41)
  - *What is the movement speed (movement rhythm) of the camera?*
  - *Is there noticeable camera shake in the video?*
- **`subject_motion_profile`** (38)
  - *Does the large passenger jet remain stationary or move during the video?*
  - *Does the person’s movement across the court occur continuously or in short bursts?*
- **`subject_orient_over_time`** (17)
  - *What happens to the kite surfer’s orientation and motion as the scene evolves?*
  - *What happens to the spacecraft’s orientation as it nears the planet?*
- **`subject_visibility`** (35)
  - *In which area of the frame is the main object located for the majority of the time?*
  - *At the end of the video, is the red fox still in the frame?*
- **`apparent_size_change`** (21)
  - *Visually, does the main object get larger or smaller over time?*
  - *Does the child wearing a yellow vest and red shoes appear visually larger or smaller over time?*
- **`camera_elevation_rel_subject`** (13)
  - *What is the camera angle relative to the person in the pink snowsuit?*
  - *What is the camera angle relative to the dog with white and black fur?*
- **`scene_revisit`** (28)
  - *Is the background scene the same at the start and end of the video?*
  - *Are the scenes (background environment) at the beginning and end of the video the same?*
- **`semantic_only`** (52)
  - *What best describes the main activity in the video?*
  - *How does the girl's interaction with the kitten evolve from the beginning to the end of the video?*
- **`OTHER`** (6)
  - *Around the blue and white tram, what is the net orbital direction of the camera from a top-down view between t=0.0s and t=3.0s?*
  - *What happens to the small white car visible in the distance near the trees as the scene unfolds?*

## 6. Scaled back to all 7000 rows

The draw was equal-allocation across question families, so the shares above are shares of the sample. Post-stratifying each (coarse task, family) cell by its true population size, counted exhaustively in `stats.json`, gives the estimate for the benchmark as a whole.

| bucket | AgentV4 coverage | estimated rows (of 7000) | share |
| --- | --- | --- | --- |
| `subject_subject_relation` | **new work** | 864 | 12% |
| `traj_subject_scene_frame` | **new work** | 773 | 11% |
| `camera_motion_profile` | **new work** | 650 | 9% |
| `traj_camera_own_frame` | `traj3d_buckets.traj_camera_own_frame` | 606 | 9% |
| `sub_obs_distance_change` | `traj3d_buckets.sub_obs_distance_change` | 593 | 8% |
| `semantic_only` | **new routing** (no measurement) | 495 | 7% |
| `subject_motion_profile` | **new work** | 424 | 6% |
| `subject_visibility` | **new work** | 321 | 5% |
| `camera_zoom_vs_dolly` | **new work** | 238 | 3% |
| `event_order` | **new work** | 234 | 3% |
| `orient_camera_subject_frame` | `traj3d_buckets.orient_camera_subject_frame` | 228 | 3% |
| `camera_elevation_rel_subject` | **new work** | 227 | 3% |
| `metric_distance` | **new work** | 225 | 3% |
| `instance_count` | **new work** | 216 | 3% |
| `metric_speed` | **new work** | 203 | 3% |
| `apparent_size_change` | **new work** | 190 | 3% |
| `traj_subject_camera_frame` | **new work** | 183 | 3% |
| `scene_revisit` | **new work** | 164 | 2% |
| `traj_subject_own_frame` | `traj3d_buckets.traj_subject_own_frame` | 83 | 1% |
| `OTHER` | n/a | 64 | 1% |
| `subject_orient_over_time` | **new work** | 18 | 0% |

Estimated from 7000 of 7000 rows; 0 rows fall in (task, family) cells the sample did not reach and are left out rather than averaged in.

## 7. Cross-tab — paper sub-task × measurement bucket

- **`Act. & Obj. Desc.`** (31) → `semantic_only` 22, `instance_count` 9
- **`Move. & Temp. Dyn.`** (129) → `subject_motion_profile` 37, `event_order` 36, `metric_speed` 17, `subject_orient_over_time` 16, `instance_count` 9, `semantic_only` 8, `traj_subject_scene_frame` 4, `traj_subject_camera_frame` 1, `subject_subject_relation` 1
- **`Spatial Rel. & Change`** (78) → `subject_subject_relation` 55, `metric_distance` 14, `traj_subject_scene_frame` 8, `subject_visibility` 1
- **`Mov. Patterns & Traj.`** (54) → `traj_subject_camera_frame` 22, `traj_subject_scene_frame` 20, `traj_subject_own_frame` 10, `subject_motion_profile` 1, `subject_orient_over_time` 1
- **`Spatial Rel. & Comp.`** (50) → `traj_subject_scene_frame` 19, `subject_visibility` 16, `semantic_only` 7, `scene_revisit` 3, `apparent_size_change` 3, `OTHER` 2
- **`Scene Focus & Dyn.`** (19) → `semantic_only` 15, `traj_subject_scene_frame` 2, `subject_visibility` 1, `traj_subject_camera_frame` 1
- **`Cam. Motion & Orient.`** (58) → `camera_motion_profile` 40, `traj_camera_own_frame` 18
- **`Cam-Obj. Interaction`** (100) → `sub_obs_distance_change` 47, `orient_camera_subject_frame` 25, `camera_elevation_rel_subject` 13, `traj_subject_camera_frame` 6, `subject_visibility` 4, `metric_distance` 3, `OTHER` 1, `camera_motion_profile` 1
- **`Temp. & Visual Change`** (79) → `scene_revisit` 25, `camera_zoom_vs_dolly` 22, `apparent_size_change` 18, `subject_visibility` 13, `OTHER` 1
- **`OTHER`** (2) → `OTHER` 2

## 8. Sampling families vs assigned buckets

| sampling family | regex intent | n | buckets the LLM assigned |
| --- | --- | --- | --- |
| `unfamilied` | matched no family regex | 66 | `semantic_only` 13, `scene_revisit` 10, `traj_subject_scene_frame` 10 |
| `activity_semantic` | what is happening, described rather than measured | 55 | `semantic_only` 32, `traj_subject_scene_frame` 9, `subject_visibility` 5 |
| `object_object_relation` | one scene object's position relative to another | 53 | `subject_subject_relation` 30, `traj_subject_scene_frame` 18, `sub_obs_distance_change` 2 |
| `subject_trajectory` | where a scene object went | 51 | `traj_subject_camera_frame` 12, `traj_subject_scene_frame` 11, `traj_subject_own_frame` 7 |
| `camera_trajectory` | where the camera went, in its own frame | 40 | `traj_camera_own_frame` 12, `traj_subject_camera_frame` 9, `camera_motion_profile` 5 |
| `event_order` | which of two named events happens first | 35 | `event_order` 35 |
| `subject_orientation` | which way a scene object faces, and how that changes | 34 | `subject_orient_over_time` 17, `orient_camera_subject_frame` 13, `traj_subject_camera_frame` 1 |
| `camera_subject_distance` | range between camera and subject, ordinally | 33 | `sub_obs_distance_change` 33 |
| `timestamped` | state at or between named timestamps | 33 | `subject_subject_relation` 9, `traj_camera_own_frame` 5, `camera_elevation_rel_subject` 4 |
| `visibility` | whether and where the subject is present in the image frame | 26 | `subject_visibility` 20, `subject_subject_relation` 4, `apparent_size_change` 1 |
| `apparent_size` | projected size in the image plane | 20 | `apparent_size_change` 16, `camera_zoom_vs_dolly` 3, `OTHER` 1 |
| `camera_rhythm` | camera speed profile over time | 19 | `camera_motion_profile` 18, `subject_motion_profile` 1 |
| `camera_shake` | high-frequency camera jitter | 18 | `camera_motion_profile` 17, `subject_motion_profile` 1 |
| `instance_count` | how many distinct instances | 18 | `instance_count` 18 |
| `metric_speed` | absolute speed, answered in physical units | 17 | `metric_speed` 17 |
| `motion_continuity` | whether the subject moves at all, and how continuously | 17 | `subject_motion_profile` 17 |
| `metric_distance` | absolute distance, answered in physical units | 17 | `metric_distance` 17 |
| `shooting_angle` | camera elevation or bearing relative to the subject | 16 | `camera_elevation_rel_subject` 9, `orient_camera_subject_frame` 7 |
| `camera_zoom` | whether focal length changed | 16 | `camera_zoom_vs_dolly` 16 |
| `scene_revisit` | whether start and end show the same place | 16 | `scene_revisit` 16 |

## 9. Review queue

544 of 600 classifications queued for review, 544 still awaiting `human_verdict`.

| reason queued | n |
| --- | --- |
| candidate bucket not implemented in AgentV4 | 494 |
| level mismatch | 178 |
| bucket is OTHER | 6 |
| paper_task is OTHER | 2 |

## 10. Data facts worth carrying into any Dyn-Bench evaluation

| coarse task | rows | answer is the longest option | answer is A |
| --- | --- | --- | --- |
| `cameraqa` | 3000 | 24.3% | 41.2% |
| `qa` | 2000 | 34.4% | 38.0% |
| `sceneqa` | 2000 | 64.7% | 36.4% |
| `all` | 7000 | 38.7% | 38.9% |

These are properties of the shipped option text, not model results -- recorded because they bound how much of this benchmark is a vision task.

4 rows ship a lowercase `answer` letter and are folded to upper case on load; 0 rows have a structural issue that survives loading (see `stats.json`).

## 11. Scope and next step

This pass covers the 7000-row VQA track only. The 3000 dynamic-object-grounding rows (`cameramask`/`objmask`/`scenemask`, scored by J&F over per-frame masks) are untouched, and are the largest single capability gap: AgentV4's `final_answer` action is text-only (`kind` must be `"text"`), so it has no way to emit geometry at all.

Turning this taxonomy into AgentV5 buckets and tools is not designed here. See `BUCKET_REQUIREMENTS_V5.md` for the per-bucket input/output specs and `AGENT_V5_GAPS.md` for what has to change in `d4rt_agent/`.
