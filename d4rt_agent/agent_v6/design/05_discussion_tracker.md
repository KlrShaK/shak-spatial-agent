# 05 — Discussion tracker for Shak's findings

Every point from [04_shak_findings.md](04_shak_findings.md) has an ID, my first take, the evidence
I checked, and a status (`open`, `agreed`, `changed`, `dropped`). The points are grouped into five
themes; each theme becomes one design file once we settle it. "My take" is a starting position
for discussion, not a decision.

## Theme 1 — What a tool returns

**Settled 2026-10-17** → [10_tool_reports.md](10_tool_reports.md). Also decided there: reports
never mention options; options visible from planning; merged-phase tables with majority motion
first, then minority motions; full contract on a tool's first report.

| ID | Your point | My take | Evidence | Status |
|---|---|---|---|---|
| T1, P3 | Outputs are raw numbers, not an LLM-readable summary; keep and build on V4's style | Agree. V5's track does print an "In time order" line, but it comes after an axis table, star marks and caveats. V4 led with one sentence giving the dominant direction, net and ranked components in the requested frame, then the table, then a bucket-specific reading contract. Proposal: every V6 report is **headline → short table (≤10 rows) → only the reading rules that apply**. | V4 report: `results/agent_v4/AGENT_V4_RESULTS.md` §1; V5 track renderer: `agent_v5/tools.py:669` | **agreed** 2026-10-17: headline commits to direction words, with size and noise floor beside them; then table; then report-specific reading notes ([10](10_tool_reports.md)) |
| P9 | Prefer whole-window profiles over the model choosing timestamps | Agree strongly. V4 always reported 10 equal-time segments over the whole clip. In V6, tools default to the whole clip; a window is set only when the question names a time, and the host derives it from the question. | V5 compositional: 178 refused calls over window formats (03 §6) | **agreed** 2026-10-17 (your proposal) |

## Theme 2 — The toolset

**Settled 2026-10-17** → [11_toolset.md](11_toolset.md): eight measurement tools plus `get_frames` (#41). `detect_all_entities`,
`track_camera`, `track_entity(main, others, frame_of_reference)`, `track_orientation`,
`track_multiple_entities`, `track_camera_zoom`, `check_scene_revisit`, `get_segmentation_mask`;
events, actions and the counting decision go to the VLM. Log #10, #12, #13, #15, #17–25.
Confirmations G1–G3 pending. The draft table below is kept as history.

Your overall claim, that accuracy comes from using few tools well rather than from many tools, fits
the history: v2 and v3 lost to the control, V4 won with 4 buckets, and V5's 23 kinds never beat
frames-only Qwen on Dyn.

| ID | Your point | My take | Evidence | Status |
|---|---|---|---|---|
| T2 | Orientation and projected rotation overlap; one rotation tool for the subject(s) over an interval | Agree. One `rotation` report: heading over time, accumulated (unwrapped) turn, facing at an instant on request. The real problem is OAV2 finding no front, which is the top unavailability reason in both V4 and V5. We need a fallback, e.g. heading from the direction of travel for moving subjects. | `orientation` unavailable on 102/252 VLM4D calls and 36/96 DSI calls; `projected_rotation` on 6/7 | **agreed in principle** 2026-10-17 (your proposal); fallback when OAV2 fails: VLM facing labels (#54); motion never depends on orientation (#56) |
| T3 | Kinematics and motion profile are redundant with the track; rename if speed-only | Agree: fold into the track report as a speed section (per-segment speed, speeds up / slows / stops). Keep metric speed separate and flagged, since its estimator is wrong (24.7%). | `subject_motion_profile` 71.4% on Dyn: the content works, the packaging changes | **agreed in principle** 2026-10-17 (your proposal) |
| T4, T6 | Camera profile should be "camera track" (world by default, or relative to a subject); camera motion vs profile is confusing | Agree. One `camera_track` = V4's `traj_camera_own_frame` (translation and pan/tilt/roll per segment in the gravity-levelled start frame) plus tempo and jitter lines. The two V5 tools could disagree about whether the camera moved at all. | V5 DSI camera own-frame 45% vs V4 78% (01); static-label conflicts (`WORKFLOW_REVIEW.md` §3) | **agreed in principle** 2026-10-17 (your proposal); "relative to a subject" depends on Q2.2 |
| T5 | Orientation: when to call orbit, when rotation | Split by question: the subject's own turning → `rotation`; which side of the subject the camera is on, and how that changes (V4's c5 bucket, V5's viewpoint/orbit) → `viewpoint`. Facing at one instant is an option of `rotation`, not a tool. | V4 c5 bucket worked (std 55%, 4-aug 53%) | **agreed** 2026-10-17: own turning → `subject_rotation`; the camera's side of the subject → `relative_track(camera, subject)` |
| T7, T16 | Rename relative position, or merge it into the track; main entity + other entity | Design choice to make together. My lean: one track tool with a `relative_to` argument (world by default, camera, another entity, or a region). The report gives distance and direction of the subject from the reference over time. This replaces `relative_position`, pair `distance` and `region_relation` with one familiar tool. The alternative is a separate `relative_track` tool with a clearer name and one more menu item. | Scalar distance cannot decide left/right: `pair_relation` 56.4% vs frames 62.8% | **agreed** 2026-10-17: one `relative_track(main, reference)` ([11](11_toolset.md)) |
| T8 | Elevation is redundant | Agree to remove it as a tool. It scores below frames-only (35.4 vs 45.1 on 82 Dyn questions) and DSI never needs it. If kept at all, it becomes one line in `viewpoint`. | 01, Dyn per-kind table; D7 study: evidence 50.5 vs frames 49.5 | **agreed in principle** 2026-10-17 (your proposal) |
| T9 | Region relation is covered by the track | Mostly agree. One nuance: distance to an extended region (road, wall, net) is distance to its surface, not to its centre. That fits as `relative_to=<region>` in the track (T7). | Used on 51 Dyn and 10 VLM4D questions | **agreed** 2026-10-17: a region is a possible reference of `relative_track` |
| T10 | Group geometry: rename; all tracked instances; how distance and speed evolve between them | Agree. Rename it, e.g. `group_track`. Input: every instance of a category. Output: per-instance net direction, spread (clustering vs scattering) over time, counts by direction. This also covers the per-instance filtering that `evaluate_predicate` tried to do (T14, P12). | Seals "scatter vs gather" and two-dog regressions (`WORKFLOW_REVIEW.md` §6, pilot `MANUAL_REVIEW.md`) | **agreed in principle** 2026-10-17 (your proposal); name in theme 2. Caution: counts from SAM3 instances scored 43.8% vs letters 81.2% (table below) |
| T11 | Census / view state: detect all entities with visibility timelines, before family/kind; helps grounding and false positives | Agree, and make it the first host step (theme 3). One caveat: SAM3 is text-prompted, so it detects every *named* entity (nouns from question and options, maybe a caption), not every object. | VLM4D false-positive questions 24.4%; an absence option is offered on 795 questions, gold on 12% | **agreed** 2026-10-17: SAM3 on nouns from question, options and the caption (P8) |
| T12 | Keep camera zoom | Agree. | 91.2% vs frames 79.4% (Point4D); 72.1% with D4RT, so backbone matters | **agreed** 2026-10-17 |
| T13 | How does scene revisit work? Maybe VLM-only | It is geometric: the share of the first frame's 3D points seen again, at a consistent depth, from the last camera. It works: 71.7% vs frames-only 58.3% on 60 Dyn questions (76.7% with D4RT). I'd keep it and rename it `view_overlap`. | `agent_v5/tools.py:627` | **agreed** 2026-10-17: keep, under a self-describing name (name settled in theme 2) |
| T14 | Remove evaluate predicate | Agree as a model-facing tool. Its computation (which instances move right in a given frame) moves into `group_track`, so the model reads "4 of 6 fish move right in the camera's view" instead of calling a filter. | 15 calls in the compositional VLM4D run | **agreed in principle** 2026-10-17 (your proposal) |
| T15 | Remove event tools; event recognition is visual, ask the VLM | Agree for actions (raising an arm, dribbles). Geometric transitions (starts or stops moving, enters or leaves view) can stay as lines in track and census reports, not as tools. | `event_order` 67.4% vs frames 78.3%; `event_onset` unavailable on 57/179 Dyn calls; D7 study: evidence 73.2 vs frames 76.3 | **agreed in principle** 2026-10-17 (your proposal). My addition, object if wrong: geometric transitions stay as report lines |
| P7, P11 | Names must state their purpose; each tool gets a one-line when / when-not example | Agree. Naming pattern `<who>_<what>` (camera_track, subject_track, group_track, viewpoint, rotation, zoom, view_overlap, scene_inventory). V4's bucket descriptions already said when not to use each ("Needs no subject", "No directions involved"). | `agent_v4/buckets.py` | **mostly agreed** 2026-10-17: names state the reference type (#16); a few names still open (Q2.12). The one-line cue and example per frame (#15) is the first instance of P11 |

**Does measuring help, per question kind?** These come from the D7 study's 2,800-question Dyn eval
(`dyn_dataset_exploration/outputs/decomposition_eval/eval_rows.json`). Kinds are the study's rule
labels. "Evidence" means Qwen answering with our measurement in the prompt; "letters" means Qwen's
option log-probabilities. Four options, so chance is 25%.

| Kind | n | Frames only | Letters | Evidence | Reading |
|---|---:|---:|---:|---:|---|
| scene_revisit | 56 | 50.0 | 60.7 | **71.4** | measurement helps |
| camera_zoom | 74 | 77.0 | 77.0 | **89.2** | measurement helps |
| cam_subject_elevation | 101 | 49.5 | 60.4 | 50.5 | no gain over frames |
| event_order | 97 | 76.3 | 75.3 | 73.2 | no gain |
| landmark_relation | 258 | 67.1 | 70.5 | 66.3 | no gain |
| pair_relation | 331 | 63.1 | 63.1 | 58.3 | measurement hurts |
| metric_distance | 73 | 31.5 | 37.0 | 30.1 | everyone near chance |
| metric_speed | 92 | 35.9 | 38.0 | **18.5** | measurement hurts badly |
| instance_count | 32 | 65.6 | **81.2** | 43.8 | measurement hurts badly |

**Draft toolset for discussion** (my proposal, built from your points; not decided):

| Working name | Replaces | Reports, over the whole clip by default |
|---|---|---|
| `scene_inventory` | census, view_state, visibility, count | Per named entity: instances, visibility timeline, share of frames visible |
| `camera_track` | camera_motion, camera_profile | Translation and pan/tilt/roll per segment in the camera's start frame; tempo and jitter |
| `subject_track` (`relative_to=`) | entity_track, motion_profile, kinematics, speed, relative_position, distance, region_relation | Path in the requested frame or relative to a reference; distance trend; speed section |
| `group_track` | group_geometry, evaluate_predicate | Per-instance direction, spread over time, counts by direction |
| `rotation` | orientation, projected_rotation | Heading over time, accumulated turn, facing at an instant |
| `viewpoint` | camera_viewpoint, orbit, elevation | Which side of the subject the camera is on, over time |
| `zoom` | camera_zoom | Focal ratio and forward motion |
| `view_overlap` | scene_revisit | How much of the start view is seen again at the end |
| `look` | show, inspect | Chosen moments with masks, by time in seconds |
| *(VLM only)* | event tools, elevation, metric speed until fixed | — |

## Theme 3 — Planning before measuring

**Flow agreed 2026-10-17** → [12_planning.md](12_planning.md): caption from a separate captioning
subagent; caption + question + options to the main agent; about → quantities → entities → plan →
tools; inventory and frames only on request (#28–31). P5, P8, P12 and T11 are covered there; P6 is
step 2 of the flow.
Later the same day: step order about → entities → quantities and plan (#37); single-choice
menu (#43); ask-video tool (#44); the agent orchestrates its calls (#45); budget of 4 plan items and
14 calls (#46); direct calls when revising (#47); colour-coded candidate choice (#48). The
quantity vocabulary comes from data ([06](06_question_census.md)). Open: how the plan is written
(proposal: plain-text numbered plan with host reminders).

| ID | Your point | My take | Evidence | Status |
|---|---|---|---|---|
| P5 | The agent is a spatial reasoning agent using tools, not a video analyser | Agree. V4's system prompt already says this ("You cannot measure 3D motion by looking … decide what should be measured … read the result honestly"). Start V6's prompt from V4's. | `agent_v4/prompts/agent_v4_system.md` | **agreed** 2026-10-17: your opening sentence, V4/V5 mix (#37) |
| P6 | First ask which physical quantities are needed (distance observer–object, object–object, relation, speed), then pick tools | Agree. This is V4's step 1 ("Name the one physical quantity … and the frame of reference"). V5 replaced it with "whose state" (family), which mixes concepts. Constraint: still one small decision per call. | `agent_v4/prompts/bucket_choice.md`; 03 §2 | **agreed** 2026-10-17: plan items name quantities (#50) |
| P8 | A short video description (e.g. egocentric: observer = subject) | Cautious agree. It is useful for planning (113 VLM4D Ego4D questions where "person" means the camera wearer). Risk: a caption is a prior, and priors are what reversed video punishes. I'd use it for planning and grounding only, not show it at the answer, and test it. | Caption-only answering scored 47.1% on Dyn (jev experiment) | **agreed** 2026-10-17: about 50 words, for planning and grounding only (finding entities the question does not name), never shown at the answer |
| P12 | Allow tool combinations (fish example); plan high-level steps, then adapt to outputs | Agree with a guard. Open-ended multi-step plans from the 8B model failed three times (v2, v3, compositional). Proposal: the plan is a short ordered list (≤4 steps) drawn from the tool names. The host runs common chains as one recipe (fish: inventory → track every fish → directions in the camera frame → count). The model adapts only at checkpoints after each report: continue, revise, or answer. | 03 §1 | **agreed** 2026-10-17: tagged plain-text plan, ≤4 items, 14 calls, reminders (#46, #49–51) |
| T11 | Census before routing | See theme 2: the host runs `scene_inventory` on the nouns in the question and options before planning. | | **changed** 2026-10-17: the inventory runs only when the agent calls `detect_all_entities` (#30) |

## Theme 4 — Calling tools reliably

**Settled 2026-10-17** → [13_tool_calls.md](13_tool_calls.md): design the causes away, then normaliser, then doctor; perception fallbacks for grounding and facing.

| ID | Your point | My take | Evidence | Status |
|---|---|---|---|---|
| P1 | Many tool failures; an agentic doctor that repairs malformed calls using tool docs and history | Agree there is a real problem, but it has three layers (03 §6). (a) Broken JSON is rare in legacy V5 except at the answer turn. (b) Host-refused arguments are frequent and mostly caused by the interface: entity IDs, generator-frame windows, inspection handles, `cam` not registered. (c) Unavailable measurements are perception failures. Proposal: first remove the causes of (b) by design; then a deterministic normaliser (aliases, fuzzy names, defaults); then your doctor, on local Qwen, for whatever is left. (c) needs a fallback policy, which the doctor cannot provide. | Compositional DSI: 100/252 questions had a refused call (32% correct on them) | **agreed** 2026-10-17 → [13](13_tool_calls.md) (#59–63). **Your position (2026-10-17):** the doctor should handle broken output *and* refused calls, and some unavailable measurements, e.g. re-query SAM3 with another keyword, or ask a VLM for the rough orientation of a cropped subject. Detail in theme 4. |
| P4 | What are generator frame indices (`inspect`)? | They are positions in the clip's full frame sequence used by Dyn-Bench's question generator (0 … L−1, e.g. 0–240). The model only sees 8–16 frames labelled "Frame 37 (t=1.5 s)", yet `inspect` and windows expect those raw integers. Agree they should never reach the model: it should refer to time in seconds or "shown frame k", and the host maps. | 178 refusals; `agent_v5/capabilities.py:266`, `clips.py:71` | **agreed** 2026-10-17: every report and frame tool uses seconds (or the labels of frames shown); generator frames never reach the model (10, #33, #41) |
| P10 | After a grounding failure: one forced retry with a frame, a different name, singular nouns | Agree. V5 already retries with shorter phrases ("brown horse with rider" → "brown horse" → "horse") and singularises for its word check, but never shows the model a frame. Also drop the host word check that rejects good names: 238 rejections on Dyn. | SAM3 found nothing: Dyn 88, DSI 80, VLM4D 183 | **agreed** 2026-10-17: one retry with a new singular name, then `get_frames`/`ask_video` (#53) |

## Theme 5 — From measurement to answer

| ID | Your point | My take | Evidence | Status |
|---|---|---|---|---|
| P2 | Measurement reasoning matches GT, then the final step flips to a wrong answer | Agree; this is the most important point. Possible causes, untested: (a) the answer turn re-injects frames and a long dossier, the model reasons visually again and the prior wins (reversed video); (b) options arrive late, so the measurement reasoning was not written against them; (c) a fresh context drops that reasoning. Proposal: the model states its reading of the measurement in one line before seeing options. The answer turn must say which option matches that reading, or why it overrides it. We first measure how often flips happen on saved traces, then test on frozen evidence. | 5 pilot regressions where the review was right and the answer flipped; compositional DSI 67% when measurements worked (03 §6) | **agreed** 2026-10-17 → [14](14_answering.md): same conversation, no added frames, precedence rule, V4 answer format, flip audit (#64–68) |

## Suggested order

1. **Theme 1 (report format).** Everything else consumes it.
2. **Theme 2 (toolset)**, using the draft table as the starting point.
3. **Theme 3 (planning)**, which depends on the tool names.
4. **Theme 4 (robust calls)** and **Theme 5 (answering)**, then evaluation.

All themes settled on 2026-10-17; evaluation is in [15_evaluation.md](15_evaluation.md).
