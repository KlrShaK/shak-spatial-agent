# 02 — How V4 and V5 run a question

Three controllers exist: V4 (DSI only), V5 legacy (`--stack legacy`, the stack behind the full
Dyn, DSI-800 and VLM4D runs) and V5 compositional (`--stack compositional`, the Oct 14–15 subset
run). The table at the end puts them side by side.

## V4 — host-driven buckets

The host owns the plan. The model makes at most three small decisions, then reads one report.

```
32 frames (round(linspace(0, N-1, 32))), full resolution, + question stem (no options)
  1. choose_bucket   one of 4 buckets, each described by quantity and frame
  2. name_subject    ≤5-word SAM3 phrase; 2 attempts; skipped for the camera bucket
  3. choose_candidate only if SAM3 is ambiguous: numbered crops
host: runs only that bucket's stages (segment → D4RT subject/camera queries → OAV2 → analyze)
  4. final_answer    same conversation, so all 32 frames are still in context
                     + the report (prose, per-segment table, totals with noise floors)
                     + the bucket's reading contract + the options, shown for the first time
```

Typical cost is 2–4 model calls. An unavailable measurement is reported as such, and the prompt
tells the model to answer from the frames rather than pick "cannot be determined".

| Piece | Where | What it does |
|---|---|---|
| Bucket registry | `agent_v4/buckets.py` | 4 buckets: `traj_subject_own_frame` (c0, c1), `traj_camera_own_frame` (c2, c3), `sub_obs_distance_change` (c4), `orient_camera_subject_frame` (c5). Each fixes stages, a reading contract and its DSI `cate` (used only for scoring). |
| Controller | `agent_v4/orchestrator.py` `AgentV4Orchestrator` | Enforced sequence `choose_bucket → name_subject → (choose_candidate) → final_answer`; `_deliver` composes the answer turn; `_render_metrics` prints segment and checkpoint tables. |
| Pipeline | `agent_v4/pipeline.py` `QuestionPipeline` | `open / ground / commit / measure`; models stay resident; stage outputs cached per (video, subject). |
| SAM3 | `sam_orientany_d4rt_test/traj3d_segment.py` | Image model grounds on frame 0; video model propagates; identity pinned by IoU with the frame-0 mask. |
| D4RT | `traj3d_query.py` | Subject: 20 points, `t_tgt = 0..31`, `t_cam = 0`. Camera: 12 × 12 grid, `t_src = t_tgt = 0`, one query per `t_cam` (pose recovered by viewing one instant from every viewpoint). |
| OAV2 | `traj3d_orient.py` | Per-frame facing with the SAM3 mask as matte; corrected for crop position and camera rotation. |
| Analysis | `traj3d_analyze.py`, `traj3d_geometry.py`, `traj3d_buckets.py` | World-up from the dominant plane; horizontal axes are the camera's frame-0 heading. Path cut into 10 equal-time segments, smoothed (running mean, 5), noise floors, then prose. Distance and orientation use initial/mid/final checkpoints. |
| Prompts | `agent_v4/prompts/` | System (610 words), bucket choice (178), four contracts (212–524). Contracts teach sign conventions, gravity versus camera axes, and the orbit rule (translate left while panning right → clockwise). |

## V5 legacy — family → kind → tool loop

The model plans the measurement within a kind-scoped menu.

```
8 keyframes + question (no options)
  D1 choose_family   7 families (semantic skips to the answer)
  D2 choose_kind     2–7 kinds within the family (23 kinds in all)
  D3 name_entity     one turn per slot, checked against the question's words
  D4 choose_window   regex first; a model turn only for event-bounded windows
MEASURE  kind playbook + 3–5 tools + ground/select/show/switch_kind (once)/done_measuring;
         each result is a time-ordered table in seconds, * above noise, a descriptive label
ANSWER   same conversation: answer frames (16 Dyn / 32 DSI) + every tool result + options → letter
```

One conversation per question under a 16,000-token budget (`context.py` trims old images, then old
tables). Defaults: 16 measuring steps; 256 / 384 / 512 tokens for decision / tool / answer turns.

## V5 compositional — requirements, composition, rated answer

Same family → kind skeleton with a new menu (6 families, 21 recipes), plus typed requirements.

```
8 keyframes + question + options (options shown from the start)
  choose_family → choose_kind → slots
  requirement fields, one decision each: set_frame (scene/world0/camera/image/body_initial/
      body_current), set_count_unit, set_predicate, declare_event × n
MEASURE  recipe tools + ground/select/inspect/observe/merge_identity/track_quality/
         add_entity/add_requirement/event_window/done_measuring; ≤32 steps
ANSWER   fresh context: question/options + current evidence summaries + ≤8 targeted visuals
         (none by default for measurement questions) → assess each option in prose (0–3)
         → final JSON; optional one repair round of ≤4 calls; 3 submission attempts
```

## V5 code map

| Piece | Where |
|---|---|
| CLI, benchmark loaders, resume | `agent_v5/run.py` (`--bench dynbench\|dsi\|vlm4d`, `--stack`, `--answer-policy`, `--answer-views`, `--backbone`) |
| Legacy controller | `agent_v5/orchestrator.py` `AgentV5`: `_decompose`, `_fill_slots`, `_program`, `_measure`, `_switch`, `_answer` |
| Compositional controller | `agent_v5/composition.py` `CompositionalAgentV5`: `_requirement_fields`, `_allowed`, `_measure`, `_compose`, `_gaps`, `_repair` |
| Rated answer, final packet | `agent_v5/answering.py` `answer`; `agent_v5/final_context.py` `build_packet` |
| Menus and slots | `agent_v5/kinds.py` (`LEGACY_FAMILIES`, `FAMILIES`, `tools_for`, `validate_slot`) |
| Typed program | `agent_v5/program.py` (`Entity`, `Window`, `Event`, `Requirement`, `FrameSpec`, `EvidenceRecord`, `QueryProgram`, `window_from_question`) |
| Legacy tools | `agent_v5/tools.py` (camera_motion/profile/zoom, scene_revisit, entity_track, distance, region_relation, elevation, visibility, apparent_size, motion_profile, speed, count, event_onset, orientation, show) |
| Compositional capabilities | `agent_v5/capabilities.py` (census, count_entities, inspect, observe, relative_position, kinematics, orbit, group_geometry, event_*), `agent_v5/spatial_tools.py` (track_quality, projected_rotation, region_geometry, encounters) |
| Numerics and scene | `agent_v5/measure.py`, `agent_v5/scene.py` (smoothing, metric scale, instance merging, windows), `agent_v5/backbones/{point4d,d4rt}.py` |
| Evidence and rendering | `agent_v5/evidence.py`, `agent_v5/render.py`, `agent_v5/context.py` |
| Prompts | `agent_v5/prompts/` (system, choose_family, choose_kind, name_entity, choose_window, measure, compose, answer, assess_answer) and 42 playbooks |
| Research tooling | `agent_v5/research/` (FINDINGS, counterexamples, protocol, subset_eval, snapshot) |

## Side by side

| | V4 | V5 legacy | V5 compositional |
|---|---|---|---|
| Scope | DSI only; 4 buckets cover its 6 categories | Dyn, DSI, VLM4D; 7 families, 23 kinds | Same; 6 families, 21 recipes |
| Model decisions before measuring | 1–3 | 3–6 | 4–10+ |
| Who picks the computation | Host (bucket fixes the stages) | Model, from a 3–5 tool menu | Model, from recipe tools plus composition actions |
| Options visible | Only at the answer | Only at the answer | From the first turn |
| Geometry | D4RT, 32 frames | Point4D (default) or D4RT, DA3 metric scale, Savitzky–Golay smoothing, boundary rule | Same, plus correspondence checks |
| "Own frame" for subject motion | OAV2 facing at first resolved frame, gravity-levelled | `entity_track(frame=own)` exists but DSI subject questions mostly route to the scene frame | `set_frame`, with a regex guard for "starting orientation" |
| What the measurement returns | One report: prose, 10-segment table, totals, noise floors | Tables in seconds, * above noise, a descriptive label, notes | Evidence records with status (supported / partial / conflicting / unresolved) |
| How to read it | A per-bucket reading contract shown with the report | A per-kind playbook shown while measuring | Playbook while measuring; generic assessment rules at the answer |
| Final answer sees | All 32 frames, one report, its contract, options | Answer frames, all tool results, options | Text packet, ≤8 targeted visuals (often none), options |
| Answer format | Text beginning with the letter | JSON letter (template example shows "A") | Prose rating of every option, then JSON |
