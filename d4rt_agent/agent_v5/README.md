# AgentV5

A 4D spatial-reasoning agent for Dyn-Bench (and still DSI-Bench). It builds on AgentV4's
host-ordered steps and on the decision tree found in
`dyn_dataset_exploration/testing_decomposition.ipynb`, with one change of principle: **no
measurement decides an answer**. Tools return numbers, noise floors and descriptive labels;
Qwen3-VL-8B reads them and chooses.

## Decision flow

```
QUESTION + OPTIONS + CLIP
  host: frames on disk, time axis (Dyn-Bench GF + per-source fps), regex windows;
        lazily the backbone scene (BACKBONE=point4d|d4rt) + its metric scale
PHASE 1  decompose -- host-ordered, one small decision per turn (reasoning, then one JSON);
         the answer options are NOT shown
  D1  8 keyframes + question + 7 families  -> choose_family
      (semantic -> straight to phase 3)
  D2  the family's 2-7 kinds               -> choose_kind
  D3  one turn per slot of the kind        -> name_entity (validated against the question's words)
  D4  window: regex; one turn only when it is bounded by an event -> choose_window
PHASE 2  measure -- model-driven; the kind's playbook and 3-5 tools are revealed
  ground / select / <the kind's tools> / show (annotated keyframes) / switch_kind (once)
  each result: time-ordered table in seconds, * above noise, a chronological line,
  a descriptive label, reliability notes  ->  ... -> done_measuring (or the step budget)
PHASE 3  answer -- the benchmark's frames (16 Dyn / 32 DSI), every tool result in full,
         and the options for the first time -> final_answer{letter}
```

One conversation per question, kept under a token budget (`context.py`): old `show` images, then
old tool tables, then rejected exchanges are replaced by one-line summaries; the answer turn
re-renders every result from the host's record.

## Families, kinds, tools

| family | kinds | tools |
|---|---|---|
| camera | camera_path, camera_translation, orbit_direction, camera_shake, camera_rhythm, camera_zoom, scene_revisit | camera_motion, camera_profile, camera_zoom, scene_revisit (+orientation for orbit) |
| subject_motion | subject_direction, traj_subject_own_frame, subject_motion_profile, metric_speed | entity_track (frame scene / camera / image / own), orientation, motion_profile, speed |
| subject_vs_camera | cam_subject_range, cam_subject_elevation, camera_viewpoint, subject_visibility, apparent_size | distance, elevation, orientation, visibility, apparent_size |
| two_entities | pair_relation, metric_distance, landmark_relation | distance, region_relation, entity_track |
| count | instance_count, tracked_objects_count | count |
| events | event_order | event_onset, camera_motion, visibility |
| semantic | semantic | none |

Every measured kind has a playbook in `prompts/playbooks/<kind>.md`, written from the
calibration findings in `dyn_dataset_exploration/logs/DECOMPOSITION.md`.

## Rules from `v5-agent_draft.md`

- **Boundary rule.** `sampling.interior_cells`: a depth cell must be at least
  `max(4 px, 25% of the inscribed radius, one cell)` inside the mask, and not on a depth
  discontinuity of more than 10%. The margin backs off to one cell and is never removed. The
  decomposition notebook's Step 4b shows it.
- **Smoothed trajectories.** Every entity path goes through `scene.smooth_track`
  (Savitzky-Golay). The camera's pan, tilt and roll are smoothed too, before any motion is read
  off them. Raw series are used only for noise floors and the shake residual. Notebook Step 4c.
- **Chronological order.** The system prompt rule, the time-ordered tables, the merged
  "In time order:" line, and the answer prompt.

## Backbones

`--backbone point4d|d4rt` (`BACKBONE=` in slurm).

| backbone | frames | metric scale |
|---|---|---|
| Point4D | up to 400 GF frames, chunked 48/8 | DA3-Nested depth ratio |
| D4RT | at most 32 frames: the first, the answer frames, fill-ins | DA3-Nested depth ratio; old `d4rt-native` cache metadata is treated as unscaled |

Scenes are cached per backbone under `<results>/cache/clips/<slug>/<backbone>/`.

## Running (GPU, from the repo or worktree root)

```bash
BACKBONE=point4d sbatch scripts/run_agent_v5_smoke.slurm             # 25 questions over all groups; peak memory
BACKBONE=point4d ./scripts/run_agent_v5_race.sh --limit 200           # A100 + Blackwell race, resumable
BACKBONE=d4rt    ./scripts/run_agent_v5_race.sh --shard 0/4           # shards: same RUN_DIR, disjoint clips
BENCH=dsi        ./scripts/run_agent_v5_race.sh --split std --random-sample 200
python -m d4rt_agent.agent_v5.report <RUN_DIR>                        # vs the decomposition baselines
python -m d4rt_agent.agent_v5.run --bench dynbench --results-dir /tmp/x --dry-run --limit 5   # no models
```

The default Dyn-Bench set is the decomposition study's 2800-question eval split, and the report
pairs it question for question with that study's baselines. Use `--all` for the whole benchmark.
Tests: `python -m pytest d4rt_agent/agent_v5/test_agent_v5.py` (GPU-free).

## Not verified yet

Qwen, SAM3, Point4D (or D4RT), DA3 and OAV2 have never been loaded together in one process;
the smoke job checks this first. The thresholds behind the descriptive labels were fitted on
Point4D scenes (`thresholds.json`, dev clips only), so with D4RT they are less trustworthy. The
numbers are always printed next to them.
