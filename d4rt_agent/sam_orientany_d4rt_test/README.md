# SAM3 + Orient-Anything-V2 + D4RT

Turns a text-named subject in a video into a plotted 3D motion trajectory with
per-frame facing direction.

```
text phrase ──SAM3──▶ per-frame masks ──┬──▶ 20 seed points ──D4RT──▶ 3D tracks ──▶ trajectory
                                        └──▶ masked crops ──OrientAnythingV2──▶ facing
```

## Targets

| Video (under DSI-Bench `videos/std/`) | Subject | Frames | Res |
|---|---|---|---|
| `CameraBench/M0jmSsQ5ptw.3.12.mp4` | "the runner" | 42 | 480×270 |
| `CameraBench/99f9276…c7de.0.mp4` | "the person in the yellow jacket" | 489 | 480×270 |
| `SynFMC/…/dynamic/6/video_480p.mp4` | "the character in pink" | 72 | 720×480 |

## Running it

Once, on a login node (needs internet; ~8.5 GB):

```bash
./scripts/prepare_traj3d_weights.sh
```

Then submit. Executing the launcher (rather than `sbatch`-ing it) lets it resolve
the run directory and validate the weight cache before queueing:

```bash
./scripts/run_traj3d_blackwell.slurm
```

Resubmit with the **same** `RUN_DIR` to continue a job that hit the time limit —
each stage records completion per video and finished work is skipped:

```bash
RUN_DIR=d4rt_agent/results/traj3d_20260818T194241Z ./scripts/run_traj3d_blackwell.slurm
```

Plots are CPU-only and run on the login node after the job:

```bash
$PY -m d4rt_agent.sam_orientany_d4rt_test.traj3d_plot --run-dir <run_dir>
```

Useful knobs: `STAGES="segment query"`, `ONLY=<slug>`, `FORCE=1`.

## Files

| File | Stage | Needs GPU |
|---|---|---|
| `traj3d_common.py` | targets, paths, atomic JSON, run metadata | no |
| `traj3d_geometry.py` | all the math — frames, Kabsch, aggregation | no |
| `traj3d_segment.py` | 1 — SAM3 grounding, propagation, point seeding | yes |
| `traj3d_query.py` | 2 — D4RT subject + camera-recovery queries | yes |
| `traj3d_orient.py` | 3a — Orient-Anything-V2 per-frame facing | yes |
| `traj3d_analyze.py` | 3b — camera recovery, track pruning, CSVs | no |
| `traj3d_plot.py` | 4 — 3D PNG, orbit GIF, 2D overlay | no |
| `traj3d_run.py` | orchestration, ordered by model so each loads once | — |
| `test_traj3d.py` | 20 tests, incl. checks against the vendored sources | no |

Launcher and weight-staging scripts stay in `scripts/`, alongside the repo's
other `.slurm` files.

## Three things worth knowing

**`visibility` and `confidence` are raw logits.** `_query_explicit_points` is
below the layer that applies the sigmoid (`src/model/heads.py:16,19` are bare
`nn.Linear`). Thresholding them directly keeps almost everything.

**Camera-recovery points need not be static.** From the training target
definition (`src/data/kubric_full_robust_dataset.py:970-976`),
`X(t_cam) = T_cw[t_cam]·p_world(t_tgt)`; querying one `t_tgt` from two `t_cam`
makes `p_world` cancel, leaving only the camera transform. So a plain uniform
image grid is correct, and no background masking is needed. (The "static
background points" advice in `prompts/dsi_bench_system.md` is a heuristic for the
text agent, not a constraint on a programmatic pipeline.)

**Every threshold is scale-relative.** D4RT's units are arbitrary — the loss
normalises each clip by its own mean depth — so thresholds are multiples of the
clip's own `cloud_scale`, never absolute distances.

## Outputs

```
d4rt_agent/results/traj3d_<UTCSTAMP>/
  run_metadata.json  run_status.json  orientation_mirror_test.json
  <video_slug>/
    frames/  masks/  masks_overlay/  candidates.jpg  segmentation.json  points.json
    d4rt/         subject.npz camera.npz queries.json
    orientation/  crop_%02d.png orientation.json
    analysis/     trajectory.csv orientation.csv camera_poses.npz trajectory.npz summary.json
    plots/        trajectory.png trajectory_orbit.gif tracks_2d_overlay.jpg
```

`candidates.jpg` is the one to check first: SAM3 returns every instance matching
the phrase, and silently tracking the wrong person is the failure nothing
downstream can detect. `summary.json` carries a `warnings` list for conditions
that make the output untrustworthy without raising.
