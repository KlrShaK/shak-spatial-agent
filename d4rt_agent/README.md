# D4RT Agent

This package evaluates whether Qwen3-VL can answer video-based spatial
questions more reliably by using live D4RT 4D geometry as a tool.

```text
video + question
      -> exact 32-frame CPU sample
      -> Qwen tool loop
      -> cached live D4RT decoder
      -> restricted numerical calculation
      -> evidence-backed answer and trace
```

Qwen handles semantic reasoning and visual grounding. D4RT supplies 3D
positions across time. The host performs calculations in a restricted numeric
interpreter and validates the evidence cited by the final answer. Ground truth
is used only for evaluation.

## Agent versions — developer notes

Four generations. Constant throughout: 3D positions from D4RT, arithmetic in the
host, GT never shown to the model. The axis that moved is **who plans the
measurement** — from the model (v2) to the host (v4).

| | agent | plans | measures | DSI-Bench 25q |
| --- | --- | --- | --- | --- |
| v1 | `simple_v1.py` *(deleted)* | host, one shot | `demo_data.json`, precomputed | — |
| v2 | `simple_v2*.py` | model, free-form | D4RT, live | 10/25 |
| v3 | `qwen_grounding_tool.py` | model, free-form | D4RT, live | 9/25 |
| v4 | `agent_v4/` | host | SAM3 + OAV2 + D4RT | 16/25 |

Tool-free Qwen control on the same frames: 13/25. v2 and v3 both lost to it.

### v1 — one-shot plan, precomputed geometry

Removed in `refactor(d4rt-agent): remove first-generation pipeline`. No loop:
Qwen saw frame 0 and emitted `{"tool", "point_2d", "frame"}` (coords 0–1000), the
host snapped the point to a track in the demo bundle and ran
`endpoint_displacement` or `path_length` (visible frames only, no interpolation
across occlusion). D4RT never ran live. Basketball: 0.324 m vs 0.563 m GT,
grounding 19.4 px off.

Established the split kept by everything since — model grounds and chooses, host
measures and computes, GT scores only.

### v2 — live D4RT, free-form evidence loop

32 frames, `round(linspace(0, N - 1, 32))`, all handed over as labelled
full-resolution images. Three actions: `query_d4rt` (box → 3D positions at named
frames), `python_math` (restricted AST over recorded evidence only — no imports,
loops, attributes, comprehensions, files, network, builtins), `final_answer`
(must cite evidence IDs; value must match a cited calculation).

Point policy is host config, absent from every schema the model sees:
`centroid`, or `ensemble5` (four seed-42 offsets, ≥3 finite visible required).
Rejected actions return for correction; traces replay on CPU.

Basketball endpoint error 0.085 m, down from v1's 0.238 m — live D4RT works.
DSI-Bench 10/25 vs 13/25 control. Failure mode was planning, not measurement:
frame-by-frame walks, re-querying never-visible targets, budget exhausted before
answering.

### v3 — grounding isolated

Hypothesis: grounding and reasoning interfere inside one conversation. Grounding
became a single-frame call with no orchestration context, producing immutable
`qg_N` groundings that `query_d4rt` cites by ID instead of re-specifying a box.
It may return `null` rather than invent a target. Also: section-addressable
system prompt, vector primitives in the calculator, self-contained rejections.

9/25. 19/60 groundings found nothing; 65 downstream calls rejected against 66
accepted; 9 steps/question mean, 28 max. Forced-grounding variant was worse
(7/25, 4 unanswered), archived on its own branch.

Grounding quality improved, score did not. Two generations of tool work had now
failed to beat no tools — the loop was the problem.

### v4 — host-driven buckets over SAM3 + OAV2 + D4RT

Host owns the plan; the model never issues a D4RT query. Enforced sequence:
`choose_bucket` → `name_subject` (if the bucket needs one) → host runs only that
bucket's stages → returns one bucket report + its reading contract →
`final_answer`. The sequence is enforced because a model asked to choose a bucket
will otherwise just answer the question.

Buckets (`agent_v4/buckets.py`), each with `stages`, a contract file and DSI
`cates` for scoring: `traj_subject_own_frame`, `traj_camera_own_frame`,
`sub_obs_distance_change`, `orient_camera_subject_frame`.

Pipeline — `sam_orientany_d4rt_test/`, driven per-question by
`agent_v4/pipeline.py`. All four checkpoints stay resident and videos stream past
them (`release_image_processor` is deliberately unused; reloading SAM3's 3.45 GB
per question costs more than the measurements). Cache key is `(video, subject)`,
not video — two questions can name different subjects in one clip.

- **SAM3** `traj3d_segment.py` — image model grounds the phrase on frame 0 for
  candidates; video model propagates a masklet through occlusion. Identity pinned
  by IoU against the frame-0 mask, not by re-prompting: `"the runner"` matches
  every runner.
- **D4RT** `traj3d_query.py` — two batches. Subject: 20 points, `t_tgt=0..31`,
  `t_cam=0` pinned so coordinates share one basis and can be differenced. Camera:
  12×12 grid, `t_src=t_tgt=0`, one query per `t_cam` — D4RT emits no pose, so it
  is recovered by viewing one instant from every viewpoint. Raw logits are
  written to disk; conversion happens in `analyze`.
- **OAV2** `traj3d_orient.py` — per-frame facing. SAM3's mask is the matte: OAV2
  composites RGBA onto white, so `rembg` is unused. Angles stay in OAV2's frame
  here.
- **Analyze** `traj3d_analyze.py` / `traj3d_geometry.py` — CPU, pure NumPy.
  Camera motion from the grid, 20 subject tracks pruned to one trajectory, OAV2
  angles → OpenCV camera frame (`OAV2_TO_OPENCV`), corrected for crop position
  and camera rotation. World-up from the scene's dominant plane; horizontal axes
  stay the camera's frame-0 heading. These fail *silently* — a wrong basis or a
  logit read as a probability produces plausible output — hence GPU-free tests.

200q uniform draw, four augmentations (leaderboard protocol):

| | std | reverse | hflip | rev+hflip | mean | robust ≥3/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 | 66.5% | 54.5% | 68.0% | 52.5% | **60.4%** | **51.0%** |
| control | 58.5% | 25.0% | 56.5% | 26.0% | 41.5% | 19.0% |

Control drops to chance on reversed video (below chance on c0/c1 — a prior
applied confidently); v4 loses 6.1 points against its 17.0. Scoring verified
against DSI-Bench's unmodified `evaluate.py`. Full census done: 1769 × 4 = 7,076.

Sections below cover v2/v3. v4: [results/agent_v4/AGENT_V4_RESULTS.md](results/agent_v4/AGENT_V4_RESULTS.md),
[results/EVALUATION_METHODOLOGY.md](results/EVALUATION_METHODOLOGY.md).

## Live agent

The main entry point is:

```bash
python -m d4rt_agent.simple_v2 --point-mode centroid
python -m d4rt_agent.simple_v2 --point-mode ensemble5
```

Every video is sampled with `round(linspace(0, N - 1, 32))`. Qwen receives all
32 sampled frames as separately labelled, full-resolution images. Videos
shorter than 32 frames are rejected, and result artifacts preserve the mapping
between sampled and original frame indices.

Qwen can emit exactly three actions:

- `query_d4rt`: ground an object with a normalized bounding box and request its
  3D position at selected sampled frames.
- `python_math`: calculate from previously recorded evidence using the
  restricted numeric interpreter.
- `final_answer`: provide an answer with evidence IDs and limitations.

Malformed actions are rejected and returned to Qwen for correction. Numeric
answers must cite both D4RT and calculation evidence, and the answer value must
match a cited calculation output. Saved calculations can be replayed on CPU.

The point policy is immutable host configuration and is never chosen by Qwen:

- `centroid` queries the center of Qwen's bounding box.
- `ensemble5` adds four deterministic seed-42 offsets within the box, image,
  and a 12-pixel centroid disk. At least three finite visible predictions are
  required for a target.

The live backend encodes the sampled clip once, caches D4RT video memory, and
supports repeated `(Tsrc, Ttgt, Tcam=0)` decoder calls. It never consumes
predicted trajectories from a pre-built demo bundle. For the basketball
experiment, it reads only the saved alignment scale from demo metadata and
loads matched ground truth directly from the WorldTrack NPZ.

Run the controlled basketball comparison on A100:

```bash
sbatch --export=ALL,POINT_MODE=centroid scripts/run_simple_v2_a100.slurm
sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_simple_v2_a100.slurm
```

Or on Blackwell:

```bash
sbatch --export=ALL,POINT_MODE=centroid scripts/run_simple_v2_blackwell.slurm
sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_simple_v2_blackwell.slurm
```

After both point-policy artifacts exist:

```bash
python -m d4rt_agent.simple_v2 --aggregate
```

The aggregate compares centroid and ensemble-5 against matched WorldTrack
ground truth. The comparison carries an important caveat: point ensembling
approximates an object-center trajectory, while WorldTrack supplies a sparse
surface-point trajectory.

Implementation details and the controlled protocol are recorded in
[V2_PROGRESS.md](V2_PROGRESS.md).

## DSI-Bench evaluation

The DSI-Bench harness evaluates the same live agent on a reproducible
25-question pilot and compares it with a tool-free Qwen control.

```bash
python -m d4rt_agent.dsi_bench_run --dry-run
```

The manifest builder selects five questions per source dataset using a fixed
Latin-rectangle coverage pattern across the six DSI task categories. The
manifest records source metadata, selected rows, video hashes, option parsing,
and ground-truth balance.

Run the agent or control on A100:

```bash
sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_dsi_bench_a100.slurm
sbatch --export=ALL,POINT_MODE=ensemble5,MODE=baseline scripts/run_dsi_bench_a100.slurm
```

Or on Blackwell:

```bash
sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_dsi_bench_blackwell.slurm
sbatch --export=ALL,POINT_MODE=ensemble5,MODE=baseline scripts/run_dsi_bench_blackwell.slurm
```

The runner loads Qwen and D4RT once, writes one JSON file per question, and
skips completed results when resubmitted. The baseline uses the same Qwen model
and sampled frames without geometry tools.

Review results with:

```bash
python -m d4rt_agent.dsi_bench_show --list
python -m d4rt_agent.dsi_bench_show <video-or-question-substring>
python -m d4rt_agent.dsi_bench_report
```

The report contains the sampled-frame animation, indexed contact sheet,
grounding boxes, complete tool trace, agent answer, control answer, and ground
truth for each question. Copied source videos are ignored by Git.

The recorded pilot and analysis are in:

- [DSI_BENCH_RESULTS.md](results/dsi_bench/DSI_BENCH_RESULTS.md)
- [FINDINGS.md](results/dsi_bench/FINDINGS.md)

## Code map

| File | Purpose |
|---|---|
| `simple_v2.py` | Offline Qwen loading, evidence-bound tool loop, trace replay, and CLI |
| `simple_v2_backend.py` | Cached live D4RT encoder/decoder and point aggregation |
| `simple_v2_contracts.py` | Video sampling, action validation, grounding policies, and restricted math |
| `simple_v2_eval.py` | Matched WorldTrack ground truth, scoring, and point-policy aggregation |
| `prompts/simple_v2_system.md` | Generic live-agent policy and worked tool examples |
| `dsi_bench_data.py` | DSI metadata parsing, balanced sampling, and manifest generation |
| `dsi_bench_run.py` | Restartable agent and tool-free benchmark passes |
| `prompts/dsi_bench_system.md` | DSI-specific spatial reasoning policy |
| `dsi_bench_show.py` | Individual trace inspection |
| `dsi_bench_report.py` | Visual benchmark report generation |
| `test_simple_v2.py` | CPU tests for live-agent contracts, orchestration, scoring, and aggregation |
| `test_dsi_bench.py` | CPU tests for dataset selection and benchmark behavior |

## Verification

Run the CPU suite with:

```bash
python -m unittest \
  d4rt_agent.test_simple_v2 \
  d4rt_agent.test_dsi_bench -v
```

GPU smoke-test only the live decoder with:

```bash
python -m d4rt_agent.simple_v2 --point-mode centroid --smoke
```

Qwen3-VL dependencies and weights are prepared once with
`scripts/prepare_qwen3vl_login.sh`. The SLURM launchers then run with Hugging
Face and Transformers offline modes enabled.
