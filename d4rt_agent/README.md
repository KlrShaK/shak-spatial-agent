# D4RT Agent (v1)

A VLM (Qwen3-VL) that orchestrates **D4RT 4D geometry as callable tools** to answer
spatial-reasoning questions about a video. v1 reads the pre-built demo bundles
(`demo/<case>/assets/demo_data.json`) instead of re-running D4RT, so the whole
pipeline runs on CPU except the VLM step.

## Simple basketball pipeline

The recommended entry point for the phased, easy-to-understand implementation is:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python \
  -m d4rt_agent.simple_v1 --phase deterministic
```

It defaults to `demo/pstudio_mini/basketball_6`, but `--demo-dir`, `--seed-u`,
`--seed-v`, and `--seed-frame` make the same script reusable for another compatible
demo bundle. The detailed design is in [BASKETBALL_V1_PROPOSAL.md](BASKETBALL_V1_PROPOSAL.md).

## How the pipeline works

```text
question + video frame
        |
        v
Qwen3-VL chooses a measurement and grounds the basketball
        |  {tool, point_2d, frame}
        v
Python validates the plan and snaps the point to a D4RT track
        |
        v
D4RT geometry tool computes the metric value
        |
        v
answer + trace + comparison with ground truth
```

The division of responsibility is intentional:

- **Qwen understands the question and image.** It decides whether the user wants
  endpoint displacement or travelled path length, then identifies the basketball.
- **D4RT supplies 3D geometry over time.** The VLM never receives a raw point cloud.
- **Python performs the arithmetic.** This avoids mixing VLM arithmetic errors with
  grounding or geometry errors.
- **WorldTrack ground truth is used only for scoring.** It is never shown to Qwen.

This is a robust one-step tool-use prototype, not a general ReAct agent. Qwen emits
one constrained JSON plan; Python executes it once and returns the result. There is
no iterative correction loop in v1.

### The two supported questions

| User intent | Tool selected by Qwen | Calculation |
|---|---|---|
| Distance between start and end | `endpoint_displacement` | `||xyz(last) - xyz(first)||` |
| Total distance covered | `path_length` | Sum of consecutive visible 3D steps |

The distinction matters. Endpoint displacement ignores the route taken. Path length
adds motion over time, but only across consecutive visible frames; it does not invent
a straight-line jump across an occlusion.

### Files to read, in order

| File | What to learn from it |
|---|---|
| [`simple_v1.py`](simple_v1.py) | The complete experiment: CLI, prompts, Qwen loading, five phases, scoring, and result writing |
| [`geometry_tools.py`](geometry_tools.py) | Loading `demo_data.json`, pixel-to-track grounding, visibility handling, displacement, path length, and duplicate removal |
| [`test_simple_v1.py`](test_simple_v1.py) | Six small examples that define the expected geometry and parser behavior |
| [`tools.py`](tools.py) | A broader JSON-schema tool interface prepared for a future multi-call agent |
| [`prepare_qwen3vl_login.sh`](../scripts/prepare_qwen3vl_login.sh) | One-time dependency and model preparation on an internet-connected login node |
| [`run_simple_d4rt_qwen_a100.slurm`](../scripts/run_simple_d4rt_qwen_a100.slurm) | Offline A100 execution and environment setup |
| [`BASKETBALL_V1_PROPOSAL.md`](BASKETBALL_V1_PROPOSAL.md) | The original phased design and success criteria |

`simple_v1.py` calls `DemoGeometry` directly after validating Qwen's plan. The
generic schemas in `tools.py` are committed for the next iteration, but were not used
to produce the basketball numbers. Older experimental working-tree files such as
`agent.py`, `qgen.py`, and `qwen_backend.py` are likewise not part of this measured
path.

### Data entering the geometry layer

The script reads `<demo-dir>/assets/demo_data.json`. The fields that matter are:

| JSON section | Purpose |
|---|---|
| `meta` | Frame count, FPS, width, and height |
| `tracks` | D4RT-predicted 3D positions, projected pixels, visibility, and confidence |
| `tracksGt` | Matching reference trajectories used for evaluation only |
| `points` | Dense point metadata, available for the broader tool interface |

For `basketball_6`, the video has 64 frames at 640 x 360 and 15 FPS. `xyzRef0` is
treated as meters because the demo construction aligned D4RT predictions to
WorldTrack scale. This does not imply that arbitrary monocular video has known
absolute scale.

### From Qwen output to a metric answer

For each question, `run_qwen()` follows the same short path:

1. Load frame 0 and ask Qwen for JSON such as
   `{"tool":"path_length","point_2d":[546,406],"frame":0}`.
2. Reject unsupported tools, malformed coordinates, or a non-zero grounding frame.
3. Convert Qwen's `[0,1000]` coordinates to image pixels.
4. Find visible D4RT tracks near that pixel; fall back to the nearest visible track.
5. Collapse exact duplicate trajectories.
6. Compute the selected metric in NumPy and save the complete trace as JSON.

The basketball bundle contains six local track entries, IDs `5-10`, but all six are
exact copies. Deduplication correctly reduces them to one trajectory rather than
pretending they are six independent estimates.

### Why the work is split into phases

| Phase | What changes | What it isolates |
|---|---|---|
| `deterministic` | Known basketball pixel, no VLM | D4RT geometry error |
| `qwen` | Qwen selects the tool and pixel | Planning and grounding error |
| `robust` | Nearby tracks are grouped and deduplicated | Track-selection stability |
| `direct` | Qwen must estimate meters from RGB only | Value of the D4RT tool |
| `aggregate` | Existing result files are merged | Final comparison; no inference |

Three evaluation lenses motivated this structure:

- **Oracle grounding + GT geometry** verifies the measurement logic.
- **Oracle grounding + D4RT geometry** is the best achievable answer given the
  predicted track; this isolates the geometry ceiling.
- **Qwen grounding + D4RT geometry** is the complete v1 system. Its gap from the
  oracle-grounded result is the orchestration cost.

The earlier Apartment demos were useful for static spatial questions but contained
little object motion. The basketball clip was chosen because it genuinely exercises
endpoint displacement, trajectory length, occlusion, and metric error.

## Results by phase

### Phase 1 results: deterministic geometry baseline

Status: **complete**

The known basketball point `(346, 166)` in frame 0 snaps to predicted track 5 at
1.53 pixels. This phase intentionally uses oracle grounding so its errors come from
geometry rather than Qwen.

| Measurement | D4RT prediction | WorldTrack GT | Absolute error |
|---|---:|---:|---:|
| Start-to-end displacement | 0.3243 m | 0.5625 m | 0.2382 m |
| Visible trajectory length | 3.6049 m | 3.1581 m | 0.4468 m |

The predicted track is visible in frames `0-33` and `56-63`: 42/64 frames, or
65.6% coverage. Path length does not bridge the missing interval. The machine-readable
result is saved in `results/basketball_6/phase1_deterministic.json`.

Verification:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python \
  -m unittest d4rt_agent.test_simple_v1 -v
```

### Phase 2 status: Qwen planner

Status: **complete**

The same entry point supports `--phase qwen`. Qwen3-VL receives frame 0 and must
return one constrained JSON plan containing the selected measurement tool and a
normalized basketball point. Geometry and arithmetic remain deterministic. The A100
launcher is `scripts/run_simple_d4rt_qwen_a100.slurm`.

GPU nodes do not need internet access. Prepare dependencies and weights once on an
internet-connected login node, then submit the offline job:

```bash
bash scripts/prepare_qwen3vl_login.sh
sbatch scripts/run_simple_d4rt_qwen_a100.slurm
```

The preparation script stores only the Transformers-layer Python packages under
`.cache/qwen3vl_python` and downloads Qwen into the shared Hugging Face cache. It
reuses the existing D4RT Torch/CUDA stack. At runtime, the SLURM script copies the
small Python layer to node-local storage, enables `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE`, resolves the cached snapshot to an absolute path, and uses
`local_files_only=True`. A missing dependency or weight snapshot causes an immediate
preflight failure instead of a network attempt.

Qwen3-VL-8B was validated offline on one 40 GB A100. Peak PyTorch allocation was
16.63 GiB, leaving substantial headroom in that run. Qwen and live D4RT inference
were not loaded concurrently, so combined memory use is not yet measured.

| Question | Selected tool | Grounding error | D4RT answer | Planning correct |
|---|---|---:|---:|---:|
| Start/end separation | `endpoint_displacement` | 19.43 px | 0.3243 m | yes |
| Distance covered | `path_length` | 20.14 px | 3.6049 m | yes |

Planning accuracy was 2/2 and grounding within the predefined 40-pixel tolerance
was 2/2. Without D4RT, Qwen answered `cannot be determined` for both metric
questions. The machine-readable result is `results/basketball_6/phase2_qwen.json`.

### Phase 3 results: duplicate-aware grounding

Status: **complete**

Run:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python \
  -m d4rt_agent.simple_v1 --phase robust
```

Within a conservative 12-pixel radius of the basketball seed, the bundle contains
six track entries (`5-10`). They are exact copies of one trajectory, so the robust
grounder collapses them to one unique track before measurement. Consequently the
metric results correctly remain unchanged and their cross-track spread is zero:

| Measurement | Robust median | Min/max | Absolute error vs GT |
|---|---:|---:|---:|
| Start-to-end displacement | 0.3243 m | 0.3243/0.3243 m | 0.2382 m |
| Visible trajectory length | 3.6049 m | 3.6049/3.6049 m | 0.4468 m |

This is an honest negative result for ensembling on this bundle: duplicate removal
makes object identity explicit, but cannot improve geometry when every local track is
identical. The machine-readable result is
`results/basketball_6/phase3_robust.json`.

### Phase 4 results: aggregate evaluation

Status: **complete**

The final report is [results/basketball_6/AGGREGATE_RESULTS.md](results/basketball_6/AGGREGATE_RESULTS.md).

| Component | Result |
|---|---:|
| Qwen tool selection | 100% (2/2) |
| Qwen grounding within 40 px | 100% (2/2) |
| Endpoint D4RT absolute error | 0.2382 m |
| Visible path D4RT absolute error | 0.4468 m |
| Direct Qwen metric answers | 0/2; correctly abstained twice |
| Duplicate local tracks removed | 5 |

The main limitation is D4RT geometry, not Qwen orchestration: Qwen chose the right
operation and track for both prompts, while the resulting numerical error was already
present in the deterministic D4RT ceiling. Path length covers only 42/64 visible
frames and does not bridge the occlusion gap.

### Phase 5 results: forced pure-VLM metric estimates

Status: **complete**

The earlier direct baseline was allowed to abstain. For a more revealing comparison,
Qwen was shown eight RGB frames and required to provide its best numerical estimate.
It received no D4RT tracks, depth, camera calibration, or metric geometry.

| Measurement | Pure Qwen | D4RT tool | GT | Pure-Qwen error | D4RT error |
|---|---:|---:|---:|---:|---:|
| Start-to-end displacement | 1.5000 m | 0.3243 m | 0.5625 m | 0.9375 m (166.7%) | 0.2382 m (42.3%) |
| Distance travelled | 2.0000 m | 3.6049 m | 3.1581 m | 1.1581 m (36.7%) | 0.4468 m (14.1%) |

The pure VLM recognized the qualitative motion but inferred scale poorly. D4RT reduced
absolute error by about 4x for endpoint displacement and 2.6x for path length on this
single example. The machine-readable result is
`results/basketball_6/phase5_direct_vlm.json`.

## Reproducing the complete experiment

Set the interpreter once:

```bash
D4RT_PY=/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python
```

Run CPU phases and tests:

```bash
$D4RT_PY -m d4rt_agent.simple_v1 --phase deterministic
$D4RT_PY -m d4rt_agent.simple_v1 --phase robust
$D4RT_PY -m unittest d4rt_agent.test_simple_v1 -v
```

Prepare Qwen once on a login node and submit the integrated offline run:

```bash
bash scripts/prepare_qwen3vl_login.sh
sbatch scripts/run_simple_d4rt_qwen_a100.slurm
```

Submit the forced pure-VLM ablation with the same launcher:

```bash
sbatch --export=ALL,PHASE=direct scripts/run_simple_d4rt_qwen_a100.slurm
```

Regenerate the aggregate report without a GPU:

```bash
$D4RT_PY -m d4rt_agent.simple_v1 --phase aggregate
```

Inspect every CLI option:

```bash
$D4RT_PY -m d4rt_agent.simple_v1 --help
```

## One worked trace

For the endpoint question, Qwen returned:

```json
{"tool":"endpoint_displacement","point_2d":[546,408],"frame":0}
```

The runtime converted this normalized point to approximately `(349.44, 146.88)` in
the 640 x 360 image. Its error relative to the oracle seed was 19.43 pixels. Local
grounding selected ball track 5. D4RT returned endpoint positions whose Euclidean
separation was 0.3243 m; GT was 0.5625 m.

The distance-travelled question followed the same path with:

```json
{"tool":"path_length","point_2d":[546,406],"frame":0}
```

It grounded the same trajectory and accumulated only consecutive visible steps:
3.6049 m over 42 visible frames, compared with 3.1581 m GT.

## Result artifacts

| Artifact | Contents |
|---|---|
| `phase1_deterministic.json` | Oracle grounding, predicted/GT measurements, visibility diagnostics, errors |
| `phase2_qwen.json` | Raw Qwen plans, pixel conversion, track grouping, D4RT results, planning/grounding scores, conservative direct baseline |
| `phase3_robust.json` | Candidate IDs, deduplicated IDs, group metric spread |
| `phase5_direct_vlm.json` | Forced Qwen estimates, explanations, GT, absolute and relative errors |
| `aggregate.json` | Machine-readable merger of every phase |
| `AGGREGATE_RESULTS.md` | Human-readable final tables |

Each artifact includes a UTC creation timestamp. The aggregate phase reads existing
artifacts; it does not rerun Qwen or D4RT.

## Reusing the script

For another compatible demo bundle:

```bash
$D4RT_PY -m d4rt_agent.simple_v1 \
  --phase deterministic \
  --demo-dir demo/<new-case> \
  --seed-u <pixel-x> \
  --seed-v <pixel-y> \
  --seed-frame <frame> \
  --output d4rt_agent/results/<new-case>/phase1_deterministic.json
```

Check that the object has a visible nearby track, inspect visibility gaps, confirm
the coordinate scale, and use a separate results directory. The v1 prompt and two
questions still say "basketball"; generalizing the CLI to accept an object label and
arbitrary question is future work.

## Limitations and interpretation

| Limitation | Consequence |
|---|---|
| Two questions on one video | These are case-study results, not benchmark accuracy |
| Precomputed `demo_data.json` | D4RT inference did not run live beside Qwen |
| One-step planning | No iterative correction or follow-up tool calls |
| Aligned metric scale | In-the-wild monocular video still needs scale recovery |
| 42/64 visible frames | Path length excludes the occluded interval rather than interpolating it |
| Raw frame-to-frame sum | 3D jitter can inflate path length |
| 16.63 GiB Qwen peak | D4RT and Qwen were not tested concurrently in GPU memory |

The strongest conclusion supported by this v1 is narrow: Qwen correctly recognized
which geometric operation the two questions required and grounded the basketball;
D4RT then produced substantially better metric estimates than a forced RGB-only VLM
guess on this example. The experiment does not yet establish general 4D-reasoning
performance.

## Commit milestones

The phase-completion commits make the development history easy to inspect:

| Commit | Milestone |
|---|---|
| `9ef017b` | Phase 1 deterministic geometry baseline |
| `5468fb0` | Phase 2 offline Qwen tool-use result |
| `6d42443` | Phase 3 duplicate-aware grounding |
| `9de5bc1` | Phase 4 initial aggregate evaluation |
| `b73397f` | Phase 5 forced pure-VLM baseline |
| `916cdeb` | Raw D4RT predictions added to comparison table |

Use `git show <commit>` to inspect a milestone or `git log -- d4rt_agent scripts`
to follow the full sequence of atomic implementation and infrastructure fixes.
