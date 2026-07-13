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

Qwen3-VL-8B was validated offline on one 40 GB A100. Peak allocated GPU memory
was 16.63 GiB, leaving approximately 23 GiB unallocated for Open-D4RT or runtime
headroom.

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

## Design

```
question + video ─▶ VLM (Qwen3-VL)  ──calls──▶  geometry tools ──▶ D4RT tracks
                        │  orchestrates              (all math here,        (demo_data.json)
                        ▼                             returns scalars)
                   final answer
```

All geometry math lives in `geometry_tools.py` and returns scalars — the VLM never
sees raw point clouds (it grounds a pixel to a `track_id`, then asks for a number).

| File | Role |
|------|------|
| `geometry_tools.py` | `DemoGeometry`: grounding, position, distance, displacement, speed, motion label. `pred`/`gt` backends. |
| `tools.py` | Qwen/OpenAI tool schemas + `ToolDispatcher` executing calls. |
| `qgen.py` | Auto-generates QA **with ground truth** from `tracksGt` (no manual annotation). |
| `agent.py` | Agent loop + `OracleLLM` (CPU, scripted — validates tools & sets the D4RT ceiling). |
| `qwen_backend.py` | `Qwen3VLBackend`: the real VLM orchestrator (GPU, transformers). |
| `eval.py` | Scoring: numeric tolerance, label/choice accuracy, MAE. |
| `run_demo.py` | CLI tying it together over the demo bundles. |

## Run

```bash
D4RT_PY=/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python3.10

# Harness sanity: Oracle on GT geometry -> expect ~1.0 everywhere
$D4RT_PY -m d4rt_agent.run_demo --backend oracle --geometry-source gt

# D4RT accuracy ceiling: Oracle on predicted geometry
$D4RT_PY -m d4rt_agent.run_demo --backend oracle --geometry-source pred

# Real VLM orchestrator (GPU node)
sbatch scripts/run_d4rt_agent_qwen_slurm.sh
```

Three evaluation lenses:
- **Oracle + gt** → proves the harness (should be ~1.0).
- **Oracle + pred** → upper bound achievable given D4RT's geometry (isolates geometry error).
- **Qwen + pred** → the full system; gap to the Oracle ceiling = orchestration cost.

## Known limitation (important)

The three `Apartment_release_*` demo clips are **near-static** (max GT object
displacement 0.0–2.05 m; almost everything is background). They exercise *static*
spatial reasoning (inter-object distance) well but are weak for *dynamic* 4D reasoning
(speed/trajectory). To evaluate motion, build a **dynamic** case first, e.g. the
`pstudio_mini/juggle_5.npz` clip:

```bash
DEMO_CASE="pstudio_mini/juggle_5.npz" OUTPUT_DIR="demo/juggle_5" NUM_FRAMES=64 \
  bash run_build_worldtrack_demo.sh
# then point run_demo at demo/juggle_5
```
