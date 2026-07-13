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

Status: **GPU validation pending**

The same entry point supports `--phase qwen`. Qwen3-VL receives frame 0 and must
return one constrained JSON plan containing the selected measurement tool and a
normalized basketball point. Geometry and arithmetic remain deterministic. The A100
launcher is `scripts/run_simple_d4rt_qwen_a100.slurm`.

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
