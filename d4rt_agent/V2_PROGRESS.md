# Phase 7: Live Qwen–D4RT agent progress

## Fixed experiment

- Entry point: `python -m d4rt_agent.simple_v2 --point-mode {centroid,ensemble5}`
- Video: `demo/pstudio_mini/basketball_6/assets/input_video.mp4`
- D4RT: local 32-frame config and `opend4rt.ckpt`
- Qwen: local-only Qwen3-VL-8B-Instruct, greedy decoding, seed 42
- Hardware per run: one 80 GB A100
- Temporal contract: `round(linspace(0, N - 1, 32))`, decoded on CPU
- Camera time: `Tcam=0`

The host fixes `point_mode` before model execution. It is absent from the Qwen
schemas and stripped from tool results returned to Qwen. The full artifact still
records it at the run, question, and live-query levels.

## Implementation status

| Milestone | Status | Evidence |
|---|---|---|
| 7.1 sampling and contracts | implemented | `simple_v2_contracts.py`, CPU tests |
| 7.2 live D4RT inference | implemented | cached `LiveD4RTBackend`, `--smoke` mode |
| 7.3 Qwen orchestration | implemented | three-action evidence loop and CPU trace replay |
| 7.4 two-job evaluation | implemented | reusable `run_simple_v2_a100.slurm` |
| 7.5 aggregate results | pending GPU artifacts | `--aggregate` validates and compares both runs |

## Safety and provenance

- Live predictions come only from `model.encode_video` plus `model.decode_queries`.
- V2 does not import v1 `DemoGeometry` and does not read predicted tracks from
  `demo_data.json`.
- Only the bounded metadata prefix is read to recover the existing GT-derived
  global scale. GT trajectories come directly from the WorldTrack NPZ and are
  recomputed at the 32 rounded source-frame indices.
- Restricted calculations use a small AST interpreter. Imports, loops, attributes,
  comprehensions, files, network, subprocesses, and arbitrary builtins are rejected.
- Qwen receives the same 32-frame sample as 32 separately labelled full-resolution
  images. Every live query declares the sampled source frame used for its box; there
  is no separate frame-inspection action.
- The generic system prompt gives endpoint and travelled-path planning/tool examples
  using unrelated objects. It explicitly prioritizes travel wording and requires the
  full ordered 32-frame target list for path length.
- The ensemble uses four seed-42 offsets clamped to the Qwen box and image, each no
  more than 12 pixels from the centroid. A target requires three valid visible
  point predictions.

## Verification commands

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python -m unittest \
  d4rt_agent.test_simple_v2 -v

# One live decoder query, no Qwen:
/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python -m d4rt_agent.simple_v2 \
  --point-mode centroid --smoke
```

## Exact two-job protocol

```bash
sbatch --export=ALL,POINT_MODE=centroid scripts/run_simple_v2_a100.slurm
sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_simple_v2_a100.slurm
```

After both artifacts exist:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python -m d4rt_agent.simple_v2 --aggregate
```

The aggregate report retains the evaluation caveat: an ensemble average
approximates an object-center trajectory, whereas the available WorldTrack GT is a
sparse surface-point trajectory.
