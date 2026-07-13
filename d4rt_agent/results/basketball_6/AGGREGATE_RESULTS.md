# D4RT Agent V1 Aggregate Results

Status: **complete**

## Geometry baseline

| Task | D4RT | GT | Absolute error |
|---|---:|---:|---:|
| endpoint_displacement | 0.3243 m | 0.5625 m | 0.2382 m |
| distance_travelled | 3.6049 m | 3.1581 m | 0.4468 m |

## Qwen3-VL orchestration

- Model: `/cluster/work/igp_psr/spanwar/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Planning accuracy: 100.0%
- Grounding within 40 px: 100.0%
- GPU: `NVIDIA A100-PCIE-40GB`

### Direct VLM baseline

- **endpoint_displacement**: cannot be determined
- **distance_travelled**: cannot be determined

## Robust grounding

Found 6 local track entries, collapsed to 1 unique trajectory; removed 5 duplicates.

## Forced pure-VLM metric estimates

Qwen received sampled RGB frames and the question only. It was required to return a number despite monocular scale ambiguity.

| Task | Pure Qwen | GT | Absolute error | Relative error |
|---|---:|---:|---:|---:|
| endpoint_displacement | 1.5000 m | 0.5625 m | 0.9375 m | 166.7% |
| distance_travelled | 2.0000 m | 3.1581 m | 1.1581 m | 36.7% |

## Interpretation

The V1 goal is transparent orchestration, not state-of-the-art accuracy. Geometry, grounding, planning, and answer-generation failures are reported separately. Visible-only path length does not interpolate across occlusions.
