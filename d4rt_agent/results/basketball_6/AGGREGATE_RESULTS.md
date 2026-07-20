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

| Task | Pure Qwen | D4RT tool | GT | Pure-Qwen error | D4RT error |
|---|---:|---:|---:|---:|---:|
| endpoint_displacement | 1.5000 m | 0.3243 m | 0.5625 m | 0.9375 m (166.7%) | 0.2382 m (42.3%) |
| distance_travelled | 2.0000 m | 3.6049 m | 3.1581 m | 1.1581 m (36.7%) | 0.4468 m (14.1%) |

## Interpretation

The V1 goal is transparent orchestration, not state-of-the-art accuracy. Geometry, grounding, planning, and answer-generation failures are reported separately. Visible-only path length does not interpolate across occlusions.

---

## Simple v2 — improved-prompt ensemble run (job 7194772)

- Run: SLURM `7194772`, A100-80GB (`eu-ts-01`), env `envs/d4rt`, 15m16s, `COMPLETED`.
- Pipeline: `simple_v2`, `POINT_MODE=ensemble5` (5 grounding points per query, host-aggregated).
- Prompt: improved v2 prompt — grounding now succeeds (raw values are real ball motion, unlike the earlier collapsed corner-tracking runs).
- Ground truth (v2, occlusion-aware visible-only path): endpoint `0.5625 m`, distance `3.1001 m`.
- Benchmark scale: `1.1193` (`global_median_scale`, `demo_data.json → meta.worldtrack.trackAlignment`).

| Task | raw D4RT | aligned (×1.1193) | GT | absolute error | relative error |
|---|---:|---:|---:|---:|---:|
| endpoint_displacement | 0.3764 m | 0.4213 m | 0.5625 m | 0.1412 m | 25.1% |
| distance_travelled | 2.2194 m | 2.4843 m | 3.1001 m | 0.6158 m | 19.9% |

### Alignment scale is not the lever

Both metrics under-shoot GT, but the dataset offers no better scale. The current `1.1193` is a global median; a sequence-specific scale computed from basketball_6's own tracks (`demo_data.json`: `tracksRaw` ↔ `tracksGt`, 1196 matched visible points) is *smaller*, so it makes errors worse:

| Scale source | value | endpoint err | distance err |
|---|---:|---:|---:|
| global median (current) | 1.1193 | 25.1% | 19.9% |
| Umeyama sim3 (this sequence) | 0.9482 | 36.5% | 32.1% |
| median magnitude ratio (this sequence) | 0.9313 | 37.7% | 33.3% |

### Ensemble point outliers (background-stuck grounding points)

Of the 5 ensemble grounding points D4RT tracked, **2 stick to the background and barely move**, compressing the averaged trajectory. Per-point aligned path length:

| point | endpoint path | distance path | classification |
|---|---:|---:|---|
| 0 | 0.675 m | 4.169 m | on-ball |
| 1 | 0.685 m | 4.094 m | on-ball |
| 2 | 0.639 m | 4.178 m | on-ball |
| 3 | 0.040 m | 0.263 m | **background-stuck** |
| 4 | 0.091 m | 0.569 m | **background-stuck** |

The host policy averages all 5, so the static points drag motion down. But the on-ball points also **jitter**, which *inflates* accumulated path length — so the current all-5 mean accidentally cancels two opposing errors (compression vs jitter) and lands at a mediocre value for the wrong reason.

### Outlier rejection + temporal smoothing

- **Outlier rejection:** drop points whose full-trajectory motion `< 0.4 × median` of the cohort (removes points 3 & 4).
- **Temporal smoothing (`wN`):** an N-frame centered moving-average applied to the aggregated trajectory *before* path integration, to remove per-frame jitter. `w3` = ±1 frame (3-frame window), `w5` = ±2 frames (5-frame window). Smoothing only affects path length (`distance_travelled`); it is a no-op for `endpoint_displacement` (start-vs-end only).

**endpoint_displacement** (GT 0.5625 m):

| aggregation | value | error |
|---|---:|---:|
| all-5 mean (current policy) | 0.4213 m | 25.1% |
| inliers only [0,1,2] | 0.6657 m | **18.4%** |
| all-5 per-frame median (robust) | 0.6847 m | 21.7% |

**distance_travelled** (GT 3.1001 m):

| aggregation | value | error |
|---|---:|---:|
| all-5 mean (current policy) | 2.4843 m | 19.9% |
| inliers only [0,1,2] | 4.1400 m | 33.5% |
| all-5 per-frame median (robust) | 4.0760 m | 31.5% |
| inliers + smooth(w3) | 3.7581 m | 21.2% |
| inliers + smooth(w5) | 3.3738 m | **8.8%** |
| all-5 median + smooth(w3) | 3.7015 m | 19.4% |

**Best achievable on this run:** endpoint `25.1% → 18.4%` (drop background-stuck points), distance `19.9% → 8.8%` (drop background-stuck points + `w5` smoothing).

### Caveats

- The `0.4 × median` motion threshold and the `w5` window are tuned on this single clip. The *direction* (reject background-stuck points; smooth before path integration) is principled, but the exact parameters need validating across sequences — the smoothing window should ideally be tied to frame rate rather than hard-coded.
- Errors here are computed against the v2 occlusion-aware GT (endpoint 0.5625 m, distance 3.1001 m), which differs from the V1 sections above (distance GT 3.1581 m).
