"""Matched ground truth, scoring, and aggregation for simple v2.

Ground truth is loaded from the original WorldTrack NPZ.  This file never imports
``DemoGeometry`` and never reads D4RT predictions from ``demo_data.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from .simple_v2_contracts import NUM_SAMPLED_FRAMES


DEFAULT_WORLDTRACK_NPZ = Path("data/worldtrack_release/pstudio_mini/basketball_6.npz")
DEFAULT_DEMO_DATA = Path("demo/pstudio_mini/basketball_6/assets/demo_data.json")
DEFAULT_GT_SEED = (346.0, 166.0, 0)


def load_alignment_scale_from_metadata(demo_data_path: str | Path) -> tuple[float, dict[str, Any]]:
    """Read only the JSON metadata prefix containing the saved GT-derived scale.

    The potentially large predicted-track arrays occur after the top-level ``meta``
    object.  Reading a bounded prefix makes the no-precomputed-predictions contract
    mechanically explicit.
    """

    path = Path(demo_data_path)
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(4 * 1024 * 1024)
    tracks_marker = prefix.find('\n  "tracks":')
    if tracks_marker >= 0:
        prefix = prefix[:tracks_marker]
    pattern = re.compile(
        r'"trackAlignment"\s*:\s*\{.*?"type"\s*:\s*"global_median_scale"'
        r'.*?"scale"\s*:\s*([-+0-9.eE]+)',
        flags=re.DOTALL,
    )
    match = pattern.search(prefix)
    if match is None:
        raise ValueError(f"GT-derived global alignment scale missing from metadata: {path}")
    scale = float(match.group(1))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid alignment scale in {path}: {scale}")
    return scale, {
        "type": "gt_derived_global_median_scale",
        "scale": scale,
        "source": str(path),
        "source_field": "meta.worldtrack.trackAlignment",
        "predicted_track_arrays_read": False,
    }


class MatchedWorldTrackGT:
    """Sparse surface-point GT evaluated at the exact 32 source indices."""

    def __init__(
        self,
        npz_path: str | Path,
        original_indices: Sequence[int],
        seed: tuple[float, float, int] = DEFAULT_GT_SEED,
    ) -> None:
        self.path = Path(npz_path)
        package = np.load(self.path, allow_pickle=True)
        self.xyz = np.asarray(package["tracks_XYZ"], dtype=np.float64)
        self.visibility = np.asarray(package["visibility"], dtype=bool)
        self.intrinsics = np.asarray(package["fx_fy_cx_cy"], dtype=np.float64).reshape(-1)
        self.original_indices = np.asarray(original_indices, dtype=np.int64)
        if self.original_indices.shape != (NUM_SAMPLED_FRAMES,):
            raise ValueError("matched GT requires exactly 32 original frame indices")
        if self.original_indices.min() < 0 or self.original_indices.max() >= self.xyz.shape[0]:
            raise ValueError("sampled video indices exceed available WorldTrack GT")
        self.uv = self._project(self.xyz)
        seed_u, seed_v, seed_frame = seed
        self.canonical_track, seed_distance = self.nearest_visible_track(
            float(seed_u), float(seed_v), int(seed_frame)
        )
        self.seed = {
            "pixel_uv": [float(seed_u), float(seed_v)],
            "original_frame": int(seed_frame),
            "track_index": self.canonical_track,
            "nearest_distance_px": seed_distance,
            "nearest_uv_px": _vector(self.uv[int(seed_frame), self.canonical_track]),
        }

    def _project(self, xyz: np.ndarray) -> np.ndarray:
        fx, fy, cx, cy = self.intrinsics[:4]
        safe_z = np.where(np.abs(xyz[..., 2]) > 1e-12, xyz[..., 2], np.nan)
        return np.stack(
            [xyz[..., 0] / safe_z * fx + cx, xyz[..., 1] / safe_z * fy + cy], axis=-1
        )

    def nearest_visible_track(self, u: float, v: float, original_frame: int) -> tuple[int, float]:
        frame = int(original_frame)
        distances = np.linalg.norm(self.uv[frame] - np.asarray([u, v]), axis=-1)
        valid = self.visibility[frame] & np.isfinite(distances)
        distances = np.where(valid, distances, np.inf)
        if not np.isfinite(distances).any():
            raise ValueError(f"no visible GT points in original frame {frame}")
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def canonical_trajectory(self) -> dict[str, Any]:
        xyz = self.xyz[self.original_indices, self.canonical_track]
        visible = self.visibility[self.original_indices, self.canonical_track]
        uv = self.uv[self.original_indices, self.canonical_track]
        return {
            "source": "WorldTrack sparse surface-point trajectory",
            "npz": str(self.path),
            "track_index": self.canonical_track,
            "sampled_frame_indices": list(range(NUM_SAMPLED_FRAMES)),
            "original_frame_indices": [int(value) for value in self.original_indices.tolist()],
            "xyz_m": [_vector(value) for value in xyz],
            "uv_px": [_vector(value) for value in uv],
            "visibility": [bool(value) for value in visible.tolist()],
            "visibility_coverage": float(np.mean(visible)),
            "seed": self.seed,
        }

    def measurement(self, task_id: str) -> dict[str, Any]:
        trajectory = self.canonical_trajectory()
        xyz = np.asarray(trajectory["xyz_m"], dtype=np.float64)
        visible = np.asarray(trajectory["visibility"], dtype=bool)
        if task_id == "endpoint_displacement":
            value = float(np.linalg.norm(xyz[-1] - xyz[0]))
            endpoint_visible = [bool(visible[0]), bool(visible[-1])]
            details: dict[str, Any] = {"endpoint_visible": endpoint_visible}
        elif task_id == "distance_travelled":
            value = visible_path_length(xyz, visible)
            details = {"visible_segments": visible_segments(visible)}
        else:
            raise ValueError(f"unsupported task: {task_id}")
        return {
            "task_id": task_id,
            "value_m": value,
            "visibility_coverage": float(np.mean(visible)),
            **details,
            "trajectory": trajectory,
        }

    def grounding_diagnostics(
        self, bbox_2d_1000: Sequence[float], sampled_t_src: int, width: int, height: int
    ) -> dict[str, Any]:
        from .simple_v2_contracts import bbox_1000_to_pixels

        x0, y0, x1, y1 = bbox_1000_to_pixels(bbox_2d_1000, width, height)
        centroid = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
        original_frame = int(self.original_indices[int(sampled_t_src)])
        track, distance = self.nearest_visible_track(centroid[0], centroid[1], original_frame)
        nearest_uv = self.uv[original_frame, track]
        canonical_uv = self.uv[original_frame, self.canonical_track]
        return {
            "bbox_2d_1000": [float(value) for value in bbox_2d_1000],
            "bbox_pixel": [x0, y0, x1, y1],
            "centroid_pixel_uv": centroid,
            "sampled_t_src": int(sampled_t_src),
            "original_t_src": original_frame,
            "nearest_gt_track": track,
            "nearest_gt_uv_px": _vector(nearest_uv),
            "nearest_distance_px": distance,
            "canonical_gt_track": self.canonical_track,
            "canonical_gt_uv_px": _vector(canonical_uv),
            "canonical_centroid_error_px": float(
                np.linalg.norm(np.asarray(centroid) - canonical_uv)
            ),
            "matches_canonical_track": track == self.canonical_track,
        }


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float64).tolist()]


def visible_segments(visibility: np.ndarray) -> list[list[int]]:
    result: list[list[int]] = []
    start: int | None = None
    for index, visible in enumerate(np.append(np.asarray(visibility, dtype=bool), False)):
        if bool(visible) and start is None:
            start = index
        elif not bool(visible) and start is not None:
            result.append([start, index - 1])
            start = None
    return result


def visible_path_length(xyz: np.ndarray, visibility: np.ndarray) -> float:
    total = 0.0
    previous: np.ndarray | None = None
    for point, visible in zip(xyz, visibility, strict=True):
        if not bool(visible) or not np.isfinite(point).all():
            previous = None
            continue
        if previous is not None:
            total += float(np.linalg.norm(point - previous))
        previous = point
    return total


def measure_d4rt_result(task_id: str, query_result: dict[str, Any], aligned: bool) -> float:
    key = "benchmark_aligned_xyz_m" if aligned else "raw_xyz"
    xyz = []
    visible = []
    for prediction in query_result["predictions"]:
        value = prediction.get(key)
        xyz.append(value if value is not None else [0.0, 0.0, 0.0])
        visible.append(bool(prediction.get("visible", False) and value is not None))
    xyz_array = np.asarray(xyz, dtype=np.float64)
    visible_array = np.asarray(visible, dtype=bool)
    targets = [int(value) for value in query_result["t_tgt"]]
    by_target = {target: offset for offset, target in enumerate(targets)}
    if task_id == "endpoint_displacement":
        if 0 not in by_target or 31 not in by_target:
            raise ValueError("endpoint D4RT evidence does not contain sampled frames 0 and 31")
        if not visible_array[by_target[0]] or not visible_array[by_target[31]]:
            raise ValueError("endpoint D4RT evidence is invalid or invisible at an endpoint")
        return float(
            np.linalg.norm(xyz_array[by_target[31]] - xyz_array[by_target[0]])
        )
    if task_id == "distance_travelled":
        if set(targets) != set(range(NUM_SAMPLED_FRAMES)):
            raise ValueError("path D4RT evidence does not contain all sampled frames 0-31")
        order = np.asarray([by_target[index] for index in range(NUM_SAMPLED_FRAMES)])
        return visible_path_length(xyz_array[order], visible_array[order])
    raise ValueError(f"unsupported task: {task_id}")


def merge_d4rt_results(query_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge targets from multiple cited calls without changing their predictions."""

    if not query_results:
        raise ValueError("no D4RT evidence to merge")
    point_modes = {item["point_mode"] for item in query_results}
    if len(point_modes) != 1:
        raise ValueError("cited D4RT evidence mixes immutable point modes")
    by_target: dict[int, dict[str, Any]] = {}
    for query in query_results:
        for target, prediction in zip(query["t_tgt"], query["predictions"], strict=True):
            by_target[int(target)] = prediction
    ordered_targets = sorted(by_target)
    predictions = [by_target[target] for target in ordered_targets]
    return {
        "point_mode": query_results[0]["point_mode"],
        "t_tgt": ordered_targets,
        "predictions": predictions,
        "visibility_coverage": float(
            np.mean([bool(item.get("visible", False)) for item in predictions])
        ),
    }


def score_question(
    *,
    task_id: str,
    final_answer: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    gt: MatchedWorldTrackGT,
    width: int,
    height: int,
) -> dict[str, Any]:
    d4rt_ids = [item for item in final_answer["evidence_ids"] if item.startswith("d4rt_")]
    math_ids = [item for item in final_answer["evidence_ids"] if item.startswith("math_")]
    if not d4rt_ids or not math_ids:
        raise ValueError("final answer must cite both D4RT and calculation call IDs")
    unknown = [query_id for query_id in d4rt_ids if query_id not in evidence]
    if unknown:
        raise ValueError(f"final answer cites unknown D4RT evidence: {unknown}")
    queries = [evidence[query_id] for query_id in d4rt_ids]
    merged_query = merge_d4rt_results(queries)
    raw_value = measure_d4rt_result(task_id, merged_query, aligned=False)
    aligned_value = measure_d4rt_result(task_id, merged_query, aligned=True)
    gt_record = gt.measurement(task_id)
    gt_value = float(gt_record["value_m"])
    absolute_error = abs(aligned_value - gt_value)
    return {
        "task_id": task_id,
        "d4rt_evidence_ids": d4rt_ids,
        "math_evidence_ids": math_ids,
        "raw_d4rt_value": raw_value,
        "benchmark_aligned_value_m": aligned_value,
        "agent_final_value": float(final_answer["value"]),
        "agent_final_unit": final_answer["unit"],
        "gt_value_m": gt_value,
        "absolute_error_m": absolute_error,
        "relative_error": absolute_error / gt_value if gt_value > 0 else None,
        "visibility_coverage": float(merged_query["visibility_coverage"]),
        "grounding": [
            {
                "d4rt_evidence_id": query_id,
                **gt.grounding_diagnostics(
                    query["bbox_2d_1000"], query["t_src"], width, height
                ),
            }
            for query_id, query in zip(d4rt_ids, queries, strict=True)
        ],
        "gt": gt_record,
        "evaluation_caveat": (
            "The D4RT point policy approximates the object-center trajectory, while "
            "WorldTrack GT is a sparse surface-point trajectory."
        ),
    }


def aggregate_files(
    centroid_path: str | Path,
    ensemble_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    centroid = json.loads(Path(centroid_path).read_text())
    ensemble = json.loads(Path(ensemble_path).read_text())
    if centroid.get("point_mode") != "centroid":
        raise ValueError("centroid artifact has the wrong immutable point mode")
    if ensemble.get("point_mode") != "ensemble5":
        raise ValueError("ensemble artifact has the wrong immutable point mode")
    if centroid["sampling"]["sampled_to_original"] != ensemble["sampling"]["sampled_to_original"]:
        raise ValueError("centroid and ensemble artifacts used different frame samples")
    if centroid["configuration"]["qwen_decoding"] != ensemble["configuration"]["qwen_decoding"]:
        raise ValueError("centroid and ensemble artifacts used different Qwen decoding")
    if centroid["configuration"]["seed"] != ensemble["configuration"]["seed"]:
        raise ValueError("centroid and ensemble artifacts used different seeds")
    if centroid["qwen"]["model"] != ensemble["qwen"]["model"]:
        raise ValueError("centroid and ensemble artifacts used different Qwen models")
    for field in ("model_config", "checkpoint", "device", "dtype"):
        if centroid["d4rt"][field] != ensemble["d4rt"][field]:
            raise ValueError(f"centroid and ensemble artifacts differ in D4RT {field}")
    by_mode = {
        "centroid": {item["task_id"]: item for item in centroid["scores"]},
        "ensemble5": {item["task_id"]: item for item in ensemble["scores"]},
    }
    rows = []
    for task_id in ("endpoint_displacement", "distance_travelled"):
        c = by_mode["centroid"][task_id]
        e = by_mode["ensemble5"][task_id]
        rows.append({
            "task": task_id,
            "centroid_d4rt_m": c["benchmark_aligned_value_m"],
            "ensemble5_d4rt_m": e["benchmark_aligned_value_m"],
            "gt_m": c["gt_value_m"],
            "centroid_error_m": c["absolute_error_m"],
            "ensemble5_error_m": e["absolute_error_m"],
        })
    baseline_paths = {
        "v1": Path("d4rt_agent/results/basketball_6/aggregate.json"),
        "pure_qwen": Path("d4rt_agent/results/basketball_6/phase5_direct_vlm.json"),
        "spatialstack": Path("d4rt_agent/results/basketball_6/phase6_spatialstack.json"),
    }
    baseline_records: dict[str, Any] = {}
    if baseline_paths["v1"].exists():
        artifact = json.loads(baseline_paths["v1"].read_text())
        baseline_records["v1"] = {
            item["id"]: {
                "estimate_m": item["d4rt_m"],
                "gt_m": item["gt_m"],
                "absolute_error_m": item["absolute_error_m"],
            }
            for item in artifact.get("geometry_baseline", [])
        }
    for name in ("pure_qwen", "spatialstack"):
        path = baseline_paths[name]
        if path.exists():
            artifact = json.loads(path.read_text())
            baseline_records[name] = {
                item["id"]: {
                    "estimate_m": item.get("estimate_m"),
                    "gt_m": item.get("gt_m"),
                    "absolute_error_m": item.get("absolute_error_m"),
                }
                for item in artifact.get("questions", [])
            }

    def _failures(artifact: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "task_id": question["id"],
                "step": entry["step"],
                "error": entry.get("error"),
                "raw_qwen_response": entry.get("raw_qwen_response"),
            }
            for question in artifact["questions"]
            for entry in question["trace"]
            if entry.get("status") == "rejected"
        ]

    result = {
        "phase": "simple_v2_aggregate",
        "status": "complete",
        "inputs": {"centroid": str(centroid_path), "ensemble5": str(ensemble_path)},
        "sampling": centroid["sampling"],
        "comparison": rows,
        "baselines": {
            name: {"artifact": str(path), "results": baseline_records.get(name)}
            for name, path in baseline_paths.items()
        },
        "mode_diagnostics": {
            "centroid": {
                "gpu": centroid.get("gpu"),
                "visibility": {
                    item["task_id"]: item["visibility_coverage"] for item in centroid["scores"]
                },
                "grounding": {
                    item["task_id"]: item["grounding"] for item in centroid["scores"]
                },
                "failures": _failures(centroid),
            },
            "ensemble5": {
                "gpu": ensemble.get("gpu"),
                "visibility": {
                    item["task_id"]: item["visibility_coverage"] for item in ensemble["scores"]
                },
                "grounding": {
                    item["task_id"]: item["grounding"] for item in ensemble["scores"]
                },
                "failures": _failures(ensemble),
            },
        },
        "caveat": (
            "Ensemble averaging approximates an object-center trajectory, while the "
            "available GT is a sparse surface-point trajectory."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Simple v2 centroid vs ensemble-5",
        "",
        "| Task | Centroid D4RT | Ensemble-5 D4RT | GT | Centroid error | Ensemble error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['centroid_d4rt_m']:.4f} m | "
            f"{row['ensemble5_d4rt_m']:.4f} m | {row['gt_m']:.4f} m | "
            f"{row['centroid_error_m']:.4f} m | {row['ensemble5_error_m']:.4f} m |"
        )
    if baseline_records:
        lines.extend([
            "",
            "## Historical baselines",
            "",
            "These use their original Phase 1/5/6 protocols and GT values; v2's primary "
            "comparison above uses newly matched rounded-frame GT.",
            "",
            "| Task | V1 D4RT | Pure Qwen | SpatialStack |",
            "|---|---:|---:|---:|",
        ])
        for task_id in ("endpoint_displacement", "distance_travelled"):
            values = []
            for name in ("v1", "pure_qwen", "spatialstack"):
                value = baseline_records.get(name, {}).get(task_id, {}).get("estimate_m")
                values.append("-" if value is None else f"{float(value):.4f} m")
            lines.append(f"| {task_id} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(["", result["caveat"], ""])
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    return result
