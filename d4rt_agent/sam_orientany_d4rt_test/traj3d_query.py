"""Stage 2: ask D4RT where the subject's points are, in 3D, through the clip.

Two query batches, with different shapes and different purposes:

    subject   20 points, t_src=<seed frame from stage 1>, t_tgt=0..31, t_cam=0
              The trajectory.  Holding t_cam fixed means every frame's position
              is expressed in the same camera frame, so the coordinates can be
              differenced -- which is exactly what a trajectory is.

    camera    144 grid points, t_src=t_tgt=0, t_cam=t for each t
              The camera's own motion.  D4RT emits no pose, so it has to be
              recovered by looking at the same instant from every viewpoint.

Both go through _query_explicit_points, which is the lowest-level entry point
and the only one that accepts caller-supplied UV.  It also returns *raw logits*
for visibility and confidence; that conversion is left to traj3d_analyze so the
numbers on disk are exactly what the model produced.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import (
    D4RT_CHECKPOINT,
    D4RT_CONFIG,
    Target,
    TARGETS,
    default_run_dir,
    read_json,
    video_paths,
    write_json,
)

NUM_FRAMES = 32


def _points_payload(uv_norm: list[list[float]]) -> list[dict[str, Any]]:
    return [{"d4rt_uv_norm": [float(u), float(v)]} for u, v in uv_norm]


def query_subject(
    backend: Any, uv_norm: list[list[float]], t_src: int = 0
) -> dict[str, np.ndarray]:
    """Positions of the subject points at every frame, all in the frame-0 camera.

    ``t_src`` is the frame the points were seeded on, which is not necessarily 0
    -- stage 1 picks whichever frame resolves the subject best. It is ``t_cam=0``
    that puts every returned position in the frame-0 camera, so the trajectory is
    in the requested coordinate frame regardless of where the points came from.
    """

    _, out = backend._query_explicit_points(
        points=_points_payload(uv_norm),
        t_src=int(t_src),
        t_tgt=list(range(NUM_FRAMES)),
        t_cam=0,
    )
    return {key: np.asarray(value) for key, value in out.items()}


def query_camera(backend: Any, uv_norm: list[list[float]]) -> dict[str, np.ndarray]:
    """The same frame-0 geometry seen from each of the 32 camera viewpoints.

    ``t_src = t_tgt = 0`` throughout, so the model never has to solve a temporal
    correspondence; only the viewpoint changes.  That isolates the camera
    transform instead of folding tracking error into it.
    """

    points = _points_payload(uv_norm)
    stacks: dict[str, list[np.ndarray]] = {}
    for t_cam in range(NUM_FRAMES):
        _, out = backend._query_explicit_points(
            points=points, t_src=0, t_tgt=[0], t_cam=t_cam
        )
        for key, value in out.items():
            # (P, 1, ...) -> (P, ...); the single target axis carries no information.
            stacks.setdefault(key, []).append(np.asarray(value)[:, 0])
    # (T, P, ...) so indexing by viewpoint reads naturally downstream.
    return {key: np.stack(value, axis=0) for key, value in stacks.items()}


def query_target(target: Target, run_dir: Path, backend_holder: dict[str, Any]) -> dict[str, Any]:
    from d4rt_agent.simple_v2_backend import LiveD4RTBackend
    from d4rt_agent.simple_v2_contracts import sample_video_cpu

    paths = video_paths(run_dir, target)
    if not paths.points_json.exists():
        raise FileNotFoundError(f"stage 1 output missing: {paths.points_json}")
    points = read_json(paths.points_json)

    sampled = sample_video_cpu(target.video_path)

    backend = backend_holder.get("backend")
    if backend is None:
        # point_mode and benchmark_scale are required by the constructor but
        # only affect the higher-level bbox helpers, which this stage bypasses.
        backend = LiveD4RTBackend(
            sampled_video=sampled,
            point_mode="centroid",
            benchmark_scale=1.0,
            model_config=D4RT_CONFIG,
            checkpoint=D4RT_CHECKPOINT,
        )
        backend_holder["backend"] = backend
    else:
        # Re-encodes the clip in seconds instead of reloading the 14 GB weights.
        backend.rebind_video(sampled)

    started = time.time()
    seed_frame = int(points.get("seed_frame", 0))
    subject = query_subject(backend, points["subject_points_uv_norm"], seed_frame)
    camera = query_camera(backend, points["camera_grid_uv_norm"])
    elapsed = time.time() - started

    np.savez_compressed(
        paths.d4rt / "subject.npz",
        **{key: value for key, value in subject.items()},
    )
    np.savez_compressed(
        paths.d4rt / "camera.npz",
        **{key: value for key, value in camera.items()},
    )

    record = {
        "slug": target.slug,
        "subject_keys": {k: list(v.shape) for k, v in subject.items()},
        "camera_keys": {k: list(v.shape) for k, v in camera.items()},
        "subject_query": {
            "t_src": seed_frame,
            "t_tgt": list(range(NUM_FRAMES)),
            "t_cam": 0,
            "num_points": len(points["subject_points_uv_norm"]),
        },
        "camera_query": {
            "t_src": 0,
            "t_tgt": 0,
            "t_cam": list(range(NUM_FRAMES)),
            "num_points": len(points["camera_grid_uv_norm"]),
        },
        "note": "visibility and confidence are RAW LOGITS; apply sigmoid before use",
        "checkpoint": str(D4RT_CHECKPOINT),
        "wall_seconds": elapsed,
    }
    write_json(paths.d4rt / "queries.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    holder: dict[str, Any] = {}
    for target in TARGETS:
        if args.only and target.slug not in args.only:
            continue
        record = query_target(target, run_dir, holder)
        print(
            f"{target.slug}: subject {record['subject_keys']['xyz_3d']} "
            f"camera {record['camera_keys']['xyz_3d']} "
            f"in {record['wall_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
