"""Closed-loop check: does the reconstruction reproject onto the subject?

Every other diagnostic in this pipeline measures one stage against its own
assumptions -- track retention says the aggregation was self-consistent, camera
RMSE says the grid points agreed with a rigid motion.  None of them would catch
the whole chain being coherently wrong, and the trajectories are in arbitrary
units in a camera frame nobody can eyeball, so "does it look right" is not
available either.

This closes the loop against the pixels:

    X_0     aggregated 3D position at time t, in the frame-0 camera
    X_t     = R_t^T (X_0 - p_t)          using the recovered camera pose
    (u, v)  = project(X_t, K)            using intrinsics fitted from D4RT
    compare against the SAM3 mask centroid at frame t

Agreement requires D4RT's depths, the RANSAC camera recovery, and the track
aggregation to all be right *together*, so it is the one number that would move
if any of them broke.

The intrinsics are fitted here rather than assumed: the camera grid gives 144
correspondences between known pixels and D4RT's own 3D points at t_cam=0, so
(fx, fy, cx, cy) come out by least squares.  That the fitted principal point
lands on the image centre is itself a check -- nothing in the fit forces it.

Expect a few percent of the image diagonal, not zero: the 3D aggregate is a
geometric median of points spread over the subject's body and the comparison is
against a 2D area centroid, which are different quantities, and the trajectory
is Savitzky-Golay smoothed on top.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import (
    TARGETS,
    Target,
    read_json,
    video_paths,
    write_json,
)


def fit_intrinsics(grid_px: np.ndarray, xyz_cam0: np.ndarray) -> np.ndarray:
    """Least-squares (fx, fy, cx, cy) from pixel <-> 3D correspondences."""

    depth = xyz_cam0[:, 2]
    ok = np.isfinite(depth) & (depth > 1e-6) & np.isfinite(xyz_cam0).all(axis=1)
    if ok.sum() < 8:
        raise RuntimeError(f"only {ok.sum()} usable grid points for the intrinsics fit")

    design_u = np.stack([xyz_cam0[ok, 0] / depth[ok], np.ones(int(ok.sum()))], axis=1)
    fx, cx = np.linalg.lstsq(design_u, grid_px[ok, 0], rcond=None)[0]
    design_v = np.stack([xyz_cam0[ok, 1] / depth[ok], np.ones(int(ok.sum()))], axis=1)
    fy, cy = np.linalg.lstsq(design_v, grid_px[ok, 1], rcond=None)[0]
    return np.array([fx, fy, cx, cy], dtype=np.float64)


def mask_centroids(masks: np.ndarray) -> np.ndarray:
    out = np.full((masks.shape[0], 2), np.nan)
    for t, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if xs.size:
            out[t] = (xs.mean(), ys.mean())
    return out


def verify_target(target: Target, run_dir: Path) -> dict[str, Any]:
    paths = video_paths(run_dir, target)
    points = read_json(paths.points_json)
    width, height = int(points["width"]), int(points["height"])

    camera = np.load(paths.d4rt / "camera.npz")
    # camera.npz is stacked over t_cam, so index 0 is the t_cam=0 view.
    xyz_cam0 = np.asarray(camera["xyz_3d"])[0]
    intrinsics = fit_intrinsics(np.array(points["camera_grid_px"], dtype=np.float64), xyz_cam0)
    fx, fy, cx, cy = intrinsics

    poses = np.load(paths.analysis / "camera_poses.npz")
    rotation, translation = poses["rotation"], poses["translation"]

    table = np.genfromtxt(paths.analysis / "trajectory.csv", delimiter=",", names=True)
    track = np.stack([table["smooth_x"], table["smooth_y"], table["smooth_z"]], axis=1)
    observed = mask_centroids(np.load(paths.masks / "masks.npy"))

    errors, per_frame = [], []
    for t in range(len(track)):
        if not (np.isfinite(track[t]).all() and np.isfinite(observed[t]).all()):
            continue
        in_camera_t = rotation[t].T @ (track[t] - translation[t])
        if in_camera_t[2] <= 1e-6:
            continue
        u = fx * in_camera_t[0] / in_camera_t[2] + cx
        v = fy * in_camera_t[1] / in_camera_t[2] + cy
        error = float(np.hypot(u - observed[t, 0], v - observed[t, 1]))
        errors.append(error)
        per_frame.append({"frame": t, "reprojected_uv": [u, v],
                          "observed_uv": observed[t].tolist(), "error_px": error})

    if not errors:
        return {"slug": target.slug, "usable_frames": 0,
                "verdict": "NO OVERLAP: nothing to compare"}

    errors_arr = np.asarray(errors)
    diagonal = float(np.hypot(width, height))
    median_fraction = float(np.median(errors_arr) / diagonal)
    return {
        "slug": target.slug,
        "subject": target.subject,
        "usable_frames": len(errors),
        "image_size": [width, height],
        "intrinsics_fxfycxcy": intrinsics.tolist(),
        # Nothing in the fit forces this onto the image centre, so a large
        # offset means D4RT's point cloud is not a pinhole projection of
        # the pixels we asked about.
        "principal_point_offset_px": [float(cx - width / 2), float(cy - height / 2)],
        "reprojection_px": {
            "median": float(np.median(errors_arr)),
            "mean": float(errors_arr.mean()),
            "p90": float(np.percentile(errors_arr, 90)),
            "max": float(errors_arr.max()),
        },
        "median_fraction_of_diagonal": median_fraction,
        "verdict": "consistent" if median_fraction < 0.05 else "INCONSISTENT",
        "per_frame": per_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    results = []
    for target in TARGETS:
        if args.only and target.slug not in args.only:
            continue
        if not (args.run_dir / target.slug / "analysis" / "camera_poses.npz").exists():
            print(f"{target.slug}: no analysis output, skipping", flush=True)
            continue
        record = verify_target(target, args.run_dir)
        results.append(record)
        if record["usable_frames"]:
            print(
                f"{record['slug'][:44]:46s} reproj median "
                f"{record['reprojection_px']['median']:6.2f} px "
                f"({record['median_fraction_of_diagonal'] * 100:.1f}% of diagonal)  "
                f"{record['verdict']}",
                flush=True,
            )

    write_json(args.run_dir / "verification.json", results)
    print(f"\nwrote {args.run_dir / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
