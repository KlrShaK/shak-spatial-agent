"""Stage 3b: turn raw model output into one trajectory and one orientation track.

Runs on CPU.  Three things happen here, in order, because each depends on the
last:

  1. Camera motion is recovered from the two-viewpoint grid queries.
  2. The 20 subject point tracks are pruned and collapsed into one trajectory.
  3. Orientation angles are converted into camera axes, corrected for the crop's
     position in the frame, and optionally rotated into the frame-0 camera.

Step 3 needs step 1 because Orient-Anything answers relative to the camera *in
that frame*, whereas the trajectory lives in the frame-0 camera.  On a moving
camera those two disagree, so both versions are written out and the difference
between them is exactly the camera's rotation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import (
    Target,
    TARGETS,
    default_run_dir,
    read_json,
    video_paths,
    write_json,
)
from d4rt_agent.sam_orientany_d4rt_test.traj3d_geometry import (
    aggregate_trajectory,
    cloud_scale,
    estimate_intrinsics,
    oav2_axes_in_opencv,
    offaxis_correction,
    estimate_world_up,
    robust_rigid,
    select_tracks,
    sigmoid,
    smooth_trajectory,
)

VISIBILITY_THRESHOLD = 0.5
# From the confidence objective c*e - 0.2*log(c), whose minimiser is c = 0.2/e.
# c = 0.2 therefore corresponds to a predicted error of order the scene depth.
CONFIDENCE_THRESHOLD = 0.2


def recover_camera_poses(camera: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recover each frame's camera pose relative to frame 0.

    ``camera["xyz_3d"]`` is (T, P, 3): the same P points at t_tgt=0, seen from
    each of the T viewpoints.  Because both views describe identical world
    geometry, they differ by exactly the camera's rigid motion:

        X_0 = T_cw[0] p_world  and  X_t = T_cw[t] p_world
        =>  X_0 = T_cw[0] T_cw[t]^-1 X_t

    p_world cancels, which is why the points need not be static -- or even off
    the subject.  (Proof is in the training target definition,
    src/data/kubric_full_robust_dataset.py:970-976.)
    """

    xyz = np.asarray(camera["xyz_3d"], dtype=np.float64)
    weights = sigmoid(camera["confidence"]) ** 2
    visible = sigmoid(camera["visibility"]) > VISIBILITY_THRESHOLD
    weights = np.where(visible, weights, 0.0)

    reference = xyz[0]
    num_frames = xyz.shape[0]
    rotations = np.tile(np.eye(3), (num_frames, 1, 1))
    translations = np.zeros((num_frames, 3))
    rmse = np.full(num_frames, np.nan)
    scales = np.full(num_frames, np.nan)
    inlier_counts = np.zeros(num_frames, dtype=int)
    conditioning = np.full(num_frames, np.nan)
    failures: list[int] = []

    for t in range(num_frames):
        if t == 0:
            inlier_counts[0] = int((weights[0] > 0).sum())
            rmse[0] = 0.0
            scales[0] = 1.0
            continue
        fit = robust_rigid(
            xyz[t], reference, weights=np.minimum(weights[t], weights[0]), seed=t
        )
        if fit is None:
            failures.append(t)
            continue
        rotations[t] = fit.rotation
        translations[t] = fit.translation
        rmse[t] = fit.rmse
        scales[t] = fit.scale
        inlier_counts[t] = int(fit.inliers.sum())
        conditioning[t] = fit.conditioning

    scene_scale = cloud_scale(reference)
    return {
        "rotation": rotations,
        "translation": translations,
        "rmse": rmse,
        "rmse_relative": rmse / max(scene_scale, 1e-9),
        "sim3_scale": scales,
        "inlier_counts": inlier_counts,
        "conditioning": conditioning,
        "failed_frames": failures,
        "scene_scale": scene_scale,
    }


def build_trajectory(subject: dict[str, np.ndarray]) -> dict[str, Any]:
    """Prune the point tracks and collapse them into one trajectory."""

    xyz = np.asarray(subject["xyz_3d"], dtype=np.float64)  # (P, T, 3)
    visibility = sigmoid(subject["visibility"])
    confidence = sigmoid(subject["confidence"])

    valid = (
        np.isfinite(xyz).all(axis=-1)
        & (visibility > VISIBILITY_THRESHOLD)
        & (confidence > CONFIDENCE_THRESHOLD)
    )
    weights = confidence**2

    selection = select_tracks(xyz, valid)
    aggregated = aggregate_trajectory(xyz, valid, weights, selection.keep)
    smoothed = smooth_trajectory(aggregated["ransac_median"])

    return {
        "xyz": xyz,
        "valid": valid,
        "visibility": visibility,
        "confidence": confidence,
        "keep": selection.keep,
        "reasons": selection.reasons,
        "scale": selection.scale,
        "aggregated": aggregated,
        "smoothed": smoothed,
    }


def build_orientation(
    orientation: dict[str, Any],
    poses: dict[str, Any],
    intrinsics: np.ndarray,
) -> list[dict[str, Any]]:
    """Convert OAV2 angles into camera-frame axes, corrected and compensated."""

    rows: list[dict[str, Any]] = []
    for entry in orientation["frames"]:
        frame = int(entry["frame"])
        row: dict[str, Any] = {"frame": frame, "status": entry.get("status", "missing")}
        if entry.get("status") != "ok":
            rows.append(row)
            continue

        az = float(entry["azimuth_deg"])
        el = float(entry["elevation_deg"])
        ro = float(entry["roll_deg"])
        alpha = float(entry.get("alpha", 0.0))

        forward, lateral, up = oav2_axes_in_opencv(az, el, ro)
        centre = entry.get("crop_centre_px")
        if centre is not None:
            forward = offaxis_correction(forward, centre, intrinsics)
            lateral = offaxis_correction(lateral, centre, intrinsics)
            up = offaxis_correction(up, centre, intrinsics)

        rotation = poses["rotation"][frame]
        row.update(
            {
                "azimuth_deg": az,
                "elevation_deg": el,
                "roll_deg": ro,
                "alpha": alpha,
                # alpha == 1 means a single distinguishable front. Anything else
                # (0 = unresolvable, 2/4 = symmetric) makes facing meaningless.
                "orientation_usable": bool(alpha == 1.0),
                "forward_cam_t": forward,
                "lateral_cam_t": lateral,
                "up_cam_t": up,
                # Directions rotate but do not translate, so only R is applied.
                "forward_cam0": rotation @ forward,
                "lateral_cam0": rotation @ lateral,
                "up_cam0": rotation @ up,
            }
        )
        rows.append(row)
    return rows


def _write_trajectory_csv(path: Path, trajectory: dict[str, Any]) -> None:
    aggregated = trajectory["aggregated"]
    smoothed = trajectory["smoothed"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "mean_x", "mean_y", "mean_z",
                "median_x", "median_y", "median_z",
                "ransac_x", "ransac_y", "ransac_z",
                "smooth_x", "smooth_y", "smooth_z",
                "n_inliers",
            ]
        )
        for t in range(aggregated["ransac_median"].shape[0]):
            writer.writerow(
                [t]
                + [f"{v:.6f}" for v in aggregated["mean"][t]]
                + [f"{v:.6f}" for v in aggregated["median"][t]]
                + [f"{v:.6f}" for v in aggregated["ransac_median"][t]]
                + [f"{v:.6f}" for v in smoothed[t]]
                + [int(aggregated["n_inliers"][t])]
            )


def _write_orientation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame", "status", "azimuth_deg", "elevation_deg", "roll_deg",
                "alpha", "orientation_usable",
                "fwd_camt_x", "fwd_camt_y", "fwd_camt_z",
                "fwd_cam0_x", "fwd_cam0_y", "fwd_cam0_z",
                "up_cam0_x", "up_cam0_y", "up_cam0_z",
            ]
        )
        for row in rows:
            if row.get("status") != "ok":
                writer.writerow([row["frame"], row.get("status", "missing")] + [""] * 14)
                continue
            writer.writerow(
                [
                    row["frame"], "ok",
                    f"{row['azimuth_deg']:.2f}",
                    f"{row['elevation_deg']:.2f}",
                    f"{row['roll_deg']:.2f}",
                    f"{row['alpha']:.0f}",
                    int(row["orientation_usable"]),
                ]
                + [f"{v:.6f}" for v in row["forward_cam_t"]]
                + [f"{v:.6f}" for v in row["forward_cam0"]]
                + [f"{v:.6f}" for v in row["up_cam0"]]
            )


# A grid point this dark in every single frame has no image content behind it.
LETTERBOX_LUMA = 12.0


def padding_grid_points(frames_dir: Path, grid_px: np.ndarray) -> np.ndarray:
    """Which grid points sit on letterbox padding rather than on the image.

    These clips are letterboxed and the grid is uniform over the *stored* frame,
    so a good number of points land on black bars. D4RT invents a near-constant
    depth for them, which makes them perfect rigid inliers and perfect planar
    inliers -- padding fits *better* than real content.

    That matters here because the largest plane in the cloud decides the world
    vertical. On the yellow-jacket clip -- a level shot across a frozen lake --
    the bars outvoted the ice and returned an "up" tilted 59 degrees. Excluding
    them is what makes the estimate mean anything.

    Returns a boolean mask over ``grid_px``; all-False when the clip has no bars.
    """

    from PIL import Image

    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        return np.zeros(len(grid_px), dtype=bool)
    luma = np.stack([
        np.asarray(Image.open(f).convert("L"), dtype=np.float64) for f in frames
    ])
    height, width = luma.shape[1:]
    index = np.round(np.asarray(grid_px)).astype(int)
    index[:, 0] = np.clip(index[:, 0], 0, width - 1)
    index[:, 1] = np.clip(index[:, 1], 0, height - 1)
    return (luma[:, index[:, 1], index[:, 0]] < LETTERBOX_LUMA).all(axis=0)


def analyze_target(
    target: Target, run_dir: Path, *, paths: Any | None = None
) -> dict[str, Any]:
    """Turn one target's raw model output into a trajectory and an orientation track.

    ``paths`` overrides the slug-derived location, for the per-question agent
    driver which keys its work directories on (video, subject) rather than on the
    video alone.

    The subject arrays are optional: a camera-only question never segments a
    subject, and camera recovery needs only the grid.
    """

    paths = video_paths(run_dir, target) if paths is None else paths
    camera = dict(np.load(paths.d4rt / "camera.npz"))
    points = read_json(paths.points_json)

    subject_path = paths.d4rt / "subject.npz"
    subject = dict(np.load(subject_path)) if subject_path.exists() else None

    poses = recover_camera_poses(camera)
    trajectory = build_trajectory(subject) if subject is not None else None

    # The 144-point camera grid, NOT the 20 subject points.
    #
    # The estimator inverts fx = z*|u - cx| / |x| per point. That is exact on
    # exact inputs wherever the points sit, so the issue is conditioning, not
    # clustering as such: subject points all sit inside one small mask, where
    # both |x| and |u - cx| are tiny, and D4RT's depth noise lands on the
    # denominator. On the yellow-jacket clip (a 114 px mask) that returned fx=51
    # against a true ~256, fx and fy disagreeing four-fold about a square pixel.
    # The grid spans the frame and recovers fx=256.7, fy=252.4.
    #
    # This feeds offaxis_correction, so a focal length that wrong silently
    # mis-rotates every facing direction it is supposed to be fixing.
    intrinsics = estimate_intrinsics(
        np.asarray(camera["xyz_3d"])[0],
        np.asarray(points["camera_grid_uv_norm"]),
        (points["height"], points["width"]),
    )

    # World up, for the reports that need a vertical the camera's own frame does
    # not supply. Padding points are excluded first -- see padding_grid_points.
    grid_px = np.asarray(points["camera_grid_px"], dtype=np.float64)
    grid_xyz = np.asarray(camera["xyz_3d"])[0]
    grid_weights = np.where(
        sigmoid(np.asarray(camera["visibility"])[0]) > VISIBILITY_THRESHOLD,
        sigmoid(np.asarray(camera["confidence"])[0]) ** 2,
        0.0,
    )
    is_padding = padding_grid_points(paths.frames, grid_px)
    grid_weights = np.where(is_padding, 0.0, grid_weights)
    world_up, world_up_fraction = estimate_world_up(grid_xyz, grid_weights)

    orientation_file = paths.orientation / "orientation.json"
    orientation_rows: list[dict[str, Any]] = []
    if orientation_file.exists():
        orientation_rows = build_orientation(read_json(orientation_file), poses, intrinsics)

    if trajectory is not None:
        _write_trajectory_csv(paths.analysis / "trajectory.csv", trajectory)
    if orientation_rows:
        _write_orientation_csv(paths.analysis / "orientation.csv", orientation_rows)

    np.savez_compressed(
        paths.analysis / "camera_poses.npz",
        world_up_cam0=(world_up if world_up is not None else np.full(3, np.nan)),
        world_up_inlier_fraction=np.float64(world_up_fraction),
        grid_padding_fraction=np.float64(float(is_padding.mean())),
        rotation=poses["rotation"],
        translation=poses["translation"],
        rmse=poses["rmse"],
        rmse_relative=poses["rmse_relative"],
        sim3_scale=poses["sim3_scale"],
        inlier_counts=poses["inlier_counts"],
        conditioning=poses["conditioning"],
    )
    if trajectory is not None:
        _write_trajectory_npz(paths, trajectory, orientation_rows)

    summary = _summarise(target, trajectory, poses, orientation_rows, intrinsics)
    write_json(paths.analysis / "summary.json", summary)
    return summary


def _write_trajectory_npz(
    paths: Any, trajectory: dict[str, Any], orientation_rows: list[dict[str, Any]]
) -> None:
    np.savez_compressed(
        paths.analysis / "trajectory.npz",
        xyz=trajectory["xyz"],
        valid=trajectory["valid"],
        keep=trajectory["keep"],
        mean=trajectory["aggregated"]["mean"],
        median=trajectory["aggregated"]["median"],
        ransac_median=trajectory["aggregated"]["ransac_median"],
        smoothed=trajectory["smoothed"],
        n_inliers=trajectory["aggregated"]["n_inliers"],
        forward_cam0=np.array(
            [
                r["forward_cam0"] if r.get("status") == "ok" else np.full(3, np.nan)
                for r in orientation_rows
            ]
        )
        if orientation_rows
        else np.zeros((0, 3)),
        up_cam0=np.array(
            [
                r["up_cam0"] if r.get("status") == "ok" else np.full(3, np.nan)
                for r in orientation_rows
            ]
        )
        if orientation_rows
        else np.zeros((0, 3)),
        orientation_usable=np.array(
            [bool(r.get("orientation_usable", False)) for r in orientation_rows]
        )
        if orientation_rows
        else np.zeros((0,), dtype=bool),
    )


def _summarise(
    target: Target,
    trajectory: dict[str, Any] | None,
    poses: dict[str, Any],
    orientation_rows: list[dict[str, Any]],
    intrinsics: np.ndarray,
) -> dict[str, Any]:
    """The run's own account of what it produced and how far to trust it.

    Subject fields are None on a camera-only run rather than absent, so a reader
    can tell "there was no subject" from "the subject failed", and so every
    summary has the same shape.
    """

    summary: dict[str, Any] = {
        "slug": target.slug,
        "subject": target.subject,
        "camera": {
            "scene_scale": poses["scene_scale"],
            "median_relative_rmse": float(np.nanmedian(poses["rmse_relative"][1:])),
            "max_relative_rmse": float(np.nanmax(poses["rmse_relative"][1:])),
            "median_sim3_scale": float(np.nanmedian(poses["sim3_scale"][1:])),
            "min_inliers": int(poses["inlier_counts"][1:].min()),
            "failed_frames": poses["failed_frames"],
            "total_rotation_deg": _total_rotation_degrees(poses["rotation"]),
        },
        "orientation": {
            "frames_ok": sum(1 for r in orientation_rows if r.get("status") == "ok"),
            "frames_usable": sum(
                1 for r in orientation_rows if r.get("orientation_usable")
            ),
            "intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        },
    }

    if trajectory is None:
        summary.update({
            "tracks_kept": None, "tracks_total": None, "tracks_rejected": None,
            "subject_scale": None, "trajectory_path_length": None,
            "displacement_start_to_end": None,
            "warnings": _warnings(None, poses),
        })
        return summary

    finite = np.isfinite(trajectory["smoothed"]).all(axis=1)
    path_length = float(
        np.linalg.norm(np.diff(trajectory["smoothed"][finite], axis=0), axis=1).sum()
    ) if finite.sum() > 1 else float("nan")

    summary.update({
        "tracks_kept": int(trajectory["keep"].sum()),
        "tracks_total": int(trajectory["keep"].size),
        "tracks_rejected": trajectory["reasons"],
        "subject_scale": trajectory["scale"],
        "trajectory_path_length": path_length,
        "displacement_start_to_end": float(
            np.linalg.norm(
                trajectory["smoothed"][finite][-1] - trajectory["smoothed"][finite][0]
            )
        )
        if finite.sum() > 1
        else float("nan"),
        "warnings": _warnings(trajectory, poses),
    })
    return summary


def _total_rotation_degrees(rotations: np.ndarray) -> float:
    """How far the camera swung overall, as a single readable number."""

    total = 0.0
    for t in range(1, rotations.shape[0]):
        relative = rotations[t] @ rotations[t - 1].T
        cos = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        total += float(np.degrees(np.arccos(cos)))
    return total


def _warnings(trajectory: dict[str, Any] | None, poses: dict[str, Any]) -> list[str]:
    """Conditions that make the output untrustworthy but do not raise."""

    out: list[str] = []
    if trajectory is not None:
        kept = int(trajectory["keep"].sum())
        if kept < 6:
            out.append(
                f"only {kept} of {trajectory['keep'].size} tracks survived; "
                "this usually means the seeding frame was bad, not the points"
            )
    median_rmse = float(np.nanmedian(poses["rmse_relative"][1:]))
    if np.isfinite(median_rmse) and median_rmse > 0.05:
        out.append(
            f"camera alignment residual is {median_rmse:.3f} of scene scale; "
            "D4RT cross-t_cam consistency is the limiting factor here"
        )
    scale = float(np.nanmedian(poses["sim3_scale"][1:]))
    if np.isfinite(scale) and abs(scale - 1.0) > 0.05:
        out.append(
            f"sim3 scale across viewpoints is {scale:.3f}, not ~1; "
            "rigid alignment may be the wrong model for this clip"
        )
    if poses["failed_frames"]:
        out.append(f"camera pose failed for frames {poses['failed_frames']}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    for target in TARGETS:
        if args.only and target.slug not in args.only:
            continue
        summary = analyze_target(target, run_dir)
        print(
            f"{target.slug}: kept {summary['tracks_kept']}/{summary['tracks_total']} tracks, "
            f"camera rmse {summary['camera']['median_relative_rmse']:.4f}, "
            f"{summary['orientation']['frames_usable']} usable orientations",
            flush=True,
        )
        for warning in summary["warnings"]:
            print(f"  WARNING: {warning}", flush=True)


if __name__ == "__main__":
    main()
