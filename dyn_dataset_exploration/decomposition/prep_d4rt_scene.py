"""GPU prep: a D4RT scene pass per clip, in the same file layout as Point4D's.

``scene4d.BACKEND = "d4rt"`` reads ``clips/<slug>/d4rt/scene.npz`` instead of ``point4d/scene.npz``
(the notebook's ``BACKEND`` switch). Everything downstream -- SAM3 masks lifted through ``points``,
tracks, camera poses, ``K`` for zoom -- is unchanged; only who produced the 3D differs.

What D4RT gives, and how it becomes a Point4D-shaped scene (world = the camera of the first scene
frame, +X right, +Y down, +Z forward, model-relative units):

* ``points[t]``  -- every cell of a stride-8 grid of frame ``t`` queried with ``t_src = t_tgt = t``
  in the frame-0 camera (``tracks_xyz_ref0``): the world position of what frame ``t`` shows.
* ``cam_R, cam_t`` -- the same queries also come back in frame ``t``'s own camera
  (``tracks_xyz_local``). Both describe the same points, so they differ by the camera's rigid
  motion: ``ref0 = R @ local + t``; RANSAC-Kabsch (``traj3d_geometry.robust_rigid``) on the
  confident cells gives ``R = cam_R[t]`` (camera -> world) and ``t = cam_t[t]`` (camera centre).
* ``K[t]``       -- D4RT has no intrinsics output; focal length is the median of
  ``(u - cx) * Z / X`` over the cells of the local cloud (principal point at the image centre).
* ``grid_traj``  -- a frame-0 grid tracked through the clip (``t_src = 0``, ``tracks_xyz_ref0``).

**Frames.** D4RT's window is 32 frames, so the scene is at most 32 generator frames: the benchmark's
16 sampled frames plus evenly spaced fill-ins (all of them when the clip has 32 or fewer). Stitching
several windows is not done; the time resolution is coarser than Point4D's on long clips.

    PYTHONPATH=<repo>:<repo>/third-party-tools/Open-D4RT:<Point4D> python -m \\
        dyn_dataset_exploration.decomposition.prep_d4rt_scene --out <catalog dir> [--shard i/n] [--only slug ...]

The metric scale (``d4rt/metric.json``) is the DA3-Nested stage of ``prep_point4d`` run on this scene:
``prep_point4d --stages metric --backend d4rt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

from dyn_dataset_exploration.dyn_bench_data import read_manifest, write_manifest
from dyn_dataset_exploration.decomposition.prep_point4d import LANDSCAPE_HW, PORTRAIT_HW, load_frames, query_grid

NUM_FRAMES = 32               # D4RT's clip length
CELL_PX = 8                   # dense grid spacing at model_hw (points[...] index = pixel // stride)
MIN_CELLS_FOR_FIT = 60
BACKEND_DIR = "d4rt"


def pick_frames(gf_index: np.ndarray, sampled16: Sequence[int], n: int = NUM_FRAMES) -> np.ndarray:
    """Positions (into ``gf_index``) of at most ``n`` frames: the first frame and the benchmark's 16
    sampled ones, then in-between frames until D4RT's window is full -- each new frame is the midpoint of
    the widest remaining gap, so the extra frames are spread evenly between the sampled ones."""

    total = len(gf_index)
    if total <= n:
        return np.arange(total)
    chosen = sorted({0} | {int(np.argmin(np.abs(gf_index - g))) for g in sampled16})
    while len(chosen) < n:
        gaps = np.diff(chosen)
        widest = int(np.argmax(gaps))
        if gaps[widest] < 2:                  # every frame in between is already taken
            break
        chosen.insert(widest + 1, chosen[widest] + int(gaps[widest]) // 2)
    return np.array(chosen)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def dense_grid(hw: tuple[int, int], cell: int = CELL_PX) -> tuple[np.ndarray, tuple[int, int]]:
    """Normalised (u, v) of every cell centre, row-major, and the grid shape."""

    h, w = hw
    gh, gw = -(-h // cell), -(-w // cell)
    ys = np.minimum((np.arange(gh) + 0.5) * cell, h - 1) / h
    xs = np.minimum((np.arange(gw) + 0.5) * cell, w - 1) / w
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32), (gh, gw)


def focal_from_points(local: np.ndarray, uv_norm: np.ndarray, weight: np.ndarray, hw: tuple[int, int]) -> tuple[float, float]:
    """Pinhole focal lengths (px at ``hw``) from a camera-frame cloud and the pixels it came from."""

    h, w = hw
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    du, dv = (uv_norm[:, 0] - 0.5) * w, (uv_norm[:, 1] - 0.5) * h
    ok = np.isfinite(local).all(1) & (z > 1e-6) & (weight > 0.1)

    def ratio(d, c, size):
        use = ok & (np.abs(d) > 0.1 * size) & (np.abs(c) > 1e-6)
        return float(np.median(d[use] * z[use] / c[use])) if use.sum() >= 20 else float("nan")

    return ratio(du, x, w), ratio(dv, y, h)


def stage(out: Path, slugs: Sequence[str]) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third-party-tools" / "Open-D4RT"))
    from infer_track_3d import _infer_tracks, _resize_video
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_geometry import robust_rigid
    from dyn_dataset_exploration.decomposition.prep_d4rt import load_model

    model, size = load_model()
    for slug in slugs:
        clip = json.loads((out / "clips" / slug / "clip.json").read_text())
        target = out / "clips" / slug / BACKEND_DIR / "scene.npz"
        if target.exists():
            continue
        started = time.time()
        frames_all, gf_all = load_frames(out, clip)
        keep = pick_frames(gf_all, clip["sampled16"])
        frames, gf_index = frames_all[keep], gf_all[keep]
        n = len(keep)
        hw = PORTRAIT_HW if clip["portrait"] else LANDSCAPE_HW
        video = _resize_video(frames, image_hw=size)

        uv_cell, (gh, gw) = dense_grid(hw)
        cells = len(uv_cell)
        # every cell of every frame, sourced on its own frame; only the diagonal (target = source) is kept
        queries = np.tile(uv_cell, (n, 1))
        source = np.repeat(np.arange(n, dtype=np.int64), cells)
        dense = _infer_tracks(model=model, video_model_rgb=video, query_uv_norm=queries, query_chunk_size=8192,
                              query_src_indices_global=source)
        idx = np.arange(n)
        ref = np.stack([dense["tracks_xyz_ref0"][f * cells:(f + 1) * cells, f] for f in idx])        # (n, cells, 3)
        local = np.stack([dense["tracks_xyz_local"][f * cells:(f + 1) * cells, f] for f in idx])
        conf = np.stack([_sigmoid(dense["tracks_confidence"][f * cells:(f + 1) * cells, f]) *
                         _sigmoid(dense["tracks_visibility_logits"][f * cells:(f + 1) * cells, f]) for f in idx])
        del dense

        cam_R, cam_t = np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3))
        rmse, fit_scale, failed = np.zeros(n), np.ones(n), []
        for f in range(1, n):
            fit = robust_rigid(local[f], ref[f], weights=np.minimum(conf[f], 1.0) ** 2, seed=f)
            if fit is None or fit.inliers.sum() < MIN_CELLS_FOR_FIT:
                failed.append(f)
                cam_R[f], cam_t[f] = cam_R[f - 1], cam_t[f - 1]     # hold the last pose; reported
                continue
            cam_R[f], cam_t[f], rmse[f], fit_scale[f] = fit.rotation, fit.translation, fit.rmse, fit.scale

        K = np.zeros((n, 3, 3))
        h, w = hw
        fx_last, fy_last = float(w), float(w)
        for f in range(n):
            fx, fy = focal_from_points(local[f], uv_cell, conf[f], hw)
            fx = fx if np.isfinite(fx) else fx_last
            fy = fy if np.isfinite(fy) else fx                      # square pixels when only one axis is observable
            fx_last = fx
            K[f] = [[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]]

        grid_uv = query_grid(hw)
        track = _infer_tracks(model=model, video_model_rgb=video,
                              query_uv_norm=(grid_uv / np.array([w, h], dtype=np.float32)).astype(np.float32),
                              query_chunk_size=8192)
        grid_traj = np.transpose(track["tracks_xyz_ref0"], (1, 0, 2)).astype(np.float32)           # (n, N, 3)
        grid_conf = np.transpose(_sigmoid(track["tracks_confidence"]) *
                                 _sigmoid(track["tracks_visibility_logits"]), (1, 0)).astype(np.float32)

        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target, gf_index=gf_index, model_hw=np.asarray(hw), source_hw=np.asarray(clip["source_hw"]),
            cam_R=cam_R, cam_t=cam_t, K=K, points=ref.reshape(n, gh, gw, 3).astype(np.float16),
            geometry_stride=np.int32(CELL_PX), grid_uv=grid_uv, grid_traj=grid_traj, grid_conf=grid_conf)
        report = {"clip_slug": slug, "frames": int(n), "gf_index": gf_index.tolist(), "model_hw": list(hw),
                  "seconds": round(time.time() - started, 1), "pose_fit_failed_frames": failed,
                  "pose_fit_rmse_median": float(np.median(rmse[1:])) if n > 1 else 0.0,
                  "pose_fit_sim3_scale_median": float(np.median(fit_scale[1:])) if n > 1 else 1.0,
                  "focal_px_min_max": [float(K[:, 0, 0].min()), float(K[:, 0, 0].max())],
                  "finite_points": float(np.isfinite(ref).all(-1).mean()),
                  **compare_with_point4d(out, slug, gf_index, cam_R, cam_t, ref.reshape(n, gh, gw, 3))}
        write_manifest(report, target.with_suffix(".json"))
        print(f"d4rt scene {slug}: {json.dumps(report)}", flush=True)


def compare_with_point4d(out: Path, slug: str, gf_index: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray,
                         points: np.ndarray) -> dict:
    """How far the D4RT scene agrees with Point4D's up to the (arbitrary) scale ratio -- a sanity check,
    not ground truth. Camera path and depth are compared on the frames both scenes have."""

    path = out / "clips" / slug / "point4d" / "scene.npz"
    if not path.exists():
        return {}
    with np.load(path) as z:
        p_gf, p_t, p_R, p_pts = z["gf_index"], z["cam_t"], z["cam_R"], z["points"].astype(np.float32)
    pos = np.array([int(np.argmin(np.abs(p_gf - g))) for g in gf_index])
    ok = p_gf[pos] == gf_index
    if ok.sum() < 4:
        return {}
    pos, a_t, b_t = pos[ok], cam_t[ok], p_t[pos[ok]]
    la, lb = np.linalg.norm(np.diff(a_t, axis=0), axis=1).sum(), np.linalg.norm(np.diff(b_t, axis=0), axis=1).sum()
    # per-frame camera-to-point depth along the optical axis, correlated across frames and cells
    def depth(P, R, t):
        return np.einsum("thwi,tij->thwj", P - t[:, None, None, :], R)[..., 2]
    d_a = depth(points[ok].astype(np.float32), cam_R[ok], a_t)
    d_b = depth(p_pts[pos], p_R[pos], b_t)
    # the grids differ (stride 8 vs Point4D's 4): sample Point4D at every second cell
    d_b = d_b[:, 1::2, 1::2][:, :d_a.shape[1], :d_a.shape[2]]
    d_a = d_a[:, :d_b.shape[1], :d_b.shape[2]]
    m = np.isfinite(d_a) & np.isfinite(d_b) & (d_a > 0) & (d_b > 0)
    if m.sum() < 100:
        return {}
    ratio = np.median(d_a[m] / d_b[m])
    return {"vs_point4d": {"depth_scale_ratio_d4rt_over_p4d": float(ratio),
                           "log_depth_corr": float(np.corrcoef(np.log(d_a[m]), np.log(d_b[m]))[0, 1]),
                           "camera_path_length_ratio": float(la / max(lb, 1e-9)),
                           "frames_compared": int(ok.sum())}}


def main(argv: Sequence[str] | None = None) -> None:
    from dyn_dataset_exploration.decomposition.build_catalog import DEFAULT_OUT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stages", nargs="+", default=["scene"], choices=["scene"])   # the job launcher passes --stages
    parser.add_argument("--only", nargs="*", default=None, help="clip slugs")
    parser.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th clip")
    args = parser.parse_args(argv)
    slugs = sorted({e["clip_slug"] for e in read_manifest(args.out / "catalog.json")["questions"]})
    if args.only:
        slugs = [s for s in slugs if s in set(args.only)]
    index, count = (int(v) for v in args.shard.split("/"))
    stage(args.out, slugs[index::count])


if __name__ == "__main__":
    main()
