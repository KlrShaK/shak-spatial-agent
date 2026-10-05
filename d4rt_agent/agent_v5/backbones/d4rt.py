"""D4RT (Open-D4RT, 32-frame checkpoint) as a scene backbone, in the same layout as Point4D's.

Live port of ``dyn_dataset_exploration/decomposition/prep_d4rt_scene.py``:

* ``points[t]``: every cell of a stride-8 grid of frame ``t`` queried with ``t_src = t_tgt = t`` in
  the frame-0 camera (``tracks_xyz_ref0``) -- the world position of what frame ``t`` shows.
* ``cam_R, cam_t``: the same queries in frame ``t``'s own camera (``tracks_xyz_local``) differ from
  ``ref0`` by the camera's rigid motion; RANSAC-Kabsch (``traj3d_geometry.robust_rigid``) on the
  confident cells gives it.
* ``K[t]``: D4RT has no intrinsics output; focal length is the median of ``(u - cx) * Z / X``.
* ``grid_traj``: a frame-0 grid tracked through the clip.

**Frames.** D4RT's window is 32 frames: the first frame, the answer frames, and evenly spread
fill-ins (``pick_frames``). Long clips are therefore covered more coarsely than by Point4D.

**Metric.** Backbone-native, as chosen for v5: D4RT's depth units are taken as metres
(``source = "d4rt-native"``). The research pipeline measured them at about 0.45x DA3-Nested's
scale on dev clips, so every metric reading names this source.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from d4rt_agent.agent_v5.backbones.base import Backbone, model_hw, query_grid
from d4rt_agent.agent_v5.clips import MAX_GF_FRAMES, Clip

NUM_FRAMES = 32
CELL_PX = 8
MIN_CELLS_FOR_FIT = 60
CHECKPOINT_NAME = "OpenD4RT_32CLIP_9Dataset_NoAUG"


def open_d4rt_root() -> Path:
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import D4RT_CORE_ROOT

    return Path(D4RT_CORE_ROOT)


def pick_frames(gf_index: np.ndarray, sampled: Sequence[int], n: int = NUM_FRAMES) -> np.ndarray:
    """Positions (into ``gf_index``) of at most ``n`` frames: the first and the answer frames, then
    midpoints of the widest gaps until the window is full."""

    total = len(gf_index)
    if total <= n:
        return np.arange(total)
    chosen = sorted({0} | {int(np.argmin(np.abs(gf_index - g))) for g in sampled})
    if len(chosen) > n:                      # e.g. DSI's 32 answer frames plus frame 0
        chosen = [chosen[i] for i in np.unique(np.round(np.linspace(0, len(chosen) - 1, n)).astype(int))]
    while len(chosen) < n:
        gaps = np.diff(chosen)
        widest = int(np.argmax(gaps))
        if gaps[widest] < 2:
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


def focal_from_points(local: np.ndarray, uv_norm: np.ndarray, weight: np.ndarray,
                      hw: tuple[int, int]) -> tuple[float, float]:
    """Pinhole focal lengths (px at ``hw``) from a camera-frame cloud and the pixels it came from."""

    h, w = hw
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    du, dv = (uv_norm[:, 0] - 0.5) * w, (uv_norm[:, 1] - 0.5) * h
    ok = np.isfinite(local).all(1) & (z > 1e-6) & (weight > 0.1)

    def ratio(d, c, size):
        use = ok & (np.abs(d) > 0.1 * size) & (np.abs(c) > 1e-6)
        return float(np.median(d[use] * z[use] / c[use])) if use.sum() >= 20 else float("nan")

    return ratio(du, x, w), ratio(dv, y, h)


class D4RTBackbone(Backbone):
    name = "d4rt"

    def __init__(self) -> None:
        self._model: Any = None
        self._size: tuple[int, int] | None = None

    def model(self) -> tuple[Any, tuple[int, int]]:
        if self._model is None:
            import torch

            root = open_d4rt_root()
            sys.path.insert(0, str(root))
            from infer_track_3d import _unwrap_state_dict
            from src.core import load_checkpoint, load_yaml_config
            from src.model import build_model

            ckpt = root / "checkpoints" / CHECKPOINT_NAME
            cfg = load_yaml_config(str(ckpt / "model.yaml"))
            model = build_model(cfg["model"]).eval()
            model.load_state_dict(_unwrap_state_dict(load_checkpoint(ckpt / "opend4rt.ckpt", map_location="cpu")),
                                  strict=False)
            size = cfg.get_path("model.input.image_size", [256, 256])
            self._model, self._size = model.to(torch.device("cuda")).eval(), (int(size[0]), int(size[1]))
        return self._model, self._size

    def reconstruct(self, clip: Clip) -> dict[str, np.ndarray]:
        from d4rt_agent.sam_orientany_d4rt_test.traj3d_geometry import robust_rigid

        model, size = self.model()
        from infer_track_3d import _infer_tracks, _resize_video

        frames_all, gf_all = clip.all_frames(MAX_GF_FRAMES)
        keep = pick_frames(gf_all, clip.answer_frames)
        frames, gf_index = frames_all[keep], gf_all[keep]
        n = len(keep)
        hw = model_hw(clip)
        video = _resize_video(frames, image_hw=size)

        uv_cell, (gh, gw) = dense_grid(hw)
        cells = len(uv_cell)
        queries = np.tile(uv_cell, (n, 1))
        source = np.repeat(np.arange(n, dtype=np.int64), cells)
        dense = _infer_tracks(model=model, video_model_rgb=video, query_uv_norm=queries, query_chunk_size=8192,
                              query_src_indices_global=source)
        ref = np.stack([dense["tracks_xyz_ref0"][f * cells:(f + 1) * cells, f] for f in range(n)])
        local = np.stack([dense["tracks_xyz_local"][f * cells:(f + 1) * cells, f] for f in range(n)])
        conf = np.stack([_sigmoid(dense["tracks_confidence"][f * cells:(f + 1) * cells, f]) *
                         _sigmoid(dense["tracks_visibility_logits"][f * cells:(f + 1) * cells, f]) for f in range(n)])
        del dense

        cam_R, cam_t = np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3))
        for f in range(1, n):
            fit = robust_rigid(local[f], ref[f], weights=np.minimum(conf[f], 1.0) ** 2, seed=f)
            if fit is None or fit.inliers.sum() < MIN_CELLS_FOR_FIT:
                cam_R[f], cam_t[f] = cam_R[f - 1], cam_t[f - 1]      # hold the last pose
                continue
            cam_R[f], cam_t[f] = fit.rotation, fit.translation

        K = np.zeros((n, 3, 3))
        h, w = hw
        fx_last = float(w)
        for f in range(n):
            fx, fy = focal_from_points(local[f], uv_cell, conf[f], hw)
            fx = fx if np.isfinite(fx) else fx_last
            fy = fy if np.isfinite(fy) else fx
            fx_last = fx
            K[f] = [[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]]

        grid_uv = query_grid(hw)
        track = _infer_tracks(model=model, video_model_rgb=video,
                              query_uv_norm=(grid_uv / np.array([w, h], dtype=np.float32)).astype(np.float32),
                              query_chunk_size=8192)
        grid_traj = np.transpose(track["tracks_xyz_ref0"], (1, 0, 2)).astype(np.float32)
        grid_conf = np.transpose(_sigmoid(track["tracks_confidence"]) *
                                 _sigmoid(track["tracks_visibility_logits"]), (1, 0)).astype(np.float32)
        return {
            "gf_index": gf_index, "model_hw": np.asarray(hw), "source_hw": np.asarray(clip.record["source_hw"]),
            "cam_R": cam_R, "cam_t": cam_t, "K": K, "points": ref.reshape(n, gh, gw, 3).astype(np.float16),
            "geometry_stride": np.int32(CELL_PX), "grid_uv": grid_uv, "grid_traj": grid_traj, "grid_conf": grid_conf,
        }

    def metric(self, clip: Clip, arrays: Mapping[str, np.ndarray]) -> dict[str, Any] | None:
        return {"metres_per_unit": 1.0, "iqr_over_median": None, "source": "d4rt-native",
                "note": "D4RT depth units taken as metres; measured at ~0.45x DA3-Nested's scale on dev clips"}
