"""Synthetic scenes with known answers, for testing S3/S4 without any model.

A static wall at ``z = wall_z`` in world coordinates, a camera on a prescribed path, and
optional entities that follow a prescribed path. ``points`` holds what each frame's depth
map would give (the wall through every cell, the entity's world position through the cells
it occupies) and the ``Grounding`` holds matching masks, so ``lift_instances`` can be
exercised exactly as it is on real data.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from d4rt_agent.sam_orientany_d4rt_test import traj3d_buckets
from dyn_dataset_exploration.decomposition.scene4d import Grounding, Scene

MODEL_HW = (48, 80)
STRIDE = 4
MASK_HW = (96, 160)
FOCAL = 60.0


def rot_y(deg: float) -> np.ndarray:
    """Turning right by ``deg`` (camera forward moves toward +X)."""

    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_x(deg: float) -> np.ndarray:
    """Tilting up by ``deg`` (camera forward moves toward -Y, since +Y is down)."""

    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rays(cam_R: np.ndarray, gh: int, gw: int) -> np.ndarray:
    h, w = MODEL_HW
    ys = np.arange(gh) * STRIDE
    xs = np.arange(gw) * STRIDE
    u, v = np.meshgrid(xs, ys)
    d = np.stack([(u - w / 2) / FOCAL, (v - h / 2) / FOCAL, np.ones_like(u, dtype=float)], axis=-1)
    return d @ cam_R.T


def build_scene(
    frames: int = 40,
    fps: float | None = 6.0,
    *,
    camera_t: Callable[[int], Sequence[float]] = lambda t: (0.0, 0.0, 0.0),
    camera_R: Callable[[int], np.ndarray] = lambda t: np.eye(3),
    entities: Sequence[tuple] = (),
    focal: Callable[[int], float] | None = None,
    wall_z: float = 6.0,
    metres_per_unit: float | None = None,
    rng: np.random.Generator | None = None,
    dataset: str = "davis",
) -> tuple[Scene, dict[str, Grounding]]:
    """A scene and one Grounding per entity phrase (its single instance is index 0).

    An entity is ``(phrase, path)`` or ``(phrase, path, radius)``; the path returns its world
    position (None where it is absent). With a radius, the mask covers a disc of that world
    size, so it grows as the entity nears the camera. ``focal(t)`` varies the focal length.
    """

    gh, gw = len(range(0, MODEL_HW[0], STRIDE)), len(range(0, MODEL_HW[1], STRIDE))
    cam_R = np.stack([camera_R(t) for t in range(frames)])
    cam_t = np.array([camera_t(t) for t in range(frames)], dtype=float)
    fl = np.array([FOCAL if focal is None else focal(t) for t in range(frames)])
    K = np.stack([np.array([[f, 0, MODEL_HW[1] / 2], [0, f, MODEL_HW[0] / 2], [0, 0, 1.0]]) for f in fl])
    points = np.zeros((frames, gh, gw, 3), dtype=np.float32)
    for t in range(frames):
        d = _rays(cam_R[t], gh, gw)
        s = (wall_z - cam_t[t, 2]) / np.where(np.abs(d[..., 2]) > 1e-6, d[..., 2], 1e-6)
        points[t] = cam_t[t] + s[..., None] * d

    mh, mw = MASK_HW
    groundings: dict[str, Grounding] = {}
    for spec in entities:
        phrase, path = spec[0], spec[1]
        radius = spec[2] if len(spec) > 2 else None
        masks = np.zeros((1, frames, mh, mw), dtype=bool)
        probs = np.full((1, frames), np.nan, dtype=np.float32)
        boxes = np.full((1, frames, 4), np.nan, dtype=np.float32)
        area = np.zeros((1, frames), dtype=np.float32)
        border = np.zeros((1, frames), dtype=bool)
        for t in range(frames):
            pos = path(t)
            if pos is None:
                continue
            pos = np.asarray(pos, dtype=float)
            local = (pos - cam_t[t]) @ cam_R[t]
            if local[2] <= 0.1:
                continue
            u = MODEL_HW[1] / 2 + FOCAL * local[0] / local[2]
            v = MODEL_HW[0] / 2 + FOCAL * local[1] / local[2]
            ci, cj = int(v // STRIDE), int(u // STRIDE)
            half = 2 if radius is None else max(1, int(round(radius * FOCAL / local[2] / STRIDE)))
            for i in range(max(ci - half, 0), min(ci + half + 1, gh)):
                for j in range(max(cj - half, 0), min(cj + half + 1, gw)):
                    points[t, i, j] = pos + (rng.normal(0, 0.004, 3) if rng is not None else 0.0)
                    masks[0, t, i * mh // gh:(i + 1) * mh // gh, j * mw // gw:(j + 1) * mw // gw] = True
            if masks[0, t].any():
                ys, xs = np.nonzero(masks[0, t])
                probs[0, t] = 0.9
                boxes[0, t] = (xs.min() / mw, ys.min() / mh, (xs.max() + 1) / mw, (ys.max() + 1) / mh)
                area[0, t] = masks[0, t].mean()
                border[0, t] = bool(masks[0, t, 0].any() or masks[0, t, -1].any()
                                    or masks[0, t, :, 0].any() or masks[0, t, :, -1].any())
        groundings[phrase] = Grounding(phrase, np.array([0]), masks, probs, boxes, area, border)

    up = np.array([0.0, -1.0, 0.0])
    scene = Scene(
        slug="synthetic", clip={"dataset": dataset}, gf_index=np.arange(frames), cam_R=cam_R, cam_t=cam_t, K=K,
        points=points, model_hw=MODEL_HW, source_hw=MODEL_HW, stride=STRIDE,
        grid_traj=np.zeros((frames, 1, 3)), grid_conf=np.ones((frames, 1)), up=up, up_fraction=1.0,
        up_source="plane", level=traj3d_buckets.level_basis(up), scale=1.0, fps=fps,
        metres_per_unit=metres_per_unit, metric_iqr=0.05 if metres_per_unit else None,
    )
    return scene, groundings
