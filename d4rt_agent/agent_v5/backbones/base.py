"""The backbone interface, the registry behind ``--backbone``, and the per-clip scene cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from d4rt_agent.agent_v5.clips import Clip
from d4rt_agent.agent_v5.scene import Scene, scene_from_arrays

LANDSCAPE_HW = (294, 518)
PORTRAIT_HW = (518, 294)
GRID_STEP_PX = 16              # frame-0 query grid spacing at model resolution
SCENE_KEYS = ("gf_index", "model_hw", "source_hw", "cam_R", "cam_t", "K", "points", "geometry_stride",
              "grid_uv", "grid_traj", "grid_conf")


class Backbone:
    """Turns a clip into scene arrays and (optionally) a metric scale. Models load on first use."""

    name = "base"

    def reconstruct(self, clip: Clip) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def metric(self, clip: Clip, arrays: Mapping[str, np.ndarray]) -> dict[str, Any] | None:
        """``{"metres_per_unit", "iqr_over_median", "source", ...}`` or None."""

        raise NotImplementedError


def model_hw(clip: Clip) -> tuple[int, int]:
    """Point4D resizes without keeping the aspect ratio: portrait clips run at (518, 294)."""

    return PORTRAIT_HW if clip.record.get("portrait") else LANDSCAPE_HW


def query_grid(hw: tuple[int, int], step: int = GRID_STEP_PX) -> np.ndarray:
    h, w = hw
    ys = np.arange(step / 2, h, step, dtype=np.float32)
    xs = np.arange(step / 2, w, step, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def camera_depth(points: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """Per-frame depth along each frame's own optical axis, from world-frame clouds."""

    local = np.einsum("thwi,tij->thwj", points - cam_t[:, None, None, :], cam_R)
    return local[..., 2]


def _point4d() -> Backbone:
    from d4rt_agent.agent_v5.backbones.point4d import Point4DBackbone
    return Point4DBackbone()


def _d4rt() -> Backbone:
    from d4rt_agent.agent_v5.backbones.d4rt import D4RTBackbone
    return D4RTBackbone()


BACKBONES: dict[str, Callable[[], Backbone]] = {"point4d": _point4d, "d4rt": _d4rt}


def get_backbone(name: str) -> Backbone:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name]()


class SceneCache:
    """``<cache>/clips/<slug>/<backbone>/{scene.npz, metric.json, scene.json}``: a clip is
    reconstructed once per backbone, and a resumed run reads it back."""

    def __init__(self, backbone: Backbone, cache: Path) -> None:
        self.backbone = backbone
        self.cache = Path(cache)
        self._last: tuple[str, Scene] | None = None

    def directory(self, clip: Clip) -> Path:
        return self.cache / "clips" / clip.slug / self.backbone.name

    def scene(self, clip: Clip) -> Scene:
        if self._last is not None and self._last[0] == clip.slug:
            return self._last[1]
        base = self.directory(clip)
        target = base / "scene.npz"
        if target.exists():
            with np.load(target) as z:
                arrays = {k: z[k] for k in z.files}
        else:
            started = time.time()
            arrays = self.backbone.reconstruct(clip)
            base.mkdir(parents=True, exist_ok=True)
            tmp = base / "scene.tmp.npz"
            np.savez_compressed(tmp, **{k: v for k, v in arrays.items()})
            os.replace(tmp, target)
            _write_json(base / "scene.json", {"backbone": self.backbone.name, "frames": int(len(arrays["gf_index"])),
                                              "seconds": round(time.time() - started, 1)})
        metric_path = base / "metric.json"
        if metric_path.exists():
            metric = json.loads(metric_path.read_text())
        else:
            metric = self.backbone.metric(clip, arrays) or {"metres_per_unit": None, "source": None}
            _write_json(metric_path, metric)
        scene = scene_from_arrays(clip.slug, clip.record, arrays, metric)
        self._last = (clip.slug, scene)
        return scene


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
