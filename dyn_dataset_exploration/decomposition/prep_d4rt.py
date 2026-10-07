"""GPU prep: a second metric scale per clip, from D4RT's own (near-metric) depth.

Point4D's units are model-relative; ``prep_point4d`` turns them into metres with DA3-Nested. D4RT
(``third-party-tools/Open-D4RT``) predicts in a scale that experiments found much closer to metric,
so here it is run on the 16 sampled frames of each clip, its per-frame depth is read at a grid of
pixels, and compared with Point4D's camera depth at the same pixels::

    metres_per_unit = median over frames of median(D4RT depth / Point4D depth)

Written to ``clips/<slug>/point4d/d4rt_metric.json`` (same fields as ``metric.json``); the scene
loader picks it up and ``scene4d.SCALE_SOURCE`` (calibrated on dev) chooses between the two.

    PYTHONPATH=<repo>:<repo>/third-party-tools/Open-D4RT:<Point4D> python -m \\
        dyn_dataset_exploration.decomposition.prep_d4rt --out <catalog dir> [--shard i/n]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from dyn_dataset_exploration.dyn_bench_data import write_manifest
from dyn_dataset_exploration.decomposition.prep_point4d import camera_depth, load_frames

OPEN_D4RT = Path(__file__).resolve().parents[2] / "third-party-tools" / "Open-D4RT"
CKPT = OPEN_D4RT / "checkpoints" / "OpenD4RT_32CLIP_9Dataset_NoAUG"
GRID = 16


def load_model(device: str = "cuda"):
    sys.path.insert(0, str(OPEN_D4RT))
    import torch
    from infer_track_3d import _unwrap_state_dict
    from src.core import load_checkpoint, load_yaml_config
    from src.model import build_model

    cfg = load_yaml_config(str(CKPT / "model.yaml"))
    model = build_model(cfg["model"]).eval()
    state = _unwrap_state_dict(load_checkpoint(CKPT / "opend4rt.ckpt", map_location="cpu"))
    model.load_state_dict(state, strict=False)
    size = cfg.get_path("model.input.image_size", [256, 256])
    return model.to(torch.device(device)).eval(), (int(size[0]), int(size[1]))


def d4rt_depth(model, size, frames: np.ndarray) -> np.ndarray:
    """(N, GRID, GRID) depth along each frame's own optical axis, at grid cell centres."""

    from infer_track_3d import _infer_tracks, _resize_video

    n = len(frames)
    video = _resize_video(frames, image_hw=size)
    cells = (np.arange(GRID, dtype=np.float32) + 0.5) / GRID
    gx, gy = np.meshgrid(cells, cells)
    uv = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    queries = np.tile(uv, (n, 1))
    source = np.repeat(np.arange(n, dtype=np.int64), len(uv))
    out = _infer_tracks(model=model, video_model_rgb=video, query_uv_norm=queries, query_chunk_size=8192,
                        query_src_indices_global=source)
    xyz = out["tracks_xyz_local"] if "tracks_xyz_local" in out else out["xyz_local"]
    depth = np.stack([xyz[i * len(uv):(i + 1) * len(uv), i, 2] for i in range(n)])
    return depth.reshape(n, GRID, GRID)


def stage(out: Path, slugs: Sequence[str]) -> None:
    model, size = load_model()
    for slug in slugs:
        clip = json.loads((out / "clips" / slug / "clip.json").read_text())
        scene_path = out / "clips" / slug / "point4d" / "scene.npz"
        target = out / "clips" / slug / "point4d" / "d4rt_metric.json"
        if target.exists() or not scene_path.exists():
            continue
        with np.load(scene_path) as scene:
            gf_index = scene["gf_index"]
            p4d = camera_depth(scene["points"].astype(np.float32), scene["cam_R"], scene["cam_t"])
        positions = [int(np.argmin(np.abs(gf_index - i))) for i in clip["sampled16"]]
        frames, _ = load_frames(out, clip)
        depth = d4rt_depth(model, size, frames[positions])
        per_frame = []
        for n, position in enumerate(positions):
            ref = p4d[position]
            h, w = ref.shape
            ys = np.clip(((np.arange(GRID) + 0.5) / GRID * h).astype(int), 0, h - 1)
            xs = np.clip(((np.arange(GRID) + 0.5) / GRID * w).astype(int), 0, w - 1)
            r = ref[np.ix_(ys, xs)]
            m = depth[n]
            ok = np.isfinite(r) & (r > 1e-6) & np.isfinite(m) & (m > 0)
            if ok.sum() < 20:
                continue
            ok &= (m < np.percentile(m[ok], 95)) & (r < np.percentile(r[ok], 95))
            per_frame.append(float(np.median(m[ok] / r[ok])))
        if not per_frame:
            continue
        arr = np.asarray(per_frame)
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        write_manifest({
            "clip_slug": slug, "model": "OpenD4RT_32CLIP_9Dataset_NoAUG", "metres_per_unit": float(q50),
            "iqr_over_median": float((q75 - q25) / max(q50, 1e-9)), "frames_used": len(per_frame),
            "method": "median over the 16 sampled frames of median(D4RT depth / Point4D camera depth) on a "
                      f"{GRID}x{GRID} grid; pixels above either map's 95th percentile excluded",
        }, target)
        print(f"d4rt {slug}: {q50:.3f} m/unit (spread {(q75 - q25) / max(q50, 1e-9):.2f})", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    index, count = (int(v) for v in args.shard.split("/"))
    slugs = sorted(p.parent.name for p in (args.out / "clips").glob("*/clip.json"))
    if args.only:
        slugs = [s for s in slugs if s in set(args.only)]
    stage(args.out, slugs[index::count])


if __name__ == "__main__":
    main()
