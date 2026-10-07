"""GPU prep, part 2: one Point4D scene pass per clip, and its metric scale.

Runs in ``envs/point4d``::

    python -m dyn_dataset_exploration.decomposition.prep_point4d --stages scene metric

- ``scene``: Point4D over the GF frames, queried with a uniform grid on GF frame 0, with
  ``return_geometry=True``. Entity-agnostic on purpose: every entity's 3D path is later
  read off these depth clouds through its SAM3 masks, so all entities share one frame and
  one scale, entities that enter late are covered, and SAM3 and Point4D never wait on each
  other. Portrait clips run at (518, 294) instead of the landscape (294, 518): Point4D
  resizes without keeping the aspect ratio, which would squash them ~3x.
- ``metric``: DA3-Nested (metric) on the 16 sampled frames; metres per Point4D unit is the
  median ratio of its depth to Point4D's camera depth on the same pixels.

Writes ``clips/<slug>/point4d/scene.npz`` + ``scene.json`` (sanity numbers) and
``clips/<slug>/point4d/metric.json``. World = the camera of GF frame 0 (+X right, +Y down,
+Z forward), in Point4D's model-relative units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from dyn_dataset_exploration.dyn_bench_data import read_manifest, write_manifest
from dyn_dataset_exploration.decomposition.build_catalog import DEFAULT_OUT

LANDSCAPE_HW = (294, 518)
PORTRAIT_HW = (518, 294)
GRID_STEP_PX = 16              # frame-0 query grid spacing at model resolution
MAX_FRAMES = 400               # longer GFs are strided; the stride is recorded
DA3_NESTED = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_LONG_SIDE = 504


def load_frames(out: Path, clip: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The GF frames (T, H, W, 3) uint8 and the GF index of each."""

    import cv2

    length = clip["gf"]["length"]
    step = int(np.ceil(length / MAX_FRAMES))
    gf_index = np.arange(0, length, step)
    frames = []
    for index in gf_index:
        bgr = cv2.imread(str(out / "clips" / clip["clip_slug"] / "frames" / f"{index:06d}.jpg"))
        if bgr is None:
            raise FileNotFoundError(f"GF frame {index} of {clip['clip_slug']}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames), gf_index


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


def sanity(traj: np.ndarray, grid_uv: np.ndarray, geom: dict[str, Any],
           conf: np.ndarray | None = None) -> dict[str, Any]:
    """Consistency checks. Self-consistency is not ground truth; it catches broken runs.

    Reprojection is also reported over the more confident half of the grid: sky and far
    background points reproject badly without the scene being wrong where it matters.
    """

    K0 = geom["intrinsics"][0]
    local0 = (traj[0] - geom["cam_t"][0]) @ geom["cam_R"][0]
    uvw = local0 @ K0.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    reproj = np.linalg.norm(uv - grid_uv, axis=1)
    R = geom["cam_R"]
    orth = np.linalg.norm(R @ np.swapaxes(R, 1, 2) - np.eye(3), axis=(1, 2))
    cloud0 = geom["points"][0].reshape(-1, 3)
    cloud0 = cloud0[np.isfinite(cloud0).all(1)]
    scale = float(np.median(np.linalg.norm(cloud0 - np.median(cloud0, 0), axis=1)))
    step = np.linalg.norm(np.diff(geom["cam_t"], axis=0), axis=1)
    focal = geom["intrinsics"][:, 0, 0]
    confident = conf[0] > np.median(conf[0]) if conf is not None else np.ones(len(reproj), bool)
    return {
        "frame0_reprojection_px_confident_median": float(np.median(reproj[confident])),
        "frame0_reprojection_px_confident_p95": float(np.percentile(reproj[confident], 95)),
        "frame0_reprojection_px_median": float(np.median(reproj)),
        "frame0_reprojection_px_p95": float(np.percentile(reproj, 95)),
        "rotation_orthogonality_max": float(orth.max()),
        "scene_scale_units": scale,
        "camera_step_over_scale_median": float(np.median(step) / max(scale, 1e-9)),
        "camera_step_over_scale_max": float(step.max() / max(scale, 1e-9)) if len(step) else 0.0,
        "focal_px_min_max": [float(focal.min()), float(focal.max())],
        "finite_tracks": float(np.isfinite(traj).all(-1).mean()),
    }


def stage_scene(out: Path, slugs: Sequence[str]) -> None:
    import torch
    from point4d.inference import LongTrackConfig, track_long_video
    from point4d.loader import load_model

    model = load_model(device="cuda")
    for slug in slugs:
        clip = json.loads((out / "clips" / slug / "clip.json").read_text())
        target = out / "clips" / slug / "point4d" / "scene.npz"
        if target.exists():
            continue
        started = time.time()
        frames, gf_index = load_frames(out, clip)
        hw = PORTRAIT_HW if clip["portrait"] else LANDSCAPE_HW
        cfg = LongTrackConfig(chunk_size=48, overlap=8, image_hw=hw, geometry_stride=4)
        grid_uv = query_grid(hw)
        with torch.inference_mode():
            traj, conf, geom = track_long_video(model, frames, grid_uv, cfg, return_geometry=True)
        stride = cfg.chunk_size - cfg.overlap
        chunk_starts = np.arange(0, max(len(frames) - cfg.overlap, 1), stride)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target, gf_index=gf_index, model_hw=np.asarray(hw), source_hw=np.asarray(clip["source_hw"]),
            cam_R=geom["cam_R"], cam_t=geom["cam_t"], K=geom["intrinsics"],
            points=geom["points"].astype(np.float16), geometry_stride=np.int32(cfg.geometry_stride),
            grid_uv=grid_uv, grid_traj=traj.astype(np.float32), grid_conf=conf.astype(np.float32),
            chunk_starts=chunk_starts, chunk_size=np.int32(cfg.chunk_size), overlap=np.int32(cfg.overlap),
        )
        report = {"clip_slug": slug, "frames": int(len(frames)), "model_hw": list(hw),
                  "portrait": clip["portrait"], "seconds": round(time.time() - started, 1),
                  "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  **sanity(traj, grid_uv, geom, conf)}
        write_manifest(report, target.with_suffix(".json"))
        print(f"scene {slug}: {json.dumps(report)}", flush=True)
        torch.cuda.empty_cache()


def stage_sanity(out: Path, slugs: Sequence[str]) -> None:
    """Recompute ``scene.json``'s checks from ``scene.npz`` (CPU)."""

    for slug in slugs:
        target = out / "clips" / slug / "point4d" / "scene.npz"
        if not target.exists():
            continue
        report = json.loads(target.with_suffix(".json").read_text())
        with np.load(target) as z:
            geom = {"intrinsics": z["K"], "cam_R": z["cam_R"], "cam_t": z["cam_t"],
                    "points": z["points"].astype(np.float32)}
            report.update(sanity(z["grid_traj"], z["grid_uv"], geom, z["grid_conf"]))
        write_manifest(report, target.with_suffix(".json"))
        print(f"sanity {slug}: confident reprojection "
              f"{report['frame0_reprojection_px_confident_median']:.1f}/"
              f"{report['frame0_reprojection_px_confident_p95']:.1f} px", flush=True)


def load_da3_nested(device: str = "cuda") -> Any:
    """DA3-Nested through the DA3 code vendored in Point4D (the official package pulls
    xformers/open3d; the network definition is all that is needed)."""

    import torch
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from point4d.backbone.depth_anything_3.cfg import create_object

    root = Path(snapshot_download(DA3_NESTED, local_files_only=True))
    text = (root / "config.json").read_text().replace(
        '"depth_anything_3.', '"point4d.backbone.depth_anything_3.')
    config = json.loads(text)["config"]
    # The Gaussian-splatting head is not vendored and not needed for depth.
    for key in ("gs_head", "gs_adapter"):
        config["anyview"].pop(key, None)
    # Built on the target device: initialising 1.4B parameters on the CPU takes minutes and the
    # checkpoint overwrites every one of them anyway.
    with torch.device(device):
        model = create_object(OmegaConf.create(config))
    state = {k[len("model."):]: v for k, v in load_file(str(root / "model.safetensors")).items()
             if k.startswith("model.") and ".gs_" not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[da3-nested] {len(missing)} missing keys, e.g. {missing[:3]}", flush=True)
    if unexpected:
        print(f"[da3-nested] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}", flush=True)
    return model.eval().to(device)


def da3_input(frames: np.ndarray) -> "torch.Tensor":
    """(1, N, 3, H, W): long side 504, sides multiples of 14, ImageNet-normalised."""

    import cv2
    import torch

    h, w = frames.shape[1:3]
    scale = DA3_LONG_SIDE / max(h, w)
    nh, nw = max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14)
    resized = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames])
    x = torch.from_numpy(resized).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def stage_metric(out: Path, slugs: Sequence[str], backend: str = "point4d") -> None:
    import cv2
    import torch

    model = load_da3_nested()
    for slug in slugs:
        clip = json.loads((out / "clips" / slug / "clip.json").read_text())
        scene_path = out / "clips" / slug / backend / "scene.npz"
        target = out / "clips" / slug / backend / "metric.json"
        if target.exists() or not scene_path.exists():
            continue
        with np.load(scene_path) as scene:
            gf_index = scene["gf_index"]
            depth = camera_depth(scene["points"].astype(np.float32), scene["cam_R"], scene["cam_t"])
        positions = [int(np.argmin(np.abs(gf_index - i))) for i in clip["sampled16"]]
        frames_all, gf_all = load_frames(out, clip)          # the scene may hold only a subset of the GF frames
        frames = frames_all[[int(np.argmin(np.abs(gf_all - gf_index[p]))) for p in positions]]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(da3_input(frames).to("cuda"))
        metric = output.depth[0].float().cpu().numpy()          # (N, h, w) metres
        ratios, per_frame = [], []
        for n, position in enumerate(positions):
            p4d = depth[position]
            m = cv2.resize(metric[n], (p4d.shape[1], p4d.shape[0]), interpolation=cv2.INTER_AREA)
            ok = np.isfinite(p4d) & (p4d > 1e-6) & np.isfinite(m) & (m > 0)
            ok &= (m < np.percentile(m[ok], 95)) & (p4d < np.percentile(p4d[ok], 95))  # sky, far
            r = m[ok] / p4d[ok]
            per_frame.append(float(np.median(r)))
            ratios.append(r)
        per_frame_arr = np.asarray(per_frame)
        q25, q50, q75 = np.percentile(per_frame_arr, [25, 50, 75])
        report = {
            "clip_slug": slug, "model": DA3_NESTED,
            "metres_per_unit": float(q50),
            "iqr_over_median": float((q75 - q25) / max(q50, 1e-9)),
            "per_frame": dict(zip(map(str, clip["sampled16"]), map(float, per_frame_arr))),
            "da3_scale_factor": float(getattr(output, "scale_factor", float("nan"))),
            "backend": backend,
            "method": f"median over the 16 sampled frames of median(DA3-Nested depth / {backend} "
                      "camera depth); pixels above either map's 95th percentile excluded",
        }
        write_manifest(report, target)
        print(f"metric {slug}: {report['metres_per_unit']:.4f} m/unit, IQR/median "
              f"{report['iqr_over_median']:.3f}", flush=True)
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stages", nargs="+", default=["scene", "metric"],
                        choices=["scene", "metric", "sanity"])
    parser.add_argument("--only", nargs="*", default=None, help="clip slugs")
    parser.add_argument("--backend", default="point4d", choices=["point4d", "d4rt"],
                        help="which scene the metric stage scales (d4rt: after prep_d4rt_scene)")
    parser.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th clip")
    args = parser.parse_args(argv)
    slugs = sorted({e["clip_slug"] for e in read_manifest(args.out / "catalog.json")["questions"]})
    if args.only:
        slugs = [s for s in slugs if s in set(args.only)]
    index, count = (int(v) for v in args.shard.split("/"))
    slugs = slugs[index::count]
    for stage in args.stages:
        started = time.time()
        if stage == "metric":
            stage_metric(args.out, slugs, args.backend)
        else:
            {"scene": stage_scene, "sanity": stage_sanity}[stage](args.out, slugs)
        print(f"stage {stage} done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
