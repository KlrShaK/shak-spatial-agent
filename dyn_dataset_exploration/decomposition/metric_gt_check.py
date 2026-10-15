"""Does Point4D become metric with a metric DepthAnything? A check against metric ground truth.

Point4D's geometry (depth clouds, camera translation, tracks) is in the units of its DA3-giant depth
head. The 3D-query decoder never sees those units: queries are divided by the chunk's mean scene
distance and its outputs multiplied back (``point4d.inference.geometry.scene_scale``), so scaling the
depth scales every output by the same factor. A metric DepthAnything cannot replace the backbone's
*weights* (DA3-Metric-Large is a ViT-L; Point4D's trunk is a fine-tuned ViT-g with time tokens the
decoder was trained on), but it can replace its *scale*: ``point4d_inject`` puts each chunk's depth and
camera translation on DA3-Nested's metric depth inside the backbone output, before the decoder,
``scene_scale`` and the chunk-to-chunk Sim(3) see it.

Ground truth (official releases, under ``DATA``): TUM RGB-D fr1/desk and fr3/walking_xyz, Bonn RGB-D
Dynamic person_tracking (Kinect depth, png / 5000), KITTI raw 2011_09_26 drives 0002 and 0005 (Velodyne
projected into image_02 as in monodepth2's Eigen protocol, Garg crop, <= 80 m). Each sequence is a
window of <= 96 frames; the 16 scored frames are ``floor(linspace(0, T-1, 16))`` as in Dyn-Bench, and
every method is read at the same <= 4000 GT pixels per scored frame.

    python -m dyn_dataset_exploration.decomposition.metric_gt_check prep           # CPU: windows
    python -m dyn_dataset_exploration.decomposition.metric_gt_check run report     # GPU (PART=metric_gt)
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Sequence

import numpy as np

DATA = Path("/cluster/work/igp_psr/spanwar/datasets/metric_depth_gt")
OUT = Path(__file__).resolve().parents[1] / "outputs" / "metric_gt_check"
KITTI_CALIB = DATA / "kitti_calib_2011_09_26" / "2011_09_26"
WINDOW = 96
N_EVAL = 16
PIXELS_PER_FRAME = 4000
P4D_LONG_SIDE = 518            # aspect kept: 4:3 -> (392, 518), KITTI -> (154, 518)
GEOMETRY_STRIDE = 2
INJECT_EVERY = 6               # DA3-Nested reads every 6th frame of a 48-frame chunk
SEQUENCES = {                  # name: (format, directory under DATA, frame stride, max GT depth in m)
    "tum_fr1_desk": ("tum", "tum_fr1_desk/rgbd_dataset_freiburg1_desk", 2, 10.0),
    "tum_fr3_walking_xyz": ("tum", "tum_fr3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz", 2, 10.0),
    "bonn_person_tracking": ("tum", "bonn_person_tracking/rgbd_bonn_person_tracking", 2, 10.0),
    "kitti_0002": ("kitti", "kitti_0002/2011_09_26/2011_09_26_drive_0002_sync", 1, 80.0),
    "kitti_0005": ("kitti", "kitti_0005/2011_09_26/2011_09_26_drive_0005_sync", 1, 80.0),
}
TUM_FX = {"tum_fr1_desk": 517.3, "tum_fr3_walking_xyz": 535.4, "bonn_person_tracking": 542.8}
METHODS = ("point4d_raw", "point4d_x_da3", "point4d_inject", "da3_nested", "da3_metric_large",
           "d4rt_raw", "point4d_x_d4rt")


# --------------------------------------------------------------------------- #
# prep (CPU): frames, GT depth and the scored pixels of each window
# --------------------------------------------------------------------------- #

def _index(path: Path) -> tuple[np.ndarray, list[str]]:
    rows = [line.split() for line in path.read_text().splitlines() if line and not line.startswith("#")]
    return np.array([float(r[0]) for r in rows]), [r[1] for r in rows]


def tum_pairs(root: Path, stride: int) -> list[tuple[Path, Path]]:
    """Each RGB frame with the depth image nearest in time (<= 20 ms), then every ``stride``-th."""

    t_rgb, rgb = _index(root / "rgb.txt")
    t_depth, depth = _index(root / "depth.txt")
    pairs = []
    for t, f in zip(t_rgb, rgb):
        j = int(np.argmin(np.abs(t_depth - t)))
        if abs(t_depth[j] - t) < 0.02:
            pairs.append((root / f, root / depth[j]))
    return pairs[::stride][:WINDOW]


def _calib(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        try:
            out[key] = np.array([float(v) for v in value.split()])
        except ValueError:
            pass
    return out


def kitti_depth(velo_path: Path, hw: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Velodyne points in the rectified left colour camera (monodepth2's ``generate_depth_map``)."""

    cam = _calib(KITTI_CALIB / "calib_cam_to_cam.txt")
    velo = _calib(KITTI_CALIB / "calib_velo_to_cam.txt")
    velo2cam = np.eye(4)
    velo2cam[:3, :3], velo2cam[:3, 3] = velo["R"].reshape(3, 3), velo["T"]
    rect = np.eye(4)
    rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
    P = cam["P_rect_02"].reshape(3, 4)
    pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4).astype(np.float64)
    pts = pts[pts[:, 0] >= 0]
    pts[:, 3] = 1.0
    im = (P @ rect @ velo2cam @ pts.T).T
    z = im[:, 2]
    u, v = np.round(im[:, 0] / z) - 1, np.round(im[:, 1] / z) - 1
    ok = (u >= 0) & (v >= 0) & (u < hw[1]) & (v < hw[0]) & (z > 0)
    u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
    depth = np.zeros(hw, np.float32)
    order = np.argsort(-z)                      # the nearest return is written last and wins
    depth[v[order], u[order]] = z[order]
    return depth, float(P[0, 0])


def valid_gt(gt: np.ndarray, fmt: str, max_depth: float) -> np.ndarray:
    ok = (gt > (0.1 if fmt == "tum" else 1e-3)) & (gt < max_depth)
    if fmt == "kitti":                          # Garg crop
        h, w = gt.shape
        crop = np.zeros_like(ok)
        crop[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
        ok &= crop
    return ok


def prep(names: Sequence[str]) -> None:
    import cv2

    rng = np.random.default_rng(0)
    for name in names:
        fmt, rel, stride, max_depth = SEQUENCES[name]
        root = DATA / rel
        if fmt == "tum":
            pairs = tum_pairs(root, stride)
            rgb_paths = [p for p, _ in pairs]
        else:
            rgb_paths = sorted((root / "image_02" / "data").glob("*.png"))[::stride][:WINDOW]
        frames = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in rgb_paths])
        pos = np.floor(np.linspace(0, len(frames) - 1, N_EVAL)).astype(int)
        gts, fx = [], TUM_FX.get(name, float("nan"))
        for p in pos:
            if fmt == "tum":
                gts.append(cv2.imread(str(pairs[p][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0)
            else:
                depth, fx = kitti_depth(root / "velodyne_points" / "data" / f"{rgb_paths[p].stem}.bin",
                                        frames.shape[1:3])
                gts.append(depth)
        frame_of, vs, us = [], [], []
        for n, gt in enumerate(gts):
            v, u = np.nonzero(valid_gt(gt, fmt, max_depth))
            pick = rng.choice(len(v), size=min(PIXELS_PER_FRAME, len(v)), replace=False)
            frame_of.append(np.full(len(pick), n))
            vs.append(v[pick])
            us.append(u[pick])
        frame_of, v, u = (np.concatenate(a) for a in (frame_of, vs, us))
        values = np.stack(gts)[frame_of, v, u]
        target = DATA / "windows" / f"{name}.npz"
        target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target, frames=frames, eval_pos=pos, pix_frame=frame_of, pix_v=v, pix_u=u,
                            gt=values, fx_gt=np.float32(fx))
        print(f"prep {name}: {len(frames)} frames {frames.shape[1:3]}, {len(v)} scored pixels, GT depth "
              f"p5/p50/p95 {np.percentile(values, [5, 50, 95]).round(2).tolist()} m", flush=True)


# --------------------------------------------------------------------------- #
# run (GPU)
# --------------------------------------------------------------------------- #

def model_hw(src_hw: tuple[int, int], long_side: int = P4D_LONG_SIDE) -> tuple[int, int]:
    h, w = src_hw
    s = long_side / max(h, w)
    return max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)


def at_pixels(maps: np.ndarray, frame: np.ndarray, v: np.ndarray, u: np.ndarray, src_hw: tuple[int, int],
              stride: int = 1, image_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Nearest-neighbour read of (N, h, w) maps at source pixels. ``maps`` may be the ``stride``
    subsampled grid of an ``image_hw`` image (Point4D's geometry)."""

    H, W = src_hw
    mh, mw = image_hw or maps.shape[1:]
    i = np.clip(np.round(((v + 0.5) * mh / H - 0.5) / stride), 0, maps.shape[1] - 1).astype(int)
    j = np.clip(np.round(((u + 0.5) * mw / W - 0.5) / stride), 0, maps.shape[2] - 1).astype(int)
    return maps[frame, i, j]


def median_ratio(num: np.ndarray, den: np.ndarray) -> float:
    """median(num / den) where both are positive and below their 95th percentiles (sky, far): the rule
    of ``prep_point4d.stage_metric`` and ``prep_d4rt``."""

    ok = np.isfinite(num) & (num > 0) & np.isfinite(den) & (den > 1e-6)
    if ok.sum() < 20:
        return float("nan")
    ok &= (num < np.percentile(num[ok], 95)) & (den < np.percentile(den[ok], 95))
    return float(np.median(num[ok] / den[ok]))


def point4d_depth(model, frames: np.ndarray, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame camera depth (T, h, w) of a scene pass with the pipeline's chunking, and focal (px at hw)."""

    import torch
    from point4d.inference import LongTrackConfig, track_long_video
    from dyn_dataset_exploration.decomposition.prep_point4d import camera_depth, query_grid

    cfg = LongTrackConfig(chunk_size=48, overlap=8, image_hw=hw, geometry_stride=GEOMETRY_STRIDE)
    with torch.inference_mode():
        _, _, geom = track_long_video(model, frames, query_grid(hw), cfg, return_geometry=True)
    return camera_depth(geom["points"], geom["cam_R"], geom["cam_t"]), geom["intrinsics"][:, 0, 0]


@contextmanager
def metric_scale_injected(model, da3, log: list[float]):
    """Point4D whose depth head is put on DA3-Nested's metric scale chunk by chunk (``log`` gets each
    chunk's factor). Depth and camera translation are scaled together, so the chunk stays consistent."""

    import torch

    original = model.forward_backbone
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)

    def forward_backbone(images):
        cache = original(images)
        pred = cache["predictions"]
        idx = list(range(0, cache["S"], INJECT_EVERY))
        metric = da3((cache["images"][:, idx].float() - mean) / std).depth[0].float().cpu().numpy()
        rel = pred["depth"][0, idx].float().cpu().numpy()
        s = float(np.nanmedian([median_ratio(m, r) for m, r in zip(metric, rel)]))
        pred["depth"] = pred["depth"] * s
        pred["extrinsics"] = pred["extrinsics"].clone()
        pred["extrinsics"][..., :3, 3] *= s
        log.append(s)
        return cache

    model.forward_backbone = forward_backbone
    try:
        yield
    finally:
        del model.forward_backbone


def da3_depths(da3, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DA3-Nested metric depth, its DA3-Metric-Large branch alone (metres via DA3's focal, as inside
    Nested), and DA3's focal (px at its input size), on frames run jointly at long side 504."""

    import torch
    from point4d.backbone.depth_anything_3.utils.alignment import apply_metric_scaling
    from dyn_dataset_exploration.decomposition.prep_point4d import da3_input

    x = da3_input(frames).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = da3(x)
        mono = apply_metric_scaling(da3.da3_metric(x).depth.float(), out.intrinsics.float())
    return (out.depth[0].float().cpu().numpy(), mono[0].cpu().numpy(),
            out.intrinsics[0, :, 0, 0].float().cpu().numpy())


def d4rt_depths(model, size, frames: np.ndarray, pos: np.ndarray, pix_frame: np.ndarray, v: np.ndarray,
                u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D4RT depth (z in the frame's own camera) at the scored pixels and on prep_d4rt's GRID x GRID cells,
    from one 32-frame window (``prep_d4rt_scene.pick_frames``) queried with t_src = t_tgt = the frame."""

    from infer_track_3d import _infer_tracks, _resize_video
    from dyn_dataset_exploration.decomposition.prep_d4rt import GRID
    from dyn_dataset_exploration.decomposition.prep_d4rt_scene import pick_frames

    keep = pick_frames(np.arange(len(frames)), pos)
    local = np.searchsorted(keep, pos)
    H, W = frames.shape[1:3]
    cells = (np.arange(GRID, dtype=np.float32) + 0.5) / GRID
    gx, gy = np.meshgrid(cells, cells)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    uv = np.concatenate([np.stack([(u + 0.5) / W, (v + 0.5) / H], axis=-1), np.tile(grid, (len(pos), 1))])
    src = np.concatenate([local[pix_frame], np.repeat(local, len(grid))]).astype(np.int64)
    out = _infer_tracks(model=model, video_model_rgb=_resize_video(frames[keep], image_hw=size),
                        query_uv_norm=uv.astype(np.float32), query_chunk_size=8192,
                        query_src_indices_global=src)
    z = out["tracks_xyz_local"][np.arange(len(uv)), src, 2]
    return z[:len(u)], z[len(u):].reshape(len(pos), GRID, GRID)


def run(names: Sequence[str]) -> None:
    import gc
    import cv2
    import torch
    from point4d.loader import load_model
    from dyn_dataset_exploration.decomposition.prep_d4rt import GRID, load_model as load_d4rt
    from dyn_dataset_exploration.decomposition.prep_point4d import load_da3_nested

    windows = {name: dict(np.load(DATA / "windows" / f"{name}.npz")) for name in names}
    p4d, da3 = load_model(device="cuda"), load_da3_nested()
    found: dict[str, dict] = {}
    for name, w in windows.items():
        frames, pos = w["frames"], w["eval_pos"]
        src_hw, hw = frames.shape[1:3], model_hw(frames.shape[1:3])
        f, v, u = w["pix_frame"], w["pix_v"], w["pix_u"]
        raw, fx_p4d = point4d_depth(p4d, frames, hw)
        chunk_scales: list[float] = []
        with metric_scale_injected(p4d, da3, chunk_scales):
            injected, _ = point4d_depth(p4d, frames, hw)
        nested, mono, fx_da3 = da3_depths(da3, frames[pos])
        # the pipeline's per-clip scale (prep_point4d.stage_metric): DA3-Nested over Point4D on the 16 frames
        s_da3 = float(np.median([median_ratio(cv2.resize(nested[n], (raw.shape[2], raw.shape[1]),
                                                         interpolation=cv2.INTER_AREA), raw[p])
                                 for n, p in enumerate(pos)]))
        raw_pix = at_pixels(raw, pos[f], v, u, src_hw, GEOMETRY_STRIDE, hw)
        inj_pix = at_pixels(injected, pos[f], v, u, src_hw, GEOMETRY_STRIDE, hw)
        found[name] = {
            "pred": {"point4d_raw": raw_pix, "point4d_x_da3": raw_pix * s_da3, "point4d_inject": inj_pix,
                     "da3_nested": at_pixels(nested, f, v, u, src_hw),
                     "da3_metric_large": at_pixels(mono, f, v, u, src_hw)},
            "raw_eval": raw[pos],
            "meta": {"frames": int(len(frames)), "source_hw": list(map(int, src_hw)), "point4d_hw": list(hw),
                     "da3_hw": list(nested.shape[1:]), "point4d_per_clip_da3_scale": s_da3,
                     "inject_chunk_scales": chunk_scales,
                     "inject_over_raw_per_eval_frame": [float(np.median(inj_pix[f == n] / raw_pix[f == n]))
                                                        for n in range(len(pos))],
                     "fx_gt": float(w["fx_gt"]),
                     "fx_point4d": float(np.median(fx_p4d) * src_hw[1] / hw[1]),
                     "fx_da3": float(np.median(fx_da3) * src_hw[1] / nested.shape[2]),
                     "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)},
        }
        print(f"point4d/da3 {name}: {json.dumps(found[name]['meta'])}", flush=True)
        torch.cuda.empty_cache()
    del p4d, da3
    gc.collect()
    torch.cuda.empty_cache()

    d4rt, size = load_d4rt()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, w in windows.items():
        frames, pos, f = w["frames"], w["eval_pos"], w["pix_frame"]
        z_pix, z_grid = d4rt_depths(d4rt, size, frames, pos, f, w["pix_v"], w["pix_u"])
        # the pipeline's D4RT scale (prep_d4rt): D4RT over Point4D on a GRID x GRID cell grid of the 16 frames
        raw_eval = found[name].pop("raw_eval")
        h, wd = raw_eval.shape[1:]
        ys = np.clip(((np.arange(GRID) + 0.5) / GRID * h).astype(int), 0, h - 1)
        xs = np.clip(((np.arange(GRID) + 0.5) / GRID * wd).astype(int), 0, wd - 1)
        s_d4rt = float(np.nanmedian([median_ratio(z_grid[n], raw_eval[n][np.ix_(ys, xs)])
                                     for n in range(len(pos))]))
        pred = found[name]["pred"]
        pred["d4rt_raw"] = z_pix
        pred["point4d_x_d4rt"] = pred["point4d_raw"] * s_d4rt
        found[name]["meta"]["point4d_per_clip_d4rt_scale"] = s_d4rt
        np.savez_compressed(OUT / f"{name}_pred.npz", gt=w["gt"], pix_frame=f, **pred)
        print(f"d4rt {name}: per-clip scale {s_d4rt:.4f}", flush=True)
    (OUT / "meta.json").write_text(json.dumps({n: r["meta"] for n, r in found.items()}, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# report (CPU)
# --------------------------------------------------------------------------- #

def metrics(gt: np.ndarray, pred: np.ndarray, frame: np.ndarray) -> dict[str, float]:
    """Unaligned (is it metric?) and median-aligned per sequence (is the shape right?)."""

    ok = np.isfinite(pred) & (pred > 1e-6)
    g, p, f = gt[ok], pred[ok], frame[ok]
    s = float(np.median(g / p))
    per_frame = np.array([np.median(g[f == n] / p[f == n]) for n in np.unique(f)])

    def abs_rel(q):
        return float(np.mean(np.abs(q - g) / g))

    def delta1(q):
        return float(np.mean(np.maximum(q / g, g / q) < 1.25))

    return {"coverage": float(ok.mean()), "gt_over_pred": s, "abs_rel": abs_rel(p), "delta1": delta1(p),
            "abs_rel_aligned": abs_rel(p * s), "delta1_aligned": delta1(p * s),
            "frame_scale_cv": float(per_frame.std() / per_frame.mean())}


def report(names: Sequence[str]) -> None:
    meta = json.loads((OUT / "meta.json").read_text())
    results = {}
    for name in names:
        with np.load(OUT / f"{name}_pred.npz") as z:
            results[name] = {m: metrics(z["gt"], z[m], z["pix_frame"]) for m in METHODS}
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    def table(title: str, key: str, fmt: str, spread: bool = False) -> list[str]:
        head = f"| method | {' | '.join(names)} |" + (" max/min |" if spread else "")
        rows = [f"### {title}", "", head, "|---" * (len(names) + 1 + spread) + "|"]
        for m in METHODS:
            vals = [results[n][m][key] for n in names]
            extra = f" {max(vals) / min(vals):.2f} |" if spread else ""
            rows.append(f"| {m} | " + " | ".join(format(x, fmt) for x in vals) + " |" + extra)
        return rows + [""]

    lines = ["# Point4D / DA3 / D4RT depth against metric ground truth", "",
             "Pooled over the 16 scored frames of each window (`metric_gt_check.py`). "
             "`gt_over_pred` = median(GT / prediction): 1.00 is metric.", ""]
    lines += table("Scale error: median(GT / pred)", "gt_over_pred", ".3f", spread=True)
    lines += table("AbsRel, unaligned (metric accuracy)", "abs_rel", ".3f")
    lines += table("delta < 1.25, unaligned", "delta1", ".3f")
    lines += table("AbsRel after one median scale per sequence (shape only)", "abs_rel_aligned", ".3f")
    lines += table("Per-frame scale CV within a sequence (drift)", "frame_scale_cv", ".3f")
    lines += ["### Run details", "", "```json", json.dumps(meta, indent=2), "```", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=["prep", "run", "report"])
    parser.add_argument("--only", nargs="*", default=None, help="sequence names")
    args = parser.parse_args(argv)
    names = [n for n in SEQUENCES if not args.only or n in args.only]
    for stage in args.stages:
        {"prep": prep, "run": run, "report": report}[stage](names)


if __name__ == "__main__":
    main()
