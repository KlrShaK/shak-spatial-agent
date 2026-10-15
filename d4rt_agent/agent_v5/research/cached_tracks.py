"""Compare two trajectory constructions on five cached clips; no inference or fitting.

The fixed-seed estimate is only a diagnostic: membership checks and plots do not
provide ground-truth 3D motion. A disagreement does not tell us which is correct.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from d4rt_agent.agent_v5.scene import Grounding, lift_instances, rank_instances, scene_from_arrays
from d4rt_agent.agent_v5.research.audit import OUT, REPO

CACHE = REPO / "d4rt_agent/results/agent_v5_vlm4d_point4d_20261010T012544Z/cache/clips"
CASES = [("varanus-tree", "lizard"), ("bear_", "bear"), ("bike-packing", "bike"),
         ("bmx-trees", "person_on_bicycle"), ("carousel", "toy_horse")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for fragment, phrase in CASES:
        base = next(p for p in CACHE.iterdir() if fragment in p.name)
        gp = next(p for p in (base / "sam3").glob(phrase + "_*.npz") if p.stem.rsplit("_", 1)[0] == phrase)
        clip = json.loads((base / "clip.json").read_text())
        with np.load(base / "point4d/scene.npz") as z:
            arrays = {k: z[k] for k in z.files}
        metric = json.loads((base / "point4d/metric.json").read_text())
        scene = scene_from_arrays(base.name, clip, arrays, metric)
        with np.load(gp) as z:
            shape = tuple(z["masks_shape"])
            masks = np.unpackbits(z["masks_packed"], axis=-1)[..., :shape[-1]].astype(bool).reshape(shape)
            g = Grounding(phrase, z["obj_ids"], masks, z["probs"], z["boxes"], z["area_frac"], z["border"])
        record = {"clip": fragment, "phrase": phrase, "scene_path": str(base / "point4d/scene.npz"),
                  "grounding_path": str(gp), "instances": g.count}
        if not g.count:
            record["status"] = "no cached detection"
            records.append(record)
            continue
        k = rank_instances(scene, g, "best")[0]["k"]
        lifted = lift_instances(scene, g, [k], entity_id="e1")
        selected_mask = masks[k, scene.gf_index]
        mh, mw = selected_mask.shape[1:]
        h, w = scene.model_hw
        uv = arrays["grid_uv"]
        seed_x = np.clip((uv[:, 0] * mw / w).astype(int), 0, mw - 1)
        seed_y = np.clip((uv[:, 1] * mh / h).astype(int), 0, mh - 1)
        # Use the existing frame-0 grid, with no confidence threshold fitted or
        # selected from answer performance. Report confidence separately.
        seeds = selected_mask[0, seed_y, seed_x]
        n = int(seeds.sum())
        record.update(instance_index=int(k), seed_points=n)
        if not n:
            record["status"] = "no frame-0 query inside selected instance"
            records.append(record)
            continue
        tracks = arrays["grid_traj"][:, seeds]
        valid = np.isfinite(tracks).all(2) & np.isfinite(tracks[0]).all(1)[None]
        displacements = np.where(valid[..., None], tracks - tracks[0], np.nan)
        fixed_delta = np.nanmedian(displacements, axis=1)
        surface_delta = lifted.smooth - lifted.smooth[0]
        # Projection checks geometric track versus the independently cached
        # segmentation. Out-of-view points count as outside, not as zero motion.
        local = np.einsum("tni,tij->tnj", tracks - scene.cam_t[:, None], scene.cam_R)
        projective = np.einsum("tij,tnj->tni", scene.K, local)
        projected = projective[..., :2] / projective[..., 2:3]
        good = valid & (local[..., 2] > 0) & np.isfinite(projected).all(2)
        xy = np.nan_to_num(projected) * [mw / w, mh / h]
        inside_image = good & (xy[..., 0] >= 0) & (xy[..., 0] < mw) & (xy[..., 1] >= 0) & (xy[..., 1] < mh)
        ix = np.clip(xy[..., 0].astype(int), 0, mw - 1)
        iy = np.clip(xy[..., 1].astype(int), 0, mh - 1)
        membership = selected_mask[np.arange(scene.length)[:, None], iy, ix] & inside_image
        record.update(status="compared", frame0_present=bool(lifted.present[0]),
                      mask_membership_fraction=float(membership.mean()),
                      median_grid_confidence=float(np.nanmedian(arrays["grid_conf"][:, seeds])),
                      surface_net_scales=(scene.to_level(surface_delta[-1]) / scene.scale).tolist(),
                      fixed_seed_net_scales=(scene.to_level(fixed_delta[-1]) / scene.scale).tolist(),
                      net_disagreement_scales=float(np.linalg.norm(surface_delta[-1] - fixed_delta[-1]) / scene.scale),
                      times=scene.t_s.tolist(), surface_delta=surface_delta.tolist(), fixed_seed_delta=fixed_delta.tolist(),
                      membership_per_frame=membership.mean(1).tolist())
        stem = fragment.strip("_")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for j, name in enumerate(("right", "down", "forward")):
            axes[0].plot(scene.t_s, scene.to_level(surface_delta)[:, j] / scene.scale, label=f"surface {name}")
            axes[0].plot(scene.t_s, scene.to_level(fixed_delta)[:, j] / scene.scale, "--", label=f"fixed seed {name}")
        axes[0].set(xlabel="Time (s)", ylabel="Displacement (scene scales)", title=f"{stem}: {phrase}")
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].plot(scene.t_s, membership.mean(1))
        axes[1].set(xlabel="Time (s)", ylabel="Fraction of projected seeds inside SAM mask", ylim=(-.05, 1.05))
        fig.tight_layout()
        fig.savefig(OUT / f"tracks_{stem}.png", dpi=140)
        plt.close(fig)
        sheet = Image.new("RGB", (960, 600), "white")
        frame_files = sorted((base / "frames").glob("*.jpg"))
        assert len(frame_files) == clip["gf"]["length"]
        for tile, pos in enumerate(np.linspace(0, scene.length - 1, 4).round().astype(int)):
            frame = Image.open(frame_files[scene.gf_index[pos]]).convert("RGB")
            frame.thumbnail((480, 270))
            iw, ih = frame.size
            image = np.asarray(frame).copy()
            mask = cv2.resize(selected_mask[pos].astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
            for xy, ok in zip(projected[pos] * [iw / w, ih / h], inside_image[pos]):
                if ok:
                    cv2.circle(image, tuple(xy.astype(int)), 2, (255, 70, 70), -1)
            x0, y0 = (tile % 2) * 480, (tile // 2) * 300
            sheet.paste(Image.fromarray(image), (x0, y0 + 25))
            ImageDraw.Draw(sheet).text((x0 + 4, y0 + 4), f"{stem} {scene.t_s[pos]:.2f}s: green=SAM; red=fixed seeds", fill="black")
        sheet.save(OUT / f"contact_{stem}.png")
        records.append(record)
        print(stem, n, record["mask_membership_fraction"], record["net_disagreement_scales"], flush=True)
    (OUT / "cached_tracks.json").write_text(json.dumps({"scope": "five selected diagnostic clips; no motion ground truth",
                                                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                       "cases": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
