"""Stage 3a: per-frame facing direction from Orient-Anything-V2.

The subject is cropped using its SAM3 mask and handed to OAV2 as an RGBA image.
OAV2's preprocessing composites RGBA onto white (utils/app_utils.py:281), so the
mask acts as the matte and ``rembg`` -- which the reference implementation uses
to strip the background -- is not needed at all.  That is both a better matte
(SAM3 knows which person we mean; u2net only knows "salient") and one fewer
176 MB download.

Angles come back in degrees in OAV2's own frame.  Converting them into D4RT's
OpenCV camera frame is done in traj3d_geometry, not here, so this module stays a
thin wrapper around the vendored model.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import (
    ORIENT_DIR,
    ORIENT_HF_FILE,
    ORIENT_HF_REPO,
    Target,
    TARGETS,
    default_run_dir,
    read_json,
    video_paths,
    write_json,
)

# Padding around the mask's bounding box, as a fraction of its larger side.
# OAV2 was trained on renders with the object fully inside the frame and a
# little air around it; a box cropped exactly to the silhouette is off
# distribution and measurably worse.
CROP_MARGIN = 0.12


@contextlib.contextmanager
def orient_anything_context():
    """OAV2's modules import each other as ``utils.*`` relative to its repo root."""

    original_path = list(sys.path)
    original_cwd = Path.cwd()
    sys.path.insert(0, str(ORIENT_DIR))
    os.chdir(ORIENT_DIR)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path


def checkpoint_path() -> Path:
    """Resolve the staged OAV2 checkpoint from the HF cache."""

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=ORIENT_HF_REPO, filename=ORIENT_HF_FILE))


def load_model() -> Any:
    import torch

    with orient_anything_context():
        # Deliberately NOT importing utils.axis_renderer, which the reference
        # app.py pulls in: it needs bpy, which is not installed, and only draws
        # decorative axis arrows.
        from vision_tower import VGGT_OriAny_Ref

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
        model.load_state_dict(torch.load(checkpoint_path(), map_location="cpu"))
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def crop_subject(frame_rgb: np.ndarray, mask: np.ndarray, margin: float = CROP_MARGIN):
    """Cut the subject out as an RGBA PIL image, with the mask as alpha.

    Returns ``(image, centre_xy)`` or ``None`` when the mask is empty.  The
    centre is in full-frame pixels and is what the off-axis correction needs.
    """

    from PIL import Image

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = int(round(margin * max(x1 - x0 + 1, y1 - y0 + 1)))
    height, width = mask.shape
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(width - 1, x1 + pad), min(height - 1, y1 + pad)

    patch = frame_rgb[cy0 : cy1 + 1, cx0 : cx1 + 1]
    alpha = mask[cy0 : cy1 + 1, cx0 : cx1 + 1].astype(np.uint8) * 255
    rgba = np.dstack([patch, alpha])
    centre = (float(xs.mean()), float(ys.mean()))
    return Image.fromarray(rgba, mode="RGBA"), centre


def predict_angles(model: Any, image: Any) -> dict[str, float]:
    """Run OAV2 on one crop.

    ``inf_single_case`` returns a dict of seven zero-dim torch tensors (not a
    tuple, despite the README), of which the ``ref_*`` four are the absolute
    pose; ``rel_*`` are zero when no target image is supplied.
    """

    with orient_anything_context():
        from utils.app_utils import inf_single_case

        answer = inf_single_case(model, image, None)

    return {
        "azimuth_deg": float(answer["ref_az_pred"]),
        "elevation_deg": float(answer["ref_el_pred"]),
        "roll_deg": float(answer["ref_ro_pred"]),
        # 1 = a unique front. 0 = no resolvable orientation; 2/4 = rotational
        # symmetry.  Anything but 1 makes the facing direction meaningless.
        "alpha": float(answer["ref_alpha_pred"]),
    }


def orient_target(target: Target, run_dir: Path, model: Any) -> dict[str, Any]:
    from d4rt_agent.simple_v2_contracts import sample_video_cpu

    paths = video_paths(run_dir, target)
    masks = np.load(paths.masks / "masks.npy")
    sampled = sample_video_cpu(target.video_path)
    frames = sampled.frames_rgb

    rows: list[dict[str, Any]] = []
    for t in range(frames.shape[0]):
        cropped = crop_subject(frames[t], masks[t])
        if cropped is None:
            rows.append({"frame": t, "status": "no_mask"})
            continue
        image, centre = cropped
        image.save(paths.orientation / f"crop_{t:02d}.png")
        try:
            angles = predict_angles(model, image)
        except Exception as exc:  # keep going; one bad frame is not fatal
            rows.append({"frame": t, "status": f"error: {exc}"})
            continue
        rows.append(
            {
                "frame": t,
                "status": "ok",
                "crop_centre_px": list(centre),
                **angles,
            }
        )

    record = {
        "slug": target.slug,
        "frames": rows,
        "num_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "num_ambiguous": sum(
            1 for r in rows if r.get("status") == "ok" and r.get("alpha") != 1.0
        ),
        "convention": (
            "OAV2 native: degrees, azimuth 0-359 with 0 = facing the camera, "
            "elevation -90..89, roll -180..179. Converted to OpenCV camera axes "
            "in traj3d_geometry.oav2_axes_in_opencv."
        ),
    }
    write_json(paths.orientation / "orientation.json", record)
    return record


def mirror_test(model: Any) -> dict[str, Any]:
    """Confirm the azimuth handedness assumed by traj3d_geometry.

    Objective and ground-truth-free: mirroring an image must negate azimuth and
    roll while leaving elevation alone.  If azimuth instead comes back unchanged,
    our OAV2 -> OpenCV basis has the wrong handedness and every axis triad would
    be mirrored -- a failure that looks entirely plausible on a plot, which is
    why it is checked explicitly rather than assumed.
    """

    from PIL import Image

    # bottle.jpg used to be in this list and had to go: it is rotationally
    # symmetric, comes back alpha=0 ("no resolvable orientation"), and its
    # azimuth is noise. It still scored as "supporting" on a 59-vs-65 degree
    # margin, which is a coin flip being counted as evidence.
    results = []
    for name in ("skateboard-0.jpg", "skateboard-1.jpg", "F35-0.jpg", "table-0.jpg"):
        path = ORIENT_DIR / "assets" / "examples" / name
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        mirrored = Image.fromarray(np.fliplr(np.asarray(image)))
        base = predict_angles(model, image)
        flip = predict_angles(model, mirrored)

        # alpha != 1 means the model itself reports no single resolvable
        # orientation, so its azimuth carries no information to test.
        if base.get("alpha", 1) != 1 or flip.get("alpha", 1) != 1:
            results.append({"image": name, "skipped": "alpha != 1 (no resolvable orientation)"})
            continue

        expected = (-base["azimuth_deg"]) % 360.0
        actual = flip["azimuth_deg"] % 360.0
        delta = min(abs(actual - expected), 360.0 - abs(actual - expected))
        unchanged = min(
            abs(actual - base["azimuth_deg"] % 360.0),
            360.0 - abs(actual - base["azimuth_deg"] % 360.0),
        )
        results.append(
            {
                "image": name,
                "azimuth_deg": base["azimuth_deg"],
                "azimuth_mirrored_deg": flip["azimuth_deg"],
                "elevation_deg": base["elevation_deg"],
                "elevation_mirrored_deg": flip["elevation_deg"],
                "distance_to_negated": delta,
                "distance_to_unchanged": unchanged,
                # A decisive margin, not merely the nearer of the two. Near
                # az=0 or az=180 negation barely moves the angle, so both
                # hypotheses fit and the case says nothing either way.
                "supports_current_basis": delta < unchanged and unchanged - delta > 20.0,
                "decisive": unchanged - delta > 20.0 or delta - unchanged > 20.0,
            }
        )

    decisive = [r for r in results if r.get("decisive")]
    supporting = [r for r in decisive if r["supports_current_basis"]]
    if not decisive:
        verdict = "INCONCLUSIVE: no example discriminated the two hypotheses"
    elif len(supporting) == len(decisive):
        verdict = "basis confirmed"
    else:
        verdict = "BASIS SUSPECT: azimuth handedness may be inverted"
    return {
        "per_image": results,
        "num_decisive": len(decisive),
        "num_supporting": len(supporting),
        "note": (
            "This checks azimuth HANDEDNESS only. It cannot validate the "
            "elevation sign or the azimuth->image-x mapping, because both "
            "candidate conventions are symmetric under az -> -az. Those are "
            "pinned by test_facing_matches_measured_examples instead."
        ),
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--mirror-test",
        action="store_true",
        help="run the handedness check on OAV2's own examples and exit",
    )
    args = parser.parse_args()

    model = load_model()

    if args.mirror_test:
        import json

        print(json.dumps(mirror_test(model), indent=2))
        return

    run_dir = args.run_dir or default_run_dir()
    for target in TARGETS:
        if args.only and target.slug not in args.only:
            continue
        record = orient_target(target, run_dir, model)
        print(
            f"{target.slug}: {record['num_ok']}/32 frames oriented, "
            f"{record['num_ambiguous']} ambiguous",
            flush=True,
        )


if __name__ == "__main__":
    main()
