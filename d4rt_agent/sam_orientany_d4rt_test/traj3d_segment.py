"""Stage 1: ground a text phrase to one subject and track its mask through the clip.

Two SAM3 models are used for different jobs.  The *image* model grounds the
phrase on frame 0 and gives us the candidate instances to choose between; the
*video* model propagates a masklet so later frames have a mask even when the
subject is partly occluded or the phrase would match someone else.

Identity is pinned by IoU against the chosen frame-0 mask rather than by
re-prompting per frame, because an open-vocabulary phrase like "the runner"
matches every runner in the shot and nothing in the text alone says which one we
meant.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import (
    CAMERA_GRID_SIZE,
    NUM_SUBJECT_POINTS,
    Target,
    TARGETS,
    VideoPaths,
    default_run_dir,
    video_paths,
    write_json,
)

# SAM3 returns every instance matching the concept, and a low-scoring match on a
# 480x270 frame is usually a hallucinated duplicate of the real subject.
#
# Kept below the 0.5 the examples use because the score is not a plain detection
# probability: `_forward_grounding` multiplies the per-query sigmoid by a
# whole-image presence sigmoid (sam3_image_processor.py:194-198), so a confident
# detection of a small, distant subject lands well under 0.5.
DEFAULT_CONFIDENCE = 0.3

# How much to shrink the mask before seeding query points.  Points on the
# silhouette straddle the subject and whatever is behind it, which is exactly
# where D4RT's depth is most ambiguous.
EROSION_PX = 3

# Half-width of the neighbourhood the mask-area sanity check compares against.
# Wide enough to smooth over one bad frame, narrow enough that genuine growth as
# the subject nears the camera stays inside the band.
HALF_WINDOW = 2


def _to_numpy(value: Any) -> Any:
    """Torch tensor -> numpy, tolerating the bf16 that autocast leaves behind."""

    import torch

    if isinstance(value, torch.Tensor):
        return value.float().cpu().numpy()
    return value


def bf16_autocast():
    """The autocast context every SAM3 call has to run inside.

    ``sam3.perflib.fused.addmm_act`` (used by every ViTDet MLP) casts its inputs
    and weights to bf16 unconditionally and returns bf16, but the ``fc2`` that
    consumes its output is a plain fp32 ``nn.Linear``.  The two only agree if
    autocast is active to cast fc2's weight as well -- otherwise the forward dies
    with "mat1 and mat2 must have the same dtype".

    The README snippet omits this; the authors' own notebook
    (``examples/sam3_image_predictor_example.ipynb``) enters
    ``torch.autocast("cuda", dtype=torch.bfloat16)`` globally before touching the
    model.  This is the same thing, scoped to the calls that need it.
    """

    import torch

    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    )


def write_frames(sampled: Any, paths: VideoPaths) -> list[Path]:
    """Persist the 32 sampled frames as JPEGs.

    The video predictor takes a directory of frames.  It must be *these* frames:
    the sampled clip is a subsequence of the original video, so pointing the
    predictor at the source mp4 would track through frames D4RT never saw.
    """

    from PIL import Image

    paths.frames.mkdir(parents=True, exist_ok=True)
    written = []
    for index in range(sampled.frames_rgb.shape[0]):
        destination = paths.frames / f"{index:05d}.jpg"
        Image.fromarray(sampled.frames_rgb[index]).save(destination, quality=95)
        written.append(destination)
    return written


_IMAGE_PROCESSOR: Any = None


def image_processor(confidence: float = DEFAULT_CONFIDENCE) -> Any:
    """Build the SAM3 image model once and reuse it.

    Cached at module scope because the repair pass below re-segments an
    arbitrary number of frames, and rebuilding a 3.45 GB model per frame would
    dominate the runtime of the whole stage.
    """

    global _IMAGE_PROCESSOR
    if _IMAGE_PROCESSOR is None:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model()
        _IMAGE_PROCESSOR = Sam3Processor(model, confidence_threshold=confidence)
    return _IMAGE_PROCESSOR


def release_image_processor() -> None:
    """Free the image model before the video predictor claims its own memory."""

    global _IMAGE_PROCESSOR
    if _IMAGE_PROCESSOR is None:
        return
    import torch

    _IMAGE_PROCESSOR = None
    torch.cuda.empty_cache()


def ground_frame(
    frame_rgb: np.ndarray, subject: str, *, confidence: float = DEFAULT_CONFIDENCE
) -> dict[str, np.ndarray]:
    """Run text-prompted segmentation on one frame and return every candidate."""

    import torch
    from PIL import Image

    processor = image_processor(confidence)

    # Deliberately a PIL image: Sam3Processor.set_image reads `shape[-2:]` for
    # arrays, which on an HWC numpy frame yields (width, 3) instead of (H, W).
    image = Image.fromarray(frame_rgb)
    with torch.inference_mode(), bf16_autocast():
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=subject, state=state)

    # .float() before .numpy(): under autocast these come back bf16, which numpy
    # has no dtype for and refuses to convert.
    return {
        "masks": state["masks"].squeeze(1).float().cpu().numpy() > 0.5,
        "boxes": state["boxes"].float().cpu().numpy().astype(np.float64),
        "scores": state["scores"].float().cpu().numpy().astype(np.float64),
    }


def select_anchor_frame(
    frames: np.ndarray, subject: str, *, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[int, dict[str, np.ndarray], list[float]]:
    """Pick the frame to ground on: whichever one SAM3 is most sure about.

    Frame 0 is the wrong default.  On CameraBench/M0jmSsQ5ptw.3.12 the runner
    starts ~200 px away and barely 15 px tall, and SAM3 returns nothing at all;
    by the middle of the clip the same person is unmistakable.  Grounding on the
    best frame and letting the tracker carry the identity outwards is strictly
    better than insisting on the frame that happens to be first.

    That costs nothing downstream: `propagate_in_video` defaults to
    `propagation_direction="both"` (sam3_base_predictor.py:256-301), so a prompt
    placed mid-clip still yields frame 0 -- which stage 2 needs, because every
    D4RT query is seeded at `t_src=0`.

    Returns the anchor index, its candidates, and the per-frame best score.
    """

    scan: list[float] = []
    best_per_frame: list[dict[str, np.ndarray]] = []
    for index in range(frames.shape[0]):
        try:
            candidates = ground_frame(frames[index], subject, confidence=confidence)
        except Exception:
            scan.append(0.0)
            best_per_frame.append({"scores": np.zeros(0)})
            continue
        scores = candidates["scores"]
        scan.append(float(scores.max()) if scores.size else 0.0)
        best_per_frame.append(candidates)

    anchor = int(np.argmax(scan))
    if scan[anchor] <= 0.0:
        raise RuntimeError(
            f"SAM3 found no instance of {subject!r} in any of the "
            f"{frames.shape[0]} sampled frames at confidence {confidence}"
        )
    return anchor, best_per_frame[anchor], scan


def choose_instance(candidates: dict[str, np.ndarray], override: int | None = None) -> int:
    """Pick the instance to track: highest confidence, unless overridden."""

    scores = candidates["scores"]
    if scores.size == 0:
        raise RuntimeError("SAM3 returned no instance for this phrase")
    if override is not None:
        if not 0 <= override < scores.size:
            raise ValueError(f"--instance-index {override} outside [0,{scores.size - 1}]")
        return int(override)
    return int(np.argmax(scores))


def draw_candidates(
    frame_rgb: np.ndarray,
    candidates: dict[str, np.ndarray],
    chosen: int,
    destination: Path,
    *,
    anchor: int = 0,
) -> None:
    """Overlay every candidate with its index and score.

    This is the audit surface for the single most dangerous failure in the
    pipeline: silently tracking the wrong person.  Nothing downstream can detect
    it, so it has to be visible here.
    """

    import cv2

    canvas = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    for index, (box, score) in enumerate(zip(candidates["boxes"], candidates["scores"])):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        picked = index == chosen
        colour = (0, 220, 0) if picked else (0, 140, 255)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2 if picked else 1)
        label = f"#{index} {score:.2f}" + (" CHOSEN" if picked else "")
        cv2.putText(
            canvas, label, (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA,
        )
    cv2.putText(
        canvas, f"anchor t={anchor:02d}", (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.imwrite(str(destination), canvas)


def propagate_masks(
    frames_dir: Path,
    subject: str,
    reference_mask: np.ndarray,
    num_frames: int,
    anchor: int = 0,
    reference_box_xyxy: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Track the chosen subject across the sampled clip.

    Returns a (T, H, W) bool array and a diagnostics dict.  The phrase seeds the
    tracker, but the object we follow is chosen by IoU against ``reference_mask``
    on the anchor frame, so we keep the instance the caller picked rather than
    whichever instance the phrase matches best per frame.

    The prompt goes on ``anchor`` and propagation runs in both directions from
    there, so frames before the anchor are covered too.

    **Why there is a second attempt.**  The image model and the video model are
    prompted with the same phrase and resolve it *independently*.  They can
    disagree: on CameraBench/M0jmSsQ5ptw.3.12 the phrase "the runner in orange"
    grounds fine in the image model, but the video model's instances overlap that
    pick on no frame at all, and matching raises.  Given the chosen box, the
    video model can be prompted with the geometry instead, which pins identity by
    construction rather than by hoping two models read one phrase the same way.
    The text attempt still goes first -- it carries semantics a box does not, and
    it is what every previously-validated run used.
    """

    import torch
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(gpus_to_use=range(max(torch.cuda.device_count(), 1)))

    attempts: list[dict[str, Any]] = [{"text": subject}]
    if reference_box_xyxy is not None:
        height, width = reference_mask.shape
        x0, y0, x1, y1 = (float(v) for v in reference_box_xyxy)
        # normalized [xmin, ymin, w, h], which is what add_prompt asserts on
        box = [
            max(0.0, min(1.0, x0 / max(width - 1, 1))),
            max(0.0, min(1.0, y0 / max(height - 1, 1))),
            max(0.0, min(1.0, (x1 - x0) / max(width - 1, 1))),
            max(0.0, min(1.0, (y1 - y0) / max(height - 1, 1))),
        ]
        # text="visual" is the literal gate the model checks
        # (sam3_video_inference.add_prompt: `text_str != "visual"`). With the
        # subject text passed alongside, detection stays TEXT-driven and the box
        # is only a refinement of whatever the phrase matched -- which is the
        # very disagreement being worked around, so that variant fails the same
        # way. "visual" sets TEXT_ID_FOR_VISUAL and promotes the box to the
        # actual prompt.
        attempts.append({"text": "visual", "bounding_boxes": [box], "bounding_box_labels": [1]})

    errors: list[str] = []
    try:
        for index, prompt in enumerate(attempts):
            try:
                per_frame = _run_propagation(predictor, frames_dir, prompt, anchor)
                tracked_id, iou = _match_object(per_frame.get(anchor, {}), reference_mask)
            except RuntimeError as error:
                kind = "box" if "bounding_boxes" in prompt else "text"
                errors.append(f"attempt {index} ({kind}): {error}")
                continue

            height, width = reference_mask.shape
            masks = np.zeros((num_frames, height, width), dtype=bool)
            present = []
            for frame_index in range(num_frames):
                mask = per_frame.get(frame_index, {}).get(tracked_id)
                if mask is not None and mask.shape == reference_mask.shape and mask.any():
                    masks[frame_index] = mask
                    present.append(frame_index)

            diagnostics = {
                "anchor_frame": anchor,
                "tracked_obj_id": tracked_id,
                "anchor_iou_with_reference": iou,
                "objects_seen_at_anchor": sorted(per_frame.get(anchor, {}).keys()),
                "frames_with_mask": present,
                "prompt_used": "box" if "bounding_boxes" in prompt else "text",
                "failed_attempts": errors,
            }
            return masks, diagnostics
    finally:
        del predictor
        torch.cuda.empty_cache()

    raise RuntimeError(
        "propagation returned no object overlapping the chosen instance; "
        + "; ".join(errors)
    )


def _run_propagation(
    predictor: Any, frames_dir: Path, prompt: dict[str, Any], anchor: int
) -> dict[int, dict[int, np.ndarray]]:
    """One prompt, one full propagation pass, as {frame: {obj_id: mask}}."""

    per_frame: dict[int, dict[int, np.ndarray]] = {}
    with bf16_autocast():
        session = predictor.handle_request(
            {"type": "start_session", "resource_path": str(frames_dir)}
        )
        session_id = session["session_id"]
        predictor.handle_request(
            {"type": "add_prompt", "session_id": session_id, "frame_index": anchor, **prompt}
        )

        for response in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            outputs = response["outputs"]
            frame_index = int(response["frame_index"])
            obj_ids = np.asarray(_to_numpy(outputs["out_obj_ids"])).reshape(-1)
            binary = np.asarray(_to_numpy(outputs["out_binary_masks"]))
            per_frame[frame_index] = {
                int(obj_id): np.asarray(binary[i]).squeeze().astype(bool)
                for i, obj_id in enumerate(obj_ids)
            }

        predictor.handle_request({"type": "close_session", "session_id": session_id})
    return per_frame


def _match_object(at_anchor: dict[int, np.ndarray], reference: np.ndarray) -> tuple[int, float]:
    best_id, best_iou = -1, -1.0
    for obj_id, mask in at_anchor.items():
        if mask.shape != reference.shape:
            continue
        union = np.logical_or(mask, reference).sum()
        if union == 0:
            continue
        iou = float(np.logical_and(mask, reference).sum() / union)
        if iou > best_iou:
            best_id, best_iou = int(obj_id), iou
    if best_id < 0:
        raise RuntimeError("propagation returned no object overlapping the chosen instance")
    return best_id, best_iou


def repair_masks(
    masks: np.ndarray,
    frame_rgbs: np.ndarray,
    subject: str,
    *,
    anchor: int = 0,
    area_ratio: float = 3.0,
) -> tuple[np.ndarray, list[int]]:
    """Re-segment frames where propagation clearly failed.

    Videos sampled from long clips put ~1.1 s between consecutive sampled frames,
    far outside what a tracker designed for consecutive video expects.  When the
    masklet is dropped or balloons, fall back to an independent per-frame text
    prompt and keep the candidate closest to the last good centroid.

    Both the centroid chain and the area test start at ``anchor`` and walk
    outwards in each direction, because that is the one frame we know is right --
    chaining from frame 0 would propagate a bad mask into every repair.

    The area test is against a *local* median, not against the anchor's own area.
    A subject walking toward the camera grows several-fold across the clip
    legitimately: on CameraBench/M0jmSsQ5ptw.3.12 the runner goes from 213 px to
    1152 px, so an anchor-relative 3x band condemns 28 of 32 perfectly good
    frames and hands them to a per-frame re-segmentation that can silently latch
    onto a different person. What actually signals a tracker failure is an
    *abrupt* change against the neighbouring frames, which is what this measures.
    """

    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64)
    if areas[anchor] <= 0:
        return masks, []

    suspect = set()
    for t in range(masks.shape[0]):
        if t == anchor:
            continue
        if areas[t] == 0:
            suspect.add(int(t))
            continue
        low = max(0, t - HALF_WINDOW)
        neighbourhood = areas[low : t + HALF_WINDOW + 1]
        local = neighbourhood[neighbourhood > 0]
        reference = float(np.median(local)) if local.size else areas[anchor]
        if reference <= 0:
            continue
        if areas[t] > area_ratio * reference or areas[t] < reference / area_ratio:
            suspect.add(int(t))
    if not suspect:
        return masks, []

    repaired = masks.copy()
    fixed: list[int] = []
    for order in (
        range(anchor + 1, masks.shape[0]),
        range(anchor - 1, -1, -1),
    ):
        last_centroid = _centroid(masks[anchor])
        for t in order:
            if t not in suspect:
                centroid = _centroid(repaired[t])
                if centroid is not None:
                    last_centroid = centroid
                continue
            try:
                candidates = ground_frame(frame_rgbs[t], subject)
            except Exception:
                continue
            if candidates["scores"].size == 0:
                continue
            index = _nearest_candidate(candidates["masks"], last_centroid)
            if index is None:
                continue
            repaired[t] = candidates["masks"][index]
            fixed.append(int(t))
            centroid = _centroid(repaired[t])
            if centroid is not None:
                last_centroid = centroid
    return repaired, sorted(fixed)


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def _nearest_candidate(masks: np.ndarray, centroid: np.ndarray | None) -> int | None:
    best_index, best_distance = None, np.inf
    for index, mask in enumerate(masks):
        point = _centroid(mask)
        if point is None:
            continue
        distance = 0.0 if centroid is None else float(np.linalg.norm(point - centroid))
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def sample_mask_points(mask: np.ndarray, count: int, *, erosion: int = EROSION_PX) -> np.ndarray:
    """Choose ``count`` well-spread interior points, as (N, 2) float pixel xy.

    Farthest-point sampling rather than a random draw so the points cover the
    subject evenly; on a human that means torso, head and limbs all contribute
    instead of whichever region happens to have the most pixels.
    """

    import cv2

    # Back the erosion off one pixel at a time rather than giving up on it: a
    # runner 30 px tall at 480x270 loses their whole body to a 3 px kernel, but
    # keeps a usable core at 1 px. All-or-nothing would drop straight to raw
    # silhouette points, which is where D4RT's depth is least reliable.
    interior = mask
    for radius in range(erosion, 0, -1):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        if eroded.sum() >= count:
            interior = eroded
            break

    ys, xs = np.nonzero(interior)
    if xs.size == 0:
        raise RuntimeError("cannot seed points: mask is empty")
    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    if coords.shape[0] <= count:
        return coords

    chosen = [int(np.argmin(np.linalg.norm(coords - coords.mean(axis=0), axis=1)))]
    distances = np.linalg.norm(coords - coords[chosen[0]], axis=1)
    for _ in range(count - 1):
        nxt = int(np.argmax(distances))
        chosen.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(coords - coords[nxt], axis=1))
    return coords[chosen]


def camera_grid_points(width: int, height: int, size: int = CAMERA_GRID_SIZE) -> np.ndarray:
    """A uniform image grid used to recover camera motion.

    No masking: the two-``t_cam`` scheme in traj3d_analyze is valid for any point
    regardless of whether it moved, so there is nothing to gain by excluding the
    subject and something to lose in spatial conditioning.
    """

    xs = np.linspace(0.05, 0.95, size) * (width - 1)
    ys = np.linspace(0.05, 0.95, size) * (height - 1)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    return grid.astype(np.float64)


def to_uv_norm(points_px: np.ndarray, width: int, height: int) -> list[list[float]]:
    """D4RT wants UV normalised by (W-1, H-1) and clipped into [0,1]."""

    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float64)
    return np.clip(np.asarray(points_px, dtype=np.float64) / scale, 0.0, 1.0).tolist()


def save_mask_overlays(frames: np.ndarray, masks: np.ndarray, destination: Path) -> None:
    import cv2

    destination.mkdir(parents=True, exist_ok=True)
    for index in range(frames.shape[0]):
        canvas = cv2.cvtColor(frames[index].copy(), cv2.COLOR_RGB2BGR)
        tint = np.zeros_like(canvas)
        tint[masks[index]] = (0, 200, 0)
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.45, 0.0)
        cv2.putText(
            canvas, f"t={index:02d}", (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.imwrite(str(destination / f"frame_{index:02d}.jpg"), canvas)


def segment_target(
    target: Target, run_dir: Path, *, instance_index: int | None = None
) -> dict[str, Any]:
    """Run the whole of stage 1 for one video."""

    from d4rt_agent.simple_v2_contracts import sample_video_cpu

    paths = video_paths(run_dir, target)
    paths.mkdirs()

    sampled = sample_video_cpu(target.video_path)
    write_frames(sampled, paths)
    frames = sampled.frames_rgb

    anchor, candidates, scan = select_anchor_frame(frames, target.subject)
    chosen = choose_instance(candidates, instance_index)
    draw_candidates(
        frames[anchor], candidates, chosen, paths.candidates_jpg, anchor=anchor
    )
    reference_mask = candidates["masks"][chosen]

    masks, diagnostics = propagate_masks(
        paths.frames, target.subject, reference_mask, frames.shape[0], anchor=anchor
    )
    masks[anchor] = reference_mask
    masks, repaired = repair_masks(masks, frames, target.subject, anchor=anchor)
    save_mask_overlays(frames, masks, paths.masks_overlay)
    np.save(paths.masks / "masks.npy", masks)

    # Seed the query points on the *largest* mask in the clip, and tell stage 2
    # to use that frame as t_src.
    #
    # Frame 0 is the obvious choice and the wrong one. On this clip the runner
    # starts 30 frames away from the camera and the tracker legitimately has
    # nothing to segment at t=0 or t=1; by t=30 the same person covers 1122 px.
    # Nothing requires t_src=0 -- `_query_explicit_points` takes t_src, t_tgt and
    # t_cam independently, and it is t_cam=0 that puts the answer in the frame-0
    # camera. Seeding where the subject is best resolved and querying t_tgt=0..31
    # still yields the full 32-frame trajectory, in the same coordinate frame.
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    if not areas.any():
        raise RuntimeError(f"no frame has a mask for {target.subject!r}")
    seed_frame = int(np.argmax(areas))
    subject_px = sample_mask_points(masks[seed_frame], NUM_SUBJECT_POINTS)
    grid_px = camera_grid_points(sampled.width, sampled.height)

    write_json(
        paths.points_json,
        {
            "seed_frame": seed_frame,
            "subject_points_px": subject_px.tolist(),
            "subject_points_uv_norm": to_uv_norm(subject_px, sampled.width, sampled.height),
            "camera_grid_px": grid_px.tolist(),
            "camera_grid_uv_norm": to_uv_norm(grid_px, sampled.width, sampled.height),
            "width": sampled.width,
            "height": sampled.height,
        },
    )

    record = {
        "relative_path": target.relative_path,
        "subject": target.subject,
        "slug": target.slug,
        "num_candidates": int(candidates["scores"].size),
        "candidate_scores": candidates["scores"].tolist(),
        "candidate_boxes_xyxy": candidates["boxes"].tolist(),
        "chosen_index": chosen,
        "chosen_score": float(candidates["scores"][chosen]),
        "anchor_frame": anchor,
        "seed_frame": seed_frame,
        "anchor_scan_best_score_per_frame": scan,
        "propagation": diagnostics,
        "repaired_frames": repaired,
        "mask_areas": masks.reshape(masks.shape[0], -1).sum(axis=1).tolist(),
        "frames_without_mask": [
            int(t) for t in range(masks.shape[0]) if not masks[t].any()
        ],
        "sampling": {
            "original_indices": list(sampled.original_indices),
            "total_original_frames": sampled.total_original_frames,
            "fps": sampled.fps,
            "rule": "round(linspace(0, N - 1, 32))",
        },
    }
    write_json(paths.segmentation_json, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="slugs to process")
    parser.add_argument("--instance-index", type=int, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        if args.only and target.slug not in args.only:
            continue
        record = segment_target(target, run_dir, instance_index=args.instance_index)
        print(
            f"{target.slug}: chose #{record['chosen_index']} "
            f"score={record['chosen_score']:.3f} "
            f"of {record['num_candidates']}, "
            f"{len(record['frames_without_mask'])} frames without mask",
            flush=True,
        )


if __name__ == "__main__":
    main()
