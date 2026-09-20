"""Run the measurement pipeline for one question, on demand.

`traj3d_run` is a batch job: three hardcoded targets, stages ordered by model so
each checkpoint loads once.  That shape does not fit here, because the subject is
not known until the agent has named it, and which stages to run is not known
until it has chosen a bucket.

So this module inverts the loop.  All four checkpoints stay resident and the
*videos* stream past them, one question at a time.  That is why
`release_image_processor` is deliberately not used: it exists so the batch runner
could hand SAM3's 3.45 GB to D4RT between stages, and reloading it per question
would cost more than every measurement in this file put together.

Work is cached per (video, subject), not per video.  Two questions can ask about
different subjects in the same clip, and a cache keyed on the clip alone would
silently answer the second one with the first one's subject.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from d4rt_agent.agent_v4.buckets import Bucket
from d4rt_agent.sam_orientany_d4rt_test import traj3d_buckets
from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import (
    CAMERA_GRID_SIZE,
    D4RT_CHECKPOINT,
    D4RT_CONFIG,
    NUM_SUBJECT_POINTS,
    Target,
    VideoPaths,
    write_json,
)

# SAM3 returns every instance matching the phrase, and picking the wrong one is
# the failure nothing downstream can detect -- so when the choice is genuinely
# close, the agent is asked instead of the argmax being taken silently.
#
# Two triggers, because there are two ways to be unsure. A runner-up within 15%
# of the top score means two instances fit the phrase about equally. A top score
# under 0.40 means nothing fits it well: SAM3's score multiplies a per-query
# sigmoid by a whole-image presence sigmoid, so a confident detection of a small
# subject lands far below 0.5 (which is why DEFAULT_CONFIDENCE is already 0.3).
AMBIGUITY_RELATIVE = 0.85
AMBIGUITY_FLOOR = 0.40

# How many candidates are worth showing. Beyond a handful the agent is choosing
# from noise, and every crop costs context.
MAX_CANDIDATES_SHOWN = 4

# How many anchor frames to try before giving up on tracking a subject.
#
# The anchor is chosen by argmax over a per-frame detection score, and that
# argmax is NOT stable: on CameraBench/M0jmSsQ5ptw.3.12 the same phrase picked
# frame 30 (score 0.902, a 1122 px mask) in one environment and frame 18
# (0.785, 368 px) in another, and the weaker one cannot be propagated at all.
# Betting the whole clip on one frame is the fragility; the runners-up are
# already ranked, so trying them costs a few seconds and removes it.
MAX_ANCHOR_ATTEMPTS = 3

# A propagation that holds the subject for fewer frames than this is accepted
# only if nothing better is found. Two frames out of 32 is technically a
# success and nearly useless downstream.
MIN_TRACKED_FRAMES = 4

# Minimum gap between anchor frames worth trying. The top four scores are
# routinely four CONSECUTIVE frames -- on the runner clip they were 18, 19, 20,
# 21 -- which is one view tried four times, and all four failed together.
# Spreading the retries is what makes them independent attempts.
MIN_ANCHOR_SEPARATION = 5


@dataclass
class Candidate:
    index: int
    score: float
    box_xyxy: list[float]
    crop: Any = None


@dataclass
class Grounding:
    """What SAM3 made of a subject phrase, and whether it was sure."""

    status: str                       # "ok" | "not_found"
    subject: str
    anchor_frame: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    ambiguous: bool = False
    reason: str | None = None
    # Best detection score per frame, from the anchor scan. Kept because the
    # single best frame is not reliably propagatable and the runners-up are the
    # obvious fallback -- see QuestionPipeline.commit.
    scan: list[float] = field(default_factory=list)

    @property
    def best_index(self) -> int:
        return max(self.candidates, key=lambda c: c.score).index if self.candidates else -1


class Models:
    """The four checkpoints, built on first use and then kept.

    Lazy because a camera-only question never touches SAM3 or Orient-Anything,
    and a run that happens to draw only camera questions should not pay for them.
    """

    def __init__(self, *, d4rt_config: Path = D4RT_CONFIG,
                 d4rt_checkpoint: Path = D4RT_CHECKPOINT) -> None:
        self.d4rt_config = Path(d4rt_config)
        self.d4rt_checkpoint = Path(d4rt_checkpoint)
        self._d4rt: Any = None
        self._orient: Any = None
        self._bound_video: Any = None

    def sam3_image(self) -> Any:
        from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import image_processor
        return image_processor()

    def orient(self) -> Any:
        if self._orient is None:
            from d4rt_agent.sam_orientany_d4rt_test.traj3d_orient import load_model
            self._orient = load_model()
        return self._orient

    def d4rt(self, sampled: Any) -> Any:
        """The D4RT backend, rebound to this clip.

        Rebinding re-encodes the video in seconds; constructing a new backend
        would reload 14 GB of weights.
        """

        from d4rt_agent.simple_v2_backend import LiveD4RTBackend

        if self._d4rt is None:
            self._d4rt = LiveD4RTBackend(
                sampled_video=sampled,
                point_mode="centroid",
                benchmark_scale=1.0,
                model_config=self.d4rt_config,
                checkpoint=self.d4rt_checkpoint,
            )
            self._bound_video = sampled
        elif self._bound_video is not sampled:
            self._d4rt.rebind_video(sampled)
            self._bound_video = sampled
        return self._d4rt


def subject_key(subject: str | None) -> str:
    """A short, filesystem-safe key for one subject phrase.

    Hashed rather than slugified because the phrase is model-authored free text:
    it can contain anything, and two phrases differing only in punctuation must
    not collide into one cache entry.
    """

    if not subject:
        return "camera_only"
    digest = hashlib.sha1(subject.strip().lower().encode("utf-8")).hexdigest()[:10]
    stem = "".join(ch if ch.isalnum() else "_" for ch in subject.strip().lower())
    return f"{stem[:32].strip('_')}_{digest}"


class QuestionPipeline:
    """Measure one question at a time, reusing whatever a previous one produced."""

    def __init__(self, *, run_dir: Path, models: Models | None = None,
                 lean: bool = False, max_cached_clips: int = 3) -> None:
        self.run_dir = Path(run_dir)
        self.models = models or Models()
        # On a full-census run the transient artifacts dominate: of ~10.9 MB per
        # question only ~144 KB is the measurement itself, and 7076 questions
        # would add ~150 GB and half a million inodes to a filesystem already at
        # 96%. Lean mode drops them once the measurement exists.
        self.lean = bool(lean)
        # Bounded, because a decoded clip is 12 MB (480x270) to 84 MB (720p) and
        # the census opens 943 distinct videos per split. An unbounded dict of
        # them reached the 24 GB host limit after ~1,150 questions and the job
        # was SIGKILLed mid-run.
        #
        # Small is enough: `census_rows` orders questions by relative_path, so
        # the questions sharing a clip are adjacent and an LRU of a few entries
        # captures essentially all the reuse a larger one would.
        self._sampled_cache: "OrderedDict[str, Any]" = OrderedDict()
        self.max_cached_clips = max(1, int(max_cached_clips))
        self.target: Target | None = None
        self.sampled: Any = None

    # -- clip ---------------------------------------------------------------

    def open(self, relative_path: str, subject: str | None = None,
             video_path: str | Path | None = None) -> Any:
        """Point the pipeline at one clip. Cheap and idempotent.

        ``video_path`` comes from the manifest and is authoritative. Deriving it
        from ``relative_path`` alone is wrong across augmentations: DSI-Bench
        ships every split at the same relative path, so the derived form always
        resolves to `std`.

        The sampled-clip cache is therefore keyed on the resolved path, not on
        the relative path, or two splits of one clip would share an entry.
        """

        from d4rt_agent.simple_v2_contracts import sample_video_cpu

        if video_path is not None:
            resolved = Path(video_path)
            root = Path(str(resolved)[: len(str(resolved)) - len(relative_path)].rstrip("/"))
            self.target = Target(relative_path, subject or "", video_root=root)
        else:
            self.target = Target(relative_path, subject or "")
        resolved = self.target.video_path

        key = str(resolved)
        if key in self._sampled_cache:
            self._sampled_cache.move_to_end(key)
        else:
            self._sampled_cache[key] = sample_video_cpu(resolved)
            while len(self._sampled_cache) > self.max_cached_clips:
                self._sampled_cache.popitem(last=False)
        self.sampled = self._sampled_cache[key]
        return self.sampled

    def work_dir(self, subject: str | None) -> VideoPaths:
        assert self.target is not None, "call open() first"
        root = self.run_dir / "pipeline" / self.target.slug / subject_key(subject)
        paths = VideoPaths(root=root)
        paths.mkdirs()
        return paths

    # -- stage 1: SAM3 ------------------------------------------------------

    def ground(self, subject: str) -> Grounding:
        """Find every instance of ``subject`` and say whether the choice is clear.

        The anchor frame is chosen by scanning the whole clip for the frame SAM3
        is most confident about, not by taking frame 0 -- a subject can start
        200 px away and 15 px tall, where SAM3 returns nothing at all, and be
        unmistakable by mid-clip. Propagation carries the identity outwards from
        wherever it was clearest.
        """

        from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import (
            select_anchor_frame, write_frames,
        )

        assert self.target is not None, "call open() first"
        paths = self.work_dir(subject)
        write_frames(self.sampled, paths)
        self.models.sam3_image()  # build once, before the scan

        try:
            anchor, raw, scan = select_anchor_frame(self.sampled.frames_rgb, subject)
        except RuntimeError as error:
            return Grounding(status="not_found", subject=subject, reason=str(error))

        order = np.argsort(-raw["scores"])[:MAX_CANDIDATES_SHOWN]
        candidates = [
            Candidate(
                index=int(i), score=float(raw["scores"][i]),
                box_xyxy=[float(v) for v in raw["boxes"][i]],
                crop=_crop_box(self.sampled.frames_rgb[anchor], raw["boxes"][i]),
            )
            for i in order
        ]
        top = candidates[0].score
        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        ambiguous = bool(runner_up > AMBIGUITY_RELATIVE * top or top < AMBIGUITY_FLOOR)
        reason = None
        if ambiguous:
            reason = (
                f"top score {top:.3f} is below the {AMBIGUITY_FLOOR} confidence floor"
                if top < AMBIGUITY_FLOOR
                else f"runner-up {runner_up:.3f} is within 15% of the top score {top:.3f}"
            )

        self._raw_candidates = raw
        return Grounding(
            status="ok", subject=subject, anchor_frame=int(anchor),
            candidates=candidates, ambiguous=ambiguous, reason=reason,
            scan=[float(v) for v in scan],
        )

    def commit(self, grounding: Grounding, index: int) -> dict[str, Any]:
        """Track the chosen instance through the clip and seed the query points.

        Tries the best anchor frame first, then the next best, because the single
        best-scoring frame is not reliably propagatable and the ranking is
        already in hand. The result kept is the one that holds the subject for
        the most frames, since that is what the trajectory is actually made of.
        """

        from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import (
            ground_frame, propagate_masks, repair_masks, sample_mask_points,
            save_mask_overlays,
        )

        subject = grounding.subject
        paths = self.work_dir(subject)
        frames = self.sampled.frames_rgb

        best: tuple[int, np.ndarray, dict[str, Any], int] | None = None
        errors: list[str] = []
        for anchor, reference, box in self._anchor_candidates(grounding, index, ground_frame):
            try:
                masks, diagnostics = propagate_masks(
                    paths.frames, subject, reference, frames.shape[0],
                    anchor=anchor, reference_box_xyxy=box,
                )
            except RuntimeError as error:
                errors.append(f"anchor {anchor}: {error}")
                continue
            masks[anchor] = reference
            tracked = int(sum(1 for t in range(masks.shape[0]) if masks[t].any()))
            if best is None or tracked > best[3]:
                best = (anchor, masks, diagnostics, tracked)
            if tracked >= MIN_TRACKED_FRAMES:
                break

        if best is None:
            raise RuntimeError(
                "propagation found no object overlapping the chosen instance at any of the "
                f"{MAX_ANCHOR_ATTEMPTS} best anchor frames; " + "; ".join(errors)
            )

        anchor, masks, diagnostics, tracked = best
        masks, repaired = repair_masks(masks, frames, subject, anchor=anchor)
        # Overlays are a debugging surface; the mask array is consumed by the
        # orientation stage in this same call. Neither is written on a lean run:
        # writing 32 JPEGs and a multi-megabyte bool array per question, only to
        # delete them, is pure I/O on a filesystem at 96%.
        if not self.lean:
            save_mask_overlays(frames, masks, paths.masks_overlay)
        np.savez_compressed(paths.masks / "masks.npz", masks=masks)

        # Seed on the LARGEST mask, not frame 0: t_src, t_tgt and t_cam are
        # independent, and it is t_cam=0 that puts the answer in the frame-0
        # camera. Seeding where the subject is best resolved costs nothing.
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        if not areas.any():
            raise RuntimeError(f"no frame has a mask for {subject!r}")
        seed_frame = int(np.argmax(areas))
        subject_px = sample_mask_points(masks[seed_frame], NUM_SUBJECT_POINTS)

        record = self._write_points(paths, subject_px, seed_frame)
        record.update({
            "chosen_index": int(index),
            "requested_anchor_frame": grounding.anchor_frame,
            "anchor_frame": anchor,
            "anchor_retries": errors,
            "tracked_frames_before_repair": tracked,
            "repaired_frames": repaired,
            "frames_without_mask": [int(t) for t in range(masks.shape[0]) if not masks[t].any()],
            "propagation": diagnostics,
        })
        write_json(paths.segmentation_json, record)
        return record

    def _anchor_candidates(self, grounding: Grounding, index: int, ground_frame: Any):
        """The anchor frames to try, best first, each with its reference instance.

        The frame the agent's choice was made on comes first and reuses that
        exact instance. Later frames are re-grounded, and their highest-scoring
        instance is used -- so an anchor move re-resolves identity, which is
        recorded rather than hidden.
        """

        yield (grounding.anchor_frame,
               self._raw_candidates["masks"][index],
               self._raw_candidates["boxes"][index])

        scan = np.asarray(grounding.scan, dtype=np.float64) if grounding.scan else np.zeros(0)
        if scan.size == 0:
            return
        used = [grounding.anchor_frame]
        for frame in np.argsort(-scan):
            frame = int(frame)
            if scan[frame] <= 0.0:
                break                      # the rest score nothing at all
            if len(used) > MAX_ANCHOR_ATTEMPTS:
                return
            if any(abs(frame - seen) < MIN_ANCHOR_SEPARATION for seen in used):
                continue                   # same view as one already tried
            used.append(frame)
            try:
                candidates = ground_frame(self.sampled.frames_rgb[frame], grounding.subject)
            except Exception:
                continue
            if candidates["scores"].size == 0:
                continue
            best = int(np.argmax(candidates["scores"]))
            yield frame, candidates["masks"][best], candidates["boxes"][best]

    def prepare_camera_only(self, subject: str | None = None) -> dict[str, Any]:
        """Seed just the camera grid, for a bucket that needs no subject."""

        paths = self.work_dir(subject)
        from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import write_frames
        write_frames(self.sampled, paths)
        return self._write_points(paths, None, 0)

    def _write_points(
        self, paths: VideoPaths, subject_px: np.ndarray | None, seed_frame: int
    ) -> dict[str, Any]:
        from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import (
            camera_grid_points, to_uv_norm,
        )

        width, height = self.sampled.width, self.sampled.height
        grid_px = camera_grid_points(width, height, CAMERA_GRID_SIZE)
        payload: dict[str, Any] = {
            "seed_frame": seed_frame,
            "camera_grid_px": grid_px.tolist(),
            "camera_grid_uv_norm": to_uv_norm(grid_px, width, height),
            "width": width, "height": height,
        }
        if subject_px is not None:
            payload["subject_points_px"] = subject_px.tolist()
            payload["subject_points_uv_norm"] = to_uv_norm(subject_px, width, height)
        write_json(paths.points_json, payload)
        return payload

    # -- stages 2-4: D4RT, Orient-Anything, analysis -------------------------

    def measure(self, bucket: Bucket, subject: str | None) -> dict[str, Any]:
        """Run the stages this bucket needs and return its report.

        Every stage records completion on disk and is skipped if already done,
        so two questions sharing a (video, subject) pair pay for the pipeline
        once -- and a question whose bucket needs orientation reuses the queries
        a cheaper earlier question already ran on the same pair.
        """

        from d4rt_agent.sam_orientany_d4rt_test.traj3d_analyze import analyze_target
        from d4rt_agent.sam_orientany_d4rt_test.traj3d_orient import orient_target
        from d4rt_agent.sam_orientany_d4rt_test.traj3d_query import (
            NUM_FRAMES, query_camera, query_subject,
        )

        assert self.target is not None, "call open() first"
        paths = self.work_dir(subject)
        points = json.loads(paths.points_json.read_text())
        stages = set(bucket.stages)
        timings: dict[str, float] = {}

        backend = self.models.d4rt(self.sampled)

        if "query_subject" in stages and not (paths.d4rt / "subject.npz").exists():
            started = time.time()
            subject_out = query_subject(
                backend, points["subject_points_uv_norm"], int(points.get("seed_frame", 0))
            )
            np.savez_compressed(paths.d4rt / "subject.npz", **subject_out)
            timings["query_subject"] = time.time() - started

        if not (paths.d4rt / "camera.npz").exists():
            started = time.time()
            camera_out = query_camera(backend, points["camera_grid_uv_norm"])
            np.savez_compressed(paths.d4rt / "camera.npz", **camera_out)
            timings["query_camera"] = time.time() - started

        if "orient" in stages and not (paths.orientation / "orientation.json").exists():
            started = time.time()
            orient_target(
                self.target, self.run_dir, self.models.orient(),
                paths=paths, sampled=self.sampled, save_crops=not self.lean,
            )
            timings["orient"] = time.time() - started

        started = time.time()
        summary = analyze_target(self.target, self.run_dir, paths=paths)
        timings["analyze"] = time.time() - started

        if self.lean:
            self._discard_transients(paths)

        inputs = traj3d_buckets.load_bucket_inputs(paths.root)
        report = bucket.builder(inputs)
        report["pipeline"] = {
            "work_dir": str(paths.root),
            "stages": list(bucket.stages),
            "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
            "warnings": summary.get("warnings", []),
            "tracks_kept": summary.get("tracks_kept"),
            "tracks_total": summary.get("tracks_total"),
            "camera_median_relative_rmse": summary["camera"]["median_relative_rmse"],
            "orientation_frames_usable": summary["orientation"]["frames_usable"],
        }
        return report


    @staticmethod
    def _discard_transients(paths: VideoPaths) -> None:
        """Drop everything the answer does not need, once the measurement exists.

        Kept: `points.json`, `segmentation.json`, `d4rt/*.npz`,
        `orientation/orientation.json`, `analysis/*` -- together ~144 KB, and
        enough that a later question on the same (video, subject) still skips
        segmentation, the D4RT queries and orientation, re-running only the
        CPU-side analysis.

        Dropped: the 32 extracted frames and the mask stack. On a lean run the
        overlays and orientation crops were never written in the first place.
        """

        import shutil

        for directory in (paths.frames, paths.masks, paths.masks_overlay, paths.plots):
            shutil.rmtree(directory, ignore_errors=True)
        for crop in paths.orientation.glob("crop_*.png"):
            crop.unlink(missing_ok=True)


def _crop_box(frame_rgb: np.ndarray, box_xyxy: np.ndarray, margin: float = 0.12) -> Any:
    """Cut one candidate's box out of the anchor frame, for the agent to look at.

    Padded the same way `traj3d_orient.crop_subject` pads: a box cut exactly to
    the detection is hard to identify out of context, and the surrounding pixels
    are usually what distinguishes two people wearing the same thing.
    """

    from PIL import Image

    height, width = frame_rgb.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    pad = margin * max(x1 - x0, y1 - y0)
    cx0, cy0 = int(max(0, round(x0 - pad))), int(max(0, round(y0 - pad)))
    cx1, cy1 = int(min(width, round(x1 + pad))), int(min(height, round(y1 + pad)))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return Image.fromarray(frame_rgb[cy0:cy1, cx0:cx1])
