"""The per-clip state every question on that clip is measured against.

Promoted from ``dyn_dataset_exploration/decomposition/scene4d.py``. Changes from the research
copy: the scene is built from a backbone's arrays (``scene_from_arrays``) instead of a cache
directory, the metric scale is whatever the active backbone provides (with its source named), and
masks are lifted through ``sampling.interior_cells`` -- the boundary-margin and depth-edge rule --
instead of a one-cell erosion.

This first part needs no model: which frames the benchmark's question generator looked
at, and what a second means on them.

**The generator frame list (GF).** ``multi_json/<ds>/<clip>_frame_sampling.json`` gives the
16 GF indices the benchmark samples and the file each one came from. They are
``floor(linspace(0, L - 1, 16))`` over a list of length ``L = last index + 1``, and on every
clip ``L`` equals the number of GT mask frames: the GF is the annotated frame list. That is
every file for the 2D sources, but every 5th RGB frame for DynamicReplica (60 of 300) and
PointOdyssey (~620 of ~3100). The shipped ``total_frames`` is *not* ``L`` on 106 clips
(e.g. 69 against 90 on ``davis_bmx-bumps``) and is kept only as a reported field.
``dyn_bench_data.resolve_media`` maps sampled index *i* to the *i*-th file on disk, which is
wrong for the two strided sources; rebuilding the GF from the shipped basenames is not.

**Seconds.** Question timestamps are ``gf_index / fps_gf`` with one rate per source. The
evidence for each rate is recomputed by ``build_catalog`` and checked by the tests; a
source whose rate is not established has ``fps=None``, and any timestamp on it is flagged
rather than mapped to a guessed frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_FRAME_NUMBER = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


@dataclass(frozen=True)
class FrameRate:
    fps: float | None
    status: str          # "verified" | "approximate" | "unverified"
    evidence: str


# Evidence is the question text itself: a question saying "the end (t≈15s)" about a
# 91-frame GF pins the rate at 6. ``build_catalog`` recounts, per source, how many question
# timestamps would fall past the clip's end at each candidate rate.
FPS_GF: dict[str, FrameRate] = {
    "ytvis": FrameRate(6.0, "verified", "'end (t≈5s)' on 30-frame clips (29/6 = 4.8); no timestamp past the end at 6 fps"),
    "sav": FrameRate(6.0, "verified", "'end (t≈15s)' on 90-frame clips (89/6 = 14.8); no timestamp past the end at 6 fps"),
    "davis": FrameRate(30.0, "verified", "'end (1.63s)' on a 50-frame clip, 'end (t=2.3s)' on 70 frames"),
    "dynamic_replica": FrameRate(6.0, "verified", "'end (9.83s)' = 59/6 on the 60-frame GF"),
    "dynpose": FrameRate(12.0, "approximate", "'t≈5.5s' ends 69 frames (12.4 fps), 'middle (2.08s)' of 48 (11.3)"),
    "spring": FrameRate(None, "unverified", "one timestamped question, not at a clip boundary"),
    "totalrecon": FrameRate(None, "unverified", "four timestamped questions, none at a clip boundary"),
    "point_odyssey": FrameRate(None, "unverified", "no timestamped questions"),
}


def frame_number(name: str) -> int | None:
    """The frame number encoded in a file name (``rgb_00210.jpg`` -> 210)."""

    match = _FRAME_NUMBER.search(name)
    return int(match.group(1)) if match else None


def list_frames(frames_dir: str | Path) -> list[str]:
    """Image files in a clip directory, in frame-number order, not string order."""

    names = [
        name for name in os.listdir(frames_dir)
        if Path(name).suffix.lower() in IMAGE_SUFFIXES and frame_number(name) is not None
    ]
    return sorted(names, key=lambda name: (frame_number(name), name))


@dataclass(frozen=True)
class GeneratorFrames:
    """The frames the generator reasoned over, in its order, resolved to files on disk."""

    names: tuple[str, ...]      # file name of every GF frame
    native: tuple[int, ...]     # frame number parsed from each name
    sampled: tuple[int, ...]    # the 16 GF indices the benchmark samples
    mode: str                   # "identity" | "stride" | "interpolated"
    stride: int | None
    reported_total: int         # the shipped ``total_frames``; not always the GF length

    @property
    def length(self) -> int:
        return len(self.names)


def generator_frames(names_on_disk: Sequence[str], sampling: Mapping[str, Any]) -> GeneratorFrames:
    """Rebuild the GF from a clip's files and its ``frame_sampling.json`` payload.

    The GF length is ``last sampled index + 1``. Two shapes occur on the benchmark:
    *identity*, the GF is every file (the 2D sources, TotalRecon, Spring), and *stride*, a
    constant integer step reproduces every shipped basename (DynamicReplica and
    PointOdyssey, step 5). *interpolated* (piecewise-linear through the 16 shipped frames,
    snapped to the nearest file) is a fallback that no current clip needs. Whatever the
    mode, the result must reproduce all 16 shipped basenames exactly, or this raises.
    """

    names = sorted(names_on_disk, key=lambda name: (frame_number(name), name))
    indices = [int(i) for i in sampling["frame_indices"]]
    if not indices:
        raise ValueError("frame_sampling.json lists no frame_indices")
    total = max(indices) + 1
    shipped = [os.path.basename(str(p)) for p in sampling["frame_paths"]]
    if len(indices) != len(shipped):
        raise ValueError("frame_indices and frame_paths disagree in length")
    by_number = {frame_number(name): name for name in names}
    shipped_numbers = [frame_number(name) for name in shipped]
    if any(n is None for n in shipped_numbers):
        raise ValueError(f"unparseable shipped frame names: {shipped[:3]}")

    def _checked(gf_names: list[str], mode: str, stride: int | None) -> GeneratorFrames:
        wrong = [(i, gf_names[i], s) for i, s in zip(indices, shipped) if gf_names[i] != s]
        if wrong:
            raise ValueError(f"{mode} GF disagrees with the shipped sample: {wrong[:3]}")
        return GeneratorFrames(
            names=tuple(gf_names),
            native=tuple(int(frame_number(n)) for n in gf_names),
            sampled=tuple(indices),
            mode=mode,
            stride=stride,
            reported_total=int(sampling.get("total_frames", total)),
        )

    if len(names) == total:
        try:
            return _checked(list(names), "identity", 1)
        except ValueError:
            pass

    if len(indices) >= 2 and indices[1] != indices[0]:
        step = (shipped_numbers[1] - shipped_numbers[0]) / (indices[1] - indices[0])
        if float(step).is_integer() and step > 0:
            start = shipped_numbers[0] - int(step) * indices[0]
            wanted = [start + int(step) * i for i in range(total)]
            if all(n in by_number for n in wanted) and all(
                start + int(step) * i == n for i, n in zip(indices, shipped_numbers)
            ):
                return _checked([by_number[n] for n in wanted], "stride", int(step))

    available = np.array(sorted(by_number), dtype=np.int64)
    target = np.interp(np.arange(total), indices, shipped_numbers)
    snapped = available[np.abs(available[None, :] - target[:, None]).argmin(axis=1)]
    return _checked([by_number[int(n)] for n in snapped], "interpolated", None)


def frame_rate(dataset: str) -> FrameRate:
    return FPS_GF.get(dataset, FrameRate(None, "unverified", f"no entry for {dataset!r}"))


def seconds_axis(n_frames: int, fps: float | None) -> list[float] | None:
    """``t`` of every GF frame, or None when the source's rate is not established."""

    if fps is None:
        return None
    return [round(i / fps, 6) for i in range(int(n_frames))]


def seconds_to_index(t: float, fps: float | None, n_frames: int) -> int | None:
    """Nearest GF index to a time in seconds, clipped to the clip; None without a rate."""

    if fps is None:
        return None
    return int(min(max(round(float(t) * fps), 0), int(n_frames) - 1))


# =========================================================================== #
# The Point4D scene, SAM3 groundings, and entity paths (S0 -> S2 -> S3 inputs)
# =========================================================================== #
#
# World = the camera of GF frame 0: +X right, +Y down, +Z forward, in the backbone's
# model-relative units (``metres_per_unit`` converts, with its source and spread as provenance).
# Everything below is CPU-only NumPy so the notebook kernel can import it.

from d4rt_agent.sam_orientany_d4rt_test import traj3d_buckets, traj3d_geometry  # noqa: E402
from d4rt_agent.agent_v5.program import Window  # noqa: E402

MIN_CORE_CELLS = 3            # interior mask cells needed for a confident position
DEPTH_BAND = 1.6              # cells deeper/shallower than median * / band are background bleed
GAP_MERGE_S = 2.0             # fragments of one entity: max time gap ...
JUMP_MERGE_SCALES = 0.35      # ... and max 3D jump, in scene scales
MIN_LIFESPAN_S = 0.5          # a tracklet shorter than this is not an instance (>= 3 frames)
SMOOTH_WINDOW_S = 0.6


@dataclass
class Scene:
    slug: str
    clip: dict[str, Any]
    gf_index: np.ndarray          # (T,) GF index of every scene frame (strided on long clips)
    cam_R: np.ndarray             # (T,3,3) camera-t -> world
    cam_t: np.ndarray             # (T,3) camera centre, world
    K: np.ndarray                 # (T,3,3) at model resolution
    points: np.ndarray            # (T,gh,gw,3) world coordinates
    model_hw: tuple[int, int]
    source_hw: tuple[int, int]
    stride: int
    grid_traj: np.ndarray         # (T,N,3) frame-0 grid queries tracked through the clip
    grid_conf: np.ndarray
    up: np.ndarray                # world up, unit
    up_fraction: float
    up_source: str                # "plane" | "camera"
    level: np.ndarray             # columns [right, down, forward] in world coords
    scale: float                  # scene scale, model units
    fps: float | None
    metres_per_unit: float | None     # from the active backbone's metric source
    metric_iqr: float | None
    metric_source: str | None = None  # "da3-nested" | "d4rt-native" | None
    pose_valid: np.ndarray | None = None

    @property
    def length(self) -> int:
        return len(self.gf_index)

    @property
    def t_s(self) -> np.ndarray | None:
        return None if self.fps is None else self.gf_index / self.fps

    @property
    def duration_s(self) -> float | None:
        t = self.t_s
        return None if t is None else float(t[-1] - t[0])

    def to_level(self, vectors: np.ndarray) -> np.ndarray:
        """World vectors -> [right, down, forward] components in the levelled frame."""

        return np.asarray(vectors) @ self.level

    def to_camera(self, positions: np.ndarray, t: int) -> np.ndarray:
        """World positions -> the camera of scene frame ``t``."""

        return (np.asarray(positions) - self.cam_t[t]) @ self.cam_R[t]


def scene_from_arrays(slug: str, clip: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                      metric: Mapping[str, Any] | None = None) -> Scene:
    """A Scene from a backbone's arrays (the ``scene.npz`` layout) and its metric scale.

    ``metric`` is ``{"metres_per_unit", "iqr_over_median", "source"}``, or None when the
    backbone gives no metric scale; every metric reading then says so instead of guessing.
    """

    metric = dict(metric or {})
    points = np.asarray(arrays["points"]).astype(np.float32)
    cloud0 = points[0].reshape(-1, 3)
    cloud0 = cloud0[np.isfinite(cloud0).all(1)]
    scale = float(traj3d_geometry.cloud_scale(cloud0))
    up, fraction = traj3d_geometry.estimate_world_up(cloud0[:: max(1, len(cloud0) // 4000)])
    source = "plane"
    if up is None:
        up, source = np.array([0.0, -1.0, 0.0]), "camera"
    level = traj3d_buckets.level_basis(up)
    if level is None:
        level, source = np.eye(3), "camera"
    return Scene(
        slug=slug, clip=dict(clip), gf_index=np.asarray(arrays["gf_index"]).astype(int),
        cam_R=np.asarray(arrays["cam_R"]).astype(np.float64), cam_t=np.asarray(arrays["cam_t"]).astype(np.float64),
        K=np.asarray(arrays["K"]).astype(np.float64), points=points,
        model_hw=tuple(int(v) for v in arrays["model_hw"]), source_hw=tuple(int(v) for v in arrays["source_hw"]),
        stride=int(arrays["geometry_stride"]), grid_traj=np.asarray(arrays["grid_traj"]),
        grid_conf=np.asarray(arrays["grid_conf"]),
        up=np.asarray(up, dtype=np.float64), up_fraction=float(fraction), up_source=source, level=level,
        scale=scale, fps=clip.get("fps_gf"), metres_per_unit=metric.get("metres_per_unit"),
        metric_iqr=metric.get("iqr_over_median"), metric_source=metric.get("source"),
        pose_valid=np.asarray(arrays["pose_valid"], dtype=bool) if "pose_valid" in arrays else None,
    )


@dataclass
class Grounding:
    """Every SAM3 instance found for one phrase, on all GF frames."""

    phrase: str
    obj_ids: np.ndarray           # (K,)
    masks: np.ndarray             # (K,T_gf,mh,mw) bool
    probs: np.ndarray             # (K,T_gf), NaN where absent
    boxes: np.ndarray             # (K,T_gf,4) normalised xyxy
    area_frac: np.ndarray         # (K,T_gf)
    border: np.ndarray            # (K,T_gf)
    used_phrase: str | None = None    # set when a simpler prompt than ``phrase`` had to be used
    sessions: tuple[int, ...] | None = None

    @property
    def count(self) -> int:
        return len(self.obj_ids)

    def present(self, k: int) -> np.ndarray:
        return ~np.isnan(self.probs[k])


def phrase_key(phrase: str) -> str:
    import hashlib

    stem = "".join(ch if ch.isalnum() else "_" for ch in phrase.strip().lower())[:32].strip("_")
    return f"{stem}_{hashlib.sha1(phrase.strip().lower().encode()).hexdigest()[:8]}"


def load_grounding(out: Path, slug: str, phrase: str) -> Grounding | None:
    path = Path(out) / "clips" / slug / "sam3" / f"{phrase_key(phrase)}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        shape = tuple(int(v) for v in z["masks_shape"])
        masks = np.unpackbits(z["masks_packed"], axis=-1)[..., : shape[-1]].astype(bool)
        return Grounding(phrase, z["obj_ids"], masks.reshape(shape), z["probs"], z["boxes"],
                         z["area_frac"], z["border"])


def _cell_coverage(mask: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """Fraction of each depth-cloud cell covered by ``mask`` (an integral-image box mean)."""

    mh, mw = mask.shape
    gh, gw = grid_hw
    integral = np.zeros((mh + 1, mw + 1), dtype=np.float64)
    integral[1:, 1:] = mask.astype(np.float64).cumsum(0).cumsum(1)
    r = np.linspace(0, mh, gh + 1)
    c = np.linspace(0, mw, gw + 1)
    r0, r1 = np.floor(r[:-1]).astype(int), np.maximum(np.ceil(r[1:]).astype(int), np.floor(r[:-1]).astype(int) + 1)
    c0, c1 = np.floor(c[:-1]).astype(int), np.maximum(np.ceil(c[1:]).astype(int), np.floor(c[:-1]).astype(int) + 1)
    r1, c1 = np.minimum(r1, mh), np.minimum(c1, mw)
    total = (integral[np.ix_(r1, c1)] - integral[np.ix_(r0, c1)]
             - integral[np.ix_(r1, c0)] + integral[np.ix_(r0, c0)])
    area = np.outer(r1 - r0, c1 - c0).astype(np.float64)
    return total / np.maximum(area, 1.0)


@dataclass
class EntityTrack:
    """One entity's world position per scene frame, and how much to trust each."""

    id: str
    phrase: str
    instances: list[int]                  # indices into the Grounding
    present: np.ndarray                   # (T,) a mask exists this frame
    confident: np.ndarray                 # (T,) enough interior cells for a position
    xyz: np.ndarray                       # (T,3) world, NaN where unusable
    smooth: np.ndarray                    # (T,3)
    cells: np.ndarray                     # (T,)
    area_frac: np.ndarray                 # (T,)
    border: np.ndarray                    # (T,)
    centroid: np.ndarray                  # (T,2) image coords, 0..1 (x, y)
    noise: float                          # residual scale, model units
    policy: str = "best"
    notes: list[str] = field(default_factory=list)
    margin_px: np.ndarray | None = None   # (T,) boundary margin the interior rule used, NaN = no 3D

    @property
    def first(self) -> int | None:
        idx = np.flatnonzero(self.present)
        return int(idx[0]) if len(idx) else None

    @property
    def last(self) -> int | None:
        idx = np.flatnonzero(self.present)
        return int(idx[-1]) if len(idx) else None


def effective_fps(scene: Scene) -> float:
    """Frames per second of the scene's own frames (lower than the clip's when subsampled)."""

    duration = scene.duration_s
    if duration and duration > 0:
        return (scene.length - 1) / duration
    return scene.fps or 6.0


def metric_scale(scene: Scene) -> tuple[float | None, float | None]:
    """``(metres per unit, relative spread)`` from the active backbone's metric source."""

    return scene.metres_per_unit, scene.metric_iqr


def subsample_scene(scene: Scene, gf_positions: Sequence[int]) -> Scene:
    """The scene restricted to the frames at the given GF indices (e.g. the benchmark's 16).

    The question generator saw only those frames, so a measurement made on them is the
    fairer test of "what would its text say"; the notebook can compare both.
    """

    import dataclasses

    keep = np.array([int(np.argmin(np.abs(scene.gf_index - g))) for g in gf_positions])
    keep = np.unique(keep)
    return dataclasses.replace(
        scene, gf_index=scene.gf_index[keep], cam_R=scene.cam_R[keep], cam_t=scene.cam_t[keep], K=scene.K[keep],
        points=scene.points[keep], grid_traj=scene.grid_traj[keep], grid_conf=scene.grid_conf[keep],
        pose_valid=None if scene.pose_valid is None else scene.pose_valid[keep])


def smoothing_window(scene: Scene, length: int) -> int:
    fps = effective_fps(scene)
    window = int(round(SMOOTH_WINDOW_S * fps)) | 1
    return int(min(max(window, 3), 9, length if length % 2 else length - 1))


def lift_instances(scene: Scene, g: Grounding, instances: Sequence[int], *, entity_id: str,
                   policy: str = "best") -> EntityTrack:
    """Lift the union of ``instances`` through the scene's depth clouds, frame by frame.

    Per frame: the interior cells of the mask (``sampling.interior_cells``: far enough from the
    mask's edge, and not on a depth discontinuity, so the silhouette -- where depth blends
    foreground and background -- is never sampled); cells whose camera depth is far from the
    median are dropped as background bleed; the position is the geometric median of what is left.
    A mask too thin for any interior keeps its 2D evidence (presence, area, centroid) and gets no
    3D position on that frame.
    """

    from d4rt_agent.agent_v5.sampling import interior_cells

    T = scene.length
    xyz = np.full((T, 3), np.nan)
    cells = np.zeros(T, dtype=int)
    present = np.zeros(T, dtype=bool)
    confident = np.zeros(T, dtype=bool)
    area = np.zeros(T)
    border = np.zeros(T, dtype=bool)
    centroid = np.full((T, 2), np.nan)
    margins = np.full(T, np.nan)
    for pos in range(T):
        gf = int(scene.gf_index[pos])
        mask = np.zeros(g.masks.shape[2:], dtype=bool)
        for k in instances:
            mask |= g.masks[k, gf]
            if g.present(k)[gf]:
                area[pos] += float(g.area_frac[k, gf])
                border[pos] |= bool(g.border[k, gf])
        if not mask.any():
            continue
        present[pos] = True
        ys, xs = np.nonzero(mask)
        centroid[pos] = (xs.mean() / mask.shape[1], ys.mean() / mask.shape[0])
        interior = interior_cells(mask, scene.points[pos], scene.cam_R[pos], scene.cam_t[pos],
                                  min_cells=MIN_CORE_CELLS)
        if interior.cells is None:
            continue
        margins[pos] = interior.margin_px
        pts = scene.points[pos][interior.cells]
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) == 0:
            continue
        depth = ((pts - scene.cam_t[pos]) @ scene.cam_R[pos])[:, 2]
        keep = (depth > 1e-6) & (depth < np.median(depth) * DEPTH_BAND) & (depth > np.median(depth) / DEPTH_BAND)
        if keep.sum() < MIN_CORE_CELLS:
            continue
        xyz[pos] = traj3d_geometry.geometric_median(pts[keep])
        cells[pos] = int(keep.sum())
        confident[pos] = bool(keep.sum() >= MIN_CORE_CELLS and not interior.backed_off)
    smooth = smooth_track(scene, xyz)
    resid = np.linalg.norm(xyz - smooth, axis=1)
    resid = resid[np.isfinite(resid)]
    noise = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) if len(resid) > 2 else 0.0
    phrase = g.phrase
    track = EntityTrack(entity_id, phrase, list(instances), present, confident, xyz, smooth, cells,
                        area, border, centroid, noise, policy)
    track.margin_px = margins
    return track


def smooth_track(scene: Scene, xyz: np.ndarray) -> np.ndarray:
    """The one smoothing every entity path goes through (Savitzky-Golay, order 2, about
    ``SMOOTH_WINDOW_S`` long). Measurements read this, never the raw path; the raw path is
    kept only for the noise floor (draft rule: the smoothed trajectory is the one used)."""

    t = scene.t_s
    if t is not None and len(t) == len(xyz) and len(t) > 2 and not np.allclose(np.diff(t), np.diff(t)[0]):
        # A local quadratic in seconds, rather than sample index, preserves constant
        # physical velocity across irregular sampling. Missing samples stay missing.
        values = np.asarray(xyz, dtype=float)
        result = values.copy()
        valid = np.isfinite(values).all(1)
        for i in np.flatnonzero(valid):
            use = valid & (np.abs(t - t[i]) <= SMOOTH_WINDOW_S / 2)
            if use.sum() >= 3:
                result[i] = np.polynomial.polynomial.polyfit(t[use] - t[i], values[use], 2)[0]
        return result
    return traj3d_geometry.smooth_trajectory(xyz, window=smoothing_window(scene, len(xyz)), polyorder=2)


# --- choosing which instances an entity role means -------------------------------------- #

def instance_stats(scene: Scene, g: Grounding) -> list[dict[str, Any]]:
    """Per instance: lifespan, score, size and centrality on the scene's frames."""

    frames = scene.gf_index
    rows = []
    for k in range(g.count):
        pres = g.present(k)[frames]
        if not pres.any():
            rows.append({"k": k, "frames": 0, "span": (None, None), "score": 0.0, "area": 0.0,
                         "centrality": 0.0, "presence": 0.0})
            continue
        idx = np.flatnonzero(pres)
        boxes = g.boxes[k][frames][pres]
        centre = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
        rows.append({
            "k": k, "frames": int(pres.sum()), "span": (int(idx[0]), int(idx[-1])),
            "score": float(np.nanmean(g.probs[k][frames][pres])),
            "area": float(np.mean(g.area_frac[k][frames][pres])),
            "centrality": float(1.0 - np.clip(np.linalg.norm(centre - 0.5, axis=1).mean() / 0.7071, 0, 1)),
            "presence": float(pres.mean()),
        })
    return rows


def rank_instances(scene: Scene, g: Grounding, policy: str) -> list[dict[str, Any]]:
    """Instances best-first. ``salient`` (the implicit main subject) favours large, central
    and long-lived; ``best`` favours long-lived and confident."""

    rows = instance_stats(scene, g)
    for row in rows:
        if policy == "salient":
            row["rank_score"] = row["presence"] * row["score"] * np.sqrt(row["area"]) * (0.5 + row["centrality"])
        else:
            row["rank_score"] = row["presence"] * row["score"]
    return sorted(rows, key=lambda r: -r["rank_score"])


def merge_fragments(scene: Scene, tracks: list[EntityTrack]) -> list[EntityTrack]:
    """Join tracklets that are one entity SAM3 lost and re-found.

    Two tracklets merge when their lifespans do not overlap, the gap between them is short
    and the entity did not teleport across it. Applied greedily, earliest first.
    """

    tracks = sorted(tracks, key=lambda t: t.first if t.first is not None else 10**9)
    merged = True
    while merged:
        merged = False
        for a in range(len(tracks)):
            for b in range(len(tracks)):
                if a == b or tracks[a].last is None or tracks[b].first is None:
                    continue
                ta, tb = tracks[a], tracks[b]
                if ta.last >= tb.first:
                    continue
                if scene.t_s is not None:
                    gap_ok = scene.t_s[tb.first] - scene.t_s[ta.last] <= GAP_MERGE_S
                else:
                    gap_ok = scene.gf_index[tb.first] - scene.gf_index[ta.last] <= 2
                if not gap_ok:
                    continue
                if (ta.present & tb.present).any():
                    continue
                pa, pb = ta.xyz[ta.last], tb.xyz[tb.first]
                if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                    continue
                if np.linalg.norm(pa - pb) > JUMP_MERGE_SCALES * scene.scale:
                    continue
                tracks[a] = _union(scene, ta, tb)
                tracks.pop(b)
                merged = True
                break
            if merged:
                break
    return tracks


def _union(scene: Scene, a: EntityTrack, b: EntityTrack) -> EntityTrack:
    use_b = b.present & ~a.present
    def pick(x, y):
        out = x.copy()
        out[use_b] = y[use_b]
        return out
    xyz = pick(a.xyz, b.xyz)
    smooth = smooth_track(scene, xyz)
    resid = np.linalg.norm(xyz - smooth, axis=1)
    resid = resid[np.isfinite(resid)]
    noise = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) if len(resid) > 2 else a.noise
    margins = pick(a.margin_px, b.margin_px) if a.margin_px is not None and b.margin_px is not None else None
    return EntityTrack(a.id, a.phrase, sorted(set(a.instances + b.instances)), a.present | b.present,
                       a.confident | b.confident, xyz, smooth, pick(a.cells, b.cells),
                       pick(a.area_frac, b.area_frac), a.border | b.border,
                       pick(a.centroid, b.centroid), noise, a.policy,
                       a.notes + b.notes + [f"merged fragment {b.instances} into {a.instances}"], margins)


def resolve_entities(scene: Scene, groundings: Mapping[str, Grounding | None], program: Any, *,
                     identity_aliases: Mapping[str, Mapping[int, int]] | None = None) -> dict[str, list[EntityTrack]]:
    """Every non-camera entity of a program -> its track(s), by the role's policy.

    Returns ``{entity_id: [EntityTrack, ...]}``: one track for a single entity, ``count``
    tracks for a group (ids ``g1[0]``, ``g1[1]`` ...), every surviving tracklet for a
    category. An entity SAM3 could not ground maps to an empty list -- a measurement
    outcome, never a guess.
    """

    resolved: dict[str, list[EntityTrack]] = {}
    for entity in program.entities:
        if entity.kind == "camera":
            continue
        g = groundings.get(entity.phrase) if entity.phrase else None
        if g is None or g.count == 0:
            resolved[entity.id] = []
            continue
        policy = "salient" if entity.kind == "main_subject" else "best"
        ranked = rank_instances(scene, g, policy)
        ranked = [r for r in ranked if r["frames"] > 0]
        if entity.kind == "group":
            aliases = (identity_aliases or {}).get(entity.id, {})
            members: dict[int, list[int]] = {}
            for r in ranked:
                members.setdefault(aliases.get(r["k"], r["k"]), []).append(r["k"])
            tracks = [lift_instances(scene, g, indices, entity_id=f"{entity.id}[{i}]", policy="group")
                      for i, indices in enumerate(members.values())]
            tracks = merge_fragments(scene, tracks)
            tracks = [tr for tr in tracks if tr.present.sum() >= 3 and
                      (scene.t_s is None or scene.t_s[tr.last] - scene.t_s[tr.first] >= MIN_LIFESPAN_S)]
            scores = {r["k"]: r["rank_score"] for r in ranked}
            tracks.sort(key=lambda tr: -sum(scores[k] for k in tr.instances))
            if entity.count:
                tracks = tracks[:entity.count]
            for i, tr in enumerate(tracks):
                tr.id = f"{entity.id}[{i}]"
            resolved[entity.id] = tracks
        elif entity.kind == "category":
            tracks = [lift_instances(scene, g, [r["k"]], entity_id=f"{entity.id}#{r['k']}", policy="all")
                      for r in ranked]
            resolved[entity.id] = tracks
        elif entity.kind == "landmark_region":
            resolved[entity.id] = [lift_instances(scene, g, [r["k"] for r in ranked], entity_id=entity.id,
                                                  policy="region_union")] if ranked else []
        else:
            resolved[entity.id] = [lift_instances(scene, g, [ranked[0]["k"]], entity_id=entity.id,
                                                  policy=policy)] if ranked else []
    return resolved


# --- windows --------------------------------------------------------------------------- #

class WindowUnresolved(ValueError):
    """The window names a time the clip's rate cannot map, or an event not yet located."""


def resolve_window(scene: Scene, window: Window | None) -> tuple[int, int, str]:
    """Scene positions ``(lo, hi)`` (inclusive) a window covers, and how it was read."""

    T = scene.length
    if window is None or window.kind == "whole":
        return 0, T - 1, "whole clip"
    t = scene.t_s
    if window.kind == "frames":
        lo = int(np.argmin(np.abs(scene.gf_index - float(window.t0))))
        hi = int(np.argmin(np.abs(scene.gf_index - float(window.t1))))
        return min(lo, hi), max(lo, hi), f"generator frames {window.t0}..{window.t1}"
    def at(seconds: float) -> int:
        if t is None:
            raise WindowUnresolved(f"no established frame rate for {scene.clip['dataset']}: cannot map {seconds} s")
        return int(np.argmin(np.abs(t - seconds)))
    if window.kind == "instant":
        p = at(float(window.t0))
        return p, p, f"instant {window.t0} s -> scene frame {p}"
    if window.kind == "interval":
        lo, hi = at(float(window.t0)), at(float(window.t1))
        return min(lo, hi), max(lo, hi), f"[{window.t0}, {window.t1}] s -> frames {lo}..{hi}"
    if window.kind == "phase":
        if window.t0 is not None and window.t1 is not None and t is not None:
            lo, hi = at(float(window.t0)), at(float(window.t1))
            return min(lo, hi), max(lo, hi), f"{window.phase}, named as [{window.t0}, {window.t1}] s"
        fractions = {"first_25": (0, 0.25), "middle_50": (0.25, 0.75), "last_25": (0.75, 1.0),
                     "first_half": (0, 0.5), "second_half": (0.5, 1.0), "start_to_middle": (0, 0.5),
                     "end": (0.88, 1.0)}
        f0, f1 = fractions[window.phase]
        axis = scene.gf_index.astype(float)
        bounds = axis[0] + np.array([f0, f1]) * (axis[-1] - axis[0])
        lo, hi = (int(np.argmin(np.abs(axis - v))) for v in bounds)
        return lo, hi, f"{window.phase} of elapsed clip duration"
    raise WindowUnresolved(f"event window {window.event!r} needs its event located first")
