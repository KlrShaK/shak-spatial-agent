"""S3 -- turn a Scene and its entity tracks into typed measurements.

Every function returns a ``Result``: the reduced value, the noise floor it was judged
against, a confidence, and the series behind it (for the notebook's plots). Nothing here
compares against an option; that is S4 (``interpret``). A measurement that cannot be made
comes back ``available=False`` with the reason, and S4 routes the question to a fallback
rather than guessing.

All thresholds are the named constants in ``TUNING``. They are conventions of the
benchmark's generator as much as physics ("noticeable pan", "stable distance"), which is
why they are not buried in the code: the calibration study varies them on a dev split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from d4rt_agent.sam_orientany_d4rt_test import traj3d_buckets, traj3d_geometry
from dyn_dataset_exploration.decomposition.program import Measurement, QueryProgram, Window
from dyn_dataset_exploration.decomposition.scene4d import (
    EntityTrack,
    Grounding,
    Scene,
    WindowUnresolved,
    effective_fps,
    metric_scale,
    resolve_entities,
    resolve_window,
    smoothing_window,
)

TUNING: dict[str, float] = {
    # camera_motion: a motion family "happened" when it exceeds its floor
    "cam_pan_deg": 6.0,            # net or peak rotation, degrees
    "cam_tilt_deg": 6.0,
    "cam_trans_scales": 0.12,      # net translation, in scene scales (per axis family)
    "cam_complex_ratio": 0.6,      # second family >= ratio * strongest => complex
    "cam_orbit_min_scales": 0.08,  # orbit needs lateral translation AND opposing pan
    "cam_orbit_ratio": 0.6,        # an orbit needs its lateral strength >= ratio * the pan strength
    "cam_trans_cap_scales": 1e9,   # translation beyond this many scene scales is an estimate gone wrong: ignored
    "cam_rot_peak": 1,             # rotation judged by its peak excursion (1) or its net change (0)
    "cam_trans_peak": 0,           # translation likewise
    # camera_jitter: rotation residual std (deg) and translation residual (scene scales)
    "jitter_slight_deg": 0.25,
    "jitter_shaky_deg": 0.8,
    "jitter_severe_deg": 2.0,
    # range / relation trends
    "range_rel_tol": 0.15,         # the range must change by more than this fraction (of a ratio: log1p) to leave "constant"
    "region_rel_tol": 0.15,        # a distance to a landmark region: change under this fraction of the start is "constant"
    "pair_rel_tol": 0.15,          # the same, for a pair of subjects (fitted apart: the two kinds' options differ)
    "range_noise_mult": 3.0,
    "plateau_from_frac": 0.65,     # change finished before this fraction of the window: "then stop"
    # displacement words
    "axis_dominance": 0.35,        # an axis is named when >= this fraction of the largest
    "move_floor_scales": 0.03,     # ... and the largest exceeds this many scene scales
    "image_move_floor": 0.05,      # image-plane floor, fraction of the image width
    # events
    "onset_pan_deg": 4.0,
    "onset_move_scales": 0.05,
    "tie_frames": 1.0,
    # metric: multiplies metres_per_unit (1.0 = trust DA3-Nested's scale as measured)
    "metric_scale_mult": 1.0,
    # metric speed: "net" (start-to-end displacement over the time) or "path" (distance travelled,
    # in half-second steps, each step's estimation noise taken off)
    "speed_mode": "net", "speed_noise_mult": 2.0, "speed_step_s": 0.5,
    # the camera's own displacement in direction words: its axes are unambiguous (unlike a subject's)
    "cam_dir_floor_scales": 0.05, "cam_dir_dominance": 0.35,
    "speed_scale_mult": 1.0,       # multiplies metres_per_unit for speeds (fitted apart from distances)
    # zoom: relative change of the focal length that counts as a zoom
    "zoom_rel_tol": 0.10,
    # apparent size: relative change of the mask area that counts as larger/smaller
    "size_rel_tol": 0.30,
    # visibility at the end of the clip
    "vis_out_frac": 0.35, "vis_partial_border": 0.5, "vis_partial_area": 0.5,
    # camera elevation relative to the subject, degrees
    "elev_eye_deg": 10.0, "elev_change_deg": 12.0,
    # camera rhythm: per-second floors that make the camera "moving", and shape ratios
    "rhythm_trans_floor_per_s": 0.02, "rhythm_rot_floor_deg_s": 1.5,
    "rhythm_burst_ratio": 2.2, "rhythm_cv": 0.6,
    # scene revisit: share of the first view still seen (consistently) from the last camera
    "revisit_same_frac": 0.85, "revisit_diff_frac": 0.35,
    # subject motion profile
    "speed_floor_per_s": 0.02, "motion_pause_s": 0.5, "motion_continuous_frac": 0.75, "motion_once_frac": 0.3,
}


@dataclass
class Result:
    id: str
    signal: str
    reduce: str
    available: bool = True
    reason: str | None = None
    value: dict[str, Any] = field(default_factory=dict)
    floor: dict[str, float] = field(default_factory=dict)
    confidence: str = "high"                 # high | medium | low
    window: tuple[int, int] | None = None
    window_note: str = ""
    series: dict[str, Any] = field(default_factory=dict)   # plots only; not serialised
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "signal": self.signal, "reduce": self.reduce, "available": self.available,
            "reason": self.reason, "value": _jsonable(self.value), "floor": _jsonable(self.floor),
            "confidence": self.confidence, "window": list(self.window) if self.window else None,
            "window_note": self.window_note, "notes": self.notes,
        }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def unavailable(m: Measurement, reason: str) -> Result:
    return Result(m.id, m.signal, m.reduce, available=False, reason=reason, confidence="low")


# --------------------------------------------------------------------------- #
# series helpers
# --------------------------------------------------------------------------- #

def zigzag(x: np.ndarray, tol: float) -> list[int]:
    """Indices of the turning points of ``x``, first and last included.

    A reversal counts only when it exceeds ``tol`` -- hysteresis, so jitter smaller than
    the tolerance never makes a path "fluctuate".
    """

    x = np.asarray(x, dtype=float)
    idx = np.flatnonzero(np.isfinite(x))
    if len(idx) < 2:
        return [int(i) for i in idx]
    pivots = [int(idx[0])]
    direction, ext = 0, int(idx[0])          # +1 rising (hunting a peak), -1 falling
    for i in idx[1:]:
        i = int(i)
        if direction == 0:
            if x[i] - x[pivots[0]] >= tol:
                direction, ext = 1, i
            elif x[pivots[0]] - x[i] >= tol:
                direction, ext = -1, i
        elif direction == 1:
            if x[i] > x[ext]:
                ext = i
            elif x[ext] - x[i] >= tol:
                pivots.append(ext)
                direction, ext = -1, i
        else:
            if x[i] < x[ext]:
                ext = i
            elif x[i] - x[ext] >= tol:
                pivots.append(ext)
                direction, ext = 1, i
    if direction != 0 and pivots[-1] != ext:
        pivots.append(ext)
    if pivots[-1] != int(idx[-1]):
        pivots.append(int(idx[-1]))
    return pivots


def classify_trend(series: np.ndarray, tol: float, duration_frames: int | None = None) -> dict[str, Any]:
    """Shape of a distance series: constant, one way, there and back, or fluctuating.

    ``tol`` is the smallest change worth naming. The series is reduced to the turning
    points that survive hysteresis, and the shape is read off their signs -- so "closer,
    then farther" needs a real minimum in the middle, not a wiggle.
    """

    x = np.asarray(series, dtype=float)
    finite = np.flatnonzero(np.isfinite(x))
    if len(finite) < 3:
        return {"trend": None, "reason": "fewer than 3 usable frames"}
    piv = zigzag(x, tol)
    values = [float(x[i]) for i in piv]
    swings = np.diff(values)
    swings = swings[np.abs(swings) >= tol]
    n = int(len(swings))
    if n == 0:
        net = float(x[finite[-1]] - x[finite[0]])
        return {"trend": "constant", "net": net, "swings": [], "pivots": piv, "plateau_from": None}
    signs = np.sign(swings)
    # merge consecutive same-sign swings (a slope with a small wobble)
    collapsed = [signs[0]]
    for s in signs[1:]:
        if s != collapsed[-1]:
            collapsed.append(s)
    if len(collapsed) == 1:
        name = "closer" if collapsed[0] < 0 else "farther"
    elif len(collapsed) == 2:
        name = "closer_then_farther" if collapsed[0] < 0 else "farther_then_closer"
    else:
        name = "fluctuating"
    end = float(x[finite[-1]])
    span = max(finite[-1] - finite[0], 1)
    plateau = None
    if name in ("closer", "farther"):
        # first frame after which the series stays within tol of its end value
        for j in range(len(finite)):
            if np.all(np.abs(x[finite[j:]] - end) <= tol):
                plateau = float((finite[j] - finite[0]) / span)
                break
    return {"trend": name, "net": float(end - x[finite[0]]), "swings": [float(s) for s in swings],
            "pivots": piv, "plateau_from": plateau}


def valid_range(mask: np.ndarray, lo: int, hi: int) -> np.ndarray:
    out = np.zeros(len(mask), dtype=bool)
    out[lo: hi + 1] = mask[lo: hi + 1]
    return out


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #

def camera_state(scene: Scene) -> dict[str, np.ndarray]:
    """Camera translation in the levelled frame and rotation angles, per scene frame.

    Depends only on the scene (no tunable constant), so it is computed once and kept on the scene:
    a calibration re-measures the same clip hundreds of times.
    """

    cached = getattr(scene, "_camera_cache", None)
    if cached is not None:
        return cached
    angles = traj3d_buckets.camera_pose_angles(scene.cam_R, level=scene.level)
    trans = scene.to_level(scene.cam_t)              # [right, down, forward]
    window = smoothing_window(scene, scene.length)
    smooth = traj3d_geometry.smooth_trajectory(trans, window=window, polyorder=2)
    state = {"pan": angles["pan_deg"], "tilt": angles["tilt_deg"], "roll": angles["roll_deg"],
             "trans": trans, "trans_smooth": smooth}
    scene._camera_cache = state                       # noqa: SLF001 -- a cache, not part of the scene
    return state


def measure_camera_motion(scene: Scene, window: Window | None, m: Measurement) -> Result:
    try:
        lo, hi, note = resolve_window(scene, window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    if hi - lo < 2:
        return unavailable(m, "the window covers fewer than 3 frames")
    cam = camera_state(scene)
    T = TUNING
    pan = np.unwrap(np.radians(cam["pan"][lo: hi + 1])) * 180 / np.pi
    tilt = cam["tilt"][lo: hi + 1]
    pan, tilt = pan - pan[0], tilt - tilt[0]
    trans = cam["trans_smooth"][lo: hi + 1] - cam["trans_smooth"][lo]
    # Peak excursion from the window's start, signed, so a pan-and-return still registers; or the
    # net change over the window. Which one the benchmark's generator used is calibrated.
    def extreme(x):
        return float(x[np.nanargmax(np.abs(x))])
    pan_ext = extreme(pan) if T["cam_rot_peak"] else float(pan[-1])
    tilt_ext = extreme(tilt) if T["cam_rot_peak"] else float(tilt[-1])
    if T["cam_trans_peak"]:
        right, down, fwd = (extreme(trans[:, i]) for i in range(3))
    else:
        right, down, fwd = (float(trans[-1, i]) for i in range(3))
    scale = scene.scale
    reliable = float(np.linalg.norm([right, down, fwd])) <= T["cam_trans_cap_scales"] * scale
    keep = 1.0 if reliable else 0.0
    strengths = {
        "pan": abs(pan_ext) / T["cam_pan_deg"],
        "tilt": abs(tilt_ext) / T["cam_tilt_deg"],
        "dolly": keep * abs(fwd) / (T["cam_trans_scales"] * scale),
        "truck": keep * abs(right) / (T["cam_trans_scales"] * scale),
        "pedestal": keep * abs(down) / (T["cam_trans_scales"] * scale),
    }
    signed = {"pan": pan_ext, "tilt": tilt_ext, "dolly": fwd, "truck": right, "pedestal": -down}
    direction_words = {
        "pan": ("right", "left"), "tilt": ("up", "down"), "dolly": ("forward", "backward"),
        "truck": ("right", "left"), "pedestal": ("up", "down"),
    }
    active = {k: v for k, v in strengths.items() if v >= 1.0}
    motion, direction, orbit = "static", None, None
    if active:
        order = sorted(active, key=lambda k: -active[k])
        motion = order[0]
        direction = direction_words[motion][0 if signed[motion] >= 0 else 1]
        # an orbit: lateral translation with the camera turning back toward the subject
        if "truck" in active and "pan" in active and signed["truck"] * signed["pan"] < 0 \
                and abs(right) >= T["cam_orbit_min_scales"] * scale \
                and active["truck"] >= T["cam_orbit_ratio"] * active["pan"]:
            motion = "orbit"
            direction = "cw" if right < 0 else "ccw"      # translating left + panning right = clockwise
            orbit = {"lateral": right, "pan": pan_ext}
        elif len(order) > 1 and active[order[1]] >= T["cam_complex_ratio"] * active[order[0]]:
            motion, direction = "complex", None
    value = {
        "motion": motion, "direction": direction,
        "pan_deg": pan_ext, "tilt_deg": tilt_ext, "roll_deg": float(np.nanmax(np.abs(cam["roll"][lo:hi + 1] - cam["roll"][lo]))),
        "right": right, "down": down, "forward": fwd,
        "right_over_scale": right / scale, "forward_over_scale": fwd / scale, "down_over_scale": down / scale,
        "strength": {k: round(v, 3) for k, v in strengths.items()},
        "signed": {k: float(v) for k, v in signed.items()},
        "orbit": orbit,
    }
    confidence = "high"
    notes = []
    if not reliable:
        notes.append(f"the estimated translation ({np.linalg.norm([right, down, fwd]) / scale:.1f} scene scales) is implausibly large and was ignored")
        confidence = "medium"
    if scene.up_source == "camera":
        confidence = "medium"
        note += "; no ground plane found, vertical is the camera's own"
    return Result(m.id, m.signal, m.reduce, value=value, window=(lo, hi), window_note=note, notes=notes,
                  floor={"pan_deg": T["cam_pan_deg"], "tilt_deg": T["cam_tilt_deg"],
                         "translation": T["cam_trans_scales"] * scale},
                  confidence=confidence,
                  series={"pan": pan, "tilt": tilt, "trans": trans, "frames": np.arange(lo, hi + 1)})


def measure_camera_jitter(scene: Scene, window: Window | None, m: Measurement) -> Result:
    try:
        lo, hi, note = resolve_window(scene, window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    if hi - lo < 4:
        return unavailable(m, "too few frames to separate jitter from motion")
    from scipy.signal import savgol_filter

    cam = camera_state(scene)
    n = hi - lo + 1
    win = min(smoothing_window(scene, n), n if n % 2 else n - 1)
    residuals = {}
    for name in ("pan", "tilt"):
        x = np.degrees(np.unwrap(np.radians(cam[name][lo: hi + 1])))
        smooth = savgol_filter(x, win, 2) if win > 2 else x
        residuals[name] = x - smooth
    rot = float(np.sqrt(np.mean(residuals["pan"] ** 2 + residuals["tilt"] ** 2)))
    trans = cam["trans"][lo: hi + 1]
    tsm = np.stack([savgol_filter(trans[:, i], win, 2) for i in range(3)], 1) if win > 2 else trans
    trans_resid = float(np.sqrt(np.mean(np.sum((trans - tsm) ** 2, axis=1))) / scene.scale)
    T = TUNING
    level = (0 if rot < T["jitter_slight_deg"] else 1 if rot < T["jitter_shaky_deg"]
             else 2 if rot < T["jitter_severe_deg"] else 3)
    fps = scene.fps
    notes = []
    confidence = "high"
    if fps is not None and fps <= 8:
        notes.append(f"{fps:g} fps aliases handheld shake; the residual understates it")
        confidence = "medium"
    return Result(m.id, m.signal, m.reduce, value={"level": level, "rotation_residual_deg": rot,
                                                    "translation_residual_scales": trans_resid},
                  floor={"slight": T["jitter_slight_deg"], "shaky": T["jitter_shaky_deg"],
                         "severe": T["jitter_severe_deg"]},
                  confidence=confidence, window=(lo, hi), window_note=note, notes=notes,
                  series={"pan_residual": residuals["pan"], "tilt_residual": residuals["tilt"],
                          "frames": np.arange(lo, hi + 1)})


# --------------------------------------------------------------------------- #
# entities: range, displacement, presence, counts
# --------------------------------------------------------------------------- #

def _operand_track(op: str, tracks: Mapping[str, list[EntityTrack]]) -> EntityTrack | str | None:
    """'cam' -> the string 'camera'; 'g1[1]' -> that member; 'e1' -> its track; else None."""

    if op == "camera" or op == "cam":
        return "camera"
    if "[" in op:
        eid, k = op[:-1].split("[")
        lst = tracks.get(eid, [])
        return lst[int(k)] if int(k) < len(lst) else None
    lst = tracks.get(op, [])
    return lst[0] if lst else None


def _position_series(scene: Scene, operand: EntityTrack | str) -> np.ndarray:
    if isinstance(operand, str):
        return scene.cam_t
    return operand.smooth


def measure_range(scene: Scene, program: QueryProgram, m: Measurement,
                  tracks: Mapping[str, list[EntityTrack]]) -> Result:
    a, b = (_operand_track(o, tracks) for o in m.of[:2])
    if a is None or b is None:
        missing = [o for o, t in zip(m.of[:2], (a, b)) if t is None]
        return unavailable(m, f"could not ground {missing}")
    window = m.window or program.window
    try:
        lo, hi, note = resolve_window(scene, window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    pa, pb = _position_series(scene, a), _position_series(scene, b)
    rng = np.linalg.norm(pa - pb, axis=1)
    ok = np.isfinite(rng)
    for op in (a, b):
        if not isinstance(op, str):
            ok &= op.present
    rng_w = np.where(valid_range(ok, lo, hi), rng, np.nan)
    if int(np.isfinite(rng_w).sum()) < (1 if m.reduce == "value_at" else 3):
        return unavailable(m, "the entities are not both visible on enough frames of the window")
    T = TUNING
    conf = "high"
    notes: list[str] = []
    both_conf = np.ones(len(rng), dtype=bool)
    for op in (a, b):
        if not isinstance(op, str):
            both_conf &= op.confident
    if np.isfinite(rng_w).sum() and both_conf[np.isfinite(rng_w)].mean() < 0.6:
        conf = "medium"
        notes.append("many frames rest on few interior mask cells")

    if m.reduce == "value_at":
        p = lo if lo == hi else (lo + hi) // 2
        near = np.flatnonzero(np.isfinite(rng_w))
        pos = int(near[np.argmin(np.abs(near - p))])
        value_units = float(rng_w[pos])
        base, spread = metric_scale(scene)
        mpu = None if base is None else base * TUNING["metric_scale_mult"]
        val: dict[str, Any] = {"units": value_units, "frame": pos, "requested_frame": p,
                               "frame_gap": abs(pos - p)}
        if mpu is not None:
            val.update({"metres": value_units * mpu, "metres_per_unit": mpu,
                        "scale_iqr_over_median": spread})
        else:
            notes.append("no metric scale for this clip")
            conf = "low"
        if abs(pos - p) > max(1, (scene.fps or 6) * 0.5):
            notes.append(f"nearest usable frame is {abs(pos - p)} frames from the requested one")
            conf = "low" if conf == "high" else conf
        return Result(m.id, m.signal, m.reduce, value=val, confidence=conf, window=(lo, hi), window_note=note,
                      notes=notes, series={"range": rng_w, "frames": np.arange(len(rng_w))})

    # A range is judged by how many times larger or smaller it gets, not by how many units: the
    # trend is read on log(range), where "twice as far" and "half as far" are the same size of change.
    idx = np.flatnonzero(np.isfinite(rng_w) & (rng_w > 0))
    if len(idx) < 3:
        return unavailable(m, "the entities are not both visible on enough frames of the window")
    win = smoothing_window(scene, len(idx))
    log_r = np.full_like(rng_w, np.nan)
    log_r[idx] = np.log(rng_w[idx])
    smooth_log = np.full_like(rng_w, np.nan)
    smooth_log[idx] = traj3d_geometry.smooth_trajectory(np.stack([log_r[idx]] * 3, 1), window=win, polyorder=2)[:, 0]
    resid = log_r[idx] - smooth_log[idx]
    noise_log = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) if len(resid) > 2 else 0.0
    rel_tol = T["pair_rel_tol"] if program.kind == "pair_relation" else T["range_rel_tol"]
    tol_log = max(float(np.log1p(rel_tol)), T["range_noise_mult"] * noise_log)
    shape = classify_trend(smooth_log, tol_log)
    if shape["trend"] is None:
        return unavailable(m, shape["reason"])
    smooth = np.exp(smooth_log)
    start = float(np.nanmedian(smooth[idx[: max(1, len(idx) // 5)]]))
    end = float(smooth[idx[-1]])
    tol = start * (float(np.exp(tol_log)) - 1.0)
    return Result(m.id, m.signal, m.reduce,
                  value={"trend": shape["trend"], "net_units": end - float(smooth[idx[0]]), "start_units": start,
                         "end_units": end, "tolerance_units": tol, "ratio_end_over_start": end / start,
                         "plateau_from": shape.get("plateau_from"), "pivots": shape["pivots"],
                         "frames_used": int(len(idx))},
                  floor={"tolerance_units": tol, "noise_log": noise_log}, confidence=conf, window=(lo, hi),
                  window_note=note, notes=notes,
                  series={"range": rng_w, "smooth": smooth, "tol": tol, "frames": np.arange(len(rng_w))})


def displacement_frames(scene: Scene, track: EntityTrack, lo_pos: int, hi_pos: int) -> dict[str, np.ndarray]:
    """One entity's displacement between two scene frames in every reference frame.

    Components are ``[right, down, forward]``. The frames differ in what they hold still:
    ``world0`` the first camera's axes; ``levelled0`` the same with the vertical along
    gravity; ``camera_end`` the camera at the end of the window; ``camera_rel`` the entity's
    position as the moving camera sees it (so camera motion is *not* cancelled); ``image``
    the centroid's motion in the picture, with depth from ``camera_rel``.
    """

    p0, p1 = track.smooth[lo_pos], track.smooth[hi_pos]
    d = p1 - p0
    local0 = scene.to_camera(p0, lo_pos)
    local1 = scene.to_camera(p1, hi_pos)
    c0, c1 = track.centroid[lo_pos], track.centroid[hi_pos]
    return {
        "world0": d,
        "levelled0": scene.to_level(d),
        "camera_end": scene.cam_R[hi_pos].T @ d,
        "camera_rel": local1 - local0,
        "image": np.array([c1[0] - c0[0], c1[1] - c0[1], (local1 - local0)[2]]),
    }


def measure_displacement(scene: Scene, program: QueryProgram, m: Measurement,
                         tracks: Mapping[str, list[EntityTrack]]) -> Result:
    track = _operand_track(m.of[0], tracks)
    if track is None or isinstance(track, str):
        return unavailable(m, f"could not ground {m.of[0]!r}")
    try:
        lo, hi, note = resolve_window(scene, m.window or program.window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    idx = np.flatnonzero(track.present & np.isfinite(track.smooth).all(1))
    idx = idx[(idx >= lo) & (idx <= hi)]
    if len(idx) < 2:
        return unavailable(m, "the entity is visible on fewer than 2 frames of the window")
    a, b = int(idx[0]), int(idx[-1])
    frames = displacement_frames(scene, track, a, b)
    notes = []
    if (a - lo) > 0.25 * max(hi - lo, 1) or (hi - b) > 0.25 * max(hi - lo, 1):
        notes.append(f"visible only on frames {a}..{b} of the window {lo}..{hi}")
    return Result(m.id, m.signal, m.reduce, window=(a, b), window_note=note, notes=notes,
                  confidence="high" if track.confident[[a, b]].all() else "medium",
                  value={"frames": {k: [float(x) for x in v] for k, v in frames.items()},
                         "start": a, "end": b, "noise": track.noise,
                         "scene_scale": scene.scale},
                  floor={"noise_units": track.noise, "move_floor": TUNING["move_floor_scales"] * scene.scale},
                  series={"xyz": track.xyz, "smooth": track.smooth, "frames": np.arange(scene.length)})


def measure_instances(scene: Scene, m: Measurement, tracks: Mapping[str, list[EntityTrack]],
                      groundings: Mapping[str, Grounding | None], program: QueryProgram) -> Result:
    eid = m.of[0]
    lst = tracks.get(eid)
    if lst is None or not groundings.get(next((e.phrase for e in program.entities if e.id == eid), None)):
        return unavailable(m, "the category was not found by the segmenter")
    fps = effective_fps(scene)
    counts = {}
    for label, seconds in (("0.3s", 0.3), ("0.5s", 0.5), ("1.0s", 1.0)):
        need = max(3, int(round(seconds * fps)))
        counts[label] = sum(1 for t in lst if int(t.present.sum()) >= need) if lst else 0
    stable = len(set(counts.values())) == 1
    return Result(m.id, m.signal, m.reduce, value={
        "count": len(lst), "by_min_lifespan": counts, "stable": stable,
        "lifespans": [[t.first, t.last, int(t.present.sum())] for t in lst],
    }, confidence="high" if stable else "low",
        notes=[] if stable else ["the count changes with the minimum lifespan"])


def region_points(scene: Scene, g: Grounding, instances: Sequence[int], pos: int) -> np.ndarray:
    """World points of the depth-cloud cells a region's mask covers on one frame."""

    from scipy.ndimage import binary_erosion
    from dyn_dataset_exploration.decomposition.scene4d import _cell_coverage, MIN_CORE_CELLS

    gf = int(scene.gf_index[pos])
    mask = np.zeros(g.masks.shape[2:], dtype=bool)
    for k in instances:
        mask |= g.masks[k, gf]
    if not mask.any():
        return np.zeros((0, 3))
    inside = _cell_coverage(mask, scene.points.shape[1:3]) >= 0.5
    core = binary_erosion(inside, structure=np.ones((3, 3), dtype=bool))
    if core.sum() < MIN_CORE_CELLS:
        return np.zeros((0, 3))
    pts = scene.points[pos][core]
    return pts[np.isfinite(pts).all(1)]


def measure_region_distance(scene: Scene, program: QueryProgram, m: Measurement,
                            tracks: Mapping[str, list[EntityTrack]],
                            groundings: Mapping[str, Grounding | None]) -> Result:
    subject = _operand_track(m.of[0], tracks)
    region_entity = next((e for e in program.entities if e.id == m.of[1]), None)
    g = groundings.get(region_entity.phrase) if region_entity else None
    if subject is None or isinstance(subject, str) or g is None or g.count == 0:
        return unavailable(m, "the subject or the landmark region could not be grounded")
    try:
        lo, hi, note = resolve_window(scene, m.window or program.window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    instances = [r for r in range(g.count)]
    dist = np.full(scene.length, np.nan)
    for pos in range(lo, hi + 1):
        if not subject.present[pos] or not np.isfinite(subject.smooth[pos]).all():
            continue
        pts = region_points(scene, g, instances, pos)
        if len(pts):
            dist[pos] = float(np.min(np.linalg.norm(pts - subject.smooth[pos], axis=1)))
    idx = np.flatnonzero(np.isfinite(dist))
    if len(idx) < 3:
        return unavailable(m, "the subject and the region are not both visible on 3 frames")
    smooth = np.full_like(dist, np.nan)
    smooth[idx] = traj3d_geometry.smooth_trajectory(np.stack([dist[idx]] * 3, 1),
                                                    window=smoothing_window(scene, len(idx)), polyorder=2)[:, 0]
    resid = dist[idx] - smooth[idx]
    noise = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) if len(resid) > 2 else 0.0
    start = float(np.nanmedian(smooth[idx[: max(1, len(idx) // 5)]]))
    T = TUNING
    # linear, unlike measure_range: a distance to a surface can reach zero, where a ratio is meaningless
    tol = max(T["region_rel_tol"] * max(start, 0.05 * scene.scale), T["range_noise_mult"] * noise)
    shape = classify_trend(smooth, tol)
    return Result(m.id, m.signal, m.reduce,
                  value={"trend": shape["trend"], "net_units": shape.get("net"), "start_units": start,
                         "end_units": float(smooth[idx[-1]]), "tolerance_units": tol, "plateau_from": shape.get("plateau_from"),
                         "frames_used": int(len(idx))},
                  floor={"tolerance_units": tol}, confidence="medium",
                  notes=["distance to the nearest surface point of the region, not to a centroid"],
                  window=(lo, hi), window_note=note,
                  series={"range": dist, "smooth": smooth, "tol": tol, "frames": np.arange(scene.length)})


# --------------------------------------------------------------------------- #
# events and order
# --------------------------------------------------------------------------- #

def _onset_of_swing(x: np.ndarray, sign: int, threshold: float) -> int | None:
    """Frame where the swing that first exceeds ``threshold`` in direction ``sign`` began.

    "Began" is the *last* frame the signal was still at its running minimum (within a small
    tolerance): a camera that sits still for 15 frames and then pans starts panning at 15,
    not at 0.
    """

    x = np.asarray(x, dtype=float) * sign
    eps = 0.15 * threshold
    running_min = np.inf
    start = 0
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue
        if v <= running_min + eps:
            start = i
            running_min = min(running_min, v)
        if v - running_min >= threshold:
            return int(start)
    return None


def measure_camera_displacement(scene: Scene, window: Window | None, m: Measurement) -> Result:
    """Where the camera went between two moments, as ``[right, down, forward]`` in the gravity-levelled
    frame of the first camera. Unlike a subject's displacement there is nothing to choose: the
    camera's own axes are what "rightward and forward" means."""

    try:
        lo, hi, note = resolve_window(scene, window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    if hi - lo < 2:
        return unavailable(m, "the window covers fewer than 3 frames")
    cam = camera_state(scene)
    d = cam["trans_smooth"][hi] - cam["trans_smooth"][lo]
    resid = np.linalg.norm(cam["trans"][lo: hi + 1] - cam["trans_smooth"][lo: hi + 1], axis=1)
    noise = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) if len(resid) > 2 else 0.0
    comp = [float(x) for x in d]
    return Result(m.id, m.signal, m.reduce, window=(lo, hi), window_note=note,
                  value={"frames": {name: comp for name in ("world0", "levelled0", "camera_end", "camera_rel", "image")},
                         "direct": True, "start": lo, "end": hi, "noise": noise, "scene_scale": scene.scale},
                  floor={"move_floor": TUNING["cam_dir_floor_scales"] * scene.scale, "noise_units": noise},
                  series={"trans": cam["trans_smooth"], "frames": np.arange(scene.length)})


def measure_metric_speed(scene: Scene, program: QueryProgram, m: Measurement,
                         tracks: Mapping[str, list[EntityTrack]]) -> Result:
    """An entity's average speed in the world, over the frames of the window where it is seen.

    The track is in the world frame (the first camera's axes), so the camera's own motion is
    not counted as the subject's. Two readings of "average speed" are made -- displacement from
    start to end over the time, and the distance travelled -- and calibration picks which one the
    benchmark's answers follow (``speed_mode``).
    """

    track = _single_track(m, tracks)
    if track is None:
        return unavailable(m, f"could not ground {m.of[0]!r}")
    if scene.t_s is None:
        return unavailable(m, f"no established frame rate for {scene.clip['dataset']}: cannot turn frames into seconds")
    try:
        lo, hi, note = resolve_window(scene, m.window or program.window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    idx = np.flatnonzero(track.present & np.isfinite(track.smooth).all(1))
    idx = idx[(idx >= lo) & (idx <= hi)]
    if len(idx) < 4:
        return unavailable(m, "the entity is visible on fewer than 4 frames of the window")
    T = TUNING
    t = scene.t_s[idx].astype(float)
    pos = track.smooth[idx]
    duration = float(t[-1] - t[0])
    if duration <= 0:
        return unavailable(m, "the entity is seen for no time")
    k = max(1, int(round(0.15 * len(idx))))
    head, tail = np.median(pos[:k], axis=0), np.median(pos[-k:], axis=0)
    net_dt = float(np.median(t[-k:]) - np.median(t[:k]))
    net = float(np.linalg.norm(tail - head)) / max(net_dt, 1e-6)
    # the path in steps of ~speed_step_s: per-frame steps would turn estimation jitter into speed
    grid = np.arange(t[0], t[-1] + 1e-9, T["speed_step_s"])
    picks = np.unique(np.abs(t[None, :] - grid[:, None]).argmin(1))
    steps = np.linalg.norm(np.diff(pos[picks], axis=0), axis=1)
    floor = T["speed_noise_mult"] * track.noise
    span = float(t[picks[-1]] - t[picks[0]]) if len(picks) > 1 else 0.0
    path = float(np.maximum(steps - floor, 0.0).sum()) / span if span > 0 else 0.0
    ups = net if T["speed_mode"] == "net" else path
    base, spread = metric_scale(scene)
    mpu = None if base is None else base * T["speed_scale_mult"]
    value: dict[str, Any] = {"units_per_s": ups, "net_units_per_s": net, "path_units_per_s": path,
                             "mode": T["speed_mode"], "duration_s": duration, "frames_used": int(len(idx))}
    notes, conf = [], "high" if track.confident[idx].mean() >= 0.6 else "medium"
    if mpu is None:
        notes.append("no metric scale for this clip")
        conf = "low"
    else:
        value.update({"mps": ups * mpu, "net_mps": net * mpu, "path_mps": path * mpu, "metres_per_unit": mpu,
                      "scale_iqr_over_median": spread})
    if (idx[0] - lo) > 0.25 * max(hi - lo, 1) or (hi - idx[-1]) > 0.25 * max(hi - lo, 1):
        notes.append(f"visible only on frames {idx[0]}..{idx[-1]} of the window {lo}..{hi}")
    return Result(m.id, m.signal, m.reduce, value=value, confidence=conf, window=(int(idx[0]), int(idx[-1])),
                  window_note=note, notes=notes, floor={"noise_units": track.noise},
                  series={"t": t, "pos": pos, "picks": picks, "frames": idx})


def measure_event_onset(scene: Scene, program: QueryProgram, m: Measurement,
                        tracks: Mapping[str, list[EntityTrack]]) -> Result:
    ev = next((e for e in program.events if e.id == m.of[0]), None)
    if ev is None:
        return unavailable(m, f"unknown event {m.of[0]!r}")
    T = TUNING
    frames = None
    onset: int | None = None
    notes: list[str] = []
    detail: dict[str, Any] = {"event": ev.id, "predicate": ev.predicate, "actor": ev.actor}
    conf = "high"
    if ev.actor in ("cam", "camera"):
        cam = camera_state(scene)
        if ev.predicate in ("pan_left", "pan_right", "pan_any"):
            pan = np.degrees(np.unwrap(np.radians(cam["pan"])))
            if ev.predicate == "pan_any":
                hits = [o for o in (_onset_of_swing(pan, 1, T["onset_pan_deg"]), _onset_of_swing(pan, -1, T["onset_pan_deg"]))
                        if o is not None]
                onset = min(hits) if hits else None
            else:
                onset = _onset_of_swing(pan, 1 if ev.predicate == "pan_right" else -1, T["onset_pan_deg"])
            frames = pan
            detail["threshold"] = f"{T['onset_pan_deg']} deg swing"
        elif ev.predicate in ("tilt_up", "tilt_down", "tilt_any"):
            tilt = cam["tilt"]
            if ev.predicate == "tilt_any":
                hits = [o for o in (_onset_of_swing(tilt, 1, T["onset_pan_deg"]), _onset_of_swing(tilt, -1, T["onset_pan_deg"]))
                        if o is not None]
                onset = min(hits) if hits else None
            else:
                onset = _onset_of_swing(tilt, 1 if ev.predicate == "tilt_up" else -1, T["onset_pan_deg"])
            frames = tilt
        elif ev.predicate == "starts_moving":
            trans = cam["trans_smooth"] - cam["trans_smooth"][0]
            speed = np.linalg.norm(trans, axis=1)
            hit = np.flatnonzero(speed >= T["onset_move_scales"] * scene.scale)
            onset = int(hit[0]) if len(hit) else None
            frames = speed
        else:
            return unavailable(m, f"no signal defined for camera event {ev.predicate!r}")
    else:
        lst = tracks.get(ev.actor)
        track = lst[0] if lst else None
        if track is None:
            return unavailable(m, f"could not ground {ev.actor!r}")
        pres = track.present
        if ev.predicate in ("enters_frame", "becomes_visible"):
            first = track.first
            if first is None:
                onset = None
            else:
                onset = first
                detail["present_from_first_frame"] = first == 0
                if first == 0:
                    notes.append("already in view on the first frame")
                    conf = "medium"
        elif ev.predicate == "leaves_frame":
            last = track.last
            onset = None if last is None or last >= scene.length - 1 else last + 1
        elif ev.predicate == "starts_moving":
            idx = np.flatnonzero(pres & np.isfinite(track.smooth).all(1))
            if len(idx) >= 3:
                disp = np.linalg.norm(track.smooth[idx] - track.smooth[idx[0]], axis=1)
                hit = np.flatnonzero(disp >= max(T["onset_move_scales"] * scene.scale, 3 * track.noise))
                onset = int(idx[hit[0]]) if len(hit) else None
                frames = np.where(pres, np.linalg.norm(track.smooth - track.smooth[idx[0]], axis=1), np.nan)
        else:
            return unavailable(m, f"no signal defined for {ev.predicate!r}")
    t = None if onset is None else (float(scene.t_s[onset]) if scene.t_s is not None else None)
    return Result(m.id, m.signal, m.reduce, value={"onset": onset, "onset_s": t, **detail},
                  confidence=conf if onset is not None else "medium", notes=notes,
                  series={"signal": frames, "onset": onset})


def measure_order(scene: Scene, m: Measurement, results: Mapping[str, Result]) -> Result:
    a, b = (results.get(o) for o in m.of[:2])
    if a is None or b is None or not a.available or not b.available:
        return unavailable(m, "an event onset could not be measured")
    oa, ob = a.value.get("onset"), b.value.get("onset")
    ea, eb = a.value["event"], b.value["event"]
    tie = TUNING["tie_frames"]
    if oa is None and ob is None:
        first = "none"
    elif ob is None:
        first = "one_only:" + ea
    elif oa is None:
        first = "one_only:" + eb
    elif abs(oa - ob) <= tie:
        first = "tie"
    else:
        first = ea if oa < ob else eb
    conf = "high" if min(a.confidence, b.confidence) == "high" else "medium"
    gap = None if oa is None or ob is None else abs(oa - ob)
    if gap is not None and gap <= 2 * tie:
        conf = "low"
    return Result(m.id, m.signal, m.reduce, value={"first": first, "onsets": [oa, ob], "gap_frames": gap},
                  confidence=conf, floor={"tie_frames": tie})


# --------------------------------------------------------------------------- #
# camera: zoom, rhythm, revisit
# --------------------------------------------------------------------------- #

def _smooth1d(x: np.ndarray, window: int) -> np.ndarray:
    from scipy.signal import savgol_filter

    x = np.asarray(x, dtype=float)
    window = min(window, len(x) if len(x) % 2 else len(x) - 1)
    if window < 3 or len(x) < 5:
        return x.copy()
    return savgol_filter(x, window, min(2, window - 1))


SPEED_WINDOW_S = 0.5


def _median_smooth(x: np.ndarray, window: int) -> np.ndarray:
    """A running median: unlike Savitzky-Golay it does not ring around a step, and a
    one-or-two-frame spike (Point4D's last frames) is removed instead of smeared."""

    from scipy.signal import medfilt

    x = np.asarray(x, dtype=float)
    window = min(window, len(x) if len(x) % 2 else len(x) - 1)
    if window < 3:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    return medfilt(padded, window)[pad: pad + len(x)]


def windowed_speed(positions: np.ndarray, frames: np.ndarray, fps: float,
                   seconds: float = SPEED_WINDOW_S) -> tuple[np.ndarray, np.ndarray]:
    """Speed as displacement over about ``seconds``, per second, for each usable frame.

    ``positions[i]`` is at frame ``frames[i]``. A per-frame step would turn a pose estimate's
    frame-to-frame jitter into speed (0.1 deg/frame is 3 deg/s at 30 fps); a displacement over
    half a second lets jitter cancel and makes a threshold mean the same at 6 and at 30 fps.
    Returns the speed and the frame it is assigned to (the window's midpoint).
    """

    frames = np.asarray(frames)
    need = max(1, int(round(seconds * fps)))
    speeds, at = [], []
    for i in range(len(frames)):
        j = int(np.searchsorted(frames, frames[i] + need))
        if j >= len(frames):
            break
        dt = frames[j] - frames[i]
        speeds.append(np.linalg.norm(positions[j] - positions[i]) * fps / dt)
        at.append((frames[i] + frames[j]) / 2)
    return np.asarray(speeds), np.asarray(at)


def measure_zoom(scene: Scene, window: Window | None, m: Measurement) -> Result:
    try:
        lo, hi, note = resolve_window(scene, window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    if hi - lo < 3:
        return unavailable(m, "the window covers fewer than 4 frames")
    fx = scene.K[lo: hi + 1, 0, 0]
    x = _median_smooth(np.log(fx / fx[0]), max(3, smoothing_window(scene, hi - lo + 1)))
    tol = float(np.log1p(TUNING["zoom_rel_tol"]))
    piv = zigzag(x, tol)
    swings = np.diff([x[i] for i in piv])
    swings = swings[np.abs(swings) >= tol]
    signs = [int(np.sign(s)) for s in swings]
    collapsed = [s for i, s in enumerate(signs) if i == 0 or s != signs[i - 1]]
    zoom = "none" if not collapsed else ("in" if collapsed[0] > 0 else "out") if len(collapsed) == 1 else "mixed"
    cam = camera_state(scene)
    fwd = float(cam["trans_smooth"][hi, 2] - cam["trans_smooth"][lo, 2]) / scene.scale
    notes = []
    conf = "high"
    if zoom != "none" and abs(fwd) > 0.3:
        notes.append(f"the camera also moved {fwd:+.2f} scales forward: a dolly can be read as a focal change")
        conf = "medium"
    return Result(m.id, m.signal, m.reduce,
                  value={"zoom": zoom, "focal_ratio_end": float(fx[-1] / fx[0]), "focal_ratio_min": float(fx.min() / fx[0]),
                         "focal_ratio_max": float(fx.max() / fx[0]), "forward_over_scale": fwd},
                  floor={"log_focal": tol}, confidence=conf, window=(lo, hi), window_note=note, notes=notes,
                  series={"focal": fx, "smooth_log": x, "frames": np.arange(lo, hi + 1)})


def measure_rhythm(scene: Scene, window: Window | None, m: Measurement) -> Result:
    try:
        lo, hi, note = resolve_window(scene, window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    if hi - lo < 5:
        return unavailable(m, "too few frames for a speed profile")
    T = TUNING
    cam = camera_state(scene)
    fps = effective_fps(scene)
    win = smoothing_window(scene, hi - lo + 1)
    frames = np.arange(lo, hi + 1)
    trans = np.stack([_smooth1d(cam["trans"][lo: hi + 1, i], win) for i in range(3)], 1)
    pan = _smooth1d(np.degrees(np.unwrap(np.radians(cam["pan"][lo: hi + 1]))), win)
    tilt = _smooth1d(cam["tilt"][lo: hi + 1], win)
    trans_speed, at = windowed_speed(trans, frames, fps)
    rot_speed, _ = windowed_speed(np.stack([pan, tilt, np.zeros_like(pan)], 1), frames, fps)
    if len(at) < 3:
        return unavailable(m, "too few frames for a speed profile")
    trans_speed = trans_speed / scene.scale
    e = trans_speed / T["rhythm_trans_floor_per_s"] + rot_speed / T["rhythm_rot_floor_deg_s"]
    e = _smooth1d(e, win)
    thirds = np.array_split(e, 3)
    m1, m2, m3 = (float(np.mean(t)) for t in thirds)
    mean_e = float(np.mean(e))
    cv = float(np.std(e) / mean_e) if mean_e > 1e-9 else 0.0
    jitter = measure_camera_jitter(scene, window, m)
    shaky = jitter.available and jitter.value["rotation_residual_deg"] >= T["jitter_shaky_deg"]
    burst = T["rhythm_burst_ratio"]
    if float(e.max()) < 1.0 or mean_e < 0.5:
        rhythm = "shaky" if shaky else "fixed"
    elif m1 < 1.0 and max(m2, m3) >= 1.0:
        rhythm = "still_then_burst"
    elif m1 >= 1.0 and m3 < 1.0 and m3 < 0.4 * m1:
        rhythm = "moving_then_stop"
    elif m1 >= 1.0 and m3 > burst * m1:
        rhythm = "speeds_up"
    elif cv <= T["rhythm_cv"]:
        rhythm = "steady"
    else:
        rhythm = "irregular"
    notes = [f"{fps:g} fps: the profile is coarse"] if fps <= 8 else []
    return Result(m.id, m.signal, m.reduce,
                  value={"rhythm": rhythm, "thirds": [m1, m2, m3], "mean": mean_e, "cv": cv, "max": float(e.max()),
                         "shaky": bool(shaky)},
                  floor={"trans_per_s": T["rhythm_trans_floor_per_s"], "rot_deg_s": T["rhythm_rot_floor_deg_s"]},
                  confidence="medium" if fps <= 8 else "high", window=(lo, hi), window_note=note, notes=notes,
                  series={"energy": e, "frames": at})


def measure_revisit(scene: Scene, window: Window | None, m: Measurement) -> Result:
    try:
        lo, hi, note = resolve_window(scene, window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    if hi - lo < 2:
        return unavailable(m, "the window covers fewer than 3 frames")
    cloud = scene.points[lo].reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(1)]
    gh, gw = scene.points.shape[1:3]
    R, c, K = scene.cam_R[hi], scene.cam_t[hi], scene.K[hi]
    pc = (cloud - c) @ R
    z = pc[:, 2]
    front = z > 1e-6
    u = K[0, 0] * pc[:, 0] / np.where(front, z, 1) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.where(front, z, 1) + K[1, 2]
    H, W = scene.model_hw
    inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    last = scene.points[hi]
    last_depth = ((last - c) @ R)[..., 2]
    gi = np.clip((v / scene.stride).astype(int), 0, gh - 1)
    gj = np.clip((u / scene.stride).astype(int), 0, gw - 1)
    seen = last_depth[gi, gj]
    consistent = inside & np.isfinite(seen) & (np.abs(seen - z) <= 0.15 * np.maximum(z, 1e-6))
    frac = float(consistent.mean())
    T = TUNING
    verdict = "same" if frac >= T["revisit_same_frac"] else "slight" if frac >= T["revisit_diff_frac"] else "different"
    cam = camera_state(scene)
    return Result(m.id, m.signal, m.reduce,
                  value={"revisit": verdict, "shared_fraction": frac, "in_view_fraction": float(inside.mean()),
                         "pan_deg": float(cam["pan"][hi] - cam["pan"][lo]),
                         "translation_over_scale": float(np.linalg.norm(cam["trans_smooth"][hi] - cam["trans_smooth"][lo]) / scene.scale)},
                  floor={"same": T["revisit_same_frac"], "different": T["revisit_diff_frac"]},
                  window=(lo, hi), window_note=note)


# --------------------------------------------------------------------------- #
# entities: apparent size, visibility, elevation, motion profile
# --------------------------------------------------------------------------- #

def _single_track(m: Measurement, tracks: Mapping[str, list[EntityTrack]]) -> EntityTrack | None:
    lst = tracks.get(m.of[0])
    return lst[0] if lst else None


def measure_apparent_size(scene: Scene, program: QueryProgram, m: Measurement,
                          tracks: Mapping[str, list[EntityTrack]]) -> Result:
    track = _single_track(m, tracks)
    if track is None:
        return unavailable(m, f"could not ground {m.of[0]!r}")
    try:
        lo, hi, note = resolve_window(scene, m.window or program.window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    inwin = np.zeros(scene.length, dtype=bool)
    inwin[lo: hi + 1] = True
    ok = track.present & (track.area_frac > 0) & inwin
    interior = ok & ~track.border
    notes = []
    use = interior if interior.sum() >= 5 else ok
    if interior.sum() >= 5 and interior.sum() < ok.sum():
        notes.append(f"{int(ok.sum() - interior.sum())} frames touching the image border were left out "
                     "(a cut-off mask is not a smaller object)")
    idx = np.flatnonzero(use)
    if len(idx) < 4:
        return unavailable(m, "the subject is visible on fewer than 4 frames of the window")
    x = np.full(scene.length, np.nan)
    x[idx] = _smooth1d(-np.log(track.area_frac[idx]), smoothing_window(scene, len(idx)))
    tol = float(np.log1p(TUNING["size_rel_tol"]))
    shape = classify_trend(x, tol)
    name = {"closer": "larger", "farther": "smaller", "constant": "unchanged"}.get(shape["trend"], "fluctuating")
    ratio_end = float(np.exp(-(x[idx[-1]] - x[idx[0]])))
    return Result(m.id, m.signal, m.reduce,
                  value={"size": name, "trend": shape["trend"], "area_ratio_end": ratio_end,
                         "area_start": float(track.area_frac[idx[0]]), "area_end": float(track.area_frac[idx[-1]]),
                         "frames_used": int(len(idx))},
                  floor={"log_area": tol}, confidence="high" if len(idx) >= 8 else "medium", window=(lo, hi),
                  window_note=note, notes=notes,
                  series={"area": np.where(use, track.area_frac, np.nan), "frames": np.arange(scene.length)})


def measure_visibility(scene: Scene, program: QueryProgram, m: Measurement,
                       tracks: Mapping[str, list[EntityTrack]]) -> Result:
    track = _single_track(m, tracks)
    if track is None:
        return unavailable(m, f"could not ground {m.of[0]!r}")
    try:
        lo, hi, note = resolve_window(scene, m.window or program.window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    T = TUNING
    seg = slice(lo, hi + 1)
    present = track.present[seg]
    present_frac = float(present.mean())
    pres_all = track.present & (track.area_frac > 0)
    life_area = float(np.median(track.area_frac[pres_all])) if pres_all.any() else 0.0
    if present_frac < T["vis_out_frac"]:
        verdict, border_frac, rel_area = "out", None, None
    else:
        border_frac = float(track.border[seg][present].mean())
        rel_area = float(np.median(track.area_frac[seg][present]) / life_area) if life_area > 0 else 1.0
        verdict = "partial" if (border_frac >= T["vis_partial_border"] or rel_area < T["vis_partial_area"]) else "full"
    return Result(m.id, m.signal, m.reduce,
                  value={"visible": verdict, "present_fraction": present_frac, "border_fraction": border_frac,
                         "relative_area": rel_area},
                  floor={"out": T["vis_out_frac"], "border": T["vis_partial_border"], "area": T["vis_partial_area"]},
                  window=(lo, hi), window_note=note,
                  series={"present": track.present.astype(float), "frames": np.arange(scene.length)})


def measure_elevation(scene: Scene, program: QueryProgram, m: Measurement,
                      tracks: Mapping[str, list[EntityTrack]]) -> Result:
    track = _single_track(m, tracks)
    if track is None:
        return unavailable(m, f"could not ground {m.of[0]!r}")
    try:
        lo, hi, note = resolve_window(scene, m.window or program.window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    ok = track.present & np.isfinite(track.smooth).all(1)
    ok[:lo] = False
    ok[hi + 1:] = False
    if ok.sum() < 3:
        return unavailable(m, "the subject is visible on fewer than 3 frames of the window")
    v = scene.to_level(scene.cam_t - track.smooth)          # camera minus subject, [right, down, forward]
    horizontal = np.hypot(v[:, 0], v[:, 2])
    elev = np.degrees(np.arctan2(-v[:, 1], np.maximum(horizontal, 1e-9)))
    elev = np.where(ok, elev, np.nan)
    T = TUNING
    eye = T["elev_eye_deg"]
    cat = np.where(elev > eye, 1, np.where(elev < -eye, -1, 0)).astype(float)
    cat[~ok] = np.nan
    fracs = {name: float(np.nanmean(cat == val)) for name, val in (("high", 1), ("eye", 0), ("low", -1))}
    med = float(np.nanmedian(elev))
    big = [k for k, f in fracs.items() if f >= 0.25]
    if len(big) >= 2 and (np.nanmax(elev) - np.nanmin(elev)) >= T["elev_change_deg"]:
        verdict = "changing"
    else:
        verdict = "high" if med > eye else "low" if med < -eye else "eye"
    notes, conf = [], "high"
    if scene.up_source != "plane":
        notes.append("no ground plane: gravity is the first camera's vertical")
        conf = "low"
    elif scene.up_fraction < 0.3:
        notes.append(f"the ground plane carries only {scene.up_fraction:.0%} of the cloud")
        conf = "medium"
    return Result(m.id, m.signal, m.reduce,
                  value={"angle": verdict, "median_deg": med, "min_deg": float(np.nanmin(elev)),
                         "max_deg": float(np.nanmax(elev)), "fractions": fracs},
                  floor={"eye_deg": eye}, confidence=conf, window=(lo, hi), window_note=note, notes=notes,
                  series={"elevation": elev, "frames": np.arange(scene.length)})


def measure_motion_profile(scene: Scene, program: QueryProgram, m: Measurement,
                           tracks: Mapping[str, list[EntityTrack]]) -> Result:
    track = _single_track(m, tracks)
    if track is None:
        return unavailable(m, f"could not ground {m.of[0]!r}")
    try:
        lo, hi, note = resolve_window(scene, m.window or program.window)
    except WindowUnresolved as error:
        return unavailable(m, str(error))
    ok = track.present & np.isfinite(track.smooth).all(1)
    ok[:lo] = False
    ok[hi + 1:] = False
    idx = np.flatnonzero(ok)
    if len(idx) < 5:
        return unavailable(m, "the subject is visible on fewer than 5 frames")
    T = TUNING
    fps = effective_fps(scene)
    speed, at = windowed_speed(track.smooth[idx], idx, fps)
    if len(speed) < 3:
        return unavailable(m, "the subject is visible over too short a span to measure a speed")
    speed = speed / scene.scale                              # scales per second
    floor = max(T["speed_floor_per_s"], 3 * track.noise * fps / scene.scale / max(1, int(round(SPEED_WINDOW_S * fps))))
    moving = speed >= floor
    if len(moving) >= 3:
        padded = np.pad(moving.astype(int), 1, mode="edge")
        moving = (padded[:-2] + padded[1:-1] + padded[2:]) >= 2         # 3-frame majority
    frac = float(moving.mean())
    runs, i = [], 0
    while i < len(moving):
        if moving[i]:
            j = i
            while j + 1 < len(moving) and moving[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    min_pause = max(1, int(round(T["motion_pause_s"] * fps)))
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= min_pause:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if frac < 0.1:
        verdict = "none"
    elif len(merged) == 1 and (merged[0][1] - merged[0][0] + 1) / len(moving) < T["motion_once_frac"]:
        verdict = "once"
    elif frac >= T["motion_continuous_frac"] and len(merged) <= 1:
        verdict = "continuous"
    elif len(merged) >= 2:
        verdict = "intermittent"
    else:
        verdict = "continuous" if frac >= 0.5 else "intermittent"
    return Result(m.id, m.signal, m.reduce,
                  value={"motion": verdict, "moving_fraction": frac, "runs": len(merged),
                         "floor_per_s": floor, "present_fraction": float(track.present.mean())},
                  floor={"speed_per_s": floor}, confidence="high" if track.confident[idx].mean() > 0.6 else "medium",
                  window=(lo, hi), window_note=note,
                  series={"speed": speed, "moving": moving, "frames": at})


# --------------------------------------------------------------------------- #
# the executor: a program's measurements, in order
# --------------------------------------------------------------------------- #

def run_program(scene: Scene, program: QueryProgram, groundings: Mapping[str, Grounding | None],
                tracks: dict[str, list[EntityTrack]] | None = None
                ) -> tuple[dict[str, Result], dict[str, list[EntityTrack]]]:
    """Resolve entities (unless already resolved), then run every measurement in order."""

    if tracks is None:
        tracks = resolve_entities(scene, groundings, program)
    results: dict[str, Result] = {}
    for m in program.measurements:
        if m.signal == "camera_motion":
            r = measure_camera_motion(scene, m.window or program.window, m)
        elif m.signal == "camera_jitter":
            r = measure_camera_jitter(scene, m.window or program.window, m)
        elif m.signal == "range":
            r = measure_range(scene, program, m, tracks)
        elif m.signal == "displacement":
            r = measure_displacement(scene, program, m, tracks)
        elif m.signal == "instances":
            r = measure_instances(scene, m, tracks, groundings, program)
        elif m.signal == "region_distance":
            r = measure_region_distance(scene, program, m, tracks, groundings)
        elif m.signal == "onset":
            r = measure_event_onset(scene, program, m, tracks)
        elif m.signal == "order":
            r = measure_order(scene, m, results)
        elif m.signal == "focal":
            r = measure_zoom(scene, m.window or program.window, m)
        elif m.signal == "camera_speed":
            r = measure_rhythm(scene, m.window or program.window, m)
        elif m.signal == "covisibility":
            r = measure_revisit(scene, m.window or program.window, m)
        elif m.signal == "area":
            r = measure_apparent_size(scene, program, m, tracks)
        elif m.signal == "presence":
            r = measure_visibility(scene, program, m, tracks)
        elif m.signal == "elevation":
            r = measure_elevation(scene, program, m, tracks)
        elif m.signal == "speed":
            r = measure_motion_profile(scene, program, m, tracks)
        elif m.signal == "speed_metric":
            r = measure_metric_speed(scene, program, m, tracks)
        elif m.signal == "camera_displacement":
            r = measure_camera_displacement(scene, m.window or program.window, m)
        else:
            r = unavailable(m, f"no measurement implemented for signal {m.signal!r}")
        _flag_simplified_prompts(program, groundings, m, r)
        results[m.id] = r
    return results, tracks


def _flag_simplified_prompts(program: QueryProgram, groundings: Mapping[str, Grounding | None],
                             m: Measurement, r: Result) -> None:
    """A subject found only by a simpler prompt than the question's own ("person" for "person in
    the pink suit") may be the wrong instance: the result says so, and is at best medium confidence."""

    if not r.available:
        return
    ops = list(m.of)
    if m.signal == "onset":
        ops = [e.actor for e in program.events if e.id in m.of]
    for op in ops:
        base = op.split("[")[0]
        entity = next((e for e in program.entities if e.id == base), None)
        g = groundings.get(entity.phrase) if entity is not None and entity.phrase else None
        if g is not None and g.used_phrase and g.used_phrase != entity.phrase:
            r.notes.append(f"{entity.phrase!r} was found only with the simpler prompt {g.used_phrase!r}: the instance may be another one")
            if r.confidence == "high":
                r.confidence = "medium"
