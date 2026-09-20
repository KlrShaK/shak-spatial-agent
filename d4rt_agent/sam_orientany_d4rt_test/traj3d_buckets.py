"""Bucket reports: turn one clip's finished analysis into text an agent can read.

`traj3d_analyze` leaves two arrays on disk per clip -- `analysis/trajectory.npz`
(the subject, plus its Orient-Anything facing) and `analysis/camera_poses.npz`
(the recovered camera).  This module is the layer above: it answers *one*
question about them at a time, in words, with the numbers that justify the words
attached.

The four reports correspond to the four measurement buckets DSI-Bench questions
fall into (`dsi_dataset_exploration/logs/TAXONOMY_REPORT.md`):

    traj_subject_own_frame        how the subject moved, in its own initial pose
    traj_camera_own_frame         how the observer moved AND turned, in its own
    sub_obs_distance_change       how far apart they were, start vs end
    orient_camera_subject_frame   which side of the subject the observer was on

Prototyped in `testing_splines.ipynb` cells 48-57; that notebook re-derives all
of this by hand and carries a cell asserting the two agree, which is what stops
them drifting.

Two deliberate departures from the notebook:

**One smoothing rule for all four.**  The notebook fed bucket 1 the
Savitzky-Golay `smoothed` track but buckets 3 and 4 the running mean (cells 51
vs 55/57).  Everything here uses the running mean.  It is the one the two
checkpoint buckets were validated against, and a report whose distance and whose
range come from differently-filtered copies of the same trajectory is a report
that can contradict itself.

**Rotation is reported per segment, not just at checkpoints.**  DSI-Bench asks
"moving right" and "orbiting clockwise" as options of a single question, and the
difference between them is whether lateral translation is accompanied by a
same-sign pan.  Ten segments of (direction, distance, angle deltas) makes that
visible without this module having to make the categorical call itself.

Distances are D4RT units -- scale-relative, comparable only within one clip.
Angles are degrees.  Axes are OpenCV throughout: +x right, +y DOWN, +z forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# The path is sampled at N+1 uniform times to give N segments.  Ten is a
# compromise: enough to separate a turn from a straight line, few enough that
# the table stays readable in a prompt.
N_LANGUAGE_SEGMENTS = 10

# Half-width of the running-mean window, as the notebook uses it.
SMOOTHING_WINDOW = 5

# Motion below these fractions of the relevant scale is not reported as motion.
# The subject is judged against its own extent, the camera against the scene's,
# because those are the only quantities in a scale-relative pipeline that mean
# the same thing from one clip to the next.
SUBJECT_FLOOR_RATIO = 0.10
CAMERA_FLOOR_RATIO = 0.01

# World up expressed in the OpenCV camera frame, where +y points down.  Used to
# define the roll-free "level" right axis.
UP_IN_CAMERA = np.array([0.0, -1.0, 0.0])


# ---------------------------------------------------------------------------
# smoothing and small helpers
# ---------------------------------------------------------------------------


def running_mean(track: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """Centred box average over a (T, 3) path.

    Gaps are interpolated so the filter sees a continuous signal and then
    restored to NaN, so an absent frame stays absent rather than acquiring an
    invented position.  The window shrinks at the ends instead of padding: there
    is no data outside the clip and inventing some would bias the first and last
    segments, which are exactly the ones every checkpoint report reads.
    """

    track = np.asarray(track, dtype=np.float64)
    finite = np.isfinite(track).all(axis=1)
    if finite.sum() < 2:
        return track.copy()

    filled = track.copy()
    if not finite.all():
        index = np.arange(len(track))
        for axis in range(3):
            filled[:, axis] = np.interp(index, index[finite], track[finite, axis])

    half, count = window // 2, len(filled)
    out = np.empty_like(filled)
    for t in range(count):
        out[t] = filled[max(0, t - half) : min(count, t + half + 1)].mean(axis=0)
    out[~finite] = np.nan
    return out


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def wrapped_angle_difference(a: float, b: float) -> float:
    """``a - b`` folded into (-180, 180]."""

    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def _unwrap_degrees(angles: np.ndarray) -> np.ndarray:
    """Unwrap, tolerating NaN gaps.

    ``np.unwrap`` propagates a single NaN across the whole tail, which would
    erase every angle after one unusable frame.  Unwrapping only the finite
    samples and writing them back keeps the gap local.
    """

    angles = np.asarray(angles, dtype=np.float64)
    out = np.full_like(angles, np.nan)
    finite = np.isfinite(angles)
    if finite.any():
        out[finite] = np.degrees(np.unwrap(np.radians(angles[finite])))
    return out


def _interp_angles(sample_frames: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Sample a per-frame angle track at (generally non-integer) frame times.

    The angles are unwrapped first.  Interpolating wrapped degrees would invent
    a 359-degree swing every time a track crosses the +/-180 seam, which on a
    panning camera is common and would read as a violent rotation.
    """

    unwrapped = _unwrap_degrees(angles)
    finite = np.isfinite(unwrapped)
    if finite.sum() < 2:
        return np.full(len(sample_frames), np.nan)
    frames = np.flatnonzero(finite).astype(np.float64)
    sampled = np.interp(sample_frames, frames, unwrapped[finite])
    # Do not extrapolate: a sample outside the measured span is not a measurement.
    sampled[(sample_frames < frames[0]) | (sample_frames > frames[-1])] = np.nan
    return sampled


def checkpoint_frames(mask: Sequence[bool] | np.ndarray) -> tuple[int, int, int]:
    """First, temporal-middle, and last True indices."""

    frames = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(frames) < 2:
        raise ValueError("at least two usable frames are required")
    middle_target = 0.5 * (frames[0] + frames[-1])
    middle = int(frames[np.argmin(np.abs(frames - middle_target))])
    return int(frames[0]), middle, int(frames[-1])


# ---------------------------------------------------------------------------
# orientation: what each body was pointing at, frame by frame
# ---------------------------------------------------------------------------


def level_basis(world_up: np.ndarray) -> np.ndarray | None:
    """Columns [right, down, forward] with DOWN along gravity, heading kept.

    The camera's own frame is the right answer for "which way was it facing" and
    the wrong one for "did it go up or down": those only coincide when the camera
    is level. This keeps the heading -- forward stays the direction the camera
    faced at frame 0, flattened into the horizontal plane -- while handing the
    vertical axis to gravity.

    None when the camera looked straight up or down, where a horizontal heading
    does not exist and there is nothing to preserve.
    """

    down = _unit(-np.asarray(world_up, dtype=np.float64))
    if down is None:
        return None
    camera_forward = np.array([0.0, 0.0, 1.0])
    forward = _unit(camera_forward - float(camera_forward @ down) * down)
    if forward is None:
        return None
    right = _unit(np.cross(down, forward))
    if right is None:
        return None
    return np.column_stack([right, down, forward])


def camera_pose_angles(
    rotation: np.ndarray, level: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-frame pan / tilt / roll of the camera, relative to frame 0.

    ``rotation[t]`` maps directions in camera-t into camera-0, so its columns
    *are* camera-t's axes seen from the start: column 2 is where the camera is
    looking, column 0 is its right.

    Read the angles off those axes rather than decomposing a full Euler triple.
    Euler extraction has to pick a convention and picks up gimbal degeneracies
    near the poles; two axis directions carry everything the report needs and
    degrade gracefully.

        pan  = atan2(f_x, f_z)      + is a turn to the camera's own right
        tilt = -asin(f_y)           + is upward, because +y points DOWN
        roll = signed angle from the level right axis to the actual right axis,
               measured about the viewing direction; + means the camera's right
               side dipped, i.e. the CAMERA rolled clockwise seen from behind it,
               which makes the image CONTENT appear to rotate counter-clockwise

    Roll is NaN when the camera looks straight up or down, where "level" has no
    meaning.  Frames whose pose failed to solve arrive as identity from
    `recover_camera_poses`; the caller masks those out before this is read.
    """

    rotation = np.asarray(rotation, dtype=np.float64)
    if level is not None:
        # Camera-t's axes, expressed in the levelled frame rather than in
        # camera-0's tilted one, so pan is a true azimuth about gravity and tilt
        # a true elevation above the horizon. Every angle is still reported as a
        # CHANGE from frame 0, which the subtraction below restores.
        rotation = np.einsum("ij,tjk->tik", level.T, rotation)
    count = rotation.shape[0]
    pan = np.full(count, np.nan)
    tilt = np.full(count, np.nan)
    roll = np.full(count, np.nan)

    for t in range(count):
        matrix = rotation[t]
        if not np.isfinite(matrix).all():
            continue
        forward = _unit(matrix[:, 2])
        right = _unit(matrix[:, 0])
        if forward is None or right is None:
            continue

        pan[t] = np.degrees(np.arctan2(forward[0], forward[2]))
        tilt[t] = np.degrees(-np.arcsin(np.clip(forward[1], -1.0, 1.0)))

        level_right = _unit(np.cross(forward, UP_IN_CAMERA))
        if level_right is None:
            continue  # looking along the vertical: roll is undefined, leave NaN
        roll[t] = np.degrees(
            np.arctan2(float(np.cross(level_right, right) @ forward),
                       float(level_right @ right))
        )

    angles = {"pan_deg": pan, "tilt_deg": tilt, "roll_deg": roll}
    # Relative to the start, in every basis. Without levelling rotation[0] is the
    # identity and these are already zero, so this is a no-op there.
    absolute = {f"initial_{name}": float(values[0]) for name, values in angles.items()}
    for name, values in angles.items():
        reference = values[0]
        if np.isfinite(reference):
            angles[name] = np.array([
                wrapped_angle_difference(v, reference) if np.isfinite(v) else np.nan
                for v in values
            ])
    angles["_absolute_at_frame_0"] = absolute
    return angles


def subject_heading_angles(
    forward_cam0: np.ndarray, orientation_usable: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-frame heading / elevation of the subject's own facing, in camera-0.

    ``forward_cam0`` is Orient-Anything's facing direction already rotated into
    the frame-0 camera by `build_orientation`, so camera motion is compensated
    and a change here is the subject actually turning rather than the observer
    walking around it.

    Both angles are reported **relative to the subject's first usable facing**,
    which is the same pose `initial_subject_basis` builds its axes from -- so a
    heading delta and a displacement in this report are expressed about the same
    reference and can be read together.

    Frames where OAV2 did not resolve a unique front (`alpha != 1`) are NaN.
    A symmetric or unresolvable object has no meaningful facing, and reporting
    one anyway is how a plausible-looking wrong answer gets made.
    """

    forward_cam0 = np.asarray(forward_cam0, dtype=np.float64)
    usable = np.asarray(orientation_usable, dtype=bool)
    count = forward_cam0.shape[0]
    heading = np.full(count, np.nan)
    elevation = np.full(count, np.nan)

    for t in range(count):
        if t >= len(usable) or not usable[t]:
            continue
        facing = _unit(forward_cam0[t])
        if facing is None:
            continue
        heading[t] = np.degrees(np.arctan2(facing[0], facing[2]))
        elevation[t] = np.degrees(-np.arcsin(np.clip(facing[1], -1.0, 1.0)))

    finite = np.flatnonzero(np.isfinite(heading))
    if len(finite):
        first = int(finite[0])
        heading = np.array([
            wrapped_angle_difference(value, heading[first]) if np.isfinite(value) else np.nan
            for value in heading
        ])
        elevation = elevation - elevation[first]
    return {"heading_deg": heading, "elevation_deg": elevation}


# ---------------------------------------------------------------------------
# translation: direction words and uniform-time segments
# ---------------------------------------------------------------------------

_LATTICE = np.array(
    [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
     if (x, y, z) != (0, 0, 0)],
    dtype=np.float64,
)
_UNIT_LATTICE = _LATTICE / np.linalg.norm(_LATTICE, axis=1, keepdims=True)


def direction_name(step: Sequence[float] | np.ndarray) -> str:
    """Snap a displacement to the closest of the 26 Moore-neighbour directions.

    The words follow the local OpenCV convention -- +x right, +y down, +z
    forward -- so a negative y is reported as **up**, not down.
    """

    step = np.asarray(step, dtype=np.float64)
    length = float(np.linalg.norm(step))
    if not np.isfinite(length) or length <= 1e-12:
        return "stationary"

    sx, sy, sz = _LATTICE[np.argmax(_UNIT_LATTICE @ (step / length))].astype(int)
    words = []
    if sy:
        words.append("down" if sy > 0 else "up")
    if sx:
        words.append("right" if sx > 0 else "left")
    if sz:
        words.append("forward" if sz > 0 else "backward")
    return "-".join(words)


# Signed axis names, in the local OpenCV convention: +x right, +y DOWN, +z forward.
AXIS_WORDS = (("right", "left"), ("down", "up"), ("forward", "backward"))


def axis_components(displacement) -> dict[str, float]:
    """Signed components of one displacement, keyed by the axis they lie on."""

    x, y, z = (float(v) for v in displacement)
    return {"right": x, "down": y, "forward": z}


def rank_components(displacement) -> str:
    """A displacement as its named components, largest magnitude first.

    The snapped direction word says a step went "down-forward"; it does NOT say
    which of the two dominated, because 26 lattice directions cannot express a
    ratio. Without this, an agent told to rank by measured magnitude has no
    magnitudes to rank -- and picking the wrong half of a compound word is a
    wrong answer that looks perfectly reasoned.
    """

    parts = []
    for value, (positive, negative) in zip(displacement, AXIS_WORDS):
        value = float(value)
        parts.append((abs(value), f"{abs(value):.3f} {positive if value >= 0 else negative}"))
    parts.sort(key=lambda item: -item[0])
    return ", ".join(text for _, text in parts)


def uniform_time_segments(
    track: np.ndarray, n_segments: int = N_LANGUAGE_SEGMENTS
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a finite (T, 3) path at ``n_segments + 1`` uniform frame times."""

    track = np.asarray(track, dtype=np.float64)
    finite = np.isfinite(track).all(axis=1)
    frames = np.flatnonzero(finite).astype(np.float64)
    if len(frames) < 2:
        raise ValueError("trajectory needs at least two finite samples")
    sample_frames = np.linspace(frames[0], frames[-1], n_segments + 1)
    samples = np.column_stack(
        [np.interp(sample_frames, frames, track[finite, axis]) for axis in range(3)]
    )
    return sample_frames, samples


def _running_mean_1d(values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    out = np.full_like(values, np.nan)
    count = len(values)
    half = window // 2
    for t in range(count):
        chunk = values[max(0, t - half) : min(count, t + half + 1)]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            out[t] = chunk.mean()
    return out


def angle_noise_deg(angles: np.ndarray, floor: float = 2.0) -> float:
    """Robust scatter of a per-frame angle track about its own running mean.

    The same shape of estimate as `robust_range_noise` uses for distances: a
    MAD-based sigma with a floor, so "the camera turned" is only claimed when
    the turn is larger than the frame-to-frame disagreement of the estimate that
    measured it.  The 2-degree floor reflects that pan here is recovered
    indirectly, through a RANSAC rigid fit on a 144-point grid.
    """

    unwrapped = _unwrap_degrees(np.asarray(angles, dtype=np.float64))
    residual = unwrapped - _running_mean_1d(unwrapped)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return float(floor)
    mad_sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    return float(max(2.0 * mad_sigma, floor))


def displacement_noise(raw: np.ndarray, smooth: np.ndarray, floor: float) -> float:
    """How far a path must move before the motion is a measurement, not jitter.

    Two terms, and the second is the one that matters.  The first is the scatter
    of the raw path about its smoothed self.  The second is ``floor``, supplied
    by the caller from a scale the track itself does not set -- the clip's scene
    scale for the camera, the subject's own extent for the subject.

    Referencing the track's own extent instead, as an earlier version did, is
    circular: on a static camera the scatter and the extent are both tiny, the
    floor shrinks with the signal, and the report can never say "stationary".
    That is the failure this argument exists to prevent, and "Basically
    unchange" is one of the commonest ground-truth options in the benchmark.

    Everything is scale-relative because D4RT's units are -- the loss normalises
    each clip by its own mean depth, so an absolute threshold is meaningless
    across clips (`traj3d_segment` says the same about its own thresholds).
    """

    raw = np.asarray(raw, dtype=np.float64)
    smooth = np.asarray(smooth, dtype=np.float64)
    residual = np.linalg.norm(raw - smooth, axis=1)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return float(max(floor, 1e-6))
    mad_sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    return float(max(2.0 * mad_sigma, floor, 1e-6))


def _turn_word(delta: float, noise: float, positive: str, negative: str) -> str:
    if not np.isfinite(delta):
        return "unmeasured"
    if abs(delta) <= noise:
        return "no significant change"
    return positive if delta > 0 else negative


def initial_subject_basis(
    forward_cam0: np.ndarray, up_cam0: np.ndarray, orientation_usable: np.ndarray
) -> tuple[np.ndarray | None, int | None]:
    """Columns ``[subject-right, subject-down, subject-forward]`` in camera-0 coords."""

    forward = np.asarray(forward_cam0, dtype=np.float64)
    up = np.asarray(up_cam0, dtype=np.float64)
    usable = np.asarray(orientation_usable, dtype=bool)
    if forward.size == 0:
        return None, None
    ok = usable & np.isfinite(forward).all(axis=1) & np.isfinite(up).all(axis=1)
    if not ok.any():
        return None, None

    pose_frame = int(np.flatnonzero(ok)[0])
    return _orthonormal_basis(forward[pose_frame], up[pose_frame]), pose_frame


def _orthonormal_basis(forward: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    fwd = _unit(forward)
    if fwd is None:
        return None
    up_axis = np.asarray(up, dtype=np.float64) - float(np.asarray(up) @ fwd) * fwd
    down = _unit(-up_axis)
    if down is None:
        return None
    right = _unit(np.cross(down, fwd))
    if right is None:
        return None
    down = np.cross(fwd, right)  # re-orthogonalise: right x down = forward
    return np.column_stack([right, down, fwd])


def subject_basis_at(
    forward_cam0: np.ndarray,
    up_cam0: np.ndarray,
    orientation_usable: np.ndarray,
    frame: int,
) -> np.ndarray | None:
    """The same basis, built from one specific frame's facing."""

    usable = np.asarray(orientation_usable, dtype=bool)
    if frame >= len(usable) or not bool(usable[frame]):
        return None
    forward = np.asarray(forward_cam0, dtype=np.float64)[frame]
    up = np.asarray(up_cam0, dtype=np.float64)[frame]
    if not np.isfinite(forward).all() or not np.isfinite(up).all():
        return None
    return _orthonormal_basis(forward, up)


def segment_report(
    track: np.ndarray,
    angle_tracks: Mapping[str, np.ndarray],
    *,
    basis: np.ndarray | None = None,
    n_segments: int = N_LANGUAGE_SEGMENTS,
    noise: float = 0.0,
) -> dict[str, Any]:
    """Ten uniform-duration segments of translation, each carrying its rotation.

    ``basis`` rotates the displacement into some body's own axes; passing None
    leaves it in the frame-0 camera basis, which is already the observer's own
    starting frame.  The angle tracks are sampled at the *same* frame times as
    the positions, so a segment's direction word and its angle delta describe
    one interval rather than two loosely related ones.
    """

    sample_frames, points = uniform_time_segments(track, n_segments)
    points = points - points[0]
    if basis is not None:
        points = points @ basis

    sampled_angles = {
        name: _interp_angles(sample_frames, values) for name, values in angle_tracks.items()
    }
    angle_noise = {name: angle_noise_deg(values) for name, values in angle_tracks.items()}

    steps = np.diff(points, axis=0)
    distances = np.linalg.norm(steps, axis=1)
    segments: list[dict[str, Any]] = []
    for index in range(n_segments):
        row: dict[str, Any] = {
            "segment": index + 1,
            "frame_start": float(sample_frames[index]),
            "frame_end": float(sample_frames[index + 1]),
            "direction": direction_name(steps[index]),
            "distance": float(distances[index]),
            "above_noise": bool(distances[index] > noise),
            **{f"d_{axis}": round(float(value), 4)
               for axis, value in axis_components(steps[index]).items()},
        }
        for name, values in sampled_angles.items():
            delta = values[index + 1] - values[index]
            row[f"{name}_delta"] = None if not np.isfinite(delta) else float(delta)
        segments.append(row)

    runs: list[dict[str, Any]] = []
    for segment in segments:
        if runs and runs[-1]["direction"] == segment["direction"]:
            runs[-1]["count"] += 1
            runs[-1]["distance"] += segment["distance"]
            runs[-1]["segment_end"] = segment["segment"]
        else:
            runs.append({
                "direction": segment["direction"], "count": 1,
                "distance": segment["distance"],
                "segment_start": segment["segment"], "segment_end": segment["segment"],
            })

    totals: dict[str, float | None] = {}
    spans: dict[str, list[float] | None] = {}
    for name, values in sampled_angles.items():
        ok = np.isfinite(values)
        finite = values[ok]
        totals[name] = float(finite[-1] - finite[0]) if len(finite) >= 2 else None
        # The frames the angle was actually measured over.  On the runner clip
        # Orient-Anything only resolves a front from frame 26, so a heading
        # delta there describes frames 26-31 while the displacement beside it
        # describes 0-31.  Reporting the total without its span invites reading
        # a five-frame turn as a whole-clip one.
        covered = sample_frames[ok]
        spans[name] = [float(covered[0]), float(covered[-1])] if len(covered) >= 2 else None

    net_vector = points[-1] - points[0]
    net = float(np.linalg.norm(net_vector))
    return {
        "net_components": axis_components(net_vector),
        "net_breakdown": rank_components(net_vector),
        "sample_frames": sample_frames.tolist(),
        "segments": segments,
        "runs": runs,
        "angle_totals": totals,
        "angle_spans": spans,
        "angle_noise_deg": angle_noise,
        "noise_floor": float(noise),
        "total_distance": float(np.nansum(distances)),
        "net_displacement": net,
        "net_above_noise": bool(net > noise),
        "full_span": [float(sample_frames[0]), float(sample_frames[-1])],
    }


# ---------------------------------------------------------------------------
# loading one clip's analysis into the form the four reports consume
# ---------------------------------------------------------------------------


@dataclass
class BucketInputs:
    """Everything the four reports read, already smoothed and masked.

    Built once per clip so the four reports cannot disagree about which
    trajectory, which filter, or which frames were usable.
    """

    subject: np.ndarray | None
    subject_raw: np.ndarray | None
    camera: np.ndarray
    camera_raw: np.ndarray
    rotation: np.ndarray
    forward_cam0: np.ndarray | None
    up_cam0: np.ndarray | None
    orientation_usable: np.ndarray | None
    visible: np.ndarray
    num_frames: int
    # Scale references for the noise floors, from the pipeline's own summary.
    scene_scale: float = 1.0
    subject_scale: float = 1.0
    camera_rmse_relative: float = 0.0
    # World up in camera-0 coords, recovered from the ground plane, or None.
    world_up: np.ndarray | None = None
    world_up_inlier_fraction: float = 0.0

    @property
    def has_subject(self) -> bool:
        return self.subject is not None

    @property
    def has_orientation(self) -> bool:
        return (
            self.orientation_usable is not None
            and np.asarray(self.orientation_usable, dtype=bool).any()
        )


def load_bucket_inputs(video_dir: Path | str) -> BucketInputs:
    """Read one clip's `analysis/` directory.

    `trajectory.npz` is optional: a camera-only run never segments a subject, so
    the subject arrays are simply absent and the two subject-bearing reports
    decline rather than fail.
    """

    analysis = Path(video_dir) / "analysis"
    poses = dict(np.load(analysis / "camera_poses.npz"))

    summary_path = analysis / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    scene_scale = float(summary.get("camera", {}).get("scene_scale") or 1.0)
    subject_scale = float(summary.get("subject_scale") or 1.0)
    camera_rmse_relative = float(
        summary.get("camera", {}).get("median_relative_rmse") or 0.0
    )

    # A pose that failed to solve is left as identity/zero by
    # `recover_camera_poses`, which is indistinguishable from a camera that
    # genuinely did not move.  Its rmse is the NaN that tells them apart.
    rmse = np.asarray(poses["rmse"], dtype=np.float64)
    failed = ~np.isfinite(rmse)

    camera_raw = np.asarray(poses["translation"], dtype=np.float64).copy()
    camera_raw[failed] = np.nan
    rotation = np.asarray(poses["rotation"], dtype=np.float64).copy()
    rotation[failed] = np.nan
    camera = running_mean(camera_raw)
    num_frames = camera_raw.shape[0]

    world_up = None
    stored_up = poses.get("world_up_cam0")
    if stored_up is not None and np.isfinite(np.asarray(stored_up, dtype=np.float64)).all():
        world_up = np.asarray(stored_up, dtype=np.float64)
    world_up_fraction = float(poses.get("world_up_inlier_fraction", 0.0) or 0.0)

    subject = subject_raw = forward_cam0 = up_cam0 = orientation_usable = None
    trajectory_path = analysis / "trajectory.npz"
    if trajectory_path.exists():
        trajectory = dict(np.load(trajectory_path))
        subject_raw = np.asarray(trajectory["ransac_median"], dtype=np.float64)
        subject = running_mean(subject_raw)
        if trajectory.get("forward_cam0") is not None and len(trajectory["forward_cam0"]):
            forward_cam0 = np.asarray(trajectory["forward_cam0"], dtype=np.float64)
            up_cam0 = np.asarray(trajectory["up_cam0"], dtype=np.float64)
            orientation_usable = np.asarray(trajectory["orientation_usable"], dtype=bool)

        valid = np.asarray(trajectory["valid"], dtype=bool)
        keep = np.asarray(trajectory["keep"], dtype=bool)
        tracked = valid[keep].any(axis=0) if keep.any() else valid.any(axis=0)
        visible = (
            tracked
            & np.isfinite(subject).all(axis=1)
            & np.isfinite(camera).all(axis=1)
        )
    else:
        visible = np.isfinite(camera).all(axis=1)

    return BucketInputs(
        subject=subject, subject_raw=subject_raw,
        camera=camera, camera_raw=camera_raw, rotation=rotation,
        forward_cam0=forward_cam0, up_cam0=up_cam0,
        orientation_usable=orientation_usable,
        visible=visible, num_frames=num_frames,
        scene_scale=scene_scale, subject_scale=subject_scale,
        camera_rmse_relative=camera_rmse_relative,
        world_up=world_up, world_up_inlier_fraction=world_up_fraction,
    )


def _unavailable(bucket: str, reason: str) -> dict[str, Any]:
    """A report that could not be made, phrased so the agent can act on it."""

    return {
        "bucket": bucket,
        "available": False,
        "unavailable_reason": reason,
        "text": (
            f"The {bucket} measurement is UNAVAILABLE for this clip: {reason}. "
            "No substitute measurement was made. Answer from the video frames alone, "
            "say so in your reasoning, and choose an option expressing uncertainty if "
            "one is offered."
        ),
        "metrics": {},
    }


# ---------------------------------------------------------------------------
# the four reports
# ---------------------------------------------------------------------------

# (metric key, word for a positive delta, word for a negative delta, label)
_CAMERA_ANGLES = (
    ("pan_deg", "turning right", "turning left", "pan"),
    ("tilt_deg", "tilting up", "tilting down", "tilt"),
    ("roll_deg", "rolling clockwise", "rolling counter-clockwise", "roll"),
)
_SUBJECT_ANGLES = (
    ("heading_deg", "turning to its right", "turning to its left", "heading"),
)


def _segment_prose(report: Mapping[str, Any], prefix: str, entity: str,
                   angle_specs: Sequence[tuple[str, str, str, str]]) -> str:
    floor = report["noise_floor"]

    # Lead with the verdict on whether anything moved at all.  A reader handed
    # ten direction words assumes there was motion to describe; on a static
    # camera those words are jitter, and the noise floor is the only thing that
    # says so.
    if not report["net_above_noise"]:
        text = (
            f"{prefix}, the {entity} is effectively STATIONARY: net displacement "
            f"{report['net_displacement']:.4f} units is within the {floor:.4f}-unit noise "
            f"floor, and the per-segment directions below are tracking jitter rather than "
            f"measured motion."
        )
    else:
        phrases = [
            f"{run['direction']} for {run['distance']:.3f} units"
            + (f" across {run['count']} segments" if run["count"] > 1 else "")
            for run in report["runs"]
        ]
        text = (
            f"{prefix}, the {entity} moves " + "; then ".join(phrases) + "."
            f" Net displacement {report['net_displacement']:.3f} units, against a "
            f"{floor:.4f}-unit noise floor. Broken down by axis, largest first: "
            f"{report['net_breakdown']}."
        )

    clauses = []
    full_span = report["full_span"]
    for key, positive, negative, label in angle_specs:
        total = report["angle_totals"].get(key)
        if total is None:
            clauses.append(f"{label} unmeasured")
            continue
        noise = report["angle_noise_deg"].get(key, 0.0)
        word = _turn_word(total, noise, positive, negative)
        clause = (
            f"{label} {total:+.1f} deg (within the +/-{noise:.1f} deg noise floor)"
            if word == "no significant change"
            else f"{label} {total:+.1f} deg ({word})"
        )
        span = report["angle_spans"].get(key)
        if span is not None and (span[0] > full_span[0] + 0.5 or span[1] < full_span[1] - 0.5):
            clause += f" -- measured only over frames {span[0]:.0f}-{span[1]:.0f}, NOT the whole clip"
        clauses.append(clause)
    if clauses:
        text += f" Over the same span, {entity} orientation changed: " + "; ".join(clauses) + "."
    return text


def traj_subject_own_frame(inputs: BucketInputs) -> dict[str, Any]:
    """How the subject moved, described in the subject's own initial pose.

    Rotating the displacement into the subject's own axes is what makes
    "forward" mean the direction the subject faces rather than the direction the
    camera happens to point.  It is therefore unavailable, rather than
    approximated, when Orient-Anything never resolved a unique front: falling
    back to camera axes would silently answer a different question.
    """

    bucket = "traj_subject_own_frame"
    if inputs.subject is None:
        return _unavailable(bucket, "no subject was segmented for this clip")
    if not inputs.has_orientation:
        return _unavailable(
            bucket,
            "Orient-Anything never resolved a unique front for the subject, so the "
            "subject's own forward axis is undefined",
        )

    basis, pose_frame = initial_subject_basis(
        inputs.forward_cam0, inputs.up_cam0, inputs.orientation_usable
    )
    if basis is None:
        return _unavailable(bucket, "the subject's facing could not be turned into a stable basis")

    angles = subject_heading_angles(inputs.forward_cam0, inputs.orientation_usable)
    # A person who shifts by a tenth of their own body length has not "moved".
    floor = displacement_noise(
        inputs.subject_raw, inputs.subject, SUBJECT_FLOOR_RATIO * inputs.subject_scale
    )
    try:
        report = segment_report(inputs.subject, angles, basis=basis, noise=floor)
    except ValueError as error:
        return _unavailable(bucket, str(error))

    prefix = f"In the subject's initial pose (resolved at frame {pose_frame})"
    return {
        "bucket": bucket, "available": True, "unavailable_reason": None,
        "text": _segment_prose(report, prefix, "subject", _SUBJECT_ANGLES),
        "metrics": {"pose_frame": pose_frame, "coordinate_frame": "subject", **report},
    }


def traj_camera_own_frame(inputs: BucketInputs) -> dict[str, Any]:
    """How the observer moved AND turned, in its own starting frame.

    Translation and rotation are one report rather than two because DSI-Bench
    offers them as options of a single question: an orbit is lateral translation
    accompanied by a same-sign pan, and a dolly is the same translation without
    it.  Splitting them would force a choice that loses whichever half was not
    chosen.

    The camera's frame-0 basis IS the observer's own starting frame, so no
    rotation into a body basis is needed here.
    """

    bucket = "traj_camera_own_frame"

    # Level the frame when gravity is known. Vertical goes to gravity; the
    # heading -- what "forward" and "left" mean -- stays the camera's own at
    # frame 0, which is what the question asks for.
    basis = level_basis(inputs.world_up) if inputs.world_up is not None else None
    frame_note = ""
    if basis is None:
        frame_note = (
            " NOTE: the world vertical could not be recovered for this clip, so up/down "
            "below are the camera's own axes at frame 0, NOT gravity. If the camera was "
            "tilted, a vertical motion will appear partly as forward/backward."
        )
    else:
        pitch = float(np.degrees(np.arcsin(np.clip(-inputs.world_up[2], -1.0, 1.0))))
        if abs(pitch) >= 10.0:
            frame_note = (
                f" (Up/down are measured against gravity. The camera started pitched "
                f"{abs(pitch):.0f} degrees {'down' if pitch > 0 else 'up'}, so its own axes "
                f"would have split this differently.)"
            )

    angles = camera_pose_angles(inputs.rotation, level=basis)
    initial_pose = angles.pop("_absolute_at_frame_0", {})
    # The camera pose is inferred, never observed, so its own alignment residual
    # is the honest lower bound on how well it was ever located.
    floor = displacement_noise(
        inputs.camera_raw, inputs.camera,
        max(2.0 * inputs.camera_rmse_relative * inputs.scene_scale,
            CAMERA_FLOOR_RATIO * inputs.scene_scale),
    )
    try:
        report = segment_report(inputs.camera, angles, basis=basis, noise=floor)
    except ValueError as error:
        return _unavailable(bucket, f"camera pose recovery produced too few usable frames: {error}")

    prefix = (
        "In the observer's starting frame, with up/down measured against gravity"
        if basis is not None else "In the observer's starting frame"
    )
    return {
        "bucket": bucket, "available": True, "unavailable_reason": None,
        "text": _segment_prose(report, prefix, "camera", _CAMERA_ANGLES) + frame_note,
        "metrics": {
            "coordinate_frame": "observer-levelled" if basis is not None else "observer",
            "vertical_is_gravity": basis is not None,
            "world_up_inlier_fraction": inputs.world_up_inlier_fraction,
            "initial_camera_pose_deg": initial_pose,
            **report,
        },
    }


def robust_range_noise(
    raw_distance: np.ndarray, smooth_distance: np.ndarray, usable: np.ndarray
) -> float:
    """Robust raw-vs-smoothed range scatter, plus a one-percent resolution floor."""

    residual = np.asarray(raw_distance)[usable] - np.asarray(smooth_distance)[usable]
    residual = residual[np.isfinite(residual)]
    mad_sigma = (
        1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if len(residual) else 0.0
    )
    scale = float(np.nanmedian(np.asarray(smooth_distance)[usable]))
    return float(max(2.0 * mad_sigma, 0.01 * scale, 1e-6))


def range_trend(before: float, after: float, tolerance: float) -> str:
    delta = float(after - before)
    if delta > tolerance:
        return "grew"
    if delta < -tolerance:
        return "shrank"
    return "stayed approximately the same"


def _range_checkpoint_phrase(before: float, after: float, tolerance: float) -> str:
    trend = range_trend(before, after, tolerance)
    if trend == "grew":
        return f"had grown to {after:.3f} units"
    if trend == "shrank":
        return f"had shrunk to {after:.3f} units"
    return f"had stayed approximately the same at {after:.3f} units"


def _range_overall_phrase(before: float, after: float, tolerance: float) -> str:
    trend = range_trend(before, after, tolerance)
    if trend == "grew":
        return f"grew by {after - before:.3f} units"
    if trend == "shrank":
        return f"shrank by {before - after:.3f} units"
    return f"stayed approximately the same (net change {after - before:+.3f} units)"


def sub_obs_distance_change(inputs: BucketInputs) -> dict[str, Any]:
    """How far apart observer and subject were, start vs middle vs end.

    The simplest of the four and the only one needing no orientation at all:
    range is rotation-invariant, so norms taken from different viewpoints are
    directly comparable and no common basis has to be established.
    """

    bucket = "sub_obs_distance_change"
    if inputs.subject is None:
        return _unavailable(bucket, "no subject was segmented for this clip")

    smooth_distance = np.linalg.norm(inputs.subject - inputs.camera, axis=1)
    raw_distance = np.linalg.norm(inputs.subject_raw - inputs.camera_raw, axis=1)
    usable = (
        np.asarray(inputs.visible, dtype=bool)
        & np.isfinite(smooth_distance)
        & np.isfinite(raw_distance)
    )
    try:
        first, middle, last = checkpoint_frames(usable)
    except ValueError as error:
        return _unavailable(bucket, f"too few frames have both subject and camera: {error}")

    tolerance = robust_range_noise(raw_distance, smooth_distance, usable)
    metrics: list[dict[str, Any]] = []
    previous: float | None = None
    for checkpoint, frame in (("initial", first), ("midpoint", middle), ("final", last)):
        distance = float(smooth_distance[frame])
        metrics.append({
            "checkpoint": checkpoint, "frame": frame, "distance": distance,
            "delta_from_previous": None if previous is None else distance - previous,
            "trend_from_previous": (
                None if previous is None else range_trend(previous, distance, tolerance)
            ),
        })
        previous = distance

    d0, dm, d1 = [row["distance"] for row in metrics]
    text = (
        f"Initially, at frame {first}, the distance between the observer and subject was "
        f"{d0:.3f} units. By the midpoint at frame {middle}, it "
        f"{_range_checkpoint_phrase(d0, dm, tolerance)}. By the final visible frame {last}, it "
        f"{_range_checkpoint_phrase(dm, d1, tolerance)}. Overall, it "
        f"{_range_overall_phrase(d0, d1, tolerance)}. A change smaller than the "
        f"{tolerance:.3f}-unit noise tolerance is not a change."
    )
    return {
        "bucket": bucket, "available": True, "unavailable_reason": None, "text": text,
        "metrics": {
            "noise_tolerance": tolerance, "checkpoints": metrics,
            "per_frame": [
                {"frame": int(t), "distance": float(smooth_distance[t])}
                for t in np.flatnonzero(usable)
            ],
        },
    }


_OCTANT_NAMES = ("front", "front-right", "right", "back-right",
                 "back", "back-left", "left", "front-left")
_QUADRANT_NAMES = ("front", "right", "back", "left")


def octant_sector(bearing_deg: float) -> tuple[str, float]:
    """Eight 45-degree sectors; positive bearing turns toward the subject's right."""

    angle = ((float(bearing_deg) + 180.0) % 360.0) - 180.0
    index = int(np.floor((angle + 22.5) / 45.0)) % 8
    centre = 45.0 * index
    if centre > 180.0:
        centre -= 360.0
    return _OCTANT_NAMES[index], centre


def quadrant_sector(bearing_deg: float) -> tuple[str, float]:
    """Four 90-degree sectors: front, right, back, left.

    This is the vocabulary DSI-Bench's cate-5 options are actually written in
    ("from character's left to back"), so it is what the report leads with.  The
    octant is kept alongside it: an observer at a bearing of 44 degrees is
    "front" by a hair, and a reader who sees only the quadrant has no way to
    know that.
    """

    angle = ((float(bearing_deg) + 180.0) % 360.0) - 180.0
    index = int(np.floor((angle + 45.0) / 90.0)) % 4
    centre = 90.0 * index
    if centre > 180.0:
        centre -= 360.0
    return _QUADRANT_NAMES[index], centre


def orient_camera_subject_frame(inputs: BucketInputs) -> dict[str, Any]:
    """Which side of the oriented subject the observer sits on, and how it changes.

    At each checkpoint, form the vector from subject to camera and project it
    into that frame's Orient-Anything basis.  The horizontal bearing is
    ``atan2(right, forward)``: 0 is in front of the subject, +90 is to its
    right, +/-180 behind, -90 to its left.

    Unlike the trajectory reports this needs the subject's facing *at each
    reported frame*, not just once, so its start and end are the first and last
    frames where the trajectory and an unambiguous facing both exist -- which
    may be narrower than the clip.
    """

    bucket = "orient_camera_subject_frame"
    if inputs.subject is None:
        return _unavailable(bucket, "no subject was segmented for this clip")
    if not inputs.has_orientation:
        return _unavailable(
            bucket,
            "Orient-Anything never resolved a unique front for the subject, so which side "
            "of it the observer is on is undefined",
        )

    usable = (
        np.asarray(inputs.visible, dtype=bool)
        & np.asarray(inputs.orientation_usable, dtype=bool)
        & np.isfinite(inputs.subject).all(axis=1)
        & np.isfinite(inputs.camera).all(axis=1)
    )
    try:
        first, middle, last = checkpoint_frames(usable)
    except ValueError as error:
        return _unavailable(
            bucket, f"too few frames have both a trajectory and a resolved facing: {error}"
        )

    metrics: list[dict[str, Any]] = []
    for checkpoint, frame in (("initial", first), ("midpoint", middle), ("final", last)):
        basis = subject_basis_at(
            inputs.forward_cam0, inputs.up_cam0, inputs.orientation_usable, frame
        )
        if basis is None:
            return _unavailable(bucket, f"the subject's facing is unusable at frame {frame}")
        subject_to_camera = inputs.camera[frame] - inputs.subject[frame]
        right, down, forward = subject_to_camera @ basis
        bearing = float(np.degrees(np.arctan2(right, forward)))
        quadrant, quadrant_centre = quadrant_sector(bearing)
        octant, octant_centre = octant_sector(bearing)
        metrics.append({
            "checkpoint": checkpoint, "frame": frame,
            "quadrant": quadrant, "octant": octant, "bearing_deg": bearing,
            "quadrant_margin_deg": float(45.0 - abs(wrapped_angle_difference(bearing, quadrant_centre))),
            "octant_margin_deg": float(22.5 - abs(wrapped_angle_difference(bearing, octant_centre))),
            "bearing_delta_from_previous_deg": None,
            "right": float(right), "up": float(-down), "forward": float(forward),
            "distance": float(np.linalg.norm(subject_to_camera)),
        })
    for previous, current in zip(metrics, metrics[1:]):
        current["bearing_delta_from_previous_deg"] = wrapped_angle_difference(
            current["bearing_deg"], previous["bearing_deg"]
        )

    q0, qm, q1 = [row["quadrant"] for row in metrics]
    transition = (
        f"remained on the subject's {q0} side" if q0 == q1
        else f"moved from the subject's {q0} side to its {q1}"
    )
    tight = [
        f"{row['checkpoint']} ({row['quadrant']}, {row['quadrant_margin_deg']:.1f} deg from the boundary)"
        for row in metrics if row["quadrant_margin_deg"] < 10.0
    ]
    text = (
        f"At the first jointly usable frame ({first}), the observer was on the subject's {q0} "
        f"side (bearing {metrics[0]['bearing_deg']:+.1f} deg). At the midpoint ({middle}) it was "
        f"on the {qm} side (bearing {metrics[1]['bearing_deg']:+.1f} deg), and at the final "
        f"jointly usable frame ({last}) on the {q1} side "
        f"(bearing {metrics[2]['bearing_deg']:+.1f} deg). Overall, the observer {transition}."
    )
    if tight:
        text += " Near a sector boundary, so the label is not robust at: " + "; ".join(tight) + "."
    return {
        "bucket": bucket, "available": True, "unavailable_reason": None, "text": text,
        "metrics": {
            "checkpoints": metrics,
            "sector_definition": (
                "bearing 0 = directly in front of the subject, +90 = the subject's right, "
                "+/-180 = behind it, -90 = its left; quadrants are 90 deg wide"
            ),
        },
    }


BUILDERS = {
    "traj_subject_own_frame": traj_subject_own_frame,
    "traj_camera_own_frame": traj_camera_own_frame,
    "sub_obs_distance_change": sub_obs_distance_change,
    "orient_camera_subject_frame": orient_camera_subject_frame,
}


def build_report(bucket: str, inputs: BucketInputs) -> dict[str, Any]:
    """Build one bucket's report by name."""

    if bucket not in BUILDERS:
        raise KeyError(f"unknown bucket {bucket!r}; expected one of {sorted(BUILDERS)}")
    return BUILDERS[bucket](inputs)
