"""Geometry for the traj3d pipeline: frames, camera recovery, aggregation.

Pure NumPy on purpose.  Everything here is testable on a login node without a
GPU, which matters because these are the parts that fail *silently* -- a wrong
basis convention or a logit read as a probability produces plausible-looking
output rather than an error.

Three coordinate conventions meet in this file:

    D4RT / OpenCV     +x right, +y down, +z forward.  All ``xyz_3d`` output.
    OrientAnythingV2  +x forward (facing), +y lateral, +z up.  Its angle output.
    image pixels      u right, v down, origin top-left.

``OAV2_TO_OPENCV`` converts the second into the first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# OrientAnything-V2 -> OpenCV camera frame
# ---------------------------------------------------------------------------

# This is the authors' own change of basis, lifted from
# third-party-tools/Orient-Anything-V2/utils/utils.py:342 (``R_trans_axis_left``
# inside relate_azi_ele_rot_to_OneposeRMatrix).  That function exists to compare
# OAV2 predictions against OnePose/BOP ground-truth rotations, which are in the
# OpenCV convention -- so this matrix is validated by the paper's own evaluation
# rather than guessed by us.
#
# It maps OAV2 axes to OpenCV axes:
#     +x forward -> (0, 0, -1)   toward the camera
#     +y lateral -> (1, 0, 0)    image right
#     +z up      -> (0, -1, 0)   up, since OpenCV +y points down
#
# det = +1 and B @ B.T = I, i.e. a proper rotation, asserted in the tests.
OAV2_TO_OPENCV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ]
)


def _axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Right-handed Rodrigues rotation, matching OAV2's axis_angle_rotation_batch."""

    axis = np.asarray(axis, dtype=np.float64)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    c1 = 1.0 - c
    return np.array(
        [
            [x * x * c1 + c, x * y * c1 - z * s, x * z * c1 + y * s],
            [x * y * c1 + z * s, y * y * c1 + c, y * z * c1 - x * s],
            [x * z * c1 - y * s, y * z * c1 + x * s, z * z * c1 + c],
        ]
    )


def oav2_object_matrix(az_deg: float, el_deg: float, ro_deg: float) -> np.ndarray:
    """Reproduce OAV2's ``azi_ele_rot_to_Obj_Rmatrix_batch`` in NumPy.

    Composition and signs follow utils/app_utils.py:120-149 exactly:
    azimuth is applied as a *negative* rotation about +z.  Despite the name the
    result maps camera -> object; its transpose is object -> camera, and that
    transpose's columns are the object's own axes seen by the camera.

    test_traj3d.py asserts this agrees with the vendored implementation.
    """

    r_azi = _axis_angle(np.array([0.0, 0.0, 1.0]), -np.deg2rad(az_deg))
    r_ele = _axis_angle(np.array([0.0, 1.0, 0.0]), np.deg2rad(el_deg))
    r_rot = _axis_angle(np.array([1.0, 0.0, 0.0]), np.deg2rad(ro_deg))
    return r_rot @ r_ele @ r_azi


def _world_to_camera(az_deg: float, el_deg: float) -> np.ndarray:
    """Rows are the OpenCV camera axes, for a camera at (az, el) on the unit sphere.

    (az, el) is a *viewpoint*, not an object pose: el > 0 means the camera is
    above the object.  That reading is what the shipped examples show -- the F35
    is photographed from above and predicts el=+41, the two skateboards are shot
    from below and predict el=-26 and el=-13.
    """

    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    sa, ca = np.sin(az), np.cos(az)
    se, ce = np.sin(el), np.cos(el)
    return np.array(
        [
            [-sa, ca, 0.0],          # camera right
            [se * ca, se * sa, -ce],  # camera down (OpenCV +y)
            [-ce * ca, -ce * sa, -se],  # camera forward
        ]
    )


def oav2_axes_in_opencv(
    az_deg: float, el_deg: float, ro_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the subject's (forward, lateral, up) unit axes in OpenCV camera coords.

    NOT derived from ``OAV2_TO_OPENCV @ oav2_object_matrix(...).T``, which is the
    obvious reading of the vendored change of basis and is wrong.  That form
    gives ``f = (cos el sin az, -sin el, -cos el cos az)``, which agrees with the
    formula below only at az = el = 0 -- the one case a quick sanity check looks
    at -- and disagrees in the sign of *both* x and y elsewhere.

    Measured against all four unambiguous (alpha=1) shipped examples, every one
    of which contradicts it and matches this one; see
    test_facing_matches_measured_examples.  The mirror test cannot arbitrate:
    both forms are symmetric under az -> -az, which is all it checks.

    Only ``forward`` is validated this way.  Roll about the forward axis spins
    the other two, and no shipped example pins its sign down, so treat the
    lateral/up pair as indicative rather than established.
    """

    axes = _world_to_camera(az_deg, el_deg) @ _axis_angle(
        np.array([1.0, 0.0, 0.0]), np.deg2rad(ro_deg)
    )
    return axes[:, 0], axes[:, 1], axes[:, 2]


def oav2_facing_closed_form(az_deg: float, el_deg: float) -> np.ndarray:
    """The facing direction in OpenCV coords, in closed form (roll-independent).

    Equivalent to ``oav2_axes_in_opencv(...)[0]`` -- kept as an independent
    derivation so the test comparing the two would catch a change to either.

    az=0 -> (0,0,-1), facing the camera.  az=180 -> (0,0,+1), facing away.
    el>0 tilts the facing direction *downward* in the image (+y in OpenCV),
    because el>0 means we are looking down at the subject.
    """

    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array(
        [
            -np.sin(az),
            np.sin(el) * np.cos(az),
            -np.cos(el) * np.cos(az),
        ]
    )


def offaxis_correction(
    direction: np.ndarray,
    crop_centre_px: tuple[float, float],
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Rotate a crop-relative direction into the full-frame camera axes.

    OAV2 was trained on renders where the object is centred and the camera looks
    straight at it.  A crop taken at ``(u_c, v_c)`` instead has its implicit
    optical axis along the ray through the crop centre, so its answer is
    expressed about that ray rather than about the frame's optical axis.  For a
    subject a third of the way to the frame edge this is a 10-15 degree error --
    small enough to look plausible, large enough to invalidate a facing claim.

    ``intrinsics`` is ``[fx, fy, cx, cy]``, as returned by
    src/eval/tasks.py::_estimate_intrinsics_params_from_predictions.
    """

    fx, fy, cx, cy = (float(v) for v in intrinsics)
    u_c, v_c = (float(v) for v in crop_centre_px)
    ray = np.array([(u_c - cx) / max(fx, 1e-9), (v_c - cy) / max(fy, 1e-9), 1.0])
    ray /= max(np.linalg.norm(ray), 1e-12)

    optical = np.array([0.0, 0.0, 1.0])
    axis = np.cross(optical, ray)
    sin_theta = float(np.linalg.norm(axis))
    if sin_theta < 1e-8:
        return np.asarray(direction, dtype=np.float64)
    angle = float(np.arctan2(sin_theta, float(optical @ ray)))
    return _axis_angle(axis / sin_theta, angle) @ np.asarray(direction, dtype=np.float64)


# ---------------------------------------------------------------------------
# Camera motion recovery
# ---------------------------------------------------------------------------


@dataclass
class RigidFit:
    rotation: np.ndarray  # (3,3), dst ~= R @ src + t
    translation: np.ndarray  # (3,)
    inliers: np.ndarray  # bool (N,)
    rmse: float
    scale: float  # from a Sim(3) fit on the inliers; ~1.0 expected
    conditioning: float  # smallest/largest singular value of the centred cloud


def _umeyama_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Kabsch with the reflection fix.

    Mirrors src/eval/tasks.py:50-69.  Duplicated rather than imported because
    that module imports torch at module scope, and this file is deliberately
    importable (and testable) without it.  test_traj3d.py asserts the two agree.
    """

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        return None
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    cov = ((src - mu_src).T @ (dst - mu_dst)) / float(src.shape[0])
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = mu_dst - rot @ mu_src
    if not (np.isfinite(rot).all() and np.isfinite(trans).all()):
        return None
    return rot, trans


def _sim3_scale(src: np.ndarray, dst: np.ndarray) -> float:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_src = src.mean(axis=0)
    var_src = float(((src - mu_src) ** 2).sum() / max(src.shape[0], 1))
    if var_src <= 1e-12:
        return float("nan")
    cov = ((src - mu_src).T @ (dst - dst.mean(axis=0))) / float(src.shape[0])
    _, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    u, _, _ = np.linalg.svd(cov)
    if np.linalg.det(vt.T @ u.T) < 0:
        d[-1] = -1.0
    return float((s * d).sum() / var_src)


def cloud_scale(points: np.ndarray) -> float:
    """Median radius of a point cloud about its median.

    Every threshold in this module is a multiple of this.  D4RT's units are
    arbitrary (the training loss normalises each clip by its own mean depth), so
    an absolute metric threshold would mean something different in every clip.
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.shape[0] == 0:
        return float("nan")
    centre = np.median(finite, axis=0)
    return float(np.median(np.linalg.norm(finite - centre, axis=1)))


def robust_rigid(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    threshold_ratio: float = 0.05,
    iterations: int = 1000,
    seed: int = 0,
) -> RigidFit | None:
    """RANSAC over Kabsch, with a scale-relative inlier threshold.

    Follows eval_track3d_in_worldtrack.py:167-218 (3-point minimal samples, then
    a refit on the consensus set) but takes its threshold as a fraction of the
    cloud's own scale rather than the absolute 0.05 that file hard-codes.
    """

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if weights is not None:
        valid &= np.asarray(weights, dtype=np.float64) > 0
    if int(valid.sum()) < 3:
        return None

    idx = np.flatnonzero(valid)
    scale = cloud_scale(dst[idx])
    if not np.isfinite(scale) or scale <= 0:
        return None
    threshold = threshold_ratio * scale

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1
    for _ in range(iterations):
        pick = rng.choice(idx, size=3, replace=False)
        fit = _umeyama_rigid(src[pick], dst[pick])
        if fit is None:
            continue
        rot, trans = fit
        residual = np.linalg.norm((src[idx] @ rot.T + trans) - dst[idx], axis=1)
        inliers = residual < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = idx[inliers]

    if best_inliers is None or best_inliers.size < 3:
        return None

    fit = _umeyama_rigid(src[best_inliers], dst[best_inliers])
    if fit is None:
        return None
    rot, trans = fit

    residual = np.linalg.norm((src[best_inliers] @ rot.T + trans) - dst[best_inliers], axis=1)
    inlier_mask = np.zeros(src.shape[0], dtype=bool)
    inlier_mask[best_inliers] = True

    centred = src[best_inliers] - src[best_inliers].mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    conditioning = float(sv[2] / max(sv[0], 1e-12))

    return RigidFit(
        rotation=rot,
        translation=trans,
        inliers=inlier_mask,
        rmse=float(np.sqrt(np.mean(residual**2))),
        scale=_sim3_scale(src[best_inliers], dst[best_inliers]),
        conditioning=conditioning,
    )


def estimate_intrinsics(
    xyz: np.ndarray, uv_norm: np.ndarray, image_hw: tuple[int, int]
) -> np.ndarray:
    """Recover ``[fx, fy, cx, cy]`` from D4RT's own 3D/2D predictions.

    Mirrors src/eval/tasks.py:72-95.  Reimplemented in NumPy so this module (and
    the analysis stage) stay importable without torch, which costs about two
    minutes of cold cluster-filesystem reads.

    The focal length is only used for the off-axis orientation correction, where
    a few percent of error is immaterial next to the 10-15 degree effect it
    corrects.
    """

    height, width = image_hw
    cx = 0.5 * float(max(width - 1, 1))
    cy = 0.5 * float(max(height - 1, 1))
    uv = np.asarray(uv_norm, dtype=np.float64)
    u_px = uv[..., 0] * float(max(width - 1, 1))
    v_px = uv[..., 1] * float(max(height - 1, 1))

    pred = np.asarray(xyz, dtype=np.float64)
    x, y, z = pred[..., 0], pred[..., 1], pred[..., 2]

    fx_vals = z * np.abs(u_px - cx) / np.maximum(np.abs(x), 1e-6)
    fy_vals = z * np.abs(v_px - cy) / np.maximum(np.abs(y), 1e-6)
    fx_vals = fx_vals[np.isfinite(fx_vals) & (fx_vals > 1e-6)]
    fy_vals = fy_vals[np.isfinite(fy_vals) & (fy_vals > 1e-6)]

    fx = float(np.median(fx_vals)) if fx_vals.size else float(max(width, 1))
    fy = float(np.median(fy_vals)) if fy_vals.size else float(max(height, 1))
    return np.array([fx, fy, cx, cy], dtype=np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """D4RT's visibility and confidence heads are bare nn.Linear (src/model/heads.py:16,19).

    _query_explicit_points returns those raw logits.  The higher-level wrappers in
    simple_v2_backend apply the sigmoid, but we bypass them, so we must do it
    here -- thresholding a logit at 0.5 would keep almost everything.
    """

    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Trajectory aggregation
# ---------------------------------------------------------------------------


def geometric_median(
    points: np.ndarray, weights: np.ndarray | None = None, iters: int = 64, eps: float = 1e-9
) -> np.ndarray:
    """Weighted L1 multivariate median via Weiszfeld.

    Preferred over the mean (which has breakdown 0, so one bad track moves it)
    and over a per-coordinate median (which is not rotation-equivariant, so the
    answer would depend on the arbitrary orientation of the camera frame).
    """

    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.full(3, np.nan)
    if weights is None:
        weights = np.ones(points.shape[0])
    weights = np.asarray(weights, dtype=np.float64)
    if points.shape[0] < 4:
        return np.median(points, axis=0)

    centre = np.average(points, axis=0, weights=weights)
    for _ in range(iters):
        dist = np.maximum(np.linalg.norm(points - centre, axis=1), eps)
        w = weights / dist
        nxt = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(nxt - centre) < eps * (1.0 + np.linalg.norm(centre)):
            return nxt
        centre = nxt
    return centre


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


@dataclass
class TrackSelection:
    keep: np.ndarray  # bool (P,)
    reasons: dict[str, list[int]]
    scale: float


def select_tracks(
    xyz: np.ndarray,
    valid: np.ndarray,
    *,
    membership_ratio: float = 3.0,
    motion_ratio: float = 2.0,
    lags: tuple[int, ...] = (4, 8, 16, 31),
) -> TrackSelection:
    """Reject whole tracks that do not belong to the subject's bulk motion.

    Rejection is at *track* level, not per frame.  A bad D4RT point latched onto
    the wrong surface when it was seeded, so it is wrong for its entire track --
    and the subject is non-rigid, so a per-frame consensus would swap physical
    identity between frames (a swinging forearm is an inlier in some frames and
    an outlier in others), producing a jittery trajectory with wandering bias.

    ``xyz`` is (P, T, 3), ``valid`` is (P, T) bool.
    """

    xyz = np.asarray(xyz, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    num_points = xyz.shape[0]

    masked = np.where(valid[..., None], xyz, np.nan)
    centre = np.nanmedian(masked, axis=0)  # (T, 3)
    scale = cloud_scale(masked[np.isfinite(masked).all(axis=-1)])
    if not np.isfinite(scale) or scale <= 0:
        return TrackSelection(np.ones(num_points, bool), {"degenerate_scale": []}, scale)

    reasons: dict[str, list[int]] = {"membership": [], "bulk_motion": [], "smoothness": []}
    keep = np.ones(num_points, dtype=bool)

    # B1 -- does the point even sit on the subject?  Catches background latch-on.
    radius = np.nanmedian(np.linalg.norm(masked - centre[None], axis=-1), axis=1)
    reject = np.nan_to_num(radius, nan=np.inf) > membership_ratio * scale
    reasons["membership"] = np.flatnonzero(reject).tolist()
    keep &= ~reject

    # B2 -- does the point translate with the subject?  Evaluated at long lags,
    # where bulk motion dominates; at lag 1 limb swing would dominate instead and
    # this would over-reject.
    motion_residual = np.zeros(num_points)
    for lag in lags:
        if lag >= xyz.shape[1]:
            continue
        delta = masked[:, lag:] - masked[:, :-lag]
        consensus = np.nanmedian(delta, axis=0)
        dev = np.nanmedian(np.linalg.norm(delta - consensus[None], axis=-1), axis=1)
        motion_residual = np.maximum(motion_residual, np.nan_to_num(dev, nan=np.inf))
    cutoff = motion_ratio * scale
    reject = motion_residual > cutoff
    if keep.sum() > 2:
        # Only meaningful with a population to compare against; with everything
        # already rejected the median is NaN and every comparison silently False.
        mad_cutoff = np.nanmedian(motion_residual[keep]) + 3.0 * _mad(motion_residual[keep])
        if np.isfinite(mad_cutoff):
            reject |= motion_residual > mad_cutoff
    reasons["bulk_motion"] = np.flatnonzero(reject & keep).tolist()
    keep &= ~reject

    # B3 -- flicker.  Catches tracks that pass B2 on average but oscillate.
    second = masked[:, 2:] - 2.0 * masked[:, 1:-1] + masked[:, :-2]
    jitter = np.nanmedian(np.linalg.norm(second, axis=-1), axis=1)
    if keep.sum() > 2:
        jitter_cutoff = np.nanmedian(jitter[keep]) + 3.0 * _mad(jitter[keep])
        if np.isfinite(jitter_cutoff):
            reject = np.nan_to_num(jitter, nan=np.inf) > jitter_cutoff
            reasons["smoothness"] = np.flatnonzero(reject & keep).tolist()
            keep &= ~reject

    return TrackSelection(keep=keep, reasons=reasons, scale=scale)


def aggregate_trajectory(
    xyz: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    keep: np.ndarray,
) -> dict[str, np.ndarray]:
    """Collapse the surviving point tracks into one trajectory per frame."""

    xyz = np.asarray(xyz, dtype=np.float64)
    num_frames = xyz.shape[1]
    out = {
        "mean": np.full((num_frames, 3), np.nan),
        "median": np.full((num_frames, 3), np.nan),
        "ransac_median": np.full((num_frames, 3), np.nan),
        "n_inliers": np.zeros(num_frames, dtype=int),
    }
    for t in range(num_frames):
        frame_valid = valid[:, t] & np.isfinite(xyz[:, t]).all(axis=-1)
        if frame_valid.any():
            out["mean"][t] = xyz[frame_valid, t].mean(axis=0)
            out["median"][t] = np.median(xyz[frame_valid, t], axis=0)
        sel = frame_valid & keep
        out["n_inliers"][t] = int(sel.sum())
        if sel.any():
            out["ransac_median"][t] = geometric_median(xyz[sel, t], weights[sel, t])
    return out


def smooth_trajectory(track: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing, applied only *after* outlier rejection.

    Smoothing first would blend the outliers we are trying to detect into their
    neighbours and hide them.
    """

    from scipy.signal import savgol_filter

    track = np.asarray(track, dtype=np.float64)
    finite = np.isfinite(track).all(axis=1)
    if int(finite.sum()) < max(window, polyorder + 2):
        return track.copy()

    filled = track.copy()
    if not finite.all():
        # Interpolate the gaps so the filter sees a continuous signal, then
        # restore NaN afterwards so absent frames stay absent.
        index = np.arange(track.shape[0])
        for axis in range(3):
            filled[:, axis] = np.interp(index, index[finite], track[finite, axis])

    window = min(window, filled.shape[0] if filled.shape[0] % 2 == 1 else filled.shape[0] - 1)
    if window <= polyorder:
        return track.copy()
    smoothed = savgol_filter(filled, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    smoothed[~finite] = np.nan
    return smoothed
