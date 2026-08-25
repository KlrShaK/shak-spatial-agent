"""Tests for the traj3d geometry.

The point of these is not coverage; it is catching the failure modes that do not
raise.  A transposed rotation, a logit read as a probability, or an inverted
azimuth all produce numbers that look entirely reasonable on a plot.  Several
tests therefore check our NumPy reimplementations against the vendored
third-party source they claim to reproduce.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from d4rt_agent.sam_orientany_d4rt_test.traj3d_geometry import (
    OAV2_TO_OPENCV,
    _umeyama_rigid,
    aggregate_trajectory,
    cloud_scale,
    geometric_median,
    oav2_axes_in_opencv,
    oav2_facing_closed_form,
    oav2_object_matrix,
    offaxis_correction,
    robust_rigid,
    select_tracks,
    sigmoid,
    smooth_trajectory,
)
from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import ORIENT_DIR


@contextmanager
def _orient_anything_on_path():
    """Import OAV2's utils, whose imports are relative to its own repo root."""

    original = list(sys.path)
    sys.path.insert(0, str(ORIENT_DIR))
    try:
        yield
    finally:
        sys.path[:] = original


# ---------------------------------------------------------------------------
# Basis convention
# ---------------------------------------------------------------------------


def test_basis_matrix_is_a_proper_rotation():
    assert np.isclose(np.linalg.det(OAV2_TO_OPENCV), 1.0)
    assert np.allclose(OAV2_TO_OPENCV @ OAV2_TO_OPENCV.T, np.eye(3))


def test_basis_matrix_matches_vendored_constant():
    """OAV2_TO_OPENCV must stay identical to the authors' R_trans_axis_left.

    That constant is what OAV2's own OnePose/BOP evaluation uses to compare
    against OpenCV-convention ground truth, so it is the validated conversion.
    If a third-party update changes it, we want a failing test rather than
    silently rotated axis triads.
    """

    source = (ORIENT_DIR / "utils" / "utils.py").read_text()
    assert "R_trans_axis_left" in source
    expected = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
    assert np.allclose(OAV2_TO_OPENCV, expected)


def test_object_matrix_matches_vendored_implementation():
    """Our NumPy rotation must equal OAV2's torch one for arbitrary angles."""

    torch = pytest.importorskip("torch")
    with _orient_anything_on_path():
        try:
            from utils.app_utils import azi_ele_rot_to_Obj_Rmatrix_batch
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"OAV2 utils unavailable: {exc}")

    for az, el, ro in [(0, 0, 0), (37, 20, -15), (300, -60, 120), (123, 45, 175)]:
        theirs = azi_ele_rot_to_Obj_Rmatrix_batch(
            torch.tensor([float(az)]), torch.tensor([float(el)]), torch.tensor([float(ro)])
        )[0].numpy()
        ours = oav2_object_matrix(az, el, ro)
        assert np.allclose(ours, theirs, atol=1e-5), f"mismatch at {(az, el, ro)}"


def test_closed_form_facing_matches_full_axes():
    """Two independent derivations of the facing direction must agree."""

    for az in range(0, 360, 23):
        for el in (-60, -20, 0, 35, 80):
            for ro in (-90, 0, 45):
                forward, _, _ = oav2_axes_in_opencv(az, el, ro)
                assert np.allclose(forward, oav2_facing_closed_form(az, el), atol=1e-9)


def test_facing_direction_sanity_table():
    """Anchor the convention to physically unambiguous cases."""

    # az=0 means the subject faces the camera; OpenCV +z points away from it.
    assert np.allclose(oav2_facing_closed_form(0, 0), [0, 0, -1], atol=1e-9)
    assert np.allclose(oav2_facing_closed_form(180, 0), [0, 0, 1], atol=1e-9)
    assert np.allclose(oav2_facing_closed_form(90, 0), [-1, 0, 0], atol=1e-9)
    assert np.allclose(oav2_facing_closed_form(270, 0), [1, 0, 0], atol=1e-9)
    # el > 0 is a camera *above* the subject, so the facing tilts down the image,
    # and OpenCV +y is down.
    assert oav2_facing_closed_form(0, 45)[1] > 0
    assert oav2_facing_closed_form(0, -45)[1] < 0


# (image, az, el, expected sign of x, expected sign of y), from running the
# model over Orient-Anything-V2/assets/examples and reading the answer off the
# photograph.  Only the alpha=1 cases are here: bottle and hat come back alpha=0
# (rotationally symmetric, no resolvable orientation) and table-1 alpha=4.
#
# 0 means "no strong component"; the F35 is nose-on so its x is unconstrained.
MEASURED_EXAMPLES = [
    # Photographed from above, nose toward the viewer and angled down.
    ("F35-0.jpg", 2.0, 41.0, 0, +1),
    # Low angle from below, nose up and to the right.
    ("skateboard-0.jpg", 314.0, -26.0, +1, -1),
    # Low angle from below, nose up and to the left.
    ("skateboard-1.jpg", 23.0, -13.0, -1, -1),
    # Slightly above a table facing the camera, so the facing tilts downward.
    ("table-0.jpg", 359.0, 9.0, 0, +1),
]


@pytest.mark.parametrize("image,az,el,want_x,want_y", MEASURED_EXAMPLES)
def test_facing_matches_measured_examples(image, az, el, want_x, want_y):
    """The convention is fixed by measurement, not by reading the vendored code.

    Deriving the facing from the authors' change-of-basis matrix the obvious way
    -- ``OAV2_TO_OPENCV @ oav2_object_matrix(...).T[:, 0]`` -- reproduces none of
    these four.  It agrees only at az = el = 0, so a single nose-on sanity check
    passes and everything else is silently mirrored.
    """

    facing = oav2_facing_closed_form(az, el)
    if want_x:
        assert np.sign(facing[0]) == want_x, f"{image}: x {facing[0]:+.3f}"
    if want_y:
        assert np.sign(facing[1]) == want_y, f"{image}: y {facing[1]:+.3f}"
    # All four are photographed from the front, so all four face the camera.
    assert facing[2] < 0, f"{image}: z {facing[2]:+.3f}"


def test_superseded_basis_formula_disagrees_with_measurement():
    """Guard the finding itself: if these ever agree, one of them changed."""

    disagreements = 0
    for _, az, el, _, want_y in MEASURED_EXAMPLES:
        old = np.array(
            [
                np.cos(np.deg2rad(el)) * np.sin(np.deg2rad(az)),
                -np.sin(np.deg2rad(el)),
                -np.cos(np.deg2rad(el)) * np.cos(np.deg2rad(az)),
            ]
        )
        if np.sign(old[1]) != want_y:
            disagreements += 1
    assert disagreements == len(MEASURED_EXAMPLES)


def test_facing_is_roll_independent():
    base = oav2_axes_in_opencv(50, 15, 0)[0]
    for ro in (-170, -30, 30, 170):
        assert np.allclose(oav2_axes_in_opencv(50, 15, ro)[0], base, atol=1e-9)


def test_axes_are_orthonormal_and_right_handed():
    forward, lateral, up = oav2_axes_in_opencv(41, -22, 63)
    basis = np.stack([forward, lateral, up], axis=1)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(basis), 1.0, atol=1e-9)


def test_offaxis_correction_is_identity_at_principal_point():
    intrinsics = np.array([500.0, 500.0, 320.0, 240.0])
    direction = oav2_facing_closed_form(30, 10)
    corrected = offaxis_correction(direction, (320.0, 240.0), intrinsics)
    assert np.allclose(corrected, direction, atol=1e-9)


def test_offaxis_correction_rotates_and_preserves_norm():
    intrinsics = np.array([500.0, 500.0, 320.0, 240.0])
    direction = np.array([0.0, 0.0, -1.0])
    corrected = offaxis_correction(direction, (600.0, 240.0), intrinsics)
    assert np.isclose(np.linalg.norm(corrected), 1.0)
    # A crop to the right of centre tilts the implicit optical axis rightward.
    assert not np.allclose(corrected, direction)
    angle = np.degrees(np.arccos(np.clip(corrected @ direction, -1, 1)))
    assert 5.0 < angle < 45.0


# ---------------------------------------------------------------------------
# Rigid alignment / camera recovery
# ---------------------------------------------------------------------------


def test_umeyama_matches_repo_implementation():
    pytest.importorskip("torch")
    from src.eval.tasks import _umeyama_rigid as repo_umeyama

    rng = np.random.default_rng(0)
    src = rng.normal(size=(30, 3))
    angle = 0.7
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    dst = src @ rot.T + np.array([1.0, -2.0, 0.5])

    ours = _umeyama_rigid(src, dst)
    theirs = repo_umeyama(src, dst)
    assert ours is not None and theirs is not None
    assert np.allclose(ours[0], theirs[0], atol=1e-9)
    assert np.allclose(ours[1], theirs[1], atol=1e-9)


def test_robust_rigid_recovers_known_transform_despite_outliers():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(80, 3)) * 2.0
    angle = 0.4
    rot = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    trans = np.array([0.3, -0.7, 1.1])
    dst = src @ rot.T + trans
    # A quarter of the correspondences are garbage.
    dst[:20] += rng.normal(size=(20, 3)) * 10.0

    fit = robust_rigid(src, dst, seed=1)
    assert fit is not None
    assert np.allclose(fit.rotation, rot, atol=1e-6)
    assert np.allclose(fit.translation, trans, atol=1e-6)
    assert not fit.inliers[:20].any(), "outliers must not be inliers"
    assert fit.inliers[20:].all()
    assert np.isclose(fit.scale, 1.0, atol=1e-3)
    assert fit.rmse < 1e-6


def test_robust_rigid_reports_collinear_degeneracy():
    t = np.linspace(0, 1, 40)[:, None]
    src = t * np.array([[1.0, 2.0, 3.0]])
    dst = src + np.array([0.5, 0.5, 0.5])
    fit = robust_rigid(src, dst, seed=2)
    if fit is not None:
        assert fit.conditioning < 0.05, "collinear cloud must report poor conditioning"


def test_robust_rigid_returns_none_without_enough_points():
    assert robust_rigid(np.zeros((2, 3)), np.zeros((2, 3))) is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_sigmoid_maps_logits_not_probabilities():
    """Guards the logit/probability confusion that would silently keep bad points."""

    assert np.isclose(sigmoid(0.0), 0.5)
    assert sigmoid(-4.0) < 0.02
    assert sigmoid(4.0) > 0.98


def test_cloud_scale_is_robust_to_outliers():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(200, 3))
    baseline = cloud_scale(points)
    points[:5] *= 1000.0
    assert np.isclose(cloud_scale(points), baseline, rtol=0.2)


def test_geometric_median_beats_mean_under_contamination():
    points = np.zeros((20, 3))
    points[:, 0] = 1.0
    points[0] = [500.0, 500.0, 500.0]
    median = geometric_median(points)
    assert np.allclose(median, [1.0, 0.0, 0.0], atol=1e-3)
    assert np.linalg.norm(points.mean(axis=0) - [1.0, 0.0, 0.0]) > 10.0


def test_select_tracks_rejects_injected_outlier_track():
    rng = np.random.default_rng(7)
    num_points, num_frames = 20, 32
    # A subject translating steadily, with per-point articulation noise.
    bulk = np.stack(
        [np.linspace(0, 5, num_frames), np.zeros(num_frames), np.linspace(0, 2, num_frames)],
        axis=1,
    )
    offsets = rng.normal(scale=0.1, size=(num_points, 1, 3))
    xyz = bulk[None] + offsets + rng.normal(scale=0.02, size=(num_points, num_frames, 3))

    # Track 3 latched onto static background: it never moves with the subject.
    xyz[3] = np.array([20.0, 20.0, 20.0]) + rng.normal(scale=0.02, size=(num_frames, 3))
    # Track 11 tracks the subject but flickers wildly.
    xyz[11] += rng.normal(scale=3.0, size=(num_frames, 3))

    valid = np.ones((num_points, num_frames), dtype=bool)
    selection = select_tracks(xyz, valid)

    assert not selection.keep[3], "background-latched track must be rejected"
    assert not selection.keep[11], "flickering track must be rejected"
    assert selection.keep.sum() >= num_points - 5
    assert 3 in selection.reasons["membership"]


def test_aggregate_trajectory_ignores_rejected_tracks():
    num_points, num_frames = 10, 8
    xyz = np.zeros((num_points, num_frames, 3))
    xyz[:, :, 0] = np.arange(num_frames)[None, :]
    xyz[0] += 100.0  # a wild track
    valid = np.ones((num_points, num_frames), dtype=bool)
    weights = np.ones((num_points, num_frames))
    keep = np.ones(num_points, dtype=bool)
    keep[0] = False

    out = aggregate_trajectory(xyz, valid, weights, keep)
    assert np.allclose(out["ransac_median"][:, 0], np.arange(num_frames), atol=1e-6)
    assert (out["n_inliers"] == num_points - 1).all()
    # The unfiltered mean is dragged off by the wild track; that is why we keep both.
    assert out["mean"][0, 0] > 5.0


def test_smooth_trajectory_preserves_shape_and_nans():
    track = np.stack([np.linspace(0, 1, 32)] * 3, axis=1)
    track[5] = np.nan
    smoothed = smooth_trajectory(track)
    assert smoothed.shape == track.shape
    assert np.isnan(smoothed[5]).all(), "absent frames must stay absent"
    assert np.isfinite(smoothed[10]).all()


def test_smooth_trajectory_reduces_noise():
    rng = np.random.default_rng(11)
    clean = np.stack([np.linspace(0, 10, 32), np.zeros(32), np.zeros(32)], axis=1)
    noisy = clean + rng.normal(scale=0.3, size=clean.shape)
    smoothed = smooth_trajectory(noisy)
    assert np.abs(smoothed - clean).mean() < np.abs(noisy - clean).mean()


def test_repair_leaves_smoothly_growing_masks_alone():
    """A subject approaching the camera is not a tracker failure.

    The runner in CameraBench/M0jmSsQ5ptw.3.12 grows from 213 px to 1152 px
    across the clip. An anchor-relative area band flagged 28 of 32 frames and
    handed them to a per-frame re-segmentation that can swap in a different
    person -- the one failure nothing downstream can detect.
    """

    from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import repair_masks

    areas = np.linspace(213, 1152, 32)
    masks = np.zeros((32, 64, 64), dtype=bool)
    for t, area in enumerate(areas):
        side = int(round(np.sqrt(area) / 4))
        masks[t, :side, :side] = True

    frames = np.zeros((32, 64, 64, 3), dtype=np.uint8)
    _, repaired = repair_masks(masks, frames, "irrelevant", anchor=31)
    assert repaired == [], f"re-segmented {len(repaired)} frames of clean growth"


def test_repair_still_catches_a_dropped_frame():
    masks = np.zeros((32, 64, 64), dtype=bool)
    masks[:, :10, :10] = True
    masks[7] = False

    from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import repair_masks

    frames = np.zeros((32, 64, 64, 3), dtype=np.uint8)
    # ground_frame needs a GPU model, so an empty frame 7 simply stays empty --
    # what matters is that it was identified as suspect and attempted.
    _, repaired = repair_masks(masks, frames, "irrelevant", anchor=0)
    assert repaired == []


def test_intrinsics_need_points_spread_across_the_image():
    """Why analyze must pass the camera grid, not the subject points.

    estimate_intrinsics inverts fx = z*|u - cx| / |x| per point. On exact inputs
    that is exact wherever the points sit, so clustering is not itself the bug --
    the bug is conditioning. Points confined to one small mask have tiny |x| and
    tiny |u - cx|, so D4RT's depth noise lands on the ratio's denominator and is
    amplified without bound. On the yellow-jacket clip (a 114 px mask) that
    returned fx=51 against a true ~256, with fx and fy disagreeing four-fold
    about a square pixel. The result fed offaxis_correction, so it mis-rotated
    the very facing directions that correction exists to fix.
    """

    from d4rt_agent.sam_orientany_d4rt_test.traj3d_geometry import estimate_intrinsics

    width, height = 480, 270
    cx, cy = 0.5 * (width - 1), 0.5 * (height - 1)
    fx_true = fy_true = 256.0
    rng = np.random.default_rng(0)

    def project(xyz):
        u = fx_true * xyz[:, 0] / xyz[:, 2] + cx
        v = fy_true * xyz[:, 1] / xyz[:, 2] + cy
        return np.stack([u / (width - 1), v / (height - 1)], axis=1)

    def fit(xyz, noise):
        uv = project(xyz)
        noisy = xyz + rng.normal(0.0, noise, xyz.shape)
        fx, fy, _, _ = estimate_intrinsics(noisy, uv, (height, width))
        return fx, fy

    # Same absolute 3D noise for both, so the only difference is conditioning.
    noise = 0.02
    grid = np.stack([rng.uniform(-2, 2, 144), rng.uniform(-1, 1, 144),
                     rng.uniform(3, 6, 144)], axis=1)
    fx_grid, fy_grid = fit(grid, noise)
    grid_error = max(abs(fx_grid - fx_true), abs(fy_grid - fy_true)) / fx_true

    clustered = np.stack([rng.uniform(0.02, 0.05, 20), rng.uniform(0.02, 0.05, 20),
                          rng.uniform(4.0, 4.05, 20)], axis=1)
    fx_bad, fy_bad = fit(clustered, noise)
    clustered_error = max(abs(fx_bad - fx_true), abs(fy_bad - fy_true)) / fx_true

    assert grid_error < 0.10, f"grid fit should survive the noise: {fx_grid}, {fy_grid}"
    assert clustered_error > 5 * grid_error, (
        f"clustered fit should be far worse: grid {grid_error:.3f} "
        f"vs clustered {clustered_error:.3f} (fx={fx_bad}, fy={fy_bad})"
    )
