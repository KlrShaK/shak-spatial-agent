"""Tests for the four bucket reports.

Two layers.  The synthetic tests pin the conventions -- which way is a positive
pan, where a quadrant boundary sits -- because those are the failures that
produce a confident wrong answer rather than an error.  The golden tests run the
real builders over the three clips in `results/traj3d_a100_20260819T100744Z` and
assert against numbers read out of the notebook, which is what keeps the module
and `testing_splines.ipynb` from drifting apart.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from d4rt_agent.sam_orientany_d4rt_test import traj3d_buckets as B
from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import REPO_ROOT

RUN_DIR = REPO_ROOT / "d4rt_agent/results/traj3d_a100_20260819T100744Z"
RUNNER = "camerabench_m0jmssq5ptw.3.12_ccd85243"
YELLOW = "656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228"
SYNFMC = "synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1"

needs_run = pytest.mark.skipif(not RUN_DIR.is_dir(), reason="saved traj3d run not present")


def _yaw(theta_deg: float) -> np.ndarray:
    """Rotation about the down axis: a camera that has panned right by theta."""
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), 0.0, math.sin(t)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(t), 0.0, math.cos(t)]])


def _pitch(phi_deg: float) -> np.ndarray:
    """Rotation about the right axis: a camera tilted UP by phi.

    Column 2 is the viewing direction and must come out with a NEGATIVE y to
    mean "up", because +y points down in the OpenCV frame.  The transpose of
    this matrix looks down, which is the easy mistake to make here.
    """
    t = math.radians(phi_deg)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, math.cos(t), -math.sin(t)],
                     [0.0, math.sin(t), math.cos(t)]])


def _roll(psi_deg: float) -> np.ndarray:
    """Rotation about the forward axis: the camera's right side dips by psi."""
    t = math.radians(psi_deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0],
                     [math.sin(t), math.cos(t), 0.0],
                     [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# conventions
# --------------------------------------------------------------------------


def test_pan_is_positive_to_the_cameras_right():
    angles = B.camera_pose_angles(np.stack([np.eye(3), _yaw(30.0)]))
    assert angles["pan_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert angles["pan_deg"][1] == pytest.approx(30.0, abs=1e-6)
    assert angles["tilt_deg"][1] == pytest.approx(0.0, abs=1e-6)
    assert angles["roll_deg"][1] == pytest.approx(0.0, abs=1e-6)


def test_tilt_is_positive_upward_despite_y_pointing_down():
    angles = B.camera_pose_angles(np.stack([np.eye(3), _pitch(20.0)]))
    assert angles["tilt_deg"][1] == pytest.approx(20.0, abs=1e-6)
    assert angles["pan_deg"][1] == pytest.approx(0.0, abs=1e-6)
    # ...and the opposite rotation must come back negative, not just non-zero.
    down = B.camera_pose_angles(np.stack([np.eye(3), _pitch(-20.0)]))
    assert down["tilt_deg"][1] == pytest.approx(-20.0, abs=1e-6)


def test_roll_is_positive_when_the_cameras_right_side_dips():
    angles = B.camera_pose_angles(np.stack([np.eye(3), _roll(15.0)]))
    assert angles["roll_deg"][1] == pytest.approx(15.0, abs=1e-6)
    assert angles["pan_deg"][1] == pytest.approx(0.0, abs=1e-6)
    assert angles["tilt_deg"][1] == pytest.approx(0.0, abs=1e-6)


def test_roll_is_nan_when_the_camera_looks_straight_down():
    """"Level" has no meaning along the vertical, so roll must decline, not guess."""
    straight_down = np.column_stack([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    angles = B.camera_pose_angles(np.stack([np.eye(3), straight_down]))
    assert math.isnan(angles["roll_deg"][1])
    assert angles["tilt_deg"][1] == pytest.approx(-90.0, abs=1e-6)


def test_angles_are_reported_as_change_from_the_start_in_every_basis():
    """Levelling makes frame 0's absolute pose non-zero; the reports still want
    a change from the start, and the absolute pose is kept separately."""
    level = B.level_basis(np.array([0.0, -1.0, 0.0]))
    angles = B.camera_pose_angles(np.stack([_pitch(30.0), _pitch(50.0)]), level=level)
    assert angles["tilt_deg"][0] == pytest.approx(0.0, abs=1e-6)
    assert angles["tilt_deg"][1] == pytest.approx(20.0, abs=1e-6)
    assert angles["_absolute_at_frame_0"]["initial_tilt_deg"] == pytest.approx(30.0, abs=1e-6)


def test_levelling_a_level_camera_changes_nothing():
    """The correction must be a no-op where the camera was already level, so
    clips that already read correctly are not disturbed."""
    assert np.allclose(B.level_basis(np.array([0.0, -1.0, 0.0])), np.eye(3), atol=1e-9)


def test_levelling_moves_a_pitched_cameras_descent_onto_the_vertical():
    """The staircase case: a camera pitched 50 deg down, descending vertically,
    splits that descent almost equally between its own forward and down axes."""
    pitch = math.radians(50.0)
    world_up_in_cam = np.array([0.0, -math.cos(pitch), math.sin(pitch)])
    basis = B.level_basis(world_up_in_cam)
    assert basis is not None

    # A purely vertical descent, written in the tilted camera's own coordinates.
    descent_world_down = np.array([0.0, math.cos(pitch), -math.sin(pitch)]) * 2.6
    in_camera = B.axis_components(descent_world_down)
    assert abs(in_camera["forward"]) > 1.0 and abs(in_camera["down"]) > 1.0  # a near-tie

    levelled = descent_world_down @ basis
    assert levelled[1] == pytest.approx(2.6, abs=1e-6)          # all of it is down
    assert abs(levelled[2]) < 1e-6                              # none of it is forward


def test_a_missing_world_up_leaves_the_camera_frame_and_says_so():
    """No gravity must mean a stated caveat, never a silent guess."""
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_geometry import estimate_world_up

    rng = np.random.default_rng(0)
    blob = rng.normal(size=(200, 3))          # a ball has no dominant plane
    up, fraction = estimate_world_up(blob)
    assert up is None


def test_failed_poses_arrive_as_nan_not_as_a_measurement():
    rotation = np.stack([np.eye(3), np.full((3, 3), np.nan)])
    angles = B.camera_pose_angles(rotation)
    assert math.isnan(angles["pan_deg"][1])


def test_quadrant_boundaries_match_the_benchmarks_vocabulary():
    assert B.quadrant_sector(0.0)[0] == "front"
    assert B.quadrant_sector(44.0)[0] == "front"
    assert B.quadrant_sector(46.0)[0] == "right"
    assert B.quadrant_sector(90.0)[0] == "right"
    assert B.quadrant_sector(179.0)[0] == "back"
    assert B.quadrant_sector(-179.0)[0] == "back"
    assert B.quadrant_sector(-90.0)[0] == "left"
    assert B.quadrant_sector(-46.0)[0] == "left"
    assert B.quadrant_sector(-44.0)[0] == "front"


def test_quadrant_and_octant_agree_about_which_side():
    for bearing in range(-180, 180, 7):
        quadrant = B.quadrant_sector(float(bearing))[0]
        octant = B.octant_sector(float(bearing))[0]
        assert quadrant in octant or octant in quadrant, (bearing, quadrant, octant)


def test_direction_words_use_the_opencv_axes():
    assert B.direction_name([1, 0, 0]) == "right"
    assert B.direction_name([-1, 0, 0]) == "left"
    assert B.direction_name([0, 1, 0]) == "down"      # +y is down
    assert B.direction_name([0, -1, 0]) == "up"
    assert B.direction_name([0, 0, 1]) == "forward"
    assert B.direction_name([1, 0, 1]) == "right-forward"
    assert B.direction_name([0, 0, 0]) == "stationary"


# --------------------------------------------------------------------------
# noise floors -- the regressions that matter most
# --------------------------------------------------------------------------


def test_static_track_is_stationary_because_the_floor_is_independent_of_it():
    """The bug this guards: a floor derived from the track's own extent shrinks
    with the signal, so a static camera can never be called static."""
    rng = np.random.default_rng(0)
    jitter = rng.normal(scale=1e-3, size=(32, 3))
    raw = jitter
    smooth = B.running_mean(raw)
    floor = B.displacement_noise(raw, smooth, floor=0.01 * 1.4)  # 1% of scene scale
    report = B.segment_report(smooth, {}, noise=floor)
    assert not report["net_above_noise"]
    assert report["net_displacement"] < floor


def test_real_motion_survives_the_floor():
    track = np.stack([np.linspace(0, 2, 32), np.zeros(32), np.zeros(32)], axis=1)
    floor = B.displacement_noise(track, B.running_mean(track), floor=0.014)
    report = B.segment_report(B.running_mean(track), {}, noise=floor)
    assert report["net_above_noise"]
    assert all(row["direction"] == "right" for row in report["segments"])


def test_angles_are_not_extrapolated_past_where_they_were_measured():
    """An orientation resolved only late in the clip must not be reported as
    spanning it -- on the runner clip OAV2 resolves a front from frame 28.

    Sampled at eleven uniform times over frames 0-31, only the last one lands
    inside a frames-28-31 window, so there is not even one interval to difference
    and the honest total is None rather than a number covering 3 of 32 frames.
    """
    track = np.stack([np.linspace(0, 1, 32), np.zeros(32), np.zeros(32)], axis=1)

    late = np.full(32, np.nan)
    late[28:] = [0.0, 1.0, 2.0, 3.0]
    report = B.segment_report(track, {"heading_deg": late}, noise=0.0)
    assert report["angle_totals"]["heading_deg"] is None
    assert report["angle_spans"]["heading_deg"] is None
    assert report["full_span"] == [0.0, 31.0]

    # A window wide enough to hold several samples does report, but the span it
    # reports is its own, not the clip's.
    partial = np.full(32, np.nan)
    partial[16:] = np.linspace(0.0, 8.0, 16)
    report = B.segment_report(track, {"heading_deg": partial}, noise=0.0)
    span = report["angle_spans"]["heading_deg"]
    assert span is not None and span[0] >= 16.0 and span[1] <= 31.0
    assert report["angle_totals"]["heading_deg"] is not None


def test_wrapped_angles_do_not_invent_a_full_turn_at_the_seam():
    angles = np.array([179.0, -179.0] + [np.nan] * 30)
    sampled = B._interp_angles(np.array([0.0, 0.5, 1.0]), angles)
    assert abs(sampled[-1] - sampled[0]) == pytest.approx(2.0, abs=1e-6)


def test_running_mean_keeps_gaps_absent():
    track = np.zeros((10, 3))
    track[4] = np.nan
    out = B.running_mean(track)
    assert np.isnan(out[4]).all()
    assert np.isfinite(out[[0, 3, 5, 9]]).all()


def test_subject_heading_is_relative_to_the_first_resolved_facing():
    forward = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    usable = np.array([False, True, True])
    angles = B.subject_heading_angles(forward, usable)
    assert math.isnan(angles["heading_deg"][0])       # not resolved -> no answer
    assert angles["heading_deg"][1] == pytest.approx(0.0, abs=1e-9)   # the reference
    assert angles["heading_deg"][2] == pytest.approx(-90.0, abs=1e-6)


# --------------------------------------------------------------------------
# golden: the real clips, against the notebook's numbers
# --------------------------------------------------------------------------


@needs_run
@pytest.mark.parametrize("slug", [RUNNER, YELLOW, SYNFMC])
def test_every_builder_returns_a_report_and_never_raises(slug):
    inputs = B.load_bucket_inputs(RUN_DIR / slug)
    for bucket in B.BUILDERS:
        report = B.build_report(bucket, inputs)
        assert report["bucket"] == bucket
        assert isinstance(report["text"], str) and report["text"]
        assert report["available"] is (report["unavailable_reason"] is None)


@needs_run
def test_distance_change_matches_the_notebook():
    report = B.sub_obs_distance_change(B.load_bucket_inputs(RUN_DIR / RUNNER))
    distances = [row["distance"] for row in report["metrics"]["checkpoints"]]
    assert distances[0] == pytest.approx(5.317, abs=5e-3)
    assert distances[2] == pytest.approx(3.635, abs=5e-3)
    # The runner approaches: its mask grows 213 -> 1152 px across the clip.
    assert all(row["trend_from_previous"] == "shrank" for row in report["metrics"]["checkpoints"][1:])


@needs_run
def test_static_camera_clip_reports_stationary():
    report = B.traj_camera_own_frame(B.load_bucket_inputs(RUN_DIR / RUNNER))
    assert report["available"]
    assert not report["metrics"]["net_above_noise"]
    assert "STATIONARY" in report["text"]


@needs_run
def test_moving_camera_clip_does_not_report_stationary():
    report = B.traj_camera_own_frame(B.load_bucket_inputs(RUN_DIR / SYNFMC))
    assert report["metrics"]["net_above_noise"]
    assert "STATIONARY" not in report["text"]


@needs_run
def test_orbit_signature_is_lateral_translation_with_compensating_pan():
    """The yellow-jacket clip tracks left while panning right -- the pan that
    keeps a subject centred is exactly what separates an orbit from a dolly."""
    report = B.traj_camera_own_frame(B.load_bucket_inputs(RUN_DIR / YELLOW))
    pan = report["metrics"]["angle_totals"]["pan_deg"]
    lateral = [row for row in report["metrics"]["runs"] if "left" in row["direction"]]
    assert pan > 5.0
    assert lateral, "expected leftward translation on this clip"


@needs_run
def test_unresolved_facing_declines_rather_than_substituting_camera_axes():
    """Yellow-jacket resolves alpha in {0,2,4} on every frame, never 1."""
    inputs = B.load_bucket_inputs(RUN_DIR / YELLOW)
    for bucket in ("traj_subject_own_frame", "orient_camera_subject_frame"):
        report = B.build_report(bucket, inputs)
        assert not report["available"]
        assert "UNAVAILABLE" in report["text"]
    # The two buckets that need no facing still work on the same clip.
    assert B.build_report("sub_obs_distance_change", inputs)["available"]
    assert B.build_report("traj_camera_own_frame", inputs)["available"]


@needs_run
def test_camera_only_clip_needs_no_trajectory_file(tmp_path):
    """A camera-bucket question never segments a subject, so `trajectory.npz`
    is simply absent and the subject-bearing reports must decline, not crash."""
    analysis = tmp_path / "analysis"
    analysis.mkdir(parents=True)
    source = RUN_DIR / SYNFMC / "analysis"
    (analysis / "camera_poses.npz").write_bytes((source / "camera_poses.npz").read_bytes())
    (analysis / "summary.json").write_bytes((source / "summary.json").read_bytes())

    inputs = B.load_bucket_inputs(tmp_path)
    assert not inputs.has_subject
    assert B.build_report("traj_camera_own_frame", inputs)["available"]
    assert not B.build_report("sub_obs_distance_change", inputs)["available"]


@needs_run
def test_sector_transition_is_reported_with_its_boundary_margin():
    report = B.orient_camera_subject_frame(B.load_bucket_inputs(RUN_DIR / SYNFMC))
    checkpoints = report["metrics"]["checkpoints"]
    assert checkpoints[0]["quadrant"] == "right"
    assert checkpoints[-1]["quadrant"] == "back"
    assert "right side to its back" in report["text"]
    # This clip ends 1.6 deg from a boundary; that has to be said out loud.
    assert checkpoints[-1]["quadrant_margin_deg"] < 10.0
    assert "not robust" in report["text"]
