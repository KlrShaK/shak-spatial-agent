"""Small synthetic trajectories; no models, clips, or GPU are needed."""

import inspect

import numpy as np
import pytest

from .reports import Report, load_contract, motion_report, phase_rows, segment_motion


def trajectory(deltas):
    return np.vstack([np.zeros(3), np.cumsum(deltas, axis=0)])


def test_still_then_forward_majority_and_minor_phase():
    times = np.linspace(0, 2, 21)
    points = trajectory([[0, 0, 0]] * 5 + [[0, 0, .2]] * 15)
    report = motion_report("runner — starting frame", times, points, .05)
    assert [p["label"] for p in report.raw["phases"]] == ["still", "forward"]
    assert report.headline.startswith("Moves FORWARD")
    assert "STILL at 0.0–0.5 s" in report.headline
    assert report.rows[0] == ("0.0–0.5 s", "still", "0.00")
    assert "comparable within this clip only" in report.render()


def test_early_left_then_right_orders_majority_before_earlier_minority():
    points = trajectory([[-.2, 0, 0]] * 4 + [[.2, 0, 0]] * 16)
    report = motion_report("runner", np.arange(21) * .1, points, .05)
    assert [r[1] for r in report.rows] == ["left", "right"]
    assert report.headline.startswith("Moves RIGHT")
    assert "LEFT at 0.0–0.4 s" in report.headline


def test_out_and_back_is_motion_even_when_net_is_zero():
    points = trajectory([[0, 0, .2]] * 10 + [[0, 0, -.2]] * 10)
    report = motion_report("camera", np.arange(21), points, .05)
    assert [r[1] for r in report.rows] == ["forward", "backward"]
    assert "STILL" not in report.headline
    assert "BACKWARD at 10.0–20.0 s" in report.headline
    assert np.allclose(points[-1], 0)


def test_jittery_static_yields_one_still_phase():
    rng = np.random.default_rng(41)
    report = motion_report("camera", np.linspace(0, 3, 101), rng.uniform(-.005, .005, (101, 3)), .05)
    assert len(report.rows) == 1
    assert report.rows[0][1] == "still"
    assert report.headline.startswith("STILL throughout")


def test_row_cap_preserves_every_strong_brief_phase():
    points = trajectory([[(-1) ** i, 0, 0] for i in range(20)])
    phases = segment_motion(np.arange(21), points, .05)
    rows = phase_rows(phases)
    assert len(phases) == 20
    assert len(rows) == 6
    text = "\n".join(" ".join(row) for row in rows)
    for phase in phases:
        assert f"{phase.label} at {phase.span}" in text


def test_weak_brief_phase_is_absorbed_but_strong_one_survives():
    deltas = [[.2, 0, 0]] * 9 + [[-.06, 0, 0]] + [[.2, 0, 0]] * 10
    assert len(segment_motion(np.arange(21), trajectory(deltas), .05)) == 1
    deltas[9] = [-.2, 0, 0]
    assert [p.label for p in segment_motion(np.arange(21), trajectory(deltas), .05)] == ["right", "left", "right"]


def test_nonuniform_actual_times_are_interpolated_in_seconds():
    points = [[0, 0, 0], [0, 0, .1], [0, 0, 2]]
    report = motion_report("runner", [0, .1, 2], points, .01)
    assert report.rows[0][0] == "0.0–2.0 s"
    assert report.raw["phases"][0]["distance"] == pytest.approx(2)


def test_unknown_fps_requires_and_preserves_shown_frame_labels():
    points = [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
    with pytest.raises(ValueError, match="shown frames"):
        motion_report("runner", None, points, .05)
    report = motion_report("runner", None, points, .05,
                           time_labels=["shown frame A", "shown frame B", "shown frame C"])
    assert report.rows[0][0] == "shown frame A–shown frame C"
    assert "0.0 s" not in report.render()
    assert "sample order" in report.render()


def test_units_require_named_metric_source():
    arguments = ("runner", [0, 1], [[0, 0, 0], [0, 0, 2]], .05)
    with pytest.raises(ValueError, match="metric scale source"):
        motion_report(*arguments, units="metres")
    report = motion_report(*arguments, units="metres", metric_source="Point4D × DA3")
    assert "Distances are metres; scale source: Point4D × DA3" in report.render()


def test_unavailable_has_only_failure_and_no_measurements():
    report = Report.unavailable("track runner", "no usable positions", raw={"position": 314159})
    assert report.render() == "track runner — unavailable: no usable positions"
    assert "314159" not in report.render()


def test_full_and_short_contracts_and_raw_data_are_separate():
    report = Report("camera", "Moves LEFT.", reading_full="Full orbit sign rule.",
                    reading_short="Axes unchanged.", raw={"option": "secret answer choice"})
    assert "Full orbit sign rule" in report.render()
    assert "Axes unchanged" in report.render(full=False)
    assert "Full orbit sign rule" not in report.render(full=False)
    assert "secret answer choice" not in report.render()
    assert "options" not in inspect.signature(motion_report).parameters


@pytest.mark.parametrize("name", ["traj_camera_own_frame", "traj_subject_own_frame",
                                    "sub_obs_distance_change", "orient_camera_subject_frame"])
def test_full_contracts_are_option_blind(name):
    contract = load_contract(name)
    assert "option" not in contract.lower()
    assert "10 equal" not in contract


def test_design_example_renderer_matches_fixture():
    # The example's net vector and phase distances are inconsistent, so this is
    # a rendering fixture; physically consistent tracks are tested separately.
    report = Report(
        "subject_track — the runner, in its own starting frame (forward = where it faced at 0.9 s)",
        "Moves FORWARD most of the clip: net 1.73 forward, 0.41 right (noise 0.05).\n"
        "Still for the first 0.3 s; turns 3° right only in the last 0.3 s.",
        rows=[("0.0–0.3 s", "still", "0.05", "–"),
              ("0.3–2.8 s", "forward", "1.48", "–"),
              ("2.8–3.1 s", "forward", "0.11", "+3° right")],
        columns=("time", "direction", "moved", "turn"),
        reading_full='forward/right are the runner\'s own axes, not the screen\'s. "still" means the motion\n'
                     'in that phase stayed below the noise floor.',
    )
    expected = '''subject_track — the runner, in its own starting frame (forward = where it faced at 0.9 s)
Moves FORWARD most of the clip: net 1.73 forward, 0.41 right (noise 0.05).
Still for the first 0.3 s; turns 3° right only in the last 0.3 s.
time       direction  moved  turn
0.0–0.3 s  still      0.05   –
0.3–2.8 s  forward    1.48   –
2.8–3.1 s  forward    0.11   +3° right
Reading: forward/right are the runner's own axes, not the screen's. "still" means the motion
in that phase stayed below the noise floor.'''
    assert report.render() == expected


def test_coherent_slow_drift_accumulates_above_displacement_floor():
    points = np.zeros((21, 3))
    points[:, 2] = np.linspace(0, .8, 21)
    report = motion_report("runner", np.linspace(0, 2, 21), points, .05)
    assert [p["label"] for p in report.raw["phases"]] == ["forward"]
    assert report.headline.startswith("Moves FORWARD")
    assert report.raw["phases"][0]["distance"] == pytest.approx(.8)


def test_brief_excursion_between_equal_time_samples_survives():
    points = np.zeros((101, 3))
    points[2, 2] = 1
    report = motion_report("runner", np.linspace(0, 1, 101), points, .05)
    labels = [p["label"] for p in report.raw["phases"]]
    assert "forward" in labels
    assert "backward" in labels
    assert "FORWARD at" in report.headline
    assert "BACKWARD at" in report.headline
    assert sum(p["distance"] for p in report.raw["phases"]) == pytest.approx(2)


def test_diagonal_above_euclidean_floor_counts_as_motion():
    report = motion_report("runner", [0, 1], [[0, 0, 0], [.04, .04, .04]], .05)
    assert report.raw["phases"][0]["label"] == "right+down+forward"
    assert "Euclidean displacement" in report.reading_full


def test_slow_drift_with_small_position_noise_stays_motion():
    rng = np.random.default_rng(3)
    points = rng.uniform(-.004, .004, (101, 3))
    points[:, 2] += np.linspace(0, .8, 101)
    report = motion_report("runner", np.linspace(0, 2, 101), points, .05)
    assert report.raw["majority"] == "forward"
