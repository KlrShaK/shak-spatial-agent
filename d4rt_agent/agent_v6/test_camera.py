"""GPU-free camera semantics, with synthetic wall geometry and known pose paths."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from d4rt_agent.agent_v5.synthetic import build_scene, rot_x, rot_y
from .tools.camera import check_scene_revisit, track_camera, track_camera_zoom
from .workspace import Workspace


def rot_z(degrees):
    a = np.radians(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def workspace(**kwargs):
    scene, _ = build_scene(**kwargs)
    clip = SimpleNamespace(fps=scene.fps, length=scene.length)
    ws = Workspace(clip, scene=scene)
    if scene.fps is None:
        ws.mark_shown(0)
        ws.mark_shown(scene.length - 1)
    return ws


def test_pure_roll_is_motion_with_size_and_floor():
    report = track_camera(workspace(frames=40, fps=13, camera_R=lambda t: rot_z(2 * t)))
    assert report.status == "ok"
    assert report.headline.startswith("Camera ROLLS CLOCKWISE")
    assert "+78.0°; noise 2.0°" in report.headline
    assert report.raw["majority"] == "rolls clockwise"
    assert report.rows[0][1] == "still"
    assert report.rows[0][-1] == "+78.0°"


def test_out_and_back_keeps_both_directions_and_nonzero_path_speed():
    ws = workspace(frames=61, fps=10, camera_t=lambda t: (0, 0, min(t, 60 - t) / 15))
    report = track_camera(ws)
    assert {p["translation"] for p in report.raw["phases"]} == {"forward", "backward"}
    assert report.raw["path_distance"] == pytest.approx(4)
    assert report.raw["net_displacement"] == pytest.approx(0)
    assert "path speed 0.667" in report.headline
    assert "net displacement rate 0.000" in report.headline


def test_left_translation_and_right_pan_follow_clockwise_orbit_contract():
    ws = workspace(frames=31, fps=10, camera_t=lambda t: (-t / 20, 0, 0), camera_R=rot_y)
    report = track_camera(ws)
    assert report.headline.startswith("Camera MOVES LEFT")
    assert "PANS RIGHT" in report.headline
    assert "CLOCKWISE orbit signature" in report.headline
    assert "translation LEFT with pan RIGHT is CLOCKWISE" in report.reading_full
    assert "COUNTER-CLOCKWISE orbit signature" not in report.headline


def test_translation_and_turns_have_shared_phase_spans():
    ws = workspace(frames=41, fps=10, camera_t=lambda t: (0, 0, t / 20),
                   camera_R=lambda t: rot_y(max(0, t - 30)))
    report = track_camera(ws)
    assert len(report.rows) == 2
    assert report.rows[0][0] == "0.0–3.0 s"
    assert report.rows[0][3] == "+0.0°"
    assert report.rows[1][0] == "3.0–4.0 s"
    assert report.rows[1][3] == "+10.0°"


def test_full_camera_turn_remains_360_degrees():
    ws = workspace(frames=73, fps=12, camera_R=lambda t: rot_y(5 * t))
    report = track_camera(ws)
    assert report.raw["rotation_deg"][-1][0] == pytest.approx(360)
    assert "+360.0°" in report.headline


def test_roll_is_included_in_jitter_residual():
    ws = workspace(frames=61, fps=30, camera_R=lambda t: rot_z(8 * np.sin(t * 2)))
    report = track_camera(ws)
    assert report.raw["jitter"]["rotation_deg"] > 1
    assert "pan, tilt and roll included" in report.headline


def test_focal_zoom_and_physical_dolly_are_reported_separately():
    zoom = track_camera_zoom(workspace(frames=31, focal=lambda t: 60 * (1 + t / 30)))
    dolly = track_camera_zoom(workspace(frames=31, camera_t=lambda t: (0, 0, t / 30)))
    assert zoom.headline.startswith("ZOOM IN")
    assert zoom.raw["focal_ratio_end"] == pytest.approx(2)
    assert zoom.raw["forward"] == pytest.approx(0)
    assert "NO NET FORWARD MOTION" in zoom.headline
    assert dolly.headline.startswith("FOCAL LENGTH UNCHANGED")
    assert dolly.raw["focal_ratio_end"] == pytest.approx(1)
    assert dolly.raw["forward"] == pytest.approx(1)
    assert "Camera FORWARD" in dolly.headline


def test_zoom_out_then_in_preserves_both_phases():
    report = track_camera_zoom(workspace(frames=41, focal=lambda t: 60 * (1 - min(t, 40 - t) / 40)))
    assert [p["label"] for p in report.raw["phases"]] == ["zoom out", "zoom in"]
    assert report.raw["focal_ratio_end"] == pytest.approx(1)
    assert "ZOOM OUT" in report.headline and "ZOOM IN" in report.headline


def test_scene_revisit_reports_geometric_overlap():
    same = check_scene_revisit(workspace(frames=21))
    other = check_scene_revisit(workspace(frames=21, camera_R=lambda t: rot_y(t * 9)))
    assert same.raw["shared_fraction"] > .8
    assert same.headline.startswith("MOST")
    assert other.raw["shared_fraction"] == pytest.approx(0)
    assert other.headline.startswith("NONE")
    assert "15%" in same.render()


def test_failed_poses_do_not_become_static_camera_evidence():
    ws = workspace(frames=21)
    ws._scene.pose_valid = np.ones(21, dtype=bool)
    ws._scene.pose_valid[10] = False
    report = track_camera(ws)
    assert report.status == "unavailable"
    assert "held poses are not measured stillness" in report.reason
    assert not report.rows
    zoom = track_camera_zoom(ws)
    assert zoom.status == "partial"
    assert zoom.raw["forward"] is None


def test_unknown_frame_rate_uses_shown_labels_and_no_fabricated_speed():
    report = track_camera(workspace(frames=21, fps=None, camera_t=lambda t: (t / 20, 0, 0)))
    assert report.rows[0][0] == "shown frame A–shown frame B"
    assert "Tempo unavailable: frame rate is unknown" in report.headline
    assert "units/s" not in report.render()
    assert "generator" not in report.render()


def test_host_interval_limits_measurement_and_keeps_seconds():
    ws = workspace(frames=41, fps=10, camera_t=lambda t: (0, 0, max(0, t - 20) / 10))
    first = track_camera(ws, window=[0, 1])
    second = track_camera(ws, window=[2, 4])
    assert first.raw["majority"] == "still"
    assert second.raw["majority"] == "moves forward"
    assert second.rows[0][0] == "2.0–4.0 s"
    assert second.raw["path_distance"] == pytest.approx(2)


def test_metric_conversion_requires_real_source_and_names_it():
    ws = workspace(frames=21, camera_t=lambda t: (0, 0, t / 20), metres_per_unit=2)
    relative = track_camera(ws)
    assert relative.raw["units"] == "scene-relative units"
    ws._scene = replace(ws.scene(), metric_source="Point4D × DA3")
    metric = track_camera(ws)
    assert metric.raw["path_distance"] == pytest.approx(2)
    assert metric.raw["units"] == "metres"
    assert "Point4D × DA3" in metric.render()


def test_rotation_pole_only_marks_unmeasurable_rotation_partial():
    report = track_camera(workspace(frames=21, camera_R=lambda t: rot_x(90),
                                    camera_t=lambda t: (t / 20, 0, 0)))
    assert report.status == "partial"
    assert "roll" in report.reason
    assert "MOVES RIGHT" in report.headline


def test_missing_depth_is_unavailable_for_revisit():
    ws = workspace(frames=11)
    ws.scene().points[:] = np.nan
    report = check_scene_revisit(ws)
    assert report.status == "unavailable"
    assert "no usable depth" in report.reason


def test_invalid_numeric_geometry_returns_an_unavailable_report():
    ws = workspace(frames=11)
    ws.scene().scale = float("nan")
    assert track_camera(ws).status == "unavailable"
    assert track_camera_zoom(ws).status == "unavailable"
    ws.scene().cam_R[-1] = np.nan
    assert check_scene_revisit(ws).status == "unavailable"


def test_zoom_report_keeps_out_and_back_dolly_phases():
    report = track_camera_zoom(workspace(frames=41, fps=10,
                                        camera_t=lambda t: (0, 0, min(t, 40 - t) / 10)))
    assert report.raw["forward"] == pytest.approx(0)
    assert [p["label"] for p in report.raw["forward_phases"]] == ["forward", "backward"]
    assert "Dolly profile: FORWARD" in report.headline
    assert "BACKWARD at" in report.headline


def test_missing_end_depth_never_means_no_revisit():
    ws = workspace(frames=11)
    ws.scene().points[-1] = np.nan
    report = check_scene_revisit(ws)
    assert report.status == 'unavailable'
    assert not report.rows


@pytest.mark.parametrize('endpoint,axis', [(0,0),(0,1),(-1,0),(-1,1)])
def test_revisit_rejects_nonpositive_focal_lengths(endpoint,axis):
    ws = workspace(frames=11)
    ws.scene().K[endpoint,axis,axis] = 0
    assert check_scene_revisit(ws).status == 'unavailable'


@pytest.mark.parametrize('delta,components', [
    ((1.2, -.3, 2), '2.000 forward, 1.200 right, 0.300 up'),
    ((-2, .3, 1.2), '2.000 left, 1.200 forward, 0.300 down'),
])
def test_camera_headline_and_phase_table_include_ranked_net_components(delta, components):
    ws = workspace(frames=21, camera_t=lambda t: np.asarray(delta) * t / 20)
    report = track_camera(ws)
    assert components in report.headline
    assert f'Net translation over the interval: {components}' in report.headline
    assert components in report.rows[0][2]
    np.testing.assert_allclose(report.raw['net_components'], delta)


def test_camera_out_and_back_keeps_phase_components_with_zero_net_translation():
    ws = workspace(frames=41, camera_t=lambda t: (min(t, 40 - t) / 10, 0, 0))
    report = track_camera(ws)
    assert '2.000 right' in report.rows[0][2]
    assert '2.000 left' in report.rows[1][2]
    assert 'Net translation over the interval: 0.000 right, 0.000 down, 0.000 forward' in report.headline
