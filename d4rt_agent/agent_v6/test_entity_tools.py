"""Analytic trajectories and mocked facing observations; no model is loaded."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from d4rt_agent.agent_v5.clips import Clip
from d4rt_agent.agent_v5.scene import EntityTrack, Grounding
from d4rt_agent.agent_v5.synthetic import build_scene, rot_y
from d4rt_agent.agent_v6.workspace import Workspace
from d4rt_agent.agent_v6.tools.entity import track_entity
from d4rt_agent.agent_v6.tools.group import track_multiple_entities
from d4rt_agent.agent_v6.tools.orientation import FACING_BINS, facing_rows, track_orientation


def workspace(paths, *, headings=None, category=None, fps=10, camera_R=None):
    frames = len(next(iter(paths.values())))
    scene, _ = build_scene(frames=frames, fps=fps, camera_R=camera_R or (lambda t: np.eye(3)))
    clip = Clip("analytic", {"gf": {"length": frames}, "fps_gf": fps, "sampled16": [0, frames - 1]}, Path("/unused"))
    clip._cache = {i: np.zeros((48, 80, 3), dtype=np.uint8) for i in range(frames)}
    masks = []
    groundings = {}
    for i, name in enumerate(paths):
        mask = np.zeros((1, frames, 48, 80), dtype=bool)
        mask[:, :, 8:24, 3 + i * 20:15 + i * 20] = True
        masks.append(mask)
        groundings[name] = Grounding(name, np.array([i]), mask, np.ones((1, frames)),
                                     np.zeros((1, frames, 4)), mask.mean(axis=(2, 3)), np.zeros((1, frames), bool))
    if category:
        groundings = {category: Grounding(category, np.arange(len(paths)), np.concatenate(masks),
                                          np.ones((len(paths), frames)), np.zeros((len(paths), frames, 4)),
                                          np.concatenate(masks).mean(axis=(2, 3)), np.zeros((len(paths), frames), bool))}
    ws = Workspace(clip, scene=scene, grounder=lambda clip, name, prompt: groundings[name])
    labels = {}
    for name in groundings:
        members = ws.entities(name)
        if category:
            labels.update(zip(paths, [entity.label for entity in members]))
        else:
            labels[name] = members[0].label
    for name, path in paths.items():
        xyz = np.asarray(path, dtype=float)
        valid = np.isfinite(xyz).all(axis=1)
        tr = EntityTrack(labels[name], name, [0], valid, valid, xyz, xyz.copy(),
                         np.ones(frames, int), np.ones(frames), np.zeros(frames, bool),
                         np.zeros((frames, 2)), 0.0)
        entity = ws.resolve(labels[name])
        ws.cached(("track", entity.label), (entity.category,), lambda tr=tr: tr)
    headings = {} if headings is None else headings
    by_label = {labels[name]: values for name, values in headings.items()}

    def facing(name, positions):
        label = ws.resolve(name).label
        values = by_label.get(label)
        result = []
        for pos in positions:
            if values is None or not np.isfinite(values[pos]):
                result.append({"pos": int(pos), "usable": False})
                continue
            angle = np.radians(values[pos])
            forward = np.array([np.sin(angle), 0, np.cos(angle)])
            result.append({"pos": int(pos), "usable": True, "forward_world": forward,
                           "forward_cam": scene.cam_R[pos].T @ forward,
                           "up_world": scene.up, "forward_level": scene.to_level(forward)})
        return result
    ws.facing = facing
    return ws, labels


def constant(point, n=31):
    return np.tile(point, (n, 1)).astype(float)


def test_entities_swap_sides_while_distance_stays_constant():
    theta = np.linspace(0, np.pi, 31)
    main = constant([0, 0, 4])
    other = main + np.column_stack([np.cos(theta), np.zeros(31), np.sin(theta)])
    ws, labels = workspace({"person": main, "dog": other}, headings={"person": np.zeros(31)})
    report = track_entity(ws, "person", ["dog"], "subject")
    relation = report.raw["relations"][0]
    assert relation["sides"][0] == "right" and relation["sides"][-1] == "left"
    np.testing.assert_allclose(relation["distance"], 1)
    assert "right" in report.render() and "left" in report.render()
    assert relation["other"] == labels["dog"]


def test_own_motion_uses_starting_facing_but_other_uses_current_facing():
    main = constant([0, 0, 4])
    main[:, 2] += np.linspace(0, 1, 31)
    other = main + [0, 0, 1]
    ws, labels = workspace({"runner": main, "dog": other}, headings={"runner": np.linspace(0, 90, 31)})
    report = track_entity(ws, "runner", ["dog"], "subject")
    path = np.array(report.raw["motions"][labels["runner"]]["positions"])
    np.testing.assert_allclose(path[-1] - path[0], [0, 0, 1], atol=1e-8)
    relation = report.raw["relations"][0]
    np.testing.assert_allclose(relation["offsets"][0], [0, 0, 1], atol=1e-8)
    np.testing.assert_allclose(relation["offsets"][-1], [-1, 0, 0], atol=1e-8)
    assert [r["other"] for r in report.raw["relations"]] == [labels["dog"]]
    implicit = track_entity(ws, "runner", frame_of_reference="subject")
    assert implicit.raw["relations"][0]["other"] == "camera"


def test_world_motion_survives_unavailable_or_failed_facing():
    path = constant([0, 0, 4])
    path[:, 0] = np.linspace(0, 2, 31)
    ws, labels = workspace({"runner": path})
    report = track_entity(ws, "runner")
    assert report.status == "partial"
    assert report.raw["motions"][labels["runner"]]["majority"] == "right"
    subject = track_entity(ws, "runner", frame_of_reference="subject")
    assert subject.status == "partial" and "No resolvable front" in subject.reason
    assert subject.raw["motions"][labels["runner"]]["majority"] == "right"
    ws.facing = lambda *args: (_ for _ in ()).throw(RuntimeError("model failed"))
    assert track_entity(ws, "runner").raw["motions"][labels["runner"]]["majority"] == "right"


def test_full_turn_accumulates_360_degrees():
    ws, _ = workspace({"person": constant([0, 0, 4], 37)}, headings={"person": np.linspace(0, 360, 37)})
    report = track_orientation(ws, "person", "own")
    assert report.raw["cumulative_turn_deg"] == pytest.approx(360)
    assert report.raw["angular_path_deg"] == pytest.approx(360)
    assert "+360°" in report.render()


def test_camera_rotation_is_removed_from_own_turning():
    ws, _ = workspace({"person": constant([0, 0, 4])}, headings={"person": np.zeros(31)},
                      camera_R=lambda t: rot_y(t * 3))
    report = track_orientation(ws, "person", "own")
    assert report.raw["cumulative_turn_deg"] == pytest.approx(0)
    world = track_entity(ws, "person", frame_of_reference="world")
    camera = track_entity(ws, "person", frame_of_reference="camera")
    assert world.raw["motions"]["person 1"]["majority"] == "still"
    assert camera.raw["motions"]["person 1"]["majority"] != "still"


def test_orientation_to_another_entity_uses_its_current_basis():
    ws, _ = workspace({"person": constant([0, 0, 4]), "dog": constant([1, 0, 4])},
                      headings={"person": np.zeros(31), "dog": np.linspace(0, 90, 31)})
    report = track_orientation(ws, "person", "dog")
    assert report.raw["bearing_deg"][0] == pytest.approx(0)
    assert report.raw["bearing_deg"][-1] == pytest.approx(-90)


def test_group_direction_counts_and_median_pairwise_spread():
    left, right, still = constant([-1, 0, 4]), constant([1, 0, 4]), constant([0, 0, 5])
    left[:, 0] += np.linspace(0, 1, 31)
    right[:, 0] -= np.linspace(0, 1, 31)
    ws, _ = workspace({"left": left, "right": right, "still": still}, category="car")
    report = track_multiple_entities(ws, "car")
    assert report.raw["counts"] == {"right": 1, "left": 1, "still": 1}
    assert report.raw["spread"][0] == pytest.approx(np.sqrt(2))
    assert report.raw["spread"][-1] == pytest.approx(1)


def test_group_own_omits_unresolved_front_from_direction_counts():
    forward = constant([0, 0, 4]) + np.column_stack([np.zeros(31), np.zeros(31), np.linspace(0, 1, 31)])
    ws, labels = workspace({"first": forward, "second": forward + [1, 0, 0]},
                           headings={"first": np.zeros(31)}, category="car")
    report = track_multiple_entities(ws, "car", "own")
    assert report.raw["counts"] == {"forward": 1}
    assert report.raw["unresolved"] == [labels["second"]]


class FakeQwen:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.answer


@pytest.mark.parametrize("label,degrees", list(FACING_BINS.items()))
def test_fallback_maps_all_eight_labels_to_45_degree_bins(label, degrees):
    ws, _ = workspace({"person": constant([0, 0, 4])})
    ws.scene().K[:, 0, 2], ws.scene().K[:, 1, 2] = 8.5, 15.5
    qwen = FakeQwen(label)
    rows = facing_rows(ws, "person", qwen=qwen)
    assert len(rows) == 3 and [r["pos"] for r in rows] == [0, 15, 30]
    expected = [np.sin(np.radians(degrees)), 0, np.cos(np.radians(degrees))]
    for row in rows:
        np.testing.assert_allclose(row["forward_world"], expected, atol=1e-8)
        assert row["source"].endswith("±22.5°")
    assert len(qwen.calls) == 3
    assert all(kwargs == {"max_new_tokens": 32} for _, kwargs in qwen.calls)
    assert all(message[0]["content"][1]["image"].width < 80 for message, _ in qwen.calls)
    facing_rows(ws, "person", qwen=qwen)
    assert len(qwen.calls) == 3


def test_fallback_lifts_camera_heading_into_world_axes():
    ws, _ = workspace({"person": constant([0, 0, 4])}, camera_R=lambda t: rot_y(90))
    ws.scene().K[:, 0, 2], ws.scene().K[:, 1, 2] = 8.5, 15.5
    rows = facing_rows(ws, "person", qwen=FakeQwen("facing image-left"))
    np.testing.assert_allclose(rows[0]["forward_world"], [0, 0, 1], atol=1e-8)


def test_resolved_native_facing_does_not_call_visual_fallback():
    ws, _ = workspace({"person": constant([0, 0, 4])}, headings={"person": np.zeros(31)})
    qwen = FakeQwen("unknown")
    report = track_orientation(ws, "person", "camera", qwen=qwen)
    assert report.status == "ok" and not qwen.calls


def test_missing_interval_is_not_bridged_by_motion_phases():
    path = constant([0, 0, 4])
    path[10:20] = np.nan
    path[20:, 0] = 10
    ws, labels = workspace({"runner": path})
    report = track_entity(ws, "runner")
    segments = report.raw["motions"][labels["runner"]]["segments"]
    assert len(segments) == 2
    assert all(segment["majority"] == "still" for segment in segments)


def test_question_window_keeps_original_seconds():
    path = constant([0, 0, 4])
    path[:, 0] = np.linspace(0, 3, 31)
    ws, labels = workspace({"runner": path})
    report = track_entity(ws, "runner", window=(1.0, 2.0))
    raw = report.raw["motions"][labels["runner"]]
    assert raw["sample_positions"] == list(range(10, 21))
    assert raw["times"][0] == 1.0 and raw["times"][-1] == 2.0
    assert report.raw["window"] == (1.0, 2.0)


def test_unknown_fps_reports_only_displayed_frame_labels():
    ws, _ = workspace({"person": constant([0, 0, 4])}, headings={"person": np.zeros(31)}, fps=None)
    ws.frames()
    text = track_entity(ws, "person").render()
    assert "shown frame" in text
    assert "Frame 30" not in text and "gf_index" not in text
    report = track_entity(ws, "person", window=(1, 2))
    assert report.status == "unavailable" and "frame rate is unknown" in report.reason


def test_metric_distances_use_scale_with_named_provenance():
    ws, _ = workspace({"person": constant([0, 0, 4]), "dog": constant([1, 0, 4])},
                      headings={"person": np.zeros(31)})
    ws._scene = replace(ws.scene(), metres_per_unit=2.5, metric_source="da3-nested")
    report = track_entity(ws, "person", ["dog"], "subject")
    assert report.raw["relations"][0]["distance"][0] == pytest.approx(2.5)
    assert "da3-nested" in report.render() and "metres" in report.render()


def test_subject_window_uses_facing_at_the_window_start():
    path = constant([0, 0, 4])
    path[:, 2] += np.linspace(0, 3, 31)
    ws, labels = workspace({"person": path}, headings={"person": np.linspace(0, 180, 31)})
    report = track_entity(ws, "person", frame_of_reference="subject", window=(1, 2))
    motion = report.raw["motions"][labels["person"]]
    assert motion["starting_position"] == 10
    assert motion["majority"] == "left+forward"
    np.testing.assert_allclose(np.subtract(motion['positions'][-1], motion['positions'][0]),
                               [-np.sqrt(3) / 2, 0, .5], atol=1e-8)
    assert 'facing at 1.00 s' in report.render()


def test_single_facing_observation_cannot_claim_no_turning():
    angles = np.full(31, np.nan)
    angles[15] = 90
    ws, _ = workspace({"person": constant([0, 0, 4])}, headings={"person": angles})
    report = track_orientation(ws, "person", "own")
    assert report.status == "unavailable"
    assert "One facing observation" in report.reason


def test_sparse_facing_does_not_claim_continuous_stillness_or_side():
    angles = np.full(31, np.nan)
    angles[[0, 30]] = 0
    ws, _ = workspace({"person": constant([0, 0, 4])}, headings={"person": angles})
    own = track_orientation(ws, "person", "own")
    assert own.status == "partial"
    assert own.raw["cumulative_turn_deg"] is None
    assert [segment["positions"] for segment in own.raw["segments"]] == [[0], [30]]
    assert "no turning above" not in own.render()
    camera = track_orientation(ws, "person", "camera")
    assert camera.status == "partial"
    assert len(camera.rows) == 2
    assert all(row[0] != "0.00 s–3.00 s" for row in camera.rows)
    assert "between missing observations is unavailable" in camera.headline


def test_turning_is_not_accumulated_across_missing_facing():
    angles = np.r_[np.linspace(0, 180, 15), np.nan, np.linspace(0, 180, 15)]
    ws, _ = workspace({"person": constant([0, 0, 4])}, headings={"person": angles})
    report = track_orientation(ws, "person", "own")
    assert report.raw["cumulative_turn_deg"] is None
    assert [segment["cumulative_turn_deg"] for segment in report.raw["segments"]] == pytest.approx([180, 180])
    assert "360" not in report.render()


def test_facing_camera_fallback_corrects_the_crop_ray():
    ws, _ = workspace({"person": constant([4, 0, 4])})
    ws.scene().K[:] = np.array([[20, 0, 40], [0, 20, 24], [0, 0, 1]])
    masks = ws.ground("person").grounding.masks
    masks[:] = False
    masks[:, :, 20:29, 56:65] = True  # ray (1, 0, 1), matching the subject's location
    qwen = FakeQwen("facing the camera")
    rows = facing_rows(ws, "person", qwen=qwen)
    np.testing.assert_allclose(rows[0]["forward_cam"], [-2**-0.5, 0, -2**-0.5], atol=1e-8)
    report = track_orientation(ws, "person", "camera", qwen=qwen)
    assert report.raw["sides"] == ["front", "front", "front"]
    np.testing.assert_allclose(report.raw["bearing_deg"], 0, atol=1e-8)


def test_partial_facing_retains_all_distance_observations_and_midclip_change():
    main = constant([0, 0, 4])
    other = main.copy()
    other[:, 0] = np.r_[np.linspace(2, 0.5, 16), np.linspace(0.5, 2, 16)[1:]]
    angles = np.full(31, np.nan)
    angles[[0, 30]] = 0
    ws, _ = workspace({"person": main, "dog": other}, headings={"person": angles})
    report = track_entity(ws, "person", ["dog"], "subject")
    relation = report.raw["relations"][0]
    assert relation["positions"] == [0, 30]
    assert relation["distance_positions"] == list(range(31))
    assert min(relation["distance"]) == pytest.approx(0.5)
    directions = {phase["label"] for phase in relation["distance_profile"]["phases"]}
    assert directions == {"closer together", "farther apart"}
    assert "0.50 at 1.50 s" in report.render()


def test_group_spread_reports_convergence_then_spreading_with_timed_extremum():
    first = constant([0, 0, 4])
    second = first.copy()
    second[:, 0] = np.r_[np.linspace(2, 0, 16), np.linspace(0, 2, 16)[1:]]
    ws, _ = workspace({"first": first, "second": second}, category="car")
    report = track_multiple_entities(ws, "car")
    assert {phase["label"] for phase in report.raw["spread_phases"]} == {"converging", "spreading"}
    assert report.raw["spread_extrema"][0]["minimum_position"] == 15
    assert "minimum 0.00 at 1.50 s" in report.render()
    assert "converging" in report.render() and "spreading" in report.render()


def test_m4_short_contracts_are_short_and_do_not_repeat_full_contracts():
    ws, _ = workspace({"person": constant([0, 0, 4]), "dog": constant([1, 0, 4])},
                      headings={"person": np.zeros(31), "dog": np.zeros(31)})
    reports = [track_entity(ws, "person"), track_entity(ws, "person", frame_of_reference="camera"),
               track_entity(ws, "person", ["dog"], "subject"), track_orientation(ws, "person", "own"),
               track_orientation(ws, "person", "camera")]
    group, _ = workspace({"first": constant([0, 0, 4]), "second": constant([1, 0, 4])}, category="car")
    reports.append(track_multiple_entities(group, "car"))
    for report in reports:
        assert report.reading_short != report.reading_full
        assert len(report.reading_short.split()) <= 40
        assert len(report.render(full=False)) < len(report.render(full=True))


def test_composed_and_group_temporal_rows_are_packed_without_losing_phases():
    frames = 61
    path = constant([0, 0, 4], frames)
    path[:, 0] = np.sin(np.linspace(0, 12 * np.pi, frames))
    ws, _ = workspace({"person": path, "dog": path + [2, 0, 0]},
                      headings={"person": np.linspace(0, 720, frames), "dog": np.zeros(frames)})
    report = track_entity(ws, "person", ["dog"])
    assert len(report.rows) <= 6
    assert len(report.raw["motions"]["person 1"]["phases"]) > 6
    assert "left" in report.render() and "right" in report.render()
    group, _ = workspace({"first": constant([0, 0, 4], frames), "second": path + [2, 0, 0]}, category="car")
    grouped = track_multiple_entities(group, "car")
    assert len(grouped.rows) <= 2 + 6
    assert len(grouped.raw["spread_phases"]) > 6
    assert "converging" in grouped.render() and "spreading" in grouped.render()


def test_subject_fallback_names_the_world_axes_in_the_title():
    ws, _ = workspace({"person": constant([0, 0, 4])})
    report = track_entity(ws, "person", frame_of_reference="subject")
    assert report.title.startswith("World motion")
    assert "starting-facing motion unavailable" in report.title


def test_entity_speed_uses_metric_scale_and_actual_seconds_per_phase():
    path = constant([0, 0, 4])
    path[:, 0] = np.linspace(0, 3, 31)
    ws, labels = workspace({"person": path})
    ws._scene = replace(ws.scene(), metres_per_unit=2.5, metric_source="da3-nested")
    report = track_entity(ws, "person")
    motion = report.raw["motions"][labels["person"]]
    assert motion["speed"]["duration_s"] == pytest.approx(3)
    assert motion["speed"]["path_speed"] == pytest.approx(2.5)
    assert motion["speed"]["net_displacement_rate"] == pytest.approx(2.5)
    assert motion["speed"]["units"] == "m/s"
    assert all(phase["path_speed"] == pytest.approx(2.5) for phase in motion["phase_speeds"])
    assert "Path speed 2.50 m/s" in report.render()
    assert "path speed 2.50 m/s" in str(report.rows)


def test_out_and_back_keeps_path_speed_separate_from_net_displacement_rate():
    path = constant([0, 0, 4])
    path[:, 0] = np.r_[np.linspace(0, 1.5, 16), np.linspace(1.5, 0, 16)[1:]]
    ws, labels = workspace({"person": path})
    ws._scene = replace(ws.scene(), metres_per_unit=2.0, metric_source="da3-nested")
    report = track_entity(ws, "person")
    motion = report.raw["motions"][labels["person"]]
    assert motion["speed"]["path_speed"] == pytest.approx(2)
    assert motion["speed"]["net_displacement_rate"] == pytest.approx(0)
    assert {phase["direction"] for phase in motion["phase_speeds"]} == {"right", "left"}
    assert "Path speed 2.00 m/s" in report.render()
    assert "net displacement rate 0.00 m/s" in report.render()


def test_unknown_fps_does_not_report_speed_from_sample_order():
    path = constant([0, 0, 4])
    path[:, 0] = np.linspace(0, 3, 31)
    ws, labels = workspace({"person": path}, fps=None)
    ws.frames()
    report = track_entity(ws, "person")
    motion = report.raw["motions"][labels["person"]]
    assert motion["speed"] is None and motion["phase_speeds"] == []
    assert "Speed unavailable" in report.render()
    assert "m/s" not in report.render() and "units/s" not in report.render()


def test_world_motion_with_camera_as_other_never_requests_a_camera_crop():
    path = constant([0, 0, 4])
    path[:, 0] = np.linspace(0, 3, 31)
    ws, labels = workspace({'person': path}, headings={'person': np.zeros(31)})
    native = ws.facing
    requests = []

    def facing(name, positions):
        requests.append(name)
        return native(name, positions)

    ws.facing = facing
    qwen = FakeQwen('facing away')
    report = track_entity(ws, 'person', ['camera'], 'world', qwen=qwen)
    assert report.status == 'ok'
    assert set(report.raw['motions']) == {labels['person'], 'camera'}
    assert requests == [labels['person']] and not qwen.calls
    assert facing_rows(ws, 'camera', qwen=qwen) == []


@pytest.mark.parametrize('use_fallback', [False, True])
def test_subject_uses_first_resolvable_facing_when_initial_observation_fails(use_fallback):
    path = constant([0, 0, 4])
    path[:, 0] = np.linspace(0, 3, 31)
    headings = np.full(31, 90.0)
    headings[:3] = np.nan
    ws, labels = workspace({'person': path}, headings={'person': headings})
    qwen = FakeQwen('unknown') if use_fallback else None
    report = track_entity(ws, 'person', frame_of_reference='subject', qwen=qwen)
    motion = report.raw['motions'][labels['person']]
    assert motion['frame_of_reference'] == 'subject'
    assert motion['starting_position'] == 3
    assert motion['majority'] == 'forward'
    assert 'facing at 0.30 s' in report.render()
    if qwen:
        assert len(qwen.calls) == 1


def test_missing_window_start_gets_fallback_even_when_later_native_facing_resolves():
    path = constant([0, 0, 4])
    path[:, 0] = np.linspace(0, 3, 31)
    headings = np.full(31, 90.0)
    headings[0], headings[10] = 0, np.nan
    ws, labels = workspace({'person': path}, headings={'person': headings})
    ws.scene().K[:, 0, 2], ws.scene().K[:, 1, 2] = 8.5, 15.5
    qwen = FakeQwen('facing image-right')
    report = track_entity(ws, 'person', frame_of_reference='subject', qwen=qwen, window=(1, 2))
    motion = report.raw['motions'][labels['person']]
    assert motion['frame_of_reference'] == 'subject' and motion['starting_position'] == 10
    assert motion['majority'] == 'forward'
    assert len(qwen.calls) == 1
    assert report.raw['headings'][labels['person']]['positions'] == list(range(10, 21))
    assert report.raw['headings'][labels['person']]['cumulative_turn_deg'] is None
    assert [segment['positions'] for segment in report.raw['headings'][labels['person']]['segments']] == [[10], list(range(11, 21))]
    assert 'facing at 1.00 s' in report.render()


def test_speed_and_phase_distances_use_the_same_filtered_trajectory():
    rng = np.random.default_rng(6)
    path = constant([0, 0, 4], 101) + rng.uniform(-.01, .01, (101, 3))
    path[50:, 0] += np.linspace(0, 2, 51)
    path[[0, 50, 100]] = [[0, 0, 4], [0, 0, 4], [2, 0, 4]]
    ws, labels = workspace({'person': path}, fps=50)
    ws.track_record('person').noise = .02
    report = track_entity(ws, 'person')
    motion = report.raw['motions'][labels['person']]
    assert [phase['label'] for phase in motion['phases']] == ['still', 'right']
    for phase, speed in zip(motion['phases'], motion['phase_speeds']):
        assert speed['path_length'] == pytest.approx(phase['distance'])
    assert motion['phase_speeds'][0]['path_speed'] == pytest.approx(0)
    assert motion['phase_speeds'][1]['path_speed'] == pytest.approx(2)
    assert motion['speed']['path_length'] == pytest.approx(2)
    assert 'noise-filtered trajectory' in report.render()


def test_absorbed_weak_jitter_does_not_become_travel_or_speed_in_a_still_phase():
    path = constant([0, 0, 4], 101)
    path[1::2, 0] = .06
    ws, labels = workspace({'person': path}, fps=50)
    ws.track_record('person').noise = .05 / 3
    report = track_entity(ws, 'person')
    motion = report.raw['motions'][labels['person']]
    assert motion['majority'] == 'still' and len(motion['phases']) == 1
    assert motion['phases'][0]['distance'] == pytest.approx(0)
    assert motion['speed']['path_speed'] == pytest.approx(0)
    assert motion['phase_speeds'][0]['path_length'] == pytest.approx(0)


@pytest.mark.parametrize('window,anchor', [(None, 1), ((1, 2), 11)])
def test_group_own_frame_uses_first_resolvable_facing_in_requested_interval(window, anchor):
    path = constant([0, 0, 4])
    path[:, 0] = np.linspace(0, 3, 31)
    headings = np.full(31, 90.0)
    headings[0] = 0 if window else np.nan
    headings[10] = np.nan
    ws, labels = workspace({'person': path}, headings={'person': headings}, category='car')
    report = track_multiple_entities(ws, 'car', 'own', window=window)
    assert report.raw['counts'] == {'forward': 1}
    assert report.raw['reference_positions'] == {labels['person']: anchor}
    assert f'axes from facing at {anchor / 10:.2f} s' in report.render()
