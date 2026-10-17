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
