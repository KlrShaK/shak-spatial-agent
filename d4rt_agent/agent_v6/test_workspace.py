"""Small synthetic fixtures exercise the same adapters as the runtime workspace."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from d4rt_agent.agent_v5.clips import Clip
from d4rt_agent.agent_v5.scene import Grounding
from d4rt_agent.agent_v5.synthetic import build_scene
from d4rt_agent.agent_v6.workspace import (
    CandidateRequired, EntityUnavailable, TimeUnavailable, Workspace, WorkspaceError,
)


def fixture_workspace(*, fps=6.0, instances=2, orientation_provider=None):
    specs = [("left", lambda t: (-1.0, 0.0, 4.0))]
    if instances == 2:
        specs.append(("right", lambda t: (1.0 + t / 100, 0.0, 4.0)))
    scene, groundings = build_scene(frames=31, fps=fps, entities=specs)
    names = list(groundings)
    g = Grounding("car", np.arange(instances), **{
        key: np.concatenate([getattr(groundings[name], key) for name in names])
        for key in ("masks", "probs", "boxes", "area_frac", "border")
    })
    clip = Clip("synthetic", {"dataset": "synthetic", "gf": {"length": 31}, "fps_gf": fps,
                              "sampled16": [0, 10, 20, 30]}, Path("/unused"))
    clip._cache = {i: np.full((3, 4, 3), i, dtype=np.uint8) for i in range(31)}
    calls = []
    responses = [g]

    def grounder(clip_arg, phrase, prompt_frame):
        calls.append((clip_arg, phrase, prompt_frame))
        return responses[-1]

    ws = Workspace(clip, scene=scene, grounder=grounder, orientation_provider=orientation_provider)
    return ws, g, calls, responses


def test_names_registered_without_grounding_and_camera_is_implicit():
    ws, _, calls, _ = fixture_workspace()
    assert ws.register(" The runner ") == "runner"
    assert "runner" in ws.known_names
    assert not calls
    assert ws.resolve("the viewer").is_camera
    assert ws.resolve("CAM").label == "camera"
    np.testing.assert_allclose(ws.track("camera"), ws.scene().cam_t)
    assert not calls


def test_names_labels_and_candidate_selection_share_grounding():
    ws, _, calls, _ = fixture_workspace()
    assert [e.label for e in ws.entities("the CAR")] == ["car 1", "car 2"]
    assert ws.resolve("the car 2").instances == (1,)
    with pytest.raises(CandidateRequired) as exc:
        ws.ensure_selection("car")
    assert [c.colour for c in exc.value.selection.choices] == ["red", "blue"]
    assert ws.choose("BLUE").label == "car 2"
    assert ws.resolve("car").label == "car 2"
    assert len(calls) == 1
    with pytest.raises(WorkspaceError, match="no current"):
        ws.choose("blue")


def test_candidate_choice_expires_on_another_call_and_question_reset():
    ws, _, calls, _ = fixture_workspace()
    with pytest.raises(CandidateRequired):
        ws.resolve("car")
    ws.begin_tool_call("get_frames")
    with pytest.raises(WorkspaceError):
        ws.choose("red")
    with pytest.raises(CandidateRequired):
        ws.resolve("car")
    ws.begin_tool_call("choose_segmentation_candidate")
    ws.choose("red")
    ws.begin_question()
    with pytest.raises(CandidateRequired):
        ws.resolve("car")
    assert len(calls) == 1


def test_reordering_sam_identifiers_preserves_mask_identity():
    ws, g, _, responses = fixture_workspace()
    ws.ground("car")
    responses.append(replace(g, obj_ids=np.array([17, 9]), **{
        key: getattr(g, key)[::-1].copy() for key in ("masks", "probs", "boxes", "area_frac", "border")
    }))
    ws.reground("car")
    assert ws.resolve("car 1").instances == (1,)
    assert ws.resolve("car 2").instances == (0,)
    assert ws.resolve("car 1").label != ws.resolve("car 2").label


def test_replaced_masks_do_not_inherit_retired_labels_by_sam_id():
    ws, g, _, responses = fixture_workspace(instances=1)
    ws.ground("car")
    new = g.masks.copy()
    new[:] = False
    new[:, :, :5, :5] = True
    responses.append(replace(g, masks=new))
    assert ws.reground("car").entities[0].label == "car 2"
    with pytest.raises(EntityUnavailable, match="label car 1"):
        ws.resolve("car 1")


def test_reground_invalidates_track_orientation_and_pair_dependants():
    orientation_calls = []

    def facing(clip, scene, masks, positions):
        orientation_calls.append(tuple(positions))
        return [{"pos": p, "usable": True, "mask_area": int(masks(p).sum())} for p in positions]

    ws, _, _, _ = fixture_workspace(instances=1, orientation_provider=facing)
    track = ws.track_record("car")
    ws.facing("car", [0, 10])
    ws.facing("car 1", [0, 10])
    pair = ws.cached("pair", ["car 1", "runner"], object)
    unrelated = ws.cached("camera_derived", ["camera"], object)
    revision = ws.revisions["car"]
    ws.reground("car")
    assert ws.revisions["car"] == revision + 1
    assert ws.track_record("car") is not track
    assert ws.cached("pair", ["car 1", "runner"], object) is not pair
    assert ws.cached("camera_derived", ["camera"], object) is unrelated
    ws.facing("car", [0, 10])
    assert orientation_calls == [(0, 10), (0, 10)]


def test_selection_change_invalidates_category_dependants():
    ws, _, _, _ = fixture_workspace()
    with pytest.raises(CandidateRequired):
        ws.resolve("car")
    derived = ws.cached("category_motion", ["car"], object)
    ws.choose("blue")
    assert ws.cached("category_motion", ["car"], object) is not derived
    assert ws.ground("car").revision == ws.revisions["car"]


def test_motion_does_not_require_resolvable_facing():
    ws, _, _, _ = fixture_workspace(instances=1)
    facing = ws.facing("car", [0, 15, 30])
    assert not any(row["usable"] for row in facing)
    np.testing.assert_allclose(ws.track("car"), np.tile([-1.0, 0, 4.0], (31, 1)), atol=1e-5)
    assert ws.track_record("car").id == "car 1"


def test_failed_grounding_prompts_for_name_then_visual_check():
    ws, g, _, responses = fixture_workspace()
    responses.append(replace(g, **{
        key: getattr(g, key)[:0] for key in ("obj_ids", "masks", "probs", "boxes", "area_frac", "border")
    }))
    with pytest.raises(EntityUnavailable, match="Suggest an alternative name"):
        ws.resolve("skier")
    with pytest.raises(EntityUnavailable, match="not proof that it is absent"):
        ws.resolve("person on skis")
    assert ws.grounding_misses == 2
    ws.begin_question()
    assert ws.grounding_misses == 0


def test_seconds_nearest_frames_ranges_and_scene_sampling():
    ws, _, calls, _ = fixture_workspace()
    assert ws.frame_index(1.2) == 7
    assert ws.frame_at("1.2 s").label == "1.17 s"
    assert ws.frame_index(-20) == 0
    assert ws.frame_index(100) == 30
    assert [view.index for view in ws.frames(range=[0, 5])] == [0, 10, 20, 30]
    assert [view.label for view in ws.frames(times=[0, 5])] == ["0.00 s", "5.00 s"]
    ws.ground("car", at=1.2)
    assert calls[0][2] == 7
    ws._scene = replace(ws.scene(), gf_index=np.array([0, 6, 30]))
    np.testing.assert_array_equal(ws.times(), [0, 1, 5])
    assert ws.time_labels() == ["0.00 s", "1.00 s", "5.00 s"]
    with pytest.raises(TimeUnavailable):
        ws.frames(times=[0], range=[0, 5])
    with pytest.raises(TimeUnavailable):
        ws.frames(range=[5, 0])
    with pytest.raises(TimeUnavailable):
        ws.frame_index(float("nan"))


def test_unknown_fps_uses_shown_labels_without_generator_integers():
    ws, _, calls, _ = fixture_workspace(fps=None)
    views = ws.frames()
    assert [v.label for v in views] == [f"shown frame {c}" for c in "ABCD"]
    assert ws.times() is None
    assert ws.frame_at("shown frame C").index == 20
    assert ws.frame_at("shown frame C").label == "shown frame C"
    assert ws.time_labels([0, 10, 20, 30]) == [v.label for v in views]
    assert "20" not in repr(views[2])
    with pytest.raises(TimeUnavailable, match="frame rate is unknown"):
        ws.frame_at(20)
    with pytest.raises(TimeUnavailable, match="already shown"):
        ws.frame_at("shown frame Z")
    assert [v.index for v in ws.frames(range=["shown frame A", "shown frame D"])] == [0, 10, 20, 30]
    ws.ground("car", at="shown frame C")
    assert calls[0][2] == 20


def test_metric_scale_requires_estimator_provenance():
    ws, _, _, _ = fixture_workspace()
    ws._scene = replace(ws.scene(), metres_per_unit=2.5, metric_source="da3-nested", metric_iqr=0.1)
    assert ws.metric.units == "metres"
    assert ws.metric.source == "da3-nested"
    assert ws.metric.relative_spread == 0.1
    np.testing.assert_allclose(ws.metric.convert([1, 2]), [2.5, 5])
    for source in (None, "d4rt-native", "point4d-raw", "d4rt-raw"):
        ws._scene = replace(ws.scene(), metric_source=source)
        assert not ws.metric.available
        assert ws.metric.units == "scene-relative units"
    ws._scene = replace(ws.scene(), metric_source="da3-nested", metres_per_unit=float("nan"))
    assert not ws.metric.available


def test_runtime_adapter_loads_scene_and_models_only_when_requested(monkeypatch, tmp_path):
    fixture, g, _, _ = fixture_workspace(instances=1)
    ws = Workspace.from_runtime(fixture.clip, tmp_path, backbone="point4d")
    assert ws.models.backbone_name == "point4d"
    assert ws.models._sam3 is None and ws.models._scenes is None and ws.models._orient is None
    scene_calls = []
    ground_calls = []

    def scene(clip):
        scene_calls.append(clip)
        return fixture.scene()

    monkeypatch.setattr(ws.models, "scenes", lambda: SimpleNamespace(scene=scene))
    monkeypatch.setattr(ws.models, "ground", lambda clip, name, prompt: ground_calls.append(name) or g)
    ws.track("car")
    ws.track("car")
    assert len(scene_calls) == 1
    assert ground_calls == ["car"]
    assert ws.models._orient is None
    alternative = Workspace.from_runtime(fixture.clip, tmp_path, backbone="d4rt")
    assert alternative.models.backbone_name == "d4rt"


def test_unknown_time_reports_never_register_unseen_images():
    ws, _, _, _ = fixture_workspace(fps=None)
    with pytest.raises(TimeUnavailable, match="Show frames"):
        ws.time_labels([0, 15, 30])
    with pytest.raises(TimeUnavailable, match="Show this frame"):
        ws.frame_label(15)
    assert not ws._shown
    assert ws.mark_shown(0) == "shown frame A"
    assert ws.mark_shown(30) == "shown frame B"
    assert ws.time_labels([0, 10, 20, 30]) == ["shown frame A", "shown frame A", "shown frame B", "shown frame B"]
    assert len(ws._shown) == 2
    with pytest.raises(TimeUnavailable, match="already shown"):
        ws.frame_index("shown frame C")
