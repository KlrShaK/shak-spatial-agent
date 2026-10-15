"""GPU-free regressions for identity, evidence lifetime and answer recovery."""

from dataclasses import replace
from contextlib import contextmanager
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from d4rt_agent.agent_v5 import kinds, measure, synthetic
from d4rt_agent.agent_v5.capabilities import (
    census, count_entities, event_reduce, frame_vectors, image_series, inspect,
    locate_event, merge_identity, record_event,
)
from d4rt_agent.agent_v5.context import Conversation
from d4rt_agent.agent_v5.contracts import ActionRejected, parse_answer_response, validate_action
from d4rt_agent.agent_v5.evidence import _indices
from d4rt_agent.agent_v5.final_context import render_frame
from d4rt_agent.agent_v5.models import Models
from d4rt_agent.agent_v5.orchestrator import AgentV5, State
from d4rt_agent.agent_v5.program import Entity, Event, Measurement, QueryProgram
from d4rt_agent.agent_v5.research.counterexamples import track, workspace
from d4rt_agent.agent_v5.scene import Grounding, merge_fragments, resolve_entities
from d4rt_agent.agent_v5.test_agent_v5 import FakeClip, ScriptedQwen, _act
from d4rt_agent.agent_v5.test_composition import census_workspace
from d4rt_agent.agent_v5.test_workflow import camera_actions, make_agent, packet_state, solve, submission
from d4rt_agent.agent_v5.tools import TOOLS, ToolError, Workspace


def two_people(kind="category"):
    scene, groundings = synthetic.build_scene(40, 6, metres_per_unit=1.0, entities=[
        ("still", lambda t: (-1.2, 0.0, 5.0)),
        ("moving", lambda t: (1.2, 0.5, 5.0 - 2.1 * t / 39)),
    ])
    parts = list(groundings.values())
    g = Grounding("person", np.array([10, 20]), **{
        name: np.concatenate([getattr(p, name) for p in parts])
        for name in ("masks", "probs", "boxes", "area_frac", "border")})
    g.area_frac[1] = np.linspace(0.02, 0.12, 40)
    eid = "g1" if kind == "group" else "c1"
    program = QueryProgram("q", "range", "measure", "", entities=[
        Entity("cam", "camera"), Entity(eid, kind, "person", count=2 if kind == "group" else None)])
    return Workspace(FakeClip(), program, scene_fn=lambda c: scene, ground_fn=lambda *a: g)


@pytest.mark.parametrize("name,args", [
    ("distance", {"a": "c1", "b": "cam"}),
    ("speed", {"entity": "c1"}),
    ("apparent_size", {"entity": "c1"}),
    ("elevation", {"entity": "c1"}),
    ("motion_profile", {"entity": "c1"}),
])
def test_category_measurements_follow_selected_identity(name, args):
    ws = two_people()
    before = TOOLS[name](ws, args).data["value"]
    TOOLS["select"](ws, {"entity": "c1", "instances": [1]})
    selected = TOOLS[name](ws, args)
    explicit = TOOLS[name](ws, {k: "c1[1]" if v == "c1" else v for k, v in args.items()})
    assert selected.data["available"] and explicit.data["available"]
    assert selected.data["value"] == explicit.data["value"]
    assert selected.data["value"] != before
    if name == "distance":
        assert selected.data["value"]["trend"] == "closer"


@pytest.mark.parametrize("selected", [True, False])
def test_group_binding_agrees_in_geometry_images_census_and_evidence(selected, monkeypatch):
    from PIL import ImageDraw
    ws = two_people("group")
    if selected:
        TOOLS["select"](ws, {"entity": "g1", "instances": [1, 0]})
    else:
        ws.grounding("g1").probs[0] *= 0.1  # Ranked member 0 is raw instance 1.
    tr = ws.track("g1[0]")
    assert tr.instances == [1]
    xy, present, _, _ = image_series(ws, "g1[0]")
    assert np.allclose(xy[present], tr.centroid[present], atol=0.02)
    assert _indices(ws, "g1[0]", "inspect") == [1]
    rows = census(ws, {"entity": "g1"}).data["value"]["instances"]
    assert rows[0]["id"] == "g1[0]" and rows[0]["indices"] == [1]
    labels = []
    original = ImageDraw.ImageDraw.text
    def capture(self, xy, text, *args, **kwargs):
        labels.append(text)
        return original(self, xy, text, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    render_frame(ws, "g1[0]", 10)
    assert labels == ["g1[0] (SAM 1)"]


def test_group_selection_cannot_choose_two_known_fragments_of_one_identity():
    ws = two_people("group")
    iid = inspect(ws, {}).data["value"]["inspection_id"]
    merge_identity(ws, {"entity": "g1", "instances": [0, 1], "evidence_ids": [iid]})
    with pytest.raises(ToolError, match="distinct physical identities"):
        TOOLS["select"](ws, {"entity": "g1", "instances": [0, 1]})


def test_selected_category_alias_keeps_all_fragment_masks_and_provenance():
    ws = census_workspace()
    iid = inspect(ws, {"entity": "c1"}).data["value"]["inspection_id"]
    merge_identity(ws, {"entity": "c1", "instances": [0, 2], "evidence_ids": [iid]})
    TOOLS["select"](ws, {"entity": "c1", "instances": [2]})
    selected, present, _, _ = image_series(ws, "c1")
    explicit, same_present, _, _ = image_series(ws, "c1[0]")
    assert np.array_equal(present, same_present)
    assert np.allclose(selected[present], explicit[present])
    assert _indices(ws, "c1", "view_state") == [0, 2]


def test_omitted_count_predicate_requests_binding_then_accepts_explicit_observation():
    ws = census_workspace()
    state = packet_state(ws)
    state.active_requirement.predicate = "walking to the left"
    agent, _ = make_agent([
        _act("count_entities", entity="c1"),
        _act("count_entities", entity="c1", predicates=["left:image", "walking"]),
        _act("done_measuring"),
    ])
    ws.grounding("c1")
    for predicate in ("left:image", "walking"):
        ws.observations[predicate] = {"entity": "c1[0]", "predicate": predicate, "value": "true",
            "revision": ws.revisions["c1"], "window": [0, 10]}
    conv, trace = Conversation("test", 16000), []
    agent._measure({"clip": ws.clip}, conv, trace, state)
    assert "Bind the full qualifier" in trace[0]["tool_error"]
    assert len(state.results) == 1
    assert state.results[0][2].data["value"]["count"] == 1


@pytest.mark.parametrize("localized", [False, True])
def test_occurrences_survive_unrelated_binding_changes(localized):
    ws = census_workspace()
    ws.program.entities.append(Entity("e2", "object", "person"))
    ws.grounding("e2")
    if localized:
        locate_event(ws, {"actor": "c1[2]", "predicate": "enters_frame"})
    else:
        iid = inspect(ws, {"entity": "c1"}).data["value"]["inspection_id"]
        record_event(ws, {"predicate": "passes", "participants": ["cam", "c1[2]"],
                          "start_s": 4, "end_s": 4.5, "evidence_ids": [iid]})
    assert event_reduce(ws, {}).data["value"]["result"] == 1
    ws.invalidate("e2")
    assert event_reduce(ws, {}).data["value"]["result"] == 1
    ws.invalidate("c1")
    assert event_reduce(ws, {}).data["value"]["result"] == 0


def onset(scene, predicate, actor="cam", tr=None, window=None):
    ev = Event("ev1", actor, predicate, predicate)
    ws = workspace(scene, {actor: [tr]} if tr is not None else {}, events=[ev], window=window)
    m = Measurement("m", "onset", ["ev1"], "onset")
    return ws, measure.measure_event_onset(scene, ws.program, m, ws.tracks_map(actor))


@pytest.mark.parametrize("fps", [6, 30])
def test_slow_pan_uses_total_swing_not_rate_or_sampling_density(fps):
    scene, _ = synthetic.build_scene(10 * fps + 1, fps,
        camera_R=lambda t: synthetic.rot_y(max(0, t / fps - 2)))
    _, r = onset(scene, "pan_right")
    assert r.value["onset_s"] == pytest.approx(2, abs=0.4)
    assert not r.value["left_censored"]
    assert r.floor["total_swing_deg"] == 4


def test_motion_noise_does_not_produce_a_start_and_ongoing_motion_is_censored():
    scene, _ = synthetic.build_scene(40, 6)
    xyz = np.column_stack([np.arange(40) % 2 * 0.03, np.zeros(40), np.full(40, 3)])
    noisy = replace(track(xyz), noise=0.02)
    _, r = onset(scene, "starts_moving", "e1", noisy)
    assert r.value["onset"] is None and not r.value["left_censored"]
    moving = track(np.column_stack([scene.t_s, np.zeros(40), np.full(40, 3)]))
    ws, r = onset(scene, "starts_moving", "e1", moving)
    assert r.value["onset"] is None and r.value["left_censored"]
    assert r.value["present_from_first_frame"]
    assert "already underway" in TOOLS["event_onset"](ws, {"event": "ev1"}).text
    order = measure.measure_order(scene, Measurement("order", "order", ["a", "b"], "order"), {"a": r, "b": r})
    assert not order.available


@pytest.mark.parametrize("peak", [0, 1])
def test_camera_labels_honor_net_peak_settings_and_keep_both_readings(peak, monkeypatch):
    monkeypatch.setitem(measure.TUNING, "cam_trans_peak", peak)
    monkeypatch.setitem(measure.TUNING, "cam_rot_peak", peak)
    scene, _ = synthetic.build_scene(61, 10,
        camera_t=lambda t: (2 * np.sin(np.pi * t / 60), 0, 0),
        camera_R=lambda t: synthetic.rot_y(15 * np.sin(np.pi * t / 60)))
    r = measure.measure_camera_motion(scene, None, Measurement("m", "camera_motion", [], "classify"))
    assert r.value["motion"] == "static" if not peak else r.value["motion"] != "static"
    assert abs(r.value["peak_translation"][0]) > 1.9
    assert abs(r.value["net_translation"][0]) < 0.02


def test_camera_orbit_proxy_is_available_and_explicit_about_its_limits():
    scene, _ = synthetic.build_scene(40, 6, camera_t=lambda t: (-t / 39, 0, 0),
                                     camera_R=lambda t: synthetic.rot_y(8 * t / 39))
    r = measure.measure_camera_motion(scene, None, Measurement("m", "camera_motion", [], "classify"))
    assert r.value["motion"] == "orbit" and r.value["orbit"]
    assert "proxy" in r.value["orbit"]["source"]


def test_default_group_merges_fragments_before_ranking_and_drops_brief_noise():
    ws = two_people("group")
    g = ws.grounding("g1")
    # Split the stationary person into two non-overlapping SAM tracks; retain the other person.
    masks = np.concatenate([g.masks, g.masks[:1], g.masks[:1]])
    masks[0, 20:] = False
    masks[2, :20] = False
    masks[3] = False
    masks[3, 10] = g.masks[0, 10]
    extended = Grounding("person", np.arange(4), masks,
        np.ones((4, 40)), np.concatenate([g.boxes, g.boxes[:1], g.boxes[:1]]),
        masks.mean((2, 3)), np.zeros((4, 40), bool))
    ws.groundings["g1"] = extended
    resolved = resolve_entities(ws.scene(), {"person": extended}, ws.program)["g1"]
    assert len(resolved) == 2
    assert sorted(sorted(t.instances) for t in resolved) == [[0, 2], [1]]


@pytest.mark.parametrize("fps", [None, 10])
def test_fragment_merge_uses_elapsed_time_or_generator_frames_not_sample_positions(fps):
    scene, _ = synthetic.build_scene(6, fps)
    scene.gf_index = np.array([0, 1, 2, 50, 51, 52])
    xyz = np.tile([0., 0., 3.], (6, 1))
    a = track(xyz, present=np.array([1, 1, 1, 0, 0, 0], bool))
    b = track(xyz, present=np.array([0, 0, 0, 1, 1, 1], bool))
    assert len(merge_fragments(scene, [a, b])) == 2


def test_category_fragments_need_identity_review_without_dropping_brief_tracks():
    ws = census_workspace()
    g = ws.grounding("c1")
    g.masks[0, 7:] = False
    g.masks[2] = False
    g.masks[2, 4:7] = g.masks[0, 0]
    result = count_entities(ws, {}).data["value"]
    assert result["count"] == 1  # The brief, spatially distinct encounter still counts.
    assert set(result["unresolved"]) == {"c1[0]", "c1[2]"}
    assert count_entities(ws, {"aggregation": "at_time", "at_s": 0}).data["value"]["count"] == 1
    assert count_entities(ws, {"aggregation": "max_simultaneous"}).data["value"]["count"] == 2
    iid = inspect(ws, {"entity": "c1"}).data["value"]["inspection_id"]
    merge_identity(ws, {"entity": "c1", "instances": [0, 2], "evidence_ids": [iid]})
    assert count_entities(ws, {}).data["value"]["count"] == 2


def test_camera_initial_frame_uses_pose_at_requested_anchor_without_orientation_model():
    scene, _ = synthetic.build_scene(40, 6, camera_R=lambda t: synthetic.rot_y(90 * t / 39))
    ws = workspace(scene)
    vectors = np.array([[1., 0., 0.]])
    initial = frame_vectors(ws, vectors, np.array([39]), {"frame": "body_initial", "reference": "cam"})
    anchored = frame_vectors(ws, vectors, np.array([39]),
                             {"frame": "body_initial", "reference": "cam", "anchor_s": 6.5})
    assert np.allclose(initial, vectors)
    assert np.allclose(anchored, [[0, 0, 1]])
    subject = two_people()
    assert TOOLS["entity_track"](subject, {"entity": "c1", "frame": "own", "reference": "cam"}).data["available"]


def test_camera_own_frame_question_completes_without_subject_facing():
    agent, _ = make_agent(camera_actions()[:-1] + [
        _act("set_frame", frame="body_initial", reference="cam", anchor_s=None),
        _act("entity_track", entity="cam"), _act("done_measuring"), submission(),
    ])
    result = agent.solve({"id": "q", "clip": FakeClip(),
        "question": "How does the camera move in its own reference frame?",
        "options": {"A": "still", "B": "forward"}})
    assert result["answered"], result.get("error")
    assert result["tools"][0]["data"]["available"]


def test_ongoing_pan_is_censored_instead_of_missing():
    scene, _ = synthetic.build_scene(61, 10, camera_R=lambda t: synthetic.rot_y(t / 10))
    ws, r = onset(scene, "pan_right")
    assert r.value["left_censored"] and r.value["present_from_first_frame"]
    assert r.value["onset"] is None
    assert "already underway" in TOOLS["event_onset"](ws, {"event": "ev1"}).text


@pytest.mark.parametrize("prefix", ["", "Assessment.\n"])
@pytest.mark.parametrize("language", ["json", ""])
def test_terminal_fenced_answer_is_accepted(prefix, language):
    envelope = {"action": "final_answer", "arguments": {"letter": "A", "justification": "evidence"}}
    text, parsed = parse_answer_response(prefix + f"```{language}\n" + json.dumps(envelope) + "\n```\n")
    assert text == prefix.strip() and parsed == envelope


@pytest.mark.parametrize("body", [
    '{"action":"final_answer","arguments":{"letter":"A","letter":"B"}}',
    '[{"action":"final_answer","arguments":{}}]',
    '{"action":"final_answer","arguments":{}}\n{"action":"final_answer","arguments":{}}',
])
def test_fences_do_not_relax_answer_validation(body):
    with pytest.raises(ActionRejected):
        parse_answer_response("```json\n" + body + "\n```")


def test_failed_post_repair_submission_preserves_accepted_letter():
    repair = {"requirement": "r1", "option": "A", "claim": "path", "missing": "detail",
              "evidence": [], "mode": "measure", "could_change_choice": "could distinguish motion"}
    agent, _ = make_agent(camera_actions() + [_act("done_measuring"), submission(repair_request=repair),
        _act("done_measuring")] + ["{}"] * 4)
    result = solve(agent)
    assert result["letter"] == "A" and result["answered"]
    assert result["repair_history"][0]["submission_failed"]
    assert "pre-repair" in result["limitations"]


def test_optional_nulls_use_defaults_and_invalid_types_are_retryable():
    agent, _ = make_agent(camera_actions()[:-1] + [
        _act("set_frame", frame="scene", anchor_s={}),
        _act("set_frame", frame="scene", anchor_s=None),
        _act("add_requirement", kind="visibility", entities=["cam"], depends_on=None),
        _act("done_measuring"), submission(),
    ])
    result = solve(agent)
    assert result["answered"], result.get("error")
    assert any(r.get("error") for r in result["trace"])
    assert result["program"]["requirements"][1]["depends_on"] == []
    with pytest.raises(ActionRejected):
        validate_action({"action": "set_predicate", "arguments": {"predicate": None, "justification": "x"}}, ["set_predicate"])
    assert count_entities(census_workspace(), {"predicate": None}).data["value"]["count"] == 3


@pytest.mark.parametrize("fps", [None, 6])
def test_legacy_unresolved_event_window_has_labeled_whole_clip_fallback(fps):
    agent = AgentV5(qwen=ScriptedQwen(["{}"] * 3), backbone="fake", workspace_factory=lambda *a: None)
    clip = FakeClip()
    clip.fps = fps
    state = State(kind="camera_path", slots={})
    task = {"id": "q", "clip": clip, "question": "How does the camera move to the frame where the person jumps?"}
    state.program = agent._program(task, clip, Conversation("test", 16000), [], state)
    assert state.program.window.kind == "whole"
    assert "fallback context" in state.notes[-1]


def test_legacy_menus_only_offer_executable_tools_and_reject_compositional_switch():
    for kind in kinds.LEGACY_KINDS:
        names = set(kinds.tools_for(kind.name, legacy=True)) - {"switch_kind", "done_measuring"}
        assert names <= TOOLS.keys()
        assert not names & {"add_entity", "add_requirement"}
    agent = AgentV5(qwen=ScriptedQwen([]), backbone="fake", workspace_factory=lambda *a: None)
    state = State(kind="camera_motion")
    conv = Conversation("test", 16000)
    agent._switch({}, conv, [], state, "count_entities")
    assert state.kind == "camera_motion" and state.kind_switches == 0


@pytest.mark.parametrize("metadata", [None, "invalid json"])
def test_reseed_cache_recovers_incomplete_legacy_entries_and_commits_one_file(tmp_path, metadata):
    ws = two_people()
    scene, g = ws.scene(), ws.grounding("c1")
    g.masks[1] = True  # Ensure the coarse query grid contains enough points for this cache test.
    calls = []
    def query(*args):
        calls.append(True)
        return np.zeros((scene.length, len(args[2]), 3)), [{"alignment": "measured"}]
    models = Models(qwen_model=None, backbone="point4d", cache=tmp_path)
    models._scenes = SimpleNamespace(directory=lambda c: tmp_path, backbone=SimpleNamespace(track_points=query))
    first = models.track_points(ws.clip, scene, g, [1], 0)
    target = next((tmp_path / "reseed_v2").glob("*.npz"))
    with np.load(target) as z:
        assert "diagnostics" in z
    # Simulate the old writer being killed after publishing only its array file.
    np.savez_compressed(target, tracks=first["tracks"])
    if metadata is not None:
        target.with_suffix(".json").write_text(metadata)
    recovered = models.track_points(ws.clip, scene, g, [1], 0)
    cached = models.track_points(ws.clip, scene, g, [1], 0)
    assert len(calls) == 2
    assert recovered["diagnostics"] == cached["diagnostics"] == first["diagnostics"]


def test_point4d_reseeding_uses_inference_mode_in_both_directions(monkeypatch):
    from d4rt_agent.agent_v5.backbones.point4d import Point4DBackbone
    from d4rt_agent.agent_v5.backbones import reseed
    active, calls = [], []
    @contextmanager
    def inference_mode():
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    def tracker(model, frames, uv, config, **kwargs):
        assert active
        calls.append(len(frames))
        return np.zeros((len(frames), len(uv), 3)), None, {"points": np.zeros((len(frames), 4, 3))}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=inference_mode))
    monkeypatch.setitem(sys.modules, "point4d", ModuleType("point4d"))
    monkeypatch.setitem(sys.modules, "point4d.inference", SimpleNamespace(
        LongTrackConfig=lambda **kwargs: kwargs, track_long_video=tracker))
    monkeypatch.setattr(reseed, "align_to_scene", lambda tracks, *a: (tracks, {}))
    scene, _ = synthetic.build_scene(5, 6)
    clip = SimpleNamespace(frames=lambda indices: np.zeros((len(indices), 8, 8, 3)))
    backbone = Point4DBackbone()
    backbone._model = object()
    result, diagnostics = backbone.track_points(clip, scene, np.zeros((3, 2)), 2)
    assert calls == [3, 3] and len(diagnostics) == 2
    assert result.shape == (5, 3, 3) and not active
