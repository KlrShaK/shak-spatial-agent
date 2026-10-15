"""GPU-free behavioral regressions for the compositional stack and measurement contracts."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from d4rt_agent.agent_v5 import measure, synthetic, kinds
from d4rt_agent.agent_v5.capabilities import (
    census, count_entities, event_reduce, inspect, observe, record_event, relative_position, runs,
)
from d4rt_agent.agent_v5.composition import CompositionalAgentV5
from d4rt_agent.agent_v5.program import Entity, Event, EvidenceRecord, Measurement, QueryProgram, Window
from d4rt_agent.agent_v5.research.counterexamples import track, workspace
from d4rt_agent.agent_v5.scene import Grounding, resolve_window
from d4rt_agent.agent_v5.spatial_tools import similarity_angle
from d4rt_agent.agent_v5.test_agent_v5 import FakeClip, ScriptedQwen, _act, _texts
from d4rt_agent.agent_v5.tools import TOOLS, Workspace, ToolError


def census_workspace():
    clip = FakeClip(11, 2)
    masks = np.zeros((3, 11, 8, 8), bool)
    masks[0, :3, 1:3, 1:3] = True
    masks[0, 7:, 1:3, 1:3] = True       # same identity reappears
    masks[1, 1, 4:6, 4:6] = True        # brief encounter is retained
    masks[2, 8:, 5:7, 1:3] = True       # outside the early query window
    g = Grounding("tree", np.array([10, 20, 30]), masks, np.ones((3, 11)),
                  np.zeros((3, 11, 4)), masks.mean((2, 3)), np.zeros((3, 11), bool))
    def no_geometry(clip):
        raise AssertionError("image census must not reconstruct geometry")
    program = QueryProgram("q", "count_entities", "measure", "", entities=[Entity("c1", "category", "tree")])
    return Workspace(clip, program, scene_fn=no_geometry, ground_fn=lambda c, p, f: g)


def test_census_keeps_brief_tracks_all_intervals_and_window_without_geometry():
    ws = census_workspace()
    result = census(ws, {"entity": "c1", "window": [0, 1]})
    rows = result.data["value"]["instances"]
    assert len(rows) == 2
    assert rows[0]["intervals_gf"] == [[0, 2], [7, 10]]
    assert rows[1]["brief"]
    assert count_entities(ws, {"entity": "c1", "window": [0, 1]}).data["value"]["count"] == 2
    assert count_entities(ws, {"entity": "c1"}).data["value"]["count"] == 3


def test_count_units_differ_unique_at_time_and_simultaneous():
    ws = census_workspace()
    assert count_entities(ws, {"aggregation": "at_time", "at_s": 0}).data["value"]["count"] == 1
    assert count_entities(ws, {"aggregation": "max_simultaneous"}).data["value"]["count"] == 2
    with pytest.raises(ToolError):
        count_entities(ws, {"aggregation": "at_time"})


def test_unknown_qualifying_predicate_is_not_false_or_a_confirmed_count():
    ws = census_workspace()
    inspected = inspect(ws, {"entity": "c1"})
    iid = inspected.data["value"]["inspection_id"]
    observe(ws, {"entity": "c1[0]", "predicate": "passed by rider", "value": "true", "evidence_ids": [iid]})
    observe(ws, {"entity": "c1[1]", "predicate": "passed by rider", "value": "false", "evidence_ids": [iid]})
    result = count_entities(ws, {"predicate": "passed by rider"}).data["value"]
    assert result["count"] == 1
    assert result["unresolved"] == ["c1[2]"]
    assert count_entities(ws, {"predicate": "not moving"}).data["value"]["count"] == 0
    assert len(count_entities(ws, {"predicate": "not moving"}).data["value"]["unresolved"]) == 3


def test_reselection_invalidates_orientation_observations_and_evidence():
    ws = census_workspace()
    iid = inspect(ws, {"entity": "c1"}).data["value"]["inspection_id"]
    ws._facing["c1@0,1"] = ["old"]
    ws._facing["c1[0]@0,1"] = ["old"]
    ws.evidence["x"] = EvidenceRecord("x", "range", ["c1"], {"c1": ws.revisions["c1"]}, "supported")
    TOOLS["select"](ws, {"entity": "c1", "instances": [1]})
    assert not ws._facing and ws.evidence["x"].superseded
    with pytest.raises(ToolError, match="predates"):
        observe(ws, {"entity": "c1[0]", "predicate": "moving", "value": "true", "evidence_ids": [iid]})


def test_image_ids_from_an_inspection_stand_for_that_inspection():
    ws = census_workspace()
    visual_ids = inspect(ws, {"entity": "c1"}).data["value"]["visual_ids"]
    observe(ws, {"entity": "c1[0]", "predicate": "passed by rider", "value": "true", "evidence_ids": visual_ids[:2]})
    assert count_entities(ws, {"predicate": "passed by rider"}).data["value"]["confirmed"] == ["c1[0]"]
    with pytest.raises(ToolError, match="unknown inspection"):
        observe(ws, {"entity": "c1[0]", "predicate": "green", "value": "true", "evidence_ids": ["visual:t99:0:mask"]})
    TOOLS["select"](ws, {"entity": "c1", "instances": [1]})
    with pytest.raises(ToolError, match="predates"):
        observe(ws, {"entity": "c1[0]", "predicate": "moving", "value": "true", "evidence_ids": visual_ids[:1]})


def test_event_occurrences_and_unique_trees_are_different_counts():
    ws = census_workspace()
    iid = inspect(ws, {"entity": "c1"}).data["value"]["inspection_id"]
    for a, b, target in ((0.5, 1, "c1[0]"), (2, 2.5, "c1[0]"), (4, 4.5, "c1[2]")):
        record_event(ws, {"predicate": "passes", "participants": ["cam", target], "start_s": a, "end_s": b,
                          "evidence_ids": [iid]})
    assert event_reduce(ws, {"operation": "count"}).data["value"]["result"] == 3
    assert event_reduce(ws, {"operation": "unique_participants"}).data["value"]["result"] == 2
    assert event_reduce(ws, {"window": [0, 1]}).data["value"]["result"] == 1


def test_duplicate_occurrence_does_not_increment_count_and_incomplete_cycle_excluded():
    ws = census_workspace()
    iid = inspect(ws, {}).data["value"]["inspection_id"]
    args = {"predicate": "dribble", "participants": ["cam"], "start_s": 1, "end_s": 2,
            "evidence_ids": [iid], "complete": False}
    record_event(ws, args)
    record_event(ws, args)
    assert event_reduce(ws, {"operation": "count"}).data["value"]["result"] == 1
    assert event_reduce(ws, {"operation": "repetitions"}).data["value"]["result"] == 0
    with pytest.raises(ToolError):
        record_event(ws, {**args, "evidence_ids": ["invented"]})


def test_speed_and_camera_keep_net_and_path_readings_for_a_round_trip():
    scene, _ = synthetic.build_scene(61, 10, metres_per_unit=1)
    t = scene.t_s
    x = 2 * np.sin(np.pi * t / t[-1])
    tr = track(np.column_stack([x, np.zeros(61), np.full(61, 4)]))
    ws = workspace(scene, {"e1": [tr]})
    result = TOOLS["speed"](ws, {"entity": "e1"}).data["value"]
    assert result["mode"] == "net"
    assert result["mps"] == pytest.approx(result["net_mps"])
    assert result["path_mps"] == pytest.approx(4 / 6, abs=1e-5)
    assert result["net_mps"] == pytest.approx(0, abs=1e-8)
    scene.cam_t[:, 0] = x
    camera = TOOLS["camera_motion"](ws, {}).data["value"]
    assert camera["motion"] == "static"  # The calibrated translation label uses net change.
    assert camera["translation_reduction"] == "net"
    assert camera["excursion"] > 1.9       # The round trip is still visible in the measured values.
    assert camera["path_length"] > 3.9
    camera_path = TOOLS["entity_track"](ws, {"entity": "cam"}).data
    assert camera_path["available"] and camera_path["frame"] == "scene"
    assert camera_path["value"]["segments"]


def test_actual_timestamps_preserve_constant_speed_and_elapsed_phase():
    scene, _ = synthetic.build_scene(21, 10)
    scene.gf_index = np.r_[np.arange(11), np.arange(20, 111, 10)]
    xyz = np.column_stack([scene.t_s, np.zeros(21), np.full(21, 4)])
    speed, _ = measure.windowed_speed(xyz, np.arange(21), 10, times=scene.t_s)
    assert np.allclose(speed, 1)
    _, hi, _ = resolve_window(scene, Window("phase", phase="first_half"))
    assert abs(scene.t_s[hi] - 5.5) <= 0.5
    ws = workspace(scene, {"e1": [track(xyz)]})
    assert TOOLS["kinematics"](ws, {"entity": "e1"}).data["value"]["path_speed"] == pytest.approx(1 / scene.scale)


def test_failed_camera_poses_do_not_establish_static_camera():
    scene, _ = synthetic.build_scene(21, 10)
    scene.pose_valid = np.r_[True, np.zeros(20, bool)]
    ws = workspace(scene, {})
    result = TOOLS["camera_motion"](ws, {}).data
    assert result["status"] == "unresolved"
    assert result["value"]["motion"] == "unresolved" and result["confidence"] == "low"


def test_image_displacement_survives_missing_depth_and_has_only_image_units():
    scene, _ = synthetic.build_scene(61, 10)
    ws = workspace(scene, {"e1": [track(np.full((61, 3), np.nan))]})
    result = TOOLS["entity_track"](ws, {"entity": "e1", "frame": "image"})
    assert result.data["available"]
    assert result.data["value"]["net"] == pytest.approx([0.6, 0])
    assert result.data["units"] == "image_fraction"


def test_first_entry_exit_honor_window_and_stops_have_a_signal():
    scene, _ = synthetic.build_scene(61, 10)
    present = np.zeros(61, bool)
    present[5:15] = True
    present[35:45] = True
    tr = track(np.tile([0, 0, 4], (61, 1)), present=present)
    events = [Event("enter", "e1", "enters_frame", "enters"), Event("leave", "e1", "leaves_frame", "leaves")]
    ws = workspace(scene, {"e1": [tr]}, events=events)
    enter = TOOLS["event_onset"](ws, {"event": "enter", "window": [3, 5]}).data["value"]
    assert enter["onset_s"] == 3.5
    leave = TOOLS["event_onset"](ws, {"event": "leave"}).data["value"]
    assert leave["onset_s"] == 1.5
    assert len(leave["occurrences"]) == 2
    tr.present[:] = True
    tr.smooth[:, 0] = np.minimum(scene.t_s, 3)
    ws.program.events.append(Event("stop", "e1", "stops_moving", "stops"))
    stopped = TOOLS["event_onset"](ws, {"event": "stop"}).data
    assert stopped["available"] and stopped["value"]["onset_s"] == pytest.approx(3.1)


def test_initial_body_frame_never_substitutes_later_orientation():
    scene, _ = synthetic.build_scene(61, 10)
    ws = workspace(scene, {"e1": [track(np.tile([0, 0, 4], (61, 1)))]})
    asked = []
    def facing(eid, positions):
        asked.extend(positions)
        return [{"pos": p, "usable": p != 0} for p in positions]
    ws.facing = facing
    result = TOOLS["entity_track"](ws, {"entity": "e1", "frame": "own", "window": [3, 5]})
    assert not result.data["available"]
    assert asked == [0]


def test_signed_relative_position_distinguishes_reflection():
    scene, _ = synthetic.build_scene(10, 10)
    a = track(np.tile([-1, 0, 4], (10, 1)))
    b = track(np.tile([0, 0, 4], (10, 1)), "e2")
    ws = workspace(scene, {"e1": [a], "e2": [b]})
    args = {"a": "e1", "b": "e2", "frame": "world0"}
    left = relative_position(ws, args).data["value"]["rows"][0]["offset"][0]
    a.smooth[:, 0] = 1
    right = relative_position(ws, args).data["value"]["rows"][0]["offset"][0]
    assert left < 0 < right and left == -right


def test_sparse_geometry_reports_generator_frames_for_followup_inspection():
    from d4rt_agent.agent_v5 import render
    scene, _ = synthetic.build_scene(10, 10)
    scene.gf_index = np.arange(10) * 7
    a = track(np.column_stack([np.arange(10), np.zeros(10), np.full(10, 4)]))
    b = track(np.tile([0, 0, 4], (10, 1)), "e2")
    ws = workspace(scene, {"e1": [a], "e2": [b]})
    relation = relative_position(ws, {"a": "e1", "b": "e2", "frame": "world0"}).data
    assert relation["window"] == [0, 63]
    assert relation["value"]["rows"][-1]["frame"] == 63
    assert relation["value"]["rows"][-1]["time_s"] == pytest.approx(6.3)
    path = TOOLS["entity_track"](ws, {"entity": "e1", "frame": "world0"}).data
    assert path["value"]["segments"][-1]["frames"][-1] == 63
    assert render.span_label(None, 0, 9, scene.gf_index) == "f0-f63"


def test_projected_similarity_angle_ignores_translation_and_scale():
    a = np.array([[0, 0], [2, 0], [1, 1]], float)
    angle = np.pi / 3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    b = 3 * a @ rotation.T + [5, -7]
    assert similarity_angle(a, b) == pytest.approx(angle)
    assert similarity_angle(a[:2], b[:2]) is None


def test_composed_loop_options_early_and_fresh_final_context():
    scene, groundings = synthetic.build_scene(40, 6, entities=[("dog", lambda t: (0.02 * t, 0, 4))])
    model = ScriptedQwen([
        _act("choose_family", family="motion"), _act("choose_kind", kind="trajectory"),
        _act("name_entity", slot="subject", value="dog"), _act("set_frame", frame="image"),
        _act("entity_track", entity="e1"), _act("done_measuring"), _act("final_answer", letter="A"),
    ])
    agent = CompositionalAgentV5(qwen=model, backbone="fake", workspace_factory=lambda clip, p: Workspace(
        clip, p, scene_fn=lambda c: scene, ground_fn=lambda c, phrase, frame: groundings[phrase]))
    task = {"id": "q", "clip": FakeClip(), "question": "Which way does the animal move?",
            "options": {"A": "The dog moves right", "B": "The dog moves left"}}
    result = agent.solve(task)
    assert result["letter"] == "A", result.get("error", result)
    assert "The dog moves right" in _texts(model.seen[0])
    assert result["program"]["requirements"][0]["frame"]["basis"] == "image"
    assert '"action": "choose_family"' not in _texts(model.seen[-1])
    assert result["evidence"][0]["observable"] == "entity_track"


def test_composed_explicit_initial_frame_rejects_scene_default():
    model = ScriptedQwen([
        _act("choose_family", family="motion"), _act("choose_kind", kind="trajectory"),
        _act("name_entity", slot="subject", value="dog"), _act("set_frame", frame="scene"),
        _act("set_frame", frame="body_initial", reference="e1", anchor_s=0),
        _act("done_measuring"), _act("final_answer", letter="A"),
    ])
    agent = CompositionalAgentV5(qwen=model, backbone="fake", repair_cycles=0, workspace_factory=lambda clip, p: Workspace(
        clip, p, scene_fn=lambda c: None, ground_fn=lambda c, phrase, frame: None))
    result = agent.solve({"id": "q", "clip": FakeClip(), "question": "How does the dog move relative to its starting orientation?",
                          "options": {"A": "Forward", "B": "Backward"}})
    assert result["letter"] == "A", result.get("error", result)
    assert any(t.get("phase") == "frame" and t["status"] == "rejected" for t in result["trace"])
    assert result["program"]["requirements"][0]["frame"]["basis"] == "body_initial"


def test_registry_has_six_families_and_twenty_one_nonoverlapping_recipes():
    assert len(kinds.FAMILIES) == 6 and len(kinds.KINDS) == 21
    assert len(kinds.COMPOSITIONAL_NAMES) == 21
    assert "tracked_objects_count" not in kinds.COMPOSITIONAL_NAMES
    assert all(kinds.kinds_of(f) for f in kinds.FAMILIES)


def test_reseeding_alignment_preserves_shared_world_basis():
    from d4rt_agent.agent_v5.backbones.reseed import align_to_scene
    rng = np.random.default_rng(4)
    source = rng.normal(size=(20, 3))
    rotation = synthetic.rot_y(35)
    target = source @ rotation.T * 2.3 + [4, -2, 9]
    result, metadata = align_to_scene(source[None], source, target)
    assert np.allclose(result[0], target)
    assert metadata["median_alignment_residual"] < 1e-10
    with pytest.raises(ValueError, match="not enough"):
        align_to_scene(source[None, :2], source[:2], target[:2])


def test_cross_session_ids_are_not_automatically_counted_as_new_objects():
    from d4rt_agent.agent_v5.capabilities import merge_identity
    ws = census_workspace()
    first = ws.grounding("c1")
    second = Grounding("tree", first.obj_ids[:1], first.masks[:1], first.probs[:1],
                       first.boxes[:1], first.area_frac[:1], first.border[:1])
    ws._ground_fn = lambda clip, phrase, frame: second
    ws.reground("c1", "tree", 1)
    result = count_entities(ws, {}).data["value"]
    assert result["count"] == 3 and result["unresolved"] == ["c1[3]"]
    assert ws.groundings["c1"].sessions == (0, 0, 0, 1)
    iid = inspect(ws, {"entity": "c1"}).data["value"]["inspection_id"]
    merge_identity(ws, {"entity": "c1", "instances": [0, 3], "evidence_ids": [iid]})
    result = count_entities(ws, {}).data["value"]
    assert result["count"] == 3 and not result["unresolved"]
    assert ws.instance_indices("c1", 0) == [0, 3]


def test_batch_geometric_predicates_compose_with_semantic_predicates():
    from d4rt_agent.agent_v5.capabilities import evaluate_predicate
    ws = census_workspace()
    # These masks do not move, although their visibility changes.
    result = evaluate_predicate(ws, {"entity": "c1", "operation": "stationary", "frame": "image"})
    assert result.data["value"]["predicate"] == "stationary:image"
    count = count_entities(ws, {"predicates": ["stationary:image"]}).data["value"]
    assert count["count"] == 2
    assert count["unresolved"] == ["c1[1]"]  # one image is insufficient to establish stationarity
    count = count_entities(ws, {"predicates": ["stationary:image", "green"]}).data["value"]
    assert count["count"] == 0 and len(count["unresolved"]) == 3


def test_diagnostic_selection_is_answer_blind_and_paired_comparison_clusters():
    from d4rt_agent.agent_v5.research.protocol import select_diagnostics, paired_compare
    rows = [{"benchmark": b, "qid": f"{b}{i}", "source": "source", "video": f"v{i}",
             "question": "What moves?", "options": {"A": "left", "B": "right"}, "hits": [], "answer": "A"}
            for b in ("DSI", "Dyn", "VLM4D") for i in range(5)]
    selected = select_diagnostics(rows, n=3)
    assert all(len(v) == 3 for v in selected["question_ids"].values())
    assert all("answer" not in r for r in selected["review"])
    baseline = {str(i): {"correct": False, "clip": {"relative_path": f"source/{i // 2}"}} for i in range(8)}
    candidate = {str(i): {"correct": True, "clip": {"relative_path": f"source/{i // 2}"}} for i in range(8)}
    compared = paired_compare(baseline, candidate, draws=100)
    assert compared["source_clusters"] == 4 and compared["paired_95_ci"] == [1, 1]
    with pytest.raises(ValueError, match="exactly"):
        paired_compare(baseline, {"0": candidate["0"]})


def test_full_body_turn_is_unwrapped_in_the_report():
    scene, _ = synthetic.build_scene(61, 10)
    ws = workspace(scene, {"e1": [track(np.tile([0, 0, 4], (61, 1)))]})
    def facing(eid, positions):
        rows = []
        for p in positions:
            angle = 2 * np.pi * p / 60
            forward = np.array([np.sin(angle), 0, np.cos(angle)])
            rows.append(dict(pos=int(p), usable=True, status="ok", forward_cam=forward, forward_world=forward,
                             forward_level=forward, up_world=np.array([0, -1, 0]), azimuth_deg=np.degrees(angle)))
        return rows
    ws.facing = facing
    assert TOOLS["orientation"](ws, {"entity": "e1"}).data["cumulative_turn_deg"] == pytest.approx(360)


def test_surface_centroid_is_reported_as_a_proxy_not_verified_correspondence():
    from d4rt_agent.agent_v5.scene import lift_instances
    scene, _ = synthetic.build_scene(41, 10)
    masks = np.zeros((1, 41, 96, 160), bool)
    for i in range(41):
        x = round(16 + 64 * i / 40)
        masks[0, i, 16:80, x:x + 48] = True
    g = Grounding("wall", np.array([0]), masks, np.ones((1, 41)), np.zeros((1, 41, 4)),
                  masks.mean((2, 3)), np.zeros((1, 41), bool))
    tr = lift_instances(scene, g, [0], entity_id="e1")
    ws = workspace(scene, {"e1": [tr]})
    result = TOOLS["entity_track"](ws, {"entity": "e1"})
    assert np.linalg.norm(tr.smooth[-1] - tr.smooth[0]) > 0.5
    assert result.data["reference_point"] == "visible_surface_proxy" and result.data["status"] == "partial"


def test_irregular_time_smoothing_keeps_constant_physical_velocity():
    from d4rt_agent.agent_v5.scene import smooth_track
    scene, _ = synthetic.build_scene(21, 10)
    scene.gf_index = np.r_[np.arange(11), np.arange(20, 111, 10)]
    p = np.column_stack([scene.t_s, scene.t_s * 2, np.full(21, 4)])
    assert np.allclose(smooth_track(scene, p), p)
    scene.cam_t = p
    assert np.allclose(measure.camera_state(scene)["trans_smooth"], scene.to_level(p))


def test_more_than_two_option_defined_events_are_declared_sequentially():
    options = {"A": "The dog enters", "B": "The cat leaves", "C": "The person sits", "D": "The car stops"}
    model = ScriptedQwen([_act("choose_family", family="events"), _act("choose_kind", kind="order_events"),
                         _act("set_event_count", count=4), *[_act("declare_event", text=v) for v in options.values()],
                         _act("done_measuring"), _act("final_answer", letter="A")])
    agent = CompositionalAgentV5(qwen=model, backbone="fake", repair_cycles=0,
        workspace_factory=lambda clip, p: Workspace(clip, p, scene_fn=lambda c: None, ground_fn=lambda c, p, f: None))
    result = agent.solve({"id": "q", "clip": FakeClip(), "question": "Which event happens first?", "options": options})
    assert result["letter"] == "A", result.get("error", result)
    assert [e["text"] for e in result["program"]["events"]] == list(options.values())
    assert len(result["program"]["requirements"][0]["events"]) == 4
    assert any("ev4 has not been localized" in note for note in result["notes"])


def test_unknown_fps_preserves_event_counts_order_and_frame_windows():
    from d4rt_agent.agent_v5.capabilities import event_window
    ws = census_workspace()
    ws.clip.fps = None
    ws.clip.label = lambda i: f"Generator frame {i} (FPS unknown)"
    iid = inspect(ws, {"frames": [0, 10]}).data["value"]["inspection_id"]
    ids = []
    for a, b in ((1, 3), (7, 9)):
        value = record_event(ws, {"predicate": "passes", "participants": ["cam", "c1[0]"],
            "start_frame": a, "end_frame": b, "evidence_ids": [iid]}).data["value"]
        ids.append(value["id"])
        assert value["start_s"] is None and value["start_frame"] == a
    assert event_reduce(ws, {"operation": "order"}).data["value"]["result"] == ids
    assert event_reduce(ws, {}).data["value"]["result"] == 2
    duration = event_reduce(ws, {"operation": "duration"}).data
    assert duration["units"] == "generator_frames" and duration["value"]["result"] == 4
    event_window(ws, {"start_event": ids[1]})
    assert ws.program.window.kind == "frames"
    assert event_reduce(ws, {}).data["value"]["result"] == 1
    with pytest.raises(ToolError, match="FPS"):
        record_event(ws, {"predicate": "passes", "participants": ["cam"],
                          "start_s": 1, "end_s": 2, "evidence_ids": [iid]})
    from d4rt_agent.agent_v5.capabilities import locate_event
    # The census workspace raises if any geometry is requested.
    localized = locate_event(ws, {"actor": "c1[0]", "predicate": "enters_frame", "frames": [0, 10]})
    assert localized.data["value"]["occurrences"][0]["start_frame"] == 7


@pytest.mark.parametrize("frames", ([999, 1000], ["bad", "window"]))
def test_invalid_tool_window_is_recoverable_in_composition(frames):
    model = ScriptedQwen([_act("choose_family", family="visual"), _act("choose_kind", kind="visual_state"),
                         _act("inspect", frames=frames), _act("done_measuring"),
                         _act("final_answer", letter="A")])
    agent = CompositionalAgentV5(qwen=model, backbone="fake", repair_cycles=0,
        workspace_factory=lambda clip, p: Workspace(clip, p, scene_fn=lambda c: None, ground_fn=lambda c, p, f: None))
    result = agent.solve({"id": "q", "clip": FakeClip(), "question": "What happens?", "options": {"A": "Nothing"}})
    assert result["letter"] == "A", result.get("error", result)
    assert all(e["status"] == "unresolved" for e in result["evidence"])
    assert any("no current supporting" in note for note in result["notes"])


def test_revised_event_anchor_invalidates_window_until_relocalized():
    from d4rt_agent.agent_v5.capabilities import event_window
    from d4rt_agent.agent_v5.program import Requirement
    ws = census_workspace()
    iid = inspect(ws, {}).data["value"]["inspection_id"]
    args = {"predicate": "enters", "participants": ["c1[0]"], "start_s": 1, "end_s": 2, "evidence_ids": [iid]}
    eid = record_event(ws, args).data["value"]["id"]
    ws.program.requirements.append(Requirement("r1", "trajectory", window=Window("event", event="entry")))
    event_window(ws, {"start_event": eid})
    assert ws.program.requirements[0].window.kind == "frames"
    record_event(ws, {**args, "occurrence_id": eid, "start_s": 2, "end_s": 3})
    assert ws.program.requirements[0].window.kind == "event"
    with pytest.raises(ToolError, match="needs its event located"):
        event_reduce(ws, {})


def test_dsi_augmentation_comparison_preserves_pairs_and_source_clusters(tmp_path):
    import json
    from d4rt_agent.agent_v5.research.protocol import prediction_rows, paired_compare
    directories = []
    for split in ("std", "hflip", "reverse", "reverse_hflip"):
        root = tmp_path / split
        (root / "answers").mkdir(parents=True)
        (root / "run_config.json").write_text(json.dumps({"bench": "dsi", "split": split}))
        (root / "answers/q.json").write_text(json.dumps({"question_id": "q", "correct": True,
            "clip": {"relative_path": f"{split}/source/video.mp4"}}))
        directories.append(root)
    predictions = prediction_rows(directories)
    assert len(predictions) == 4
    assert paired_compare(predictions, predictions, draws=100)["source_clusters"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        prediction_rows([directories[0], directories[0]])
