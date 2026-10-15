"""Behavioral checks for reporting, context selection and the integrated answer workflow."""

from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from d4rt_agent.agent_v5 import kinds, measure, synthetic
from d4rt_agent.agent_v5.capabilities import count_entities, inspect, observe
from d4rt_agent.agent_v5.composition import CompositionalAgentV5
from d4rt_agent.agent_v5.contracts import ActionRejected, parse_answer_response, validate_answer_submission
from d4rt_agent.agent_v5.evidence import DESCRIPTIONS, attach
from d4rt_agent.agent_v5.final_context import build_packet, render_frame, visible_refs
from d4rt_agent.agent_v5.orchestrator import State
from d4rt_agent.agent_v5.program import Entity, QueryProgram, Requirement, Window
from d4rt_agent.agent_v5.scene import Grounding
from d4rt_agent.agent_v5.test_agent_v5 import FakeClip, ScriptedQwen, _act, _texts
from d4rt_agent.agent_v5.test_composition import census_workspace
from d4rt_agent.agent_v5.tools import TOOLS, ToolResult, Workspace


OPTIONS = {"A": "The camera turns right", "B": "The camera stays still"}
ASSESSMENT = "A: 3, supported by the current measurement. B: 0, contradicted."


def submission(letter="A", text=ASSESSMENT, action="final_answer", **kwargs):
    raw = _act(action, letter=letter, **kwargs)
    return text + "\n" + raw[raw.index("{"):]


def camera_actions():
    return [_act("choose_family", family="motion"), _act("choose_kind", kind="trajectory"),
            _act("name_entity", slot="subject", value="camera"), _act("set_frame", frame="scene")]


def make_agent(actions, **kwargs):
    scene, _ = synthetic.build_scene(40, 6)
    qwen = ScriptedQwen(actions)
    agent = CompositionalAgentV5(qwen=qwen, backbone="fake", workspace_factory=lambda clip, program: Workspace(
        clip, program, scene_fn=lambda c: scene, ground_fn=lambda c, p, f: None), **kwargs)
    return agent, qwen


def solve(agent, options=OPTIONS):
    return agent.solve({"id": "q", "clip": FakeClip(), "question": "What does the camera do?", "options": options})


def packet_state(ws, kind="count_entities"):
    req = Requirement("r1", kind, [e.id for e in ws.program.entities], window=ws.program.window)
    ws.program.requirements = [req]
    state = State(program=ws.program, workspace=ws)
    state.active_requirement = req
    return state


def add_result(state, name, args, result):
    attach(state.workspace, state.active_requirement, name, args, result)
    state.results.append((name, args, result))
    for vid in result.data.get("visual_ids", []):
        state.workspace.visuals[vid]["requirement"] = state.active_requirement.id


def packet(agent, state):
    task = {"id": "q", "clip": state.workspace.clip, "question": "How many trees?", "options": {"A": "three"}}
    return build_packet(agent, task, state)


def test_every_measurement_recipe_has_reporting_descriptions():
    administrative = {"done_measuring", "switch_kind", "add_entity", "add_requirement"}
    assert set(TOOLS) - administrative <= set(DESCRIPTIONS)


def test_reporting_preserves_values_and_does_not_reconstruct_image_evidence():
    ws = census_workspace()
    state = packet_state(ws)
    result = count_entities(ws, {})
    before = deepcopy(result.data["value"])
    attach(ws, state.active_requirement, "count_entities", {}, result)
    assert result.data["value"] == before
    record = result.evidence[0]
    assert record.observations["sampling"] == "generator_frames"
    assert record.observations["detection_completeness"] == "unknown"
    assert record.selected_instances["c1"] == [0, 1, 2]
    assert record.quality["label"] == "unknown" and ws._scene is None


def test_compound_components_do_not_inherit_each_others_availability():
    ws = census_workspace()
    state = packet_state(ws, "stability")
    result = ToolResult("profile", "profile", data={"rhythm": {"available": False, "reason": "FPS unknown"},
                        "jitter": {"available": True, "confidence": "medium", "value": {"rms": 2}}})
    attach(ws, state.active_requirement, "camera_profile", {}, result)
    assert [(e.component, e.available) for e in result.evidence] == [("rhythm", False), ("jitter", True)]
    assert result.evidence[0].observations["valid_samples"] is None
    assert result.evidence[1].quality["basis"] == "heuristic"


def test_unavailable_measurement_reports_unknown_observations():
    ws = census_workspace()
    state = packet_state(ws, "range")
    result = ToolResult("failed", "failed", data={"available": False, "reason": "no depth"})
    attach(ws, state.active_requirement, "distance", {"a": "c1[0]", "b": "cam"}, result)
    record = result.evidence[0]
    assert not record.available and record.status == "unresolved"
    assert record.observations["valid_samples"] is None and record.quality["label"] == "unknown"


def test_sparse_coverage_and_pose_failures_use_actual_generator_frames():
    scene, _ = synthetic.build_scene(5, 10)
    scene.gf_index = np.array([0, 3, 11, 20, 40])
    scene.pose_valid = np.array([True, True, False, True, True])
    ws = Workspace(FakeClip(41, 10), QueryProgram("q", "trajectory", "measure", "", entities=[Entity("cam", "camera")]),
                   scene_fn=lambda c: scene, ground_fn=lambda c, p, f: None)
    ws._scene = scene
    state = packet_state(ws, "trajectory")
    result = ToolResult("motion", "motion", data={"available": True, "series": {"frames": list(range(5))}})
    attach(ws, state.active_requirement, "camera_motion", {}, result)
    observations = result.evidence[0].observations
    assert observations["intervals_gf"] == [[0, 3], [20, 40]]
    assert observations["valid_samples"] == 4 and observations["sampling"] == "sampled_geometry"


def test_measurement_packet_has_no_automatic_images_and_semantic_packet_has_standard_frames():
    agent, model = make_agent(camera_actions() + [_act("camera_motion"), _act("done_measuring"), submission()])
    out = solve(agent)
    assert out["letter"] == "A", out.get("error")
    assert out["final_visual_ids"] == [] and out["assessment_present"]
    assert not any(part["type"] == "image" for message in model.seen[-1] for part in message["content"])
    agent, model = make_agent([_act("choose_family", family="visual"), _act("choose_kind", kind="visual_state"),
                              _act("done_measuring"), submission()])
    out = solve(agent)
    assert out["letter"] == "A" and out["final_visual_ids"] == [f"frame:{f}" for f in FakeClip().answer_frames]


def test_mixed_requirements_do_not_restore_standard_frames():
    ws = census_workspace()
    state = packet_state(ws, "visual_state")
    ws.program.requirements.append(Requirement("r2", "range", ["c1"]))
    agent, _ = make_agent([])
    _, _, turns, _, _, _ = packet(agent, state)
    assert turns == []


def test_count_packet_has_few_actual_mask_overlays_and_logical_ids():
    ws = census_workspace()
    state = packet_state(ws)
    result = count_entities(ws, {})
    add_result(state, "count_entities", {}, result)
    agent, _ = make_agent([])
    conv, _, turns, manifest, _, _ = packet(agent, state)
    assert 1 <= len(turns) <= 4
    assert all(v["annotation"] == "mask" for v in manifest.values())
    assert not any(vid.startswith("frame:") for vid, _ in turns)
    assert any(np.asarray(turn.images[0][1])[1:-1, 1:-1].any() for _, turn in turns)
    assert '"count":3' in _texts(conv.messages())
    mask, _ = render_frame(ws, "c1", 0, "mask")
    boxes, _ = render_frame(ws, "c1", 0, "box")
    assert np.any(np.asarray(mask) != np.asarray(boxes))


def test_count_annotations_respect_instant_and_simultaneous_reductions():
    from d4rt_agent.agent_v5.final_context import count_frames
    ws = census_workspace()
    assert count_frames(ws, count_entities(ws, {"aggregation": "at_time", "at_s": 0}).data, "c1")[0] == [0]
    result = count_entities(ws, {"aggregation": "max_simultaneous"}).data
    assert count_frames(ws, result, "c1")[0] == [1]


def test_missing_count_segmentation_never_becomes_a_zero_visual_assertion():
    ws = census_workspace()
    image, limitation = render_frame(ws, "missing", 0)
    assert "does not establish absence" in limitation and image is not None


def test_exact_inspection_frames_and_old_scopes_preserve_dependency_images():
    ws = census_workspace()
    state = packet_state(ws)
    first = inspect(ws, {"entity": "c1[0]", "frame_ids": [0, 1]})
    add_result(state, "inspect", {"entity": "c1[0]", "frame_ids": [0, 1]}, first)
    iid = first.data["value"]["inspection_id"]
    assertion = observe(ws, {"entity": "c1[0]", "predicate": "green", "value": "true", "frames": [0, 1], "evidence_ids": [iid]})
    add_result(state, "observe", {}, assertion)
    last = inspect(ws, {"entity": "c1[0]", "frame_ids": [8, 10]})
    add_result(state, "inspect", {"entity": "c1[0]", "frame_ids": [8, 10]}, last)
    agent, _ = make_agent([])
    _, refs, turns, _, _, _ = packet(agent, state)
    ids = [vid for vid, _ in turns]
    assert any(iid in vid for vid in ids)
    assert any(last.data["value"]["inspection_id"] in vid for vid in ids)
    ws.invalidate("c1")
    _, refs, turns, _, _, _ = packet(agent, state)
    assert turns == [] and refs == {}


@pytest.mark.parametrize("invalid", ([999], [], [True], list(range(9))))
def test_inspection_rejects_invalid_exact_generator_frames(invalid):
    with pytest.raises(ValueError, match="frame_ids"):
        inspect(census_workspace(), {"frame_ids": invalid})


@pytest.mark.parametrize("failure", ("duplicate", "malformed", "multiple", "trailing", "invalid_label"))
def test_bounded_recovery_preserves_prose_and_keeps_measurement_frames_omitted(failure):
    raw = submission()
    if failure == "duplicate":
        raw = raw.replace('"letter": "A"', '"letter": "A", "letter": "B"')
    elif failure == "malformed":
        raw = raw[:-1]
    elif failure == "multiple":
        raw += "\n" + submission()
    elif failure == "trailing":
        raw += "\ntrailing text"
    else:
        raw = submission("invalid")
    agent, model = make_agent(camera_actions() + [_act("camera_motion"), _act("done_measuring"),
                             raw, raw, raw, submission("B", text="", action="answer_only")])
    out = solve(agent)
    assert out["letter"] == "B" and out["schema_recovery"] == "answer_only", out.get("error")
    assert out["assessment_text"] == ASSESSMENT and out["final_visual_ids"] == []
    assert len([r for r in out["trace"] if r["phase"] == "answer_recovery"]) == 1
    assert "unverified advisory notes" in _texts(model.seen[-1])


def test_failed_recovery_leaves_answer_empty():
    agent, _ = make_agent(camera_actions() + [_act("done_measuring"), "{}", "{}", "{}", "{}"])
    out = solve(agent)
    assert out["letter"] is None and out["schema_recovery"] == "failed"


def test_formatting_retry_does_not_repeat_or_discard_assessment():
    agent, model = make_agent(camera_actions() + [_act("done_measuring"), submission("invalid"), submission("B", text="")])
    out = solve(agent)
    assert out["letter"] == "B" and out["assessment_text"] == ASSESSMENT
    assert len(out["assessment_history"]) == 1
    assert "do not repeat assessments" in _texts(model.seen[-1])


def test_freeform_score_consistency_is_advisory_not_a_host_answer_rule():
    agent, _ = make_agent(camera_actions() + [_act("done_measuring"), submission("B")])
    out = solve(agent)
    assert out["letter"] == "B" and out["assessment_complete"] is None


def test_conditional_repair_is_one_round_and_four_calls():
    repair = {"requirement": "r1", "option": "A", "claim": "camera path", "missing": "trajectory detail",
              "evidence": [], "mode": "measure", "could_change_choice": "could distinguish turning from stillness"}
    agent, _ = make_agent(camera_actions() + [_act("done_measuring"), submission(repair_request=repair),
        *[_act("camera_motion") for _ in range(4)], submission("B")])
    out = solve(agent)
    assert out["letter"] == "B", out.get("error")
    assert len(out["repair_history"]) == 1 and out["repair_history"][0]["calls_consumed"] == 4
    assert len(out["tools"]) == 4
    assert out["repair_history"][0]["before_letter"] == "A"
    assert out["trace"][-1]["visual_ids"] == []


def test_repair_cannot_exceed_remaining_total_budget():
    repair = {"requirement": "r1", "option": "A", "claim": "path", "missing": "detail", "evidence": [],
              "mode": "measure", "could_change_choice": "could change the preferred option"}
    agent, _ = make_agent(camera_actions() + [_act("done_measuring"), submission(repair_request=repair),
                                           _act("camera_motion"), submission()], max_steps=2)
    out = solve(agent)
    assert out["letter"] == "A" and out["measuring_steps"] == 2
    assert out["repair_history"][0]["calls_consumed"] == 1


def test_current_and_supplied_references_only_and_dynamic_option_labels():
    options = {"yes": "turning", "no": "still"}
    agent, _ = make_agent(camera_actions() + [_act("done_measuring"), submission("yes")])
    assert solve(agent, options)["letter"] == "yes"
    with pytest.raises(ActionRejected, match="current"):
        validate_answer_submission({"letter": "yes", "evidence_ids": ["frame:0"]}, options, {}, repair_allowed=False)
    text, _ = parse_answer_response('A: 2, the {right} direction.\n' + _act("final_answer", letter="A"))
    assert "{right}" in text


def test_resume_guard_and_policy_defaults(tmp_path):
    from d4rt_agent.agent_v5.run import parse_args, validate_resume
    assert parse_args(["--results-dir", "/tmp/a", "--stack", "compositional"]).answer_policy == "rated_repair"
    assert parse_args(["--results-dir", "/tmp/a"]).answer_policy == "plain"
    assert parse_args(["--results-dir", "/tmp/a"]).repair_tool_limit == 4
    (tmp_path / "answers").mkdir()
    (tmp_path / "answers/q.json").write_text("{}")
    (tmp_path / "run_config.json").write_text(json.dumps({"run_signature": "same"}))
    validate_resume(tmp_path, "same")
    with pytest.raises(ValueError, match="different"):
        validate_resume(tmp_path, "changed")


def test_offline_audit_and_report_do_not_confuse_parsing_with_grounding(tmp_path):
    from d4rt_agent.agent_v5.research.assessment_audit import audit_assessment
    from d4rt_agent.agent_v5.report import report
    row = {"question_id": "qa_source_clip_0", "letter": "B", "correct": False,
        "reference": {"answer": "A", "options": OPTIONS}, "assessment_text": ASSESSMENT + " Citation: frame:0.",
        "assessment_present": True, "answer_protocol": "freeform_assessment_v1", "schema_recovery": "answer_only",
        "trace": [{"phase": "answer", "status": "accepted", "evidence_ids": [], "visual_ids": []}],
        "generation_costs": {"calls": 3, "input_tokens_estimate": 120, "output_tokens": 20}}
    audit = audit_assessment(row)
    assert audit["mechanically_complete"] and audit["assessment_answer_consistent"] is False
    assert audit["citations_not_in_final_packet"] == ["frame:0"]
    assert "manual review" in audit["grounding_review"]
    (tmp_path / "answers").mkdir()
    (tmp_path / "answers/q.json").write_text(json.dumps(row))
    (tmp_path / "answers/old.json").write_text(json.dumps({"question_id": "cameraqa_source_clip_1", "correct": True}))
    text = report(tmp_path, None)
    assert "1/1 parseable assessments" in text and "across 1/1 questions" in text


def test_source_hash_changes_when_policy_source_changes(tmp_path):
    from d4rt_agent.agent_v5.run import source_digest
    source = tmp_path / "d4rt_agent/agent_v5"
    source.mkdir(parents=True)
    (source / "policy.py").write_text("policy = 'plain'")
    before = source_digest(tmp_path)
    (source / "policy.py").write_text("policy = 'rated'")
    assert source_digest(tmp_path) != before


def test_trimmed_visuals_are_removed_from_live_reference_registry():
    ws = census_workspace()
    state = packet_state(ws)
    result = inspect(ws, {"entity": "c1", "frame_ids": [0, 1, 7, 10]})
    add_result(state, "inspect", {}, result)
    agent, _ = make_agent([], token_budget=500)
    conv, refs, turns, _, _, _ = packet(agent, state)
    conv.add("user", "answer now", kind="answer")
    conv.fit()
    supplied = visible_refs(refs, turns)
    assert not any(turn.images for _, turn in turns)
    assert not any(vid in supplied for vid, _ in turns)


def test_claim_repair_preserves_existing_evidence_and_cannot_request_a_second_round():
    repair = {"requirement": "r1", "option": "A", "claim": "path", "missing": "detail", "evidence": [],
              "mode": "measure", "could_change_choice": "could change which option wins"}
    agent, _ = make_agent(camera_actions() + [_act("camera_motion"), _act("done_measuring"),
        submission(repair_request=repair), _act("camera_motion"), _act("done_measuring"),
        submission(repair_request=repair), submission()])
    out = solve(agent)
    assert out["letter"] == "A" and len(out["repair_history"]) == 1
    assert len(out["tools"]) == 2 and len(out["evidence"]) == 2
    assert any("repair is unavailable" in r.get("error", "") for r in out["trace"])


def test_compact_trajectory_keeps_order_and_existing_turning_points():
    from d4rt_agent.agent_v5.evidence import compact_result
    scene, _ = synthetic.build_scene(10, 10)
    scene.gf_index = np.arange(10) * 7
    ws = Workspace(FakeClip(64, 10), QueryProgram("q", "range", "measure", "", entities=[Entity("cam", "camera")]),
                   scene_fn=lambda c: scene, ground_fn=lambda c, p, f: None)
    ws._scene = scene
    state = packet_state(ws, "range")
    curve = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5, 6], float)
    result = ToolResult("range", "range", data={"value": {"trend": "closer_then_farther", "pivots": [0, 4, 9]},
        "series": {"frames": list(range(10)), "smooth": curve}, "available": True})
    attach(ws, state.active_requirement, "distance", {"a": "cam", "b": "cam"}, result)
    packet = json.loads(compact_result(result, result.evidence))[0]
    assert packet["value"]["trend"] == "closer_then_farther"
    assert packet["value"]["pivots_gf"] == [0, 28, 63]
    assert result.data["value"]["pivots"] == [0, 4, 9]
    assert any(r["frame"] == 28 and r["smooth"] == 1 for r in packet["samples"]["rows"])
    assert packet["samples"]["rows"][-1]["frame"] == 63
    assert np.array_equal(result.data["series"]["smooth"], curve)


def test_source_correlation_and_projection_limitations_remain_in_packet():
    ws = census_workspace()
    state = packet_state(ws, "region_relation")
    from d4rt_agent.agent_v5.capabilities import report as observation
    one = observation("region", {"projected_inside": True}, note="Projection alone does not establish physical contact.",
                      entities=["c1"], frame="image", units="image_fraction")
    two = observation("view", {"area": 0.2}, entities=["c1"], frame="image", units="image_fraction")
    attach(ws, state.active_requirement, "region_geometry", {}, one)
    attach(ws, state.active_requirement, "view_state", {}, two)
    assert one.evidence[0].source_ids == two.evidence[0].source_ids
    assert "physical contact" in one.evidence[0].limitations[0]
    assert "no depth" in one.evidence[0].axes


def test_instance_metadata_and_visibility_are_scoped_to_the_requested_identity():
    ws = census_workspace()
    state = packet_state(ws)
    result = inspect(ws, {"entity": "c1[1]", "frame_ids": [0, 1, 2]})
    attach(ws, state.active_requirement, "inspect", {"entity": "c1[1]"}, result)
    record = result.evidence[0]
    assert record.entities == ["c1[1]"] and record.selected_instances == {"c1[1]": [1]}
    assert record.observations["visibility_gaps"]["c1[1]"] == [[0, 0], [2, 2]]
    assert ws._scene is None


def test_count_masks_include_census_candidates_after_a_single_instance_selection():
    ws = census_workspace()
    TOOLS["select"](ws, {"entity": "c1", "instances": [0]})
    state = packet_state(ws)
    result = count_entities(ws, {})
    add_result(state, "count_entities", {}, result)
    agent, _ = make_agent([])
    _, _, turns, _, _, _ = packet(agent, state)
    peak = next(turn for vid, turn in turns if ":1:mask" in vid)
    expected, _ = render_frame(ws, "c1", 1, instances=[0, 1, 2])
    assert np.array_equal(np.asarray(peak.images[0][1]), np.asarray(expected))


def test_visual_manifest_retains_artifacts_omitted_by_the_targeted_image_cap():
    ws = census_workspace()
    state = packet_state(ws)
    for frames in ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10]):
        result = inspect(ws, {"entity": "c1", "frame_ids": frames})
        add_result(state, "inspect", {"entity": "c1", "frame_ids": frames}, result)
    agent, _ = make_agent([])
    _, _, turns, manifest, _, omitted = packet(agent, state)
    assert len(turns) == 8 and len(omitted) == 3
    assert set(omitted) <= set(manifest)
    assert all(manifest[vid]["frames"] for vid in omitted)


def test_generic_show_artifact_is_invalidated_by_its_parent_binding(monkeypatch):
    from PIL import Image
    from d4rt_agent.agent_v5.context import Conversation
    ws = census_workspace()
    ws.grounding("c1")
    state = packet_state(ws)
    agent, _ = make_agent([])
    monkeypatch.setitem(TOOLS, "show", lambda ws, args: ToolResult("show", "show", images=[Image.new("RGB", (20, 20))]))
    agent._run_tool(Conversation("test", 16000), [{}], state, "show", {"target": "entities"})
    assert packet(agent, state)[2]
    ws.invalidate("c1")
    assert packet(agent, state)[2] == []


def test_complementary_requirement_repair_preserves_unaffected_evidence():
    repair = {"requirement": "r1", "option": "A", "claim": "lens or motion", "missing": "focal ratio",
              "evidence": [], "mode": "reroute", "could_change_choice": "could distinguish zoom from translation"}
    agent, _ = make_agent(camera_actions() + [_act("camera_motion"), _act("done_measuring"),
        submission(repair_request=repair), _act("add_requirement", kind="zoom", entities=["cam"]),
        _act("camera_zoom"), _act("done_measuring"), submission()])
    out = solve(agent)
    assert out["letter"] == "A", out.get("error")
    assert [r["kind"] for r in out["program"]["requirements"]] == ["trajectory", "zoom"]
    assert {e["requirement"] for e in out["evidence"] if not e["superseded"]} == {"r1", "r2"}
    assert out["repair_history"][0]["calls_consumed"] == 3


def test_legacy_value_frame_indices_are_mapped_only_in_the_answer_packet():
    from d4rt_agent.agent_v5.evidence import compact_result
    scene, _ = synthetic.build_scene(5, 10)
    scene.gf_index = np.array([0, 3, 11, 20, 40])
    ws = Workspace(FakeClip(41, 10), QueryProgram("q", "range", "measure", "", entities=[Entity("cam", "camera")]),
                   scene_fn=lambda c: scene, ground_fn=lambda c, p, f: None)
    ws._scene = scene
    state = packet_state(ws, "range")
    value = {"frame": 2, "requested_frame": 1, "frame_gap": 1, "units": 2.5}
    result = ToolResult("range", "range", data={"available": True, "value": dict(value)})
    attach(ws, state.active_requirement, "distance", {}, result)
    displayed = json.loads(compact_result(result, result.evidence))[0]["value"]
    assert displayed["frame"] == 11 and displayed["requested_frame"] == 3
    assert displayed["frame_gap_gf"] == 8 and displayed["estimator_sample_gap"] == 1
    assert result.data["value"] == value
    result = ToolResult("onset", "onset", data={"value": {"onset": 3,
        "occurrences": [{"onset": 3, "onset_s": 2.0, "bracket": [2, 3]}]}})
    attach(ws, state.active_requirement, "event_onset", {}, result)
    displayed = json.loads(compact_result(result, result.evidence))[0]["value"]
    assert displayed["onset"] == 20 and displayed["occurrences"][0]["bracket"] == [11, 20]
    assert result.data["value"]["onset"] == 3
