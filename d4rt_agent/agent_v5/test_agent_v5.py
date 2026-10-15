"""GPU-free tests for AgentV5: the boundary rule, smoothing, chronological rendering, contracts, the
three-phase loop with a scripted model, context trimming and the backbone cache.

    python -m pytest d4rt_agent/agent_v5/test_agent_v5.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from d4rt_agent.agent_v5 import kinds, render, synthetic
from d4rt_agent.agent_v5.context import Conversation
from d4rt_agent.agent_v5.contracts import ActionRejected, parse_letter, validate_action
from d4rt_agent.agent_v5.orchestrator import PROMPTS, AgentV5
from d4rt_agent.agent_v5.program import Entity, QueryProgram
from d4rt_agent.agent_v5.sampling import interior_cells
from d4rt_agent.agent_v5.scene import lift_instances
from d4rt_agent.agent_v5.tools import TOOL_SPECS, TOOLS, ToolError, Workspace

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


class FakeClip:
    """The parts of ``clips.Clip`` the loop and tools use, without files."""

    def __init__(self, length: int = 40, fps: float = 6.0) -> None:
        self.slug = "synthetic"
        self.length = length
        self.fps = fps
        self.record = {"dataset": "davis", "source_hw": list(synthetic.MODEL_HW), "portrait": False,
                       "sampled16": list(np.linspace(0, length - 1, 16).astype(int)), "gf": {"length": length}}

    @property
    def answer_frames(self) -> list[int]:
        return self.record["sampled16"]

    def keyframes(self, n: int = 8) -> list[int]:
        return self.answer_frames[:: max(1, len(self.answer_frames) // n)][:n]

    def label(self, i: int) -> str:
        return f"Frame {i} (t={i / self.fps:.2f} s)"

    def frame(self, i: int) -> np.ndarray:
        return np.zeros((*synthetic.MODEL_HW, 3), dtype=np.uint8)

    def pil(self, i: int, long_side: int = 448) -> Any:
        from PIL import Image

        return Image.new("RGB", (64, 40))


def u_path(t: int) -> tuple[float, float, float]:
    """Forward 0.6, then left 0.8, then backward 1.2 (the largest part comes last)."""

    if t <= 13:
        return (0.0, 0.0, 3.0 + 0.6 * t / 13)
    if t <= 26:
        return (-0.8 * (t - 13) / 13, 0.0, 3.6)
    return (-0.8, 0.0, 3.6 - 1.2 * (t - 26) / 13)


def make_workspace(entities, kind: str = "subject_direction", program_entities=None, events=()):
    scene, groundings = synthetic.build_scene(40, 6.0, entities=entities)
    clip = FakeClip()
    program = QueryProgram("q1", kind, "measure", "", entities=[Entity("cam", "camera", role="observer")]
                           + list(program_entities or []), events=list(events))
    ground = lambda c, phrase, frame: groundings[phrase] if phrase in groundings else _empty(c)  # noqa: E731
    return Workspace(clip, program, scene_fn=lambda c: scene, ground_fn=ground), scene


def _empty(clip):
    from d4rt_agent.agent_v5.scene import Grounding

    n = clip.length
    return Grounding("x", np.zeros(0, int), np.zeros((0, n, 1, 1), bool), np.zeros((0, n)), np.zeros((0, n, 4)),
                     np.zeros((0, n)), np.zeros((0, n), bool))


# --------------------------------------------------------------------------- #
# draft rule 1: no points near the segmentation boundary
# --------------------------------------------------------------------------- #


def _cloud(gh: int, gw: int, obj: tuple[slice, slice], obj_depth: float = 2.0, bg_depth: float = 6.0) -> np.ndarray:
    z = np.full((gh, gw), bg_depth)
    z[obj] = obj_depth
    ys, xs = np.mgrid[0:gh, 0:gw]
    return np.stack([xs * 0.01 * z, ys * 0.01 * z, z], axis=-1)


def test_interior_cells_skip_mask_rim_and_depth_edges():
    gh, gw, cell = 20, 20, 8
    cloud = _cloud(gh, gw, (slice(5, 12), slice(5, 12)))           # the object is 7x7 cells at depth 2
    mask = np.zeros((gh * cell, gw * cell), bool)
    mask[4 * cell: 13 * cell, 4 * cell: 13 * cell] = True           # the mask bleeds one cell past it
    out = interior_cells(mask, cloud, np.eye(3), np.zeros(3), margin_px=1, margin_frac=0.0)
    assert out.cells is not None and out.cells.sum() >= 3
    depth = cloud[..., 2][out.cells]
    assert np.allclose(depth, 2.0), "a background or silhouette cell was sampled"
    assert out.depth_edge_cells > 0


def test_interior_cells_thin_mask_gives_no_3d():
    gh, gw, cell = 20, 20, 8
    cloud = _cloud(gh, gw, (slice(5, 15), slice(9, 11)))
    mask = np.zeros((gh * cell, gw * cell), bool)
    mask[5 * cell: 15 * cell, 9 * cell: 11 * cell] = True           # two cells wide: no interior
    out = interior_cells(mask, cloud, np.eye(3), np.zeros(3))
    assert out.cells is None


def test_lift_records_margin_and_position():
    scene, g = synthetic.build_scene(20, 6.0, entities=[("dog", lambda t: (0.0, 0.0, 4.0))])
    track = lift_instances(scene, g["dog"], [0], entity_id="e1")
    assert np.isfinite(track.margin_px).all()
    assert np.allclose(track.xyz[np.isfinite(track.xyz).all(1)], [0.0, 0.0, 4.0], atol=1e-3)


# --------------------------------------------------------------------------- #
# draft rule 2: the smoothed trajectory is the one used
# --------------------------------------------------------------------------- #


def test_entity_track_reads_the_smoothed_path():
    rng = np.random.default_rng(0)
    scene, groundings = synthetic.build_scene(40, 6.0, entities=[("dog", lambda t: (-1 + 0.05 * t, 0.0, 4.0))],
                                              rng=rng)
    ws = Workspace(FakeClip(), QueryProgram("q", "subject_direction", "measure", "",
                                            entities=[Entity("cam", "camera"), Entity("e1", "object", "dog")]),
                   scene_fn=lambda c: scene, ground_fn=lambda c, p, f: groundings[p])
    result = TOOLS["entity_track"](ws, {"entity": "e1"})
    track = ws.track("e1")
    idx = np.flatnonzero(track.present & np.isfinite(track.smooth).all(1))
    expected = scene.to_level(track.smooth[idx[-1]] - track.smooth[idx[0]]) / scene.scale
    assert result.data["series_used"] == "smoothed"
    assert np.allclose(result.data["net"], expected)
    assert not np.allclose(track.xyz[idx[-1]], track.smooth[idx[-1]]), "the noise should be smoothed away"


def test_camera_motion_reads_smoothed_angles():
    from d4rt_agent.agent_v5 import measure

    scene, _ = synthetic.build_scene(40, 6.0, camera_R=lambda t: synthetic.rot_y(0.75 * t))
    state = measure.camera_state(scene)
    assert "pan_smooth" in state and np.isfinite(state["pan_smooth"]).all()
    ws = Workspace(FakeClip(), QueryProgram("q", "camera_path", "measure", "", entities=[Entity("cam", "camera")]),
                   scene_fn=lambda c: scene, ground_fn=lambda c, p, f: None)
    result = TOOLS["camera_motion"](ws, {})
    assert "pan" in result.summary and "right" in result.text


# --------------------------------------------------------------------------- #
# draft rule 3: chronological order
# --------------------------------------------------------------------------- #


def test_u_shaped_path_reads_in_time_order():
    ws, _ = make_workspace([("dog", u_path)], program_entities=[Entity("e1", "object", "dog")])
    result = TOOLS["entity_track"](ws, {"entity": "e1"})
    line = next(l for l in result.text.splitlines() if l.startswith("In time order:"))
    order = [line.index(w) for w in ("forward", "left", "backward")]
    assert order == sorted(order), line
    assert "backward" in line.split("[largest]")[0].rsplit("->", 1)[-1], "the backward part is the largest"
    system = (PROMPTS / "system.md").read_text()
    assert "Chronological rule" in system


def test_window_argument_accepts_the_forms_the_model_writes():
    ws, _ = make_workspace([], kind="camera_translation")
    for given in ([3.0, 6.0], "[3.0, 6.0]", "3, 6", "3s-6s"):
        w = ws.window({"window": given})
        assert (w.kind, w.t0, w.t1) == ("interval", 3.0, 6.0), given
    assert ws.window({"window": None}) is ws.program.window
    assert ws.window({}) is ws.program.window
    with pytest.raises(ToolError):
        ws.window({"window": "the middle"})


def test_chronological_merges_repeats():
    line = render.chronological([("0s", "1s", "forward", 0.2), ("1s", "2s", "forward", 0.1), ("2s", "3s", "left", 0.4)])
    assert line.count("forward") == 1 and line.index("forward") < line.index("left")


# --------------------------------------------------------------------------- #
# kinds, contracts
# --------------------------------------------------------------------------- #


def test_every_measured_kind_has_a_playbook_and_known_tools():
    for spec in kinds.KINDS:
        assert spec.family in kinds.FAMILIES
        for tool in kinds.tools_for(spec.name):
            assert tool in TOOL_SPECS
            assert tool in TOOLS or tool in ("switch_kind", "done_measuring", "add_entity", "add_requirement")
        if spec.route == "measure":
            assert (PROMPTS / "playbooks" / f"{spec.name}.md").exists(), spec.name
    assert all(kinds.kinds_of(f) for f in kinds.FAMILIES if f != "semantic")
    assert max(len(kinds.kinds_of(f)) for f in kinds.FAMILIES) <= 7


def test_validate_action():
    name, args = validate_action({"action": "choose_family", "arguments": {"family": "camera", "justification": "x"}},
                                 ["choose_family"])
    assert name == "choose_family" and args["family"] == "camera"
    with pytest.raises(ActionRejected):
        validate_action({"action": "final_answer", "arguments": {"letter": "A", "justification": "x"}}, ["choose_family"])
    with pytest.raises(ActionRejected):
        validate_action({"action": "choose_family", "arguments": {"family": "camera"}}, ["choose_family"])
    with pytest.raises(ActionRejected):
        validate_action({"action": "choose_family", "arguments": {"family": "camera", "bogus": 1,
                                                                   "justification": "x"}}, ["choose_family"])
    assert parse_letter("b", ["A", "B"]) == "B"
    with pytest.raises(ActionRejected):
        parse_letter("E", ["A", "B"])


def test_slot_validation_and_entities():
    q = "How many distinct people appear in the video?"
    slot = kinds.KIND_BY_NAME["instance_count"].slots[0]
    assert kinds.validate_slot(slot, "people", q) == "person"
    with pytest.raises(ValueError):
        kinds.validate_slot(slot, "child", q)
    q2 = "Do the two dogs get closer or farther apart?"
    ents, _ = kinds.entities_from_slots("pair_relation", {"subject": "dog", "reference": "dog"}, q2)
    assert [e.kind for e in ents] == ["camera", "group"]
    q3 = "How does the distance between the man and the camera change?"
    ref = kinds.KIND_BY_NAME["metric_distance"].slots[1]
    assert kinds.validate_slot(ref, "the camera", q3) == "camera"
    ents, _ = kinds.entities_from_slots("metric_distance", {"subject": "man", "reference": "camera"}, q3)
    assert [e.id for e in ents] == ["cam", "e1"]


# --------------------------------------------------------------------------- #
# the three-phase loop, with a scripted model
# --------------------------------------------------------------------------- #


class ScriptedQwen:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    def generate(self, messages, max_new_tokens=None):
        self.seen.append(messages)
        reply = self.replies.pop(0) if self.replies else _act("done_measuring")
        return reply


def _act(name: str, **args: Any) -> str:
    return "Reasoning. " + json.dumps({"action": name, "arguments": {**args, "justification": "because"}})


def _texts(messages) -> str:
    return "\n".join(p["text"] for m in messages for p in m["content"] if p.get("type") == "text")


def _agent(replies, entities, **kw):
    scene, groundings = synthetic.build_scene(40, 6.0, entities=entities)
    qwen = ScriptedQwen(replies)

    def factory(clip, program):
        return Workspace(clip, program, scene_fn=lambda c: scene, ground_fn=lambda c, p, f: groundings.get(p) or _empty(c))

    return AgentV5(qwen=qwen, workspace_factory=factory, backbone="test", **kw), qwen


TASK = {"id": "q1", "question": "Between 0s and 6s, which way does the dog move?",
        "options": {"A": "Forward then left", "B": "Backward then left", "C": "Stays still", "D": "Right"}}


def test_loop_end_to_end_and_options_hidden_until_answer():
    replies = [
        _act("final_answer", letter="A"),                  # out of turn: rejected, the family is asked again
        _act("choose_family", family="subject_motion"),
        _act("choose_kind", kind="subject_direction"),
        _act("name_entity", slot="subject", value="dog"),
        _act("entity_track", entity="e1"),
        _act("done_measuring"),
        _act("final_answer", letter="A"),
    ]
    agent, qwen = _agent(replies, [("dog", u_path)])
    out = agent.solve({**TASK, "clip": FakeClip()})
    assert out["status"] == "answered" and out["letter"] == "A", out.get("error")
    assert out["kind"] == "subject_direction" and out["slots"] == {"subject": "dog"}
    assert [t["tool"] for t in out["tools"]] == ["entity_track"]
    assert out["trace"][0]["status"] == "rejected"
    for messages in qwen.seen[:-1]:
        assert "Forward then left" not in _texts(messages), "options leaked before the answer turn"
    assert "Forward then left" in _texts(qwen.seen[-1])
    assert "In time order:" in _texts(qwen.seen[-1]), "the answer turn carries the full dossier"


def test_two_entity_kind_asks_one_slot_per_turn():
    replies = [
        _act("choose_family", family="two_entities"),
        _act("choose_kind", kind="pair_relation"),
        _act("name_entity", slot="subject", value="dog"),
        _act("name_entity", slot="reference", value="cat"),
        _act("distance", a="e1", b="e2"),
        _act("done_measuring"),
        _act("final_answer", letter="B"),
    ]
    agent, _ = _agent(replies, [("dog", lambda t: (-1.5 + 0.03 * t, 0.0, 4.0)), ("cat", lambda t: (1.5, 0.0, 4.0))])
    task = {**TASK, "question": "Do the dog and the cat get closer or farther apart?"}
    out = agent.solve({**task, "clip": FakeClip()})
    assert out["slots"] == {"subject": "dog", "reference": "cat"}, out.get("error")
    assert "closer" in out["tools"][0]["summary"]


def test_measuring_budget_then_forced_answer():
    replies = [_act("choose_family", family="camera"), _act("choose_kind", kind="camera_path")] \
        + [_act("camera_motion")] * 5 + [_act("final_answer", letter="C")]
    agent, _ = _agent(replies, [], max_steps=3)
    out = agent.solve({**TASK, "clip": FakeClip()})
    assert out["measuring_steps"] == 3 and out["letter"] == "C", out.get("error")
    assert any("stopped after 3 steps" in n for n in out["notes"])


def test_semantic_goes_straight_to_the_answer():
    agent, qwen = _agent([_act("choose_family", family="semantic"), _act("final_answer", letter="D")], [])
    out = agent.solve({**TASK, "clip": FakeClip()})
    assert out["kind"] == "semantic" and out["letter"] == "D" and out["tools"] == []


# --------------------------------------------------------------------------- #
# context, backbone cache
# --------------------------------------------------------------------------- #


def test_context_trims_tools_but_keeps_the_opening():
    conv = Conversation("system", budget=400)
    opening = conv.add("user", "question", images=[("k", object())], kind="opening")
    for i in range(20):
        conv.add("user", "x" * 400, kind="tool", summary=f"tool {i}")
        conv.add("assistant", "ok", kind="step")
    conv.add("user", "latest", kind="tool", summary="latest")
    conv.fit()
    assert not opening.trimmed and opening.images
    assert conv.turns[-1].text == "latest"
    assert sum(t.trimmed for t in conv.turns) >= 15


def test_scene_cache_roundtrip(tmp_path: Path):
    from d4rt_agent.agent_v5.backbones import base

    scene, _ = synthetic.build_scene(10, 6.0)
    arrays = {"gf_index": scene.gf_index, "model_hw": np.asarray(scene.model_hw), "source_hw": np.asarray(scene.source_hw),
              "cam_R": scene.cam_R, "cam_t": scene.cam_t, "K": scene.K, "points": scene.points,
              "geometry_stride": np.int32(scene.stride), "grid_uv": np.zeros((1, 2)), "grid_traj": scene.grid_traj,
              "grid_conf": scene.grid_conf}

    class Fake(base.Backbone):
        name = "fake"
        calls = 0

        def reconstruct(self, clip):
            Fake.calls += 1
            return arrays

        def metric(self, clip, arrays):
            return {"metres_per_unit": 2.0, "iqr_over_median": 0.1, "source": "test"}

    clip = FakeClip(10)
    first = base.SceneCache(Fake(), tmp_path).scene(clip)
    second = base.SceneCache(Fake(), tmp_path).scene(clip)
    assert Fake.calls == 1
    assert second.metres_per_unit == 2.0 and second.metric_source == "test"
    assert (tmp_path / "clips" / "synthetic" / "fake" / "scene.npz").exists()
    assert np.allclose(first.cam_t, second.cam_t)
    assert set(base.BACKBONES) == {"point4d", "d4rt"}
    with pytest.raises(ValueError):
        base.get_backbone("nope")


def test_d4rt_pick_frames_keeps_answer_frames():
    from d4rt_agent.agent_v5.backbones.d4rt import pick_frames

    gf = np.arange(300)
    picks = pick_frames(gf, list(np.linspace(0, 299, 16).astype(int)))
    assert len(picks) == 32 and 0 in picks and 299 in picks
    picks = pick_frames(gf, list(np.linspace(0, 299, 32).astype(int)))
    assert len(picks) <= 32
