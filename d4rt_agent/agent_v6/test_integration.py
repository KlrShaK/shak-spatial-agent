"""Controller-to-tool integration using synthetic observations and scripted Qwen."""

import json
from pathlib import Path

import numpy as np
import pytest

from d4rt_agent.agent_v5.clips import Clip
from d4rt_agent.agent_v5.scene import Grounding
from d4rt_agent.agent_v5.synthetic import build_scene
from d4rt_agent.agent_v6.agent import AgentV6
from d4rt_agent.agent_v6.workspace import Workspace


class ScriptedQwen:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        assert self.responses, "An unexpected helper or controller call exhausted the script"
        return self.responses.pop(0)


def text(messages):
    return "\n".join(message["content"] if isinstance(message["content"], str) else
                     "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
                     for message in messages)


def action(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def synthetic_workspace(*, entities=(), category=None, camera_t=lambda t: (0, 0, 0), native_facing=True):
    scene, groundings = build_scene(frames=31, fps=10, camera_t=camera_t, entities=entities)
    clip = Clip("integration", {"dataset": "synthetic", "gf": {"length": 31}, "fps_gf": 10,
                                "sampled16": list(range(0, 31, 2))}, Path("/unused"))
    clip._cache = {i: np.full((48, 80, 3), i, np.uint8) for i in range(31)}
    if category is not None:
        members = list(groundings.values())
        groundings = {category: Grounding(category, np.arange(len(members)), **{
            key: np.concatenate([getattr(member, key) for member in members])
            for key in ("masks", "probs", "boxes", "area_frac", "border")
        })}
    ground_calls = []

    def grounder(clip, phrase, anchor):
        ground_calls.append((phrase, anchor))
        return groundings[phrase]

    def orientation(clip, scene, masks, positions):
        return [{"pos": int(position), "usable": native_facing,
                 "forward_world": np.array([0., 0., 1.]), "up_world": scene.up}
                for position in positions]

    return Workspace(clip, scene=scene, grounder=grounder, orientation_provider=orientation), ground_calls


def test_real_camera_measurement_reaches_the_answer_conversation():
    workspace, ground_calls = synthetic_workspace(camera_t=lambda t: (t / 10, 0, 0))
    qwen = ScriptedQwen(
        "An outdoor scene with a wall.", '{"entities": []}',
        "1. [camera motion] Measure camera translation in its starting view.",
        action("track_camera"), '{"answer": true, "completed_plan_items": [1]}',
        '{"letter": "B", "because": "The camera measurement shows rightward translation."}',
    )
    agent = AgentV6(qwen=qwen)
    result = agent.solve({"id": "camera", "question": "Which direction does the camera move?",
                          "options": {"A": "left", "B": "right"}}, workspace=workspace)
    assert agent.dispatcher is None  # solve constructs and uses the production Toolset.
    assert result["letter"] == "B" and result["tool_calls"] == 1
    report = result["reports"][0]
    assert report["tool"] == "track_camera" and report["status"] == "ok"
    assert report["headline"].startswith("Camera MOVES RIGHT")
    assert report["raw"]["path_distance"] == pytest.approx(3)
    assert report["rendered"] in text(qwen.calls[-1][0])
    assert result["model_calls"] == len(qwen.calls) == 6
    assert not ground_calls and not qwen.responses


def test_real_ambiguity_choice_then_measurement_uses_the_chosen_instance():
    workspace, ground_calls = synthetic_workspace(category="car", entities=[
        ("left car", lambda t: (-1, 0, 4)),
        ("right car", lambda t: (0.7 + t / 60, 0, 4)),
    ])
    qwen = ScriptedQwen(
        "Two cars beside a wall.", '{"entities": ["car"]}',
        "1. [entity motion] Identify the intended car and measure its direction.",
        action("track_entity", main="car"),
        action("choose_segmentation_candidate", colour="blue"),
        action("track_entity", main="car"), '{"answer": true}',
        '{"letter": "B", "because": "The selected car has rightward world motion."}',
    )
    result = AgentV6(qwen=qwen).solve(
        {"id": "selection", "question": "How does the car on the right move?",
         "options": {"A": "left", "B": "right"}}, workspace=workspace)
    assert result["letter"] == "B" and result["tool_calls"] == 3
    ambiguity, choice, motion = result["reports"]
    assert ambiguity["title"] == "Choose a segmentation candidate"
    assert [candidate["colour"] for candidate in ambiguity["raw"]["candidates"]] == ["red", "blue"]
    assert choice["raw"]["label"] == "car 2"
    assert set(motion["raw"]["motions"]) == {"car 2"}
    assert motion["raw"]["motions"]["car 2"]["majority"] == "right"
    assert workspace.resolve("car").label == "car 2"
    assert workspace.pending_candidates is None
    assert ground_calls == [("car", 0)]
    assert "choose_segmentation_candidate" in text([qwen.calls[4][0][-1]])
    assert "choose_segmentation_candidate" not in text([qwen.calls[6][0][-1]])
    assert sum(part["type"] == "image" for message in qwen.calls[4][0]
               for part in message["content"] if isinstance(message["content"], list)) == 9
    assert motion["rendered"] in text(qwen.calls[-1][0])
    assert result["model_calls"] == len(qwen.calls) == 8 and not qwen.responses


def test_failed_facing_and_visual_fallback_preserve_real_world_translation():
    workspace, ground_calls = synthetic_workspace(
        entities=[("runner", lambda t: (-0.5 + t / 30, 0, 4))], native_facing=False)
    qwen = ScriptedQwen(
        "A person in front of a wall.", '{"entities": ["runner"]}',
        "1. [entity motion] Measure the runner relative to its starting facing.",
        action("track_entity", main="runner", frame_of_reference="subject"),
        "unknown", "unknown", "unknown",  # The real facing helper examines three crops.
        '{"answer": true}',
        '{"letter": "B", "because": "World motion is rightward; starting facing is unavailable."}',
    )
    result = AgentV6(qwen=qwen).solve(
        {"id": "missing-facing", "question": "What motion can be established for the runner?",
         "options": {"A": "left in the world", "B": "right in the world"}}, workspace=workspace)
    report = result["reports"][0]
    assert report["status"] == "partial" and report["title"].startswith("World motion")
    assert "No resolvable front" in report["reason"]
    motion = report["raw"]["motions"]["runner 1"]
    assert motion["frame_of_reference"] == "world" and motion["majority"] == "right"
    assert result["letter"] == "B" and result["tool_calls"] == 1
    fallback_calls = [call for call in qwen.calls if call[1].get("max_new_tokens") == 32]
    assert len(fallback_calls) == 3 and all(len(messages) == 1 for messages, _ in fallback_calls)
    assert report["rendered"] in text(qwen.calls[-1][0])
    assert ground_calls == [("runner", 0)]
    assert result["model_calls"] == len(qwen.calls) == 9 and not qwen.responses


def test_real_caption_and_video_tool_are_isolated_and_counted_inside_controller():
    workspace, ground_calls = synthetic_workspace()
    qwen = ScriptedQwen(
        "A skier in a red jacket on snow.", '{"entities": ["skier"]}',
        "1. [visual] Find when the skier jumps.",
        action("ask_video", question="At which time does the skier jump?"),
        "The skier jumps at 1.5 seconds.", '{"answer": true}',
        '{"letter": "B", "because": "The video viewer locates the jump at 1.5 seconds."}',
    )
    agent = AgentV6(qwen=qwen)
    result = agent.solve(
        {"id": "event", "question": "When does the skier jump?",
         "options": {"A": "0.5 seconds", "B": "1.5 seconds"}}, workspace=workspace)
    assert result["letter"] == "B" and result["reports"][0]["tool"] == "ask_video"
    assert result["reports"][0]["headline"] == "The skier jumps at 1.5 seconds."
    helpers = [(messages, kwargs) for messages, kwargs in qwen.calls if "video_metadata" in kwargs]
    assert len(helpers) == 2
    assert all(len(messages) == 2 for messages, _ in helpers)
    assert all(messages[1]["content"][0]["type"] == "video" for messages, _ in helpers)
    assert all(kwargs["video_metadata"][0]["fps"] == 10 for _, kwargs in helpers)
    assert all(kwargs["video_metadata"][0]["frames_indices"][0] == 0 and
               kwargs["video_metadata"][0]["frames_indices"][-1] == 30 for _, kwargs in helpers)
    assert "When does the skier jump?" not in text(helpers[0][0])
    assert "1. [visual]" not in text(helpers[1][0]) and "Options:" not in text(helpers[1][0])
    assert result["reports"][0]["rendered"] in text(qwen.calls[-1][0])
    assert result["model_calls"] == agent.qwen.calls == len(qwen.calls) == 7
    assert result["output_tokens"] == agent.qwen.output_tokens > 0
    assert not ground_calls and not qwen.responses
