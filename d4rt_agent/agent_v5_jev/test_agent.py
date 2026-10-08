"""The jev agent end to end on V5's synthetic scenes, with a scripted decision function (no network)."""

import json

from d4rt_agent.agent_v5 import synthetic
from d4rt_agent.agent_v5.test_agent_v5 import FakeClip, _empty, u_path
from d4rt_agent.agent_v5.tools import Workspace
from d4rt_agent.agent_v5_jev.agent import JevAgent, _candidate_calls

TASK = {"id": "q1", "question": "Between 0s and 6s, which way does the dog move?",
        "options": {"A": "Forward then left", "B": "Backward then left", "C": "Stays still", "D": "Right"}}
CAPTION = {"text": "Timeline:\n- 0-6 s: a dog walks forward then left\nEntities:\n- dog | first seen 0 s | last seen 6 s | walks",
           "entities": [{"name": "dog", "first_s": 0.0, "last_s": 6.0, "behaviour": "walks"}],
           "spans": [(0.0, 3.0), (3.0, 6.0)]}


class Scripted:
    """Answers each question with the first preferred choice that is on offer; records every request."""

    def __init__(self, prefs):
        self.prefs = prefs
        self.requests = []

    def __call__(self, text, questions):
        self.requests.append((text, [dict(q) for q in questions]))
        out = {}
        for q in questions:
            values = [c["value"] for c in q["choices"]]
            wanted = self.prefs.get(q["name"], [])
            wanted = wanted if isinstance(wanted, list) else [wanted]
            pick = next((w for w in wanted if w in values), values[0])
            out[q["name"]] = {"choice": pick, "confidence": 0.9,
                              "probs": {v: (1.0 if v == pick else 0.0) for v in values}}
        return out


def _agent(decide, entities, **kw):
    scene, groundings = synthetic.build_scene(40, 6.0, entities=entities)

    def factory(clip, program):
        return Workspace(clip, program, scene_fn=lambda c: scene,
                         ground_fn=lambda c, p, f: groundings.get(p) or _empty(c))

    return JevAgent(decide=decide, workspace_factory=factory, captions=lambda slug: CAPTION, backbone="test", **kw)


def test_end_to_end_options_hidden_until_the_answer():
    decide = Scripted({"family": "subject_motion", "kind": "subject_direction", "subject": "dog",
                       "tool": ["entity_track(entity=e1, frame=scene)", "done_measuring"], "answer": "A"})
    out = _agent(decide, [("dog", u_path)]).solve({**TASK, "clip": FakeClip()})
    assert out["status"] == "answered" and out["letter"] == "A", out.get("error")
    assert out["kind"] == "subject_direction" and out["slots"] == {"subject": "dog"}
    assert [t["tool"] for t in out["tools"]] == ["entity_track"]
    bodies = [json.dumps(r) for r in decide.requests]
    for body in bodies[:-2]:
        assert "Forward then left" not in body, "options leaked before the answer"
    assert "Forward then left" in bodies[-2] and "In time order:" in bodies[-2]
    assert out["caption_only_letter"] is not None and "In time order:" not in bodies[-1]
    assert "Forward then left" in bodies[-1]


def test_no_request_carries_the_ground_truth():
    decide = Scripted({"family": "semantic", "answer": "B"})
    out = _agent(decide, []).solve({**TASK, "clip": FakeClip(), "answer": "Z-secret", "gt": "Z-secret"})
    assert out["letter"] == "B"
    assert all("Z-secret" not in json.dumps(r) for r in decide.requests)


def test_semantic_skips_measuring():
    decide = Scripted({"family": "semantic", "answer": "D"})
    out = _agent(decide, []).solve({**TASK, "clip": FakeClip()})
    assert out["kind"] == "semantic" and out["tools"] == [] and out["letter"] == "D"


def test_two_slots_in_one_call_and_distance_between_the_pair():
    decide = Scripted({"family": "two_entities", "kind": "pair_relation", "subject": "dog", "reference": "cat",
                       "tool": ["distance(a=e1, b=e2)", "done_measuring"], "answer": "B"})
    task = {**TASK, "question": "Do the dog and the cat get closer or farther apart?", "clip": FakeClip()}
    out = _agent(decide, [("dog", lambda t: (-1.5 + 0.03 * t, 0.0, 4.0)), ("cat", lambda t: (1.5, 0.0, 4.0))]
                 ).solve(task)
    assert out["slots"] == {"subject": "dog", "reference": "cat"}, out.get("error")
    assert sum(1 for t in out["trace"] if t["step"] == "slots") == 2          # both slots, one call
    assert "closer" in out["tools"][0]["summary"]


def test_no_candidate_for_a_slot_falls_back_to_semantic():
    decide = Scripted({"family": "subject_motion", "kind": "subject_direction", "answer": "C"})
    task = {**TASK, "question": "What happens next?", "clip": FakeClip()}
    agent = _agent(decide, [])
    agent.captions = lambda slug: {**CAPTION, "entities": []}
    out = agent.solve(task)
    assert out["kind"] == "semantic" and out["decomposition_ok"] is False and out["letter"] == "C"


def test_tool_budget_and_no_repeats():
    decide = Scripted({"family": "camera", "kind": "camera_path", "tool": ["camera_motion"], "answer": "C"})
    out = _agent(decide, [], max_tool_steps=3).solve({**TASK, "clip": FakeClip()})
    assert [t["tool"] for t in out["tools"]] == ["camera_motion"]           # nothing else on offer, no repeat
    assert out["letter"] == "C"


def test_candidate_calls_describe_their_arguments():
    scene, groundings = synthetic.build_scene(40, 6.0, entities=[("dog", u_path)])
    from d4rt_agent.agent_v5.program import Entity, QueryProgram

    program = QueryProgram("q", "subject_direction", "measure", "",
                           entities=[Entity("cam", "camera"), Entity("e1", "object", "dog", role="subject")])
    ws = Workspace(FakeClip(), program, scene_fn=lambda c: scene, ground_fn=lambda c, p, f: groundings[p])
    calls = _candidate_calls(ws, "subject_direction", ["entity_track"], set())
    assert len(calls) == 4 and all("e1 = dog" in c.describe() for c in calls)
    assert _candidate_calls(ws, "subject_direction", ["entity_track"], {calls[0].key})[0].key != calls[0].key


def test_event_window_is_chosen_from_the_clips_own_frame_times():
    decide = Scripted({"family": "subject_motion", "kind": ["interval", "subject_direction"], "subject": "dog",
                       "start": "0.00", "end": ["2.33", "2.50"], "tool": ["entity_track(entity=e1, frame=scene)",
                                                                     "done_measuring"], "answer": "A"})
    task = {**TASK, "question": "From the start until the frame where the dog reaches the fence, which way does the dog move?",
            "clip": FakeClip()}
    out = _agent(decide, [("dog", u_path)]).solve(task)
    w = out["program"]["window"]
    assert w["kind"] == "interval" and w["t0"] == 0.0 and w["t1"] > 0.0, out.get("error")
    end_q = next(q for _, qs in decide.requests for q in qs if q["name"] == "end")
    assert all(float(c["value"]) > 0.0 for c in end_q["choices"])           # only times after the start
    assert any(h.startswith("window: interval") for h in out["history"])
