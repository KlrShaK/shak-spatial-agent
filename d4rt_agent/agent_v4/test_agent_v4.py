"""Orchestrator tests, driven by a scripted model and a fake pipeline.

No GPU and no checkpoints: what is under test is the host's control of the
sequence, which is where the design's guarantees live.  If the loop lets a model
answer before it has been measured, or accepts a subject for a bucket that has
none, the whole premise is gone -- and neither failure needs a real model to
reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from d4rt_agent.agent_v4 import buckets as buckets_module
from d4rt_agent.agent_v4.contracts import ActionRejected, validate_action
from d4rt_agent.agent_v4.orchestrator import AgentV4Orchestrator
from d4rt_agent.agent_v4.pipeline import Candidate, Grounding

TASK = {
    "id": "q1",
    "relative_path": "CameraBench/example.mp4",
    "question": "In this video clip, how is the observer's location moving?",
    "options": {"A": "Moving forward", "B": "Orbiting clockwise",
                "C": "Moving right", "D": "Basically unchange"},
}
SUBJECT_TASK = {**TASK, "question": "How is the character's location moving?"}


class ScriptedQwen:
    """Replays canned responses, and records what it was shown."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.seen: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]], **_: Any) -> str:
        self.seen.append(messages)
        if not self.responses:
            raise AssertionError("the loop asked for more turns than the script has")
        return self.responses.pop(0)

    def last_text(self) -> str:
        """All text the model was shown on its most recent turn."""
        return "\n".join(
            part["text"]
            for message in self.seen[-1]
            for part in message["content"]
            if part["type"] == "text"
        )


@dataclass
class FakeSampled:
    frames_rgb: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 4, 4, 3), dtype=np.uint8)
    )
    width: int = 4
    height: int = 4
    fps: float = 30.0


@dataclass
class FakePipeline:
    grounding: Grounding | None = None
    report: dict[str, Any] | None = None
    measure_error: Exception | None = None
    calls: list[str] = field(default_factory=list)
    opened_paths: list[str | None] = field(default_factory=list)

    def open(self, relative_path: str, subject: str | None = None,
             video_path: str | None = None) -> FakeSampled:
        self.calls.append(f"open:{subject}")
        self.opened_paths.append(video_path)
        return FakeSampled()

    def prepare_camera_only(self, subject: str | None = None) -> dict:
        self.calls.append("prepare_camera_only")
        return {}

    def ground(self, subject: str) -> Grounding:
        self.calls.append(f"ground:{subject}")
        return self.grounding or Grounding(status="ok", subject=subject,
                                           candidates=[Candidate(0, 0.9, [0, 0, 1, 1])])

    def commit(self, grounding: Grounding, index: int) -> dict:
        self.calls.append(f"commit:{index}")
        return {}

    def measure(self, bucket: Any, subject: str | None) -> dict:
        self.calls.append(f"measure:{bucket.name}:{subject}")
        if self.measure_error is not None:
            raise self.measure_error
        return self.report or {
            "bucket": bucket.name, "available": True, "unavailable_reason": None,
            "text": "The camera moves right for 0.412 units.",
            "metrics": {"net_displacement": 0.412, "noise_floor": 0.02},
        }


def action(name: str, **arguments: Any) -> str:
    arguments.setdefault("justification", "because")
    return "Reasoning here.\n" + json.dumps({"action": name, "arguments": arguments})


def run(responses: list[str], pipeline: FakePipeline | None = None, task=TASK, **kwargs):
    qwen = ScriptedQwen(responses)
    pipeline = pipeline or FakePipeline()
    solution = AgentV4Orchestrator(qwen=qwen, pipeline=pipeline, **kwargs).solve(task)
    return solution, qwen, pipeline


# --------------------------------------------------------------------------
# the happy paths
# --------------------------------------------------------------------------


def test_camera_bucket_never_asks_for_a_subject():
    """44% of the benchmark is camera motion, which has no subject to segment.
    Asking for one would waste SAM3 and Orient-Anything on every such question."""
    solution, _, pipeline = run([
        action("choose_bucket", bucket="traj_camera_own_frame"),
        action("final_answer", kind="text", text="C: Moving right. 0.412 units."),
    ])
    assert solution["answered"]
    assert solution["chosen_bucket"] == "traj_camera_own_frame"
    assert solution["subject"] is None
    assert not any(call.startswith("ground:") for call in pipeline.calls)
    assert "prepare_camera_only" in pipeline.calls


def test_subject_bucket_grounds_then_measures():
    solution, _, pipeline = run([
        action("choose_bucket", bucket="traj_subject_own_frame"),
        action("name_subject", subject="the runner"),
        action("final_answer", kind="text", text="A: Moving forward."),
    ], task=SUBJECT_TASK)
    assert solution["subject"] == "the runner"
    assert "ground:the runner" in pipeline.calls
    assert "commit:0" in pipeline.calls
    assert solution["measurement_available"]


def test_ambiguous_grounding_asks_the_agent_to_pick():
    pipeline = FakePipeline(grounding=Grounding(
        status="ok", subject="the runner", anchor_frame=7,
        candidates=[Candidate(0, 0.50, [0, 0, 1, 1]), Candidate(3, 0.47, [1, 1, 2, 2])],
        ambiguous=True, reason="runner-up 0.470 is within 15% of the top score 0.500",
    ))
    solution, qwen, pipeline = run([
        action("choose_bucket", bucket="orient_camera_subject_frame"),
        action("name_subject", subject="the runner"),
        action("choose_candidate", index=3),
        action("final_answer", kind="text", text="B: from character's left to right."),
    ], pipeline, task=SUBJECT_TASK)
    assert solution["chosen_candidate_index"] == 3
    assert solution["grounding_ambiguous"]
    assert "commit:3" in pipeline.calls


# --------------------------------------------------------------------------
# the host enforcing the sequence
# --------------------------------------------------------------------------


def test_answering_before_being_measured_is_refused():
    """The one thing this design exists to prevent."""
    solution, qwen, pipeline = run([
        action("final_answer", kind="text", text="C: Moving right."),
        action("choose_bucket", bucket="traj_camera_own_frame"),
        action("final_answer", kind="text", text="C: Moving right. 0.412 units."),
    ])
    assert solution["answered"]
    assert [a["status"] for a in solution["trace"]] == ["rejected", "accepted", "accepted"]
    assert "needs `choose_bucket`" in solution["trace"][0]["error"]
    assert any(call.startswith("measure:") for call in pipeline.calls)


def test_an_out_of_range_candidate_is_refused_not_committed():
    pipeline = FakePipeline(grounding=Grounding(
        status="ok", subject="the car", anchor_frame=0,
        candidates=[Candidate(0, 0.5, [0, 0, 1, 1]), Candidate(1, 0.49, [1, 1, 2, 2])],
        ambiguous=True, reason="close",
    ))
    solution, _, pipeline = run([
        action("choose_bucket", bucket="sub_obs_distance_change"),
        action("name_subject", subject="the car"),
        action("choose_candidate", index=9),
        action("choose_candidate", index=1),
        action("final_answer", kind="text", text="A: Get farther."),
    ], pipeline, task=SUBJECT_TASK)
    assert "commit:9" not in pipeline.calls
    assert "commit:1" in pipeline.calls
    rejected = [a for a in solution["trace"] if a["status"] == "rejected"]
    assert "must be one of [0, 1]" in rejected[0]["error"]


def test_a_response_with_no_action_is_rejected_and_retried():
    solution, _, _ = run([
        "I think the camera moves right, but I will not emit any JSON.",
        action("choose_bucket", bucket="traj_camera_own_frame"),
        action("final_answer", kind="text", text="C: Moving right."),
    ])
    assert solution["answered"]
    assert solution["trace"][0]["status"] == "rejected"


def test_an_unknown_bucket_name_is_rejected_with_the_valid_list():
    solution, _, _ = run([
        action("choose_bucket", bucket="camera_motion"),
        action("choose_bucket", bucket="traj_camera_own_frame"),
        action("final_answer", kind="text", text="C: Moving right."),
    ])
    error = solution["trace"][0]["error"]
    assert "traj_camera_own_frame" in error
    assert solution["answered"]


def test_the_loop_gives_up_rather_than_spinning_forever():
    solution, _, _ = run([action("choose_bucket", bucket="nope")] * 6, max_steps=4)
    assert not solution["answered"]
    assert solution["exhausted"]
    assert solution["steps_used"] == 4


# --------------------------------------------------------------------------
# failure is reported, not papered over
# --------------------------------------------------------------------------


def test_a_subject_that_cannot_be_found_gets_one_retry_then_declines():
    pipeline = FakePipeline(
        grounding=Grounding(status="not_found", subject="x", reason="no instance found")
    )
    solution, qwen, pipeline = run([
        action("choose_bucket", bucket="traj_subject_own_frame"),
        action("name_subject", subject="the pink character"),
        action("name_subject", subject="the person"),
        action("final_answer", kind="text", text="A: Moving forward."),
    ], pipeline, task=SUBJECT_TASK)
    assert solution["subject_attempts"] == 2
    assert not solution["measurement_available"]
    assert "could not find" in solution["measurement_unavailable_reason"]
    # It must never quietly measure something else instead.
    assert not any(call.startswith("measure:") for call in pipeline.calls)
    assert "UNAVAILABLE" in qwen.last_text()


def test_a_pipeline_crash_becomes_an_unavailable_measurement():
    pipeline = FakePipeline(measure_error=RuntimeError("D4RT ran out of memory"))
    solution, qwen, _ = run([
        action("choose_bucket", bucket="traj_camera_own_frame"),
        action("final_answer", kind="text", text="D: Basically unchange."),
    ], pipeline)
    assert solution["answered"]
    assert not solution["measurement_available"]
    assert "D4RT ran out of memory" in solution["measurement_unavailable_reason"]


def test_an_unavailable_report_withholds_the_contract_but_still_asks_the_question():
    """A contract for reading numbers is noise when there are no numbers; the
    question and its options still have to arrive."""
    pipeline = FakePipeline(report={
        "bucket": "traj_subject_own_frame", "available": False,
        "unavailable_reason": "no unique front", "text": "... UNAVAILABLE ...",
        "metrics": {},
    })
    _, qwen, _ = run([
        action("choose_bucket", bucket="traj_subject_own_frame"),
        action("name_subject", subject="the character"),
        action("final_answer", kind="text", text="A: Moving forward."),
    ], pipeline, task=SUBJECT_TASK)
    shown = qwen.last_text()
    assert "UNAVAILABLE" in shown
    assert "How to read this measurement" not in shown
    assert "Options:" in shown and "Orbiting clockwise" in shown


# --------------------------------------------------------------------------
# what the agent is actually shown
# --------------------------------------------------------------------------


def test_only_the_chosen_buckets_report_and_contract_are_shown():
    _, qwen, _ = run([
        action("choose_bucket", bucket="traj_camera_own_frame"),
        action("final_answer", kind="text", text="C: Moving right."),
    ])
    shown = qwen.last_text()
    assert "traj_camera_own_frame" in shown
    for other in ("sub_obs_distance_change", "orient_camera_subject_frame"):
        assert f"Measurement: `{other}`" not in shown

    # The chosen bucket's contract arrives...
    assert "How to read this measurement" in shown
    assert "This is the observer's own motion, in the frame it started in." in shown
    # ...and no other bucket's does. Phrases chosen to sit on one source line:
    # the markdown wraps, so a substring spanning a wrap never matches.
    assert "It carries no direction at all" not in shown          # distance contract
    assert "The sides are the subject's, not yours" not in shown  # sector contract


def test_the_bucket_choice_turn_lists_every_bucket_and_no_category_label():
    """Showing the model the benchmark's own `cate` would hand it the recipe."""
    _, qwen, _ = run([
        action("choose_bucket", bucket="traj_camera_own_frame"),
        action("final_answer", kind="text", text="C: Moving right."),
    ])
    first_turn = "\n".join(
        part["text"] for part in qwen.seen[0][1]["content"] if part["type"] == "text"
    )
    for name in buckets_module.NAMES:
        assert name in first_turn
    assert "cate" not in first_turn.lower().replace("indicate", "")


def test_metrics_render_as_a_table_a_human_can_scan():
    from d4rt_agent.agent_v4.orchestrator import _render_metrics

    rendered = _render_metrics({
        "segments": [
            {"segment": 1, "frame_start": 0.0, "frame_end": 3.1, "direction": "right",
             "distance": 0.12, "above_noise": True, "pan_deg_delta": -4.2},
            {"segment": 2, "frame_start": 3.1, "frame_end": 6.2, "direction": "right",
             "distance": 0.13, "above_noise": True, "pan_deg_delta": -4.4},
        ],
        "net_displacement": 0.25, "noise_floor": 0.01,
    })
    assert "pan_deg_delta" in rendered
    assert "direction" in rendered
    assert "net_displacement" in rendered


# --------------------------------------------------------------------------
# regressions from the first live smoke test
# --------------------------------------------------------------------------


def test_a_tracking_failure_declines_instead_of_losing_the_question():
    """SAM3's video model tracks its own instances and can end up overlapping
    none of the image model's pick, which raises. That happened on the runner
    clip in the first live run and took the whole question with it."""
    pipeline = FakePipeline()
    pipeline.commit = lambda g, i: (_ for _ in ()).throw(
        RuntimeError("propagation returned no object overlapping the chosen instance")
    )
    solution, qwen, _ = run([
        action("choose_bucket", bucket="traj_subject_own_frame"),
        action("name_subject", subject="the runner"),
        action("final_answer", kind="text", text="A: Moving forward."),
    ], pipeline, task=SUBJECT_TASK)
    assert solution["answered"]
    assert solution["status"] != "error" if "status" in solution else True
    assert not solution["measurement_available"]
    assert "could not be tracked" in solution["measurement_unavailable_reason"]
    assert "UNAVAILABLE" in qwen.last_text()


def test_a_grounding_crash_is_treated_as_not_found_not_as_a_lost_question():
    pipeline = FakePipeline()
    pipeline.ground = lambda s: (_ for _ in ()).throw(RuntimeError("SAM3 out of memory"))
    solution, _, _ = run([
        action("choose_bucket", bucket="sub_obs_distance_change"),
        action("name_subject", subject="the car"),
        action("name_subject", subject="the vehicle"),
        action("final_answer", kind="text", text="D: Cannot be determined."),
    ], pipeline, task=SUBJECT_TASK)
    assert solution["answered"]
    assert not solution["measurement_available"]


class GenerationExploded(RuntimeError):
    """Stands in for an OOM or a decode failure partway through a question."""


def test_an_unexpected_failure_keeps_the_state_needed_to_diagnose_it():
    """A bare traceback loses which bucket and subject were chosen -- exactly
    the two facts a diagnosis starts from.

    The failure is injected into generation rather than into the pipeline,
    because pipeline failures are already caught and turned into unavailable
    measurements; this exercises the last-resort handler around the whole loop.
    """

    class ExplodingQwen(ScriptedQwen):
        def generate(self, messages, **kwargs):
            if len(self.seen) >= 2:
                raise GenerationExploded("CUDA out of memory during generate")
            return super().generate(messages, **kwargs)

    qwen = ExplodingQwen([
        action("choose_bucket", bucket="orient_camera_subject_frame"),
        action("name_subject", subject="the man in white"),
    ])
    solution = AgentV4Orchestrator(qwen=qwen, pipeline=FakePipeline()).solve(SUBJECT_TASK)

    assert solution["status"] == "error"
    assert not solution["answered"]
    assert solution["chosen_bucket"] == "orient_camera_subject_frame"
    assert "GenerationExploded" in solution["error"]
    # The steps that DID succeed are still on record.
    assert solution["steps_used"] >= 1


def test_reports_name_the_axis_components_not_just_a_compound_word():
    """"down-forward" does not say which of the two dominated -- 26 lattice
    directions cannot express a ratio. The first live run produced exactly that
    failure: the agent asserted "forward dominates" on a 1.861 vs 1.831 split."""
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_buckets import (
        rank_components, segment_report,
    )

    assert rank_components([0.1, 2.0, 1.0]).startswith("2.000 down")
    assert "1.000 forward" in rank_components([0.1, 2.0, 1.0])
    assert "0.500 up" in rank_components([0.0, -0.5, 0.0])
    assert "0.500 left" in rank_components([-0.5, 0.0, 0.0])

    track = np.stack([np.zeros(11), np.linspace(0, 2, 11), np.linspace(0, 1, 11)], axis=1)
    report = segment_report(track, {}, noise=0.0)
    assert report["net_breakdown"].startswith("2.000 down")
    assert report["net_components"]["down"] == pytest.approx(2.0)
    assert all("d_down" in row for row in report["segments"])


# --------------------------------------------------------------------------
# anchor retry: the single best frame is not reliably propagatable
# --------------------------------------------------------------------------


def _fake_sam3(monkeypatch, *, scan, propagate):
    """Point the pipeline's SAM3 entry points at scripted stand-ins."""
    import d4rt_agent.sam_orientany_d4rt_test.traj3d_segment as seg

    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    candidates = {"masks": np.stack([mask]), "boxes": np.array([[1.0, 1.0, 3.0, 3.0]]),
                  "scores": np.array([max(scan)])}
    monkeypatch.setattr(seg, "select_anchor_frame",
                        lambda frames, subject, **k: (int(np.argmax(scan)), candidates, list(scan)))
    monkeypatch.setattr(seg, "ground_frame", lambda frame, subject, **k: candidates)
    monkeypatch.setattr(seg, "propagate_masks", propagate)
    monkeypatch.setattr(seg, "repair_masks", lambda m, f, s, anchor=0: (m, []))
    monkeypatch.setattr(seg, "save_mask_overlays", lambda *a, **k: None)
    monkeypatch.setattr(seg, "sample_mask_points", lambda m, n, **k: np.zeros((n, 2)))
    monkeypatch.setattr(seg, "write_frames", lambda sampled, paths: [])
    monkeypatch.setattr(seg, "image_processor", lambda *a, **k: None)
    return mask


def _pipeline(tmp_path):
    from d4rt_agent.agent_v4.pipeline import Models, QuestionPipeline

    pipe = QuestionPipeline(run_dir=tmp_path, models=Models())
    pipe.target = type("T", (), {"slug": "clip"})()
    pipe.sampled = FakeSampled(frames_rgb=np.zeros((32, 4, 4, 3), dtype=np.uint8))
    return pipe


def test_commit_falls_back_to_the_next_best_anchor(tmp_path, monkeypatch):
    """The clip that motivated this: the top-scoring frame could not be
    propagated at all, and the whole measurement was lost with it."""
    from d4rt_agent.agent_v4.pipeline import Grounding

    scan = [0.1] * 32
    scan[18], scan[30] = 0.90, 0.78          # 18 wins the argmax and cannot propagate
    tried: list[int] = []

    def propagate(frames_dir, subject, reference, num_frames, anchor=0, reference_box_xyxy=None):
        tried.append(anchor)
        if anchor == 18:
            raise RuntimeError("propagation returned no object overlapping the chosen instance")
        masks = np.zeros((num_frames, 4, 4), dtype=bool)
        masks[:20, 1:3, 1:3] = True
        return masks, {"prompt_used": "text", "anchor_iou_with_reference": 0.9,
                       "frames_with_mask": list(range(20))}

    mask = _fake_sam3(monkeypatch, scan=scan, propagate=propagate)
    pipe = _pipeline(tmp_path)
    grounding = pipe.ground("the runner in orange")
    assert grounding.anchor_frame == 18

    record = pipe.commit(grounding, grounding.best_index)
    assert tried[0] == 18, "the agent's own anchor must be tried first"
    assert record["requested_anchor_frame"] == 18
    assert record["anchor_frame"] == 30, "should have fallen back to the next best frame"
    assert record["anchor_retries"], "the failed attempt must be recorded, not hidden"
    # 20 propagated frames plus the anchor itself: commit stamps the chosen
    # instance's own mask onto its anchor, which is the one frame known right.
    assert record["tracked_frames_before_repair"] == 21


def test_commit_prefers_the_anchor_that_tracks_more_frames(tmp_path, monkeypatch):
    """Two frames out of 32 is technically a successful propagation and nearly
    useless downstream, so a thin result must not end the search."""
    from d4rt_agent.agent_v4.pipeline import Grounding

    scan = [0.1] * 32
    scan[18], scan[30] = 0.90, 0.78

    def propagate(frames_dir, subject, reference, num_frames, anchor=0, reference_box_xyxy=None):
        kept = 2 if anchor == 18 else 25
        masks = np.zeros((num_frames, 4, 4), dtype=bool)
        masks[:kept, 1:3, 1:3] = True
        return masks, {"prompt_used": "text", "anchor_iou_with_reference": 0.9,
                       "frames_with_mask": list(range(kept))}

    _fake_sam3(monkeypatch, scan=scan, propagate=propagate)
    pipe = _pipeline(tmp_path)
    grounding = pipe.ground("the runner")
    record = pipe.commit(grounding, grounding.best_index)
    assert record["anchor_frame"] == 30
    assert record["tracked_frames_before_repair"] == 26      # 25 propagated + the anchor


def test_commit_raises_only_after_every_anchor_has_failed(tmp_path, monkeypatch):
    def propagate(*a, **k):
        raise RuntimeError("propagation returned no object overlapping the chosen instance")

    scan = [0.5] * 32
    _fake_sam3(monkeypatch, scan=scan, propagate=propagate)
    pipe = _pipeline(tmp_path)
    grounding = pipe.ground("the runner")
    with pytest.raises(RuntimeError, match="best anchor frames"):
        pipe.commit(grounding, grounding.best_index)


def test_anchor_retries_are_spread_across_time(tmp_path, monkeypatch):
    """The top four scores are routinely four CONSECUTIVE frames -- one view
    tried four times, which on the runner clip failed four times together."""
    scan = [0.1] * 32
    for f, s in [(18, 0.90), (19, 0.89), (20, 0.88), (21, 0.87), (5, 0.60), (30, 0.55)]:
        scan[f] = s
    tried: list[int] = []

    def propagate(frames_dir, subject, reference, num_frames, anchor=0, reference_box_xyxy=None):
        tried.append(anchor)
        raise RuntimeError("propagation returned no object overlapping the chosen instance")

    _fake_sam3(monkeypatch, scan=scan, propagate=propagate)
    pipe = _pipeline(tmp_path)
    grounding = pipe.ground("the runner")
    with pytest.raises(RuntimeError):
        pipe.commit(grounding, grounding.best_index)

    assert tried[0] == 18
    assert 19 not in tried and 20 not in tried and 21 not in tried, \
        f"neighbours of the first anchor are the same view: {tried}"
    assert all(abs(a - b) >= 5 for a in tried for b in tried if a != b), tried


# --------------------------------------------------------------------------
# augmentation splits: the same relative_path exists in all four
# --------------------------------------------------------------------------


def test_the_manifest_video_path_decides_which_split_is_watched(tmp_path, monkeypatch):
    """The bug this guards cost three full evaluation runs.

    DSI-Bench ships `std`, `reverse`, `hflip` and `reverse_hflip` at the SAME
    relative_path. Deriving the clip from relative_path alone always resolves to
    `std`, so an augmentation run watches std frames while being scored against
    augmented ground truth — which lands *below* chance rather than erroring, and
    looks like a model failure instead of a plumbing failure.
    """
    from d4rt_agent.agent_v4 import pipeline as P

    opened: list[str] = []
    monkeypatch.setattr(
        "d4rt_agent.simple_v2_contracts.sample_video_cpu",
        lambda path: (opened.append(str(path)), FakeSampled())[1],
    )
    pipe = P.QuestionPipeline(run_dir=tmp_path, models=P.Models())

    pipe.open("CameraBench/x.mp4", video_path="/data/videos/reverse/CameraBench/x.mp4")
    assert opened[-1] == "/data/videos/reverse/CameraBench/x.mp4"
    assert pipe.target.video_root == Path("/data/videos/reverse")

    pipe.open("CameraBench/x.mp4", video_path="/data/videos/hflip/CameraBench/x.mp4")
    assert opened[-1] == "/data/videos/hflip/CameraBench/x.mp4", \
        "the same relative_path in a different split must NOT hit the cache"
    assert len(opened) == 2

    # ...and re-opening the identical path still caches.
    pipe.open("CameraBench/x.mp4", video_path="/data/videos/hflip/CameraBench/x.mp4")
    assert len(opened) == 2


def test_the_task_handed_to_the_agent_names_the_split(tmp_path):
    from d4rt_agent.agent_v4.run import _task_for

    entry = {"question_id": "q", "relative_path": "CameraBench/x.mp4",
             "question": "?", "options": {"A": "a", "B": "b"}, "option_letters": ["A", "B"],
             "video_path": "/data/videos/reverse/CameraBench/x.mp4", "gt": "A", "cate": 1}
    task = _task_for(entry)
    assert task["video_path"].endswith("/reverse/CameraBench/x.mp4")
    assert "gt" not in task and "cate" not in task


def test_lean_mode_keeps_the_measurement_and_drops_the_rest(tmp_path):
    """A full-census run is ~7,000 questions; at 10.9 MB each the transient
    artifacts alone would be ~150 GB and half a million inodes."""
    from d4rt_agent.agent_v4.pipeline import QuestionPipeline
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import VideoPaths

    paths = VideoPaths(root=tmp_path / "clip" / "subject")
    paths.mkdirs()
    for i in range(3):
        (paths.frames / f"{i:05d}.jpg").write_bytes(b"x" * 1000)
        (paths.masks_overlay / f"frame_{i:02d}.jpg").write_bytes(b"x" * 1000)
        (paths.orientation / f"crop_{i:02d}.png").write_bytes(b"x" * 1000)
    (paths.masks / "masks.npz").write_bytes(b"x" * 100000)
    # what the answer actually needs
    paths.points_json.write_text("{}")
    paths.segmentation_json.write_text("{}")
    (paths.d4rt / "camera.npz").write_bytes(b"x" * 100)
    (paths.orientation / "orientation.json").write_text("{}")
    (paths.analysis / "summary.json").write_text("{}")

    QuestionPipeline._discard_transients(paths)

    assert not paths.frames.exists() and not paths.masks.exists()
    assert not paths.masks_overlay.exists()
    assert not list(paths.orientation.glob("crop_*.png"))
    # ...and everything a cache hit or a re-score needs survives
    assert paths.points_json.exists() and paths.segmentation_json.exists()
    assert (paths.d4rt / "camera.npz").exists()
    assert (paths.orientation / "orientation.json").exists()
    assert (paths.analysis / "summary.json").exists()


def test_the_clip_cache_is_bounded(tmp_path, monkeypatch):
    """An unbounded cache of decoded clips (12-84 MB each) hit the 24 GB host
    limit after ~1,150 census questions and the job was SIGKILLed."""
    from d4rt_agent.agent_v4 import pipeline as P

    decoded: list[str] = []
    monkeypatch.setattr("d4rt_agent.simple_v2_contracts.sample_video_cpu",
                        lambda path: (decoded.append(str(path)), FakeSampled())[1])
    pipe = P.QuestionPipeline(run_dir=tmp_path, models=P.Models(), max_cached_clips=2)

    for i in range(5):
        pipe.open(f"d/{i}.mp4", video_path=f"/v/std/d/{i}.mp4")
    assert len(pipe._sampled_cache) == 2, "cache must not grow without bound"
    assert len(decoded) == 5

    # adjacent questions on one clip -- the census ordering -- still hit warm
    pipe.open("d/9.mp4", video_path="/v/std/d/9.mp4")
    n = len(decoded)
    pipe.open("d/9.mp4", video_path="/v/std/d/9.mp4")
    assert len(decoded) == n, "a repeat of the most recent clip must not re-decode"
