"""Small CPU fixtures for mask reports and isolated local video conversations."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from PIL import Image
import pytest

from d4rt_agent.agent_v5.clips import Clip
from d4rt_agent.agent_v5.scene import Grounding
from d4rt_agent.agent_v6.workspace import CandidateRequired, Workspace, WorkspaceError
from d4rt_agent.agent_v6.tools.ask_video import ask_video
from d4rt_agent.agent_v6.tools.captioner import (
    VIDEO_MAX_PIXELS, VideoQwen, caption_clip, video_input,
)
from d4rt_agent.agent_v6.tools.frames import get_frames
from d4rt_agent.agent_v6.tools.inventory import (
    candidate_report, choose_segmentation_candidate, detect_all_entities, get_segmentation_mask,
)


def visual_workspace(fps=2.0):
    masks = np.zeros((2, 21, 20, 20), dtype=bool)
    areas = np.rint(np.r_[np.linspace(24, 56, 11), np.linspace(56, 36, 11)[1:]]).astype(int)
    for index, count in enumerate(areas):
        patch = np.arange(100).reshape(10, 10) < count
        masks[0, index, 5:15, 5:15] = patch
    masks[1, 8:15, 4:8, 18:20] = True
    grounding = Grounding("car", np.arange(2), masks, np.ones((2, 21)),
                          np.zeros((2, 21, 4)), masks.mean(axis=(2, 3)),
                          np.zeros((2, 21), dtype=bool))
    clip = Clip("synthetic", {"dataset": "synthetic", "gf": {"length": 21},
                              "fps_gf": fps, "sampled16": [0, 7, 14, 20]}, Path("/unused"))
    clip._cache = {index: np.full((64, 64, 3), index, np.uint8) for index in range(21)}
    calls = []

    def grounder(clip, name, anchor):
        calls.append((name, anchor))
        return grounding

    workspace = Workspace(clip, grounder=grounder)
    workspace.scene = lambda: (_ for _ in ()).throw(AssertionError("Visual tools must not reconstruct a scene"))
    return workspace, calls


class Qwen:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.answer


def test_inventory_misses_do_not_advance_explicit_entity_retry_messages():
    workspace, _ = visual_workspace()
    grounder = workspace._grounder

    def empty(clip, name, anchor):
        grounding = grounder(clip, name, anchor)
        return replace(grounding, **{
            key: getattr(grounding, key)[:0]
            for key in ('obj_ids', 'masks', 'probs', 'boxes', 'area_frac', 'border')
        })

    workspace._grounder = empty
    report = detect_all_entities(workspace, Qwen('{"entities":["road","sky","skier"]}'),
                                 'Where is the skier?', {'A': 'left'}, 'An outdoor scene.')
    assert report.status == 'unavailable' and workspace.grounding_misses == 0
    first = get_segmentation_mask(workspace, ['skier'])
    assert 'Suggest an alternative name' in first.reason
    second = get_segmentation_mask(workspace, ['person on skis'])
    assert 'not proof that it is absent' in second.reason
    assert workspace.grounding_misses == 2


def test_mask_table_has_spans_size_reversal_edges_and_shared_labels():
    workspace, calls = visual_workspace()
    report = get_segmentation_mask(workspace, ["car"])
    assert report.columns == ("visible", "object", "size (% of image)", "at image edge")
    assert report.rows[0] == ("0.00–10.00 s", "car 1", "6% → 14% → 9%", "—")
    assert report.rows[1][0] == "4.00–7.00 s"
    assert report.rows[1][1] == "car 2"
    assert report.rows[1][3] == "right edge at 4.00–7.00 s"
    assert report.images == []
    assert [phase["change"] for phase in report.raw["instances"][0]["size_phases"]] == ["larger", "smaller"]
    masked = get_segmentation_mask(workspace, ["car 2"], times=[4, 6])
    assert len(masked.rows) == 1 and masked.rows[0][1] == "car 2"
    assert [label for label, image in masked.images] == ["4.00 s", "6.00 s"]
    assert all(isinstance(image, Image.Image) for _, image in masked.images)
    assert calls == [("car", 0)]


def test_mask_spans_preserve_gaps_and_visibility_does_not_depend_on_stale_probabilities():
    workspace, _ = visual_workspace()
    group = workspace.ground("car")
    group.grounding.masks[0, 6:10] = False
    report = get_segmentation_mask(workspace, ["car 1"])
    assert report.rows[0][0] == "0.00–2.50 s; 5.00–10.00 s"
    assert not report.raw["instances"][0]["visible"][7]
    assert "; " in report.rows[0][2]


def test_get_frames_range_has_four_time_labelled_plain_images_and_no_grounding():
    workspace, calls = visual_workspace()
    report = get_frames(workspace, range=[1, 7])
    assert [label for label, _ in report.images] == ["1.00 s", "3.00 s", "5.00 s", "7.00 s"]
    assert [np.asarray(image)[0, 0, 0] for _, image in report.images] == [2, 6, 10, 14]
    assert not calls
    with pytest.raises(WorkspaceError):
        get_frames(workspace, times=[0] * 5)


def test_inventory_lists_nouns_in_one_separate_call_then_reuses_mask_table():
    workspace, calls = visual_workspace()
    qwen = Qwen('{"entities": ["car", "the car", "camera"]}')
    report = detect_all_entities(workspace, qwen, "How many cars?", {"A": "two"}, "Cars beside a road.")
    assert len(qwen.calls) == 1
    messages, kwargs = qwen.calls[0]
    assert len(messages) == 2
    assert json.loads(messages[1]["content"]) == {
        "question": "How many cars?", "options": {"A": "two"}, "caption": "Cars beside a road."}
    assert report.raw["counts"] == {"car": 2}
    assert report.rows == get_segmentation_mask(workspace, ["car"]).rows
    assert report.images[0][0] == "4.00 s"  # first peak occupancy
    assert calls == [("car", 0)]


def test_candidate_image_uses_distinct_colours_and_choice_updates_workspace():
    workspace, calls = visual_workspace()
    with pytest.raises(CandidateRequired) as caught:
        workspace.resolve("car")
    report = candidate_report(workspace, caught.value.selection)
    candidates = report.raw["candidates"]
    assert len({tuple(item["rgb"]) for item in candidates}) == len(candidates) == 2
    pixels = np.asarray(report.images[0][1])
    assert all(np.any(np.all(pixels == item["rgb"], axis=-1)) for item in candidates)
    assert choose_segmentation_candidate(workspace, "BLUE").raw["label"] == "car 2"
    assert workspace.resolve("car").label == "car 2"
    assert calls == [("car", 0)]
    with pytest.raises(WorkspaceError):
        choose_segmentation_candidate(workspace, "red")


def test_caption_is_question_blind_and_cached_across_questions():
    workspace, calls = visual_workspace()
    workspace.question = "SECRET QUESTION"
    qwen = Qwen("A red car and a blue car on a road, with trees and fences.")
    first = caption_clip(workspace, qwen)
    workspace.begin_question()
    assert caption_clip(workspace, qwen) == first
    assert len(qwen.calls) == 1 and not calls
    messages, kwargs = qwen.calls[0]
    assert len(messages) == 2
    assert "no motion words" in messages[0]["content"]
    assert "SECRET QUESTION" not in str(messages)
    assert "first-person" in messages[0]["content"]
    assert messages[1]["content"][0]["type"] == "video"
    assert kwargs["video_metadata"][0]["fps"] == 2
    assert "left" not in first and "right" not in first


def test_video_input_uses_local_processor_sampler_and_keeps_source_times():
    workspace, calls = visual_workspace(fps=10)
    sampling = []

    def sample(metadata, fps):
        sampling.append((metadata.total_num_frames, metadata.fps, fps))
        return np.array([0, 7, 13, 20])

    qwen = SimpleNamespace(qwen=SimpleNamespace(processor=SimpleNamespace(
        video_processor=SimpleNamespace(sample_frames=sample))))
    content, metadata = video_input(workspace, qwen)
    assert sampling == [(21, 10, 2)]
    assert content[0]["video"].shape == (4, 64, 64, 3)
    assert metadata[0]["frames_indices"] == [0, 7, 13, 20]
    assert metadata[0]["fps"] == 10
    assert not calls


def test_video_question_has_only_requested_text_and_no_prior_history():
    workspace, calls = visual_workspace()
    qwen = Qwen("The skier jumps at 2.5 seconds.")
    first = ask_video(workspace, qwen, "When does the skier jump?")
    ask_video(workspace, qwen, "What colour is the jacket?")
    assert first.headline == qwen.answer
    assert all(len(messages) == 2 for messages, _ in qwen.calls)
    assert "When does the skier jump?" not in str(qwen.calls[1][0])
    assert "at most two sentences" in qwen.calls[0][0][0]["content"]
    assert not calls


def test_unknown_fps_uses_shown_images_without_fabricated_video_clock():
    workspace, _ = visual_workspace(fps=None)
    qwen = Qwen("The skier is airborne in shown frame D.")
    ask_video(workspace, qwen, "When does the skier jump?")
    messages, kwargs = qwen.calls[0]
    content = messages[1]["content"]
    assert "video_metadata" not in kwargs
    assert not any(item["type"] == "video" for item in content)
    assert any(item.get("text") == "shown frame A" for item in content)
    assert "frame rate is unknown" in content[0]["text"]


@pytest.mark.parametrize("frame_indices,expected_budget", [
    ([0, 7, 13, 20], VIDEO_MAX_PIXELS),
    ([0, 5, 10, 15, 20], int(VIDEO_MAX_PIXELS * 5 / 6)),
])
def test_video_adapter_passes_metadata_budget_and_deterministic_generation(frame_indices, expected_budget):
    captured = {}

    class Batch(dict):
        input_ids = np.ones((1, 3), dtype=int)

        def to(self, device):
            return self

    def process(messages, **kwargs):
        captured["processor"] = kwargs
        return Batch(input_ids=Batch.input_ids)

    def generate(**kwargs):
        captured["generation"] = kwargs
        return np.array([[1, 1, 1, 9]])

    base = SimpleNamespace(
        processor=SimpleNamespace(apply_chat_template=process, batch_decode=lambda values, **kw: ["caption"]),
        model=SimpleNamespace(device="cpu", generate=generate), seed=42, max_new_tokens=768,
        torch=SimpleNamespace(manual_seed=lambda seed: None, cuda=SimpleNamespace(is_available=lambda: False),
                              inference_mode=nullcontext),
    )
    qwen = VideoQwen(base)
    metadata = [{"total_num_frames": 21, "fps": 10, "frames_indices": frame_indices}]
    assert qwen.generate([], video_metadata=metadata, max_new_tokens=160) == "caption"
    settings = captured["processor"]["videos_kwargs"]
    assert settings["video_metadata"] == metadata
    assert settings["do_sample_frames"] is False
    assert settings["size"]["longest_edge"] == expected_budget
    assert captured["generation"]["do_sample"] is False
    assert captured["generation"]["max_new_tokens"] == 160
    assert qwen.processor is base.processor


def test_coherent_single_pixel_size_growth_is_not_discarded_as_jitter():
    workspace, _ = visual_workspace()
    group = workspace.ground("car")
    for index in range(21):
        group.grounding.masks[0, index] = False
        group.grounding.masks[0, index, 5:15, 5:15] = np.arange(100).reshape(10, 10) < 24 + index
    report = get_segmentation_mask(workspace, ["car 1"])
    assert report.rows[0][2] == "6% → 11%"


def test_unknown_fps_mask_table_reuses_only_labels_of_images_already_shown():
    workspace, _ = visual_workspace(fps=None)
    get_frames(workspace)
    existing = dict(workspace._frame_labels)
    report = get_segmentation_mask(workspace, ["car"])
    assert workspace._frame_labels == existing
    assert report.rows[0][0] == "shown frame A–shown frame D"
    assert "seconds" not in report.render()
    assert "4.00" not in report.render()


def test_caption_cache_survives_a_fresh_workspace_and_invalidates_for_model_change(tmp_path):
    workspace, _ = visual_workspace()
    workspace.models = SimpleNamespace(cache=tmp_path, qwen_model="model-one", seed=42)
    qwen = Qwen("A red car and a blue car beside a road.")
    caption = caption_clip(workspace, qwen)
    fresh = Workspace(workspace.clip, models=workspace.models)
    assert caption_clip(fresh, qwen) == caption
    assert len(qwen.calls) == 1
    assert (tmp_path / "clips" / "synthetic" / "v6_caption.json").is_file()
    changed = Workspace(workspace.clip, models=SimpleNamespace(cache=tmp_path, qwen_model="model-two", seed=42))
    caption_clip(changed, qwen)
    assert len(qwen.calls) == 2


def test_unknown_fps_video_helpers_use_the_same_keyframes_as_main_agent():
    workspace, _ = visual_workspace(fps=None)
    content, metadata = video_input(workspace, Qwen("caption"))
    assert metadata is None
    assert len([item for item in content if item["type"] == "image"]) == len(workspace.clip.keyframes(8))
    assert set(workspace._frame_labels) == set(workspace.clip.keyframes(8))


def test_candidate_montage_shows_disjoint_candidates_in_one_image():
    workspace, _ = visual_workspace()
    group = workspace.ground('car')
    group.grounding.masks[0, 8:] = False
    with pytest.raises(CandidateRequired) as error:
        workspace.resolve('car')
    report = candidate_report(workspace, error.value.selection)
    assert len(report.images) == 1
    assert len(report.raw['frame_indices']) == 2
    for entity in group.entities:
        assert any(group.grounding.masks[list(entity.instances), index].any()
                   for index in report.raw['frame_indices'])
    assert report.status == 'partial'


def test_mask_table_rejects_unknown_instance_labels_before_sam():
    workspace, calls = visual_workspace()
    workspace.ground('car')
    with pytest.raises(WorkspaceError, match='unavailable'):
        get_segmentation_mask(workspace, ['car 99'])
    assert len(calls) == 1


def test_successful_mask_table_has_a_headline():
    workspace, _ = visual_workspace()
    assert get_segmentation_mask(workspace, ['car']).headline
