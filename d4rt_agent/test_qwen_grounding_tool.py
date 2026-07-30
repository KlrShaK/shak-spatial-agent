"""CPU-only tests for the isolated bbox/points Qwen grounding core."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

import numpy as np

from d4rt_agent.qwen_grounding_tool import (
    BBOX_MAX_NEW_TOKENS,
    POINTS_MAX_NEW_TOKENS,
    GroundingResponseError,
    QwenGroundingTool,
    grounding_cache_key,
    normalize_grounding_request,
    parse_bbox_response,
    parse_grounding_response,
    parse_points_response,
    sampled_video_clip_key,
)
from d4rt_agent.simple_v2_contracts import SampledVideo


class _FakeQwen:
    model_path = "/models/fake-qwen"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages, *, max_new_tokens=None):
        self.calls.append(
            {"messages": messages, "max_new_tokens": max_new_tokens}
        )
        return self.replies.pop(0)


def _sampled_video(
    *,
    path: Path = Path("/clips/example.mp4"),
    original_indices: tuple[int, ...] = tuple(range(0, 64, 2)),
) -> SampledVideo:
    return SampledVideo(
        video_path=path,
        frames_rgb=np.zeros((32, 3, 4, 3), dtype=np.uint8),
        original_indices=original_indices,
        total_original_frames=64,
        fps=30.0,
        width=4,
        height=3,
    )


class BboxParserTests(unittest.TestCase):
    def test_accepts_strict_object_inside_fence_and_normalizes_numbers(self) -> None:
        self.assertEqual(
            parse_bbox_response(
                '```json\n{"bbox_2d_1000":[10,20.5,300,400]}\n```'
            ),
            (10.0, 20.5, 300.0, 400.0),
        )

    def test_null_is_not_found(self) -> None:
        parsed = parse_grounding_response(
            '{"bbox_2d_1000":null}',
            mode="bbox",
        )
        self.assertEqual(parsed.status, "not_found")
        self.assertIsNone(parsed.bbox_2d_1000)

    def test_rejects_unknown_field_boolean_bounds_and_inverted_boxes(self) -> None:
        invalid = (
            '{"bbox_2d_1000":[1,2,3,4],"extra":1}',
            '{"bbox_2d_1000":[true,2,3,4]}',
            '{"bbox_2d_1000":[-1,2,3,4]}',
            '{"bbox_2d_1000":[1,2,1001,4]}',
            '{"bbox_2d_1000":[3,2,1,4]}',
            '{"bbox_2d_1000":[1,4,3,2]}',
            '{"bbox_2d_1000":[1,2,3]}',
            '{"bbox_2d_1000":"1,2,3,4"}',
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_bbox_response(raw)

    def test_rejects_non_finite_json_constants(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_bbox_response('{"bbox_2d_1000":[0,0,NaN,10]}')


class PointsParserTests(unittest.TestCase):
    def test_accepts_fewer_points_and_uses_immutable_internal_values(self) -> None:
        parsed = parse_grounding_response(
            '{"points_2d_1000":['
            '{"xy":[10,20],"description":"chest"},'
            '{"xy":[30.5,40],"description":"back"}]}',
            mode="points",
            requested_count=3,
        )
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(len(parsed.points_2d_1000 or ()), 2)
        self.assertEqual(parsed.points_2d_1000[1].xy, (30.5, 40.0))
        with self.assertRaises(FrozenInstanceError):
            parsed.points_2d_1000[0].description = "changed"

    def test_null_and_empty_list_are_not_found(self) -> None:
        for raw in (
            '{"points_2d_1000":null}',
            '{"points_2d_1000":[]}',
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(
                    parse_points_response(raw, requested_count=5)
                )
                self.assertEqual(
                    parse_grounding_response(
                        raw,
                        mode="points",
                        requested_count=5,
                    ).status,
                    "not_found",
                )

    def test_rejects_more_than_requested_and_duplicate_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "more points"):
            parse_points_response(
                '{"points_2d_1000":['
                '{"xy":[1,2],"description":"a"},'
                '{"xy":[3,4],"description":"b"}]}',
                requested_count=1,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_points_response(
                '{"points_2d_1000":['
                '{"xy":[1,2],"description":"a"},'
                '{"xy":[1.0,2.0],"description":"b"}]}',
                requested_count=2,
            )

    def test_rejects_bad_point_shapes_coordinates_and_descriptions(self) -> None:
        invalid = (
            '{"points_2d_1000":[{"xy":[1,2],"description":"a","extra":1}]}',
            '{"points_2d_1000":[{"xy":[1],"description":"a"}]}',
            '{"points_2d_1000":[{"xy":[true,2],"description":"a"}]}',
            '{"points_2d_1000":[{"xy":[1,1001],"description":"a"}]}',
            '{"points_2d_1000":[{"xy":[1,2],"description":"  "}]}',
            '{"points_2d_1000":[{"xy":[1,2],"description":3}]}',
            '{"points_2d_1000":[{"xy":[1,2],"description":"'
            + ("x" * 201)
            + '"}]}',
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_points_response(raw, requested_count=2)

    def test_count_must_be_one_through_eight(self) -> None:
        for count in (True, 0, 9):
            with self.subTest(count=count), self.assertRaises(ValueError):
                parse_points_response(
                    '{"points_2d_1000":null}',
                    requested_count=count,
                )


class CacheKeyTests(unittest.TestCase):
    def test_request_normalization_is_nfkc_trim_collapse_and_casefold(self) -> None:
        self.assertEqual(
            normalize_grounding_request("  Ｒunner\t CHEST  "),
            "runner chest",
        )

    def test_identical_normalized_requests_have_identical_keys(self) -> None:
        sampled = _sampled_video()
        left = grounding_cache_key(
            sampled, "points", 2, " Runner\tChest ", 2
        )
        right = grounding_cache_key(
            sampled, "points", 2, "runner chest", 2
        )
        self.assertEqual(left, right)
        self.assertEqual(left[0], sampled_video_clip_key(sampled))

    def test_relevant_request_or_clip_changes_produce_different_keys(self) -> None:
        sampled = _sampled_video()
        base = grounding_cache_key(sampled, "points", 2, "runner", 2)
        variants = (
            grounding_cache_key(sampled, "points", 3, "runner", 2),
            grounding_cache_key(sampled, "points", 2, "runner", 3),
            grounding_cache_key(sampled, "points", 2, "jogger", 2),
            grounding_cache_key(sampled, "bbox", 2, "runner", None),
            grounding_cache_key(
                _sampled_video(
                    original_indices=tuple(range(1, 65, 2))
                ),
                "points",
                2,
                "runner",
                2,
            ),
        )
        for variant in variants:
            self.assertNotEqual(base, variant)


class GroundingToolTests(unittest.TestCase):
    def test_bbox_call_has_one_image_fresh_context_and_host_provenance(self) -> None:
        qwen = _FakeQwen(['{"bbox_2d_1000":[100,200,300,400]}'])
        tool = QwenGroundingTool(qwen)
        sampled = _sampled_video()

        result = tool.ground(
            sampled_video=sampled,
            mode="bbox",
            t_src=3,
            request=" runner ",
            count=None,
        )

        self.assertEqual(
            {key: result[key] for key in ("status", "mode", "t_src", "request")},
            {
                "status": "ok",
                "mode": "bbox",
                "t_src": 3,
                "request": "runner",
            },
        )
        self.assertEqual(result["bbox_2d_1000"], [100.0, 200.0, 300.0, 400.0])
        self.assertNotIn("grounding_id", result)
        self.assertNotIn("point_id", result)

        call = qwen.calls[0]
        self.assertEqual(call["max_new_tokens"], BBOX_MAX_NEW_TOKENS)
        messages = call["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        content = messages[1]["content"]
        self.assertEqual(
            sum(item.get("type") == "image" for item in content), 1
        )
        all_text = messages[0]["content"] + " " + " ".join(
            str(item.get("text", "")) for item in content
        )
        self.assertIn("runner", all_text)
        self.assertNotIn("answer choice", messages[1]["content"][-1]["text"].lower())

        provenance = result["host_provenance"]
        self.assertEqual(provenance["original_t_src"], 6)
        self.assertEqual(provenance["qwen_model_path"], qwen.model_path)
        self.assertEqual(provenance["generation_seed"], 42)
        self.assertEqual(provenance["max_new_tokens"], BBOX_MAX_NEW_TOKENS)
        self.assertEqual(
            provenance["raw_grounder_response"],
            '{"bbox_2d_1000":[100,200,300,400]}',
        )
        self.assertFalse(provenance["cache_hit"])
        self.assertRegex(provenance["clip_key"], r"^clip_sha256:[0-9a-f]{64}$")
        self.assertRegex(provenance["cache_key_digest"], r"^[0-9a-f]{64}$")
        self.assertGreaterEqual(provenance["inference_wall_time_seconds"], 0.0)

    def test_points_call_uses_larger_budget_and_reports_counts_without_ids(self) -> None:
        qwen = _FakeQwen(
            [
                '{"points_2d_1000":['
                '{"xy":[11,22],"description":"chest"},'
                '{"xy":[33,44],"description":"back"}]}'
            ]
        )
        result = QwenGroundingTool(qwen).ground(
            sampled_video=_sampled_video(),
            mode="points",
            t_src=0,
            request="runner's chest and back",
            count=3,
        )
        self.assertEqual(qwen.calls[0]["max_new_tokens"], POINTS_MAX_NEW_TOKENS)
        self.assertEqual(result["requested_count"], 3)
        self.assertEqual(result["returned_count"], 2)
        self.assertEqual(
            result["points_2d_1000"],
            [
                {"xy": [11.0, 22.0], "description": "chest"},
                {"xy": [33.0, 44.0], "description": "back"},
            ],
        )
        self.assertNotIn("point_id", str(result["points_2d_1000"]))

    def test_not_found_is_successful_parsed_evidence_shape(self) -> None:
        qwen = _FakeQwen(['{"points_2d_1000":[]}'])
        result = QwenGroundingTool(qwen).ground(
            sampled_video=_sampled_video(),
            mode="points",
            t_src=1,
            request="five static points on the distant background",
            count=5,
        )
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["returned_count"], 0)
        self.assertEqual(result["points_2d_1000"], [])

    def test_each_call_uses_a_new_message_list_and_image(self) -> None:
        qwen = _FakeQwen(
            [
                '{"bbox_2d_1000":[1,2,3,4]}',
                '{"bbox_2d_1000":[2,3,4,5]}',
            ]
        )
        tool = QwenGroundingTool(qwen)
        sampled = _sampled_video()
        tool.ground(
            sampled_video=sampled,
            mode="bbox",
            t_src=0,
            request="runner",
        )
        tool.ground(
            sampled_video=sampled,
            mode="bbox",
            t_src=1,
            request="runner",
        )
        self.assertIsNot(
            qwen.calls[0]["messages"],
            qwen.calls[1]["messages"],
        )
        first_image = qwen.calls[0]["messages"][1]["content"][0]["image"]
        second_image = qwen.calls[1]["messages"][1]["content"][0]["image"]
        self.assertIsNot(first_image, second_image)

    def test_parse_failure_preserves_raw_reply_and_provenance(self) -> None:
        raw = "I cannot provide JSON."
        qwen = _FakeQwen([raw])
        with self.assertRaises(GroundingResponseError) as raised:
            QwenGroundingTool(qwen).ground(
                sampled_video=_sampled_video(),
                mode="bbox",
                t_src=0,
                request="runner",
            )
        self.assertEqual(raised.exception.raw_response, raw)
        self.assertEqual(
            raised.exception.host_provenance["raw_grounder_response"],
            raw,
        )
        self.assertNotIn("grounding_id", raised.exception.host_provenance)

    def test_input_validation_happens_before_qwen_generation(self) -> None:
        cases = (
            {"mode": "other", "t_src": 0, "request": "x", "count": None},
            {"mode": "bbox", "t_src": 0, "request": "x", "count": 1},
            {"mode": "points", "t_src": 0, "request": "x", "count": None},
            {"mode": "points", "t_src": 0, "request": "x", "count": 9},
            {"mode": "bbox", "t_src": True, "request": "x", "count": None},
            {"mode": "bbox", "t_src": 32, "request": "x", "count": None},
            {"mode": "bbox", "t_src": 0, "request": " ", "count": None},
            {"mode": "bbox", "t_src": 0, "request": "x" * 501, "count": None},
        )
        qwen = _FakeQwen([])
        tool = QwenGroundingTool(qwen)
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                tool.ground(sampled_video=_sampled_video(), **arguments)
        self.assertEqual(qwen.calls, [])

    def test_custom_prompt_file_controls_version_and_system_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.md"
            path.write_text("isolated test prompt\n", encoding="utf-8")
            qwen = _FakeQwen(['{"bbox_2d_1000":null}'])
            tool = QwenGroundingTool(qwen, prompt_path=path)
            result = tool.ground(
                sampled_video=_sampled_video(),
                mode="bbox",
                t_src=0,
                request="missing object",
            )
        self.assertEqual(
            qwen.calls[0]["messages"][0]["content"],
            "isolated test prompt",
        )
        self.assertEqual(
            result["host_provenance"]["grounding_system_prompt"],
            "isolated test prompt",
        )
        self.assertRegex(
            result["host_provenance"]["grounding_prompt_version"],
            r"^[0-9a-f]{64}$",
        )


if __name__ == "__main__":
    unittest.main()
