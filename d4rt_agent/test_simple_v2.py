"""CPU contract and orchestration tests for simple v2."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from d4rt_agent.simple_v2 import (
    SYSTEM_PROMPT,
    SimpleV2Orchestrator,
    _validate_final_evidence,
    replay_tool_trace,
    resolve_bindings,
)
from d4rt_agent.simple_v2_backend import LiveD4RTBackend
from d4rt_agent.simple_v2_contracts import (
    ACTION_SCHEMAS,
    grounding_points,
    restricted_python_math,
    uniform_sample_indices,
    validate_action,
)
from d4rt_agent.simple_v2_eval import (
    DEFAULT_DEMO_DATA,
    DEFAULT_WORLDTRACK_NPZ,
    MatchedWorldTrackGT,
    aggregate_files,
    load_alignment_scale_from_metadata,
    measure_d4rt_result,
    merge_d4rt_results,
    score_question,
)
from d4rt_agent.simple_v2_30b import QWEN_30B_MODEL_ID, prepare_30b_arguments


class SamplingContractTest(unittest.TestCase):
    def test_exact_rounded_32_frame_contract(self) -> None:
        indices = uniform_sample_indices(64)
        expected = tuple(int(value) for value in np.rint(np.linspace(0, 63, 32)))
        self.assertEqual(indices, expected)
        self.assertEqual(indices[:4], (0, 2, 4, 6))
        self.assertEqual(indices[-2:], (61, 63))

    def test_short_video_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 32"):
            uniform_sample_indices(31)


class ActionContractTest(unittest.TestCase):
    def test_qwen_schemas_do_not_contain_point_mode(self) -> None:
        self.assertNotIn("point_mode", json.dumps(ACTION_SCHEMAS))

    def test_qwen_has_only_three_actions_without_frame_inspection(self) -> None:
        self.assertEqual(
            [schema["name"] for schema in ACTION_SCHEMAS],
            ["query_d4rt", "python_math", "final_answer"],
        )

    def test_system_prompt_is_generic_and_teaches_complete_path_queries(self) -> None:
        prompt = SYSTEM_PROMPT.lower()
        self.assertNotIn("basketball", prompt)
        self.assertIn("distance covered", prompt)
        self.assertIn("travel wording has priority", prompt)
        self.assertIn("decide the requested measurement and its time interval separately", prompt)
        self.assertIn("first and last visible\n  appearance", prompt)
        self.assertIn("disjoint\nvisible segments", prompt)
        self.assertIn("never bridges a disappearance/reappearance gap", prompt)
        self.assertIn("path(0-10) + path(20-31)", prompt)
        self.assertIn('"t_tgt":[5,24]', SYSTEM_PROMPT)
        full_targets = json.dumps(list(range(32)), separators=(",", ":"))
        self.assertIn(f'"t_tgt":{full_targets}', SYSTEM_PROMPT)
        self.assertIn("[0,31] alone is never a path over the full video", SYSTEM_PROMPT)
        self.assertIn("Never repeat rejected arguments", SYSTEM_PROMPT)

    def test_system_prompt_tool_examples_are_valid_actions(self) -> None:
        examples = [
            json.loads(line)
            for line in SYSTEM_PROMPT.splitlines()
            if line.startswith('{"action":"') and "ACTION_NAME" not in line
        ]
        self.assertEqual(len(examples), 8)
        self.assertEqual(
            [validate_action(example)[0] for example in examples],
            [
                "query_d4rt",
                "python_math",
                "query_d4rt",
                "python_math",
                "final_answer",
                # Example C composes speed from displacement and elapsed time.
                "python_math",
                # Example D selects a non-zero viewpoint for a directional question
                # and answers it in words.
                "query_d4rt",
                "final_answer",
            ],
        )

    def test_final_answer_kind_defaults_to_numeric(self) -> None:
        name, args = validate_action({
            "action": "final_answer",
            "arguments": {
                "value": 1.25,
                "unit": "meters",
                "evidence_ids": ["d4rt_1", "math_1"],
                "limitations": "",
            },
        })
        self.assertEqual(name, "final_answer")
        self.assertEqual(args["kind"], "numeric")
        self.assertEqual(args["value"], 1.25)

    def test_text_answer_needs_text_and_allows_empty_evidence(self) -> None:
        _, args = validate_action({
            "action": "final_answer",
            "arguments": {"kind": "text", "text": "  I am doing well.  "},
        })
        self.assertEqual(args["kind"], "text")
        self.assertEqual(args["text"], "I am doing well.")
        self.assertEqual(args["evidence_ids"], [])
        with self.assertRaisesRegex(ValueError, "non-empty text"):
            validate_action({
                "action": "final_answer",
                "arguments": {"kind": "text", "text": "   "},
            })
        with self.assertRaisesRegex(ValueError, "exceeds 2000 characters"):
            validate_action({
                "action": "final_answer",
                "arguments": {"kind": "text", "text": "x" * 2001},
            })
        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            validate_action({
                "action": "final_answer",
                "arguments": {"kind": "direction", "text": "left"},
            })

    def test_numeric_answer_still_requires_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "must cite evidence_ids"):
            validate_action({
                "action": "final_answer",
                "arguments": {"value": 1.0, "unit": "meters", "evidence_ids": []},
            })

    def test_qwen_chooses_any_viewpoint_within_the_sampled_range(self) -> None:
        def _query(t_cam: int) -> dict[str, object]:
            return {
                "action": "query_d4rt",
                "arguments": {
                    "label": "cupboard",
                    "bbox_2d_1000": [300, 240, 470, 640],
                    "t_src": 20,
                    "t_tgt": [25],
                    "t_cam": t_cam,
                    "justification": "Express the position from the requested viewpoint.",
                },
            }

        for t_cam in (0, 25, 31):
            self.assertEqual(validate_action(_query(t_cam))[1]["t_cam"], t_cam)
        for outside in (-1, 32):
            with self.assertRaisesRegex(ValueError, r"t_cam is outside \[0, 31\]"):
                validate_action(_query(outside))

    def test_merge_refuses_evidence_from_different_camera_frames(self) -> None:
        def _result(t_cam: int, target: int) -> dict[str, object]:
            return {
                "point_mode": "centroid",
                "t_cam": t_cam,
                "t_tgt": [target],
                "predictions": [{
                    "benchmark_aligned_xyz_m": [1.0, 0.0, 0.0], "visible": True
                }],
            }

        merged = merge_d4rt_results([_result(25, 5), _result(25, 24)])
        self.assertEqual(merged["t_tgt"], [5, 24])
        with self.assertRaisesRegex(ValueError, "mixes camera frames"):
            merge_d4rt_results([_result(0, 5), _result(25, 24)])
        with self.assertRaisesRegex(ValueError, "missing t_cam"):
            merge_d4rt_results([{"point_mode": "centroid", "t_tgt": [5], "predictions": []}])

    def test_30b_entry_point_injects_and_enforces_model(self) -> None:
        arguments = prepare_30b_arguments(["--point-mode", "ensemble5"])
        self.assertEqual(arguments[-2:], ["--qwen-model", QWEN_30B_MODEL_ID])
        cached = (
            "/cache/models--Qwen--Qwen3-VL-30B-A3B-Instruct/"
            "snapshots/revision"
        )
        self.assertEqual(
            prepare_30b_arguments(["--qwen-model", cached]),
            ["--qwen-model", cached],
        )
        with self.assertRaisesRegex(ValueError, "requires Qwen/Qwen3-VL-30B"):
            prepare_30b_arguments(["--qwen-model", "Qwen/Qwen3-VL-8B-Instruct"])

    def test_qwen_cannot_emit_point_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "never select or emit"):
            validate_action({
                "action": "query_d4rt",
                "arguments": {
                    "point_mode": "centroid",
                    "label": "object",
                    "bbox_2d_1000": [0, 0, 10, 10],
                    "t_src": 0,
                    "t_tgt": [31],
                    "t_cam": 0,
                    "justification": "Need geometry.",
                },
            })
        with self.assertRaisesRegex(ValueError, "never select or emit"):
            validate_action({
                "action": "python_math",
                "arguments": {
                    "bindings": {"x": {"point_mode": "centroid"}},
                    "code": "x = 1",
                    "justification": "Calculate a value.",
                },
            })

    def test_query_times_and_justification_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            validate_action({
                "action": "query_d4rt",
                "arguments": {
                    "label": "object",
                    "bbox_2d_1000": [0, 0, 10, 10],
                    "t_src": 0,
                    "t_tgt": [32],
                    "t_cam": 0,
                    "justification": "Need geometry.",
                },
            })
        with self.assertRaisesRegex(ValueError, "declare the sampled source frame"):
            validate_action({
                "action": "query_d4rt",
                "arguments": {
                    "label": "object",
                    "bbox_2d_1000": [0, 0, 10, 10],
                    "t_tgt": [31],
                    "t_cam": 0,
                    "justification": "Need geometry.",
                },
            })

    def test_final_evidence_accepts_question_selected_intervals(self) -> None:
        endpoint_evidence = {
            "d4rt_1": {"t_tgt": [5, 24]},
            "math_1": {"outputs": {"value": 2.0}},
        }
        _validate_final_evidence(
            "endpoint_displacement",
            {"value": 2.0, "unit": "meters", "evidence_ids": ["d4rt_1", "math_1"]},
            endpoint_evidence,
        )

        path_evidence = {
            "d4rt_1": {"t_tgt": [5, 6, 7]},
            "math_1": {"outputs": {"value": 3.0}},
        }
        _validate_final_evidence(
            "distance_travelled",
            {"value": 3.0, "unit": "m", "evidence_ids": ["d4rt_1", "math_1"]},
            path_evidence,
        )
        path_evidence["d4rt_1"]["t_tgt"] = [5, 7]
        with self.assertRaisesRegex(ValueError, "selected inclusive interval"):
            _validate_final_evidence(
                "distance_travelled",
                {"value": 3.0, "unit": "m", "evidence_ids": ["d4rt_1", "math_1"]},
                path_evidence,
            )

    def test_unit_gate_applies_only_to_the_metre_scored_tasks(self) -> None:
        evidence = {
            "d4rt_1": {"t_tgt": [0, 31]},
            "math_1": {"outputs": {"value": 2.0}},
        }
        with self.assertRaisesRegex(ValueError, "scored in meters"):
            _validate_final_evidence(
                "endpoint_displacement",
                {"value": 2.0, "unit": "centimeters", "evidence_ids": ["d4rt_1", "math_1"]},
                evidence,
            )
        # An open-ended task carries no metres GT, so any sensible unit is allowed and
        # the unknown task id must not raise.
        _validate_final_evidence(
            "object_speed",
            {"value": 2.0, "unit": "m/s", "evidence_ids": ["d4rt_1", "math_1"]},
            evidence,
        )

    def test_text_answer_needs_no_measurement_evidence(self) -> None:
        _validate_final_evidence(
            "small_talk", {"kind": "text", "text": "Doing well.", "evidence_ids": []}, {}
        )
        # Cited ids must still exist, even for text.
        with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
            _validate_final_evidence(
                "small_talk",
                {"kind": "text", "text": "Doing well.", "evidence_ids": ["d4rt_9"]},
                {},
            )


class UnscoredAnswerTest(unittest.TestCase):
    def test_text_answer_scores_as_unscored(self) -> None:
        score = score_question(
            task_id="endpoint_displacement",
            final_answer={
                "kind": "text",
                "text": "It moved toward the camera.",
                "evidence_ids": ["d4rt_1", "math_1"],
            },
            evidence={},
            gt=None,
            width=640,
            height=480,
        )
        self.assertFalse(score["scored"])
        self.assertEqual(score["answer_kind"], "text")
        self.assertEqual(score["agent_final_text"], "It moved toward the camera.")
        self.assertIn("no automated scoring", score["unscored_reason"])

    def test_task_without_ground_truth_is_unscored_not_an_error(self) -> None:
        score = score_question(
            task_id="object_speed",
            final_answer={"value": 1.5, "unit": "m/s", "evidence_ids": ["d4rt_1", "math_1"]},
            evidence={},
            gt=None,
            width=640,
            height=480,
        )
        self.assertFalse(score["scored"])
        self.assertEqual(score["answer_kind"], "numeric")
        self.assertIn("no WorldTrack ground truth", score["unscored_reason"])


class GroundingPolicyTest(unittest.TestCase):
    def test_centroid_is_one_box_center(self) -> None:
        points = grounding_points("centroid", [250, 250, 750, 750], 641, 361)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["pixel_uv"], [320.0, 180.0])
        self.assertEqual(points[0]["offset_px"], [0.0, 0.0])

    def test_ensemble_is_reproducible_radius_and_bounds_constrained(self) -> None:
        bbox = [450, 400, 550, 600]
        first = grounding_points("ensemble5", bbox, 640, 360)
        second = grounding_points("ensemble5", bbox, 640, 360)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        x0, x1 = 0.45 * 639, 0.55 * 639
        y0, y1 = 0.4 * 359, 0.6 * 359
        for point in first:
            x, y = point["pixel_uv"]
            dx, dy = point["offset_px"]
            self.assertLessEqual(math.hypot(dx, dy), 12.0 + 1e-9)
            self.assertGreaterEqual(x, x0)
            self.assertLessEqual(x, x1)
            self.assertGreaterEqual(y, y0)
            self.assertLessEqual(y, y1)
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 639)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, 359)

    def test_tiny_edge_box_still_constrains_offsets(self) -> None:
        points = grounding_points("ensemble5", [0, 0, 4, 4], 640, 360)
        center = np.asarray(points[0]["pixel_uv"])
        for item in points:
            point = np.asarray(item["pixel_uv"])
            self.assertTrue(np.all(point >= 0))
            self.assertLessEqual(float(np.linalg.norm(point - center)), 12.0 + 1e-9)

    def _aggregation_backend(self, mode: str):
        from d4rt_agent.simple_v2_contracts import SampledVideo

        backend = object.__new__(LiveD4RTBackend)
        backend.point_mode = mode
        backend.benchmark_scale = 2.0
        backend.sampled_video = SampledVideo(
            video_path=Path("fake.mp4"),
            frames_rgb=np.zeros((32, 4, 4, 3), dtype=np.uint8),
            original_indices=tuple(range(32)),
            total_original_frames=32,
            fps=15.0,
            width=4,
            height=4,
        )
        return backend

    def test_centroid_preserves_raw_xyz_when_invisible(self) -> None:
        backend = self._aggregation_backend("centroid")
        points = grounding_points("centroid", [0, 0, 1000, 1000], 4, 4)
        output = {
            "xyz_3d": np.asarray([[[1.0, 2.0, 3.0]]]),
            "uv_2d": np.asarray([[[0.5, 0.5]]]),
            "visibility": np.asarray([[-1.0]]),
            "confidence": np.asarray([[0.2]]),
        }
        result = backend._aggregate_target(points, output, 0, 0)
        self.assertEqual(result["raw_xyz"], [1.0, 2.0, 3.0])
        self.assertEqual(result["benchmark_aligned_xyz_m"], [2.0, 4.0, 6.0])
        self.assertFalse(result["visible"])

    def test_ensemble_requires_three_visible_finite_points(self) -> None:
        backend = self._aggregation_backend("ensemble5")
        points = grounding_points("ensemble5", [0, 0, 1000, 1000], 4, 4)
        output = {
            "xyz_3d": np.asarray([[[float(index), 0.0, 0.0]] for index in range(5)]),
            "uv_2d": np.asarray([[[0.5, 0.5]]] * 5),
            "visibility": np.asarray([[1.0], [1.0], [1.0], [-1.0], [-1.0]]),
            "confidence": np.asarray([[0.2]] * 5),
        }
        result = backend._aggregate_target(points, output, 0, 0)
        self.assertTrue(result["visible"])
        self.assertEqual(result["valid_count"], 3)
        self.assertEqual(result["raw_xyz"], [1.0, 0.0, 0.0])
        output["visibility"] = np.asarray([[1.0], [1.0], [-1.0], [-1.0], [-1.0]])
        result = backend._aggregate_target(points, output, 0, 0)
        self.assertFalse(result["visible"])
        self.assertIsNone(result["raw_xyz"])


class RestrictedMathTest(unittest.TestCase):
    def test_endpoint_and_visible_path(self) -> None:
        result = restricted_python_math(
            {
                "points": [
                    [0, 0, 0], [1, 0, 0], [9, 0, 0], [11, 0, 0], [13, 0, 0]
                ],
                "visible": [True, True, False, True, True],
            },
            "endpoint = dist(points[0], points[-1])\ntravel = path_length(points, visible)",
        )
        self.assertEqual(result["endpoint"], 13.0)
        # Segment [0,1] contributes 1 and segment [3,4] contributes 2.  The
        # disappearance at index 2 prevents an incorrect 10-unit bridge.
        self.assertEqual(result["travel"], 3.0)

    def test_import_loop_files_and_attributes_are_rejected(self) -> None:
        rejected = (
            "import os",
            "for x in values:\n    y = x",
            "value = open(1)",
            "value = values.shape",
            "value = __import__(1)",
        )
        for code in rejected:
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    restricted_python_math({"values": [1, 2]}, code)

    def test_non_finite_result_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            restricted_python_math({"x": 1.0}, "bad = x / 0")

    def test_string_index_error_explains_separate_numeric_bindings(self) -> None:
        with self.assertRaisesRegex(ValueError, "bind each evidence field"):
            restricted_python_math(
                {"points": [[0.0, 0.0, 0.0]]},
                "value = path_length(points, points['visibility'])",
            )


class GroundTruthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DEFAULT_WORLDTRACK_NPZ.exists() or not DEFAULT_DEMO_DATA.exists():
            raise unittest.SkipTest("basketball WorldTrack data is unavailable")

    def test_scale_reads_metadata_only(self) -> None:
        scale, provenance = load_alignment_scale_from_metadata(DEFAULT_DEMO_DATA)
        self.assertAlmostEqual(scale, 1.1193330652951246)
        self.assertFalse(provenance["predicted_track_arrays_read"])

    def test_gt_is_recomputed_on_rounded_sample(self) -> None:
        gt = MatchedWorldTrackGT(DEFAULT_WORLDTRACK_NPZ, uniform_sample_indices(64))
        endpoint = gt.measurement("endpoint_displacement")
        path = gt.measurement("distance_travelled")
        self.assertAlmostEqual(endpoint["value_m"], 0.5624797, places=6)
        self.assertAlmostEqual(path["value_m"], 3.1000854, places=6)
        self.assertEqual(path["visible_segments"], [[0, 3], [21, 31]])

    def test_gt_measurements_use_the_selected_interval_and_disjoint_segments(self) -> None:
        gt = MatchedWorldTrackGT(DEFAULT_WORLDTRACK_NPZ, uniform_sample_indices(64))
        endpoint = gt.measurement("endpoint_displacement", 1, 3)
        trajectory = gt.canonical_trajectory()["xyz_m"]
        self.assertAlmostEqual(
            endpoint["value_m"], math.dist(trajectory[1], trajectory[3]), places=9
        )
        self.assertEqual(endpoint["endpoint_sampled_frames"], [1, 3])

        path = gt.measurement("distance_travelled", 1, 22)
        self.assertEqual(path["sampled_interval"], [1, 22])
        self.assertEqual(path["visible_segments"], [[1, 3], [21, 22]])


class _ScriptedQwen:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.message_snapshots: list[str] = []
        self.initial_content_layout: list[dict] | None = None

    def generate(self, messages):
        if self.initial_content_layout is None:
            self.initial_content_layout = []
            for part in messages[1]["content"]:
                if part["type"] == "image":
                    self.initial_content_layout.append({
                        "type": "image",
                        "size": list(part["image"].size),
                    })
                else:
                    self.initial_content_layout.append({
                        "type": "text",
                        "text": part["text"],
                    })
        # Images are represented by their type only so the snapshot is JSON-safe.
        serializable = []
        for message in messages:
            content = []
            for part in message["content"]:
                content.append({key: value for key, value in part.items() if key != "image"})
            serializable.append({"role": message["role"], "content": content})
        self.message_snapshots.append(json.dumps(serializable))
        return next(self.responses)


class _FakeBackend:
    point_mode = "ensemble5"

    def query(self, **kwargs):
        predictions = []
        for target in kwargs["t_tgt"]:
            value = float(target) / 31.0
            predictions.append({
                "sampled_frame_index": target,
                "original_frame_index": target,
                "raw_xyz": [value, 0.0, 0.0],
                "benchmark_aligned_xyz_m": [value, 0.0, 0.0],
                "visible": True,
                "math_xyz_aligned_m": [value, 0.0, 0.0],
            })
        return {
            "point_mode": self.point_mode,
            "bbox_2d_1000": kwargs["bbox_2d_1000"],
            "t_src": kwargs["t_src"],
            "t_tgt": kwargs["t_tgt"],
            "t_cam": kwargs["t_cam"],
            "predictions": predictions,
            "math_trajectory_aligned_xyz_m": [
                item["math_xyz_aligned_m"] for item in predictions
            ],
            "math_visibility": [True] * len(predictions),
            "visibility_coverage": 1.0,
        }


class OrchestratorTest(unittest.TestCase):
    def test_evidence_loop_and_replay_without_exposing_policy(self) -> None:
        qwen = _ScriptedQwen([
            json.dumps({
                "action": "query_d4rt",
                "arguments": {
                    "label": "ball",
                    "bbox_2d_1000": [400, 400, 500, 500],
                    "t_src": 0,
                    "t_tgt": [0],
                    "t_cam": 0,
                    "justification": "Need the two endpoint positions.",
                },
            }),
            json.dumps({
                "action": "query_d4rt",
                "arguments": {
                    "label": "ball",
                    "bbox_2d_1000": [400, 400, 500, 500],
                    "t_src": 0,
                    "t_tgt": [31],
                    "t_cam": 0,
                    "justification": "Need the ending position.",
                },
            }),
            json.dumps({
                "action": "python_math",
                "arguments": {
                    "bindings": {
                        "start": {"evidence_id": "d4rt_1", "path": ["predictions", 0, "benchmark_aligned_xyz_m"]},
                        "end": {"evidence_id": "d4rt_2", "path": ["predictions", 0, "benchmark_aligned_xyz_m"]},
                    },
                    "code": "value = dist(start, end)",
                    "justification": "Compute the requested Euclidean displacement.",
                },
            }),
            json.dumps({
                "action": "final_answer",
                "arguments": {
                    "value": 1.0,
                    "unit": "m",
                    "evidence_ids": ["d4rt_1", "d4rt_2", "math_1"],
                    "limitations": "Sparse point evidence.",
                },
            }),
        ])
        from d4rt_agent.simple_v2_contracts import SampledVideo

        sampled = SampledVideo(
            video_path=Path("fake.mp4"),
            frames_rgb=np.zeros((32, 4, 4, 3), dtype=np.uint8),
            original_indices=tuple(range(32)),
            total_original_frames=32,
            fps=15.0,
            width=4,
            height=4,
        )
        solved = SimpleV2Orchestrator(
            qwen=qwen, backend=_FakeBackend(), sampled_video=sampled
        ).solve({"id": "endpoint_displacement", "question": "How far did it move?"})
        self.assertEqual(solved["final_answer"]["value"], 1.0)
        self.assertEqual(replay_tool_trace(solved["trace"])["replayed_math_calls"], 1)
        self.assertEqual(replay_tool_trace(solved["trace"])["status"], "complete")
        # The host policy appears in the artifact, but in no message shown to Qwen.
        self.assertEqual(solved["point_mode"], "ensemble5")
        self.assertTrue(all("point_mode" not in snapshot for snapshot in qwen.message_snapshots))
        self.assertIsNotNone(qwen.initial_content_layout)
        layout = qwen.initial_content_layout
        self.assertEqual(len(layout), 65)
        for index in range(32):
            self.assertEqual(layout[2 * index], {
                "type": "text", "text": f"Sampled frame {index}:"
            })
            self.assertEqual(layout[2 * index + 1], {
                "type": "image", "size": [4, 4]
            })
        self.assertIn("Ground bbox_2d_1000", layout[-1]["text"])

        merged = merge_d4rt_results([
            solved["evidence"]["d4rt_1"], solved["evidence"]["d4rt_2"]
        ])
        self.assertEqual(measure_d4rt_result("endpoint_displacement", merged, aligned=True), 1.0)

    def test_binding_resolution_rejects_literal_numbers(self) -> None:
        with self.assertRaises(ValueError):
            resolve_bindings({"x": 3.0}, {"d4rt_1": {"x": 3.0}})

    def test_path_math_recovers_after_separate_binding_hint(self) -> None:
        frames = list(range(32))
        qwen = _ScriptedQwen([
            json.dumps({
                "action": "query_d4rt",
                "arguments": {
                    "label": "object",
                    "bbox_2d_1000": [400, 400, 500, 500],
                    "t_src": 0,
                    "t_tgt": frames,
                    "t_cam": 0,
                    "justification": "Need the full visible trajectory.",
                },
            }),
            json.dumps({
                "action": "python_math",
                "arguments": {
                    "bindings": {
                        "track": {
                            "evidence_id": "d4rt_1",
                            "path": ["math_trajectory_aligned_xyz_m"],
                        }
                    },
                    "code": "value = path_length(track, track['math_visibility'])",
                    "justification": "Compute the visible path.",
                },
            }),
            json.dumps({
                "action": "python_math",
                "arguments": {
                    "bindings": {
                        "track": {
                            "evidence_id": "d4rt_1",
                            "path": ["math_trajectory_aligned_xyz_m"],
                        },
                        "visible": {
                            "evidence_id": "d4rt_1",
                            "path": ["math_visibility"],
                        },
                    },
                    "code": "value = path_length(track, visible)",
                    "justification": "Compute the visible path with separate numeric bindings.",
                },
            }),
            json.dumps({
                "action": "final_answer",
                "arguments": {
                    "value": 1.0,
                    "unit": "m",
                    "evidence_ids": ["d4rt_1", "math_1"],
                    "limitations": "Sparse point evidence.",
                },
            }),
        ])
        from d4rt_agent.simple_v2_contracts import SampledVideo

        sampled = SampledVideo(
            video_path=Path("fake.mp4"),
            frames_rgb=np.zeros((32, 4, 4, 3), dtype=np.uint8),
            original_indices=tuple(frames),
            total_original_frames=32,
            fps=15.0,
            width=4,
            height=4,
        )
        solved = SimpleV2Orchestrator(
            qwen=qwen, backend=_FakeBackend(), sampled_video=sampled, max_steps=4
        ).solve({"id": "distance_travelled", "question": "How far did it travel?"})
        self.assertEqual(solved["final_answer"]["value"], 1.0)
        self.assertEqual(solved["trace"][1]["status"], "rejected")
        self.assertIn("bind each evidence field", solved["trace"][1]["error"])
        self.assertTrue(any("bind each evidence field" in item for item in qwen.message_snapshots))


class AggregateTest(unittest.TestCase):
    def _artifact(self, mode: str, value: float) -> dict:
        scores = []
        questions = []
        for task_id, gt_value in (("endpoint_displacement", 0.5), ("distance_travelled", 3.0)):
            scores.append({
                "task_id": task_id,
                "benchmark_aligned_value_m": value,
                "gt_value_m": gt_value,
                "absolute_error_m": abs(value - gt_value),
                "visibility_coverage": 1.0,
                "grounding": {"centroid_pixel_uv": [1.0, 2.0]},
            })
            questions.append({"id": task_id, "trace": []})
        return {
            "status": "complete",
            "point_mode": mode,
            "sampling": {"sampled_to_original": list(range(32))},
            "configuration": {"seed": 42, "qwen_decoding": {"do_sample": False}},
            "qwen": {"model": "qwen"},
            "d4rt": {
                "model_config": "model.yaml",
                "checkpoint": "model.ckpt",
                "device": "cuda",
                "dtype": "bfloat16",
            },
            "gpu": {"name": "fake"},
            "scores": scores,
            "questions": questions,
        }

    def test_two_mode_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            centroid = root / "centroid.json"
            ensemble = root / "ensemble.json"
            output = root / "aggregate.json"
            report = root / "report.md"
            centroid.write_text(json.dumps(self._artifact("centroid", 1.0)))
            ensemble.write_text(json.dumps(self._artifact("ensemble5", 2.0)))
            result = aggregate_files(centroid, ensemble, output, report)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["comparison"]), 2)
            self.assertTrue(output.exists())
            self.assertIn("Centroid D4RT", report.read_text())

    def test_unscored_answer_does_not_break_aggregation(self) -> None:
        """An open-ended question rides along without measurement diagnostics."""

        def with_unscored(mode: str, value: float) -> dict:
            artifact = self._artifact(mode, value)
            artifact["scores"].append({
                "task_id": "person_motion_description",
                "answer_kind": "text",
                "scored": False,
                "unscored_reason": "answer kind 'text' has no automated scoring",
                "agent_final_text": "The person walked toward the camera.",
            })
            artifact["questions"].append({"id": "person_motion_description", "trace": []})
            return artifact

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            centroid = root / "centroid.json"
            ensemble = root / "ensemble.json"
            centroid.write_text(json.dumps(with_unscored("centroid", 1.0)))
            ensemble.write_text(json.dumps(with_unscored("ensemble5", 2.0)))
            result = aggregate_files(
                centroid, ensemble, root / "aggregate.json", root / "report.md"
            )
            # The scored comparison is unchanged by the extra question.
            self.assertEqual(len(result["comparison"]), 2)
            diagnostics = result["mode_diagnostics"]["centroid"]
            self.assertNotIn("person_motion_description", diagnostics["visibility"])
            self.assertEqual(
                [item["task_id"] for item in diagnostics["unscored"]],
                ["person_motion_description"],
            )
            self.assertEqual(
                diagnostics["unscored"][0]["answer"],
                "The person walked toward the camera.",
            )

    def test_incomplete_artifact_is_rejected_before_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            centroid = root / "centroid.json"
            ensemble = root / "ensemble.json"
            incomplete = self._artifact("centroid", 1.0)
            incomplete["status"] = "failed"
            centroid.write_text(json.dumps(incomplete))
            ensemble.write_text(json.dumps(self._artifact("ensemble5", 2.0)))
            with self.assertRaisesRegex(ValueError, "centroid artifact is incomplete"):
                aggregate_files(
                    centroid,
                    ensemble,
                    root / "aggregate.json",
                    root / "report.md",
                )


if __name__ == "__main__":
    unittest.main()
