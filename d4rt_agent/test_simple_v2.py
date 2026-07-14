"""CPU contract and orchestration tests for simple v2."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from d4rt_agent.simple_v2 import (
    SimpleV2Orchestrator,
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
)


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
        with self.assertRaisesRegex(ValueError, "justification"):
            validate_action({
                "action": "inspect_frames",
                "arguments": {"frame_indices": [0], "justification": ""},
            })


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
                "points": [[0, 0, 0], [1, 0, 0], [9, 0, 0], [11, 0, 0]],
                "visible": [True, True, False, True],
            },
            "endpoint = dist(points[0], points[-1])\ntravel = path_length(points, visible)",
        )
        self.assertEqual(result["endpoint"], 11.0)
        self.assertEqual(result["travel"], 1.0)

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


class _ScriptedQwen:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.message_snapshots: list[str] = []

    def generate(self, messages):
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
                    "t_tgt": [0, 31],
                    "t_cam": 0,
                    "justification": "Need the two endpoint positions.",
                },
            }),
            json.dumps({
                "action": "python_math",
                "arguments": {
                    "bindings": {
                        "start": {"evidence_id": "d4rt_1", "path": ["predictions", 0, "benchmark_aligned_xyz_m"]},
                        "end": {"evidence_id": "d4rt_1", "path": ["predictions", 1, "benchmark_aligned_xyz_m"]},
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
                    "evidence_ids": ["d4rt_1", "math_1"],
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
        # The host policy appears in the artifact, but in no message shown to Qwen.
        self.assertEqual(solved["point_mode"], "ensemble5")
        self.assertTrue(all("point_mode" not in snapshot for snapshot in qwen.message_snapshots))

    def test_binding_resolution_rejects_literal_numbers(self) -> None:
        with self.assertRaises(ValueError):
            resolve_bindings({"x": 3.0}, {"d4rt_1": {"x": 3.0}})


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


if __name__ == "__main__":
    unittest.main()
