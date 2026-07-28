from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from .qwen_grounding_debug import (
    box_iou,
    center_distance_1000,
    extract_requests,
    parse_detection,
)


class GroundingDebugTest(unittest.TestCase):
    def test_parse_detection_strict_json_inside_fence(self) -> None:
        self.assertEqual(
            parse_detection('```json\n{"bbox_2d_1000":[10,20,30,40]}\n```'),
            [10.0, 20.0, 30.0, 40.0],
        )
        self.assertIsNone(parse_detection('{"bbox_2d_1000":null}'))
        with self.assertRaises(ValueError):
            parse_detection('{"bbox_2d_1000":[30,20,10,40]}')

    def test_box_metrics(self) -> None:
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [10, 10, 20, 20]), 0.0)
        self.assertEqual(center_distance_1000([0, 0, 10, 10], [10, 0, 20, 10]), 10.0)

    def test_extracts_ok_and_rejected_requests_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answers = Path(directory)
            payload = {
                "question_id": "q1",
                "dataset": "test",
                "category_name": "object",
                "status": "complete",
                "sampling": {
                    "video": "/tmp/example.mp4",
                    "sampled_to_original": [
                        {"sampled_frame": 0, "original_frame": 3},
                    ],
                },
                "trace": [
                    {
                        "step": 1,
                        "status": "ok",
                        "parsed_action": {
                            "action": "query_d4rt",
                            "arguments": {
                                "label": "car",
                                "t_src": 0,
                                "bbox_2d_1000": [10, 20, 30, 40],
                            },
                        },
                        "result": {
                            "predictions": [
                                {"sampled_frame_index": 0, "visible": True},
                            ],
                        },
                    },
                    {
                        "step": 2,
                        "status": "rejected",
                        "parsed_action": {
                            "action": "query_d4rt",
                            "arguments": {
                                "label": "car",
                                "t_src": 0,
                                "bbox_2d_1000": [10, 20, 30, 40],
                            },
                        },
                    },
                ],
            }
            (answers / "q1.json").write_text(json.dumps(payload), encoding="utf-8")
            occurrences, probes = extract_requests(answers)
            self.assertEqual(len(occurrences), 2)
            self.assertEqual(len(probes), 1)
            self.assertEqual(probes[0]["occurrence_ids"], ["q1:step_01", "q1:step_02"])
            self.assertTrue(occurrences[0]["source_frame_visible"])
            self.assertIsNone(occurrences[1]["source_frame_visible"])


if __name__ == "__main__":
    unittest.main()
