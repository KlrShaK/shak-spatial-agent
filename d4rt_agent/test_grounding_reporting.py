"""CPU-only tests for Phase 2 grounding provenance and visual overlays."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from d4rt_agent.dsi_bench_report import (
    _grounding_details,
    build_grounding_sheet,
    build_qwen_grounding_media,
    iter_grounding_evidence,
    parse_args,
)
from d4rt_agent.dsi_bench_show import render_trace, render_trace_markdown


def _step(
    step: int,
    action: str,
    arguments: dict,
    call_id: str,
    *,
    cache_hit: bool = False,
    reused_evidence_id: str | None = None,
) -> dict:
    return {
        "step": step,
        "status": "ok",
        "raw_qwen_response": f'reason\n{{"action":"{action}"}}',
        "effective_response": f'reason\n{{"action":"{action}"}}',
        "discarded_suffix": "",
        "parsed_action": {"action": action, "arguments": arguments},
        "call_id": call_id,
        "cache_hit": cache_hit,
        "reused_evidence_id": reused_evidence_id,
    }


def _bbox(qg_id: str, t_src: int, *, status: str = "ok") -> dict:
    return {
        "grounding_id": qg_id,
        "status": status,
        "mode": "bbox",
        "t_src": t_src,
        "request": "red car <nearest>",
        "bbox_2d_1000": [100, 200, 400, 600] if status == "ok" else None,
        "host_provenance": {
            "original_t_src": 17 + t_src,
            "raw_grounder_response": '{"bbox_2d_1000":[100,200,400,600]} <raw>',
            "cache_hit": False,
        },
    }


def _points(qg_id: str, t_src: int) -> dict:
    return {
        "grounding_id": qg_id,
        "status": "ok",
        "mode": "points",
        "t_src": t_src,
        "request": "runner's chest and back",
        "requested_count": 2,
        "returned_count": 2,
        "points_2d_1000": [
            {"point_id": "p1", "xy": [500, 500], "description": "chest"},
            {"point_id": "p2", "xy": [600, 520], "description": "back"},
        ],
        "host_provenance": {
            "original_t_src": 29,
            "raw_grounder_response": '{"points_2d_1000":[...]}',
        },
    }


def _attempt(groundings: dict[str, dict], *, cached: bool = False) -> dict:
    evidence: dict[str, dict] = dict(groundings)
    trace: list[dict] = []
    step_number = 1
    for qg_id, grounding in groundings.items():
        arguments = {
            "mode": grounding["mode"],
            "request": grounding["request"],
            "t_src": grounding["t_src"],
        }
        if grounding["mode"] == "points":
            arguments["count"] = grounding["requested_count"]
        trace.append(_step(step_number, "ground_with_qwen", arguments, qg_id))
        step_number += 1
    first_id = next(iter(groundings))
    if cached:
        trace.append(
            _step(
                step_number,
                "ground_with_qwen",
                {
                    "mode": groundings[first_id]["mode"],
                    "request": groundings[first_id]["request"],
                    "t_src": groundings[first_id]["t_src"],
                },
                first_id,
                cache_hit=True,
                reused_evidence_id=first_id,
            )
        )
        step_number += 1
    if groundings[first_id]["status"] == "ok":
        d4rt = {
            "grounding_id": first_id,
            "grounding_mode": groundings[first_id]["mode"],
            "point_ids": ["p1", "p2"] if groundings[first_id]["mode"] == "points" else None,
            "predictions": [],
        }
        evidence["d4rt_1"] = d4rt
        trace.append(
            _step(
                step_number,
                "query_d4rt",
                {
                    "grounding_id": first_id,
                    "point_ids": d4rt["point_ids"],
                    "t_tgt": [0, 31],
                    "t_cam": 0,
                },
                "d4rt_1",
            )
        )
    return {"trace": trace, "evidence": evidence}


def _record() -> dict:
    strict = _attempt(
        {"qg_1": _bbox("qg_1", 0), "qg_2": _bbox("qg_2", 1, status="not_found")},
        cached=True,
    )
    relaxed = _attempt({"qg_1": _points("qg_1", 1)})
    return {**relaxed, "gated_attempt": strict}


class GroundingEvidenceTraversalTest(unittest.TestCase):
    def test_namespaces_strict_and_relaxed_registries(self) -> None:
        rows = iter_grounding_evidence(_record())
        self.assertEqual(
            [row["display_id"] for row in rows],
            ["strict/qg_1", "strict/qg_2", "relaxed/qg_1"],
        )

    def test_markdown_exposes_provenance_cache_geometry_and_raw_response(self) -> None:
        record = _record()
        markdown = "\n".join(_grounding_details(record, "question_1", "clip"))
        self.assertIn("`strict/qg_1`", markdown)
        self.assertIn("`relaxed/qg_1`", markdown)
        self.assertIn("cache hit on 1 later call", markdown)
        self.assertIn("strict/d4rt_1", markdown)
        self.assertIn("not_found", markdown)
        self.assertIn("original frame `17`", markdown)
        self.assertIn("Raw grounding-model response", markdown)
        self.assertIn("&lt;raw&gt;", markdown)
        self.assertIn("grounding_frames/question_1_f00.jpg", markdown)
        self.assertIn("grounding_frames/question_1_f01.jpg", markdown)

    def test_trace_renderers_show_new_calls_and_namespaced_provenance(self) -> None:
        attempt = _attempt({"qg_1": _points("qg_1", 1)})
        markdown = "\n".join(render_trace_markdown(attempt, evidence_namespace="strict"))
        terminal = "\n".join(render_trace(attempt, evidence_namespace="strict"))
        for output in (markdown, terminal):
            self.assertIn("ground_with_qwen", output)
            self.assertIn("strict/qg_1", output)
            self.assertIn("strict/d4rt_1", output)
            self.assertIn("p1", output)


class GroundingOverlayTest(unittest.TestCase):
    def test_combines_groundings_by_exact_source_frame(self) -> None:
        frames = np.zeros((2, 300, 400, 3), dtype=np.uint8)
        frames[0, :, :, 0] = 210
        frames[1, :, :, 1] = 190
        record = _record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = build_qwen_grounding_media(
                frames,
                record,
                question_id="question_1",
                frames_directory=root / "grounding_frames",
                sheet_destination=root / "groundings" / "clip.jpg",
            )
            self.assertEqual(
                [path.name for path in paths],
                ["question_1_f00.jpg", "question_1_f01.jpg"],
            )
            self.assertTrue((root / "groundings" / "clip.jpg").is_file())

            import cv2

            frame_zero = cv2.imread(str(paths[0]))
            frame_one = cv2.imread(str(paths[1]))
            self.assertEqual(frame_zero.shape[:2], (300, 400))
            self.assertEqual(frame_one.shape[:2], (300, 400))
            # RGB source red becomes BGR [0,0,210], while source green remains
            # BGR [0,190,0]. This proves each overlay used its requested frame.
            self.assertGreater(int(frame_zero[-10, -10, 2]), 150)
            self.assertGreater(int(frame_one[-10, -10, 1]), 140)

    def test_refresh_rewrites_current_files_and_removes_stale_frames(self) -> None:
        frames = np.zeros((2, 300, 400, 3), dtype=np.uint8)
        record = _record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_dir = root / "grounding_frames"
            sheet = root / "groundings" / "clip.jpg"
            paths = build_qwen_grounding_media(
                frames,
                record,
                question_id="question_1",
                frames_directory=frames_dir,
                sheet_destination=sheet,
            )
            paths[0].write_bytes(b"stale")
            stale = frames_dir / "question_1_f31.jpg"
            stale.write_bytes(b"stale")
            build_qwen_grounding_media(
                frames,
                record,
                question_id="question_1",
                frames_directory=frames_dir,
                sheet_destination=sheet,
                refresh=True,
            )
            self.assertGreater(paths[0].stat().st_size, len(b"stale"))
            self.assertFalse(stale.exists())

    def test_legacy_d4rt_box_sheet_remains_supported(self) -> None:
        frames = np.zeros((1, 300, 400, 3), dtype=np.uint8)
        legacy = {
            "evidence": {
                "d4rt_1": {
                    "t_src": 0,
                    "label": "runner",
                    "bbox_pixel": [10, 20, 100, 200],
                    "query_points": [],
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "legacy.jpg"
            self.assertTrue(build_grounding_sheet(frames, legacy, destination))
            self.assertTrue(destination.is_file())

    def test_refresh_flag_parses_with_skip_media(self) -> None:
        args = parse_args(["--refresh-groundings", "--skip-media"])
        self.assertTrue(args.refresh_groundings)
        self.assertTrue(args.skip_media)


if __name__ == "__main__":
    unittest.main()
