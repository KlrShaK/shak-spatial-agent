"""Contract tests for the model-facing Phase 2 grounding workflows."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from .simple_v2_contracts import validate_action


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PATHS = (
    PROMPT_DIR / "simple_v2_system.md",
    PROMPT_DIR / "dsi_bench_system.md",
)
ACTION_LINE = re.compile(r"^\{\"action\":.*\}$", re.MULTILINE)
ASSISTANT_TURN = re.compile(
    r"ASSISTANT TURN \d+\n(?P<body>.*?)"
    r"--- END ASSISTANT TURN; (?:STOP AND WAIT FOR HOST|TASK COMPLETE) ---",
    re.DOTALL,
)


def _read_prompts() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in PROMPT_PATHS}


def _actions(text: str) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for match in ACTION_LINE.finditer(text):
        try:
            actions.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            # The response-format metavariable uses ``{...}`` deliberately; only
            # concrete example actions form part of this contract.
            continue
    return actions


class GroundingPromptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompts = _read_prompts()

    def test_prompts_teach_exactly_the_four_model_facing_tools(self) -> None:
        expected = {
            "ground_with_qwen",
            "query_d4rt",
            "python_math",
            "final_answer",
        }
        for name, text in self.prompts.items():
            with self.subTest(prompt=name):
                self.assertIn("four supplied schemas", text)
                self.assertEqual(expected, {action["action"] for action in _actions(text)})
                for action in _actions(text):
                    validate_action(action)

    def test_no_legacy_geometry_instructions_or_query_arguments(self) -> None:
        forbidden_query_fields = {"label", "bbox_2d_1000", "t_src", "point_mode"}
        for name, text in self.prompts.items():
            with self.subTest(prompt=name):
                self.assertNotIn("bbox_2d_1000", text)
                self.assertNotIn("three supplied schemas", text)
                for action in _actions(text):
                    if action["action"] != "query_d4rt":
                        continue
                    arguments = action["arguments"]
                    self.assertIsInstance(arguments, dict)
                    self.assertTrue(forbidden_query_fields.isdisjoint(arguments))
                    self.assertIn("grounding_id", arguments)
                    self.assertIn("t_tgt", arguments)
                    self.assertIn("t_cam", arguments)

    def test_every_example_assistant_turn_contains_one_action(self) -> None:
        for name, text in self.prompts.items():
            turns = list(ASSISTANT_TURN.finditer(text))
            with self.subTest(prompt=name):
                self.assertGreaterEqual(len(turns), 10)
                for turn in turns:
                    self.assertEqual(
                        1,
                        len(ACTION_LINE.findall(turn.group("body"))),
                        msg=turn.group("body"),
                    )

    def test_bbox_points_and_recovery_rules_are_explicit(self) -> None:
        required_fragments = (
            'mode="bbox"',
            'mode="points"',
            "whole visible object",
            "named parts",
            "static",
            "rigid",
            "not_found",
            "fewer points than requested",
            "never invent",
            "immutable `grounding_id`",
            "separate physical",
        )
        for name, text in self.prompts.items():
            with self.subTest(prompt=name):
                normalized = text.lower()
                for fragment in required_fragments:
                    self.assertIn(fragment.lower(), normalized)

    def test_complete_bbox_workflow_composes_all_four_tools(self) -> None:
        for name, text in self.prompts.items():
            bbox_example = text.split("### Example A", 1)[1].split("### Example B", 1)[0]
            names = [action["action"] for action in _actions(bbox_example)]
            with self.subTest(prompt=name):
                self.assertEqual("ground_with_qwen", names[0])
                self.assertIn("query_d4rt", names[1:-2])
                self.assertEqual(["python_math", "final_answer"], names[-2:])
                first = _actions(bbox_example)[0]["arguments"]
                self.assertEqual("bbox", first["mode"])
                self.assertNotIn("count", first)
                query = next(
                    action for action in _actions(bbox_example)
                    if action["action"] == "query_d4rt"
                )
                self.assertEqual("qg_1", query["arguments"]["grounding_id"])

    def test_static_background_workflow_preserves_point_tracks(self) -> None:
        for name, text in self.prompts.items():
            static_example = text.split("### Example B", 1)[1].split(
                "### Example C", 1
            )[0]
            actions = _actions(static_example)
            with self.subTest(prompt=name):
                grounding = next(
                    action for action in actions
                    if action["action"] == "ground_with_qwen"
                )
                self.assertEqual("points", grounding["arguments"]["mode"])
                self.assertGreaterEqual(grounding["arguments"]["count"], 2)
                queries = [
                    action for action in actions if action["action"] == "query_d4rt"
                ]
                self.assertTrue(queries)
                self.assertTrue(all(query["arguments"].get("point_ids") for query in queries))
                self.assertIn("point_tracks", static_example)
                self.assertIn("independent", static_example)

    def test_recovery_example_waits_then_issues_a_new_grounding_action(self) -> None:
        for name, text in self.prompts.items():
            recovery = text.split("### Example C", 1)[-1]
            if name == "dsi_bench_system.md":
                recovery = text.split("#### Recovery pattern", 1)[1]
            actions = _actions(recovery)
            grounding_actions = [
                action for action in actions
                if action["action"] == "ground_with_qwen"
            ]
            with self.subTest(prompt=name):
                self.assertIn('status="not_found"', recovery)
                self.assertGreaterEqual(len(grounding_actions), 2)
                self.assertNotEqual(
                    grounding_actions[0]["arguments"]["t_src"],
                    grounding_actions[1]["arguments"]["t_src"],
                )
                self.assertIn("STOP AND WAIT FOR HOST", recovery)


if __name__ == "__main__":
    unittest.main()
