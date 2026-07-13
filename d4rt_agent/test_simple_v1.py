"""CPU tests for the simple D4RT tool-use entry point."""

from pathlib import Path
import unittest

from d4rt_agent.geometry_tools import DemoGeometry
from d4rt_agent.simple_v1 import DEFAULT_DEMO, _parse_planner_json, run_deterministic


class BasketballDeterministicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (DEFAULT_DEMO / "assets" / "demo_data.json").exists():
            raise unittest.SkipTest("basketball demo bundle is not available")

    def test_known_results_and_visibility(self) -> None:
        result = run_deterministic(DEFAULT_DEMO, (346.0, 166.0, 0))
        by_id = {item["id"]: item for item in result["questions"]}
        self.assertAlmostEqual(
            by_id["endpoint_displacement"]["predicted"]["endpoint_displacement_m"],
            0.3243,
            places=4,
        )
        path = by_id["distance_travelled"]["predicted"]
        self.assertAlmostEqual(path["path_length_m"], 3.6049, places=4)
        self.assertEqual(path["visible_segments"], [[0, 33], [56, 63]])
        self.assertEqual(path["visible_frames"], 42)

    def test_path_does_not_bridge_visibility_gaps(self) -> None:
        geometry = DemoGeometry(DEFAULT_DEMO, source="pred")
        measured = geometry.path_measurement(5, 0, 63)
        segment_sum = geometry.path_length(5, 0, 33) + geometry.path_length(5, 56, 63)
        self.assertAlmostEqual(measured["path_length_m"], segment_sum, places=4)

    def test_qwen_planner_json_parser(self) -> None:
        plan = _parse_planner_json(
            '```json\n{"tool":"path_length","point_2d":[541,461],"frame":0}\n```'
        )
        self.assertEqual(plan["tool"], "path_length")
        self.assertEqual(plan["point_2d"], [541.0, 461.0])

    def test_qwen_planner_json_rejects_unknown_tool(self) -> None:
        with self.assertRaises(ValueError):
            _parse_planner_json('{"tool":"distance","point_2d":[500,500],"frame":0}')


if __name__ == "__main__":
    unittest.main()
