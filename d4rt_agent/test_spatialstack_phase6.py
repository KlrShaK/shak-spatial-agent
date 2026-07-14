"""CPU tests for the isolated Phase 6 SpatialStack helper."""

import unittest

from d4rt_agent.spatialstack_phase6 import parse_metric_answer, sample_indices


class SpatialStackPhase6Test(unittest.TestCase):
    def test_frame_sampling_matches_spatialstack_default(self) -> None:
        self.assertEqual(sample_indices(64, 32), list(range(0, 62, 2)) + [63])

    def test_metric_parser_ignores_frame_numbers(self) -> None:
        self.assertEqual(
            parse_metric_answer("The distance from Frame-0 to Frame-31 is 0.75 meters."),
            0.75,
        )

    def test_metric_parser_rejects_ambiguous_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_metric_answer("It may be 0.5 m or 0.8 m.")


if __name__ == "__main__":
    unittest.main()
