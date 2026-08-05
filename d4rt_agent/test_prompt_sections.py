"""Tests for section-addressable prompts.

The point of these is attribution: an ablation flag that silently did nothing
would be recorded as an ablation and quietly corrupt a comparison, so the
failure modes here are all "refuse loudly" rather than "carry on".
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from d4rt_agent.dsi_bench_run import SYSTEM_PROMPT_PATH
from d4rt_agent.prompt_sections import (
    load_prompt,
    parse_override_args,
    render_prompt,
    section_names,
)


SAMPLE = """<!-- section: alpha -->
# Alpha

Alpha body.
<!-- endsection -->

<!-- section: beta -->
## Beta

Beta body.
<!-- endsection -->
"""


class RenderTest(unittest.TestCase):
    def test_names_are_returned_in_document_order(self) -> None:
        self.assertEqual(section_names(SAMPLE), ("alpha", "beta"))

    def test_rendering_without_ablations_strips_every_marker(self) -> None:
        rendered = render_prompt(SAMPLE)
        self.assertNotIn("<!--", rendered)
        self.assertIn("Alpha body.", rendered)
        self.assertIn("Beta body.", rendered)

    def test_excluding_a_section_removes_only_that_section(self) -> None:
        rendered = render_prompt(SAMPLE, exclude=("alpha",))
        self.assertNotIn("Alpha body.", rendered)
        self.assertIn("Beta body.", rendered)

    def test_overriding_replaces_the_body(self) -> None:
        rendered = render_prompt(SAMPLE, override={"beta": "## Beta\n\nReplaced."})
        self.assertIn("Replaced.", rendered)
        self.assertNotIn("Beta body.", rendered)
        self.assertIn("Alpha body.", rendered)

    def test_unknown_section_raises_rather_than_silently_doing_nothing(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown prompt section"):
            render_prompt(SAMPLE, exclude=("gamma",))
        with self.assertRaisesRegex(ValueError, "unknown prompt section"):
            render_prompt(SAMPLE, override={"gamma": "x"})

    def test_excluding_and_overriding_the_same_section_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "both excluded and overridden"):
            render_prompt(SAMPLE, exclude=("alpha",), override={"alpha": "x"})

    def test_duplicate_names_raise(self) -> None:
        doubled = SAMPLE + SAMPLE
        with self.assertRaisesRegex(ValueError, "duplicate prompt section"):
            render_prompt(doubled)

    def test_unbalanced_marker_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unbalanced prompt section marker"):
            render_prompt("<!-- section: alpha -->\nbody, never closed\n")


class OverrideArgTest(unittest.TestCase):
    def test_parses_name_equals_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body.md"
            path.write_text("replacement", encoding="utf-8")
            parsed = parse_override_args([f"alpha={path}"])
            self.assertEqual(parsed, {"alpha": path})

    def test_missing_file_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            parse_override_args(["alpha=/nonexistent/body.md"])

    def test_malformed_argument_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "expects name=path"):
            parse_override_args(["alpha"])


class LivePromptTest(unittest.TestCase):
    """The shipped DSI prompt must actually be addressable."""

    def test_dsi_prompt_declares_the_expected_sections(self) -> None:
        names = section_names(SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            names,
            (
                "role",
                "method",
                # Contracts sits third: step 3 of the method names its fields, and
                # it holds the most-violated content (the binding shape), which
                # does not belong in the middle of the prompt.
                "contracts",
                "coordinate_frame",
                "decision_tree",
                "troubleshooting",
                "example",
            ),
        )

    def test_rendered_prompt_carries_no_markers_and_shrank(self) -> None:
        rendered = load_prompt(SYSTEM_PROMPT_PATH)
        self.assertNotIn("<!--", rendered)
        # The archived forced-grounding prompt was 28.4k characters and its
        # instruction load is what the restructure exists to reduce.
        self.assertLess(len(rendered), 18_000)

    def test_every_section_can_be_dropped_independently(self) -> None:
        text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        full = load_prompt(SYSTEM_PROMPT_PATH)
        for name in section_names(text):
            ablated = load_prompt(SYSTEM_PROMPT_PATH, exclude=(name,))
            self.assertLess(len(ablated), len(full), name)
            self.assertNotIn("<!--", ablated)


if __name__ == "__main__":
    unittest.main()
