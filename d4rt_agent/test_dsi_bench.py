"""CPU tests for the DSI-Bench evaluation harness.

Everything here runs without a GPU, a model, or the D4RT checkpoint, so the
sampling design and the answer rules can be checked before any job is submitted.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from d4rt_agent.dsi_bench_data import (
    CATEGORY_NAMES,
    DATASETS,
    NUM_CATEGORIES,
    QUESTIONS_PER_DATASET,
    TOTAL_QUESTIONS,
    build_manifest,
    find_balanced_seed,
    format_question_prompt,
    gt_histogram,
    latin_rectangle_cells,
    load_rows,
    metadata_csv,
    parse_choice_letter,
    parse_options,
    sample_rows,
    video_slug,
)
from d4rt_agent.dsi_bench_run import DSIOrchestrator, SYSTEM_PROMPT_PATH, _task_for
from d4rt_agent.simple_v2_contracts import restricted_python_math, validate_action


CSV_PATH = metadata_csv()
BENCHMARK_AVAILABLE = CSV_PATH.exists()


class OptionParsingTest(unittest.TestCase):
    def test_letters_and_bodies_are_split(self) -> None:
        options = parse_options("A: Moving upward; B: Moving backward; C: Moving right; D: Moving downward")
        self.assertEqual(options, {
            "A": "Moving upward",
            "B": "Moving backward",
            "C": "Moving right",
            "D": "Moving downward",
        })

    def test_trailing_semicolon_does_not_create_a_fifth_option(self) -> None:
        """Three rows of std.csv end with ';' and would parse as a blank option."""

        options = parse_options("A: Get farther; B: Get closer; C: Remain unchanged; D: Cannot be determined;")
        self.assertEqual(sorted(options), ["A", "B", "C", "D"])

    def test_malformed_and_duplicate_options_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unparsable option"):
            parse_options("A: Left; not an option")
        with self.assertRaisesRegex(ValueError, "duplicate option letter"):
            parse_options("A: Left; A: Right")


class SlugTest(unittest.TestCase):
    def test_slug_is_safe_for_paths_and_urls(self) -> None:
        slug = video_slug("k700/land sailing/AbC_000012_000022/video-scene-00001.mp4")
        self.assertNotIn(" ", slug)
        self.assertNotIn("/", slug)
        self.assertEqual(slug, slug.lower())

    def test_distinct_paths_get_distinct_slugs(self) -> None:
        first = video_slug("internet/a/video-scene-00001.mp4")
        second = video_slug("internet/b/video-scene-00001.mp4")
        self.assertNotEqual(first, second)


class ChoiceLetterTest(unittest.TestCase):
    """Recovering the control's answer from however it chose to phrase it.

    Job 8071005 answered all 25 questions as '<B>' rather than the documented
    '<answer>B</answer>', so a parser matching only the documented tag recorded 25
    non-answers from 25 real ones.
    """

    def test_documented_tag(self) -> None:
        self.assertEqual(parse_choice_letter("<answer>B</answer>", "ABCD"), "B")

    def test_the_abbreviation_the_model_actually_emits(self) -> None:
        self.assertEqual(parse_choice_letter("<B>", "ABCD"), "B")
        self.assertEqual(parse_choice_letter("<c>", "ABCD"), "C")
        # Qwen also emits a half-closed tag; two of 25 replies looked like this.
        self.assertEqual(parse_choice_letter("<B</answer>", "ABCD"), "B")

    def test_the_answer_tag_itself_is_not_mistaken_for_a_letter(self) -> None:
        """'<answer>' begins with '<a', which a lax pattern reads as choice A."""

        self.assertEqual(parse_choice_letter("<answer>C</answer>", "ABCD"), "C")

    def test_bare_and_prefixed_letters(self) -> None:
        self.assertEqual(parse_choice_letter("B", "ABCD"), "B")
        self.assertEqual(parse_choice_letter("**D**", "ABCD"), "D")
        self.assertEqual(parse_choice_letter("C. Moving forward", "ABCD"), "C")
        self.assertEqual(parse_choice_letter("The answer is A", "ABCD"), "A")

    def test_a_letter_outside_the_options_is_not_an_answer(self) -> None:
        self.assertIsNone(parse_choice_letter("<E>", "ABCD"))

    def test_unreadable_replies_return_none(self) -> None:
        self.assertIsNone(parse_choice_letter("", "ABCD"))
        self.assertIsNone(parse_choice_letter("I cannot tell from this video.", "ABCD"))

    def test_the_documented_tag_wins_over_stray_letters(self) -> None:
        self.assertEqual(
            parse_choice_letter("Maybe C, but <answer>B</answer>", "ABCD"), "B"
        )


class LatinRectangleTest(unittest.TestCase):
    def test_shape_is_balanced_across_datasets_and_tasks(self) -> None:
        cells = latin_rectangle_cells()
        self.assertEqual(len(cells), TOTAL_QUESTIONS)
        self.assertEqual(len(set(cells)), TOTAL_QUESTIONS)
        per_dataset: dict[str, int] = {}
        per_category: dict[int, int] = {}
        for dataset, cate in cells:
            per_dataset[dataset] = per_dataset.get(dataset, 0) + 1
            per_category[cate] = per_category.get(cate, 0) + 1
        self.assertEqual(set(per_dataset), set(DATASETS))
        self.assertEqual(set(per_dataset.values()), {QUESTIONS_PER_DATASET})
        self.assertEqual(sorted(per_category), list(range(NUM_CATEGORIES)))
        # 25 questions over 6 tasks is as even as [4,4,4,4,5,4].
        self.assertEqual(sorted(per_category.values()), [4, 4, 4, 4, 4, 5])


@unittest.skipUnless(BENCHMARK_AVAILABLE, f"DSI-Bench not present at {CSV_PATH}")
class RealBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_rows(CSV_PATH)

    def test_every_row_has_four_options_containing_its_ground_truth(self) -> None:
        self.assertEqual(len(self.rows), 1769)
        for row in self.rows:
            self.assertEqual(len(row.options), 4, row.relative_path)
            self.assertIn(row.gt, row.options, row.relative_path)

    def test_every_selected_cell_has_candidates(self) -> None:
        for dataset, cate in latin_rectangle_cells():
            pool = [r for r in self.rows if r.dataset == dataset and r.cate == cate]
            self.assertTrue(pool, f"empty cell ({dataset}, cate={cate})")

    def test_sampling_is_deterministic_and_uses_distinct_videos(self) -> None:
        first = sample_rows(self.rows, seed=20260721)
        second = sample_rows(self.rows, seed=20260721)
        self.assertEqual([r.question_id for r in first], [r.question_id for r in second])
        self.assertEqual(len(first), TOTAL_QUESTIONS)
        self.assertEqual(len({r.relative_path for r in first}), TOTAL_QUESTIONS)

    def test_a_different_seed_selects_a_different_sample(self) -> None:
        first = sample_rows(self.rows, seed=20260721)
        second = sample_rows(self.rows, seed=20260722)
        self.assertNotEqual([r.question_id for r in first], [r.question_id for r in second])

    def test_balanced_seed_evens_out_the_answer_key(self) -> None:
        _, selected = find_balanced_seed(self.rows)
        histogram = gt_histogram(selected)
        self.assertEqual(sorted(histogram), ["A", "B", "C", "D"])
        self.assertTrue(all(5 <= count <= 8 for count in histogram.values()), histogram)

    def test_manifest_hides_the_answer_key_from_the_model(self) -> None:
        manifest = build_manifest(seed=20260721)
        entry = manifest["questions"][0]
        task = _task_for(entry)
        # The prompt may legitimately contain option text; it must not contain the
        # category, which would hand the model its reasoning recipe.
        self.assertNotIn(str(entry["cate"]), task["question"].split("Options:")[0])
        for leak in (entry["category_name"], entry["dataset"], entry["others"] or "\0"):
            self.assertNotIn(leak, task["question"])
        self.assertIn(entry["question"], task["question"])
        for letter in entry["option_letters"]:
            self.assertIn(entry["options"][letter], task["question"])


class QuestionPromptTest(unittest.TestCase):
    def test_options_are_rendered_in_letter_order(self) -> None:
        entry = {
            "question": "How did it move?",
            "options": {"B": "Left", "A": "Right"},
            "option_letters": ["A", "B"],
        }
        rendered = format_question_prompt(entry)
        self.assertLess(rendered.index("A: Right"), rendered.index("B: Left"))


class _StubOrchestrator(DSIOrchestrator):
    """DSIOrchestrator with the model and backend left out.

    ``_validate_final`` touches neither, so the answer rules can be tested without
    loading a 14 GB checkpoint.
    """

    def __init__(self, require_d4rt: bool = True, max_steps: int = 14) -> None:
        self.require_d4rt = require_d4rt
        self.max_steps = max_steps


class StepDeadlineTest(unittest.TestCase):
    """The host must tell the model when its budget is nearly spent.

    Job 8070124 lost three of its first four questions to agents that kept
    measuring -- scanning t_tgt one frame at a time -- until the loop was
    abandoned. Remaining steps are host knowledge, so only the host can say it.
    """

    def test_no_notice_while_there_is_room_to_work(self) -> None:
        orchestrator = _StubOrchestrator(max_steps=14)
        self.assertIsNone(orchestrator._step_notice(1))
        self.assertIsNone(orchestrator._step_notice(10))

    def test_notice_appears_and_counts_down_near_the_limit(self) -> None:
        orchestrator = _StubOrchestrator(max_steps=14)
        self.assertIn("3 step(s) left", orchestrator._step_notice(11))
        self.assertIn("1 step(s) left", orchestrator._step_notice(13))

    def test_notice_tells_the_model_to_answer_rather_than_measure(self) -> None:
        notice = _StubOrchestrator(max_steps=14)._step_notice(13)
        self.assertIn("final_answer", notice)
        self.assertIn("limitations", notice)

    def test_no_notice_once_the_budget_is_gone(self) -> None:
        self.assertIsNone(_StubOrchestrator(max_steps=14)._step_notice(14))

    def test_base_orchestrator_stays_silent(self) -> None:
        """The metric pipeline must see the conversation it always saw."""

        from d4rt_agent.simple_v2 import SimpleV2Orchestrator

        base = SimpleV2Orchestrator.__new__(SimpleV2Orchestrator)
        base.max_steps = 12
        self.assertIsNone(base._step_notice(11))


class FinalAnswerRuleTest(unittest.TestCase):
    TASK = {"id": "internet_c4_example"}
    EVIDENCE = {"d4rt_1": {"t_tgt": [0]}, "math_1": {"outputs": {}}}

    def test_a_text_answer_citing_d4rt_is_accepted(self) -> None:
        _StubOrchestrator()._validate_final(
            self.TASK,
            {"kind": "text", "text": "B: Get closer.", "evidence_ids": ["d4rt_1", "math_1"]},
            self.EVIDENCE,
        )

    def test_an_answer_with_no_d4rt_call_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one query_d4rt call"):
            _StubOrchestrator()._validate_final(
                self.TASK,
                {"kind": "text", "text": "B: Get closer.", "evidence_ids": ["math_1"]},
                self.EVIDENCE,
            )

    def test_unknown_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            _StubOrchestrator()._validate_final(
                self.TASK,
                {"kind": "text", "text": "B", "evidence_ids": ["d4rt_9"]},
                self.EVIDENCE,
            )

    def test_a_numeric_answer_is_rejected_for_multiple_choice(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple choice"):
            _StubOrchestrator()._validate_final(
                self.TASK,
                {"kind": "numeric", "value": 1.0, "unit": "m", "evidence_ids": ["d4rt_1"]},
                self.EVIDENCE,
            )

    def test_the_relaxed_retry_drops_only_the_d4rt_requirement(self) -> None:
        relaxed = _StubOrchestrator(require_d4rt=False)
        relaxed._validate_final(
            self.TASK, {"kind": "text", "text": "B", "evidence_ids": []}, self.EVIDENCE
        )
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            relaxed._validate_final(
                self.TASK, {"kind": "text", "text": "B", "evidence_ids": ["d4rt_9"]}, self.EVIDENCE
            )

    def test_single_target_evidence_is_allowed(self) -> None:
        """Camera-to-object range legitimately queries one frame per viewpoint.

        The metric pipeline's two-target minimum would reject exactly the recipe
        the DSI prompt teaches, so it deliberately does not apply here.
        """

        _StubOrchestrator()._validate_final(
            self.TASK,
            {"kind": "text", "text": "A: Get farther.", "evidence_ids": ["d4rt_1"]},
            {"d4rt_1": {"t_tgt": [0], "t_cam": 0}},
        )


class MathDiagnosisTest(unittest.TestCase):
    """An occluded frame must be reported as occluded, not as bad arithmetic.

    D4RT returns benchmark_aligned_xyz_m: null for a target it did not observe, and
    the calculator rejects it as "non-finite or non-numeric values". Three
    questions in job 8071987 were lost to agents rebinding that same null, one of
    them twelve times, because nothing in the message mentioned visibility.
    """

    EVIDENCE = {
        "d4rt_1": {
            "t_tgt": [0],
            "predictions": [{"benchmark_aligned_xyz_m": [0.1, 0.0, 4.2], "visible": True}],
        },
        "d4rt_2": {
            "t_tgt": [31],
            "predictions": [{"benchmark_aligned_xyz_m": None, "visible": False}],
        },
    }
    BINDINGS = {
        "start": {"evidence_id": "d4rt_1", "path": ["predictions", 0, "benchmark_aligned_xyz_m"]},
        "end": {"evidence_id": "d4rt_2", "path": ["predictions", 0, "benchmark_aligned_xyz_m"]},
    }

    def test_the_invisible_frame_is_named(self) -> None:
        message = _StubOrchestrator()._diagnose_math(
            {"bindings": self.BINDINGS},
            self.EVIDENCE,
            ValueError("math input contains non-finite or non-numeric values"),
        )
        self.assertIn("`end`", message)
        self.assertIn("d4rt_2", message)
        self.assertIn("sampled frame 31", message)
        self.assertIn("visibility", message)
        # The visible binding is not blamed.
        self.assertNotIn("`start`", message)

    def test_the_message_offers_a_next_action(self) -> None:
        message = _StubOrchestrator()._diagnose_math(
            {"bindings": self.BINDINGS}, self.EVIDENCE, ValueError("boom")
        )
        self.assertIn("math_visibility", message)
        self.assertIn("limitations", message)

    def test_unrelated_failures_are_passed_through_unchanged(self) -> None:
        error = ValueError("code exceeds 32 statements")
        message = _StubOrchestrator()._diagnose_math(
            {"bindings": {"a": {"evidence_id": "d4rt_1", "path": ["predictions", 0, "visible"]}}},
            self.EVIDENCE,
            error,
        )
        self.assertEqual(message, str(error))


class HardDeadlineTest(unittest.TestCase):
    """The final step is reserved for the answer.

    Warnings alone did not stop two agents in job 8071987 from issuing fourteen
    queries and never answering.
    """

    def test_tools_are_refused_on_the_last_step(self) -> None:
        orchestrator = _StubOrchestrator(max_steps=14)
        with self.assertRaisesRegex(ValueError, "must be final_answer"):
            orchestrator._execute_action("query_d4rt", {}, {}, step=13)

    def test_tools_are_allowed_while_steps_remain(self) -> None:
        orchestrator = _StubOrchestrator(max_steps=14)
        # Reaches the base implementation, which rejects an unknown action name --
        # proving the deadline did not fire.
        with self.assertRaisesRegex(ValueError, "host cannot execute action"):
            orchestrator._execute_action("not_a_tool", {}, {}, step=5)

    def test_a_missing_step_never_triggers_the_deadline(self) -> None:
        with self.assertRaisesRegex(ValueError, "host cannot execute action"):
            _StubOrchestrator(max_steps=14)._execute_action("not_a_tool", {}, {})


class BlindQueryTest(unittest.TestCase):
    """Querying for an object that has left the shot must be cut off.

    The last two failures of job 8084609 both spent every remaining step hunting
    for a vanished object -- one sweeping t_cam, which cannot affect visibility at
    all -- and ignored both the countdown and the hard deadline.
    """

    @staticmethod
    def _evidence(blind: int, visible: int = 1) -> dict[str, dict[str, object]]:
        evidence: dict[str, dict[str, object]] = {}
        for index in range(visible):
            evidence[f"d4rt_{index + 1}"] = {"t_tgt": [0], "math_visibility": [True]}
        for index in range(blind):
            evidence[f"d4rt_{visible + index + 1}"] = {"t_tgt": [31], "math_visibility": [False]}
        return evidence

    def test_queries_allowed_below_the_threshold(self) -> None:
        orchestrator = _StubOrchestrator()
        orchestrator._refuse_hopeless_query(self._evidence(blind=2))

    def test_query_refused_once_enough_come_back_empty(self) -> None:
        orchestrator = _StubOrchestrator()
        with self.assertRaisesRegex(ValueError, "nothing visible"):
            orchestrator._refuse_hopeless_query(self._evidence(blind=3))

    def test_refusal_supplies_a_template_and_blames_the_right_thing(self) -> None:
        try:
            _StubOrchestrator()._refuse_hopeless_query(self._evidence(blind=4))
        except ValueError as error:
            message = str(error)
        self.assertIn("final_answer", message)
        self.assertIn("limitations", message)
        self.assertIn("t_cam cannot make an unobserved target visible", message)

    def test_partly_visible_results_do_not_count_as_blind(self) -> None:
        evidence = {
            f"d4rt_{index}": {"t_tgt": [0, 31], "math_visibility": [True, False]}
            for index in range(1, 6)
        }
        _StubOrchestrator()._refuse_hopeless_query(evidence)

    def test_the_deadline_message_carries_the_template(self) -> None:
        try:
            _StubOrchestrator(max_steps=14)._execute_action("query_d4rt", {}, {}, step=13)
        except ValueError as error:
            message = str(error)
        self.assertIn('"action":"final_answer"', message)


class SystemPromptTest(unittest.TestCase):
    PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    def test_prompt_never_mentions_the_host_grounding_policy(self) -> None:
        self.assertNotIn("point_mode", self.PROMPT)

    def test_prompt_teaches_the_scalar_versus_vector_frame_rule(self) -> None:
        lowered = self.PROMPT.lower()
        self.assertIn("frame-invariant", lowered)
        self.assertIn("static background", lowered)
        # The metric prompt's blanket ban would forbid the range-change recipe.
        self.assertNotIn("never mix frames", lowered)

    def test_prompt_states_that_positions_are_unscaled(self) -> None:
        self.assertIn("arbitrary unit", self.PROMPT)

    def test_prompt_says_the_viewpoint_cannot_change_visibility(self) -> None:
        """An agent that thinks it can re-view an occluded frame loops until it dies.

        Job 8068823 spent all 10 steps re-asking for one invisible target frame
        under eight different t_cam values.
        """

        self.assertIn("never make an invisible target visible", self.PROMPT)
        self.assertIn("math_visibility", self.PROMPT)

    def test_tool_examples_are_valid_actions_that_measure_before_answering(self) -> None:
        """Every worked example must show the calls it cites.

        An example that jumps straight to an answer is one the model reproduces,
        which is how the metric prompt lost its measurements (job 7985547).
        """

        examples = [
            json.loads(line)
            for line in self.PROMPT.splitlines()
            if line.startswith('{"action":"') and "ACTION_NAME" not in line
        ]
        self.assertEqual(len(examples), 13)
        names = [validate_action(example)[0] for example in examples]
        self.assertEqual(
            names,
            [
                # A: range change, two viewpoints
                "query_d4rt", "query_d4rt", "python_math", "final_answer",
                # B: camera egomotion from a static background point
                "query_d4rt", "query_d4rt", "python_math", "final_answer",
                # C: object motion projected onto its own measured facing axis
                "query_d4rt", "query_d4rt", "query_d4rt", "python_math", "final_answer",
            ],
        )
        for example in examples:
            if validate_action(example)[0] == "final_answer":
                self.assertEqual(example["arguments"]["kind"], "text")
                cited = example["arguments"]["evidence_ids"]
                self.assertTrue(any(item.startswith("d4rt_") for item in cited), cited)

    def test_example_calculations_run_under_the_restricted_evaluator(self) -> None:
        values = {
            "start": [1.0, 0.5, 8.0],
            "end": [0.6, 0.4, 4.3],
            "spread": [0.03, 0.02, 0.04],
            "front": [0.6, 0.3, 4.0],
            "rear": [0.2, 0.3, 4.4],
        }
        for line in self.PROMPT.splitlines():
            if not line.startswith('{"action":"python_math"'):
                continue
            action = json.loads(line)
            bindings = {name: values[name] for name in action["arguments"]["bindings"]}
            outputs = restricted_python_math(bindings, action["arguments"]["code"])
            self.assertTrue(outputs)


if __name__ == "__main__":
    unittest.main()
