"""CPU-only tests for the DSI-Bench LLM judge.

No network, no model, no API key.  The request builders, the defensive result
parser, and the scorer are all pure functions, so the whole judge pipeline can
be checked before a single batch is submitted.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from d4rt_agent.dsi_bench_judge import (
    SENTINEL_NO_ANSWER,
    SENTINEL_UNMAPPABLE,
    SYSTEM_PROMPT_PATH,
    batch_line,
    build_score_report,
    extract_choice,
    format_options,
    judge_messages,
    response_format,
    _result_content,
    _select_complete,
)


def _record(qid="internet_c1_x", status="complete", text="C: Basically unchange. Measured 0.01.",
            gt="C", letters=("A", "B", "C", "D"), cate=1) -> dict:
    options = {L: f"option {L}" for L in letters}
    options["C"] = "Basically unchange"
    rec = {
        "question_id": qid,
        "status": status,
        "dataset": "internet",
        "cate": cate,
        "question": "Where does the car move?",
        "options": options,
        "option_letters": list(letters),
        "gt": gt,
        "d4rt_used": True,
    }
    if status == "complete":
        rec["final_answer"] = {"kind": "text", "text": text, "evidence_ids": ["d4rt_1"]}
    else:
        rec["error"] = "Qwen did not produce a valid final answer in 14 steps"
    return rec


class RequestBuildingTest(unittest.TestCase):
    def test_options_render_in_letter_order(self) -> None:
        rec = _record()
        self.assertEqual(
            format_options(rec),
            "A: option A\nB: option B\nC: Basically unchange\nD: option D",
        )

    def test_messages_never_leak_ground_truth(self) -> None:
        rec = _record(gt="C")
        messages = judge_messages(rec, "SYSTEM")
        joined = json.dumps(messages)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(rec["question"], messages[1]["content"])
        self.assertIn("Basically unchange", messages[1]["content"])
        # The gt letter as a token must not be handed to the judge as the answer key.
        self.assertNotIn('"gt"', joined)
        self.assertNotIn("ground truth", joined.lower())

    def test_enum_is_this_questions_letters_plus_sentinel(self) -> None:
        rf = response_format(["A", "B", "C"])
        enum = rf["json_schema"]["schema"]["properties"]["choice"]["enum"]
        self.assertEqual(enum, ["A", "B", "C", SENTINEL_UNMAPPABLE])
        self.assertTrue(rf["json_schema"]["strict"])
        self.assertFalse(rf["json_schema"]["schema"]["additionalProperties"])

    def test_batch_line_shape(self) -> None:
        line = batch_line(_record(qid="q1"), model="m", system_prompt="S", max_completion_tokens=40)
        self.assertEqual(line["custom_id"], "q1")
        self.assertEqual(line["method"], "POST")
        self.assertEqual(line["url"], "/v1/chat/completions")
        self.assertEqual(line["body"]["model"], "m")
        self.assertEqual(line["body"]["max_completion_tokens"], 40)
        self.assertIn("response_format", line["body"])

    def test_select_complete_filters_and_orders(self) -> None:
        answers = {
            "b": _record(qid="b"),
            "a": _record(qid="a"),
            "f": _record(qid="f", status="failed"),
        }
        picked = _select_complete(answers, limit=None, only=None)
        self.assertEqual([r["question_id"] for r in picked], ["a", "b"])  # sorted, no failed
        self.assertEqual(_select_complete(answers, limit=1, only=None)[0]["question_id"], "a")
        self.assertEqual(
            [r["question_id"] for r in _select_complete(answers, limit=None, only=["b"])], ["b"]
        )


class ExtractChoiceTest(unittest.TestCase):
    LETTERS = ["A", "B", "C", "D"]

    def test_structured_json_choice(self) -> None:
        choice, rationale = extract_choice('{"choice":"C","rationale":"opens with C"}', self.LETTERS)
        self.assertEqual(choice, "C")
        self.assertEqual(rationale, "opens with C")

    def test_structured_unmappable(self) -> None:
        choice, _ = extract_choice('{"choice":"UNMAPPABLE","rationale":"hedged"}', self.LETTERS)
        self.assertEqual(choice, SENTINEL_UNMAPPABLE)

    def test_out_of_enum_letter_becomes_unmappable(self) -> None:
        """A model that ignores strict mode and returns 'E' must not invent an option."""

        choice, _ = extract_choice('{"choice":"E","rationale":"?"}', self.LETTERS)
        self.assertEqual(choice, SENTINEL_UNMAPPABLE)

    def test_bare_letter_fallback(self) -> None:
        # Not JSON at all -- salvage via parse_choice_letter.
        choice, _ = extract_choice("B", self.LETTERS)
        self.assertEqual(choice, "B")

    def test_unmappable_token_in_free_text(self) -> None:
        choice, _ = extract_choice("I think this is UNMAPPABLE", self.LETTERS)
        self.assertEqual(choice, SENTINEL_UNMAPPABLE)

    def test_empty_content_is_unmappable(self) -> None:
        self.assertEqual(extract_choice("", self.LETTERS)[0], SENTINEL_UNMAPPABLE)
        self.assertEqual(extract_choice(None, self.LETTERS)[0], SENTINEL_UNMAPPABLE)


class ResultContentTest(unittest.TestCase):
    def test_success_line(self) -> None:
        line = {"custom_id": "q", "error": None, "response": {"status_code": 200,
                "body": {"choices": [{"message": {"content": "hi"}}]}}}
        content, error = _result_content(line)
        self.assertEqual(content, "hi")
        self.assertIsNone(error)

    def test_request_level_error(self) -> None:
        content, error = _result_content({"custom_id": "q", "error": {"message": "boom"}})
        self.assertIsNone(content)
        self.assertIn("request_error", error)

    def test_non_200(self) -> None:
        content, error = _result_content({"error": None, "response": {"status_code": 400, "body": {}}})
        self.assertIsNone(content)
        self.assertIn("http_400", error)


class SystemPromptTest(unittest.TestCase):
    def test_prompt_states_the_hard_constraints(self) -> None:
        text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").lower()
        self.assertIn(SENTINEL_UNMAPPABLE.lower(), text)
        self.assertIn("outside knowledge", text)
        self.assertIn("guess", text)
        # It must tell the judge not to grade correctness.
        self.assertTrue("correct" in text and "not" in text)


class ScoreReportTest(unittest.TestCase):
    """The scorer's arithmetic and grouping, on a tiny hand-built results dir."""

    def _build(self, tmp: Path) -> None:
        answers_dir = tmp / "answers"
        answers_dir.mkdir(parents=True)
        judge_dir = tmp / "judge"
        judge_dir.mkdir(parents=True)

        # 4 questions: one correct, one wrong-but-mapped, one unmappable, one failed.
        recs = [
            _record(qid="internet_c1_correct", text="C: Basically unchange.", gt="C", cate=1),
            _record(qid="internet_c1_wrong", text="A: option A.", gt="C", cate=1),
            _record(qid="internet_c2_hedge", text="It is unclear.", gt="B", cate=2),
            _record(qid="internet_c2_fail", status="failed", gt="D", cate=2),
        ]
        for r in recs:
            (answers_dir / f"{r['question_id']}.json").write_text(json.dumps(r))

        questions = [
            {"question_id": r["question_id"], "gt": r["gt"], "cate": r["cate"],
             "dataset": r["dataset"], "option_letters": r["option_letters"]}
            for r in recs
        ]
        (tmp / "manifest.json").write_text(json.dumps({"questions": questions}))

        judgments = {
            "internet_c1_correct": {"choice": "C", "source": "gpt"},
            "internet_c1_wrong": {"choice": "A", "source": "gpt"},
            "internet_c2_hedge": {"choice": SENTINEL_UNMAPPABLE, "source": "gpt"},
            "internet_c2_fail": {"choice": SENTINEL_NO_ANSWER, "source": "auto_failed"},
        }
        (judge_dir / "judgments.json").write_text(json.dumps(judgments))

    def test_report_counts_and_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            self._build(tmp)
            report = build_score_report(tmp)

        # Overall: n=4, mapped=2 (C,A), 1 unmappable, 1 no_answer, correct=1.
        # acc_all = 1/4 = 25.0%, acc_mapped = 1/2 = 50.0%, map_rate = 2/4 = 50.0%.
        self.assertIn("| all questions | 4 | 2 | 1 | 1 | 1 | 25.0% | 50.0% | 50.0% |", report)
        # Task split: c1 has the two mapped (1 correct); c2 has hedge+fail (0 mapped).
        self.assertIn("c1 · Obj:moving cam", report)
        self.assertIn("c2 · Cam:static scene", report)
        # Judge validation line present.
        self.assertIn("Judge validation", report)


if __name__ == "__main__":
    unittest.main()
