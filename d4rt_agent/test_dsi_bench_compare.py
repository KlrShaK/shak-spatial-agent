"""CPU tests for reproducible DSI result comparison."""

from __future__ import annotations

import unittest

from d4rt_agent.dsi_bench_compare import summarize


class CompareTest(unittest.TestCase):
    @staticmethod
    def _manifest():
        return {
            "questions": [
                {
                    "question_id": question_id,
                    "gt": gt,
                    "option_letters": ["A", "B"],
                    "dataset": "Synthetic",
                    "category_name": "Obj:static cam",
                }
                for question_id, gt in (
                    ("q1", "A"),
                    ("q2", "B"),
                    ("q3", "A"),
                    ("q4", "B"),
                )
            ]
        }

    @staticmethod
    def _answer(letter, status="complete", trace=None):
        value = {
            "status": status,
            "wall_seconds": 2.0,
            "final_answer": (
                {"kind": "text", "text": f"{letter}: option", "evidence_ids": []}
                if letter
                else None
            ),
            "trace": trace or [],
            "steps": len(trace or []),
        }
        return value

    def test_completed_failed_relaxed_and_unparseable_are_counted(self) -> None:
        trace = [
            {
                "status": "ok",
                "parsed_action": {"action": "ground_with_qwen"},
                "call_id": "qg_1",
                "cache_hit": False,
                "discarded_suffix": "",
                "result": {"mode": "bbox", "status": "ok"},
            },
            {
                "status": "ok",
                "parsed_action": {"action": "query_d4rt"},
                "result": {"grounding_mode": "bbox", "visibility_coverage": 0.75},
                "discarded_suffix": "",
            },
            {
                "status": "rejected",
                "parsed_action": {
                    "action": "ground_with_qwen",
                    "arguments": {"mode": "bbox"},
                },
                "grounder_failure": {
                    "raw_grounder_response": "bad",
                    "host_provenance": {},
                },
                "discarded_suffix": "",
            },
        ]
        current = {
            "q1": self._answer("A", trace=trace),
            "q2": self._answer("B", status="complete_relaxed"),
            "q3": self._answer(None, status="failed"),
            "q4": self._answer("A"),
        }
        previous = {key: self._answer("A") for key in current}
        failed = {key: self._answer(None, status="failed") for key in current}
        baseline = {
            "q1": {"status": "complete", "raw_response": "<A>", "parsed_letter": None},
            "q2": {"status": "complete", "raw_response": "<A>", "parsed_letter": None},
            "q3": {
                "status": "complete",
                "raw_response": "cannot determine",
                "parsed_letter": None,
            },
            "q4": {
                "status": "complete",
                "raw_response": "<answer>B</answer>",
                "parsed_letter": None,
            },
        }

        result = summarize(
            manifest=self._manifest(),
            current=current,
            previous=previous,
            failed=failed,
            baseline=baseline,
        )

        self.assertEqual(result["current"]["correct"], 2)
        self.assertEqual(result["current"]["total"], 4)
        self.assertEqual(result["completion"]["strict"], 2)
        self.assertEqual(result["completion"]["relaxed"], 1)
        self.assertEqual(result["completion"]["failed"], 1)
        self.assertEqual(result["completion"]["strict_rate"], 0.5)
        self.assertEqual(result["grounding"]["qg_calls"], 2)
        self.assertEqual(result["grounding"]["qg_inferences"], 2)
        self.assertEqual(result["grounding"]["malformed"], 1)
        self.assertEqual(result["grounding"]["rejected"], 1)
        self.assertEqual(result["visibility_coverage_by_mode"]["bbox"]["mean"], 0.75)
        self.assertEqual(result["current"]["accuracy_among_completed"], 2 / 3)
        self.assertEqual(result["per_task"], {
            "Obj:static cam": {
                "correct": 2,
                "total": 4,
                "accuracy": 0.5,
                "statuses": {
                    "complete": 2,
                    "complete_relaxed": 1,
                    "failed": 1,
                },
            }
        })

    def test_unparseable_completed_answer_is_incorrect(self) -> None:
        current = {
            key: self._answer(None, status="complete")
            for key in ("q1", "q2", "q3", "q4")
        }
        baseline = {
            key: {"status": "complete", "raw_response": "unparseable"}
            for key in current
        }
        result = summarize(
            manifest=self._manifest(),
            current=current,
            previous=current,
            failed=current,
            baseline=baseline,
        )
        self.assertEqual(result["current"]["correct"], 0)
        self.assertEqual(result["current"]["parsed_answers"], 0)

    def test_failed_record_with_stale_final_text_is_incorrect(self) -> None:
        current = {
            key: self._answer("A", status="failed")
            for key in ("q1", "q2", "q3", "q4")
        }
        baseline = {
            key: {"status": "complete", "raw_response": "unparseable"}
            for key in current
        }
        result = summarize(
            manifest=self._manifest(),
            current=current,
            previous=current,
            failed=current,
            baseline=baseline,
        )
        self.assertEqual(result["current"]["correct"], 0)
        self.assertEqual(result["current"]["parsed_answers"], 0)


if __name__ == "__main__":
    unittest.main()
