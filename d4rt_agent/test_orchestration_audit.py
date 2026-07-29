"""CPU-only tests for orchestration trace auditing and rendering."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from d4rt_agent.dsi_bench_show import render_trace, render_trace_markdown
from d4rt_agent.orchestration_audit import (
    action_objects,
    audit_answers_dir,
    audit_records,
    count_action_objects,
    main,
)


def _state(
    available: list[dict[str, object]],
    *,
    accepted: bool,
    new_id: str | None,
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "available": available,
        "last_action": "accepted" if accepted else "rejected",
        "new_evidence_id": new_id,
        "reason": reason,
    }


def _row(
    *,
    step: int,
    action: dict[str, object] | None,
    status: str,
    available: list[dict[str, object]],
    call_id: str | None = None,
    result: object = None,
    error: str | None = None,
    failure_stage: str | None = None,
    raw: str | None = None,
    effective: str | None = None,
    suffix: str = "",
) -> dict[str, object]:
    if raw is None:
        raw = json.dumps(action, separators=(",", ":")) if action is not None else "No action."
    if effective is None:
        effective = raw.removesuffix(suffix) if suffix else raw
    return {
        "step": step,
        "kind": "model_action",
        "raw_qwen_response": raw,
        "effective_response": effective,
        "discarded_suffix": suffix,
        "action_span": (
            {"start": effective.find("{"), "end": len(effective)}
            if action is not None else None
        ),
        "parsed_action": action,
        "status": status,
        "failure_stage": failure_stage,
        "call_id": call_id,
        "result": result,
        "error": error,
        "evidence_state": _state(
            available,
            accepted=status == "ok",
            new_id=call_id if status == "ok" and call_id and not call_id.startswith("final_") else None,
            reason=error,
        ),
    }


def _valid_record(*, suffix: str = "") -> dict[str, object]:
    query = {
        "action": "query_d4rt",
        "arguments": {
            "label": "runner",
            "bbox_2d_1000": [100, 100, 300, 500],
            "t_src": 0,
            "t_tgt": [0, 31],
            "t_cam": 0,
            "justification": "Measure motion.",
        },
    }
    query_effective = "I will measure the runner.\n" + json.dumps(
        query, separators=(",", ":")
    )
    query_raw = query_effective + suffix
    entry = {
        "evidence_id": "d4rt_1",
        "tool_name": "query_d4rt",
        "created_at_step": 1,
    }
    final = {
        "action": "final_answer",
        "arguments": {
            "kind": "text",
            "text": "A: left",
            "evidence_ids": ["d4rt_1"],
            "limitations": "",
        },
    }
    result = {
        "question_id": "q1",
        "status": "complete",
        "trace": [
            _row(
                step=1,
                action=query,
                status="ok",
                available=[entry],
                call_id="d4rt_1",
                result={"t_tgt": [0, 31]},
                raw=query_raw,
                effective=query_effective,
                suffix=suffix,
            ),
            _row(
                step=2,
                action=final,
                status="ok",
                available=[entry],
                call_id="final_1",
                result=final["arguments"],
            ),
        ],
        "evidence": {"d4rt_1": {"t_tgt": [0, 31]}},
    }
    return result


class ActionScanningTest(unittest.TestCase):
    def test_scans_actions_in_order_without_counting_plain_json(self) -> None:
        text = (
            '{"note":{"nested":true}}\n'
            '{"action":"query_d4rt","arguments":{"x":{"y":"}"}}}\n'
            'future {"action":"python_math","arguments":{}}'
        )
        self.assertEqual(
            [item["action"] for item in action_objects(text)],
            ["query_d4rt", "python_math"],
        )
        self.assertEqual(count_action_objects(text), 2)

    def test_malformed_prefix_does_not_hide_later_action(self) -> None:
        text = '{bad json\n{"action":"final_answer","arguments":{}}'
        self.assertEqual(count_action_objects(text), 1)

    def test_nested_name_field_is_not_a_second_action(self) -> None:
        text = json.dumps({
            "action": "query_d4rt",
            "arguments": {
                "metadata": {"name": "runner"},
                "justification": "Nested data is part of one action.",
            },
        })
        self.assertEqual(count_action_objects(text), 1)


class AuditTest(unittest.TestCase):
    def test_valid_trace_passes_and_counts_discarded_future_action(self) -> None:
        suffix = (
            '\nTool result d4rt_1: invented\n'
            '{"action":"python_math","arguments":{"bindings":{}}}'
        )
        result = audit_records([_valid_record(suffix=suffix)])
        self.assertEqual(result["status"], "pass")
        summary = result["summary"]
        self.assertEqual(summary["strict_attempts"], 1)
        self.assertEqual(summary["relaxed_attempts"], 0)
        self.assertEqual(summary["total_model_turns"], 2)
        self.assertEqual(summary["turns_with_discarded_suffix"], 1)
        self.assertEqual(summary["discarded_suffixes_with_another_action"], 1)
        self.assertEqual(summary["evidence_ids_created"], 1)
        self.assertEqual(summary["successful_actions"]["query_d4rt"], 1)

    def test_ledger_mismatch_is_structural_failure(self) -> None:
        record = _valid_record()
        record["trace"][0]["evidence_state"]["available"] = []
        result = audit_records([record])
        self.assertEqual(result["status"], "fail")
        codes = {item["code"] for item in result["violations"]}
        self.assertIn("ledger_registry_mismatch", codes)

    def test_accepted_unknown_evidence_is_structural_failure(self) -> None:
        record = _valid_record()
        record["trace"][1]["parsed_action"]["arguments"]["evidence_ids"] = [
            "d4rt_1", "math_9"
        ]
        result = audit_records([record])
        self.assertEqual(result["status"], "fail")
        violation = next(
            item for item in result["violations"]
            if item["code"] == "accepted_unknown_evidence"
        )
        self.assertIn("math_9", violation["message"])

    def test_multiple_actions_in_effective_history_fail(self) -> None:
        record = _valid_record()
        extra = '\n{"action":"python_math","arguments":{}}'
        record["trace"][0]["effective_response"] += extra
        result = audit_records([record])
        self.assertEqual(result["summary"]["effective_turns_with_multiple_actions"], 1)
        self.assertIn(
            "multiple_effective_actions",
            {item["code"] for item in result["violations"]},
        )

    def test_raw_boundary_tampering_is_structural_failure(self) -> None:
        record = _valid_record()
        row = record["trace"][0]
        row["raw_qwen_response"] += (
            '\n{"action":"python_math","arguments":{"bindings":{}}}'
        )
        result = audit_records([record])
        self.assertEqual(result["status"], "fail")
        self.assertIn(
            "boundary_reconstruction_mismatch",
            {item["code"] for item in result["violations"]},
        )

    def test_boundary_must_select_first_raw_action(self) -> None:
        record = _valid_record()
        row = record["trace"][0]
        first = row["raw_qwen_response"]
        second = json.dumps(row["parsed_action"], separators=(",", ":"))
        row["raw_qwen_response"] = first + "\n" + second
        start = len(first) + 1
        row["action_span"] = {"start": start, "end": start + len(second)}
        row["effective_response"] = row["raw_qwen_response"][: start + len(second)]
        row["discarded_suffix"] = ""
        result = audit_records([record])
        self.assertEqual(result["status"], "fail")
        self.assertIn(
            "first_action_boundary_mismatch",
            {item["code"] for item in result["violations"]},
        )

    def test_parsed_arguments_must_match_raw_action(self) -> None:
        record = _valid_record()
        record["trace"][0]["parsed_action"]["arguments"]["label"] = "different object"
        result = audit_records([record])
        self.assertEqual(result["status"], "fail")
        self.assertIn(
            "parsed_action_payload_mismatch",
            {item["code"] for item in result["violations"]},
        )

    def test_no_action_boundary_cannot_carry_parsed_action(self) -> None:
        record = _valid_record()
        row = record["trace"][0]
        row["raw_qwen_response"] = "plain prose"
        row["effective_response"] = "plain prose"
        row["discarded_suffix"] = ""
        row["action_span"] = None
        result = audit_records([record])
        self.assertEqual(result["status"], "fail")
        self.assertIn(
            "parsed_action_without_boundary",
            {item["code"] for item in result["violations"]},
        )

    def test_unknown_evidence_rejection_is_counted_without_creating_id(self) -> None:
        record = _valid_record()
        rejected_action = {
            "action": "python_math",
            "arguments": {
                "bindings": {"x": {"evidence_id": "d4rt_9", "path": []}},
                "code": "y = norm(x)",
                "justification": "Calculate.",
            },
        }
        entry = record["trace"][0]["evidence_state"]["available"][0]
        rejected = _row(
            step=2,
            action=rejected_action,
            status="rejected",
            available=[entry],
            error=(
                "Evidence ID(s) ['d4rt_9'] do not exist. "
                "Available evidence IDs: ['d4rt_1']."
            ),
            failure_stage="execution",
        )
        record["trace"].insert(1, rejected)
        record["trace"][2]["step"] = 3
        result = audit_records([record])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["summary"]["unknown_evidence_rejections"], 1)
        self.assertEqual(result["summary"]["evidence_ids_created"], 1)

    def test_strict_and_relaxed_attempts_are_counted_separately(self) -> None:
        strict = _valid_record()
        relaxed = _valid_record()
        relaxed["question_id"] = "q2"
        relaxed["status"] = "complete_relaxed"
        relaxed["gated_attempt"] = {
            "error": "strict exhausted",
            "trace": strict["trace"],
            "evidence": strict["evidence"],
        }
        result = audit_records([relaxed])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["summary"]["strict_attempts"], 1)
        self.assertEqual(result["summary"]["relaxed_attempts"], 1)
        self.assertEqual(result["summary"]["evidence_ids_created"], 2)

    def test_legacy_trace_is_readable_without_phase1_ledger(self) -> None:
        legacy = {
            "question_id": "old",
            "status": "complete",
            "trace": [{
                "step": 1,
                "status": "ok",
                "call_id": "d4rt_1",
                "parsed_action": {
                    "action": "query_d4rt",
                    "arguments": {},
                },
                "raw_qwen_response": '{"action":"query_d4rt","arguments":{}}',
            }],
            "evidence": {"d4rt_1": {}},
        }
        result = audit_records([legacy])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["summary"]["legacy_attempts"], 1)

    def test_directory_and_cli_write_machine_readable_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers = root / "answers"
            answers.mkdir()
            (answers / "q1.json").write_text(
                json.dumps(_valid_record()), encoding="utf-8"
            )
            direct = audit_answers_dir(answers)
            self.assertEqual(direct["status"], "pass")
            output = root / "audit.json"
            self.assertEqual(
                main(["--answers-dir", str(answers), "--output", str(output)]),
                0,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "pass")
            self.assertEqual(saved["summary"]["records"], 1)

    def test_empty_answers_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = audit_answers_dir(Path(directory))
        self.assertEqual(result["status"], "fail")
        self.assertIn(
            "no_answer_records",
            {item["code"] for item in result["violations"]},
        )


class TraceRenderingCompatibilityTest(unittest.TestCase):
    def test_normal_render_uses_effective_response_and_diagnoses_raw_suffix(self) -> None:
        suffix = (
            '\nTool result d4rt_1: <fabricated & unsafe>\n'
            '{"action":"python_math","arguments":{}}'
        )
        record = _valid_record(suffix=suffix)
        markdown = "\n".join(render_trace_markdown(record))
        self.assertIn("I will measure the runner.", markdown)
        self.assertIn("Discarded suffix", markdown)
        self.assertIn("&lt;fabricated &amp; unsafe&gt;", markdown)
        normal_text = "\n".join(render_trace(record, raw=False))
        self.assertNotIn("<fabricated & unsafe>", normal_text)
        raw_text = "\n".join(render_trace(record, raw=True))
        self.assertIn("<fabricated & unsafe>", raw_text)

    def test_old_trace_without_effective_fields_still_renders(self) -> None:
        record = {
            "trace": [{
                "step": 1,
                "status": "rejected",
                "raw_qwen_response": "I forgot the action.",
                "error": "no action",
            }],
            "evidence": {},
        }
        self.assertIn("I forgot the action.", "\n".join(render_trace_markdown(record)))

    def test_parse_rejection_shows_complete_effective_malformed_action(self) -> None:
        effective = 'Thinking first.\n{"action": broken payload'
        record = {
            "trace": [{
                "step": 1,
                "status": "rejected",
                "failure_stage": "parse",
                "raw_qwen_response": effective,
                "effective_response": effective,
                "discarded_suffix": "",
                "action_span": None,
                "parsed_action": None,
                "call_id": None,
                "result": None,
                "error": "Qwen did not emit one JSON action",
                "evidence_state": _state(
                    [], accepted=False, new_id=None, reason="parse failure"
                ),
            }],
            "evidence": {},
        }
        markdown = "\n".join(render_trace_markdown(record))
        self.assertIn("Thinking first.", markdown)
        self.assertIn('{"action": broken payload', markdown)


if __name__ == "__main__":
    unittest.main()
