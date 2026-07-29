"""Audit saved agent traces for single-action and evidence-ledger invariants.

This module is deliberately CPU-only.  It reads answer JSON files, checks the
host/model boundary recorded by Phase 1, and writes one machine-readable
summary.  It never imports or loads Qwen, D4RT, Torch, or Transformers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def action_objects(text: str) -> list[dict[str, Any]]:
    """Return complete JSON action objects in textual order.

    Scanning every opening brace deliberately mirrors the live extractor.  An
    outer JSON object can contain nested braces, so offsets consumed by a prior
    decode are not used to skip future candidates.
    """

    decoder = json.JSONDecoder()
    found: list[tuple[int, dict[str, Any]]] = []
    for match in re.finditer(r"\{", str(text)):
        try:
            value, _ = decoder.raw_decode(str(text)[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ("action" in value or "name" in value):
            found.append((match.start(), value))
    return [value for _, value in sorted(found, key=lambda item: item[0])]


def count_action_objects(text: str) -> int:
    """Count complete action objects in text without executing them."""

    return len(action_objects(text))


def iter_attempts(record: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield strict and relaxed attempt payloads in execution order."""

    gated = record.get("gated_attempt")
    if isinstance(gated, Mapping):
        yield "strict", gated
        yield "relaxed", record
    else:
        yield "strict", record


def _action_name(row: Mapping[str, Any]) -> str:
    action = row.get("parsed_action")
    if isinstance(action, Mapping):
        name = action.get("action", action.get("name"))
        if isinstance(name, str) and name:
            return name
    return "unparsed"


def _referenced_evidence(row: Mapping[str, Any]) -> set[str]:
    action = row.get("parsed_action")
    if not isinstance(action, Mapping):
        return set()
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        return set()
    name = action.get("action", action.get("name"))
    if name == "python_math":
        bindings = arguments.get("bindings")
        if not isinstance(bindings, Mapping):
            return set()
        return {
            specification["evidence_id"]
            for specification in bindings.values()
            if isinstance(specification, Mapping)
            and isinstance(specification.get("evidence_id"), str)
        }
    if name == "final_answer":
        evidence_ids = arguments.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            return set()
        return {value for value in evidence_ids if isinstance(value, str)}
    return set()


def _ledger_entries(row: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    state = row.get("evidence_state")
    if not isinstance(state, Mapping):
        return None
    available = state.get("available")
    if not isinstance(available, list):
        return None
    entries: list[dict[str, Any]] = []
    for item in available:
        if not isinstance(item, Mapping):
            return None
        evidence_id = item.get("evidence_id")
        tool_name = item.get("tool_name")
        created_at_step = item.get("created_at_step")
        if (
            not isinstance(evidence_id, str)
            or not isinstance(tool_name, str)
            or not isinstance(created_at_step, int)
            or isinstance(created_at_step, bool)
        ):
            return None
        entries.append({
            "evidence_id": evidence_id,
            "tool_name": tool_name,
            "created_at_step": created_at_step,
        })
    return entries


def _violation(
    violations: list[dict[str, Any]],
    *,
    question_id: str,
    attempt: str,
    code: str,
    message: str,
    step: int | None = None,
) -> None:
    item: dict[str, Any] = {
        "question_id": question_id,
        "attempt": attempt,
        "code": code,
        "message": message,
    }
    if step is not None:
        item["step"] = step
    violations.append(item)


def _audit_attempt(
    question_id: str,
    attempt_name: str,
    payload: Mapping[str, Any],
    metrics: dict[str, Any],
    violations: list[dict[str, Any]],
) -> dict[str, Any]:
    trace = payload.get("trace")
    evidence = payload.get("evidence")
    if not isinstance(trace, list):
        _violation(
            violations,
            question_id=question_id,
            attempt=attempt_name,
            code="invalid_trace",
            message="attempt trace is missing or is not a list",
        )
        trace = []
    if not isinstance(evidence, Mapping):
        _violation(
            violations,
            question_id=question_id,
            attempt=attempt_name,
            code="invalid_registry",
            message="attempt evidence registry is missing or is not an object",
        )
        evidence = {}

    # Old committed artifacts predate Phase 1.  They remain readable and useful
    # for comparisons, but do not claim to carry ledger/effective-history data.
    phase1_rows = [
        row for row in trace
        if isinstance(row, Mapping)
        and ("effective_response" in row or "evidence_state" in row)
    ]
    is_phase1_trace = bool(phase1_rows)
    if trace and not is_phase1_trace:
        metrics["legacy_attempts"] += 1

    expected_entries: list[dict[str, Any]] = []
    prior_ids: set[str] = set()
    attempt_created: list[str] = []
    attempt_max_step = 0

    for position, raw_row in enumerate(trace, start=1):
        if not isinstance(raw_row, Mapping):
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                code="invalid_trace_row",
                message=f"trace row {position} is not an object",
            )
            continue
        row = raw_row
        step_value = row.get("step", position)
        step = (
            step_value
            if isinstance(step_value, int) and not isinstance(step_value, bool)
            else position
        )
        attempt_max_step = max(attempt_max_step, step)
        metrics["total_model_turns"] += 1
        name = _action_name(row)
        status = str(row.get("status", ""))
        if status == "ok":
            metrics["successful_actions"][name] += 1
        elif status == "rejected":
            metrics["rejected_actions"][name] += 1
            stage = row.get("failure_stage")
            metrics["rejections_by_failure_stage"][
                str(stage) if stage is not None else "unspecified"
            ] += 1
            error = str(row.get("error", ""))
            if (
                "evidence" in error.casefold()
                and (
                    "unknown" in error.casefold()
                    or "do not exist" in error.casefold()
                    or "does not exist" in error.casefold()
                )
            ):
                metrics["unknown_evidence_rejections"] += 1

        suffix = str(row.get("discarded_suffix", ""))
        if suffix.strip():
            metrics["turns_with_discarded_suffix"] += 1
            if count_action_objects(suffix):
                metrics["discarded_suffixes_with_another_action"] += 1

        if not is_phase1_trace:
            continue

        required_fields = (
            "raw_qwen_response",
            "effective_response",
            "discarded_suffix",
            "action_span",
            "parsed_action",
            "failure_stage",
            "call_id",
            "result",
            "error",
            "evidence_state",
        )
        missing = [field for field in required_fields if field not in row]
        if missing:
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="missing_trace_fields",
                message=f"Phase 1 trace row is missing fields: {missing}",
            )

        effective = str(row.get("effective_response", ""))
        effective_actions = count_action_objects(effective)
        if effective_actions > 1:
            metrics["effective_turns_with_multiple_actions"] += 1
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="multiple_effective_actions",
                message=f"effective response contains {effective_actions} action objects",
            )
        if re.search(
            r"^\s*(?:Tool result\b|HOST TURN\b)",
            effective,
            re.IGNORECASE | re.MULTILINE,
        ):
            metrics["effective_turns_simulating_host"] += 1
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="simulated_host_in_effective_history",
                message="effective response contains model-authored host/tool-result text",
            )

        if status == "ok":
            unavailable = sorted(_referenced_evidence(row) - prior_ids)
            if unavailable:
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    step=step,
                    code="accepted_unknown_evidence",
                    message=(
                        f"accepted {name} references unavailable evidence {unavailable}; "
                        f"available before action: {sorted(prior_ids)}"
                    ),
                )

        state = row.get("evidence_state")
        entries = _ledger_entries(row)
        if entries is None:
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="invalid_ledger",
                message="evidence_state.available is missing or malformed",
            )
            entries = []

        if not isinstance(state, Mapping):
            state = {}
        last_action = state.get("last_action")
        expected_last = "accepted" if status == "ok" else "rejected"
        if last_action != expected_last:
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="ledger_action_mismatch",
                message=f"ledger last_action={last_action!r}, expected {expected_last!r}",
            )

        new_id = state.get("new_evidence_id")
        call_id = row.get("call_id")
        evidence_action = status == "ok" and name in {"query_d4rt", "python_math"}
        if evidence_action:
            if not isinstance(call_id, str):
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    step=step,
                    code="missing_call_id",
                    message=f"successful {name} is missing its call_id",
                )
            if new_id != call_id:
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    step=step,
                    code="ledger_new_id_mismatch",
                    message=f"ledger created {new_id!r}, trace call_id is {call_id!r}",
                )
            if isinstance(call_id, str):
                if call_id in prior_ids:
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="duplicate_evidence_id",
                        message=f"evidence ID {call_id!r} was created more than once",
                    )
                else:
                    expected_entries.append({
                        "evidence_id": call_id,
                        "tool_name": name,
                        "created_at_step": step,
                    })
                    prior_ids.add(call_id)
                    attempt_created.append(call_id)
        elif new_id is not None:
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="rejection_created_evidence",
                message=f"non-evidence action reports new evidence {new_id!r}",
            )

        if entries != expected_entries:
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="ledger_registry_mismatch",
                message=(
                    f"ledger available entries {entries!r} do not match "
                    f"host-created entries {expected_entries!r}"
                ),
            )

    metrics["maximum_steps"] = max(metrics["maximum_steps"], attempt_max_step)
    metrics["evidence_ids_created"] += len(attempt_created)

    if is_phase1_trace:
        registry_ids = list(evidence)
        if registry_ids != attempt_created:
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                code="final_registry_mismatch",
                message=(
                    f"final evidence registry IDs {registry_ids!r} do not match "
                    f"trace-created IDs {attempt_created!r}"
                ),
            )
        for prefix in ("d4rt", "math"):
            numbers = [
                int(match.group(1))
                for evidence_id in attempt_created
                if (match := re.fullmatch(rf"{prefix}_(\d+)", evidence_id))
            ]
            if numbers and numbers != list(range(1, len(numbers) + 1)):
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    code="evidence_counter_gap",
                    message=f"{prefix} evidence sequence has a gap: {numbers}",
                )

    return {
        "name": attempt_name,
        "turns": len(trace),
        "legacy_trace": bool(trace and not is_phase1_trace),
        "evidence_ids": list(evidence),
        "maximum_step": attempt_max_step,
    }


def audit_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit already-loaded records and return a JSON-safe report."""

    metrics: dict[str, Any] = {
        "records": len(records),
        "record_statuses": Counter(),
        "strict_attempts": 0,
        "relaxed_attempts": 0,
        "legacy_attempts": 0,
        "total_model_turns": 0,
        "successful_actions": Counter(),
        "rejected_actions": Counter(),
        "rejections_by_failure_stage": Counter(),
        "turns_with_discarded_suffix": 0,
        "discarded_suffixes_with_another_action": 0,
        "effective_turns_with_multiple_actions": 0,
        "effective_turns_simulating_host": 0,
        "evidence_ids_created": 0,
        "unknown_evidence_rejections": 0,
        "maximum_steps": 0,
    }
    violations: list[dict[str, Any]] = []
    record_summaries: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        question_id = str(record.get("question_id", f"record_{index + 1}"))
        metrics["record_statuses"][str(record.get("status", "unknown"))] += 1
        attempts = []
        for attempt_name, payload in iter_attempts(record):
            metrics[f"{attempt_name}_attempts"] += 1
            attempts.append(
                _audit_attempt(question_id, attempt_name, payload, metrics, violations)
            )
        record_summaries.append({
            "question_id": question_id,
            "status": record.get("status"),
            "attempts": attempts,
        })

    serializable_metrics = {
        key: dict(sorted(value.items())) if isinstance(value, Counter) else value
        for key, value in metrics.items()
    }
    return {
        "status": "pass" if not violations else "fail",
        "generated_at": _utc_now(),
        "summary": serializable_metrics,
        "records": record_summaries,
        "violations": violations,
    }


def audit_answers_dir(answers_dir: Path) -> dict[str, Any]:
    """Load every answer JSON in a directory and audit it."""

    answers_dir = Path(answers_dir)
    records: list[Mapping[str, Any]] = []
    load_violations: list[dict[str, Any]] = []
    for path in sorted(answers_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError("top-level JSON value is not an object")
            records.append(value)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            load_violations.append({
                "question_id": path.stem,
                "attempt": "file",
                "code": "record_load_error",
                "message": f"{type(error).__name__}: {error}",
            })
    result = audit_records(records)
    result["answers_dir"] = str(answers_dir.resolve())
    if load_violations:
        result["violations"] = load_violations + result["violations"]
        result["status"] = "fail"
        result["summary"]["record_load_errors"] = len(load_violations)
    else:
        result["summary"]["record_load_errors"] = 0
    return result


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_answers_dir(args.answers_dir)
    _write_atomic(args.output, result)
    summary = result["summary"]
    print(
        f"orchestration audit {result['status']}: "
        f"{summary['records']} record(s), "
        f"{summary['total_model_turns']} turn(s), "
        f"{len(result['violations'])} violation(s)"
    )
    print(f"Wrote: {args.output}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
