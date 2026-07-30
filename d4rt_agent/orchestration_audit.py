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
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

try:
    from .simple_v2_contracts import validate_action
except ImportError:  # Support ``python d4rt_agent/orchestration_audit.py``.
    from simple_v2_contracts import validate_action


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_action(
    text: str,
) -> tuple[dict[str, Any], int, int] | None:
    """Return the first action and exact source span using the live scan rule."""

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", str(text)):
        try:
            value, consumed = decoder.raw_decode(str(text)[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ("action" in value or "name" in value):
            return value, match.start(), match.start() + consumed
    return None


def action_objects(text: str) -> list[dict[str, Any]]:
    """Return complete JSON action objects in textual order.

    Once an action object is decoded, its entire span is consumed so nested
    objects with ordinary ``name`` fields are not mistaken for second actions.
    Non-action and malformed objects advance one brace so a later action can
    still be found, matching the live extractor's recovery behavior.
    """

    decoder = json.JSONDecoder()
    source = str(text)
    found: list[dict[str, Any]] = []
    offset = 0
    while match := re.search(r"\{", source[offset:]):
        start = offset + match.start()
        try:
            value, consumed = decoder.raw_decode(source[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict) and ("action" in value or "name" in value):
            found.append(value)
            offset = start + consumed
        else:
            offset = start + 1
    return found


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


def _normalize_legacy_bbox_query(action: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize pre-Phase-2 query_d4rt actions for archived-trace audits."""

    name = action.get("action", action.get("name"))
    arguments = action.get("arguments")
    if name != "query_d4rt" or not isinstance(arguments, Mapping):
        return None
    if "bbox_2d_1000" not in arguments or "grounding_id" in arguments:
        return None
    args = dict(arguments)
    try:
        args["bbox_2d_1000"] = [float(value) for value in args["bbox_2d_1000"]]
        args["t_src"] = int(args["t_src"])
        args["t_tgt"] = [int(value) for value in args["t_tgt"]]
        args["t_cam"] = int(args["t_cam"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"action": "query_d4rt", "arguments": args}


def _referenced_evidence(row: Mapping[str, Any]) -> set[str]:
    action = row.get("parsed_action")
    if not isinstance(action, Mapping):
        return set()
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        return set()
    name = action.get("action", action.get("name"))
    if name == "query_d4rt":
        grounding_id = arguments.get("grounding_id")
        return {grounding_id} if isinstance(grounding_id, str) else set()
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


def _grounding_result_problems(
    grounding_id: str,
    arguments: Mapping[str, Any],
    result: Any,
) -> list[str]:
    """Validate the immutable qg payload saved by a successful grounding call."""

    if not isinstance(result, Mapping):
        return ["successful grounding result is not an object"]
    problems: list[str] = []
    mode = arguments.get("mode")
    status = result.get("status")
    if result.get("grounding_id") != grounding_id:
        problems.append(
            f"grounding_id={result.get('grounding_id')!r}, expected {grounding_id!r}"
        )
    if result.get("mode") != mode:
        problems.append(
            f"grounding mode={result.get('mode')!r}, action requested {mode!r}"
        )
    for field in ("t_src", "request"):
        if result.get(field) != arguments.get(field):
            problems.append(
                f"grounding {field}={result.get(field)!r}, action requested "
                f"{arguments.get(field)!r}"
            )
    if status not in {"ok", "not_found"}:
        problems.append(f"grounding status is invalid: {status!r}")

    provenance = result.get("host_provenance")
    if not isinstance(provenance, Mapping):
        problems.append("grounding host_provenance is missing")
    else:
        for field in (
            "clip_key",
            "resolved_video_path",
            "sampled_mapping_digest",
            "cache_key_digest",
            "raw_grounder_response",
            "grounding_prompt_version",
        ):
            if not isinstance(provenance.get(field), str) or not provenance.get(field):
                problems.append(f"grounding provenance {field!r} is missing")
        if not isinstance(provenance.get("original_t_src"), int):
            problems.append("grounding provenance original_t_src is missing")

    if mode == "bbox":
        bbox = result.get("bbox_2d_1000")
        if status == "not_found":
            if bbox is not None:
                problems.append("not_found bbox grounding contains geometry")
        elif (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1000.0
                for value in (bbox if isinstance(bbox, list) else [])
            )
            or (
                isinstance(bbox, list)
                and len(bbox) == 4
                and (float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]))
            )
        ):
            problems.append("bbox grounding geometry is invalid")
    elif mode == "points":
        points = result.get("points_2d_1000")
        if not isinstance(points, list):
            problems.append("points grounding geometry is not a list")
            points = []
        requested = arguments.get("count")
        if result.get("requested_count") != requested:
            problems.append("points grounding requested_count does not match action")
        if result.get("returned_count") != len(points):
            problems.append("points grounding returned_count does not match geometry")
        if status == "not_found" and points:
            problems.append("not_found points grounding contains geometry")
        if status == "ok" and not points:
            problems.append("ok points grounding contains no points")
        if isinstance(requested, int) and len(points) > requested:
            problems.append("points grounding returned more points than requested")
        expected_ids = [f"p{index}" for index in range(1, len(points) + 1)]
        actual_ids: list[Any] = []
        coordinates: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, Mapping):
                problems.append("points grounding contains a non-object point")
                continue
            actual_ids.append(point.get("point_id"))
            xy = point.get("xy")
            if (
                not isinstance(xy, list)
                or len(xy) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1000.0
                    for value in (xy if isinstance(xy, list) else [])
                )
            ):
                problems.append("points grounding contains invalid coordinates")
            else:
                coordinates.append((float(xy[0]), float(xy[1])))
            if not isinstance(point.get("description"), str) or not str(
                point.get("description")
            ).strip():
                problems.append("points grounding contains an empty description")
        if actual_ids != expected_ids:
            problems.append(
                f"point IDs are not host-assigned in order: {actual_ids!r}"
            )
        if len(set(coordinates)) != len(coordinates):
            problems.append("points grounding contains duplicate coordinates")
    return problems


def _d4rt_grounding_problems(
    arguments: Mapping[str, Any],
    result: Any,
    evidence: Mapping[str, Any],
) -> list[str]:
    """Validate that one accepted D4RT result resolves its cited qg exactly."""

    if not isinstance(result, Mapping):
        return ["successful D4RT result is not an object"]
    grounding_id = arguments.get("grounding_id")
    grounding = evidence.get(grounding_id) if isinstance(grounding_id, str) else None
    if not isinstance(grounding, Mapping):
        return [f"D4RT grounding {grounding_id!r} is absent from the registry"]
    problems: list[str] = []
    mode = grounding.get("mode")
    if result.get("grounding_id") != grounding_id:
        problems.append("D4RT result grounding_id does not match its action")
    if result.get("grounding_mode") != mode:
        problems.append("D4RT result grounding_mode does not match qg mode")
    if result.get("t_src") != grounding.get("t_src"):
        problems.append("D4RT source frame does not match qg source frame")
    if mode == "bbox":
        if result.get("point_mode") != "ensemble5":
            problems.append("bbox D4RT result did not use host-fixed ensemble5")
        if arguments.get("point_ids") is not None:
            problems.append("bbox D4RT action contains point_ids")
    elif mode == "points":
        points = grounding.get("points_2d_1000")
        available = [
            point.get("point_id")
            for point in points or []
            if isinstance(point, Mapping)
        ]
        selected = arguments.get("point_ids")
        expected = available if selected is None else list(selected)
        if any(point_id not in available for point_id in expected):
            problems.append("point-mode D4RT action selects unknown qg point IDs")
        if result.get("point_ids") != expected:
            problems.append("point-mode D4RT result point_ids do not match selection")
        tracks = result.get("point_tracks")
        track_ids = [
            track.get("point_id")
            for track in tracks or []
            if isinstance(track, Mapping)
        ]
        if track_ids != expected:
            problems.append("point-mode D4RT tracks do not match selected point IDs")
    return problems


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
    grounding_clip_key: str | None = None

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

        raw_value = row.get("raw_qwen_response")
        effective_value = row.get("effective_response")
        suffix_value = row.get("discarded_suffix")
        raw = raw_value if isinstance(raw_value, str) else ""
        effective = effective_value if isinstance(effective_value, str) else ""
        suffix = suffix_value if isinstance(suffix_value, str) else ""
        if not all(
            isinstance(value, str)
            for value in (raw_value, effective_value, suffix_value)
        ):
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="invalid_boundary_text",
                message="raw/effective/discarded response fields must all be strings",
            )

        span = row.get("action_span")
        if span is None:
            if effective != raw or suffix != "":
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    step=step,
                    code="boundary_reconstruction_mismatch",
                    message=(
                        "a no-action turn must retain the complete raw response as "
                        "effective history and have an empty discarded suffix"
                    ),
                )
            if action_objects(raw):
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    step=step,
                    code="missing_action_boundary",
                    message="raw response contains an action but action_span is null",
                )
            if row.get("parsed_action") is not None:
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    step=step,
                    code="parsed_action_without_boundary",
                    message="parsed_action must be null when no action was extracted",
                )
        elif isinstance(span, Mapping):
            start = span.get("start")
            end = span.get("end")
            valid_offsets = (
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end <= len(raw)
            )
            if not valid_offsets:
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    step=step,
                    code="invalid_action_span",
                    message=f"action_span {dict(span)!r} is outside the raw response",
                )
            else:
                decoder = json.JSONDecoder()
                try:
                    decoded, consumed = decoder.raw_decode(raw[start:])
                except json.JSONDecodeError:
                    decoded, consumed = None, -1
                if (
                    consumed != end - start
                    or not isinstance(decoded, dict)
                    or not ("action" in decoded or "name" in decoded)
                ):
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="action_span_decode_mismatch",
                        message="action_span does not select one complete action object",
                    )
                first = _first_action(raw)
                if first is None or (start, end) != (first[1], first[2]):
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="first_action_boundary_mismatch",
                        message=(
                            f"recorded action span {(start, end)!r} does not match "
                            f"the first raw action span "
                            f"{None if first is None else (first[1], first[2])!r}"
                        ),
                    )
                if effective != raw[:end].strip() or suffix != raw[end:]:
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="boundary_reconstruction_mismatch",
                        message=(
                            "effective_response/discarded_suffix do not reconstruct "
                            "the recorded raw response at action_span.end"
                        ),
                    )
                parsed_name = _action_name(row)
                decoded_name = (
                    decoded.get("action", decoded.get("name"))
                    if isinstance(decoded, Mapping) else None
                )
                if parsed_name != decoded_name:
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="parsed_action_mismatch",
                        message=(
                            f"action span names {decoded_name!r}, but parsed_action "
                            f"names {parsed_name!r}"
                        ),
                    )
                parsed_action = row.get("parsed_action")
                expected_action: Any = decoded
                if row.get("failure_stage") != "validation":
                    try:
                        normalized_name, normalized_arguments = validate_action(decoded)
                        expected_action = {
                            "action": normalized_name,
                            "arguments": normalized_arguments,
                        }
                    except (KeyError, TypeError, ValueError):
                        # Archived Phase 1 traces used raw bbox queries. They
                        # remain auditable even though Phase 2 no longer exposes
                        # that schema to the model.
                        expected_action = (
                            _normalize_legacy_bbox_query(decoded) or decoded
                        )
                if parsed_action != expected_action:
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="parsed_action_payload_mismatch",
                        message=(
                            "parsed_action does not match the action selected by "
                            "action_span after host normalization"
                        ),
                    )
        else:
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="invalid_action_span",
                message="action_span must be an object or null",
            )

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
        expected_last = {
            "ok": "accepted",
            "rejected": "rejected",
            "error": "error",
        }.get(status, "rejected")
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
        evidence_action = status == "ok" and name in {
            "ground_with_qwen",
            "query_d4rt",
            "python_math",
        }
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
            cache_hit = bool(row.get("cache_hit"))
            reused_id = row.get("reused_evidence_id")
            if cache_hit:
                if name != "ground_with_qwen":
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="invalid_cache_hit",
                        message=f"only ground_with_qwen may reuse evidence, got {name}",
                    )
                if reused_id != call_id or call_id not in prior_ids or new_id is not None:
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="invalid_cache_reuse",
                        message=(
                            f"cache hit must reuse an existing call_id without creating "
                            f"new evidence; call_id={call_id!r}, reused={reused_id!r}, "
                            f"new={new_id!r}, prior={sorted(prior_ids)!r}"
                        ),
                    )
            elif new_id != call_id:
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    step=step,
                    code="ledger_new_id_mismatch",
                    message=f"ledger created {new_id!r}, trace call_id is {call_id!r}",
                )
            if isinstance(call_id, str) and not cache_hit:
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
            if isinstance(call_id, str) and call_id in evidence:
                if row.get("result") != evidence.get(call_id):
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="trace_registry_result_mismatch",
                        message=(
                            f"trace result for {call_id!r} differs from the immutable "
                            "evidence registry"
                        ),
                    )
            action = row.get("parsed_action")
            arguments = (
                action.get("arguments")
                if isinstance(action, Mapping)
                and isinstance(action.get("arguments"), Mapping)
                else {}
            )
            if name == "ground_with_qwen" and isinstance(call_id, str):
                for problem in _grounding_result_problems(
                    call_id, arguments, row.get("result")
                ):
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="invalid_grounding_provenance",
                        message=problem,
                    )
                result = row.get("result")
                provenance = (
                    result.get("host_provenance")
                    if isinstance(result, Mapping) else None
                )
                clip_key = (
                    provenance.get("clip_key")
                    if isinstance(provenance, Mapping) else None
                )
                if isinstance(clip_key, str):
                    if grounding_clip_key is None:
                        grounding_clip_key = clip_key
                    elif grounding_clip_key != clip_key:
                        _violation(
                            violations,
                            question_id=question_id,
                            attempt=attempt_name,
                            step=step,
                            code="cross_clip_grounding_registry",
                            message="one attempt contains qg records from different clips",
                        )
            elif name == "query_d4rt" and isinstance(
                arguments.get("grounding_id"), str
            ):
                for problem in _d4rt_grounding_problems(
                    arguments, row.get("result"), evidence
                ):
                    _violation(
                        violations,
                        question_id=question_id,
                        attempt=attempt_name,
                        step=step,
                        code="invalid_d4rt_grounding_provenance",
                        message=problem,
                    )
        elif new_id is not None:
            _violation(
                violations,
                question_id=question_id,
                attempt=attempt_name,
                step=step,
                code="rejection_created_evidence",
                message=f"non-evidence action reports new evidence {new_id!r}",
            )

        grounder_failure = row.get("grounder_failure")
        if name == "ground_with_qwen" and grounder_failure is not None:
            valid_failure = (
                status == "rejected"
                and isinstance(grounder_failure, Mapping)
                and isinstance(grounder_failure.get("raw_grounder_response"), str)
                and isinstance(grounder_failure.get("host_provenance"), Mapping)
                and isinstance(
                    grounder_failure["host_provenance"].get("clip_key"), str
                )
            )
            if not valid_failure:
                _violation(
                    violations,
                    question_id=question_id,
                    attempt=attempt_name,
                    step=step,
                    code="invalid_grounder_failure_provenance",
                    message=(
                        "grounder parse failure must be a rejected action with raw "
                        "response and clip-bound host provenance"
                    ),
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
        for prefix in ("qg", "d4rt", "math"):
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
    if not records and not load_violations:
        load_violations.append({
            "question_id": "<directory>",
            "attempt": "file",
            "code": "no_answer_records",
            "message": f"no answer JSON files found in {answers_dir}",
        })
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
