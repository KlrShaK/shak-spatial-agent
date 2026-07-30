"""Compare one fixed-manifest DSI run with prior agent and baseline results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .dsi_bench_data import parse_choice_letter, read_manifest


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _records(directory: Path, subdirectory: str, expected: Sequence[str]) -> dict[str, dict[str, Any]]:
    source = directory / subdirectory
    found = {path.stem: path for path in source.glob("*.json")}
    expected_set = set(expected)
    if set(found) != expected_set:
        missing = sorted(expected_set - set(found))
        extra = sorted(set(found) - expected_set)
        raise ValueError(
            f"{source} does not match manifest IDs; missing={missing}, extra={extra}"
        )
    records: dict[str, dict[str, Any]] = {}
    for question_id in expected:
        record = _read_json(found[question_id])
        if str(record.get("question_id")) != question_id:
            raise ValueError(
                f"{found[question_id]} payload question_id="
                f"{record.get('question_id')!r}, expected {question_id!r}"
            )
        records[question_id] = record
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_run_identity(
    records: Mapping[str, Mapping[str, Any]],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Validate one implementation/config identity while allowing resumed jobs."""

    invariant_keys = (
        "code_sha",
        "manifest_sha256",
        "qwen_model_path",
        "d4rt",
        "seed",
        "main_max_new_tokens",
        "grounder_bbox_max_new_tokens",
        "grounder_points_max_new_tokens",
        "max_steps",
        "point_mode",
        "prompt_sha256",
    )
    identities: dict[str, str] = {}
    jobs: set[str] = set()
    nodes: set[str] = set()
    run_ids: set[str] = set()
    hardware_names: set[str] = set()
    first_metadata: dict[str, Any] | None = None
    for question_id, record in records.items():
        metadata = record.get("run_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{question_id} is missing run_metadata")
        if metadata.get("manifest_sha256") != manifest_sha256:
            raise ValueError(
                f"{question_id} manifest SHA does not match current manifest"
            )
        if first_metadata is None:
            first_metadata = dict(metadata)
            identities = {
                key: json.dumps(metadata.get(key), sort_keys=True, separators=(",", ":"))
                for key in invariant_keys
            }
        else:
            for key in invariant_keys:
                encoded = json.dumps(
                    metadata.get(key), sort_keys=True, separators=(",", ":")
                )
                if encoded != identities[key]:
                    raise ValueError(
                        f"mixed run metadata for {key!r}; first and "
                        f"{question_id} differ"
                    )
        if metadata.get("slurm_job_id"):
            jobs.add(str(metadata["slurm_job_id"]))
        if metadata.get("slurm_node"):
            nodes.add(str(metadata["slurm_node"]))
        if metadata.get("run_id"):
            run_ids.add(str(metadata["run_id"]))
        hardware = metadata.get("hardware")
        if isinstance(hardware, Mapping) and hardware.get("name"):
            hardware_names.add(str(hardware["name"]))
    if len(hardware_names) > 1:
        raise ValueError(f"mixed GPU models in current records: {sorted(hardware_names)}")
    identity = first_metadata or {}
    identity["slurm_job_ids"] = sorted(jobs)
    identity["slurm_nodes"] = sorted(nodes)
    identity["run_ids"] = sorted(run_ids)
    if hardware_names:
        identity["hardware_names"] = sorted(hardware_names)
    return identity


def _attempts(record: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    gated = record.get("gated_attempt")
    if isinstance(gated, Mapping):
        yield "strict", gated
    if isinstance(record.get("trace"), list):
        name = "relaxed" if isinstance(gated, Mapping) else "strict"
        yield name, record


def _letter(record: Mapping[str, Any], letters: Sequence[str]) -> str | None:
    if record.get("status") not in {"complete", "complete_relaxed"}:
        return None
    final = record.get("final_answer")
    if not isinstance(final, Mapping):
        return None
    text = final.get("text")
    if not isinstance(text, str):
        return None
    return parse_choice_letter(text, letters)


def _visibility(result: Mapping[str, Any]) -> float | None:
    value = result.get("visibility_coverage")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    tracks = result.get("point_tracks")
    values = [
        float(track["visibility_coverage"])
        for track in tracks or []
        if isinstance(track, Mapping)
        and isinstance(track.get("visibility_coverage"), (int, float))
    ]
    return mean(values) if values else None


def summarize(
    *,
    manifest: Mapping[str, Any],
    current: Mapping[str, Mapping[str, Any]],
    previous: Mapping[str, Mapping[str, Any]],
    failed: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    questions = list(manifest["questions"])
    count = len(questions)
    order = [str(item["question_id"]) for item in questions]
    gt = {str(item["question_id"]): str(item["gt"]) for item in questions}
    letters = {
        str(item["question_id"]): list(item["option_letters"]) for item in questions
    }

    def score(records: Mapping[str, Mapping[str, Any]], *, baseline_mode: bool = False) -> dict[str, Any]:
        parsed: dict[str, str | None] = {}
        for question_id in order:
            record = records[question_id]
            parsed[question_id] = (
                parse_choice_letter(
                    str(record.get("raw_response", "")),
                    letters[question_id],
                )
                if baseline_mode and record.get("status") == "complete"
                else _letter(record, letters[question_id])
                if not baseline_mode
                else None
            )
        completed = sum(value is not None for value in parsed.values())
        correct = sum(parsed[question_id] == gt[question_id] for question_id in order)
        completed_ids = [
            question_id
            for question_id in order
            if records[question_id].get("status") in {"complete", "complete_relaxed"}
        ]
        completed_correct = sum(
            parsed[question_id] == gt[question_id]
            for question_id in completed_ids
        )
        return {
            "correct": correct,
            "total": count,
            "accuracy": correct / count if count else 0.0,
            "parsed_answers": completed,
            "accuracy_among_parsed": correct / completed if completed else 0.0,
            "completed_answers": len(completed_ids),
            "correct_among_completed": completed_correct,
            "accuracy_among_completed": (
                completed_correct / len(completed_ids) if completed_ids else 0.0
            ),
            "letters": parsed,
        }

    current_score = score(current)
    previous_score = score(previous)
    failed_score = score(failed)
    baseline_score = score(baseline, baseline_mode=True)
    statuses = Counter(str(current[item].get("status", "unknown")) for item in order)
    tools = Counter()
    qg_modes = Counter()
    not_found = 0
    malformed_grounder = 0
    discarded = 0
    model_turns = 0
    steps: list[int] = []
    visibility: dict[str, list[float]] = defaultdict(list)
    per_question_steps: dict[str, int] = {}

    for question_id in order:
        record = current[question_id]
        question_steps = 0
        saw_trace = False
        for _, attempt in _attempts(record):
            trace = attempt.get("trace")
            if isinstance(trace, list):
                saw_trace = True
                question_steps += len(trace)
        if not saw_trace and isinstance(record.get("steps"), int):
            question_steps = int(record["steps"])
        steps.append(question_steps)
        per_question_steps[question_id] = question_steps
        for _, attempt in _attempts(record):
            trace = attempt.get("trace")
            if not isinstance(trace, list):
                continue
            for row in trace:
                if not isinstance(row, Mapping):
                    continue
                model_turns += 1
                action = (row.get("parsed_action") or {}).get("action")
                if row.get("status") == "rejected":
                    tools["rejected"] += 1
                    if action == "ground_with_qwen" and isinstance(
                        row.get("grounder_failure"), Mapping
                    ):
                        tools["qg_calls"] += 1
                        tools["qg_inferences"] += 1
                        malformed_grounder += 1
                        arguments = (row.get("parsed_action") or {}).get("arguments") or {}
                        qg_modes[str(arguments.get("mode", "unknown"))] += 1
                if row.get("discarded_suffix"):
                    discarded += 1
                if row.get("status") != "ok":
                    continue
                if action == "ground_with_qwen":
                    tools["qg_calls"] += 1
                    if row.get("cache_hit"):
                        tools["qg_cache_hits"] += 1
                    else:
                        tools["qg_inferences"] += 1
                    result = row.get("result") or {}
                    qg_modes[str(result.get("mode", "unknown"))] += 1
                    if result.get("status") == "not_found":
                        not_found += 1
                elif action == "query_d4rt":
                    tools["d4rt_calls"] += 1
                    result = row.get("result") or {}
                    mode = str(result.get("grounding_mode", "unknown"))
                    coverage = _visibility(result)
                    if coverage is not None:
                        visibility[mode].append(coverage)
                elif action == "python_math":
                    tools["math_calls"] += 1

    flips: dict[str, list[dict[str, Any]]] = {}
    for name, reference in (
        ("previous_agent", previous_score),
        ("failed_forced_grounding", failed_score),
        ("baseline", baseline_score),
    ):
        rows = []
        for question_id in order:
            before = reference["letters"][question_id]
            after = current_score["letters"][question_id]
            if before != after:
                rows.append({
                    "question_id": question_id,
                    "before": before,
                    "after": after,
                    "gt": gt[question_id],
                    "change": (
                        "gain"
                        if after == gt[question_id] and before != gt[question_id]
                        else "loss"
                        if before == gt[question_id] and after != gt[question_id]
                        else "changed_wrong"
                    ),
                })
        flips[name] = rows

    runtime_values = [
        float(current[item]["wall_seconds"])
        for item in order
        if isinstance(current[item].get("wall_seconds"), (int, float))
    ]
    per_question = [{
        "question_id": question_id,
        "gt": gt[question_id],
        "current": current_score["letters"][question_id],
        "previous": previous_score["letters"][question_id],
        "failed": failed_score["letters"][question_id],
        "baseline": baseline_score["letters"][question_id],
        "status": current[question_id].get("status"),
        "dataset": next(
            str(item["dataset"]) for item in questions
            if str(item["question_id"]) == question_id
        ),
        "task": next(
            str(item["category_name"]) for item in questions
            if str(item["question_id"]) == question_id
        ),
        "steps": per_question_steps[question_id],
        "runtime_seconds": current[question_id].get("wall_seconds"),
        "correct": current_score["letters"][question_id] == gt[question_id],
    } for question_id in order]
    task_results: dict[str, dict[str, Any]] = {}
    for task in sorted({str(item["category_name"]) for item in questions}):
        rows = [row for row in per_question if row["task"] == task]
        task_results[task] = {
            "correct": sum(bool(row["correct"]) for row in rows),
            "total": len(rows),
            "accuracy": (
                sum(bool(row["correct"]) for row in rows) / len(rows)
                if rows else 0.0
            ),
            "statuses": dict(Counter(str(row["status"]) for row in rows)),
        }
    sample_metadata = current[order[0]].get("run_metadata") if order else None
    run_metadata = dict(sample_metadata) if isinstance(sample_metadata, Mapping) else {}

    return {
        "questions": count,
        "manifest": {
            "benchmark": manifest.get("benchmark"),
            "split": manifest.get("split"),
            "sampling": manifest.get("sampling"),
        },
        "run_metadata": run_metadata,
        "current": current_score,
        "previous_agent": previous_score,
        "failed_forced_grounding": failed_score,
        "baseline": baseline_score,
        "completion": {
            "strict": statuses.get("complete", 0),
            "strict_rate": statuses.get("complete", 0) / count if count else 0.0,
            "relaxed": statuses.get("complete_relaxed", 0),
            "relaxed_rate": (
                statuses.get("complete_relaxed", 0) / count if count else 0.0
            ),
            "failed": statuses.get("failed", 0),
            "failed_rate": statuses.get("failed", 0) / count if count else 0.0,
            "error": statuses.get("error", 0),
            "statuses": dict(statuses),
        },
        "grounding": {
            **dict(tools),
            "modes": dict(qg_modes),
            "not_found": not_found,
            "malformed": malformed_grounder,
        },
        "discarded_suffix_turns": discarded,
        "model_turns": model_turns,
        "discarded_suffix_rate": (
            discarded / model_turns if model_turns else 0.0
        ),
        "steps": {
            "total": sum(steps),
            "mean": mean(steps) if steps else 0.0,
            "max": max(steps) if steps else 0,
        },
        "visibility_coverage_by_mode": {
            mode: {"mean": mean(values), "count": len(values)}
            for mode, values in sorted(visibility.items())
        },
        "runtime": {
            "total_seconds": sum(runtime_values),
            "mean_seconds": mean(runtime_values) if runtime_values else 0.0,
        },
        "answer_flips": flips,
        "per_task": task_results,
        "per_question": per_question,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    current = summary["current"]
    completion = summary["completion"]
    grounding = summary["grounding"]
    metadata = summary.get("run_metadata", {})
    hardware = metadata.get("hardware", {}) if isinstance(metadata, Mapping) else {}
    failed_completion = summary["failed_forced_grounding"]["completed_answers"]
    lines = [
        "# DSI-Bench grounding-tool comparison",
        "",
        "## Run identity",
        "",
        f"- Implementation commit: `{metadata.get('code_sha', 'unknown')}`",
        (
            f"- Blackwell Slurm job(s): "
            f"`{metadata.get('slurm_job_ids', [metadata.get('slurm_job_id', 'unknown')])}` "
            f"on `{metadata.get('slurm_nodes', [metadata.get('slurm_node', 'unknown')])}`"
        ),
        (
            f"- GPU: "
            f"`{metadata.get('hardware_names', [hardware.get('name', 'unknown')])}`"
        ),
        (
            f"- Fixed manifest: {summary['questions']} questions from "
            f"`{summary.get('manifest', {}).get('benchmark', 'DSI-Bench')}` "
            f"`{summary.get('manifest', {}).get('split', 'std')}`"
        ),
        "",
        "## Headline results",
        "",
        f"- Current: {current['correct']}/{current['total']}",
        f"- Previous agent: {summary['previous_agent']['correct']}/{summary['questions']}",
        (
            f"- Failed forced-grounding run: "
            f"{summary['failed_forced_grounding']['correct']}/{summary['questions']} "
            f"with {summary['questions'] - failed_completion} unanswered/failed"
        ),
        f"- Unchanged Qwen baseline: {summary['baseline']['correct']}/{summary['questions']}",
        (
            f"- Completion: {completion['strict']} strict, {completion['relaxed']} relaxed, "
            f"{completion['failed']} failed, {completion['error']} infrastructure errors"
        ),
        (
            f"- Completion rates: {100 * completion['strict_rate']:.1f}% strict, "
            f"{100 * completion['relaxed_rate']:.1f}% relaxed, "
            f"{100 * completion['failed_rate']:.1f}% failed"
        ),
        (
            f"- Accuracy among completed statuses: "
            f"{current['correct_among_completed']}/{current['completed_answers']} "
            f"({100 * current['accuracy_among_completed']:.1f}%)"
        ),
        "",
        "## Tool and orchestration metrics",
        "",
        (
            f"- Grounding: {grounding.get('qg_calls', 0)} calls, "
            f"{grounding.get('qg_inferences', 0)} inferences, "
            f"{grounding.get('qg_cache_hits', 0)} cache hits, "
            f"{grounding.get('not_found', 0)} not found, "
            f"{grounding.get('malformed', 0)} malformed"
        ),
        f"- Grounding modes: {grounding.get('modes', {})}",
        (
            f"- Downstream calls: {grounding.get('d4rt_calls', 0)} D4RT, "
            f"{grounding.get('math_calls', 0)} math, "
            f"{grounding.get('rejected', 0)} rejected"
        ),
        (
            f"- Steps: {summary['steps']['total']} total, "
            f"{summary['steps']['mean']:.2f} mean, {summary['steps']['max']} max"
        ),
        (
            f"- Discarded-suffix turns: {summary['discarded_suffix_turns']}/"
            f"{summary['model_turns']} "
            f"({100 * summary['discarded_suffix_rate']:.1f}%)"
        ),
        f"- D4RT visibility by grounding mode: {summary['visibility_coverage_by_mode']}",
        (
            f"- Runtime: {summary['runtime']['total_seconds'] / 60:.1f} minutes total, "
            f"{summary['runtime']['mean_seconds']:.1f} seconds/question"
        ),
        "",
        "## Per task",
        "",
        "| Task | Correct | Total | Accuracy | Statuses |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for task, row in summary["per_task"].items():
        lines.append(
            f"| {task} | {row['correct']} | {row['total']} | "
            f"{100 * row['accuracy']:.1f}% | {row['statuses']} |"
        )
    lines += [
        "",
        "## Answer flips",
        "",
    ]
    for reference, flips in summary["answer_flips"].items():
        gains = sum(row["change"] == "gain" for row in flips)
        losses = sum(row["change"] == "loss" for row in flips)
        changed_wrong = sum(row["change"] == "changed_wrong" for row in flips)
        lines.append(
            f"- Versus {reference}: {gains} gains, {losses} losses, "
            f"{changed_wrong} changed-wrong answers."
        )
        for row in flips:
            lines.append(
                f"  - `{row['question_id']}`: {row['before'] or '—'} → "
                f"{row['after'] or '—'} (GT {row['gt']}, {row['change']})"
            )
    lines += [
        "",
        "## Per question",
        "",
        "| Question | Task | Current | Previous | Failed prompt | Baseline | GT | Status | Steps | Seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in summary["per_question"]:
        lines.append(
            f"| {row['question_id']} | {row['task']} | {row['current'] or '—'} | "
            f"{row['previous'] or '—'} | {row['failed'] or '—'} | "
            f"{row['baseline'] or '—'} | {row['gt']} | {row['status']} | "
            f"{row['steps']} | {row['runtime_seconds'] or 0} |"
        )
    lines += [
        "",
        "## Visual review and interpretation",
        "",
        "Grounding overlays, failure taxonomy, audit status, limitations, and the final "
        "hypothesis assessment are added after exhaustive staged-artifact review.",
    ]
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def compare(
    current_results: Path,
    previous_results: Path,
    failed_results: Path,
) -> dict[str, Any]:
    manifest = read_manifest(current_results / "manifest.json")
    order = [str(item["question_id"]) for item in manifest["questions"]]
    current = _records(current_results, "answers", order)
    previous = _records(previous_results, "answers", order)
    failed = _records(failed_results, "answers", order)
    baseline = _records(current_results, "baseline", order)
    previous_manifest_path = previous_results / "manifest.json"
    if previous_manifest_path.exists() and (
        _sha256(previous_manifest_path)
        != _sha256(current_results / "manifest.json")
    ):
        raise ValueError("previous-agent manifest differs from current manifest")
    summary = summarize(
        manifest=manifest,
        current=current,
        previous=previous,
        failed=failed,
        baseline=baseline,
    )
    summary["run_metadata"] = _validated_run_identity(
        current,
        manifest_sha256=_sha256(current_results / "manifest.json"),
    )
    if manifest.get("benchmark") == "DSI-Bench" and len(order) == 25:
        expected = {
            "previous_agent": 10,
            "failed_forced_grounding": 7,
            "baseline": 13,
        }
        actual = {
            name: summary[name]["correct"]
            for name in expected
        }
        if actual != expected:
            raise ValueError(
                f"reference headline mismatch: expected {expected}, got {actual}"
            )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-results", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path, required=True)
    parser.add_argument("--failed-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = compare(args.current_results, args.previous_results, args.failed_results)
    _write(args.output_json, json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _write(args.output_markdown, _markdown(summary))
    print(
        f"current {summary['current']['correct']}/{summary['questions']}; "
        f"previous {summary['previous_agent']['correct']}/{summary['questions']}; "
        f"failed {summary['failed_forced_grounding']['correct']}/{summary['questions']}; "
        f"baseline {summary['baseline']['correct']}/{summary['questions']}"
    )


if __name__ == "__main__":
    main()
