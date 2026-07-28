#!/usr/bin/env python
"""Print one DSI-Bench question's full agent trace: thinking, tool calls, numbers.

The per-question JSON holds everything, but ``raw_qwen_response`` interleaves the
model's prose with the JSON action it ends on, and the measurements live in a
separate ``evidence`` map keyed by call id.  This stitches the two back together
in the order the model saw them.

Usage::

    python d4rt_agent/dsi_bench_show.py M0jmSsQ5ptw          # match by video path
    python d4rt_agent/dsi_bench_show.py CameraBench_c0        # or by question id
    python d4rt_agent/dsi_bench_show.py --list                # what is available
    python d4rt_agent/dsi_bench_show.py M0jmSsQ5ptw --raw     # unabridged prose
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Mapping

RESULTS = Path(__file__).resolve().parent / "results" / "dsi_bench"
RULE = "=" * 100
UNPARSED_LIMIT = 600


def _load(directory: str) -> list[dict[str, Any]]:
    root = RESULTS / directory
    return [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]


def _wrap(text: str, indent: str = "    ") -> str:
    paragraphs = [p.strip() for p in text.strip().split("\n") if p.strip()]
    return "\n".join(
        textwrap.fill(p, width=100, initial_indent=indent, subsequent_indent=indent)
        for p in paragraphs
    )


def _thinking(raw: str) -> str:
    """The model reasons in prose, then emits one JSON action; keep the prose."""

    cut = raw.find('{"action"')
    # ``cut == 0`` means the model skipped the prose and went straight to the
    # action -- that is an empty thought, not an unparseable one.
    if cut >= 0:
        return raw[:cut].strip()
    # No action at all: the step was rejected, and the reply is sometimes the
    # previous tool result echoed back verbatim. Cap it so one malformed step
    # cannot bury the rest of the trace.
    text = raw.strip()
    return text if len(text) <= UNPARSED_LIMIT else text[:UNPARSED_LIMIT] + " […truncated]"


def _rejection(step: Mapping[str, Any]) -> str:
    """Why the host refused the step. Older traces recorded it under `result`."""

    return str(step.get("error") or step.get("result") or "unspecified")


def _xyz(values: list[float] | None) -> str:
    return "null" if not values else "[" + ", ".join(f"{v:+.4f}" for v in values) + "]"


def _describe_query(record: dict[str, Any]) -> list[str]:
    """One line per returned frame: the numbers the model actually got back."""

    lines = []
    for prediction in record.get("predictions", []):
        visible = prediction.get("visible")
        lines.append(
            f"      t_tgt={prediction['sampled_frame_index']:<3} "
            f"visible={str(visible):<5} "
            f"valid={prediction.get('valid_count')}/5  "
            f"xyz={_xyz(prediction.get('benchmark_aligned_xyz_m'))}  "
            f"std={_xyz(prediction.get('benchmark_aligned_xyz_std_m'))}"
        )
    return lines


def _quote(text: str) -> list[str]:
    """Markdown blockquote. Blank lines need their own ``>`` or the quote breaks."""

    return [f"> {line}" if line.strip() else ">" for line in text.strip().split("\n")]


def _markdown_query_table(record: Mapping[str, Any]) -> list[str]:
    """What came back, one row per requested frame.

    The interesting comparison is xyz against std -- a displacement smaller than
    the spread beside it is noise -- so they sit in adjacent columns.
    """

    rows = [
        "| `t_tgt` | visible | valid | `benchmark_aligned_xyz_m` | ± spread |",
        "| ---: | :---: | :---: | --- | --- |",
    ]
    for prediction in record.get("predictions", []):
        visible = prediction.get("visible")
        xyz = prediction.get("benchmark_aligned_xyz_m")
        std = prediction.get("benchmark_aligned_xyz_std_m")
        rows.append(
            f"| {prediction['sampled_frame_index']} "
            f"| {'✓' if visible else '✗'} "
            f"| {prediction.get('valid_count')}/5 "
            f"| {'`' + _xyz(xyz) + '`' if xyz else '—'} "
            f"| {'`' + _xyz(std) + '`' if std else '—'} |"
        )
    return rows


def render_trace_markdown(record: Mapping[str, Any]) -> list[str]:
    """The same trace as :func:`render_trace`, but as markdown rather than text.

    Three things need to be told apart at a glance -- what the model thought,
    what it called, and what came back -- so each gets its own form: a
    blockquote, a fenced call, and a table.  Prose is left unwrapped here
    because markdown reflows it; the text renderer wraps because a terminal
    does not.
    """

    evidence = record.get("evidence", {})
    lines: list[str] = []
    for step in record.get("trace", []):
        action = step.get("parsed_action") or {}
        name = action.get("action", "-")
        status = step["status"]
        badge = "" if status == "ok" else f" · **{status}**"
        lines += [f"**Step {step['step']}** · `{name}`{badge}", ""]

        thought = _thinking(step.get("raw_qwen_response") or "")
        if thought:
            lines += _quote(thought) + [""]

        arguments = action.get("arguments") or {}
        if name == "query_d4rt":
            lines += [
                "```python",
                f"query_d4rt(label={arguments.get('label')!r}, "
                f"bbox_2d_1000={arguments.get('bbox_2d_1000')},",
                f"           t_src={arguments.get('t_src')}, "
                f"t_tgt={arguments.get('t_tgt')}, t_cam={arguments.get('t_cam')})",
                "```",
                "",
            ]
        elif name == "python_math":
            lines += ["```python", *(arguments.get("code") or "").split("\n"), "```", ""]
            bound = [
                f"`{key}` ← `{value.get('evidence_id')}."
                + ".".join(str(part) for part in value.get("path", []))
                + "`"
                for key, value in (arguments.get("bindings") or {}).items()
            ]
            if bound:
                lines += [f"*Bound from evidence:* {', '.join(bound)}", ""]
        elif name == "final_answer":
            cited = arguments.get("evidence_ids") or []
            lines += [
                f"**Answer.** {arguments.get('text', '')}",
                "",
                f"*Cites* {', '.join(f'`{i}`' for i in cited) if cited else '**nothing**'} · "
                f"*limitations:* {arguments.get('limitations', '') or '—'}",
                "",
            ]

        justification = arguments.get("justification")
        if justification:
            lines += [f"*Why:* {justification}", ""]

        call_id = step.get("call_id")
        found = evidence.get(call_id) if call_id else None
        if found and call_id.startswith("d4rt_"):
            lines += [f"*Returned* `{call_id}`:", "", *_markdown_query_table(found), ""]
        elif found and call_id.startswith("math_"):
            outputs = ", ".join(
                f"`{key} = {value}`" for key, value in (found.get("outputs") or {}).items()
            )
            lines += [f"*Returned* `{call_id}`: {outputs}", ""]
        if status != "ok":
            lines += [f"> ⛔ **Host rejected this step.** {_rejection(step)[:400]}", ""]
        lines += ["---", ""]
    # The separator belongs *between* steps; the report supplies its own after
    # the closing </details>.
    while lines and lines[-1] in {"", "---"}:
        lines.pop()
    return lines + [""]


def render_trace(record: Mapping[str, Any], raw: bool = False) -> list[str]:
    """The model's reasoning, its call, and what came back, step by step.

    Returned rather than printed so the markdown report can embed the same text
    it prints here -- one rendering, two consumers.
    """

    evidence = record.get("evidence", {})
    lines: list[str] = []
    for step in record.get("trace", []):
        action = step.get("parsed_action") or {}
        name = action.get("action", "-")
        lines.append(f"--- step {step['step']}  {name}  [{step['status']}] " + "-" * 40)
        prose = step.get("raw_qwen_response") or ""
        thought = prose if raw else _thinking(prose)
        if thought:
            lines.append(_wrap(thought))
        arguments = action.get("arguments") or {}
        if name == "query_d4rt":
            lines.append(
                f"    CALL  label={arguments.get('label')!r} "
                f"bbox={arguments.get('bbox_2d_1000')} "
                f"t_src={arguments.get('t_src')} "
                f"t_tgt={arguments.get('t_tgt')} "
                f"t_cam={arguments.get('t_cam')}"
            )
        elif name == "python_math":
            lines.append(f"    CALL  bindings={json.dumps(arguments.get('bindings', {}))}")
            lines += [f"          | {line}" for line in (arguments.get("code") or "").split("\n")]
        elif name == "final_answer":
            lines += [
                f"    ANSWER  {arguments.get('text', '')}",
                f"    cites   {arguments.get('evidence_ids')}",
                f"    limits  {arguments.get('limitations', '')}",
            ]

        call_id = step.get("call_id")
        found = evidence.get(call_id) if call_id else None
        if found and call_id.startswith("d4rt_"):
            lines.append(f"    -> {call_id}")
            lines += _describe_query(found)
        elif found and call_id.startswith("math_"):
            lines.append(f"    -> {call_id}  {json.dumps(found.get('outputs', {}))}")
        if step["status"] != "ok":
            lines.append(f"    !! REJECTED: {_rejection(step)[:400]}")
        lines.append("")
    return lines


def show(record: dict[str, Any], baseline: dict[str, Any] | None, raw: bool) -> None:
    options = record["options"]
    gt = record["gt"]
    print(RULE)
    print(f"{record['relative_path']}   [{record['category_name']} / {record['dataset']}]")
    print(RULE)
    print(f"Q: {record['question']}")
    for letter, text in options.items():
        mark = "  <-- GROUND TRUTH" if letter == gt else ""
        print(f"   {letter}: {text}{mark}")
    sampling = record["sampling"]
    print(
        f"\nvideo {sampling['width']}x{sampling['height']} @ {sampling['fps']}fps, "
        f"{sampling['total_original_frames']} frames -> 32 sampled"
    )
    print(
        f"steps={record['steps']}  status={record['status']}  "
        f"d4rt_used={record['d4rt_used']}  {record.get('wall_seconds', 0):.0f}s"
    )
    print()
    print("\n".join(render_trace(record, raw)))

    if baseline is not None:
        print(f"{'-' * 40} Qwen-only baseline (no tools) {'-' * 29}")
        print(_wrap(baseline.get("raw_response", "")))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", help="substring of the video path or question id")
    parser.add_argument("--list", action="store_true", help="list every question and exit")
    parser.add_argument("--raw", action="store_true", help="keep the emitted JSON in the prose")
    parser.add_argument("--no-baseline", action="store_true", help="skip the tool-free reply")
    args = parser.parse_args()

    answers = _load("answers")
    if args.list or not args.pattern:
        for record in answers:
            print(
                f"{record['relative_path']:<62} {record['category_name']:<20} "
                f"GT={record['gt']}  steps={record['steps']}"
            )
        return

    needle = args.pattern.lower()
    matches = [
        record
        for record in answers
        if needle in record["relative_path"].lower() or needle in record["question_id"].lower()
    ]
    if not matches:
        raise SystemExit(f"no question matches {args.pattern!r}; try --list")

    baselines = {record["question_id"]: record for record in _load("baseline")}
    for record in matches:
        show(
            record,
            None if args.no_baseline else baselines.get(record["question_id"]),
            args.raw,
        )


if __name__ == "__main__":
    main()
