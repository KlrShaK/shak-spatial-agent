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
import html
import json
import textwrap
from pathlib import Path
from typing import Any, Mapping

try:
    from .orchestration_audit import count_action_objects, iter_attempts
except ImportError:  # Support the documented ``python d4rt_agent/dsi_bench_show.py`` form.
    from orchestration_audit import count_action_objects, iter_attempts

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


def _effective_response(step: Mapping[str, Any]) -> str:
    """Return the response that actually entered model-visible history."""

    value = step.get("effective_response")
    if isinstance(value, str):
        return value
    return str(step.get("raw_qwen_response") or "")


def _effective_thinking(step: Mapping[str, Any]) -> str:
    """Keep reasoning before the submitted action, using exact Phase 1 bounds."""

    effective = _effective_response(step)
    span = step.get("action_span")
    if isinstance(span, Mapping):
        start = span.get("start")
        raw = str(step.get("raw_qwen_response") or "")
        if isinstance(start, int) and not isinstance(start, bool) and 0 <= start <= len(raw):
            # action_span indexes the unstripped raw generation, whereas
            # effective_response is intentionally stripped before history.
            return raw[:start].strip()
    if "action_span" in step and span is None:
        # A Phase 1 parse rejection has no submitted action boundary. The whole
        # response entered effective history, including any malformed
        # action-like tail, so the normal trace view must show the whole turn.
        return (
            effective
            if len(effective) <= UNPARSED_LIMIT
            else effective[:UNPARSED_LIMIT] + " […truncated]"
        )
    return _thinking(effective)


def trace_boundary_metrics(record: Mapping[str, Any]) -> dict[str, int]:
    """Summarize model output discarded by the one-action boundary."""

    suffixes = [
        str(step.get("discarded_suffix", ""))
        for step in record.get("trace", [])
        if isinstance(step, Mapping)
    ]
    nonempty = [suffix for suffix in suffixes if suffix.strip()]
    return {
        "turns": len(suffixes),
        "discarded_suffixes": len(nonempty),
        "discarded_suffixes_with_action": sum(
            1 for suffix in nonempty if count_action_objects(suffix)
        ),
    }


def _markdown_boundary_diagnostic(step: Mapping[str, Any]) -> list[str]:
    """Collapsed, HTML-safe raw/effective/suffix diagnostics for one turn."""

    suffix = str(step.get("discarded_suffix", ""))
    if not suffix.strip():
        return []
    raw = str(step.get("raw_qwen_response", ""))
    effective = _effective_response(step)
    actions = count_action_objects(suffix)
    action_label = f"; {actions} additional action(s)" if actions else ""
    return [
        "<details><summary>Turn-boundary diagnostic — "
        f"discarded {len(suffix)} character(s){action_label}</summary>",
        "",
        "<p><strong>Effective response retained in history</strong></p>",
        f"<pre><code>{html.escape(effective)}</code></pre>",
        "<p><strong>Discarded suffix</strong></p>",
        f"<pre><code>{html.escape(suffix)}</code></pre>",
        "<p><strong>Complete raw model response</strong></p>",
        f"<pre><code>{html.escape(raw)}</code></pre>",
        "",
        "</details>",
        "",
    ]


def _rejection(step: Mapping[str, Any]) -> str:
    """Why the host refused the step. Older traces recorded it under `result`."""

    return str(step.get("error") or step.get("result") or "unspecified")


def _xyz(values: list[float] | None) -> str:
    return "null" if not values else "[" + ", ".join(f"{v:+.4f}" for v in values) + "]"


def _namespaced_id(evidence_id: str, namespace: str | None) -> str:
    """Disambiguate IDs whose counters restart on the relaxed retry."""

    return f"{namespace}/{evidence_id}" if namespace else evidence_id


def _host_provenance(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return Phase 2's diagnostic-only grounding provenance, if present."""

    for key in ("host_provenance", "grounding_provenance", "provenance"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _grounding_cache_description(
    step: Mapping[str, Any], record: Mapping[str, Any]
) -> tuple[bool, str | None]:
    """Read cache metadata from both live and early Phase 2 record layouts."""

    candidates: list[Mapping[str, Any]] = [step, record, _host_provenance(record)]
    result = step.get("result")
    if isinstance(result, Mapping):
        candidates.insert(1, result)
    cache_hit = any(value.get("cache_hit") is True for value in candidates)
    reused = next(
        (
            str(value["reused_evidence_id"])
            for value in candidates
            if isinstance(value.get("reused_evidence_id"), str)
        ),
        None,
    )
    return cache_hit, reused


def _grounding_geometry(record: Mapping[str, Any]) -> str:
    mode = record.get("mode")
    if mode == "bbox":
        return f"bbox_2d_1000={record.get('bbox_2d_1000')}"
    if mode == "points":
        points = record.get("points_2d_1000") or []
        return "points_2d_1000=" + json.dumps(points, ensure_ascii=False)
    return "geometry=unknown"


def _grounding_raw_response(record: Mapping[str, Any]) -> str:
    host = _host_provenance(record)
    value = host.get("raw_grounder_response", record.get("raw_grounder_response", ""))
    return str(value or "")


def _describe_grounding(
    record: Mapping[str, Any],
    evidence_id: str,
    step: Mapping[str, Any],
    namespace: str | None,
) -> list[str]:
    cache_hit, reused = _grounding_cache_description(step, record)
    cache = "hit" if cache_hit else "miss"
    if reused:
        cache += f", reused {_namespaced_id(reused, namespace)}"
    original = _host_provenance(record).get("original_t_src")
    source = f"sampled frame {record.get('t_src')}"
    if original is not None:
        source += f" (original frame {original})"
    return [
        f"    -> {_namespaced_id(evidence_id, namespace)}  "
        f"status={record.get('status', '?')} mode={record.get('mode', '?')} cache={cache}",
        f"       source={source} request={record.get('request', '')!r}",
        f"       {_grounding_geometry(record)}",
    ]


def _describe_query(record: dict[str, Any]) -> list[str]:
    """One line per returned frame: the numbers the model actually got back."""

    tracks = record.get("point_tracks")
    if isinstance(tracks, list):
        lines: list[str] = []
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            lines.append(
                f"      {track.get('point_id')} {track.get('source_xy_1000')} "
                f"{track.get('description', '')}"
            )
            for prediction in track.get("predictions", []):
                lines.append(
                    f"        t_tgt={prediction['sampled_frame_index']:<3} "
                    f"visible={str(prediction.get('visible')):<5} "
                    f"xyz={_xyz(prediction.get('benchmark_aligned_xyz_m'))}"
                )
        return lines

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


def _markdown_query_result(record: Mapping[str, Any]) -> list[str]:
    """Render bbox aggregation or independent Phase 2 point trajectories."""

    tracks = record.get("point_tracks")
    if not isinstance(tracks, list):
        return _markdown_query_table(record)
    lines: list[str] = []
    for track in tracks:
        if not isinstance(track, Mapping):
            continue
        lines += [
            f"**Point `{track.get('point_id', '?')}`** — "
            f"`source_xy_1000={track.get('source_xy_1000')}`; "
            f"{track.get('description', '')}",
            "",
            *_markdown_query_table(track),
            "",
        ]
    return lines or ["_(no point tracks returned)_", ""]


def render_trace_markdown(
    record: Mapping[str, Any], evidence_namespace: str | None = None
) -> list[str]:
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

        thought = _effective_thinking(step)
        if thought:
            lines += _quote(thought) + [""]

        arguments = action.get("arguments") or {}
        if name == "ground_with_qwen":
            lines += [
                "```python",
                f"ground_with_qwen(mode={arguments.get('mode')!r}, "
                f"request={arguments.get('request')!r},",
                f"                 t_src={arguments.get('t_src')}, "
                f"count={arguments.get('count')})",
                "```",
                "",
            ]
        elif name == "query_d4rt" and arguments.get("grounding_id"):
            lines += [
                "```python",
                f"query_d4rt(grounding_id={arguments.get('grounding_id')!r}, "
                f"point_ids={arguments.get('point_ids')},",
                f"           t_tgt={arguments.get('t_tgt')}, "
                f"t_cam={arguments.get('t_cam')})",
                "```",
                "",
            ]
        elif name == "query_d4rt":
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
        if found and call_id.startswith("qg_"):
            shown_id = _namespaced_id(call_id, evidence_namespace)
            cache_hit, reused = _grounding_cache_description(step, found)
            cache = "hit" if cache_hit else "miss"
            if reused:
                cache += f"; reused `{_namespaced_id(reused, evidence_namespace)}`"
            host = _host_provenance(found)
            original = host.get("original_t_src")
            source = f"sampled frame `{found.get('t_src')}`"
            if original is not None:
                source += f", original frame `{original}`"
            lines += [
                f"*Returned* `{shown_id}` — status `{found.get('status', '?')}`, "
                f"mode `{found.get('mode', '?')}`, cache {cache}.",
                "",
                f"*Request:* {found.get('request', '')}",
                "",
                f"*Source:* {source}.",
                "",
                f"`{_grounding_geometry(found)}`",
                "",
            ]
            raw_grounder = _grounding_raw_response(found)
            if raw_grounder:
                lines += [
                    "<details><summary>Raw grounding-model response</summary>",
                    "",
                    f"<pre><code>{html.escape(raw_grounder)}</code></pre>",
                    "",
                    "</details>",
                    "",
                ]
        elif found and call_id.startswith("d4rt_"):
            grounding_id = found.get("grounding_id")
            point_ids = found.get("point_ids")
            provenance = ""
            if isinstance(grounding_id, str):
                provenance = (
                    f" via `{_namespaced_id(grounding_id, evidence_namespace)}`"
                    + (f", points `{point_ids}`" if point_ids else "")
                )
            lines += [
                f"*Returned* `{_namespaced_id(call_id, evidence_namespace)}`{provenance}:",
                "",
                *_markdown_query_result(found),
                "",
            ]
        elif found and call_id.startswith("math_"):
            outputs = ", ".join(
                f"`{key} = {value}`" for key, value in (found.get("outputs") or {}).items()
            )
            lines += [
                f"*Returned* `{_namespaced_id(call_id, evidence_namespace)}`: {outputs}",
                "",
            ]
        if status != "ok":
            stage = step.get("failure_stage")
            stage_text = f" during `{stage}`" if stage else ""
            lines += [
                f"> ⛔ **Host rejected this step{stage_text}.** {_rejection(step)[:400]}",
                "",
            ]
        lines += _markdown_boundary_diagnostic(step)
        lines += ["---", ""]
    # The separator belongs *between* steps; the report supplies its own after
    # the closing </details>.
    while lines and lines[-1] in {"", "---"}:
        lines.pop()
    return lines + [""]


def render_trace(
    record: Mapping[str, Any],
    raw: bool = False,
    evidence_namespace: str | None = None,
) -> list[str]:
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
        prose = str(step.get("raw_qwen_response") or "")
        thought = prose if raw else _effective_thinking(step)
        if thought:
            lines.append(_wrap(thought))
        arguments = action.get("arguments") or {}
        if name == "ground_with_qwen":
            lines.append(
                f"    CALL  mode={arguments.get('mode')!r} "
                f"request={arguments.get('request')!r} "
                f"t_src={arguments.get('t_src')} count={arguments.get('count')}"
            )
        elif name == "query_d4rt" and arguments.get("grounding_id"):
            lines.append(
                f"    CALL  grounding_id="
                f"{_namespaced_id(str(arguments.get('grounding_id')), evidence_namespace)!r} "
                f"point_ids={arguments.get('point_ids')} "
                f"t_tgt={arguments.get('t_tgt')} t_cam={arguments.get('t_cam')}"
            )
        elif name == "query_d4rt":
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
        if found and call_id.startswith("qg_"):
            lines += _describe_grounding(found, call_id, step, evidence_namespace)
        elif found and call_id.startswith("d4rt_"):
            shown_id = _namespaced_id(call_id, evidence_namespace)
            grounding_id = found.get("grounding_id")
            point_ids = found.get("point_ids")
            suffix = ""
            if isinstance(grounding_id, str):
                suffix = f" via {_namespaced_id(grounding_id, evidence_namespace)}"
                if point_ids:
                    suffix += f" points={point_ids}"
            lines.append(f"    -> {shown_id}{suffix}")
            lines += _describe_query(found)
        elif found and call_id.startswith("math_"):
            lines.append(
                f"    -> {_namespaced_id(call_id, evidence_namespace)}  "
                f"{json.dumps(found.get('outputs', {}))}"
            )
        if step["status"] != "ok":
            stage = step.get("failure_stage")
            stage_text = f" [{stage}]" if stage else ""
            lines.append(f"    !! REJECTED{stage_text}: {_rejection(step)[:400]}")
        suffix = str(step.get("discarded_suffix", ""))
        if suffix.strip() and not raw:
            lines.append(
                f"    .. discarded suffix: {len(suffix)} chars, "
                f"{count_action_objects(suffix)} additional action(s); use --raw to inspect"
            )
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
    attempts = list(iter_attempts(record))
    for attempt_name, attempt in attempts:
        if len(attempts) > 1:
            print(f"{'-' * 38} {attempt_name.upper()} ATTEMPT {'-' * 44}")
        print("\n".join(render_trace(attempt, raw, evidence_namespace=attempt_name)))

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
