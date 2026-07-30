"""Assemble the DSI-Bench run into a reviewable markdown report.

Runs on CPU after the GPU passes::

    python -m d4rt_agent.dsi_bench_report

Copies each clip beside its result, renders the exact 32 frames the models saw as
both an animation and an indexed contact sheet, and lays ground truth next to the
agent's answer and the tool-free control's answer.

**Nothing here is scored.** No accuracy is computed and no letter is extracted
from the agent's prose, because the answers are meant to be judged by reading
them.  The one letter reported for the control is the tag its own prompt asks it
to emit, recorded verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from .dsi_bench_data import CATEGORY_NAMES, parse_choice_letter, read_manifest
from .dsi_bench_show import render_trace_markdown, trace_boundary_metrics
from .orchestration_audit import count_action_objects, iter_attempts
from .simple_v2_contracts import sample_video_cpu


DEFAULT_RESULTS_DIR = Path("d4rt_agent/results/dsi_bench")
REPORT_NAME = "DSI_BENCH_RESULTS.md"

GIF_WIDTH = 320
GIF_FRAME_MS = 150
SHEET_COLUMNS = 8
SHEET_TILE_WIDTH = 240
SHEET_JPEG_QUALITY = 88

# One grounding tile is one GIF frame wide, so the box can be checked against
# the animation at the same scale.
GROUNDING_TILE_WIDTH = GIF_WIDTH
GROUNDING_COLUMNS = 4
GROUNDING_CAPTION_PX = 20
GROUNDING_LEGEND_LINE_PX = 22

# BGR colours chosen to remain distinct on both bright and dark video frames.
GROUNDING_COLOURS = (
    (37, 215, 255),
    (255, 144, 30),
    (87, 230, 80),
    (220, 80, 220),
    (60, 80, 255),
    (235, 210, 70),
    (180, 120, 255),
    (70, 220, 190),
)


def _load_records(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["question_id"]] = record
    return records


def build_gif(frames: np.ndarray, destination: Path) -> None:
    """Animate the sampled frames so the motion is visible inline.

    Markdown renderers that strip <video> still render an animated GIF, and every
    question here is about motion, so a still frame would hide the evidence.
    """

    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    size = (GIF_WIDTH, max(1, round(height * GIF_WIDTH / width)))
    images = [Image.fromarray(frame).resize(size, Image.BILINEAR) for frame in frames]
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=True,
    )


def build_contact_sheet(frames: np.ndarray, destination: Path) -> None:
    """Lay the 32 frames out in an indexed grid.

    The index labels are the point: the agent's `t_src`, `t_tgt` and `t_cam`
    arguments are frame numbers, so a reviewer needs to see which image each of
    those refers to.
    """

    import cv2

    destination.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for index, frame in enumerate(frames):
        height, width = frame.shape[:2]
        tile_height = max(1, round(height * SHEET_TILE_WIDTH / width))
        tile = cv2.resize(frame, (SHEET_TILE_WIDTH, tile_height))
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        label = str(index)
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    tallest = max(tile.shape[0] for tile in tiles)
    tiles = [
        cv2.copyMakeBorder(tile, 0, tallest - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for tile in tiles
    ]
    rows = [
        np.hstack(tiles[start : start + SHEET_COLUMNS])
        for start in range(0, len(tiles), SHEET_COLUMNS)
    ]
    cv2.imwrite(str(destination), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY])


def _grounding_tile(frame: np.ndarray, query: Mapping[str, Any], call_id: str) -> np.ndarray:
    """One `query_d4rt` call: the frame it grounded on, with the box it drew."""

    import cv2

    height, width = frame.shape[:2]
    scale = GROUNDING_TILE_WIDTH / width
    tile = cv2.resize(frame, (GROUNDING_TILE_WIDTH, max(1, round(height * scale))))
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)

    x0, y0, x1, y1 = (round(value * scale) for value in query["bbox_pixel"])
    cv2.rectangle(tile, (x0, y0), (x1, y1), (0, 230, 0), 1, cv2.LINE_AA)

    # point_index 0 carries offset [0, 0] -- it is the centroid, and the rest are
    # the ensemble's seeded offsets around it.
    for point in query.get("query_points", []):
        u, v = (round(value * scale) for value in point["pixel_uv"])
        if point.get("point_index") == 0:
            cv2.circle(tile, (u, v), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(tile, (u, v), 4, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(tile, (u, v), 2, (0, 220, 255), 1, cv2.LINE_AA)

    index = str(query["t_src"])
    for colour, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(tile, index, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thickness, cv2.LINE_AA)

    tile = cv2.copyMakeBorder(
        tile, 0, GROUNDING_CAPTION_PX, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    caption = f"{call_id}  f{query['t_src']}  {str(query.get('label', ''))[:26]}"
    cv2.putText(
        tile,
        caption,
        (4, tile.shape[0] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return tile


def build_grounding_sheet(
    frames: np.ndarray, record: Mapping[str, Any] | None, destination: Path
) -> bool:
    """Every box the agent grounded, on the frame it grounded it on.

    The numeric trace says `bbox_2d_1000=[480, 300, 540, 620]`, which is not
    something a reviewer can check by eye.  Drawing it answers the question the
    trace cannot: was the agent actually looking at the object it named?
    """

    import cv2

    queries: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(record, Mapping):
        for attempt_name, attempt in iter_attempts(record):
            evidence = attempt.get("evidence")
            if not isinstance(evidence, Mapping):
                continue
            queries.extend(
                (f"{attempt_name}/{call_id}", value)
                for call_id, value in evidence.items()
                if isinstance(call_id, str)
                and call_id.startswith("d4rt_")
                and isinstance(value, Mapping)
                and value.get("bbox_pixel")
            )
    if not queries:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    tiles = [
        _grounding_tile(frames[value["t_src"]], value, call_id) for call_id, value in queries
    ]
    blank = np.zeros_like(tiles[0])
    while len(tiles) % GROUNDING_COLUMNS:
        tiles.append(blank)
    rows = [
        np.hstack(tiles[start : start + GROUNDING_COLUMNS])
        for start in range(0, len(tiles), GROUNDING_COLUMNS)
    ]
    cv2.imwrite(str(destination), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY])
    return True


def _host_provenance(grounding: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("host_provenance", "grounding_provenance", "provenance"):
        value = grounding.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _is_grounding_evidence(evidence_id: str, value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and (
            evidence_id.startswith("qg_")
            or (
                value.get("mode") in {"bbox", "points"}
                and isinstance(value.get("t_src"), int)
                and "request" in value
            )
        )
    )


def iter_grounding_evidence(
    record: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return every immutable qg record, namespaced by execution attempt.

    Strict and relaxed retries both start their evidence counters at ``qg_1``.
    The namespace therefore is part of the review identity even for a normal
    one-attempt record, where the only namespace is ``strict``.
    """

    rows: list[dict[str, Any]] = []
    if not isinstance(record, Mapping):
        return rows
    for attempt_name, attempt in iter_attempts(record):
        evidence = attempt.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        trace = attempt.get("trace")
        steps = trace if isinstance(trace, list) else []
        for evidence_id, value in evidence.items():
            if not isinstance(evidence_id, str) or not _is_grounding_evidence(evidence_id, value):
                continue
            uses = [
                step
                for step in steps
                if isinstance(step, Mapping)
                and (
                    step.get("call_id") == evidence_id
                    or step.get("reused_evidence_id") == evidence_id
                    or (
                        isinstance(step.get("result"), Mapping)
                        and step["result"].get("reused_evidence_id") == evidence_id
                    )
                )
            ]
            rows.append({
                "attempt": attempt_name,
                "evidence_id": evidence_id,
                "display_id": f"{attempt_name}/{evidence_id}",
                "grounding": value,
                "steps": uses,
                "attempt_record": attempt,
            })
    return rows


def _grounding_colour(display_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(display_id.encode("utf-8")).digest()
    return GROUNDING_COLOURS[int.from_bytes(digest[:2], "big") % len(GROUNDING_COLOURS)]


def _valid_source_frame(value: Any, frame_count: int) -> int | None:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < frame_count
    ):
        return value
    return None


def _draw_qwen_grounding_frame(
    frame: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    sampled_frame: int,
) -> np.ndarray:
    """Draw every Phase 2 grounding made on one exact sampled RGB frame."""

    import cv2

    canvas = cv2.cvtColor(np.asarray(frame).copy(), cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]
    thickness = max(2, round(min(width, height) / 300))
    radius = max(4, round(min(width, height) / 120))
    font_scale = max(0.45, min(0.75, width / 1200))

    legend: list[tuple[str, tuple[int, int, int]]] = []
    for row in rows:
        grounding = row["grounding"]
        display_id = str(row["display_id"])
        colour = _grounding_colour(display_id)
        status = str(grounding.get("status", "?"))
        mode = str(grounding.get("mode", "?"))
        request = " ".join(str(grounding.get("request", "")).split())
        legend.append(
            (
                f"{display_id}  {mode}  {status}  {request}",
                colour,
            )
        )

        if status != "ok":
            continue
        if mode == "bbox":
            box = grounding.get("bbox_2d_1000")
            if (
                isinstance(box, list)
                and len(box) == 4
                and all(isinstance(value, (int, float)) for value in box)
            ):
                x0 = round(float(box[0]) * max(width - 1, 1) / 1000.0)
                y0 = round(float(box[1]) * max(height - 1, 1) / 1000.0)
                x1 = round(float(box[2]) * max(width - 1, 1) / 1000.0)
                y1 = round(float(box[3]) * max(height - 1, 1) / 1000.0)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thickness, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    display_id,
                    (max(2, x0), max(18, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    colour,
                    thickness,
                    cv2.LINE_AA,
                )
        elif mode == "points":
            points = grounding.get("points_2d_1000")
            if not isinstance(points, list):
                continue
            for point_index, point in enumerate(points, start=1):
                if not isinstance(point, Mapping):
                    continue
                xy = point.get("xy")
                if (
                    not isinstance(xy, list)
                    or len(xy) != 2
                    or not all(isinstance(value, (int, float)) for value in xy)
                ):
                    continue
                u = round(float(xy[0]) * max(width - 1, 1) / 1000.0)
                v = round(float(xy[1]) * max(height - 1, 1) / 1000.0)
                point_id = str(point.get("point_id") or f"p{point_index}")
                cv2.circle(canvas, (u, v), radius, colour, -1, cv2.LINE_AA)
                cv2.circle(canvas, (u, v), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    f"{display_id}/{point_id}",
                    (u + radius + 3, max(18, v - radius)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    colour,
                    thickness,
                    cv2.LINE_AA,
                )

    # Include not_found entries in the same legend even though there is no fake
    # geometry to draw. Limit text to the image width without dropping identity,
    # mode, or status.
    max_chars = max(30, round(width / max(5.0, 8.0 * font_scale)))
    legend = [(text[:max_chars], colour) for text, colour in legend]
    legend_height = GROUNDING_LEGEND_LINE_PX * (len(legend) + 1) + 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, min(height, legend_height)), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0)
    cv2.putText(
        canvas,
        f"sampled frame {sampled_frame}",
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for line_index, (text, colour) in enumerate(legend, start=1):
        cv2.putText(
            canvas,
            text,
            (8, 19 + line_index * GROUNDING_LEGEND_LINE_PX),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _write_grounding_contact_sheet(images: Sequence[np.ndarray], destination: Path) -> None:
    import cv2

    if not images:
        return
    tiles: list[np.ndarray] = []
    for image in images:
        height, width = image.shape[:2]
        tile_height = max(1, round(height * GROUNDING_TILE_WIDTH / width))
        tiles.append(cv2.resize(image, (GROUNDING_TILE_WIDTH, tile_height)))
    tallest = max(tile.shape[0] for tile in tiles)
    tiles = [
        cv2.copyMakeBorder(tile, 0, tallest - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for tile in tiles
    ]
    blank = np.zeros_like(tiles[0])
    while len(tiles) % GROUNDING_COLUMNS:
        tiles.append(blank)
    rows = [
        np.hstack(tiles[start : start + GROUNDING_COLUMNS])
        for start in range(0, len(tiles), GROUNDING_COLUMNS)
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY])


def build_qwen_grounding_media(
    frames: np.ndarray,
    record: Mapping[str, Any] | None,
    *,
    question_id: str,
    frames_directory: Path,
    sheet_destination: Path,
    refresh: bool = False,
) -> list[Path]:
    """Render combined exact-frame images and a per-question contact sheet."""

    import cv2

    rows = iter_grounding_evidence(record)
    prefix = f"{question_id}_f"
    frames_directory.mkdir(parents=True, exist_ok=True)
    if refresh:
        for old_path in frames_directory.iterdir():
            if old_path.is_file() and old_path.name.startswith(prefix) and old_path.suffix == ".jpg":
                old_path.unlink()

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        t_src = _valid_source_frame(row["grounding"].get("t_src"), len(frames))
        if t_src is not None:
            grouped.setdefault(t_src, []).append(row)
    if not grouped:
        if refresh and sheet_destination.exists():
            sheet_destination.unlink()
        return []

    paths: list[Path] = []
    rendered: list[np.ndarray] = []
    for t_src, frame_rows in sorted(grouped.items()):
        image = _draw_qwen_grounding_frame(frames[t_src], frame_rows, t_src)
        destination = frames_directory / f"{question_id}_f{t_src:02d}.jpg"
        if refresh or not destination.exists():
            cv2.imwrite(
                str(destination),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY],
            )
        paths.append(destination)
        rendered.append(image)
    if refresh or not sheet_destination.exists():
        _write_grounding_contact_sheet(rendered, sheet_destination)
    return paths


def _baseline_letter(record: Mapping[str, Any] | None, entry: Mapping[str, Any]) -> str | None:
    """Re-derive the control's letter from its raw reply.

    Parsing at read time rather than trusting the stored field keeps old runs
    readable after the extractor improves -- the raw text is the record of what
    the model said, and the letter is only an interpretation of it.
    """

    if not record or record.get("status") != "complete":
        return None
    return parse_choice_letter(record.get("raw_response", ""), entry["option_letters"])


def _final_text(record: Mapping[str, Any] | None) -> str:
    if not record:
        return "_(not run)_"
    if record.get("status") in {"failed", "error"}:
        return f"_({record['status']}: {str(record.get('error', ''))[:160]})_"
    answer = record.get("final_answer") or {}
    return str(answer.get("text") or answer.get("value") or "_(no answer)_")


def _first_clause(text: str, limit: int = 64) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = line.replace("|", "\\|")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _trace_rows(record: Mapping[str, Any]) -> list[str]:
    rows = []
    for entry in record.get("trace", []):
        action = entry.get("parsed_action") or {}
        arguments = action.get("arguments") or {}
        name = action.get("action", "—")
        target = arguments.get("t_tgt")
        suffix = str(entry.get("discarded_suffix", ""))
        if suffix.strip():
            extra_actions = count_action_objects(suffix)
            suffix_text = f"yes ({extra_actions} action{'s' if extra_actions != 1 else ''})"
        else:
            suffix_text = "no"
        subject = (
            arguments.get("request")
            or arguments.get("grounding_id")
            or arguments.get("label", "")
        )
        t_src = arguments.get("t_src")
        if t_src is None and name == "query_d4rt" and arguments.get("grounding_id"):
            t_src = "from qg"
        rows.append(
            "| {step} | {status} | {stage} | {name} | {label} | {t_src} | {t_tgt} | "
            "{t_cam} | {suffix} | {why} |".format(
                step=entry.get("step", ""),
                status=entry.get("status", ""),
                stage=entry.get("failure_stage") or "—",
                name=name,
                label=str(subject)[:28].replace("|", "\\|"),
                t_src=t_src if t_src is not None else "",
                t_tgt=(f"{target[0]}…{target[-1]} ({len(target)})" if isinstance(target, list) and len(target) > 3
                       else (target if target is not None else "")),
                t_cam=arguments.get("t_cam", ""),
                suffix=suffix_text,
                why=str(arguments.get("justification", entry.get("error", "")))[:90].replace("|", "\\|"),
            )
        )
    return rows


def _grounding_cache_summary(row: Mapping[str, Any]) -> str:
    grounding = row["grounding"]
    host = _host_provenance(grounding)
    hit_steps = [
        step
        for step in row["steps"]
        if step.get("cache_hit") is True
        or (
            isinstance(step.get("result"), Mapping)
            and step["result"].get("cache_hit") is True
        )
    ]
    if hit_steps:
        return f"created once; cache hit on {len(hit_steps)} later call(s)"
    if grounding.get("cache_hit") is True or host.get("cache_hit") is True:
        return "cache hit"
    return "cache miss"


def _d4rt_grounding_uses(row: Mapping[str, Any]) -> list[str]:
    evidence_id = row["evidence_id"]
    attempt = row["attempt_record"]
    evidence = attempt.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    uses: list[str] = []
    for step in attempt.get("trace", []):
        if not isinstance(step, Mapping):
            continue
        action = step.get("parsed_action")
        if not isinstance(action, Mapping) or action.get("action") != "query_d4rt":
            continue
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping) or arguments.get("grounding_id") != evidence_id:
            continue
        call_id = step.get("call_id")
        result = evidence.get(call_id) if isinstance(call_id, str) else None
        point_ids = arguments.get("point_ids")
        if isinstance(result, Mapping) and result.get("point_ids"):
            point_ids = result.get("point_ids")
        target = arguments.get("t_tgt")
        uses.append(
            f"{row['attempt']}/{call_id or 'rejected'}"
            f" (points {point_ids if point_ids else 'all'}, t_tgt={target}, "
            f"t_cam={arguments.get('t_cam')})"
        )
    return uses


def _markdown_grounding_geometry(grounding: Mapping[str, Any]) -> str:
    if grounding.get("mode") == "bbox":
        return f"`bbox_2d_1000={grounding.get('bbox_2d_1000')}`"
    points = grounding.get("points_2d_1000")
    if not isinstance(points, list):
        return "`points_2d_1000=[]`"
    values = []
    for index, point in enumerate(points, start=1):
        if not isinstance(point, Mapping):
            continue
        point_id = point.get("point_id") or f"p{index}"
        description = str(point.get("description", "")).replace("|", "\\|")
        values.append(f"`{point_id}={point.get('xy')}` ({description or 'no description'})")
    return "; ".join(values) or "`points_2d_1000=[]`"


def _grounding_details(
    record: Mapping[str, Any] | None,
    question_id: str,
    video_slug: str,
) -> list[str]:
    rows = iter_grounding_evidence(record)
    if not rows:
        return []
    lines = [
        f"<details><summary>Qwen grounding evidence ({len(rows)} immutable groundings)</summary>",
        "",
        "Grounding IDs are prefixed with their attempt because strict and relaxed retries "
        "have independent evidence registries.",
        "",
    ]
    source_frames: list[int] = []
    for row in rows:
        grounding = row["grounding"]
        display_id = row["display_id"]
        host = _host_provenance(grounding)
        t_src = grounding.get("t_src")
        if isinstance(t_src, int) and not isinstance(t_src, bool):
            source_frames.append(t_src)
        original = host.get("original_t_src")
        source = f"sampled frame `{t_src}`"
        if original is not None:
            source += f" / original frame `{original}`"
        requested_count = grounding.get("requested_count")
        returned_count = grounding.get("returned_count")
        count = ""
        if requested_count is not None:
            count = f"; requested `{requested_count}`, returned `{returned_count}`"
        uses = _d4rt_grounding_uses(row)
        lines += [
            f"### `{display_id}`",
            "",
            f"- Request: {grounding.get('request', '')}",
            f"- Result: status `{grounding.get('status', '?')}`, mode "
            f"`{grounding.get('mode', '?')}`{count}",
            f"- Source: {source}",
            f"- Cache: {_grounding_cache_summary(row)}",
            f"- Parsed geometry: {_markdown_grounding_geometry(grounding)}",
            (
                "- D4RT provenance: " + "; ".join(f"`{value}`" for value in uses)
                if uses
                else "- D4RT provenance: not queried"
            ),
            "",
        ]
        raw_grounder = str(
            host.get("raw_grounder_response", grounding.get("raw_grounder_response", ""))
            or ""
        )
        if raw_grounder:
            lines += [
                "<details><summary>Raw grounding-model response</summary>",
                "",
                f"<pre><code>{html.escape(raw_grounder)}</code></pre>",
                "",
                "</details>",
                "",
            ]
    lines += [
        "**Combined exact-source overlays.** Boxes and points come only from "
        "`ground_with_qwen`; no legacy agent-authored boxes are drawn.",
        "",
    ]
    for t_src in sorted(set(source_frames)):
        lines += [
            f"![{question_id} grounding frame {t_src}]"
            f"(grounding_frames/{question_id}_f{t_src:02d}.jpg)",
            "",
        ]
    lines += [
        f"![grounding contact sheet](groundings/{video_slug}.jpg)",
        "",
        "</details>",
        "",
    ]
    return lines


def _header(manifest: Mapping[str, Any], agent: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    sampling = manifest["sampling"]
    counts = manifest["counts"]
    sample_agent = next(iter(agent.values()), {})
    sample_baseline = next(iter(baseline.values()), {})
    lines = [
        "# DSI-Bench evaluation — D4RT agent vs. tool-free Qwen",
        "",
        "25 questions from DSI-Bench (`std` split), sampled evenly across the benchmark's",
        "5 source datasets and 6 motion-reasoning tasks. Each row shows the clip, the",
        "question, the dataset's ground-truth answer, the D4RT agent's answer, and the same",
        "Qwen model answering from the same 32 frames with no tools.",
        "",
        "**These answers are not auto-scored.** The agent replies in prose so its reasoning",
        "stays visible, and the comparison against ground truth is meant to be read.",
        "",
        "## Run configuration",
        "",
        "| | |",
        "| --- | --- |",
        f"| Questions | {len(manifest['questions'])} |",
        f"| Sampling design | {sampling['design']} — {sampling['description']} |",
        f"| Sampling seed | `{sampling['seed']}` (requested `{sampling['requested_seed']}`) |",
        f"| Source CSV | `{manifest['csv_path']}` |",
        f"| CSV sha256 | `{manifest['csv_sha256'][:16]}…` |",
        f"| Ground-truth histogram | {counts['gt_histogram']} |",
        f"| Per dataset | {counts['per_dataset']} |",
        f"| Per task | {counts['per_category']} |",
    ]
    if sample_agent:
        alignment = sample_agent.get("benchmark_alignment", {})
        lines += [
            f"| Point mode | `{sample_agent.get('point_mode', '?')}` |",
            f"| Position scale | `{alignment.get('type', '?')}` (scale {alignment.get('scale', '?')}) |",
        ]
    if sample_baseline:
        lines.append(f"| Qwen model | `{sample_baseline.get('qwen_model', '?')}` |")
    statuses: dict[str, int] = {}
    for record in agent.values():
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    if statuses:
        lines.append(f"| Agent run status | {statuses} |")
        tracker_backed = sum(1 for record in agent.values() if record.get("d4rt_used"))
        lines.append(f"| Answers citing D4RT | {tracker_backed} / {len(agent)} |")
        boundary = {
            "turns": 0,
            "discarded_suffixes": 0,
            "discarded_suffixes_with_action": 0,
        }
        for record in agent.values():
            for _, attempt in iter_attempts(record):
                metrics = trace_boundary_metrics(attempt)
                for key in boundary:
                    boundary[key] += metrics[key]
        lines += [
            f"| Model turns audited | {boundary['turns']} |",
            (
                "| Turns with discarded suffixes | "
                f"{boundary['discarded_suffixes']} / {boundary['turns']} |"
            ),
            (
                "| Discarded suffixes containing another action | "
                f"{boundary['discarded_suffixes_with_action']} |"
            ),
        ]
    lines += [
        "",
        "> **Read this as a diagnostic, not a benchmark score.** The sample is balanced by",
        "> design — each task carries 4–5 questions, while the full benchmark is 33%",
        "> *Obj:moving cam* and 31% *Cam:dynamic scene* — so it is built to show where the",
        "> tracker helps and where it does not, not to estimate DSI-Bench accuracy. With 25",
        "> four-way questions, chance alone lands around 6 correct.",
        "",
        "> **Positions carry no absolute scale.** DSI-Bench ships no ground-truth geometry, so",
        "> the agent's numbers are in consistent but arbitrary units. Signs, ratios and",
        "> directions are meaningful; magnitudes are not metres.",
        "",
    ]
    return lines


def _index_table(
    manifest: Mapping[str, Any], agent: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[str]:
    lines = [
        "## At a glance",
        "",
        "| # | Dataset | Task | GT | Agent answer | Qwen-only | D4RT |",
        "| ---: | --- | --- | :---: | --- | :---: | :---: |",
    ]
    for index, entry in enumerate(manifest["questions"], start=1):
        question_id = entry["question_id"]
        agent_record = agent.get(question_id)
        baseline_record = baseline.get(question_id)
        gt_text = f"**{entry['gt']}**"
        agent_text = _first_clause(_final_text(agent_record))
        letter = _baseline_letter(baseline_record, entry)
        if letter:
            baseline_text = f"**{letter}**" if letter == entry["gt"] else letter
        elif baseline_record and baseline_record.get("status") == "complete":
            baseline_text = "?"
        elif baseline_record:
            baseline_text = "err"
        else:
            baseline_text = "—"
        used = agent_record.get("d4rt_used") if agent_record else None
        lines.append(
            f"| [{index}](#{index}-{entry['dataset'].lower()}-task-{entry['cate']}) "
            f"| {entry['dataset']} | {CATEGORY_NAMES[entry['cate']]} | {gt_text} "
            f"| {agent_text} | {baseline_text} | {'yes' if used else 'no' if agent_record else '—'} |"
        )
    lines.append("")
    return lines


def _question_section(
    index: int,
    entry: Mapping[str, Any],
    agent_record: Mapping[str, Any] | None,
    baseline_record: Mapping[str, Any] | None,
) -> list[str]:
    slug = entry["video_slug"]
    lines = [
        f"## {index}. {entry['dataset']} — task {entry['cate']}: {CATEGORY_NAMES[entry['cate']]}",
        "",
        f"![clip {index}](gifs/{slug}.gif)",
        "",
        f"[▶ full-quality MP4](videos/{slug}.mp4) · source: `{entry['relative_path']}`",
        "",
        f'<video controls width="480" src="videos/{slug}.mp4"></video>',
        "",
        f"**Question.** {entry['question']}",
        "",
    ]
    for letter in entry["option_letters"]:
        marker = " ← **ground truth**" if letter == entry["gt"] else ""
        lines.append(f"- **{letter}.** {entry['options'][letter]}{marker}")
    lines += ["", "**Agent answer.**", "", f"> {_final_text(agent_record).strip()}", ""]

    if agent_record:
        answer = agent_record.get("final_answer") or {}
        limitations = str(answer.get("limitations", "")).strip()
        if limitations:
            lines += [f"*Agent-stated limitations:* {limitations}", ""]
        if agent_record.get("status") == "complete_relaxed":
            lines += [
                "> ⚠️ This answer was produced on the relaxed retry: the agent could not reach",
                "> an answer while required to cite a D4RT measurement, so it is not",
                "> tracker-backed.",
                "",
            ]

    if baseline_record:
        letter = _baseline_letter(baseline_record, entry)
        raw = str(baseline_record.get("raw_response", baseline_record.get("error", ""))).strip()
        if letter:
            verdict = "matches ground truth" if letter == entry["gt"] else f"ground truth is {entry['gt']}"
            shown = f"**{letter}** — {entry['options'].get(letter, '?')} ({verdict}). Raw reply: `{raw[:80]}`"
        else:
            shown = f"_no letter could be read from the reply:_ `{raw[:200]}`"
        lines += ["**Qwen-only baseline.**", "", f"> {shown}", ""]

    if agent_record:
        attempts = list(iter_attempts(agent_record))
        for attempt_name, attempt in attempts:
            if not attempt.get("trace"):
                continue
            rows = _trace_rows(attempt)
            boundary = trace_boundary_metrics(attempt)
            label = (
                "Agent"
                if len(attempts) == 1
                else f"{attempt_name.capitalize()} attempt"
            )
            timing = (
                f", {agent_record.get('wall_seconds', 0):.0f}s"
                if len(attempts) == 1 else ""
            )
            lines += [
                f"<details><summary>{label} trace "
                f"({len(rows)} steps{timing}; "
                f"{boundary['discarded_suffixes']} discarded suffixes, "
                f"{boundary['discarded_suffixes_with_action']} with another action)"
                "</summary>",
                "",
                "| Step | Status | Stage | Action | Label | t_src | t_tgt | t_cam | "
                "Discarded suffix | Justification |",
                "| ---: | --- | --- | --- | --- | ---: | --- | ---: | --- | --- |",
                *rows,
                "",
                "</details>",
                "",
            ]
            # The table above says what the agent did; this says why it thought
            # so, based on the effective model-visible response.
            lines += [
                f"<details><summary>{label} thinking log "
                f"({len(rows)} steps)</summary>",
                "",
                *render_trace_markdown(attempt, evidence_namespace=attempt_name),
                "</details>",
                "",
            ]
    qwen_groundings = iter_grounding_evidence(agent_record)
    if qwen_groundings:
        lines += _grounding_details(agent_record, entry["question_id"], slug)
    grounded = []
    if not qwen_groundings and agent_record:
        for attempt_name, attempt in iter_attempts(agent_record):
            grounded.extend(
                f"{attempt_name}/{call_id}"
                for call_id in attempt.get("evidence", {})
                if call_id.startswith("d4rt_")
            )
    if grounded:
        lines += [
            f"<details><summary>Grounded objects ({len(grounded)} query_d4rt calls)</summary>",
            "",
            f"![groundings {index}](groundings/{slug}.jpg)",
            "",
            "Each tile is one `query_d4rt` call, drawn on its `t_src` frame: green box as the "
            "agent placed it, red dot the centroid, yellow dots the four other `ensemble5` "
            "sample points. The number top-left is the sampled frame index.",
            "",
            "</details>",
            "",
        ]
    lines += [
        "<details><summary>The 32 sampled frames the models saw</summary>",
        "",
        f"![contact sheet {index}](contact_sheets/{slug}.jpg)",
        "",
        "</details>",
        "",
        "---",
        "",
    ]
    return lines


def build_report(
    results_dir: Path,
    skip_media: bool = False,
    refresh_groundings: bool = False,
) -> Path:
    results_dir = Path(results_dir)
    manifest = read_manifest(results_dir / "manifest.json")
    agent = _load_records(results_dir / "answers")
    baseline = _load_records(results_dir / "baseline")

    lines = _header(manifest, agent, baseline)
    lines += _index_table(manifest, agent, baseline)
    lines += ["## Questions", ""]

    for index, entry in enumerate(manifest["questions"], start=1):
        slug = entry["video_slug"]
        question_id = entry["question_id"]
        agent_record = agent.get(question_id)
        source = Path(entry["video_path"])
        destination = results_dir / "videos" / f"{slug}.mp4"
        gif_path = results_dir / "gifs" / f"{slug}.gif"
        sheet_path = results_dir / "contact_sheets" / f"{slug}.jpg"
        grounding_path = results_dir / "groundings" / f"{slug}.jpg"
        grounding_frames_dir = results_dir / "grounding_frames"
        qwen_groundings = iter_grounding_evidence(agent_record)
        expected_qg_paths = [
            grounding_frames_dir / f"{question_id}_f{t_src:02d}.jpg"
            for t_src in sorted({
                row["grounding"].get("t_src")
                for row in qwen_groundings
                if isinstance(row["grounding"].get("t_src"), int)
                and not isinstance(row["grounding"].get("t_src"), bool)
            })
        ]
        needs_standard = not skip_media and (
            not gif_path.exists() or not sheet_path.exists()
        )
        needs_qg = bool(qwen_groundings) and (
            refresh_groundings
            or (
                not skip_media
                and (
                    not grounding_path.exists()
                    or any(not path.exists() for path in expected_qg_paths)
                )
            )
        )
        needs_legacy_grounding = not qwen_groundings and (
            refresh_groundings or (not skip_media and not grounding_path.exists())
        )
        if needs_standard or needs_qg or needs_legacy_grounding:
            if not skip_media:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(source, destination)
            sampling_source = destination if destination.exists() else source
            sampled = sample_video_cpu(sampling_source)
            if needs_standard:
                if not gif_path.exists():
                    build_gif(sampled.frames_rgb, gif_path)
                if not sheet_path.exists():
                    build_contact_sheet(sampled.frames_rgb, sheet_path)
            if needs_qg:
                build_qwen_grounding_media(
                    sampled.frames_rgb,
                    agent_record,
                    question_id=question_id,
                    frames_directory=grounding_frames_dir,
                    sheet_destination=grounding_path,
                    refresh=refresh_groundings,
                )
            elif needs_legacy_grounding:
                if refresh_groundings and grounding_path.exists():
                    grounding_path.unlink()
                build_grounding_sheet(sampled.frames_rgb, agent_record, grounding_path)
            del sampled
        if not skip_media or refresh_groundings:
            # Refresh only touches grounding artifacts. Videos, GIFs and ordinary
            # contact sheets are neither rebuilt nor required by that operation.
            media_label = "groundings refreshed" if refresh_groundings else "media ready"
            print(
                f"[{index:2d}/{len(manifest['questions'])}] {media_label} for {slug[:48]}",
                flush=True,
            )
        lines += _question_section(
            index, entry, agent_record, baseline.get(question_id)
        )

    report_path = results_dir / REPORT_NAME
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--skip-media", action="store_true", help="rebuild only the markdown, reusing existing media"
    )
    parser.add_argument(
        "--refresh-groundings",
        action="store_true",
        help=(
            "rebuild grounding frame overlays and grounding contact sheets; "
            "leave videos, GIFs, and ordinary contact sheets untouched"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    path = build_report(
        args.results_dir,
        skip_media=args.skip_media,
        refresh_groundings=args.refresh_groundings,
    )
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
