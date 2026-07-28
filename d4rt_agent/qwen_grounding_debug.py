"""Probe Qwen's object grounding independently of the D4RT tool loop.

The saved DSI-Bench traces contain the label, source frame, and bounding box
chosen for every ``query_d4rt`` action.  This module replays only the grounding
part: Qwen sees one source image and one fixed detection prompt, with no video
question, tool schema, D4RT result, or conversation history.

Repeated trace requests with the same ``(question, t_src, label)`` are evaluated
once because greedy decoding with a reset seed is deterministic.  The result is
then mapped back to every original trace occurrence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from .simple_v2 import DEFAULT_QWEN_MODEL, OfflineQwen
from .simple_v2_contracts import sample_video_cpu


DEFAULT_ANSWERS_DIR = Path("d4rt_agent/results/dsi_bench/answers")
DEFAULT_RESULTS_DIR = Path("d4rt_agent/results/qwen_grounding_debug")
PROMPT_TEMPLATE = """Ground the target named "{label}" in this image inside one tight bounding box.

Return exactly one JSON object and no other text:
{{"bbox_2d_1000":[x_min,y_min,x_max,y_max]}}

Follow the agent's coordinate convention: normalize every bounding-box coordinate
to an integer from 0 to 1000, with origin at the top-left of the image. If the
named target is not visible, return {{"bbox_2d_1000":null}}."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _probe_id(question_id: str, t_src: int, label: str) -> str:
    digest = hashlib.sha256(
        f"{question_id}\0{t_src}\0{label}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{question_id}_f{t_src:02d}_{digest}"


def _as_box(value: Any) -> list[float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError("bbox_2d_1000 must be null or four finite numbers")
    box = [float(item) for item in value]
    if not all(0.0 <= item <= 1000.0 for item in box):
        raise ValueError("bbox_2d_1000 coordinates must lie in [0,1000]")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox_2d_1000 must have positive width and height")
    return box


def parse_detection(raw: str) -> list[float] | None:
    """Parse the first JSON object containing exactly ``bbox_2d_1000``."""

    decoder = json.JSONDecoder()
    for offset, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == {"bbox_2d_1000"}:
            return _as_box(value["bbox_2d_1000"])
    raise ValueError(f"no strict bbox JSON object found in response: {raw[:300]!r}")


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_distance_1000(a: Sequence[float], b: Sequence[float]) -> float:
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    return math.hypot(ax - bx, ay - by)


def _d4rt_visibility(trace_row: Mapping[str, Any], t_src: int) -> dict[str, Any]:
    result = trace_row.get("result")
    if not isinstance(result, Mapping):
        return {
            "requested_targets": 0,
            "visible_targets": 0,
            "source_frame_requested": False,
            "source_frame_visible": None,
        }
    predictions = result.get("predictions")
    if not isinstance(predictions, list):
        predictions = []
    visible_targets = sum(
        1 for prediction in predictions
        if isinstance(prediction, Mapping) and prediction.get("visible") is True
    )
    source_predictions = [
        prediction for prediction in predictions
        if isinstance(prediction, Mapping) and prediction.get("sampled_frame_index") == t_src
    ]
    return {
        "requested_targets": len(predictions),
        "visible_targets": visible_targets,
        "source_frame_requested": bool(source_predictions),
        "source_frame_visible": (
            any(prediction.get("visible") is True for prediction in source_predictions)
            if source_predictions else None
        ),
    }


def extract_requests(answers_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return all trace occurrences and unique image/label probes."""

    occurrences: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for answer_path in sorted(answers_dir.glob("*.json")):
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        question_id = str(answer["question_id"])
        sampling = answer.get("sampling") or {}
        video_path = sampling.get("video")
        if not isinstance(video_path, str):
            raise ValueError(f"{answer_path}: missing sampling.video")
        for row in answer.get("trace") or []:
            action = row.get("parsed_action") or {}
            if action.get("action") != "query_d4rt":
                continue
            arguments = action.get("arguments") or {}
            label = arguments.get("label")
            t_src = arguments.get("t_src")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{answer_path}: trace step {row.get('step')} has no label")
            if not isinstance(t_src, int):
                raise ValueError(f"{answer_path}: trace step {row.get('step')} has invalid t_src")
            agent_box = _as_box(arguments.get("bbox_2d_1000"))
            if agent_box is None:
                raise ValueError(f"{answer_path}: trace grounding box cannot be null")
            occurrence = {
                "occurrence_id": f"{question_id}:step_{int(row['step']):02d}",
                "question_id": question_id,
                "dataset": answer.get("dataset"),
                "category_name": answer.get("category_name"),
                "answer_status": answer.get("status"),
                "agent_step": int(row["step"]),
                "trace_status": row.get("status"),
                "label": label,
                "t_src": t_src,
                "original_frame": next(
                    (
                        item.get("original_frame")
                        for item in sampling.get("sampled_to_original") or []
                        if item.get("sampled_frame") == t_src
                    ),
                    None,
                ),
                "agent_bbox_2d_1000": agent_box,
                **_d4rt_visibility(row, t_src),
            }
            occurrences.append(occurrence)
            grouped[(question_id, t_src, label)].append(occurrence)

    probes: list[dict[str, Any]] = []
    answer_lookup = {
        json.loads(path.read_text(encoding="utf-8"))["question_id"]:
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(answers_dir.glob("*.json"))
    }
    for (question_id, t_src, label), members in sorted(grouped.items()):
        answer = answer_lookup[question_id]
        boxes = []
        for member in members:
            box = member["agent_bbox_2d_1000"]
            if box not in boxes:
                boxes.append(box)
        probes.append({
            "probe_id": _probe_id(question_id, t_src, label),
            "question_id": question_id,
            "dataset": answer.get("dataset"),
            "category_name": answer.get("category_name"),
            "video_path": answer["sampling"]["video"],
            "sampling": answer["sampling"],
            "label": label,
            "t_src": t_src,
            "original_frame": members[0]["original_frame"],
            "agent_bboxes_2d_1000": boxes,
            "occurrence_ids": [member["occurrence_id"] for member in members],
        })
    return occurrences, probes


def _probe_metrics(agent_boxes: Sequence[Sequence[float]], detected: Sequence[float] | None) -> dict[str, Any]:
    comparisons = []
    if detected is not None:
        comparisons = [
            {
                "agent_bbox_2d_1000": list(agent_box),
                "iou": box_iou(agent_box, detected),
                "center_distance_1000": center_distance_1000(agent_box, detected),
            }
            for agent_box in agent_boxes
        ]
    return {
        "comparisons": comparisons,
        "max_iou": max((item["iou"] for item in comparisons), default=None),
        "min_center_distance_1000": min(
            (item["center_distance_1000"] for item in comparisons), default=None
        ),
    }


def _draw_combined_frame(
    frame: Any,
    items: Sequence[tuple[Mapping[str, Any], Sequence[float] | None]],
    path: Path,
) -> None:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    colors = [
        (255, 0, 255),
        (0, 220, 255),
        (255, 165, 0),
        (50, 255, 50),
        (255, 80, 80),
        (160, 100, 255),
        (255, 255, 0),
        (0, 180, 120),
    ]

    def pixels(box: Sequence[float]) -> tuple[float, float, float, float]:
        return (
            float(box[0]) * width / 1000.0,
            float(box[1]) * height / 1000.0,
            float(box[2]) * width / 1000.0,
            float(box[3]) * height / 1000.0,
        )

    missing = []
    for index, (probe, detected) in enumerate(items):
        color = colors[index % len(colors)]
        if detected is None:
            missing.append(str(probe["label"]))
            continue
        box_px = pixels(detected)
        draw.rectangle(box_px, outline=color, width=3)
        label = str(probe["label"])
        text_box = draw.textbbox((0, 0), label)
        label_x = max(0.0, min(box_px[0], width - text_box[2] - 6))
        label_y = max(0.0, box_px[1] - text_box[3] - 6)
        draw.rectangle(
            (label_x, label_y, label_x + text_box[2] + 6, label_y + text_box[3] + 4),
            fill=(0, 0, 0),
        )
        draw.text((label_x + 3, label_y + 2), label, fill=color)
    if missing:
        banner = "null: " + ", ".join(missing)
        text_box = draw.textbbox((0, 0), banner)
        banner_y = max(0, height - text_box[3] - 6)
        draw.rectangle((0, banner_y, min(width, text_box[2] + 8), height), fill=(0, 0, 0))
        draw.text((4, banner_y + 2), banner, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92)


def _frame_output_name(question_id: str, t_src: int) -> str:
    return f"{question_id}_f{t_src:02d}.jpg"


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_summary(
    results_dir: Path,
    occurrences: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]],
) -> None:
    detection_dir = results_dir / "detections"
    records = []
    for probe in probes:
        path = detection_dir / f"{probe['probe_id']}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))

    complete = [record for record in records if record.get("status") == "complete"]
    boxes = [record for record in complete if record.get("detected_bbox_2d_1000") is not None]
    nulls = [record for record in complete if record.get("detected_bbox_2d_1000") is None]
    errors = [record for record in records if record.get("status") != "complete"]
    ious = [
        float(record["metrics"]["max_iou"])
        for record in boxes
        if record.get("metrics", {}).get("max_iou") is not None
    ]
    centers = [
        float(record["metrics"]["min_center_distance_1000"])
        for record in boxes
        if record.get("metrics", {}).get("min_center_distance_1000") is not None
    ]

    record_by_id = {record["probe_id"]: record for record in records}
    probe_by_occurrence = {
        occurrence_id: probe["probe_id"]
        for probe in probes
        for occurrence_id in probe["occurrence_ids"]
    }
    occurrence_rows = []
    for occurrence in occurrences:
        probe_id = probe_by_occurrence[occurrence["occurrence_id"]]
        record = record_by_id.get(probe_id)
        row = dict(occurrence)
        row["probe_id"] = probe_id
        row["detected_bbox_2d_1000"] = (
            record.get("detected_bbox_2d_1000") if record else None
        )
        row["iou"] = (
            box_iou(occurrence["agent_bbox_2d_1000"], record["detected_bbox_2d_1000"])
            if record and record.get("detected_bbox_2d_1000") is not None else None
        )
        occurrence_rows.append(row)
    _write_atomic(results_dir / "requests.json", {"requests": occurrence_rows})

    per_question: dict[str, dict[str, Any]] = {}
    for probe in probes:
        question_id = probe["question_id"]
        summary = per_question.setdefault(question_id, {
            "dataset": probe.get("dataset"),
            "unique_probes": 0,
            "trace_occurrences": 0,
            "detected": 0,
            "ious": [],
        })
        summary["unique_probes"] += 1
        summary["trace_occurrences"] += len(probe["occurrence_ids"])
        record = record_by_id.get(probe["probe_id"])
        if record and record.get("detected_bbox_2d_1000") is not None:
            summary["detected"] += 1
            if record["metrics"]["max_iou"] is not None:
                summary["ious"].append(float(record["metrics"]["max_iou"]))

    summary_payload = {
        "created_at": _utc_now(),
        "trace_questions": len({item["question_id"] for item in occurrences}),
        "trace_grounding_requests": len(occurrences),
        "executed_trace_requests": sum(item["trace_status"] == "ok" for item in occurrences),
        "rejected_trace_requests": sum(item["trace_status"] == "rejected" for item in occurrences),
        "unique_probes": len(probes),
        "records_present": len(records),
        "complete": len(complete),
        "detected_boxes": len(boxes),
        "not_visible_responses": len(nulls),
        "errors": len(errors),
        "median_iou_with_agent_box": _median(ious),
        "iou_at_least_0_5": sum(value >= 0.5 for value in ious),
        "iou_at_least_0_75": sum(value >= 0.75 for value in ious),
        "median_center_distance_1000": _median(centers),
    }
    _write_atomic(results_dir / "summary.json", summary_payload)

    lines = [
        "# Plain-Qwen grounding debug",
        "",
        "This probe isolates Qwen's 2D grounding from the D4RT reasoning loop. For each",
        "unique `(question, t_src, label)` requested in the saved agent traces, Qwen saw",
        "only that one sampled image and the same fixed object-detection prompt. It received",
        "no video question, other frames, tool schema, D4RT output, or conversation history.",
        "",
        "Repeated trace requests with identical image and label were evaluated once and",
        "mapped back to every occurrence. Each saved source-frame image combines all new",
        "plain-Qwen detections requested on that frame. It does not draw the original",
        "agent boxes.",
        "",
        "## Status",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Questions | {summary_payload['trace_questions']} |",
        f"| Original grounding requests | {summary_payload['trace_grounding_requests']} |",
        f"| Unique image/label probes | {summary_payload['unique_probes']} |",
        f"| Completed probes | {summary_payload['complete']} |",
        f"| Parsed boxes | {summary_payload['detected_boxes']} |",
        f"| Explicit not-visible responses | {summary_payload['not_visible_responses']} |",
        f"| Errors | {summary_payload['errors']} |",
        f"| Median IoU with original agent box | {_fmt(summary_payload['median_iou_with_agent_box'])} |",
        f"| IoU ≥ 0.50 | {summary_payload['iou_at_least_0_5']} / {len(ious)} |",
        f"| IoU ≥ 0.75 | {summary_payload['iou_at_least_0_75']} / {len(ious)} |",
        f"| Median box-center distance (0–1000 coordinates) | {_fmt(summary_payload['median_center_distance_1000'], 1)} |",
        "",
        "> IoU measures whether two Qwen prompting contexts chose the same region; it is",
        "> not localization accuracy because DSI-Bench supplies no ground-truth boxes.",
        "> The overlays must be inspected to decide whether agreement reproduces a mistake.",
        "",
        "## Per question",
        "",
        "| Question | Dataset | Trace requests | Unique probes | Detected | Median IoU |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for question_id, values in sorted(per_question.items()):
        lines.append(
            f"| `{question_id}` | {values['dataset']} | {values['trace_occurrences']} | "
            f"{values['unique_probes']} | {values['detected']} | "
            f"{_fmt(_median(values['ious']))} |"
        )
    lines.extend([
        "",
        "## Probe details",
        "",
        "| Question / frame | Label | Repetitions | Qwen box | IoU | Frame detections |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ])
    for probe in probes:
        record = record_by_id.get(probe["probe_id"])
        detected = record.get("detected_bbox_2d_1000") if record else None
        iou = record.get("metrics", {}).get("max_iou") if record else None
        detected_text = "`null`" if detected is None else f"`{[round(v, 1) for v in detected]}`"
        frame_image = f"frames/{_frame_output_name(probe['question_id'], probe['t_src'])}"
        lines.append(
            f"| `{probe['question_id']}` / {probe['t_src']} | {probe['label']} | "
            f"{len(probe['occurrence_ids'])} | {detected_text} | {_fmt(iou)} | "
            f"[view]({frame_image}) |"
        )
    lines.extend([
        "",
        "Raw model replies and parsed boxes are in `detections/`; `requests.json` maps",
        "the unique results back to every original trace step.",
        "",
    ])
    (results_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    from PIL import Image

    answers_dir = Path(args.answers_dir)
    results_dir = Path(args.results_dir)
    occurrences, probes = extract_requests(answers_dir)
    if args.limit is not None:
        probes = probes[: args.limit]
        kept = {item for probe in probes for item in probe["occurrence_ids"]}
        occurrences = [item for item in occurrences if item["occurrence_id"] in kept]
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(results_dir / "manifest.json", {
        "phase": "qwen_grounding_debug",
        "created_at": _utc_now(),
        "answers_dir": str(answers_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "prompt_template": PROMPT_TEMPLATE,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "trace_questions": len({item["question_id"] for item in occurrences}),
        "trace_grounding_requests": len(occurrences),
        "unique_probes": len(probes),
        "deduplication_key": ["question_id", "t_src", "label"],
        "probes": probes,
    })
    if args.dry_run:
        print(
            f"{len(occurrences)} trace requests across "
            f"{len({item['question_id'] for item in occurrences})} questions -> "
            f"{len(probes)} unique probes"
        )
        return 0

    qwen = OfflineQwen(args.qwen_model, args.max_new_tokens, args.seed)
    print(f"qwen ready ({qwen.model_path}); {len(probes)} unique probes", flush=True)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for probe in probes:
        by_question[probe["question_id"]].append(probe)

    completed = 0
    started = time.monotonic()
    for question_index, (question_id, question_probes) in enumerate(sorted(by_question.items()), 1):
        sampled = sample_video_cpu(Path(question_probes[0]["video_path"]))
        expected_mapping = question_probes[0]["sampling"].get("sampled_to_original")
        if expected_mapping is not None and sampled.mapping() != expected_mapping:
            raise RuntimeError(f"{question_id}: sampled-frame mapping changed")
        for probe in question_probes:
            output_path = results_dir / "detections" / f"{probe['probe_id']}.json"
            if output_path.exists() and not args.force:
                completed += 1
                continue

            prompt = PROMPT_TEMPLATE.format(label=probe["label"])
            record: dict[str, Any] = {
                "phase": "qwen_grounding_debug",
                "created_at": _utc_now(),
                **probe,
                "prompt": prompt,
                "qwen_model": qwen.model_path,
            }
            probe_started = time.monotonic()
            try:
                content = [
                    {"type": "image", "image": Image.fromarray(sampled.frames_rgb[probe["t_src"]])},
                    {"type": "text", "text": prompt},
                ]
                raw = qwen.generate([{"role": "user", "content": content}])
                detected = parse_detection(raw)
                record.update(
                    status="complete",
                    raw_qwen_response=raw,
                    detected_bbox_2d_1000=detected,
                    metrics=_probe_metrics(probe["agent_bboxes_2d_1000"], detected),
                )
            except Exception as error:  # keep the run restartable after any malformed reply
                detected = None
                record.update(
                    status="error",
                    error=f"{type(error).__name__}: {error}",
                    detected_bbox_2d_1000=None,
                    metrics=_probe_metrics(probe["agent_bboxes_2d_1000"], None),
                )
            record["wall_seconds"] = round(time.monotonic() - probe_started, 3)
            _write_atomic(output_path, record)
            completed += 1
            print(
                f"[{completed:2d}/{len(probes)}] {record['status']:8s} "
                f"f={probe['t_src']:2d} label={probe['label']!r} "
                f"iou={_fmt(record['metrics']['max_iou'])}",
                flush=True,
            )
        frame_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for probe in question_probes:
            frame_groups[probe["t_src"]].append(probe)
        for t_src, frame_probes in sorted(frame_groups.items()):
            items = []
            for probe in frame_probes:
                record_path = results_dir / "detections" / f"{probe['probe_id']}.json"
                record = json.loads(record_path.read_text(encoding="utf-8"))
                items.append((probe, record.get("detected_bbox_2d_1000")))
            _draw_combined_frame(
                sampled.frames_rgb[t_src],
                items,
                results_dir / "frames" / _frame_output_name(question_id, t_src),
            )
        del sampled
        gc.collect()
        print(
            f"question {question_index:2d}/{len(by_question)} done: {question_id}",
            flush=True,
        )

    write_summary(results_dir, occurrences, probes)
    print(f"grounding probe done in {(time.monotonic() - started) / 60:.1f} min", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-dir", type=Path, default=DEFAULT_ANSWERS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
