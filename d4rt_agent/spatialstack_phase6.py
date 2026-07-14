#!/usr/bin/env python3
"""Prepare and score the Phase 6 native SpatialStack basketball baseline.

SpatialStack itself is executed by its existing ``infer_batch.py``. This helper
only extracts video frames and converts its raw text answers into an auditable
metric-error artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


NUMBER_ONLY = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)?\s*[.]?\s*$", re.I)
NUMBER_WITH_UNIT = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b", re.I)


def sample_indices(total: int, count: int) -> list[int]:
    """Match ``np.linspace(0, total - 1, count, dtype=int)`` without NumPy."""
    if total <= count:
        return list(range(total))
    return [int(i * (total - 1) / (count - 1)) for i in range(count)]


def prepare_frames(video: Path, scenes_dir: Path) -> dict:
    """Extract every source frame; SpatialStack will uniformly sample 32 later."""
    import decord
    from PIL import Image

    frames_dir = scenes_dir / "basketball_6" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    reader = decord.VideoReader(str(video))
    for index in range(len(reader)):
        Image.fromarray(reader[index].asnumpy()).save(frames_dir / f"frame_{index:06d}.png")

    summary = {
        "video": str(video.resolve()),
        "total_frames": len(reader),
        "spatialstack_input_frames": min(32, len(reader)),
        "sampled_original_indices": sample_indices(len(reader), 32),
        "frames_dir": str(frames_dir),
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_metric_answer(text: str) -> float:
    """Parse a metric answer while avoiding frame numbers in verbose responses."""
    match = NUMBER_ONLY.match(text)
    if not match:
        matches = NUMBER_WITH_UNIT.findall(text)
        if len(matches) != 1:
            raise ValueError(f"expected one metric value, found {len(matches)} in: {text!r}")
        value = float(matches[0])
    else:
        value = float(match.group(1))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid non-negative metric distance: {value}")
    return value


def score(raw_path: Path, output_path: Path) -> dict:
    raw = json.loads(raw_path.read_text())
    questions = []
    for item in raw.get("results", []):
        response = item["model_answer"]
        gt = float(item["ground_truth"])
        record = {
            "id": item["prompt_id"],
            "question": item["question"],
            "raw_response": response,
            "gt_m": gt,
        }
        try:
            estimate = parse_metric_answer(response)
            absolute_error = abs(estimate - gt)
            record.update({
                "estimate_m": estimate,
                "absolute_error_m": absolute_error,
                "relative_error": absolute_error / gt,
            })
        except ValueError as error:
            record["parse_error"] = str(error)
        questions.append(record)

    sampled = sample_indices(64, 32)
    result = {
        "phase": "phase6_spatialstack",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": raw.get("metadata", {}).get("model_path"),
        "model_family": raw.get("metadata", {}).get("model_family"),
        "input": {
            "video": "demo/pstudio_mini/basketball_6/assets/input_video.mp4",
            "total_video_frames": 64,
            "num_visuals": 32,
            "sampling": "uniform, SpatialStack default",
            "sampled_original_frame_indices": sampled,
            "frame_label_mapping": {f"Frame-{i}": source for i, source in enumerate(sampled)},
            "add_frame_index": True,
        },
        "decoding": {
            "dtype": "bfloat16",
            "temperature": 0.0,
            "num_beams": 1,
            "max_new_tokens": 512,
            "disable_thinking": True,
        },
        "questions": questions,
        "raw_result": str(raw_path.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print("Task                       SpatialStack   GT       Abs. error   Relative error")
    print("-------------------------  -------------  -------  -----------  --------------")
    for item in questions:
        if "parse_error" in item:
            print(f"{item['id']:<25}  PARSE ERROR: {item['parse_error']}")
        else:
            print(
                f"{item['id']:<25}  {item['estimate_m']:>10.4f} m  {item['gt_m']:>6.4f} m  "
                f"{item['absolute_error_m']:>8.4f} m  {100 * item['relative_error']:>12.1f}%"
            )
    print(f"Saved {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="extract video frames for SpatialStack")
    prepare.add_argument("--video", type=Path, required=True)
    prepare.add_argument("--scenes-dir", type=Path, required=True)

    score_parser = subparsers.add_parser("score", help="score SpatialStack raw responses")
    score_parser.add_argument("--raw", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_frames(args.video, args.scenes_dir)
    else:
        score(args.raw, args.output)


if __name__ == "__main__":
    main()
