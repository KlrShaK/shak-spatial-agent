"""Run the live isolated bbox/points tool on fixed one-frame smoke fixtures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from .dsi_bench_data import read_manifest
from .qwen_grounding_tool import GroundingResponseError, QwenGroundingTool
from .simple_v2 import DEFAULT_QWEN_MODEL, OfflineQwen
from .simple_v2_contracts import sample_video_cpu


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _draw(frame: Any, result: Mapping[str, Any], path: Path) -> None:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    color = (255, 55, 40)
    bbox = result.get("bbox_2d_1000")
    if isinstance(bbox, list):
        xy = [
            bbox[0] * (width - 1) / 1000.0,
            bbox[1] * (height - 1) / 1000.0,
            bbox[2] * (width - 1) / 1000.0,
            bbox[3] * (height - 1) / 1000.0,
        ]
        draw.rectangle(xy, outline=color, width=max(2, width // 300))
    for point_index, point in enumerate(result.get("points_2d_1000") or [], start=1):
        x = point["xy"][0] * (width - 1) / 1000.0
        y = point["xy"][1] * (height - 1) / 1000.0
        radius = max(4, width // 160)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        draw.text((x + radius + 2, y - radius), f"p{point_index}", fill=color)
    legend = (
        f"{result['mode']} | {result['status']} | {result['request']}"
        + (
            f" | {result.get('returned_count', 0)}/{result.get('requested_count')}"
            if result["mode"] == "points"
            else ""
        )
    )
    draw.rectangle((0, 0, min(width, 12 + 7 * len(legend)), 24), fill=(0, 0, 0))
    draw.text((6, 6), legend, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=94)


def run(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    by_id = {
        str(entry["question_id"]): entry for entry in manifest["questions"]
    }
    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    if not isinstance(requests, list) or not requests:
        raise ValueError("request fixture must be a non-empty JSON list")
    qwen = OfflineQwen(args.qwen_model, seed=args.seed)
    tool = QwenGroundingTool(qwen=qwen)
    print(f"one physical Qwen checkpoint: {qwen.model_path}", flush=True)

    for index, request in enumerate(requests, start=1):
        question_id = str(request["question_id"])
        if question_id not in by_id:
            raise ValueError(f"unknown fixture question ID: {question_id}")
        output = args.results_dir / f"request_{index:02d}.json"
        if output.exists() and not args.force:
            print(f"[{index}/{len(requests)}] skip {output.name}", flush=True)
            continue
        sampled = sample_video_cpu(Path(by_id[question_id]["video_path"]))
        try:
            result = tool.ground(
                sampled_video=sampled,
                mode=str(request["mode"]),
                t_src=int(request["t_src"]),
                request=str(request["request"]),
                count=request.get("count"),
            )
            parser_error = None
        except GroundingResponseError as error:
            # A malformed model reply is a smoke-test observation, not a reason
            # to lose the remaining fixed fixtures.
            parser_error = str(error)
            result = {
                "status": "parse_error",
                "mode": str(request["mode"]),
                "t_src": int(request["t_src"]),
                "request": str(request["request"]),
                "requested_count": request.get("count"),
                "bbox_2d_1000": None,
                "points_2d_1000": [],
                "host_provenance": error.host_provenance,
            }
        overlay = args.results_dir / "overlays" / f"request_{index:02d}.jpg"
        _draw(sampled.frames_rgb[int(request["t_src"])], result, overlay)
        record = {
            "phase": "qwen_grounding_tool_smoke",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fixture_index": index,
            "question_id": question_id,
            "video_path": str(sampled.video_path),
            "source_frame": {
                "sampled": int(request["t_src"]),
                "original": int(sampled.original_indices[int(request["t_src"])]),
            },
            "request": request,
            "result": result,
            "parser_error": parser_error,
            "raw_grounder_response": (
                result.get("host_provenance", {}).get("raw_grounder_response")
            ),
            "wall_seconds": result.get("host_provenance", {}).get(
                "inference_wall_time_seconds"
            ),
            "overlay": str(overlay.relative_to(args.results_dir)),
        }
        _atomic(output, record)
        print(
            f"[{index}/{len(requests)}] {result['status']:9s} "
            f"{request['mode']:6s} {question_id[:38]}",
            flush=True,
        )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--qwen-model", default=os.environ.get("MODEL_DIR", DEFAULT_QWEN_MODEL))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
