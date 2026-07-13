"""One-command, intentionally simple D4RT tool-use demonstration.

The script is reusable for any compatible ``demo_data.json`` bundle. Basketball
defaults make the first run concrete and reproducible:

    python -m d4rt_agent.simple_v1 --phase deterministic

Later phases add Qwen behind the same entry point; the deterministic phase is the
geometry and evaluation baseline that does not need a GPU.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .geometry_tools import DemoGeometry


DEFAULT_DEMO = Path("demo/pstudio_mini/basketball_6")
DEFAULT_SEED = (346.0, 166.0, 0)
QUESTIONS = [
    {
        "id": "endpoint_displacement",
        "question": "What is the metric distance between the starting and ending position of the basketball?",
        "tool": "endpoint_displacement",
    },
    {
        "id": "distance_travelled",
        "question": "How much distance did the basketball cover between the first and last frame of the video?",
        "tool": "path_length",
    },
]


def _measure(geometry: DemoGeometry, track_id: int, tool: str) -> dict[str, Any]:
    last = geometry.num_frames - 1
    if tool == "endpoint_displacement":
        return geometry.endpoint_measurement(track_id, 0, last)
    if tool == "path_length":
        return geometry.path_measurement(track_id, 0, last)
    raise ValueError(f"Unsupported measurement tool: {tool}")


def run_deterministic(demo_dir: Path, seed: tuple[float, float, int]) -> dict[str, Any]:
    """Run known-pixel grounding against predicted and GT geometry."""
    pred = DemoGeometry(demo_dir, source="pred")
    gt = DemoGeometry(demo_dir, source="gt")
    u, v, t = seed
    pred_hit = pred.nearest_track(u, v, t)
    gt_hit = gt.nearest_track(u, v, t)

    results = []
    for question in QUESTIONS:
        predicted = _measure(pred, pred_hit.track_id, question["tool"])
        reference = _measure(gt, gt_hit.track_id, question["tool"])
        key = "endpoint_displacement_m" if question["tool"] == "endpoint_displacement" else "path_length_m"
        error = abs(float(predicted[key]) - float(reference[key]))
        results.append({
            **question,
            "predicted": predicted,
            "reference": reference,
            "absolute_error_m": round(error, 4),
        })

    return {
        "phase": "deterministic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "demo_dir": str(demo_dir),
        "video_meta": {
            "num_frames": pred.num_frames,
            "fps": pred.fps,
            "width": pred.width,
            "height": pred.height,
        },
        "grounding": {
            "label": "basketball",
            "requested_uvt": [u, v, t],
            "predicted_track_id": pred_hit.track_id,
            "predicted_pixel_distance": round(pred_hit.pixel_dist, 4),
            "gt_track_id": gt_hit.track_id,
            "gt_pixel_distance": round(gt_hit.pixel_dist, 4),
        },
        "questions": results,
    }


def _print_deterministic(result: dict[str, Any]) -> None:
    grounding = result["grounding"]
    print(f"Demo: {result['demo_dir']}")
    print(
        "Grounding: pixel "
        f"{grounding['requested_uvt']} -> predicted track {grounding['predicted_track_id']} "
        f"({grounding['predicted_pixel_distance']:.2f}px away)"
    )
    print("\nMeasurement                 D4RT (m)    GT (m)    Abs. error (m)")
    print("--------------------------  ----------  --------  --------------")
    for item in result["questions"]:
        key = "endpoint_displacement_m" if item["tool"] == "endpoint_displacement" else "path_length_m"
        print(
            f"{item['id']:<26}  {item['predicted'][key]:>10.4f}  "
            f"{item['reference'][key]:>8.4f}  {item['absolute_error_m']:>14.4f}"
        )
        if item["tool"] == "path_length":
            pred = item["predicted"]
            print(
                f"  visible coverage: {pred['visible_frames']}/{pred['total_frames']} frames "
                f"({100 * pred['coverage']:.1f}%), segments={pred['visible_segments']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["deterministic"], default="deterministic")
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--seed-u", type=float, default=DEFAULT_SEED[0])
    parser.add_argument("--seed-v", type=float, default=DEFAULT_SEED[1])
    parser.add_argument("--seed-frame", type=int, default=DEFAULT_SEED[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("d4rt_agent/results/basketball_6/phase1_deterministic.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_deterministic(args.demo_dir, (args.seed_u, args.seed_v, args.seed_frame))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    _print_deterministic(result)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
