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
import math
import re
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
QWEN_SYSTEM_PROMPT = """You are the semantic planner for a 4D geometry tool.
You see frame 0 of a video and receive a question about the basketball.

Choose exactly one tool:
- endpoint_displacement: straight-line separation between the start and end positions.
- path_length: accumulated distance covered/travelled along intermediate positions.

Ground the center of the basketball in frame 0. Coordinates are normalized integers
from 0 to 1000, where [0,0] is top-left and [1000,1000] is bottom-right.

Return JSON only, with no markdown or explanation:
{"tool":"endpoint_displacement or path_length","point_2d":[x,y],"frame":0}
"""


def _measure(geometry: DemoGeometry, track_id: int, tool: str) -> dict[str, Any]:
    last = geometry.num_frames - 1
    if tool == "endpoint_displacement":
        return geometry.endpoint_measurement(track_id, 0, last)
    if tool == "path_length":
        return geometry.path_measurement(track_id, 0, last)
    raise ValueError(f"Unsupported measurement tool: {tool}")


def _measure_group(geometry: DemoGeometry, track_ids: list[int], tool: str) -> dict[str, Any]:
    last = geometry.num_frames - 1
    if tool == "endpoint_displacement":
        return geometry.group_endpoint_measurement(track_ids, 0, last)
    if tool == "path_length":
        return geometry.group_path_measurement(track_ids, 0, last)
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


def run_robust(demo_dir: Path, seed: tuple[float, float, int], radius_px: float) -> dict[str, Any]:
    """Run duplicate-aware local track grouping for the same two measurements."""
    pred = DemoGeometry(demo_dir, source="pred")
    gt = DemoGeometry(demo_dir, source="gt")
    u, v, t = seed
    pred_group = pred.ground_track_group(u, v, t, radius_px=radius_px)
    gt_group = gt.ground_track_group(u, v, t, radius_px=radius_px)
    results = []
    for question in QUESTIONS:
        predicted = _measure_group(pred, pred_group["unique_track_ids"], question["tool"])
        reference = _measure_group(gt, gt_group["unique_track_ids"], question["tool"])
        key = "endpoint_displacement_m" if question["tool"] == "endpoint_displacement" else "path_length_m"
        results.append({
            **question,
            "predicted": predicted,
            "reference": reference,
            "absolute_error_m": round(abs(float(predicted[key]) - float(reference[key])), 4),
        })
    return {
        "phase": "robust",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "demo_dir": str(demo_dir),
        "grounding": {"requested_uvt": [u, v, t], "predicted": pred_group, "gt": gt_group},
        "questions": results,
    }


def _parse_planner_json(text: str) -> dict[str, Any]:
    """Extract and validate Qwen's deliberately small planner response."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Qwen did not return a JSON object: {text!r}")
    plan = json.loads(match.group(0))
    tool = plan.get("tool")
    if tool not in {"endpoint_displacement", "path_length"}:
        raise ValueError(f"unsupported tool in Qwen plan: {tool!r}")
    point = plan.get("point_2d")
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError(f"point_2d must be [x, y], got {point!r}")
    x, y = float(point[0]), float(point[1])
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ValueError(f"normalized point is out of range: {point!r}")
    frame = int(plan.get("frame", 0))
    if frame != 0:
        raise ValueError(f"V1 grounding must use frame 0, got frame {frame}")
    return {"tool": tool, "point_2d": [x, y], "frame": frame}


def _parse_metric_estimate(text: str) -> dict[str, Any]:
    """Parse the constrained, no-tool metric estimate returned by Qwen."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Qwen did not return a JSON object: {text!r}")
    value = json.loads(match.group(0))
    estimate = float(value["estimate_m"])
    if not math.isfinite(estimate) or estimate < 0:
        raise ValueError(f"invalid metric estimate: {estimate!r}")
    return {"estimate_m": estimate, "reason": str(value.get("reason", ""))}


def _read_video_frame(video_path: Path, frame_index: int):
    """Read one RGB frame lazily so deterministic runs do not require OpenCV."""
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


class QwenPlanner:
    """Small constrained Qwen3-VL planner: one image in, one JSON tool call out."""

    def __init__(self, model_id: str, max_new_tokens: int = 128) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        supplied_path = Path(model_id).expanduser()
        model_path = (
            supplied_path.resolve()
            if supplied_path.exists()
            else Path(snapshot_download(repo_id=model_id, local_files_only=True)).resolve()
        )
        self.model_path = str(model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
        )

    def plan(self, image, question: str) -> tuple[dict[str, Any], str]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {question}"},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return _parse_planner_json(raw), raw

    def answer_without_tools(self, labelled_images: list[tuple[int, Any]], question: str) -> str:
        """Direct VLM baseline using sampled frames but no D4RT measurements."""
        content: list[dict[str, Any]] = []
        for frame_index, image in labelled_images:
            content.append({"type": "text", "text": f"Frame {frame_index}:"})
            content.append({"type": "image", "image": image})
        content.append({
            "type": "text",
            "text": (
                f"Question: {question}\n"
                "Answer directly without external tools. Give a metric number only if the "
                "video itself provides enough metric scale; otherwise say it cannot be determined."
            ),
        })
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def estimate_without_tools(
        self, labelled_images: list[tuple[int, Any]], question: str
    ) -> tuple[dict[str, Any], str]:
        """Force a best-effort metric estimate from pixels alone for ablation."""
        content: list[dict[str, Any]] = []
        for frame_index, image in labelled_images:
            content.append({"type": "text", "text": f"Frame {frame_index}:"})
            content.append({"type": "image", "image": image})
        content.append({
            "type": "text",
            "text": (
                f"Question: {question}\n\n"
                "Use only these video frames and visual common-sense priors. Do not use external tools, "
                "tracking, depth, camera calibration, or supplied geometry. You must make your best "
                "non-negative numerical estimate in meters even though monocular scale is uncertain. "
                "Return JSON only: {\"estimate_m\": number, \"reason\": \"brief explanation\"}"
            ),
        })
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return _parse_metric_estimate(raw), raw


def run_qwen(
    demo_dir: Path,
    reference_seed: tuple[float, float, int],
    model_id: str,
    max_new_tokens: int,
    grounding_radius: float,
) -> dict[str, Any]:
    """Let Qwen choose the tool and grounding point, then execute D4RT geometry."""
    pred = DemoGeometry(demo_dir, source="pred")
    gt = DemoGeometry(demo_dir, source="gt")
    planner = QwenPlanner(model_id, max_new_tokens=max_new_tokens)
    image = _read_video_frame(demo_dir / "assets" / "input_video.mp4", 0)
    reference_u, reference_v, reference_t = reference_seed
    reference_gt_hit = gt.nearest_track(reference_u, reference_v, reference_t)

    results = []
    for question in QUESTIONS:
        item: dict[str, Any] = {**question}
        try:
            plan, raw = planner.plan(image, question["question"])
            u = plan["point_2d"][0] * pred.width / 1000.0
            v = plan["point_2d"][1] * pred.height / 1000.0
            group = pred.ground_track_group(u, v, plan["frame"], radius_px=grounding_radius)
            measurement = _measure_group(pred, group["unique_track_ids"], plan["tool"])
            reference = _measure(gt, reference_gt_hit.track_id, question["tool"])
            selected_key = (
                "endpoint_displacement_m" if plan["tool"] == "endpoint_displacement" else "path_length_m"
            )
            expected_key = (
                "endpoint_displacement_m" if question["tool"] == "endpoint_displacement" else "path_length_m"
            )
            point_error = math.hypot(u - reference_u, v - reference_v)
            item.update({
                "raw_qwen_response": raw,
                "plan": plan,
                "grounding_pixel": [round(u, 2), round(v, 2)],
                "grounding_pixel_error": round(point_error, 2),
                "grounded_track_id": group["representative_track_id"],
                "track_group": group,
                "track_snap_distance": group["nearest_pixel_distance"],
                "planning_correct": plan["tool"] == question["tool"],
                "grounding_within_40px": point_error <= 40.0,
                "measurement": measurement,
                "reference": reference,
                "answer": f"{measurement[selected_key]:.4f} m",
            })
            if plan["tool"] == question["tool"]:
                item["absolute_error_m"] = round(
                    abs(float(measurement[selected_key]) - float(reference[expected_key])), 4
                )
            else:
                item["absolute_error_m"] = None
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["planning_correct"] = False
            item["grounding_within_40px"] = False
        results.append(item)

    sample_indices = sorted({0, pred.num_frames // 3, 2 * pred.num_frames // 3, pred.num_frames - 1})
    labelled_images = [
        (index, _read_video_frame(demo_dir / "assets" / "input_video.mp4", index))
        for index in sample_indices
    ]
    direct_baseline = []
    for question in QUESTIONS:
        try:
            response = planner.answer_without_tools(labelled_images, question["question"])
            direct_baseline.append({**question, "response": response})
        except Exception as exc:
            direct_baseline.append({**question, "error": f"{type(exc).__name__}: {exc}"})

    gpu = {}
    if planner.torch.cuda.is_available():
        gpu = {
            "name": planner.torch.cuda.get_device_name(0),
            "max_memory_allocated_gib": round(planner.torch.cuda.max_memory_allocated(0) / 2**30, 3),
        }
    return {
        "phase": "qwen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "demo_dir": str(demo_dir),
        "model": model_id,
        "local_model_path": planner.model_path,
        "gpu": gpu,
        "reference_grounding": [reference_u, reference_v, reference_t],
        "questions": results,
        "direct_vlm_baseline": direct_baseline,
        "summary": {
            "questions": len(results),
            "successful": sum("error" not in item for item in results),
            "planning_correct": sum(bool(item.get("planning_correct")) for item in results),
            "grounding_within_40px": sum(bool(item.get("grounding_within_40px")) for item in results),
        },
    }


def run_direct_vlm(
    demo_dir: Path,
    reference_seed: tuple[float, float, int],
    model_id: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Force Qwen to estimate both metric quantities from pixels alone."""
    gt = DemoGeometry(demo_dir, source="gt")
    planner = QwenPlanner(model_id, max_new_tokens=max_new_tokens)
    reference_hit = gt.nearest_track(*reference_seed)
    sample_indices = sorted({round(i * (gt.num_frames - 1) / 7) for i in range(8)})
    labelled_images = [
        (index, _read_video_frame(demo_dir / "assets" / "input_video.mp4", index))
        for index in sample_indices
    ]
    results = []
    for question in QUESTIONS:
        item: dict[str, Any] = {**question}
        try:
            estimate, raw = planner.estimate_without_tools(labelled_images, question["question"])
            reference = _measure(gt, reference_hit.track_id, question["tool"])
            key = "endpoint_displacement_m" if question["tool"] == "endpoint_displacement" else "path_length_m"
            gt_value = float(reference[key])
            error = abs(estimate["estimate_m"] - gt_value)
            item.update({
                "raw_qwen_response": raw,
                "estimate_m": estimate["estimate_m"],
                "reason": estimate["reason"],
                "gt_m": gt_value,
                "absolute_error_m": round(error, 4),
                "relative_error": round(error / gt_value, 4) if gt_value else None,
            })
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
    gpu = {}
    if planner.torch.cuda.is_available():
        gpu = {
            "name": planner.torch.cuda.get_device_name(0),
            "max_memory_allocated_gib": round(planner.torch.cuda.max_memory_allocated(0) / 2**30, 3),
        }
    return {
        "phase": "direct_vlm",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "demo_dir": str(demo_dir),
        "model": model_id,
        "local_model_path": planner.model_path,
        "sampled_frames": sample_indices,
        "information_available_to_model": "RGB frames and question only",
        "gpu": gpu,
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


def _print_qwen(result: dict[str, Any]) -> None:
    print(f"Demo: {result['demo_dir']}")
    print(f"Model: {result['model']}")
    if result.get("gpu"):
        print(
            f"GPU: {result['gpu']['name']}; peak allocated model/process memory "
            f"{result['gpu']['max_memory_allocated_gib']:.3f} GiB"
        )
    for item in result["questions"]:
        print(f"\n[{item['id']}] {item['question']}")
        if "error" in item:
            print(f"  ERROR: {item['error']}")
            continue
        print(f"  Qwen: {item['raw_qwen_response']}")
        print(
            f"  pixel={item['grounding_pixel']}, track={item['grounded_track_id']}, "
            f"tool={item['plan']['tool']}, answer={item['answer']}"
        )
        print(
            f"  planning_correct={item['planning_correct']}, "
            f"grounding_error={item['grounding_pixel_error']:.2f}px"
        )
    print(f"\nSummary: {json.dumps(result['summary'])}")
    print("\nDirect VLM baseline (no D4RT):")
    for item in result.get("direct_vlm_baseline", []):
        print(f"  {item['id']}: {item.get('response', item.get('error'))}")


def _print_robust(result: dict[str, Any]) -> None:
    pred_group = result["grounding"]["predicted"]
    print(f"Demo: {result['demo_dir']}")
    print(
        f"Local grounding found {len(pred_group['candidate_track_ids'])} track entries, "
        f"collapsed to {len(pred_group['unique_track_ids'])} unique trajectory."
    )
    print("Measurement                 Median (m)  Spread min/max (m)  Abs. error (m)")
    print("--------------------------  ----------  ------------------  --------------")
    for item in result["questions"]:
        key = "endpoint_displacement_m" if item["tool"] == "endpoint_displacement" else "path_length_m"
        spread = item["predicted"]["metric_spread_m"]
        print(
            f"{item['id']:<26}  {item['predicted'][key]:>10.4f}  "
            f"{spread['min']:.4f}/{spread['max']:.4f}         {item['absolute_error_m']:.4f}"
        )


def run_aggregate(results_dir: Path) -> dict[str, Any]:
    """Combine the saved phase artifacts without rerunning either model."""
    paths = {
        "phase1": results_dir / "phase1_deterministic.json",
        "phase2": results_dir / "phase2_qwen.json",
        "phase3": results_dir / "phase3_robust.json",
    }
    phases: dict[str, Any] = {}
    missing = []
    for name, path in paths.items():
        if path.exists():
            phases[name] = json.loads(path.read_text())
        else:
            missing.append(str(path))

    aggregate: dict[str, Any] = {
        "phase": "aggregate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not missing else "partial",
        "missing": missing,
    }
    if "phase1" in phases:
        aggregate["geometry_baseline"] = [
            {
                "id": item["id"],
                "d4rt_m": item["predicted"][
                    "endpoint_displacement_m" if item["tool"] == "endpoint_displacement" else "path_length_m"
                ],
                "gt_m": item["reference"][
                    "endpoint_displacement_m" if item["tool"] == "endpoint_displacement" else "path_length_m"
                ],
                "absolute_error_m": item["absolute_error_m"],
            }
            for item in phases["phase1"]["questions"]
        ]
    if "phase2" in phases:
        phase2 = phases["phase2"]
        total = max(int(phase2["summary"]["questions"]), 1)
        aggregate["qwen"] = {
            "model": phase2["model"],
            "gpu": phase2.get("gpu", {}),
            "planning_accuracy": phase2["summary"]["planning_correct"] / total,
            "grounding_within_40px_accuracy": phase2["summary"]["grounding_within_40px"] / total,
            "tool_questions": phase2["questions"],
            "direct_vlm_baseline": phase2.get("direct_vlm_baseline", []),
        }
    if "phase3" in phases:
        group = phases["phase3"]["grounding"]["predicted"]
        aggregate["robust_grounding"] = {
            "candidate_tracks": len(group["candidate_track_ids"]),
            "unique_tracks": len(group["unique_track_ids"]),
            "duplicates_removed": group["duplicate_tracks_removed"],
        }
    return aggregate


def _aggregate_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# D4RT Agent V1 Aggregate Results",
        "",
        f"Status: **{result['status']}**",
        "",
    ]
    if result.get("missing"):
        lines.extend(["Missing artifacts:", ""] + [f"- `{path}`" for path in result["missing"]] + [""])
    if "geometry_baseline" in result:
        lines.extend([
            "## Geometry baseline",
            "",
            "| Task | D4RT | GT | Absolute error |",
            "|---|---:|---:|---:|",
        ])
        for item in result["geometry_baseline"]:
            lines.append(
                f"| {item['id']} | {item['d4rt_m']:.4f} m | {item['gt_m']:.4f} m | "
                f"{item['absolute_error_m']:.4f} m |"
            )
        lines.append("")
    if "qwen" in result:
        qwen = result["qwen"]
        lines.extend([
            "## Qwen3-VL orchestration",
            "",
            f"- Model: `{qwen['model']}`",
            f"- Planning accuracy: {100 * qwen['planning_accuracy']:.1f}%",
            f"- Grounding within 40 px: {100 * qwen['grounding_within_40px_accuracy']:.1f}%",
            f"- GPU: `{qwen.get('gpu', {}).get('name', 'unknown')}`",
            "",
            "### Direct VLM baseline",
            "",
        ])
        for item in qwen["direct_vlm_baseline"]:
            lines.append(f"- **{item['id']}**: {item.get('response', item.get('error', 'missing'))}")
        lines.append("")
    if "robust_grounding" in result:
        robust = result["robust_grounding"]
        lines.extend([
            "## Robust grounding",
            "",
            f"Found {robust['candidate_tracks']} local track entries, collapsed to "
            f"{robust['unique_tracks']} unique trajectory; removed {robust['duplicates_removed']} duplicates.",
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "The V1 goal is transparent orchestration, not state-of-the-art accuracy. Geometry, grounding, "
        "planning, and answer-generation failures are reported separately. Visible-only path length does "
        "not interpolate across occlusions.",
        "",
    ])
    return "\n".join(lines)


def _print_aggregate(result: dict[str, Any]) -> None:
    print(_aggregate_markdown(result))


def _print_direct_vlm(result: dict[str, Any]) -> None:
    print(f"Pure VLM baseline: {result['model']}")
    print(f"Sampled frames: {result['sampled_frames']}")
    print("Measurement                 Qwen (m)  GT (m)  Abs. error  Relative error")
    print("--------------------------  --------  ------  ----------  --------------")
    for item in result["questions"]:
        if "error" in item:
            print(f"{item['id']:<26}  ERROR: {item['error']}")
            continue
        print(
            f"{item['id']:<26}  {item['estimate_m']:>8.4f}  {item['gt_m']:>6.4f}  "
            f"{item['absolute_error_m']:>10.4f}  {100 * item['relative_error']:>12.1f}%"
        )
        print(f"  reason: {item['reason']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["deterministic", "qwen", "robust", "aggregate", "direct"], default="deterministic"
    )
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--seed-u", type=float, default=DEFAULT_SEED[0])
    parser.add_argument("--seed-v", type=float, default=DEFAULT_SEED[1])
    parser.add_argument("--seed-frame", type=int, default=DEFAULT_SEED[2])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--grounding-radius", type=float, default=12.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("d4rt_agent/results/basketball_6"))
    parser.add_argument(
        "--report", type=Path, default=Path("d4rt_agent/results/basketball_6/AGGREGATE_RESULTS.md")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = (args.seed_u, args.seed_v, args.seed_frame)
    if args.phase == "deterministic":
        result = run_deterministic(args.demo_dir, seed)
        output = args.output or Path("d4rt_agent/results/basketball_6/phase1_deterministic.json")
        printer = _print_deterministic
    elif args.phase == "qwen":
        result = run_qwen(
            args.demo_dir,
            seed,
            args.model,
            args.max_new_tokens,
            args.grounding_radius,
        )
        output = args.output or Path("d4rt_agent/results/basketball_6/phase2_qwen.json")
        printer = _print_qwen
    elif args.phase == "robust":
        result = run_robust(args.demo_dir, seed, args.grounding_radius)
        output = args.output or Path("d4rt_agent/results/basketball_6/phase3_robust.json")
        printer = _print_robust
    elif args.phase == "aggregate":
        result = run_aggregate(args.results_dir)
        output = args.output or args.results_dir / "aggregate.json"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_aggregate_markdown(result))
        printer = _print_aggregate
    else:
        result = run_direct_vlm(args.demo_dir, seed, args.model, args.max_new_tokens)
        output = args.output or Path("d4rt_agent/results/basketball_6/phase5_direct_vlm.json")
        printer = _print_direct_vlm
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    printer(result)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
