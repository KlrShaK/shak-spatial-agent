"""Simple offline Qwen -> live D4RT agent (Phase 7).

Examples::

    python -m d4rt_agent.simple_v2 --point-mode centroid
    python -m d4rt_agent.simple_v2 --point-mode ensemble5

The selected point policy is immutable host configuration.  It is intentionally
absent from every schema and message shown to Qwen.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .simple_v2_backend import (
    DEFAULT_D4RT_CHECKPOINT,
    DEFAULT_D4RT_CONFIG,
    LiveD4RTBackend,
)
from .simple_v2_contracts import (
    ACTION_SCHEMAS,
    DEFAULT_ASSUMED_FPS,
    NUM_SAMPLED_FRAMES,
    POINT_MODES,
    SampledVideo,
    restricted_python_math,
    sample_video_cpu,
    validate_action,
)
from .simple_v2_eval import (
    DEFAULT_DEMO_DATA,
    DEFAULT_GT_SEED,
    DEFAULT_WORLDTRACK_NPZ,
    METRE_SCORED_TASKS,
    MatchedWorldTrackGT,
    aggregate_files,
    load_alignment_scale_from_metadata,
    score_question,
)


DEFAULT_DEMO_DIR = Path("demo/pstudio_mini/basketball_6")
DEFAULT_VIDEO = DEFAULT_DEMO_DIR / "assets" / "input_video.mp4"
DEFAULT_RESULTS_DIR = Path("d4rt_agent/results/basketball_6")
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QUESTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "endpoint_displacement",
        "question": (
            "What is the metric distance between the starting and ending position "
            "of the basketball?"
        ),
    },
    {
        "id": "distance_travelled",
        "question": (
            "How much distance did the basketball cover between the first and last "
            "frame of the video?"
        ),
    },
    # Open-ended: no WorldTrack ground truth, so this is answered and recorded but
    # scored: false.  It exercises the descriptive/text answer path.
    {
        "id": "person_motion_description",
        "question": "How did the person move during this clip?",
    },
)


# The system prompt outgrew this module.  It lives beside it as markdown so the
# prompt can be read and reviewed on its own, and is loaded once at import.
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "simple_v2_system.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_action_json(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object representing an action."""

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ("action" in value or "name" in value):
            return value
    raise ValueError(f"Qwen did not emit one JSON action: {text[:500]!r}")


def _redact_host_policy(result: dict[str, Any]) -> dict[str, Any]:
    """Remove host policy metadata before returning a tool result to Qwen."""

    redacted = {
        key: value for key, value in result.items() if key not in {"point_mode", "query_points"}
    }
    predictions = redacted.get("predictions")
    if isinstance(predictions, list):
        redacted["predictions"] = [
            {
                key: value
                for key, value in prediction.items()
                if key != "individual_point_results"
            }
            for prediction in predictions
        ]
    return redacted


def _resolve_path(value: Any, path: Sequence[Any]) -> Any:
    current = value
    for component in path:
        if isinstance(current, Mapping) and isinstance(component, str):
            if component not in current:
                raise ValueError(f"binding path field not found: {component!r}")
            current = current[component]
        elif isinstance(current, (list, tuple)) and isinstance(component, int):
            try:
                current = current[component]
            except IndexError as error:
                raise ValueError(f"binding path index out of range: {component}") from error
        else:
            raise ValueError(f"cannot apply binding path component {component!r}")
    return current


def resolve_bindings(
    specifications: Mapping[str, Any], evidence: Mapping[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve Qwen binding specifications only from recorded evidence."""

    values: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for variable, specification in specifications.items():
        if not isinstance(variable, str) or not variable.isidentifier() or variable.startswith("_"):
            raise ValueError(f"invalid binding variable: {variable!r}")
        if not isinstance(specification, Mapping):
            raise ValueError(f"binding {variable!r} must be an evidence reference object")
        evidence_id = specification.get("evidence_id")
        path = specification.get("path", [])
        if not isinstance(evidence_id, str) or evidence_id not in evidence:
            raise ValueError(f"binding {variable!r} cites unknown evidence: {evidence_id!r}")
        if not isinstance(path, list) or not all(isinstance(item, (str, int)) for item in path):
            raise ValueError(f"binding {variable!r} path must contain string fields or integer indices")
        values[variable] = _resolve_path(evidence[evidence_id], path)
        provenance[variable] = {"evidence_id": evidence_id, "path": path}
    if not values:
        raise ValueError("python_math requires at least one prior-evidence binding")
    return values, provenance


def _all_finite_scalars(value: Any) -> list[float]:
    found: list[float] = []
    if isinstance(value, Mapping):
        for child in value.values():
            found.extend(_all_finite_scalars(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_all_finite_scalars(child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        found.append(float(value))
    return found


def _validate_final_evidence(
    task_id: str,
    arguments: dict[str, Any],
    evidence: Mapping[str, dict[str, Any]],
) -> None:
    evidence_ids = arguments["evidence_ids"]
    if any(item not in evidence for item in evidence_ids):
        unknown = [item for item in evidence_ids if item not in evidence]
        raise ValueError(f"final answer cites unknown evidence IDs: {unknown}")
    if arguments.get("kind", "numeric") == "text":
        # Descriptive answers carry no scored number, so there is nothing to trace
        # back to a calculation.  They are recorded as unscored instead.
        return
    d4rt_ids = [item for item in evidence_ids if item.startswith("d4rt_")]
    math_ids = [item for item in evidence_ids if item.startswith("math_")]
    if not d4rt_ids or not math_ids:
        raise ValueError("final answer must cite both a d4rt_* and math_* call")
    targets = {
        int(value)
        for evidence_id in d4rt_ids
        for value in evidence[evidence_id]["t_tgt"]
    }
    if len(targets) < 2:
        raise ValueError("measurement evidence must query at least two sampled frames")
    interval_start, interval_end = min(targets), max(targets)
    if task_id == "endpoint_displacement":
        # The question determines the endpoints.  They may be explicit sampled
        # frames or the object's visually determined first/last appearance.
        pass
    elif task_id == "distance_travelled":
        expected = set(range(interval_start, interval_end + 1))
        if targets != expected:
            missing = sorted(expected - targets)
            raise ValueError(
                "path evidence must query every sampled frame in the selected "
                f"inclusive interval [{interval_start},{interval_end}]; missing {missing}"
            )
    # Any other task id simply has no bespoke evidence rule yet; the two-frame
    # minimum above still applies.  Task ids are host-supplied, never model-supplied,
    # so falling back here cannot be used to weaken a rule the model was given.
    candidates: list[float] = []
    for evidence_id in math_ids:
        candidates.extend(_all_finite_scalars(evidence[evidence_id].get("outputs", {})))
    value = float(arguments["value"])
    tolerance = max(1e-3, abs(value) * 1e-3)
    if not any(abs(value - candidate) <= tolerance for candidate in candidates):
        raise ValueError("final answer value does not match a cited calculation output")
    # Only the GT-scored tasks are pinned to meters, because their answer is compared
    # against a metres ground truth where "centimeters" would be a silent 100x error.
    if task_id in METRE_SCORED_TASKS:
        if arguments["unit"].strip().lower() not in {"m", "meter", "meters", "metre", "metres"}:
            raise ValueError(f"{task_id} answers are scored in meters and must use meters")


class OfflineQwen:
    """Deterministic local-only Qwen3-VL generator."""

    def __init__(self, model_id: str, max_new_tokens: int = 768, seed: int = 42) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.seed = int(seed)
        supplied = Path(model_id).expanduser()
        if supplied.exists():
            model_path = supplied.resolve()
        else:
            model_path = Path(
                snapshot_download(repo_id=model_id, local_files_only=True)
            ).resolve()
        self.model_path = str(model_path)
        self.device_map_policy = os.environ.get("QWEN_DEVICE_MAP", "auto").strip().lower()
        if self.device_map_policy == "auto":
            device_map: str | dict[str, int] = "auto"
        elif self.device_map_policy == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("QWEN_DEVICE_MAP=cuda requires an available CUDA device")
            device_map = {"": 0}
        else:
            raise ValueError("QWEN_DEVICE_MAP must be 'auto' or 'cuda'")
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        self.max_new_tokens = int(max_new_tokens)

    def generate(self, messages: list[dict[str, Any]]) -> str:
        self.torch.manual_seed(self.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.seed)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        new_tokens = output[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class OrchestrationError(RuntimeError):
    """Agent exhaustion that retains all replayable partial state."""

    def __init__(self, message: str, trace: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]) -> None:
        super().__init__(message)
        self.trace = trace
        self.evidence = evidence


class SimpleV2Orchestrator:
    """Host-enforced three-action loop with replayable evidence."""

    def __init__(
        self,
        *,
        qwen: OfflineQwen,
        backend: LiveD4RTBackend,
        sampled_video: SampledVideo,
        max_steps: int = 12,
        system_prompt: str | None = None,
    ) -> None:
        self.qwen = qwen
        self.backend = backend
        self.sampled_video = sampled_video
        self.max_steps = max(1, int(max_steps))
        self.system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt

    def _validate_final(
        self,
        task: Mapping[str, Any],
        arguments: dict[str, Any],
        evidence: Mapping[str, dict[str, Any]],
    ) -> None:
        """Check a final answer against the evidence recorded for this task.

        Subclasses serving other benchmarks override this to impose their own
        evidence rules; the metric pipeline keeps the task-id-keyed rules.
        """

        _validate_final_evidence(task["id"], arguments, evidence)

    def _step_notice(self, step: int) -> str | None:
        """Optional message appended after each step, e.g. a remaining-step warning.

        Returning ``None`` leaves the conversation untouched, which is the
        behaviour every caller had before this hook existed.
        """

        return None

    def solve(self, task: dict[str, str]) -> dict[str, Any]:
        from PIL import Image

        tool_text = json.dumps(ACTION_SCHEMAS, separators=(",", ":"))
        initial_content: list[dict[str, Any]] = []
        for index, frame in enumerate(self.sampled_video.frames_rgb):
            initial_content.append({"type": "text", "text": f"Sampled frame {index}:"})
            initial_content.append({"type": "image", "image": Image.fromarray(frame)})
        fps = float(self.sampled_video.fps)
        if not math.isfinite(fps) or fps <= 0:
            fps = DEFAULT_ASSUMED_FPS
        step_seconds = (
            self.sampled_video.original_indices[1] - self.sampled_video.original_indices[0]
        ) / fps
        initial_content.append({
            "type": "text",
            "text": (
                "Each image above is full resolution. Ground bbox_2d_1000 in the "
                "sampled frame declared by t_src.\n"
                f"Timing: consecutive sampled frames are {step_seconds:.4f} seconds "
                "apart, so the elapsed time between sampled frames A and B is "
                f"{step_seconds:.4f} * (B - A) seconds, and the whole clip spans "
                f"{step_seconds * 31:.4f} seconds.\n"
                f"Available action schemas: {tool_text}\n\n"
                f"Question: {task['question']}"
            ),
        })
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": initial_content},
        ]
        trace: list[dict[str, Any]] = []
        evidence: dict[str, dict[str, Any]] = {}
        counters = {"d4rt": 0, "math": 0, "final": 0}

        for step in range(1, self.max_steps + 1):
            raw = self.qwen.generate(messages)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})
            attempt: dict[str, Any] = {
                "step": step,
                "kind": "model_action",
                "raw_qwen_response": raw,
            }
            try:
                if re.search(r"\bpoint_mode\b", raw):
                    raise ValueError("Qwen must never select or emit point_mode")
                parsed = _extract_action_json(raw)
                name, arguments = validate_action(parsed)
                attempt["parsed_action"] = {"action": name, "arguments": arguments}
                if name == "final_answer":
                    self._validate_final(task, arguments, evidence)
                    counters["final"] += 1
                    call_id = f"final_{counters['final']}"
                    record = {
                        **attempt,
                        "call_id": call_id,
                        "status": "ok",
                        "result": arguments,
                    }
                    trace.append(record)
                    return {
                        "point_mode": self.backend.point_mode,
                        **task,
                        "final_answer": arguments,
                        "trace": trace,
                        "evidence": evidence,
                        "steps": step,
                    }
                result, response_content, prefix = self._execute_action(
                    name, arguments, evidence, step=step
                )
                counters[prefix] += 1
                call_id = f"{prefix}_{counters[prefix]}"
                evidence[call_id] = result
                record = {
                    **attempt,
                    "call_id": call_id,
                    "status": "ok",
                    "justification": arguments["justification"],
                    "result": result,
                }
                trace.append(record)
                if response_content is None:
                    response_content = [{
                        "type": "text",
                        "text": f"Tool result {call_id}: {json.dumps(_redact_host_policy(result))}",
                    }]
                else:
                    response_content.insert(0, {
                        "type": "text",
                        "text": f"Tool result {call_id}: {json.dumps(_redact_host_policy(result))}",
                    })
                notice = self._step_notice(step)
                if notice:
                    response_content.append({"type": "text", "text": notice})
                messages.append({"role": "user", "content": response_content})
            except (KeyError, TypeError, ValueError) as error:
                attempt.update(status="rejected", error=str(error))
                trace.append(attempt)
                rejection = [{
                    "type": "text",
                    "text": (
                        f"Action rejected: {error}. Return one corrected JSON action using "
                        "only the supplied schemas."
                    ),
                }]
                notice = self._step_notice(step)
                if notice:
                    rejection.append({"type": "text", "text": notice})
                messages.append({"role": "user", "content": rejection})
        raise OrchestrationError(
            f"Qwen did not produce a valid final answer in {self.max_steps} steps; "
            f"last trace entry: {trace[-1] if trace else 'none'}",
            trace,
            evidence,
        )

    def _execute_action(
        self,
        name: str,
        arguments: dict[str, Any],
        evidence: dict[str, dict[str, Any]],
        step: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]] | None, str]:
        """Run one tool call. ``step`` lets subclasses enforce a step budget."""

        if name == "query_d4rt":
            result = self.backend.query(
                label=arguments["label"],
                bbox_2d_1000=arguments["bbox_2d_1000"],
                t_src=arguments["t_src"],
                t_tgt=arguments["t_tgt"],
                t_cam=arguments["t_cam"],
            )
            return result, None, "d4rt"
        if name == "python_math":
            values, provenance = resolve_bindings(arguments["bindings"], evidence)
            outputs = restricted_python_math(values, arguments["code"])
            result = {
                "bindings": provenance,
                "resolved_bindings": values,
                "code": arguments["code"],
                "outputs": outputs,
            }
            return result, None, "math"
        raise ValueError(f"host cannot execute action: {name}")


def replay_tool_trace(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate trace structure and replay every restricted calculation on CPU."""

    evidence: dict[str, dict[str, Any]] = {}
    replayed_math = 0
    final_answers = 0
    for entry in trace:
        if entry.get("status") != "ok":
            continue
        call_id = entry.get("call_id")
        if not isinstance(call_id, str):
            raise ValueError("successful trace entry is missing call_id")
        result = entry.get("result")
        if call_id.startswith("math_"):
            if not isinstance(result, Mapping):
                raise ValueError(f"invalid calculation result in {call_id}")
            replayed = restricted_python_math(result["resolved_bindings"], result["code"])
            if replayed != result["outputs"]:
                raise ValueError(f"calculation replay mismatch in {call_id}")
            replayed_math += 1
        if call_id.startswith("final_"):
            final_answers += 1
        if call_id.startswith(("d4rt_", "math_")):
            evidence[call_id] = dict(result)
    return {
        "status": "complete" if final_answers == 1 else "incomplete",
        "successful_evidence_calls": len(evidence),
        "replayed_math_calls": replayed_math,
        "final_answers": final_answers,
    }


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    sampled = sample_video_cpu(args.video)
    scale, scale_provenance = load_alignment_scale_from_metadata(args.demo_data)
    gt = MatchedWorldTrackGT(
        args.gt_npz,
        sampled.original_indices,
        seed=(args.gt_seed_u, args.gt_seed_v, args.gt_seed_frame),
    )
    backend = LiveD4RTBackend(
        sampled_video=sampled,
        point_mode=args.point_mode,
        benchmark_scale=scale,
        model_config=args.d4rt_config,
        checkpoint=args.d4rt_checkpoint,
        device=args.device,
        dtype=args.d4rt_dtype,
        query_chunk_size=args.query_chunk_size,
    )
    if args.smoke:
        result = backend.query(
            label="smoke-test object",
            bbox_2d_1000=[520.0, 360.0, 580.0, 480.0],
            t_src=0,
            t_tgt=[31],
            t_cam=0,
        )
        return {
            "phase": "simple_v2_live_smoke",
            "created_at": _utc_now(),
            "point_mode": args.point_mode,
            "sampling": _sampling_record(sampled),
            "scale_provenance": scale_provenance,
            "d4rt": backend.metadata(),
            "query": result,
            "gpu": backend.gpu_memory(),
        }

    qwen = OfflineQwen(args.qwen_model, args.max_new_tokens, args.seed)
    orchestrator = SimpleV2Orchestrator(
        qwen=qwen,
        backend=backend,
        sampled_video=sampled,
        max_steps=args.max_steps,
    )
    question_results = []
    scores = []
    failures = []
    for task in QUESTIONS:
        try:
            solved = orchestrator.solve(dict(task))
        except OrchestrationError as error:
            solved = {
                "point_mode": args.point_mode,
                **task,
                "status": "failed",
                "error": str(error),
                "trace": error.trace,
                "evidence": error.evidence,
            }
            question_results.append(solved)
            failures.append({"task_id": task["id"], "error": str(error)})
            break
        # Record the solved question before scoring: a scoring error must never
        # discard an expensive completed run.
        question_results.append(solved)
        try:
            score = score_question(
                task_id=task["id"],
                final_answer=solved["final_answer"],
                evidence=solved["evidence"],
                gt=gt,
                width=sampled.width,
                height=sampled.height,
            )
        except Exception as error:  # noqa: BLE001 - never lose the artifact to scoring
            failures.append({
                "task_id": task["id"],
                "stage": "scoring",
                "error": f"{type(error).__name__}: {error}",
            })
            continue
        if score.get("scored", True):
            score["agent_vs_recomputed_aligned_delta_m"] = abs(
                float(score["agent_final_value"]) - float(score["benchmark_aligned_value_m"])
            )
        scores.append(score)

    return {
        "phase": "simple_v2_live_agent",
        "status": "failed" if failures else "complete",
        "created_at": _utc_now(),
        "point_mode": args.point_mode,
        "offline": True,
        "configuration": {
            "seed": args.seed,
            "qwen_decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
            },
            "point_mode_selected_by": "CLI or POINT_MODE environment before execution",
            "point_mode_exposed_to_qwen": False,
            "t_cam_selected_by": "Qwen, per query, over the full sampled range",
        },
        "sampling": _sampling_record(sampled),
        "qwen": {
            "model": qwen.model_path,
            "device_map": qwen.device_map_policy,
            "action_schemas": list(ACTION_SCHEMAS),
        },
        "d4rt": backend.metadata(),
        "scale_provenance": scale_provenance,
        "ground_truth": gt.canonical_trajectory(),
        "questions": question_results,
        "scores": scores,
        "failures": failures,
        "gpu": backend.gpu_memory(),
        "trace_replay": [
            replay_tool_trace(item.get("trace", [])) for item in question_results
        ],
    }


def _sampling_record(sampled: SampledVideo) -> dict[str, Any]:
    return {
        "contract": "round(linspace(0, N - 1, 32))",
        "decoded_on": "CPU",
        "video": str(sampled.video_path),
        "total_original_frames": sampled.total_original_frames,
        "num_sampled_frames": NUM_SAMPLED_FRAMES,
        "sampled_to_original": sampled.mapping(),
        "fps": sampled.fps,
        "width": sampled.width,
        "height": sampled.height,
        "same_frames_for_qwen_and_d4rt": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point-mode",
        choices=POINT_MODES,
        default=os.environ.get("POINT_MODE", "centroid"),
        help="immutable host grounding policy (never included in the Qwen schema)",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--qwen-model", default=os.environ.get("MODEL_DIR", DEFAULT_QWEN_MODEL))
    parser.add_argument("--d4rt-config", type=Path, default=DEFAULT_D4RT_CONFIG)
    parser.add_argument("--d4rt-checkpoint", type=Path, default=DEFAULT_D4RT_CHECKPOINT)
    parser.add_argument("--d4rt-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-data", type=Path, default=DEFAULT_DEMO_DATA)
    parser.add_argument("--gt-npz", type=Path, default=DEFAULT_WORLDTRACK_NPZ)
    parser.add_argument("--gt-seed-u", type=float, default=DEFAULT_GT_SEED[0])
    parser.add_argument("--gt-seed-v", type=float, default=DEFAULT_GT_SEED[1])
    parser.add_argument("--gt-seed-frame", type=int, default=DEFAULT_GT_SEED[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="run one live D4RT query without Qwen")
    parser.add_argument("--replay-trace", type=Path, help="CPU-validate traces in an existing result")
    parser.add_argument("--aggregate", action="store_true", help="aggregate completed centroid and ensemble artifacts")
    parser.add_argument(
        "--centroid-result", type=Path, default=DEFAULT_RESULTS_DIR / "simple_v2_centroid.json"
    )
    parser.add_argument(
        "--ensemble-result", type=Path, default=DEFAULT_RESULTS_DIR / "simple_v2_ensemble5.json"
    )
    parser.add_argument(
        "--aggregate-output", type=Path, default=DEFAULT_RESULTS_DIR / "simple_v2_aggregate.json"
    )
    parser.add_argument(
        "--aggregate-report", type=Path, default=DEFAULT_RESULTS_DIR / "SIMPLE_V2_RESULTS.md"
    )
    args = parser.parse_args(argv)
    if args.point_mode not in POINT_MODES:
        parser.error(f"unsupported POINT_MODE default: {args.point_mode!r}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.aggregate:
        result = aggregate_files(
            args.centroid_result,
            args.ensemble_result,
            args.aggregate_output,
            args.aggregate_report,
        )
        print(json.dumps(result["comparison"], indent=2))
        print(f"Saved: {args.aggregate_output}")
        return
    if args.replay_trace is not None:
        artifact = json.loads(args.replay_trace.read_text())
        reports = [replay_tool_trace(item["trace"]) for item in artifact["questions"]]
        print(json.dumps(reports, indent=2))
        if artifact.get("status") != "complete" or any(
            report["status"] != "complete" for report in reports
        ):
            raise SystemExit(1)
        return
    result = run_live(args)
    output = args.output
    if output is None:
        suffix = "_smoke" if args.smoke else ""
        output = DEFAULT_RESULTS_DIR / f"simple_v2_{args.point_mode}{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if "scores" in result:
        for score in result["scores"]:
            task = score.get("task_id", "?")
            if not score.get("scored", True):
                answer = score.get("agent_final_text") or score.get("agent_final_value")
                print(
                    f"{task}: UNSCORED ({score.get('unscored_reason', 'no reason given')}) "
                    f"answer={answer!r}"
                )
                continue
            print(
                f"{task}: raw={score['raw_d4rt_value']:.4f} "
                f"aligned={score['benchmark_aligned_value_m']:.4f} m "
                f"GT={score['gt_value_m']:.4f} m error={score['absolute_error_m']:.4f} m"
            )
    print(f"Saved: {output}")
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
