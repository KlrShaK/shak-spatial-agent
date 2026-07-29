"""Run the simple_v2 agent, or a tool-free Qwen control, over a DSI-Bench sample.

Two passes share one manifest, one set of sampled frames, and one decoding
configuration::

    python -m d4rt_agent.dsi_bench_run --dry-run
    python -m d4rt_agent.dsi_bench_run --point-mode ensemble5
    python -m d4rt_agent.dsi_bench_run --baseline

The agent pass loads the D4RT checkpoint once and rebinds it to each clip, since
the checkpoint costs minutes to load and a clip costs seconds to encode.  Every
question is written to its own file as soon as it finishes, so an interrupted job
resumes instead of restarting.

Answers are recorded, never scored: DSI-Bench ground truth travels through to the
report so the comparison can be made by eye.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any, Mapping, Sequence

from .dsi_bench_data import (
    DEFAULT_DSI_ROOT,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    build_manifest,
    format_question_prompt,
    parse_choice_letter,
    read_manifest,
    write_manifest,
)
from .simple_v2 import (
    ActionRejected,
    DEFAULT_QWEN_MODEL,
    OfflineQwen,
    OrchestrationError,
    SimpleV2Orchestrator,
    ToolExecutionError,
    _resolve_path,
    unknown_evidence_error,
)
from .simple_v2_backend import (
    DEFAULT_D4RT_CHECKPOINT,
    DEFAULT_D4RT_CONFIG,
    LiveD4RTBackend,
)
from .simple_v2_contracts import NUM_SAMPLED_FRAMES, POINT_MODES, SampledVideo, sample_video_cpu


DEFAULT_RESULTS_DIR = Path("d4rt_agent/results/dsi_bench")
DEFAULT_MANIFEST = DEFAULT_RESULTS_DIR / "manifest.json"
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dsi_bench_system.md"

# DSI-Bench ships no ground-truth scale, so positions stay in model units and the
# artifact must not claim otherwise.
DSI_BENCHMARK_SCALE = 1.0
DSI_ALIGNMENT_TYPE = "identity_no_ground_truth_scale"

# Replaces the backend's default note, which forbids all cross-t_cam comparison.
# That rule is right for displacement inside one frame but wrong here: observer
# motion and camera-to-object range are *measured* by varying t_cam, and this
# string reaches the model on every tool result, so it has to agree with the
# system prompt rather than contradict it.
DSI_CROSS_FRAME_NOTE = (
    "Component values from queries with different t_cam are in different bases and must "
    "never be subtracted. Scalars are frame-invariant: norm() and dist() may be compared "
    "across different t_cam, which is how range change is measured. Reading a static "
    "background point across several t_cam values is how camera motion is measured."
)

ANSWER_INSTRUCTION = (
    'Answer with final_answer using kind="text". Begin the text with the letter of the '
    "option you choose, then that option's text copied exactly, then the measured "
    "quantities that chose it. Cite the query_d4rt evidence behind it."
)

# The tool-free control uses the answer format from DSI-Bench's own reference
# evaluator, so the baseline is the benchmark's intended protocol rather than a
# variant of it.
BASELINE_PROMPT = (
    "You are a vision-language expert.\n"
    "You are given a clip of video and your task is to answer a question about the video.\n"
    "You only need to provide *ONE* correct answer selecting from the options listed below.\n"
    "For example, if you think the correct answer is 'A' from 'A. Above B. Under C. Front "
    "D. Behind', your response should **only** be '<answer>A</answer>'.\n"
    "Please answer the question in this format strictly:\n"
    "<answer>[A, B, C, or D]</answer>\n"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DSIOrchestrator(SimpleV2Orchestrator):
    """The simple_v2 loop with DSI-Bench's final-answer rule.

    The metric pipeline keys its evidence rules off a task id and lets text
    answers through unchecked, which here would allow an answer read straight off
    the images.  That would make the agent indistinguishable from the tool-free
    control, so a cited D4RT call is required instead.
    """

    # How many steps before the limit the host starts telling the model to land,
    # and how many are reserved for the answer itself once warnings are ignored.
    DEADLINE_WARNING_STEPS = 3
    HARD_DEADLINE_STEPS = 1
    # How many queries may come back with nothing visible before the host stops
    # accepting more. Occlusion is a property of the clip, so the third empty
    # result is already strong evidence that a fourth will be empty too.
    MAX_BLIND_QUERIES = 3

    # Handed to the model whenever it is being told to stop measuring. A model
    # that has just issued a dozen queries follows that pattern rather than an
    # instruction; a literal skeleton to fill in is what breaks it.
    ANSWER_TEMPLATE = (
        '{"action":"final_answer","arguments":{"kind":"text",'
        '"text":"<letter>: <that option\'s text>. <the measurements behind it>",'
        '"evidence_ids":["d4rt_1"],"limitations":"<what stayed uncertain>"}}'
    )

    def __init__(self, *, require_d4rt: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.require_d4rt = bool(require_d4rt)

    def _step_notice(self, step: int) -> str | None:
        """Warn the model as its step budget runs out.

        Every option is a plausible answer, so there is no natural stopping point:
        a model that cannot measure what it wants keeps trying to measure it.  Job
        8070124 lost three questions to agents scanning frames one at a time until
        the loop ran out.  The remaining-step count is host knowledge, so the host
        is what has to say it.
        """

        remaining = self.max_steps - step
        if remaining > self.DEADLINE_WARNING_STEPS or remaining <= 0:
            return None
        return (
            f"You have {remaining} step(s) left before the run is abandoned and the "
            "question is recorded unanswered. Stop gathering evidence now. If you still "
            "need a calculation, make it your next action, then answer. Otherwise emit "
            "final_answer immediately, choosing the best-supported option from the "
            "evidence you already have and naming what stayed uncertain in `limitations`. "
            "An answer with acknowledged uncertainty is worth far more than no answer."
        )

    def _execute_action(
        self,
        name: str,
        arguments: dict[str, Any],
        evidence: dict[str, dict[str, Any]],
        step: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]] | None, str]:
        if step is not None and self.max_steps - step <= self.HARD_DEADLINE_STEPS:
            # Warnings alone did not stop agents from measuring until the loop
            # died (job 8071987), so the last step is reserved for the answer.
            raise ActionRejected(
                "no steps remain for measurement — this is your last action, so it must "
                "be final_answer. Choose the best-supported option from the evidence you "
                "already have. Emit exactly this shape, filled in: " + self.ANSWER_TEMPLATE
            )
        if name == "query_d4rt":
            self._refuse_hopeless_query(evidence)
        try:
            return super()._execute_action(name, arguments, evidence, step=step)
        except ActionRejected as error:
            if name == "python_math":
                raise ActionRejected(
                    self._diagnose_math(arguments, evidence, error)
                ) from error
            raise

    def _refuse_hopeless_query(self, evidence: Mapping[str, dict[str, Any]]) -> None:
        """Stop the model hunting for a frame where an absent object is visible.

        Both questions still failing after job 8084609 died the same way: the
        object left the shot, and the agent spent its whole budget looking for it
        -- one sweeping `t_cam` (which cannot affect visibility at all), the other
        walking `t_tgt` backwards a frame at a time. Neither responded to being
        told to stop, because a dozen queries in context outweigh an instruction.
        Refusing the query is the only reliable way to end the pattern.
        """

        blind = [
            call_id
            for call_id, prior in evidence.items()
            if call_id.startswith("d4rt_") and not any(prior.get("math_visibility") or [True])
        ]
        if len(blind) < self.MAX_BLIND_QUERIES:
            return
        raise ActionRejected(
            f"{len(blind)} queries ({', '.join(sorted(blind))}) have already come back with "
            "nothing visible, so the tracked point is absent from those frames and no further "
            "query will recover it. Changing t_cam cannot make an unobserved target visible. "
            "Stop measuring and answer now from the frames that did resolve, plus what the "
            "images show, recording the gap in `limitations`. Emit exactly this shape, filled "
            "in: " + self.ANSWER_TEMPLATE
        )

    def _diagnose_math(
        self,
        arguments: Mapping[str, Any],
        evidence: Mapping[str, dict[str, Any]],
        error: Exception,
    ) -> str:
        """Explain a failed calculation in terms the model can act on.

        An occluded target returns ``benchmark_aligned_xyz_m: null``, and binding
        it fails with "non-finite or non-numeric values" -- a message that never
        mentions visibility, so the model rebinds the same null. Three questions
        in job 8071987 were lost to exactly that loop, one of them retrying twelve
        times. Naming the invisible frame turns a dead end into a next action.
        """

        message = str(error)
        culprits: list[str] = []
        for variable, specification in (arguments.get("bindings") or {}).items():
            if not isinstance(specification, Mapping):
                continue
            evidence_id = specification.get("evidence_id")
            source = evidence.get(evidence_id) if isinstance(evidence_id, str) else None
            if source is None:
                continue
            path = specification.get("path") or []
            try:
                value = _resolve_path(source, path)
            except ValueError:
                continue
            if value is None or (isinstance(value, list) and any(item is None for item in value)):
                frames = source.get("t_tgt", [])
                index = next((item for item in path if isinstance(item, int)), None)
                frame = frames[index] if isinstance(index, int) and index < len(frames) else "?"
                culprits.append(f"`{variable}` (from {evidence_id}, sampled frame {frame})")
        if not culprits:
            return message
        return (
            f"{message}. The cause is visibility, not arithmetic: {', '.join(culprits)} "
            "resolved to null because D4RT did not observe that point at that frame. "
            "Rebinding the same field will fail again. Instead, query a spread of target "
            "frames in one call (for example t_tgt [0,3,6,9,12,15,18,21,24,27,31]), read "
            "`math_visibility`, and use the first and last frames whose entry is true. If "
            "too few frames are visible to measure what you wanted, answer now from what "
            "you have and say so in `limitations`."
        )

    def _validate_final(
        self,
        task: Mapping[str, Any],
        arguments: dict[str, Any],
        evidence: Mapping[str, dict[str, Any]],
    ) -> None:
        if arguments.get("kind") != "text":
            raise ValueError(
                'DSI-Bench answers are multiple choice: use kind="text" and begin the text '
                "with the option letter you chose."
            )
        evidence_ids = arguments.get("evidence_ids", [])
        unknown = [item for item in evidence_ids if item not in evidence]
        if unknown:
            raise ValueError(
                "final answer: " + unknown_evidence_error(unknown, evidence)
            )
        if self.require_d4rt and not any(item.startswith("d4rt_") for item in evidence_ids):
            raise ValueError(
                "the answer must cite at least one query_d4rt call, because an answer read "
                "from the images alone cannot be told apart from a guess; measure the "
                "quantity that separates the options, then cite it"
            )


def _task_for(entry: Mapping[str, Any]) -> dict[str, str]:
    """Build the model-facing task.

    Only the question and its options cross this boundary.  ``cate``, ``dataset``,
    ``GT``, ``others`` and ``video_type`` stay behind: telling the model its
    question's category would hand it the reasoning recipe and void the run.
    """

    return {
        "id": entry["question_id"],
        "question": f"{format_question_prompt(entry)}\n\n{ANSWER_INSTRUCTION}",
    }


def _entry_facts(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "question_id": entry["question_id"],
        "dataset": entry["dataset"],
        "cate": entry["cate"],
        "category_name": entry["category_name"],
        "relative_path": entry["relative_path"],
        "video_slug": entry["video_slug"],
        "question": entry["question"],
        "options": entry["options"],
        "option_letters": entry["option_letters"],
        "gt": entry["gt"],
    }


def _sampling_record(sampled: SampledVideo) -> dict[str, Any]:
    return {
        "contract": "round(linspace(0, N - 1, 32))",
        "decoded_on": "CPU",
        "video": str(sampled.video_path),
        "total_original_frames": sampled.total_original_frames,
        "num_sampled_frames": NUM_SAMPLED_FRAMES,
        "fps": sampled.fps,
        "width": sampled.width,
        "height": sampled.height,
        "sampled_to_original": sampled.mapping(),
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write one result so an interrupted job never leaves a half-parsed file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _pending(entries: Sequence[Mapping[str, Any]], directory: Path, force: bool) -> list[Mapping[str, Any]]:
    if force:
        return list(entries)
    return [entry for entry in entries if not (directory / f"{entry['question_id']}.json").exists()]


def _select(manifest: Mapping[str, Any], args: argparse.Namespace) -> list[Mapping[str, Any]]:
    entries = list(manifest["questions"])
    if args.only:
        wanted = set(args.only)
        entries = [entry for entry in entries if entry["question_id"] in wanted]
        missing = wanted - {entry["question_id"] for entry in entries}
        if missing:
            raise SystemExit(f"unknown question ids: {sorted(missing)}")
    if args.num_shards > 1:
        entries = [entry for index, entry in enumerate(entries) if index % args.num_shards == args.shard]
    if args.limit is not None:
        entries = entries[: args.limit]
    return entries


def run_dry(manifest: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> int:
    """Check every selected clip decodes into 32 frames, without loading a model."""

    print(f"manifest seed={manifest['sampling']['seed']} "
          f"gt_histogram={manifest['counts']['gt_histogram']}")
    print(f"per-dataset {manifest['counts']['per_dataset']}")
    print(f"per-category {manifest['counts']['per_category']}")
    print()
    failures = 0
    for index, entry in enumerate(entries):
        path = Path(entry["video_path"])
        try:
            sampled = sample_video_cpu(path)
        except Exception as error:  # noqa: BLE001 - report every bad clip, not just the first
            failures += 1
            print(f"[{index:2d}] FAIL {entry['question_id']}: {type(error).__name__}: {error}")
            continue
        print(
            f"[{index:2d}] ok   {entry['dataset']:12s} c{entry['cate']} "
            f"frames={sampled.total_original_frames:4d} fps={sampled.fps:5.1f} "
            f"{sampled.width}x{sampled.height} gt={entry['gt']} {entry['question_id'][:50]}"
        )
        del sampled
        gc.collect()
    print()
    print(f"checked {len(entries)} clips, {failures} unusable")
    return 1 if failures else 0


def _solve_one(
    entry: Mapping[str, Any],
    *,
    qwen: OfflineQwen,
    backend: LiveD4RTBackend,
    sampled: SampledVideo,
    system_prompt: str,
    max_steps: int,
    require_d4rt: bool,
) -> dict[str, Any]:
    orchestrator = DSIOrchestrator(
        qwen=qwen,
        backend=backend,
        sampled_video=sampled,
        max_steps=max_steps,
        system_prompt=system_prompt,
        require_d4rt=require_d4rt,
    )
    return orchestrator.solve(_task_for(entry))


def run_agent(args: argparse.Namespace, manifest: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> int:
    answers_dir = Path(args.results_dir) / "answers"
    pending = _pending(entries, answers_dir, args.force)
    print(
        f"{len(entries)} selected, {len(pending)} pending, "
        f"{len(entries) - len(pending)} already done",
        flush=True,
    )
    if not pending:
        return 0

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    # Built from the first pending clip so a resumed job never decodes a video it
    # is about to skip.
    first = sample_video_cpu(Path(pending[0]["video_path"]))
    backend = LiveD4RTBackend(
        sampled_video=first,
        point_mode=args.point_mode,
        benchmark_scale=DSI_BENCHMARK_SCALE,
        alignment_type=DSI_ALIGNMENT_TYPE,
        cross_frame_note=DSI_CROSS_FRAME_NOTE,
        model_config=args.d4rt_config,
        checkpoint=args.d4rt_checkpoint,
        device=args.device,
        dtype=args.d4rt_dtype,
        query_chunk_size=args.query_chunk_size,
    )
    del first
    gc.collect()
    qwen = OfflineQwen(args.qwen_model, args.max_new_tokens, args.seed)
    print(
        f"backend ready ({backend.metadata()['checkpoint']}), qwen ready ({qwen.model_path})",
        flush=True,
    )

    started = time.monotonic()
    for index, entry in enumerate(pending, start=1):
        question_start = time.monotonic()
        sampled = None
        record: dict[str, Any] = {
            "phase": "dsi_bench_agent",
            "created_at": _utc_now(),
            **_entry_facts(entry),
            "point_mode": args.point_mode,
            "benchmark_alignment": {"type": DSI_ALIGNMENT_TYPE, "scale": DSI_BENCHMARK_SCALE},
        }
        try:
            sampled = sample_video_cpu(Path(entry["video_path"]))
            backend.rebind_video(sampled)
            record["sampling"] = _sampling_record(sampled)
            try:
                solved = _solve_one(
                    entry,
                    qwen=qwen,
                    backend=backend,
                    sampled=sampled,
                    system_prompt=system_prompt,
                    max_steps=args.max_steps,
                    require_d4rt=True,
                )
                record.update(status="complete", d4rt_used=True, **_solution_fields(solved))
            except OrchestrationError as error:
                # The D4RT requirement is what failed, not necessarily the question.
                # Retry once without it so the row stays comparable to the control,
                # and record plainly that this answer is not tracker-backed.
                record["gated_attempt"] = {
                    "error": str(error),
                    "trace": error.trace,
                    "evidence": error.evidence,
                }
                try:
                    solved = _solve_one(
                        entry,
                        qwen=qwen,
                        backend=backend,
                        sampled=sampled,
                        system_prompt=system_prompt,
                        max_steps=args.max_steps,
                        require_d4rt=False,
                    )
                    record.update(
                        status="complete_relaxed", d4rt_used=False, **_solution_fields(solved)
                    )
                except OrchestrationError as relaxed_error:
                    record.update(
                        status="failed",
                        d4rt_used=False,
                        error=str(relaxed_error),
                        trace=relaxed_error.trace,
                        evidence=relaxed_error.evidence,
                    )
        except ToolExecutionError as error:
            record.update(
                status="error",
                d4rt_used=False,
                error=str(error),
                trace=error.trace,
                evidence=error.evidence,
                traceback=traceback.format_exc(),
            )
        except Exception as error:  # noqa: BLE001 - one bad clip must not cost the rest
            record.update(
                status="error",
                d4rt_used=False,
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        record["wall_seconds"] = round(time.monotonic() - question_start, 2)
        _write_atomic(answers_dir / f"{entry['question_id']}.json", record)

        elapsed = time.monotonic() - started
        remaining = (elapsed / index) * (len(pending) - index)
        print(
            f"[{index:2d}/{len(pending)}] {record['status']:17s} "
            f"d4rt={str(record.get('d4rt_used')):5s} "
            f"{record['wall_seconds']:7.1f}s eta={remaining / 60:5.1f}m "
            f"{entry['question_id'][:44]}",
            flush=True,
        )
        # Held until here so the frames are released before the next clip decodes;
        # the host allowance is 24 GiB and one clip can be half a gigabyte.
        sampled = None
        gc.collect()

    print(f"agent pass done in {(time.monotonic() - started) / 60:.1f} min")
    return 0


def _solution_fields(solved: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "final_answer": solved["final_answer"],
        "steps": solved["steps"],
        "trace": solved["trace"],
        "evidence": solved["evidence"],
    }


def run_baseline(args: argparse.Namespace, manifest: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> int:
    """Answer the same questions from the same frames with no tools at all."""

    from PIL import Image

    baseline_dir = Path(args.results_dir) / "baseline"
    pending = _pending(entries, baseline_dir, args.force)
    print(f"{len(entries)} selected, {len(pending)} pending")
    if not pending:
        return 0

    qwen = OfflineQwen(args.qwen_model, args.max_new_tokens, args.seed)
    started = time.monotonic()
    for index, entry in enumerate(pending, start=1):
        question_start = time.monotonic()
        sampled = None
        record: dict[str, Any] = {
            "phase": "dsi_bench_baseline",
            "created_at": _utc_now(),
            **_entry_facts(entry),
            "qwen_model": qwen.model_path,
        }
        try:
            sampled = sample_video_cpu(Path(entry["video_path"]))
            record["sampling"] = _sampling_record(sampled)
            # Same 32 frames, same chronological numbering the agent sees, so the
            # only difference between the two passes is the tool loop.
            content: list[dict[str, Any]] = []
            for position, frame in enumerate(sampled.frames_rgb):
                content.append({"type": "text", "text": f"Sampled frame {position}:"})
                content.append({"type": "image", "image": Image.fromarray(frame)})
            content.append(
                {"type": "text", "text": f"{BASELINE_PROMPT}\nQuestion:\n{format_question_prompt(entry)}"}
            )
            raw = qwen.generate([{"role": "user", "content": content}])
            record.update(
                status="complete",
                raw_response=raw,
                parsed_letter=parse_choice_letter(raw, entry["option_letters"]),
            )
        except Exception as error:  # noqa: BLE001
            record.update(
                status="error",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        record["wall_seconds"] = round(time.monotonic() - question_start, 2)
        _write_atomic(baseline_dir / f"{entry['question_id']}.json", record)
        print(
            f"[{index:2d}/{len(pending)}] {record['status']:9s} "
            f"answer={record.get('parsed_letter')} gt={entry['gt']} "
            f"{record['wall_seconds']:6.1f}s {entry['question_id'][:44]}",
            flush=True,
        )
        sampled = None
        gc.collect()

    print(f"baseline pass done in {(time.monotonic() - started) / 60:.1f} min")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsi-root", type=Path, default=DEFAULT_DSI_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seed-sample", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-balance-gt",
        action="store_true",
        help="skip the search for a seed with an even answer-key histogram",
    )
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="check clips decode; no model")
    parser.add_argument("--baseline", action="store_true", help="tool-free Qwen control pass")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="re-run questions already answered")
    parser.add_argument(
        "--point-mode",
        choices=POINT_MODES,
        default=os.environ.get("POINT_MODE", "ensemble5"),
        help="immutable host grounding policy (never included in the Qwen schema)",
    )
    parser.add_argument("--qwen-model", default=os.environ.get("MODEL_DIR", DEFAULT_QWEN_MODEL))
    parser.add_argument("--d4rt-config", type=Path, default=DEFAULT_D4RT_CONFIG)
    parser.add_argument("--d4rt-checkpoint", type=Path, default=DEFAULT_D4RT_CHECKPOINT)
    parser.add_argument("--d4rt-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    # The facing-axis recipe alone costs three queries plus a calculation, so a
    # budget of 10 left almost no room to recover from one wasted step.
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = args.manifest or (Path(args.results_dir) / "manifest.json")
    if args.rebuild_manifest or not manifest_path.exists():
        manifest = build_manifest(
            dsi_root=args.dsi_root,
            split=args.split,
            seed=args.seed_sample,
            balance_gt=not args.no_balance_gt,
        )
        write_manifest(manifest, manifest_path)
        print(f"wrote manifest: {manifest_path}")
    else:
        manifest = read_manifest(manifest_path)

    entries = _select(manifest, args)
    if args.dry_run:
        raise SystemExit(run_dry(manifest, entries))
    if args.baseline:
        raise SystemExit(run_baseline(args, manifest, entries))
    raise SystemExit(run_agent(args, manifest, entries))


if __name__ == "__main__":
    main()
