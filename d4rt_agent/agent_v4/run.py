"""Run agent v4 over a DSI-Bench manifest.

One JSON file per question, and a question that already has one is skipped. That
is what makes the 3 h SLURM limit survivable: the pipeline is ~10 h of GPU for
both question sets, so a run is expected to be resumed several times, and
resuming means resubmitting the identical command.

Sharding is deliberately not offered. Each shard would reload D4RT's 14 GB and
Qwen's 16 GB, which is the cost keeping everything resident exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from d4rt_agent.agent_v4.buckets import CATE_TO_BUCKET
from d4rt_agent.agent_v4.orchestrator import AgentV4Orchestrator
from d4rt_agent.agent_v4.pipeline import Models, QuestionPipeline
from d4rt_agent.dsi_bench_data import (
    DEFAULT_DSI_ROOT,
    DEFAULT_RANDOM_SIZE,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    build_manifest,
    read_manifest,
    write_manifest,
)
from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import (
    D4RT_CHECKPOINT,
    D4RT_CONFIG,
    git_sha,
)

DEFAULT_RESULTS_DIR = Path("d4rt_agent/results/agent_v4")
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_for(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The model-facing task.

    `cate`, `GT` and `category_name` are omitted on purpose. Showing the model
    which category a question belongs to would hand it the reasoning recipe the
    agent is supposed to work out for itself, and the bucket-vs-cate agreement
    is the headline diagnostic -- it means nothing if the answer was given away.
    """

    return {
        "id": entry["question_id"],
        "relative_path": entry["relative_path"],
        "question": entry["question"],
        "options": {letter: entry["options"][letter] for letter in entry["option_letters"]},
        # Authoritative: the same relative_path exists in all four augmentation
        # splits, so only this says which one to actually watch.
        "video_path": entry["video_path"],
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _pending(entries: Sequence[Mapping[str, Any]], directory: Path, force: bool):
    if force:
        return list(entries)
    return [e for e in entries if not (directory / f"{e['question_id']}.json").exists()]


def _select(manifest: Mapping[str, Any], args: argparse.Namespace) -> list[Mapping[str, Any]]:
    entries = list(manifest["questions"])
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["question_id"] in wanted or e["relative_path"] in wanted]
    if args.limit is not None:
        entries = entries[: args.limit]
    return entries


def resolve_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Load the manifest, or build it once and keep it beside the answers.

    The manifest lives inside the results directory because it *is* the
    definition of the run: a shared directory would mean two designs overwriting
    each other's selection and silently breaking the pairing with every earlier
    result.
    """

    path = args.manifest or (args.results_dir / "manifest.json")
    if path.exists() and not args.rebuild_manifest:
        return read_manifest(path)
    manifest = build_manifest(
        dsi_root=args.dsi_root, split=args.split, seed=args.seed_sample,
        balance_gt=not args.no_balance_gt,
        random_size=args.random_sample,
    )
    write_manifest(manifest, path)
    print(f"manifest written: {path} ({manifest['sampling']['design']}, "
          f"{manifest['sampling']['total_selected']} questions)", flush=True)
    return manifest


def run_dry(manifest: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> int:
    """Check every clip decodes, without loading a single model."""

    from d4rt_agent.simple_v2_contracts import sample_video_cpu

    failures = 0
    seen: dict[str, str] = {}
    for index, entry in enumerate(entries, start=1):
        path = Path(entry["video_path"])
        if entry["relative_path"] in seen:
            continue
        try:
            sampled = sample_video_cpu(path)
            seen[entry["relative_path"]] = "ok"
            print(f"[{index}/{len(entries)}] ok   {entry['relative_path']} "
                  f"{sampled.width}x{sampled.height} {sampled.frames_rgb.shape[0]} frames",
                  flush=True)
        except Exception as error:
            failures += 1
            print(f"[{index}/{len(entries)}] FAIL {entry['relative_path']}: {error}", flush=True)
    print(f"\n{len(seen)} distinct clips checked, {failures} failed", flush=True)
    return 1 if failures else 0


def run_agent(args: argparse.Namespace, manifest: Mapping[str, Any],
              entries: Sequence[Mapping[str, Any]]) -> int:
    from d4rt_agent.simple_v2 import OfflineQwen

    answers = args.results_dir / "answers"
    pending = _pending(entries, answers, args.force)
    print(f"{len(entries)} selected, {len(entries) - len(pending)} already answered, "
          f"{len(pending)} to run", flush=True)
    if not pending:
        return 0

    _write_atomic(args.results_dir / "run_metadata.json", {
        "created_at": _utc_now(),
        "agent": "v4",
        "code_sha": git_sha(),
        "qwen_model": args.qwen_model,
        "d4rt_checkpoint": str(args.d4rt_checkpoint),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sampling": manifest["sampling"],
    })

    qwen = OfflineQwen(args.qwen_model, max_new_tokens=args.max_new_tokens, seed=args.seed)
    pipeline = QuestionPipeline(
        run_dir=args.results_dir,
        models=Models(d4rt_config=args.d4rt_config, d4rt_checkpoint=args.d4rt_checkpoint),
        lean=args.lean,
    )
    orchestrator = AgentV4Orchestrator(qwen=qwen, pipeline=pipeline, max_steps=args.max_steps)

    failures = 0
    deadline = (time.monotonic() + args.max_seconds) if args.max_seconds else None
    for index, entry in enumerate(pending, start=1):
        # Stop BETWEEN questions when the budget is spent, so the in-flight
        # answer is written rather than the process being killed mid-write by
        # the scheduler. Exits 0: an out-of-time run is incomplete, not failed.
        if deadline is not None and time.monotonic() >= deadline:
            print(f"\nreached the time budget after {index - 1} of {len(pending)} "
                  f"pending questions; stopping cleanly", flush=True)
            break
        question_id = entry["question_id"]
        print(f"\n[{index}/{len(pending)}] {question_id}", flush=True)
        started = datetime.now(timezone.utc)
        try:
            solution = orchestrator.solve(_task_for(entry))
            # setdefault, not an override: solve() marks its own status when it
            # caught something internally, and stamping "ok" over that would hide
            # a failed question in the results.
            record = dict(solution)
            record.setdefault("status", "ok")
        except Exception:
            # One question's failure must not cost the rest of a multi-hour GPU
            # job: the questions are independent, and a clip that will not decode
            # is no reason to abandon 199 others.
            record = {
                "question_id": question_id, "status": "error",
                "error": traceback.format_exc(), "answered": False,
            }

        # Counted from the record so both paths -- the orchestrator's own caught
        # failure and an escape from it -- are tallied the same way, once.
        if record.get("status") == "error":
            failures += 1
            last_line = str(record.get("error", "")).strip().splitlines()[-1:]
            print(f"    FAILED: {last_line[0] if last_line else 'unknown'}", flush=True)

        record["wall_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
        # Scoring-side facts, kept out of the model's task on purpose.
        record["reference"] = {
            "cate": entry["cate"], "category_name": entry["category_name"],
            "dataset": entry["dataset"], "gt": entry["gt"],
            "options": entry["options"],
            "expected_bucket": CATE_TO_BUCKET.get(entry["cate"]),
        }
        _write_atomic(answers / f"{question_id}.json", record)

        if record.get("answered"):
            agreed = record.get("chosen_bucket") == record["reference"]["expected_bucket"]
            print(f"    bucket={record.get('chosen_bucket')} "
                  f"({'matches' if agreed else 'DIFFERS FROM'} cate {entry['cate']}), "
                  f"measured={record.get('measurement_available')}, "
                  f"{record['wall_seconds']:.0f}s", flush=True)
            print(f"    answer: {str(record.get('answer_text'))[:160]}", flush=True)

    print(f"\ndone: {len(pending) - failures}/{len(pending)} answered, {failures} errored",
          flush=True)
    return 1 if failures == len(pending) else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsi-root", type=Path, default=DEFAULT_DSI_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--seed-sample", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-balance-gt", action="store_true")
    parser.add_argument(
        "--random-sample", type=int, nargs="?", const=DEFAULT_RANDOM_SIZE, default=None,
        help=f"draw N questions uniformly instead of the Latin-rectangle 25 "
             f"(default N={DEFAULT_RANDOM_SIZE})",
    )
    parser.add_argument("--dry-run", action="store_true", help="check clips decode; no model")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true", help="re-run answered questions")
    parser.add_argument("--qwen-model", default=os.environ.get("MODEL_DIR", DEFAULT_QWEN_MODEL))
    parser.add_argument("--d4rt-config", type=Path, default=D4RT_CONFIG)
    parser.add_argument("--d4rt-checkpoint", type=Path, default=D4RT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--lean", action="store_true",
        help="delete frames/masks/overlays once a question is measured; keeps the "
             "measurement itself. ~75x less disk, needed for a full-census run.",
    )
    parser.add_argument(
        "--max-seconds", type=int, default=0,
        help="stop starting new questions after N seconds (0 = no limit). Lets a "
             "SLURM job finish its current question and exit cleanly before the wall clock.",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest = resolve_manifest(args)
    entries = _select(manifest, args)
    if not entries:
        raise SystemExit("no questions selected")
    sys.exit(run_dry(manifest, entries) if args.dry_run else run_agent(args, manifest, entries))


if __name__ == "__main__":
    main()
