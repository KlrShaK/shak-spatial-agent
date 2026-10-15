"""Dyn-Bench VQA with the benchmark's OFFICIAL protocol, made resumable and shardable.

    python -m dyn_dataset_exploration.official_eval --results-dir <dir> --model-path <snapshot> [--shard i/n]
    python -m dyn_dataset_exploration.official_eval --results-dir <dir> --summary

The protocol is the official code, imported unchanged from ``Dynamics-X/Thinking-in-Dynamics``
(cloned at ``/cluster/work/igp_psr/spanwar/Thinking-in-Dynamics``):

- ``utils.load_tasks.load_all_tasks``: all 7,000 questions; every frame file of the clip.
- ``utils.run_qa_task.run_qa_task``: question + options + "Answer with the option's letter from the
  given choices directly.", the frames as one video with ``fps: 24`` metadata, ``max_new_tokens`` 512,
  the model's own generation config (Qwen3-VL samples: temperature 0.7, top-p 0.8, top-k 20).
- ``utils.cal_metrics.compute_qa_accuracy``: the first token of the reply against the answer letter.
- Summary: per level (Inter-Object / Object-Scene / Camera-Object) and "Overall" = mean of the three
  level accuracies, as ``eval.py`` prints it; the question-weighted mean is reported as well.

Only the plumbing differs from ``eval.py``:
- one answer file per question (resume after a time limit, clip-level shards, no all-or-nothing save);
- a seed per question (derived from its id), so sampled answers are reproducible in any shard or order;
- attention ``sdpa`` instead of ``flash_attention_2``: flash-attn is not installed for torch 2.11, and
  PyTorch SDPA runs FlashAttention-2 kernels itself (``eval.py`` would fall back to ``eager``).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

OFFICIAL_REPO = Path("/cluster/work/igp_psr/spanwar/Thinking-in-Dynamics")
DYN_ROOT = Path("/cluster/work/igp_psr/spanwar/datasets/Dyn-Bench")
LEVELS = ("Inter-Object", "Object-Scene", "Camera-Object")


def task_id(task: dict[str, Any]) -> str:
    """Our question id (``{task}_{video_slug}_{index}``, as in the eval catalog), so runs can be paired."""

    from dyn_dataset_exploration.dyn_bench_data import video_slug

    video = task["scene_name"][len(task["dataset"]) + 1:]
    return f"{task['task_type']}_{video_slug(task['dataset'], video)}_{task['question_idx']}"


def load_qwen3_vl(model_path: str) -> dict[str, Any]:
    """``utils.load_model`` for ``qwen3_vl``, with attention ``sdpa`` (see the module doc)."""

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa",
        device_map="auto", local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    # the same keys ``utils.load_model.load_model`` returns (``run_qa_task`` reads ``device``)
    return {"model": model, "tokenizer": None, "processor": processor, "model_type": "qwen3_vl",
            "device": "cuda", "supports_mask": False}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(args.repo))
    from utils.cal_metrics import compute_qa_accuracy
    from utils.load_tasks import load_all_tasks
    from utils.run_qa_task import run_qa_task

    tasks = load_all_tasks(args.dyn_root, "qa")
    scenes = sorted({t["scene_name"] for t in tasks})
    index, count = (int(v) for v in args.shard.split("/"))
    mine = set(scenes[index::count])
    tasks = [t for t in tasks if t["scene_name"] in mine]
    if args.limit:
        tasks = tasks[: args.limit]
    answers = args.results_dir / "answers"

    def done(t: dict[str, Any]) -> bool:          # an answer that ended in an error is retried on resume
        path = answers / f"{task_id(t)}.json"
        return path.exists() and json.loads(path.read_text()).get("error") is None

    pending = [t for t in tasks if not done(t)]
    write_json(args.results_dir / f"run_config.shard{index}of{count}.json", {
        "protocol": "official Dynamics-X/Thinking-in-Dynamics utils (load_all_tasks, run_qa_task, compute_qa_accuracy)",
        "official_commit": os.popen(f"git -C {args.repo} rev-parse HEAD").read().strip(),
        "model_path": args.model_path, "attn_implementation": "sdpa", "seed": args.seed, "shard": args.shard,
        "questions": len(tasks), "pending": len(pending), "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    print(f"shard {args.shard}: {len(tasks)} questions on {len(mine)} clips, {len(pending)} pending", flush=True)
    model_dict = load_qwen3_vl(args.model_path)
    right = 0
    all_failed = [True]                  # stop early if the very first questions all fail (a setup bug)
    for n, task in enumerate(pending, 1):
        qid = task_id(task)
        seed = (args.seed + int(hashlib.sha1(qid.encode()).hexdigest()[:8], 16)) % (2**31)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.time()
        try:
            prediction = run_qa_task(model_dict=model_dict, frame_paths=task["frame_paths"],
                                     question=task["question"], options=task["options"])
            error = None
        except Exception as exc:                       # noqa: BLE001 -- eval.py scores a failure as wrong too
            prediction, error = "", f"{type(exc).__name__}: {exc}"
            if all_failed[0] and n >= 5:
                raise RuntimeError(f"the first 5 questions all failed ({error}); stopping") from exc
        else:
            all_failed[0] = False
        accuracy = compute_qa_accuracy(prediction, task["answer"]) if prediction else 0.0
        right += accuracy
        write_json(answers / f"{qid}.json", {
            "question_id": qid, "dataset": task["dataset"], "scene_name": task["scene_name"],
            "task_type": task["task_type"], "category": task["category"], "question_idx": task["question_idx"],
            "question": task["question"], "options": task["options"], "answer": task["answer"],
            "prediction": prediction, "accuracy": accuracy, "error": error, "seed": seed,
            "n_frames": len(task["frame_paths"]), "seconds": round(time.time() - started, 2),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{n}/{len(pending)}] {qid} pred={prediction[:20]!r} gt={task['answer']} "
              f"{time.time() - started:.1f}s acc={right / n:.3f}", flush=True)


def summary(results_dir: Path, catalog: Path | None) -> str:
    rows = [json.loads(p.read_text()) for p in sorted((results_dir / "answers").glob("*.json"))]
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r["accuracy"])
    level = {k: 100 * sum(by[k]) / len(by[k]) for k in LEVELS if by[k]}
    lines = ["# Qwen3-VL-8B on Dyn-Bench VQA, official protocol", "",
             f"{len(rows)} questions answered; {sum(1 for r in rows if r['error'])} raised an error (scored wrong).", "",
             "| | n | accuracy |", "|---|--:|--:|"]
    for k in LEVELS:
        if k in level:
            lines.append(f"| {k} | {len(by[k])} | {level[k]:.1f} |")
    if len(level) == 3:
        lines.append(f"| **Overall** (mean of the 3 levels, as `eval.py`) | {len(rows)} | **{sum(level.values()) / 3:.1f}** |")
    lines.append(f"| question-weighted mean | {len(rows)} | {100 * sum(r['accuracy'] for r in rows) / max(1, len(rows)):.1f} |")
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r["accuracy"])
    lines += ["", "| dataset | n | accuracy |", "|---|--:|--:|"]
    for k, v in sorted(by_ds.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {k} | {len(v)} | {100 * sum(v) / len(v):.1f} |")
    if catalog and catalog.exists():
        sub = {q["qid"] for q in json.loads(catalog.read_text())["questions"]}
        rs = [r for r in rows if r["question_id"] in sub]
        if rs:
            lines += ["", f"On the 2,800-question eval split used by AgentV5 ({len(rs)} matched): "
                          f"{100 * sum(r['accuracy'] for r in rs) / len(rs):.1f}% question-weighted."]
    text = "\n".join(lines) + "\n"
    (results_dir / "report.md").write_text(text)
    return text


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--model-path", default=os.environ.get("MODEL_DIR"))
    p.add_argument("--repo", type=Path, default=OFFICIAL_REPO)
    p.add_argument("--dyn-root", type=Path, default=DYN_ROOT)
    p.add_argument("--shard", default="0/1", help="i/n over clips")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary", action="store_true")
    p.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parent / "outputs" / "decomposition_eval" / "catalog.json")
    args = p.parse_args(argv)
    if args.summary:
        print(summary(args.results_dir, args.catalog))
    else:
        run(args)


if __name__ == "__main__":
    main()
