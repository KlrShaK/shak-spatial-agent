"""A representative evaluation set: random clips of the whole benchmark, every question on them.

The calibration catalog holds only questions the rules can read, on clips chosen partly for the kinds
they contain. This one is a clip-level random sample of Dyn-Bench, stratified by source dataset in
proportion to the benchmark, with **all seven questions of each sampled clip** -- templated or not --
so its accuracy estimates the pipeline's accuracy on Dyn-Bench itself.

The clips that the measurement constants and the policy were fitted on (the dev split) are left out of
the pool, so nothing here is in-sample. Clips already processed for the calibration catalog (test and
held-out clips) are reused through symlinks, with their scenes, groundings and Qwen answers.

    sbatch scripts/run_cpu_task.slurm -m dyn_dataset_exploration.decomposition.build_eval_catalog [--clips 400]
    ... --materialize            # frames of the new clips
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Sequence

from dyn_dataset_exploration.decomposition.build_catalog import clip_record
from dyn_dataset_exploration.decomposition.build_dev_catalog import MAX_SIDE, materialize
from dyn_dataset_exploration.decomposition.program import rule_parse
from dyn_dataset_exploration.dyn_bench_data import DEFAULT_DYN_ROOT, load_rows, qa_parquet, write_manifest

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
DEV = OUTPUTS / "decomposition_dev"
DEFAULT_OUT = OUTPUTS / "decomposition_eval"
SEED = 20260930


def sample_clips(clips: dict[str, list], excluded: set[str], total: int, seed: int) -> list[str]:
    """Clip keys, ``total`` of them, in proportion to each dataset's share of the non-excluded pool."""

    pool: dict[str, list[str]] = defaultdict(list)
    for key in sorted(clips):
        if key not in excluded:
            pool[key.split("/")[0]].append(key)
    size = sum(len(v) for v in pool.values())
    rng = random.Random(seed)
    chosen: list[str] = []
    for dataset, keys in sorted(pool.items()):
        quota = min(len(keys), max(1, round(total * len(keys) / size)))
        rng.shuffle(keys)
        chosen += keys[:quota]
    return sorted(chosen)


def build(dyn_root: Path, out: Path, total: int, seed: int = SEED) -> dict[str, Any]:
    rows = load_rows(qa_parquet(dyn_root))
    clips: dict[str, list] = defaultdict(list)
    for row in rows:
        clips[row.video_key].append(row)
    dev_catalog = json.loads((DEV / "catalog.json").read_text())["questions"]
    dev_clips = {f"{e['dataset']}/{e['video']}" for e in dev_catalog if e["split"] == "dev"}
    chosen = sample_clips(clips, dev_clips, total, seed)

    entries: list[dict[str, Any]] = []
    reused = 0
    for key in chosen:
        dataset, video = key.split("/", 1)
        record = clip_record(dyn_root, dataset, video)
        scale = min(1.0, MAX_SIDE / max(record["source_hw"]))
        record["frames_hw"] = [max(1, round(record["source_hw"][0] * scale)), max(1, round(record["source_hw"][1] * scale))]
        record["split"] = "eval"
        slug = record["clip_slug"]
        target = out / "clips" / slug
        shared = DEV / "clips" / slug
        if (shared / "point4d" / "scene.npz").exists():       # processed already: reuse scene, groundings, frames
            if not target.exists() and not target.is_symlink():
                (out / "clips").mkdir(parents=True, exist_ok=True)
                os.symlink(shared, target)
            reused += 1
        else:
            write_manifest(record, target / "clip.json")
        for row in sorted(clips[key], key=lambda r: r.question_id):
            try:
                program = rule_parse(row.question_id, row.question, row.options)
            except ValueError:
                program = None
            entries.append({
                "qid": row.question_id, "branch": program.kind if program else "unparsed",
                "rule_kind": program.kind if program else None, "task": row.task, "dataset": row.dataset,
                "video": row.video, "clip_slug": slug, "split": "eval", "question": row.question,
                "options": row.options, "answer": row.answer,
            })
            for sub, suffix in (("programs", "rules"),):
                if program:
                    write_manifest(program.to_dict(), out / sub / f"{row.question_id}.{suffix}.json")
            # keep the programs the GPU stages already finished (named main subjects) and every Qwen answer
            for kind_dir, name in (("programs", f"{row.question_id}.pipeline.json"),
                                   ("answers", f"{row.question_id}.frames_only.json"),
                                   ("answers", f"{row.question_id}.letters.json"),
                                   ("answers", f"{row.question_id}.evidence.json")):
                src, dst = DEV / kind_dir / name, out / kind_dir / name
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            pipe = out / "programs" / f"{row.question_id}.pipeline.json"
            if program and not pipe.exists():
                write_manifest(program.to_dict(), pipe)
    write_manifest({
        "purpose": "representative evaluation: random clips of Dyn-Bench (dev clips excluded), all their questions",
        "seed": seed, "clips": len(chosen), "dyn_root": str(dyn_root), "questions": entries,
    }, out / "catalog.json")
    return {"clips": len(chosen), "reused_clips": reused, "new_clips": len(chosen) - reused, "questions": len(entries),
            "by_dataset": dict(Counter(e["dataset"] for e in entries)), "by_task": dict(Counter(e["task"] for e in entries)),
            "read_by_rules": sum(e["rule_kind"] is not None for e in entries)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dyn-root", type=Path, default=DEFAULT_DYN_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clips", type=int, default=400)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.materialize:
        print("frames written:", materialize(args.out, args.workers))
    else:
        print(json.dumps(build(args.dyn_root, args.out, args.clips), indent=2))


if __name__ == "__main__":
    main()
