"""The calibration catalog: every rule-parsable question on a random sample of clips.

The 10-question catalog shows each branch once; it cannot tune a threshold. This one takes
a video-level random sample of clips from the sources whose time axis is established, and
keeps *all* the templated questions on them, so questions of one clip share one Point4D
pass and one SAM3 session. Clips are split dev / test by a hash of the clip name: every
threshold is chosen on dev and reported on test, and a clip never sits in both.

Runs in ``envs/spatialstack-qwen35`` (parquet)::

    python -m dyn_dataset_exploration.decomposition.build_dev_catalog
    python -m dyn_dataset_exploration.decomposition.build_dev_catalog --materialize --workers 8
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence

from dyn_dataset_exploration.dyn_bench_data import DEFAULT_DYN_ROOT, load_rows, qa_parquet, write_manifest
from dyn_dataset_exploration.decomposition.build_catalog import clip_record
from dyn_dataset_exploration.decomposition.program import rule_parse

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "decomposition_dev"
PER_DATASET = {"ytvis": 70, "sav": 60, "davis": 40, "dynpose": 40, "dynamic_replica": 12}
MEASURED_KINDS = {
    "camera_path", "camera_shake", "cam_subject_range", "subject_direction", "pair_relation",
    "metric_distance", "instance_count", "event_order", "landmark_relation",
    # iteration 2
    "camera_rhythm", "camera_zoom", "scene_revisit", "cam_subject_elevation", "apparent_size",
    "subject_visibility", "subject_motion_profile",
    # iteration 3
    "metric_speed", "camera_translation",
}
RARE_KINDS = {"subject_direction", "pair_relation", "instance_count", "metric_distance"}
MAX_SIDE = 1280
SEED = 20260929


def split_of(slug: str) -> str:
    return "dev" if int(hashlib.sha1(slug.encode()).hexdigest()[:4], 16) % 2 == 0 else "test"


THIN_KINDS = {"camera_translation", "subject_motion_profile", "pair_relation", "subject_direction",
              "metric_speed", "metric_distance", "instance_count"}


def build(dyn_root: Path, out: Path, seed: int = SEED, extend: bool = False, heldout: bool = False) -> dict[str, Any]:
    rows = load_rows(qa_parquet(dyn_root))
    by_clip: dict[str, list[Any]] = {}
    programs = {}
    for row in rows:
        if row.dataset not in PER_DATASET:
            continue
        program = rule_parse(row.question_id, row.question, row.options)
        if program is None or program.kind not in MEASURED_KINDS:
            continue
        programs[row.question_id] = program
        by_clip.setdefault(row.video_key, []).append(row)

    rng = random.Random(seed)
    chosen: list[str] = []
    heldout_clips: set[str] = set()
    if extend:
        # keep exactly the clips of the existing catalog: the GPU stages have (or will have) run
        # on them, and new questions on the same clips need no new Point4D pass
        existing = json.loads((out / "catalog.json").read_text())["questions"]
        chosen = sorted({f"{e['dataset']}/{e['video']}" for e in existing})
        if heldout:
            # every not-yet-used clip that holds a question of a thinly covered kind: a fresh set that no
            # constant or policy was fitted on, to verify the finished pipeline and thicken those kinds
            fresh = sorted(k for k, rows_ in by_clip.items() if k not in set(chosen)
                           and any(programs[r.question_id].kind in THIN_KINDS for r in rows_))
            chosen = sorted(set(chosen) | set(fresh))
            heldout_clips = set(fresh)
    else:
        for dataset, count in PER_DATASET.items():
            pool = sorted(k for k in by_clip if k.startswith(dataset + "/"))
            rng.shuffle(pool)
            chosen += pool[:count]

    # The rare kinds carry the conventions being calibrated (direction words, metric scale,
    # pair distances, counts) and a random draw leaves too few of them, so every clip with one
    # is added.
    if not extend:
        rare = {k for k, rows_ in by_clip.items()
                if any(programs[r.question_id].kind in RARE_KINDS for r in rows_)}
        chosen = sorted(set(chosen) | rare)
    chosen = sorted(chosen)

    entries: list[dict[str, Any]] = []
    clips: dict[str, dict[str, Any]] = {}
    for key in sorted(chosen):
        dataset, video = key.split("/", 1)
        record = clip_record(dyn_root, dataset, video)
        if key not in by_clip:
            continue
        scale = min(1.0, MAX_SIDE / max(record["source_hw"]))
        record["frames_hw"] = [max(1, round(record["source_hw"][0] * scale)), max(1, round(record["source_hw"][1] * scale))]
        record["split"] = "heldout" if key in heldout_clips else split_of(record["clip_slug"])
        clips[key] = record
        for row in by_clip[key]:
            program = programs[row.question_id]
            entries.append({
                "qid": row.question_id, "branch": program.kind, "rule_kind": program.kind,
                "task": row.task, "dataset": row.dataset, "video": row.video,
                "clip_slug": record["clip_slug"], "split": record["split"],
                "question": row.question, "options": row.options, "answer": row.answer,
            })
            write_manifest(program.to_dict(), out / "programs" / f"{row.question_id}.rules.json")
            # the GPU stages fill the pipeline program in (a named main subject); an extension keeps them
            target = out / "programs" / f"{row.question_id}.pipeline.json"
            if not target.exists():
                write_manifest(program.to_dict(), target)
    for record in clips.values():
        write_manifest(record, out / "clips" / record["clip_slug"] / "clip.json")
    write_manifest({
        "purpose": "calibration catalog: all rule-parsable questions on a video-level random sample",
        "seed": seed, "per_dataset": PER_DATASET, "dyn_root": str(dyn_root),
        "branches": sorted(MEASURED_KINDS), "questions": entries,
    }, out / "catalog.json")
    counts = Counter((e["split"], e["branch"]) for e in entries)
    return {"clips": len(clips), "questions": len(entries), "out": str(out),
            "by_split": dict(Counter(e["split"] for e in entries)),
            "by_kind": dict(Counter(e["branch"] for e in entries)),
            "by_split_kind": {f"{s}/{k}": v for (s, k), v in sorted(counts.items())}}


def _materialize(job: tuple[str, str, list[str], list[int]]) -> int:
    from PIL import Image

    frames_dir, out_dir, names, hw = job
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    done = 0
    for index, name in enumerate(names):
        target = dest / f"{index:06d}.jpg"
        if target.exists():
            done += 1
            continue
        with Image.open(Path(frames_dir) / name) as image:
            image = image.convert("RGB")
            if (image.height, image.width) != (hw[0], hw[1]):
                image = image.resize((hw[1], hw[0]), Image.BILINEAR)
            image.save(target, quality=92)
        done += 1
    return done


def materialize(out: Path, workers: int) -> int:
    jobs = []
    for path in sorted((out / "clips").glob("*/clip.json")):
        record = json.loads(path.read_text())
        jobs.append((record["frames_dir"], str(path.parent / "frames"), record["gf"]["names"], record["frames_hw"]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(_materialize, jobs))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dyn-root", type=Path, default=DEFAULT_DYN_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--extend", action="store_true",
                        help="keep the existing catalog's clips and add every measured-kind question on them")
    parser.add_argument("--heldout", action="store_true",
                        help="with --extend: add the clips holding thin-kind questions as split 'heldout'")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.materialize:
        print("frames written:", materialize(args.out, args.workers))
    else:
        print(json.dumps(build(args.dyn_root, args.out, extend=args.extend, heldout=args.heldout), indent=2))


if __name__ == "__main__":
    main()
