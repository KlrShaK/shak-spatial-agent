"""Resolve the decomposition catalog: its questions, their clips, and the time axis.

Runs in ``envs/spatialstack-qwen35`` because it reads the parquet::

    python -m dyn_dataset_exploration.decomposition.build_catalog --materialize

Writes under ``dyn_dataset_exploration/outputs/decomposition/``:

- ``catalog.json``: the first-iteration questions, one or two per branch of the decision
  graph (the answer is stored; the notebook hides it until the last step)
- ``gf_check.json``: the generator-frame rebuild over all 1000 clips
- ``time_axis_evidence.json``: per source, question timestamps against clip ends
- ``clips/<clip_slug>/clip.json``: GF files, frame numbers, seconds, the 16 samples, size
- ``clips/<clip_slug>/frames/000000.jpg ...``: the GF frames, JPEG, one index space shared
  by SAM3, Point4D and the notebook (``--materialize``)
- ``programs/<qid>.rules.json``: the rule parse, where a rule matches
- ``programs/<qid>.reference.json``: the hand-authored reference program
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence

from dyn_dataset_exploration.dyn_bench_data import (
    DEFAULT_DYN_ROOT,
    DynRow,
    load_rows,
    qa_parquet,
    video_slug,
    write_manifest,
)
from dyn_dataset_exploration.decomposition.program import rule_parse, timestamps
from dyn_dataset_exploration.decomposition.reference_programs import REFERENCE, reference_program
from dyn_dataset_exploration.decomposition.scene4d import (
    frame_rate,
    generator_frames,
    list_frames,
    seconds_axis,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "decomposition"

# One or two questions per branch of the decision graph. Chosen from the labelled
# 600-question sample, from sources whose time axis is established, so that together they
# exercise every stage: camera-only vocabularies, an implicit main subject, one entity, a
# group of two, a metric instant, a category count, two-branch event order, a region
# landmark (leaf B) and a semantic question (leaf C). Two clips carry two questions each,
# which is also how the per-clip cache gets exercised.
CATALOG: dict[str, list[str]] = {
    "camera_vocab": [
        "cameraqa_davis_lab_coat_6b077a14_0",
        "cameraqa_dynamic_replica_04ff5e_4_obj_source_left_f2cef79b_2",
    ],
    "cam_subject_range": ["cameraqa_dynpose_32439b6a_737d_473b_b7fb_3aead1f6d0c4_069c8138_1"],
    "subject_direction": ["qa_ytvis_1b0ad425e5_b6f634ee_0"],
    "pair_relation": ["qa_dynamic_replica_04ff5e_4_obj_source_left_f2cef79b_0"],
    "metric_distance": ["qa_dynpose_32439b6a_737d_473b_b7fb_3aead1f6d0c4_069c8138_0"],
    "instance_count": ["qa_sav_017974_manual_818c7cc6_1"],
    "event_order": ["cameraqa_sav_026989_manual_8cbd9e78_2"],
    "narrative": ["sceneqa_davis_boat_d573aaba_1"],
    "semantic": ["qa_sav_000040_manual_b7fd5545_0"],
}

CANDIDATE_FPS = (5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 24.0, 25.0, 30.0, 60.0)
# "end (1.63s)", "the end (t≈15s)": a timestamp that names the last GF frame pins the rate.
_END_TIME = re.compile(r"\bend\b[^0-9]{0,12}(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def clip_record(dyn_root: Path, dataset: str, video: str) -> dict[str, Any]:
    """Everything the later stages need to know about one clip, from disk alone."""

    from PIL import Image

    clip = f"{dataset}_{video}"
    frames_dir = Path(dyn_root) / "frames" / dataset / clip
    sampling = json.loads(
        (Path(dyn_root) / "multi_json" / dataset / f"{clip}_frame_sampling.json").read_text()
    )
    gf = generator_frames(list_frames(frames_dir), sampling)
    rate = frame_rate(dataset)
    with Image.open(frames_dir / gf.names[0]) as image:
        width, height = image.size

    masks_dir = Path(dyn_root) / "masks" / dataset / clip
    objects: Counter[str] = Counter()
    if masks_dir.is_dir():
        for name in os.listdir(masks_dir):
            match = re.search(r"_obj(\d+)\.png$", name)
            if match:
                objects[f"object_{int(match.group(1))}"] += 1

    return {
        "dataset": dataset,
        "video": video,
        "video_key": f"{dataset}/{video}",
        "clip_slug": video_slug(dataset, video),
        "frames_dir": str(frames_dir),
        "gf": {
            "names": list(gf.names),
            "native": list(gf.native),
            "mode": gf.mode,
            "stride": gf.stride,
            "length": gf.length,
            "reported_total": gf.reported_total,
        },
        "sampled16": list(gf.sampled),
        "fps_gf": rate.fps,
        "fps_status": rate.status,
        "fps_evidence": rate.evidence,
        "t_s": seconds_axis(gf.length, rate.fps),
        "duration_s": None if rate.fps is None else round((gf.length - 1) / rate.fps, 4),
        "source_hw": [height, width],
        "portrait": height > width,
        # Diagnostics only: the GT masks answer the benchmark's grounding track and must never
        # feed a VQA answer.
        "gt_masks_dir": str(masks_dir) if masks_dir.is_dir() else None,
        "gt_objects": dict(sorted(objects.items())),
    }


def materialize_frames(record: Mapping[str, Any], clip_dir: Path) -> int:
    """Write the GF as ``frames/%06d.jpg``: JPEG sources are linked, others converted.

    SAM3's video predictor reads a folder of JPEGs, and a frame index has to mean the same
    frame in SAM3, Point4D and the notebook, so every consumer reads this folder.
    """

    from PIL import Image

    out = clip_dir / "frames"
    out.mkdir(parents=True, exist_ok=True)
    source = Path(record["frames_dir"])
    for index, name in enumerate(record["gf"]["names"]):
        target = out / f"{index:06d}.jpg"
        if target.exists() or target.is_symlink():
            continue
        if Path(name).suffix.lower() in {".jpg", ".jpeg"}:
            os.symlink(source / name, target)
        else:
            with Image.open(source / name) as image:
                image.convert("RGB").save(target, quality=95)
    return len(record["gf"]["names"])


def gf_check(dyn_root: Path) -> dict[str, Any]:
    """Rebuild the GF of every clip; any disagreement with the shipped sample is listed."""

    modes: Counter[str] = Counter()
    lengths: dict[str, list[int]] = defaultdict(list)
    reported_differs: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    per_clip: dict[str, int] = {}
    for dataset in sorted(os.listdir(Path(dyn_root) / "multi_json")):
        folder = Path(dyn_root) / "multi_json" / dataset
        for name in sorted(os.listdir(folder)):
            if not name.endswith("_frame_sampling.json"):
                continue
            clip = name[: -len("_frame_sampling.json")]
            sampling = json.loads((folder / name).read_text())
            try:
                gf = generator_frames(list_frames(Path(dyn_root) / "frames" / dataset / clip), sampling)
            except Exception as error:  # noqa: BLE001 -- report every clip, never abort
                failures.append({"clip": clip, "error": str(error)})
                continue
            modes[f"{dataset}:{gf.mode}:{gf.stride}"] += 1
            lengths[dataset].append(gf.length)
            per_clip[f"{dataset}/{clip[len(dataset) + 1:]}"] = gf.length
            if gf.reported_total != gf.length:
                reported_differs[dataset] += 1
    return {
        "purpose": "generator-frame list rebuilt from shipped frame_paths, for every clip",
        "clips": sum(modes.values()) + len(failures),
        "failures": failures,
        "modes": dict(sorted(modes.items())),
        "reported_total_differs": dict(sorted(reported_differs.items())),
        "length_by_dataset": {
            ds: {"min": min(v), "median": statistics.median(v), "max": max(v)}
            for ds, v in sorted(lengths.items())
        },
        "gf_length": per_clip,
    }


def time_axis_evidence(rows: Sequence[DynRow], gf_length: Mapping[str, int]) -> dict[str, Any]:
    """Per source: how many question timestamps would fall past the clip at each rate.

    A rate under which timestamps overflow the clip is ruled out; a question that names
    the end of the clip ("the end (t≈15s)") pins the rate as ``(L - 1) / t``.
    """

    past_end: dict[str, Counter[float]] = defaultdict(Counter)
    counted: Counter[str] = Counter()
    end_rates: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        length = gf_length.get(row.video_key)
        values = [t for t in timestamps(row.question) if t > 0]
        if not length or not values:
            continue
        counted[row.dataset] += len(values)
        for fps in CANDIDATE_FPS:
            past_end[row.dataset][fps] += sum(1 for t in values if t > (length - 1) / fps + 0.25)
        end = _END_TIME.search(row.question)
        if end and float(end.group(1)) > 0:
            end_rates[row.dataset].append(round((length - 1) / float(end.group(1)), 2))
    report = {}
    for dataset in sorted(counted):
        rate = frame_rate(dataset)
        report[dataset] = {
            "fps_used": rate.fps,
            "status": rate.status,
            "timestamps": counted[dataset],
            "past_end_at_fps": {str(fps): past_end[dataset][fps] for fps in CANDIDATE_FPS},
            "end_anchored_implied_fps": sorted(end_rates[dataset]),
        }
    return {
        "purpose": "which seconds-per-frame rate each source's question timestamps are consistent with",
        "rule": "a rate is ruled out when timestamps fall past (L-1)/fps + 0.25 s",
        "by_dataset": report,
    }


def build(dyn_root: Path, out: Path, materialize: bool, skip_gf_check: bool) -> dict[str, Any]:
    rows = load_rows(qa_parquet(dyn_root))
    by_id = {row.question_id: row for row in rows}
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    if not skip_gf_check:
        check = gf_check(dyn_root)
        write_manifest(check, out / "gf_check.json")
        write_manifest(time_axis_evidence(rows, check["gf_length"]), out / "time_axis_evidence.json")
        summary["gf_failures"] = len(check["failures"])

    entries: list[dict[str, Any]] = []
    clips: dict[str, dict[str, Any]] = {}
    for branch, qids in CATALOG.items():
        for qid in qids:
            row = by_id.get(qid)
            if row is None:
                raise KeyError(f"catalog question {qid} is not in the parquet")
            record = clips.get(row.video_key) or clip_record(dyn_root, row.dataset, row.video)
            clips[row.video_key] = record
            program = rule_parse(qid, row.question, row.options)
            if program is not None:
                write_manifest(program.to_dict(), out / "programs" / f"{qid}.rules.json")
            if qid in REFERENCE:
                write_manifest(reference_program(qid, row.options).to_dict(),
                               out / "programs" / f"{qid}.reference.json")
            entries.append({
                "qid": qid,
                "branch": branch,
                "rule_kind": program.kind if program else None,
                "task": row.task,
                "dataset": row.dataset,
                "video": row.video,
                "clip_slug": record["clip_slug"],
                "question": row.question,
                "options": row.options,
                "answer": row.answer,
            })

    for record in clips.values():
        clip_dir = out / "clips" / record["clip_slug"]
        if materialize:
            record["materialized_frames"] = materialize_frames(record, clip_dir)
        write_manifest(record, clip_dir / "clip.json")

    write_manifest({
        "purpose": "first-iteration decomposition catalog: one or two questions per branch",
        "dyn_root": str(dyn_root),
        "branches": list(CATALOG),
        "questions": entries,
    }, out / "catalog.json")
    summary.update({"questions": len(entries), "clips": len(clips), "out": str(out)})
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dyn-root", type=Path, default=DEFAULT_DYN_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--materialize", action="store_true", help="write the GF frames as JPEGs")
    parser.add_argument("--skip-gf-check", action="store_true", help="skip the 1000-clip rebuild")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(build(args.dyn_root, args.out, args.materialize, args.skip_gf_check), indent=2))


if __name__ == "__main__":
    main()
