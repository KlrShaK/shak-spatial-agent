"""Seed taxonomy, exhaustive question-template scan, and stratified sampling.

This module never touches the network. It answers two separate questions:

1. ``templates`` -- does every DSI-Bench question fall into a small number of
   fixed templates, and which ``cate`` values share one?  This is a
   deterministic scan over the *entire* split (not a sample): presence or
   absence of a question shape is a fact about the dataset, not something
   worth guessing at statistically.
2. ``sample`` -- draw a small, category-balanced set of questions (200 total,
   ~33-34 per category) to send, one at a time, to a blind LLM classifier
   (``taxonomy_classify.py``) against the nine-bucket seed taxonomy below.
   Only this sample -- never the full 1769-row dataset -- is ever sent to an
   LLM; the template scan above is local regex, no API calls at all.

Run as ``python -m dsi_dataset_exploration.taxonomy_data <command>`` from the
repo root.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import re
from typing import Mapping, Sequence

from d4rt_agent.dsi_bench_data import (
    DEFAULT_DSI_ROOT,
    DEFAULT_SPLIT,
    DSIRow,
    file_sha256,
    load_rows,
    metadata_csv,
    video_root,
    video_slug,
    write_manifest,
)

DEFAULT_RESULTS_DIR = Path("dsi_dataset_exploration/logs")
# A fresh seed for this exploration, distinct from dsi_bench_data.py's own
# DEFAULT_SEED -- the two samples serve different purposes and should not be
# confused for the same draw.
DEFAULT_SEED = 20260825

# --------------------------------------------------------------------------- #
# Seed taxonomy -- tool/measurement-centric, not motion-vocabulary-centric.
# "What physical quantity, in what reference frame, does this question
# actually measure?" -- verbatim from the researcher's own nine buckets.
# --------------------------------------------------------------------------- #

BUCKETS: dict[str, str] = {
    "subject_own_frame": "Trajectory of the subject relative to its own frame",
    "subject_observer_frame": "Trajectory of the subject relative to the observer/camera",
    "distance_change": "Change of distance between the subject and the observer",
    "camera_own_frame": "Trajectory of the camera/observer relative to its own frame",
    "camera_subject_frame": "Trajectory of the camera/observer relative to the subject",
    "observer_orient_rel_subject": (
        "Orientation change of the observer relative to the subject "
        "(OpenCV coordinate convention)"
    ),
    "subject_orient_over_time": "Change of orientation of the subject through time",
    "observer_orient_over_time": "Change of orientation of the observer through time",
    "subject_orient_rel_observer": (
        "Orientation change of the subject relative to the orientation of the observer"
    ),
}
OTHER_BUCKET = "OTHER"
BUCKET_KEYS: tuple[str, ...] = (*BUCKETS.keys(), OTHER_BUCKET)

# Hypothesis, not fact: which bucket each DSI-Bench `cate` is expected to test,
# derived from the exhaustive template scan below and a geometric reading of
# cate 5's question. cate 5 is the one entry to actively confirm via the LLM
# pass and human review, not accept on this comment alone -- its question reads
# "observer's position ... relative to the subject's own orientation", i.e. the
# observer's *position* over time expressed in a frame the subject's facing
# defines. That is camera_subject_frame, not either party's own orientation
# changing (observer_orient_rel_subject was the considered and rejected
# alternative).
CATE_TO_BUCKET_HYPOTHESIS: dict[int, str] = {
    0: "subject_own_frame",     # Obj:static cam
    1: "subject_own_frame",     # Obj:moving cam -- same quantity; camera motion is the confound
    2: "camera_own_frame",      # Cam:static scene
    3: "camera_own_frame",      # Cam:dynamic scene -- same quantity; moving objects are the confound
    4: "distance_change",       # Obj-Cam distance
    5: "camera_subject_frame",  # Obj-Cam orientation -- tentative, see comment above
}


# --------------------------------------------------------------------------- #
# Exhaustive question-template scan (no network, no sampling -- every row)
# --------------------------------------------------------------------------- #

# Subject nouns seen across std.csv's questions, stripped so only the template
# skeleton remains. Deliberately over-inclusive (matches nouns that never occur
# in some categories) -- a miss here just leaves one noun in the output, still
# recognizable as belonging to the same template.
_SUBJECT_NOUN_PATTERN = re.compile(
    r"\b(character|woman|man|kid|person|camera|dog|cat|boy|girl|car|horse|plane|"
    r"motorbike|race car|boat|cyclist|truck|seal|sailboat|skier|vehicle)\b",
    re.IGNORECASE,
)
# Catches the long tail of one-off nouns (e.g. "the raccoon's", "the toy_plane's")
# that the fixed list above does not name individually.
_POSSESSIVE_NOUN_PATTERN = re.compile(r"\bthe\s+[a-zA-Z][a-zA-Z '_-]*?'s\b")


def _normalize_template(question: str) -> str:
    """Collapse a question to its template skeleton by stripping the subject noun."""

    normalized = _POSSESSIVE_NOUN_PATTERN.sub("the SUBJ's", question)
    normalized = _SUBJECT_NOUN_PATTERN.sub("SUBJ", normalized)
    return normalized


def distinct_question_templates(rows: Sequence[DSIRow]) -> dict[str, dict[str, object]]:
    """Every distinct question template in ``rows``, with counts and cate breakdown.

    Exhaustive over whatever ``rows`` it is given -- callers wanting the whole-
    dataset finding must pass every row, not a sample. Returns
    ``{template: {"count": n, "by_cate": {cate: n, ...}}}``, sorted by count
    descending.
    """

    counts: Counter[str] = Counter()
    by_cate: dict[str, Counter[int]] = {}
    for row in rows:
        template = _normalize_template(row.question)
        counts[template] += 1
        by_cate.setdefault(template, Counter())[row.cate] += 1

    return {
        template: {"count": count, "by_cate": dict(sorted(by_cate[template].items()))}
        for template, count in counts.most_common()
    }


# --------------------------------------------------------------------------- #
# Category-stratified sampling
# --------------------------------------------------------------------------- #

SAMPLE_SIZE = 200
# Sums to 200. Deliberately not proportional to the benchmark's own skew (cate 1
# has 582 rows, cate 5 only 85) -- every category needs enough sampled questions
# to say something about its own bucket, not just the largest ones.
CATEGORY_QUOTAS: dict[int, int] = {0: 33, 1: 34, 2: 33, 3: 34, 4: 33, 5: 33}


def _candidates(rows: Sequence[DSIRow], cate: int) -> list[DSIRow]:
    # Sorted explicitly so the draw does not depend on CSV row order.
    return sorted(
        (row for row in rows if row.cate == cate),
        key=lambda row: (row.relative_path, row.csv_row_index),
    )


def stratified_sample(
    rows: Sequence[DSIRow], seed: int, quotas: Mapping[int, int] = CATEGORY_QUOTAS
) -> list[DSIRow]:
    """Draw ``quotas[cate]`` rows per category, distinct videos within a category.

    Unlike ``dsi_bench_data.sample_rows`` (dataset-forced Latin rectangle at
    n=25) or ``census_rows`` (every row), this draws uniformly within each
    category only -- the taxonomy pass has no reason to force one question per
    (dataset, category) cell, only enough per category to test its bucket.
    """

    rng = random.Random(seed)
    selected: list[DSIRow] = []
    for cate, quota in sorted(quotas.items()):
        pool = _candidates(rows, cate)
        shuffled = pool[:]
        rng.shuffle(shuffled)
        used_videos: set[str] = set()
        chosen: list[DSIRow] = []
        for row in shuffled:
            if row.relative_path in used_videos:
                continue
            used_videos.add(row.relative_path)
            chosen.append(row)
            if len(chosen) == quota:
                break
        if len(chosen) < quota:
            raise ValueError(
                f"cate {cate} has only {len(chosen)} distinct-video candidates, need {quota}"
            )
        selected.extend(chosen)
    return selected


def build_taxonomy_manifest(
    dsi_root: Path = DEFAULT_DSI_ROOT,
    split: str = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    quotas: Mapping[int, int] = CATEGORY_QUOTAS,
) -> dict[str, object]:
    """Select the taxonomy sample and describe the selection well enough to repeat it."""

    csv_path = metadata_csv(dsi_root, split)
    videos = video_root(dsi_root, split)
    rows = load_rows(csv_path)
    selected = stratified_sample(rows, seed, quotas)

    entries: list[dict[str, object]] = []
    for row in selected:
        entries.append(
            {
                "cate": row.cate,
                "category_name": row.category_name,
                "dataset": row.dataset,
                "relative_path": row.relative_path,
                "question": row.question,
                "options": row.options,
                "option_letters": row.option_letters,
                "gt": row.gt,
                "others": row.others,
                "question_id": row.question_id,
                "video_slug": video_slug(row.relative_path),
                "video_path": str(videos / row.relative_path),
                "hypothesis_bucket": CATE_TO_BUCKET_HYPOTHESIS[row.cate],
            }
        )

    per_category: dict[str, int] = {}
    for row in selected:
        per_category[row.category_name] = per_category.get(row.category_name, 0) + 1

    return {
        "purpose": (
            "dsi_dataset_exploration taxonomy sample -- category-stratified, not "
            "proportional to the benchmark's own skew"
        ),
        "split": split,
        "dsi_root": str(dsi_root),
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
        "sampling": {
            "design": "category_stratified",
            "description": (
                "quotas[cate] rows drawn uniformly within each category; "
                "distinct videos within a category, not across categories"
            ),
            "seed": seed,
            "quotas": dict(quotas),
            "total_selected": len(selected),
        },
        "counts": {"per_category": per_category},
        "questions": entries,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cmd_templates(args: argparse.Namespace) -> int:
    csv_path = metadata_csv(args.dsi_root, args.split)
    rows = load_rows(csv_path)
    templates = distinct_question_templates(rows)
    payload = {
        "split": args.split,
        "csv_path": str(csv_path),
        "total_rows": len(rows),
        "distinct_templates": len(templates),
        "templates": templates,
    }
    out_path = Path(args.results_dir) / "templates.json"
    write_manifest(payload, out_path)
    print(f"{len(rows)} rows -> {len(templates)} distinct templates (written to {out_path})")
    for template, info in templates.items():
        print(f"  [{info['count']:4d}] cate={info['by_cate']} {template[:100]}")
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    manifest = build_taxonomy_manifest(
        dsi_root=args.dsi_root, split=args.split, seed=args.seed, quotas=CATEGORY_QUOTAS
    )
    out_path = Path(args.results_dir) / "manifest.json"
    write_manifest(manifest, out_path)
    print(f"wrote {manifest['sampling']['total_selected']} questions to {out_path}")
    for name, count in manifest["counts"]["per_category"].items():
        print(f"  {name}: {count}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsi-root", type=Path, default=DEFAULT_DSI_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p_templates = sub.add_parser(
        "templates", help="exhaustive question-template scan over every row (no network)"
    )
    p_templates.set_defaults(func=_cmd_templates)

    p_sample = sub.add_parser(
        "sample", help="draw the 200-question category-stratified sample (no network)"
    )
    p_sample.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_sample.set_defaults(func=_cmd_sample)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
