"""DSI-Bench question loading, and the two selections we evaluate.

DSI-Bench ships one CSV per video augmentation.  We evaluate the ``std`` split
only, so this module reads ``metadatas/std.csv`` and builds one of two
selections from it.

``latin_rectangle`` is the small, balanced, reproducible subset.  The benchmark's
(dataset x task) grid is badly unbalanced -- cell sizes range from 3 to 305 -- so
a proportional sample would be almost entirely ``internet`` and would miss whole
tasks.  :func:`latin_rectangle_cells` instead spreads the questions evenly: each
of the 5 datasets contributes 5 questions, and each of the 6 tasks receives 4 or
5.  That makes the result a diagnostic across the whole benchmark rather than an
estimate of its headline accuracy.

``census`` is every question in the split -- all 1769 of them.  It needs no seed
and no balance scan, because it selects nothing.  What it buys is coverage; what
it inherits is the benchmark's own skew, which is why results from it are only
readable grouped by dataset and by task.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence


DEFAULT_DSI_ROOT = Path("/cluster/work/igp_psr/spanwar/datasets/DSI-Bench")
DEFAULT_SPLIT = "std"

# Category names come from the benchmark's own reference evaluator (README.md),
# indexed by the integer `cate` column.
CATEGORY_NAMES: tuple[str, ...] = (
    "Obj:static cam",
    "Obj:moving cam",
    "Cam:static scene",
    "Cam:dynamic scene",
    "Obj-Cam distance",
    "Obj-Cam orientation",
)
NUM_CATEGORIES = len(CATEGORY_NAMES)

# The first path segment of `relative_path` identifies the source dataset.
DATASETS: tuple[str, ...] = ("CameraBench", "SynFMC", "internet", "k700", "llava178k")

QUESTIONS_PER_DATASET = 5
TOTAL_QUESTIONS = len(DATASETS) * QUESTIONS_PER_DATASET

# How a manifest chose its questions.  Written into the manifest so a results
# directory always states which of the two runs produced it, and so a report can
# refuse to describe a census as a balanced sample.
DESIGN_LATIN_RECTANGLE = "latin_rectangle"
DESIGN_CENSUS = "census"

DEFAULT_SEED = 20260721
# Scanned seeds must keep every ground-truth letter inside this band.  With only
# 25 four-way questions reviewed by eye, a letter-skewed sample makes a model's
# letter bias easy to mistake for competence.
GT_BALANCE_MIN = 5
GT_BALANCE_MAX = 8

_OPTION_PATTERN = re.compile(r"^([A-Z])\s*[:.]\s*(.+)$")
_SLUG_STRIP = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class DSIRow:
    """One DSI-Bench question with its video and answer key."""

    cate: int
    dataset: str
    relative_path: str
    video_type: str
    question: str
    options: dict[str, str]
    gt: str
    others: str
    csv_row_index: int

    @property
    def option_letters(self) -> list[str]:
        return sorted(self.options)

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES[self.cate]

    @property
    def question_id(self) -> str:
        return f"{self.dataset}_c{self.cate}_{video_slug(self.relative_path)}"


def video_slug(relative_path: str) -> str:
    """Build a filesystem- and URL-safe id for one benchmark video.

    104 of the benchmark's paths contain spaces and the ``llava178k`` ones run
    six levels deep, so the raw path cannot be used as a filename.  The sha1
    suffix keeps two videos that slug identically from colliding.
    """

    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    stem = relative_path.lower().rsplit(".", 1)[0]
    stem = re.sub(r"[\s/]+", "_", stem)
    stem = _SLUG_STRIP.sub("", stem).strip("_")
    # Long stems make unwieldy filenames; the digest carries the uniqueness.
    return f"{stem[-60:].strip('_')}_{digest}"


def parse_options(text: str) -> dict[str, str]:
    """Split the ``options`` column into ``{letter: option text}``.

    Three rows in ``std.csv`` end with a trailing ``;`` that would otherwise
    parse as a fifth, blank option, so empty parts are dropped before matching.
    """

    options: dict[str, str] = {}
    for part in (piece.strip() for piece in text.split(";")):
        if not part:
            continue
        match = _OPTION_PATTERN.match(part)
        if match is None:
            raise ValueError(f"unparsable option {part!r} in {text!r}")
        letter, body = match.group(1), match.group(2).strip()
        if letter in options:
            raise ValueError(f"duplicate option letter {letter!r} in {text!r}")
        if not body:
            raise ValueError(f"empty option body for {letter!r} in {text!r}")
        options[letter] = body
    if len(options) < 2:
        raise ValueError(f"expected at least two options, got {options} from {text!r}")
    return options


def parse_choice_letter(text: str, valid_letters: Iterable[str] | None = None) -> str | None:
    """Recover the option letter a tool-free reply settled on.

    The benchmark's own prompt asks for ``<answer>X</answer>``, but Qwen commonly
    abbreviates that to ``<X>`` or answers with the bare letter, so matching only
    the documented tag silently loses every answer.  Patterns are tried from most
    to least explicit, and a letter outside the question's options is treated as
    no answer rather than guessed at.
    """

    if not text:
        return None
    allowed = {letter.upper() for letter in valid_letters} if valid_letters else None
    stripped = text.strip()
    patterns = (
        r"<answer>\s*([A-Za-z])\s*</answer>",   # the documented form
        # The abbreviation Qwen actually emits, tolerating the unclosed '<B</answer>'
        # it sometimes produces.  Requiring a non-letter after the captured letter
        # is what keeps this from matching the '<answer>' tag itself.
        r"<\s*([A-Za-z])\s*(?![A-Za-z])",
        r"^\**\s*([A-Za-z])\s*[\.\):,]",        # "B." / "B)" / "B:" at the start
        r"^\**\s*([A-Za-z])\**\s*$",            # a bare letter, alone
        r"\banswer\s*(?:is)?\s*[:\-]?\s*([A-Za-z])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, stripped, re.IGNORECASE | re.MULTILINE)
        if match:
            letter = match.group(1).upper()
            if allowed is None or letter in allowed:
                return letter
    return None


def metadata_csv(dsi_root: Path = DEFAULT_DSI_ROOT, split: str = DEFAULT_SPLIT) -> Path:
    return Path(dsi_root) / "metadatas" / f"{split}.csv"


def video_root(dsi_root: Path = DEFAULT_DSI_ROOT, split: str = DEFAULT_SPLIT) -> Path:
    return Path(dsi_root) / "videos" / split


def load_rows(csv_path: Path) -> list[DSIRow]:
    """Read every question from one DSI-Bench metadata CSV.

    ``question`` and ``options`` are quoted and contain commas, so this must go
    through :class:`csv.DictReader` rather than any manual splitting.
    """

    csv_path = Path(csv_path)
    rows: list[DSIRow] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for index, record in enumerate(csv.DictReader(handle)):
            relative_path = record["relative_path"].strip()
            dataset = relative_path.split("/", 1)[0]
            if dataset not in DATASETS:
                raise ValueError(f"unknown dataset {dataset!r} in row {index}")
            options = parse_options(record["options"])
            gt = record["GT"].strip()
            if gt not in options:
                raise ValueError(
                    f"row {index} ground truth {gt!r} is not one of its options {sorted(options)}"
                )
            cate = int(record["cate"])
            if not 0 <= cate < NUM_CATEGORIES:
                raise ValueError(f"row {index} has out-of-range cate {cate}")
            rows.append(
                DSIRow(
                    cate=cate,
                    dataset=dataset,
                    relative_path=relative_path,
                    # Recorded verbatim: the column mixes '0', '0.0' and '1.0'.
                    video_type=record["video_type"].strip(),
                    question=record["question"].strip(),
                    options=options,
                    gt=gt,
                    others=record.get("others", "").strip(),
                    csv_row_index=index,
                )
            )
    if not rows:
        raise ValueError(f"no questions found in {csv_path}")
    return rows


def latin_rectangle_cells() -> list[tuple[str, int]]:
    """The 25 (dataset, category) cells the sample draws one question from each.

    Dataset ``i`` takes categories ``{i, i+1, i+2, i+3, i+4} mod 6``.  Every
    dataset therefore contributes 5 questions, and the column sums come out
    ``[4, 4, 4, 4, 5, 4]`` -- as even as 25 questions over 6 tasks can be.
    """

    return [
        (dataset, (index + offset) % NUM_CATEGORIES)
        for index, dataset in enumerate(DATASETS)
        for offset in range(QUESTIONS_PER_DATASET)
    ]


def _candidates(rows: Sequence[DSIRow], dataset: str, cate: int) -> list[DSIRow]:
    # Sorted explicitly so the draw does not depend on CSV row order.
    return sorted(
        (row for row in rows if row.dataset == dataset and row.cate == cate),
        key=lambda row: (row.relative_path, row.csv_row_index),
    )


def sample_rows(rows: Sequence[DSIRow], seed: int) -> list[DSIRow]:
    """Draw one question per Latin-rectangle cell, all with distinct videos."""

    rng = random.Random(seed)
    selected: list[DSIRow] = []
    used_videos: set[str] = set()
    for dataset, cate in latin_rectangle_cells():
        pool = [
            row
            for row in _candidates(rows, dataset, cate)
            if row.relative_path not in used_videos
        ]
        if not pool:
            raise ValueError(f"no unused candidates for cell ({dataset}, cate={cate})")
        chosen = rng.choice(pool)
        used_videos.add(chosen.relative_path)
        selected.append(chosen)
    return selected


def census_rows(rows: Sequence[DSIRow]) -> list[DSIRow]:
    """Every question in the split, in a stable, video-grouped order.

    Sorted by ``relative_path`` rather than left in CSV order for two reasons:
    the order stops depending on how the benchmark happened to write its CSV, and
    the questions sharing one video (1769 questions span only 943 videos) end up
    adjacent, so a shard covers whole clips and a reviewer reads a clip's
    questions together.  ``csv_row_index`` breaks ties, so the order is total.
    """

    return sorted(rows, key=lambda row: (row.relative_path, row.csv_row_index))


def gt_histogram(rows: Iterable[DSIRow]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for row in rows:
        histogram[row.gt] = histogram.get(row.gt, 0) + 1
    return dict(sorted(histogram.items()))


def _is_balanced(histogram: dict[str, int]) -> bool:
    if sorted(histogram) != ["A", "B", "C", "D"]:
        return False
    return all(GT_BALANCE_MIN <= count <= GT_BALANCE_MAX for count in histogram.values())


def find_balanced_seed(
    rows: Sequence[DSIRow], start_seed: int = DEFAULT_SEED, max_attempts: int = 500
) -> tuple[int, list[DSIRow]]:
    """Return the first seed at or after ``start_seed`` with a balanced answer key.

    Uniform sampling is unbiased but, at n=25, routinely produces answer keys
    skewed towards one letter.  Scanning for a balanced key is a disclosed
    variance-reduction choice: the chosen seed is recorded in the manifest, so
    the selection stays fully reproducible.
    """

    for attempt in range(max_attempts):
        seed = start_seed + attempt
        selected = sample_rows(rows, seed)
        if _is_balanced(gt_histogram(selected)):
            return seed, selected
    raise RuntimeError(
        f"no seed in [{start_seed}, {start_seed + max_attempts}) produced a "
        f"ground-truth histogram with every letter in [{GT_BALANCE_MIN}, {GT_BALANCE_MAX}]"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    dsi_root: Path = DEFAULT_DSI_ROOT,
    split: str = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    balance_gt: bool = True,
    census: bool = False,
) -> dict[str, Any]:
    """Select the questions and describe the selection well enough to repeat it.

    With ``census=True`` the split is taken whole and ``seed``/``balance_gt`` are
    ignored -- there is nothing to seed when nothing is being drawn.
    """

    csv_path = metadata_csv(dsi_root, split)
    videos = video_root(dsi_root, split)
    rows = load_rows(csv_path)
    if census:
        chosen_seed, selected = None, census_rows(rows)
    elif balance_gt:
        chosen_seed, selected = find_balanced_seed(rows, seed)
    else:
        chosen_seed, selected = seed, sample_rows(rows, seed)

    entries: list[dict[str, Any]] = []
    for row in selected:
        entries.append(
            {
                **asdict(row),
                "category_name": row.category_name,
                "option_letters": row.option_letters,
                "question_id": row.question_id,
                "video_slug": video_slug(row.relative_path),
                "video_path": str(videos / row.relative_path),
            }
        )

    # Answers are stored one file per question_id, so a collision would make two
    # questions silently overwrite each other.  The 25-question sample cannot
    # collide (it forces distinct videos); a census has no such protection, and
    # relies on cate differing whenever two questions share a video.
    id_counts: dict[str, int] = {}
    for entry in entries:
        id_counts[entry["question_id"]] = id_counts.get(entry["question_id"], 0) + 1
    duplicates = sorted(key for key, count in id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} question_id(s) are not unique, so their answers would "
            f"overwrite each other: {duplicates[:5]}"
        )

    per_dataset: dict[str, int] = {}
    per_category: dict[str, int] = {}
    for row in selected:
        per_dataset[row.dataset] = per_dataset.get(row.dataset, 0) + 1
        per_category[row.category_name] = per_category.get(row.category_name, 0) + 1

    if census:
        sampling = {
            "design": DESIGN_CENSUS,
            "description": (
                "every question in the split, ordered by relative_path then csv_row_index; "
                "nothing is drawn, so there is no seed"
            ),
            "requested_seed": None,
            "seed": None,
            "gt_balanced": False,
            "total_source_questions": len(rows),
            "total_selected": len(selected),
            "distinct_videos": len({row.relative_path for row in selected}),
        }
        caveat = (
            "The complete 'std' split, so the per-dataset and per-task counts are the "
            "benchmark's own and are strongly uneven: 'internet' supplies 891 of 1769 "
            "questions, and the six tasks range from 85 to 582. Any single overall number "
            "is dominated by the largest cells -- read results grouped by dataset and by "
            "task."
        )
    else:
        sampling = {
            "design": DESIGN_LATIN_RECTANGLE,
            "description": (
                "dataset i takes categories {i, i+1, i+2, i+3, i+4} mod 6; one question "
                "per cell, all videos distinct"
            ),
            "requested_seed": seed,
            "seed": chosen_seed,
            "gt_balanced": balance_gt,
            "gt_balance_band": [GT_BALANCE_MIN, GT_BALANCE_MAX],
            "total_source_questions": len(rows),
            "total_selected": len(selected),
        }
        caveat = (
            "Balanced by design: each task carries 4-5 questions while the full benchmark "
            "is 33% 'Obj:moving cam' and 31% 'Cam:dynamic scene'. Read these results as a "
            "diagnostic across tasks, not as an estimate of DSI-Bench accuracy."
        )

    return {
        "benchmark": "DSI-Bench",
        "split": split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dsi_root": str(dsi_root),
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
        "video_root": str(videos),
        "sampling": sampling,
        "counts": {
            "per_dataset": per_dataset,
            "per_category": per_category,
            "gt_histogram": gt_histogram(selected),
        },
        "caveat": caveat,
        "questions": entries,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_question_prompt(entry: dict[str, Any]) -> str:
    """Render the question and its options for the model.

    Only the question text and the options are included.  ``cate``, ``dataset``,
    ``GT``, ``others`` and ``video_type`` are bookkeeping: showing the model the
    category label would tell it which reasoning recipe to apply and invalidate
    the evaluation.
    """

    lines = [entry["question"], "", "Options:"]
    lines.extend(f"{letter}: {entry['options'][letter]}" for letter in entry["option_letters"])
    return "\n".join(lines)
