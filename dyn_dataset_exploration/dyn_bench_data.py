"""Dyn-Bench loading, media resolution, deterministic scans, and sampling.

This module never touches the network and never loads a model. It answers the
questions that can be settled by reading the dataset itself, and it draws the
one sample that ``taxonomy_classify.py`` is allowed to send to an LLM.

Four subcommands, in the order they are meant to run::

    python -m dyn_dataset_exploration.dyn_bench_data index
    python -m dyn_dataset_exploration.dyn_bench_data stats
    python -m dyn_dataset_exploration.dyn_bench_data templates
    python -m dyn_dataset_exploration.dyn_bench_data sample

``index`` exists because Dyn-Bench's eight source datasets do not agree on how
to name anything: frames are ``frame_0000.jpg`` in DAVIS, ``00001.jpg`` in
Spring, ``rgb_00000.jpg`` in PointOdyssey and
``046ac8-3_obj_source_left-0000.png`` in DynamicReplica; the SA-V clips lose
their ``_manual`` suffix on the way to ``videos/``; and every
``frame_sampling.json`` points at the authors' own cluster
(``/apdcephfs_zwfy/...``). Resolving that once, here, keeps it out of every
downstream module.

``templates`` is the scan ``dsi_dataset_exploration.taxonomy_data`` ran over
DSI-Bench, retuned and rerun. Part of its job is to document that it *does not*
transfer: DSI-Bench's 1769 rows collapsed to 34 templates, whereas Dyn-Bench's
``qa`` and ``sceneqa`` are free-form LLM-generated text that barely collapses at
all. That is the finding that forces the LLM pass to do open coding rather than
template matching.

``FAMILIES`` below is a **sampling device and a descriptive statistic, not the
taxonomy**. It is deliberate regex, so it is reproducible and auditable, and it
exists so that a 200-question draw does not miss the 60 scene-revisit questions
entirely. The actual taxonomy is decided by ``taxonomy_classify.py``.

This module is self-contained on purpose. ``d4rt_agent.dsi_bench_data`` is
hard-wired to DSI-Bench's five datasets, six ``cate`` values and ``"A: a; B: b"``
option syntax, none of which survive here -- but the public shape (``load_rows``,
``build_manifest``, ``write_manifest``/``read_manifest``,
``format_question_prompt``, ``file_sha256``, ``video_slug``) is kept identical so
this file can be promoted to ``d4rt_agent/dyn_bench_data.py`` unchanged when an
agent actually runs on Dyn-Bench.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
from typing import Any, Mapping, Sequence

DEFAULT_DYN_ROOT = Path("/cluster/work/igp_psr/spanwar/datasets/Dyn-Bench")
DEFAULT_RESULTS_DIR = Path("dyn_dataset_exploration/logs")
# A fresh seed for this pass, distinct from the DSI exploration's 20260825 --
# the two draws are unrelated and should not be confused for the same one.
DEFAULT_SEED = 20260921

QA_PARQUET = "qa_tasks.parquet"
TASKS: tuple[str, ...] = ("cameraqa", "qa", "sceneqa")
DATASETS: tuple[str, ...] = (
    "davis",
    "dynamic_replica",
    "dynpose",
    "point_odyssey",
    "sav",
    "spring",
    "totalrecon",
    "ytvis",
)
# Four 4D/synthetic sources with 5-13 videos each. A draw proportional to the
# benchmark's own skew (ytvis has 504 videos, spring 5) would show them barely
# or not at all, and they are exactly the clips with real 3D structure.
RARE_DATASETS = frozenset({"spring", "point_odyssey", "dynamic_replica", "totalrecon"})

# Every frame_sampling.json was written on the authors' cluster and its
# frame_paths still point there. We keep the frame *indices*, which are the only
# part that is about the data, and re-root the paths ourselves.
AUTHOR_MEDIA_ROOT = "/apdcephfs_zwfy/share_303937731/lyl1015/dynamic/ckpt/Dyn-Bench"

SAMPLE_SIZE = 600
# Flat across the three coarse tasks rather than proportional (cameraqa has 3000
# rows, the others 2000 each): each task needs enough questions to characterise
# itself, and no task needs more than 200 to do that.
TASK_QUOTAS: dict[str, int] = {"cameraqa": 200, "qa": 200, "sceneqa": 200}
# Share of a family's quota held back for the four rare source datasets, when
# they have any rows in that family at all.
RARE_RESERVE_FRACTION = 0.2


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #

_OPTION_PREFIX = re.compile(r"^\s*([A-Za-z])\s*[.):]\s*(.*)$", re.DOTALL)
_LETTERS = "ABCDEFGH"


def parse_options(raw: Sequence[str]) -> dict[str, str]:
    """Split Dyn-Bench's ``["A. text", "B. text", ...]`` into ``{letter: text}``.

    Dyn-Bench embeds the letter in the option string with a ``.`` separator,
    where DSI-Bench used one ``"A: a; B: b"`` string with ``:``. An option that
    carries no prefix at all falls back to its position, so a malformed row
    still yields four answerable choices instead of raising.
    """

    options: dict[str, str] = {}
    for position, item in enumerate(raw):
        match = _OPTION_PREFIX.match(str(item))
        if match and match.group(1).upper() not in options:
            options[match.group(1).upper()] = match.group(2).strip()
        else:
            options[_LETTERS[position]] = str(item).strip()
    return options


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return cleaned or "clip"


def video_slug(dataset: str, video: str) -> str:
    """Filename-safe, collision-proof id for one clip.

    Same construction as ``dsi_bench_data.video_slug``: a readable tail plus a
    hash of the full key, so two clips whose names differ only in a truncated
    prefix cannot collide.
    """

    key = f"{dataset}/{video}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{_slug(key)[-60:]}_{digest}"


@dataclass(frozen=True)
class DynRow:
    """One multiple-choice question, as shipped in ``qa_tasks.parquet``."""

    dataset: str
    video: str
    task: str
    qa_index: int
    question: str
    options: dict[str, str]
    answer: str
    source_file: str
    row_index: int

    @property
    def option_letters(self) -> list[str]:
        return sorted(self.options)

    @property
    def video_key(self) -> str:
        return f"{self.dataset}/{self.video}"

    @property
    def video_slug(self) -> str:
        return video_slug(self.dataset, self.video)

    @property
    def question_id(self) -> str:
        return f"{self.task}_{self.video_slug}_{self.qa_index}"


def qa_parquet(dyn_root: Path = DEFAULT_DYN_ROOT) -> Path:
    return Path(dyn_root) / QA_PARQUET


def load_rows(parquet_path: Path) -> list[DynRow]:
    """Read every VQA row, normalising the two things the export got wrong.

    ``answer`` ships four lowercase letters among 7000 rows, and the letter is
    the join key against ``options`` -- so it is upper-cased here rather than at
    every read site. ``row_index`` is kept so a row can be traced back to its
    position in the parquet after any sort.
    """

    import pyarrow.parquet as pq  # local: only this one command needs pyarrow

    table = pq.read_table(Path(parquet_path))
    rows: list[DynRow] = []
    for index, record in enumerate(table.to_pylist()):
        rows.append(
            DynRow(
                dataset=str(record["dataset"]),
                video=str(record["video"]),
                task=str(record["task"]),
                qa_index=int(record["qa_index"]),
                question=str(record["question"]).strip(),
                options=parse_options(record["options"]),
                answer=str(record["answer"]).strip().upper(),
                source_file=str(record["source_file"]),
                row_index=index,
            )
        )
    return rows


def answer_normalisations(parquet_path: Path) -> dict[str, int]:
    """Count what ``load_rows`` silently fixed, so a report can cite it.

    ``load_rows`` upper-cases the ``answer`` column, which makes the four
    lowercase rows in the shipped export invisible to every downstream check.
    Counting them here keeps the fix and the record of the fix in the same
    module instead of leaving the claim to a comment.
    """

    import pyarrow.parquet as pq

    column = pq.read_table(Path(parquet_path), columns=["answer"]).column("answer").to_pylist()
    return {
        "lowercase_answer_rows": sum(
            1 for value in column if str(value).strip() != str(value).strip().upper()
        ),
        "whitespace_answer_rows": sum(1 for value in column if str(value) != str(value).strip()),
    }


def format_question_prompt(entry: Mapping[str, Any]) -> str:
    """The blind view: the question and its options, nothing else.

    ``task``, ``dataset``, ``answer`` and the family label are all withheld --
    naming the coarse task would hand the classifier a third of the answer on
    the paper-task axis, and the family label would hand it the other axis.
    """

    lines = [str(entry["question"]), "", "Options:"]
    options = entry["options"]
    for letter in sorted(options):
        lines.append(f"{letter}: {options[letter]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Question families -- a sampling stratum, not the taxonomy
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Family:
    """One regex-defined question family.

    ``question`` and ``options`` are both optional; when both are given, both
    must match. The options pattern exists for the two families that are only
    identifiable from the answer choices -- a question asking "how fast" looks
    ordinal until you see that every option is in m/s.
    """

    name: str
    description: str
    question: re.Pattern[str] | None = None
    options: re.Pattern[str] | None = None

    def matches(self, question_text: str, options_text: str) -> bool:
        if self.question is not None and not self.question.search(question_text):
            return False
        if self.options is not None and not self.options.search(options_text):
            return False
        return self.question is not None or self.options is not None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Ordered most-specific first: the first match wins as the primary family, so a
# question that asks for a distance *in metres* at a *named second* is filed
# under metric_distance, where its hardest requirement is.
FAMILIES: tuple[Family, ...] = (
    Family(
        "metric_speed",
        "absolute speed, answered in physical units",
        options=_rx(r"\bm\s*/\s*s\b|\bmeters? per second\b|\bkm\s*/\s*h\b|\bmph\b"),
    ),
    Family(
        "metric_distance",
        "absolute distance, answered in physical units",
        options=_rx(r"\b\d+(\.\d+)?\s*(m|meters?|metres?|cm|centimet\w+|feet|ft)\b"),
    ),
    Family("instance_count", "how many distinct instances", question=_rx(r"\bhow many\b")),
    Family(
        "event_order",
        "which of two named events happens first",
        question=_rx(r"\b(happens?|comes?|occurs?|occurred)\s+first\b|\bfirst[:,]|\bwhich event\b"),
    ),
    Family(
        "timestamped",
        "state at or between named timestamps",
        question=_rx(r"\b\d+(\.\d+)?\s*(s|secs?|seconds?)\b"),
    ),
    # Before camera_zoom on purpose: "does the object appear larger or smaller
    # (considering combined effects of zoom and physical motion)" names zoom only
    # as the confound. The quantity asked for is projected size.
    Family(
        "apparent_size",
        "projected size in the image plane",
        question=_rx(
            r"apparent size|appear\w*\s+(larger|smaller|bigger)|larger or smaller"
            r"|visually appear|occupy|\bsize\b.{0,25}\bframe\b"
        ),
    ),
    Family("camera_zoom", "whether focal length changed", question=_rx(r"\bzoom")),
    Family(
        "camera_shake",
        "high-frequency camera jitter",
        question=_rx(r"\bshak\w*|\bjitter|\bstabili\w*|\bwobbl"),
    ),
    Family(
        "camera_rhythm",
        "camera speed profile over time",
        question=_rx(r"movement speed|movement rhythm|\brhythm\b|speed .{0,20}camera|camera.{0,30}\bspeed\b"),
    ),
    Family(
        "shooting_angle",
        "camera elevation or bearing relative to the subject",
        question=_rx(r"shooting angle|camera angle|\b(high|low)[- ]angle|eye[- ]?level|which side of the"),
    ),
    Family(
        "subject_orientation",
        "which way a scene object faces, and how that changes",
        question=_rx(r"\borientation\b|\bfacing\b|\bfaces?\s+(toward|away|the)|which (way|direction).{0,20}(face|point)|\bheading\b"),
    ),
    Family(
        "visibility",
        "whether and where the subject is present in the image frame",
        question=_rx(
            r"\b(in|out of|leaves?|left|enters?|entered|exits?)\s+the\s+frame\b"
            r"|still (visible|in frame)|remains? visible|(which|what) (area|part|region|portion) of the frame"
        ),
    ),
    Family(
        "scene_revisit",
        "whether start and end show the same place",
        question=_rx(r"\b(scene|background|environment|location)\b.{0,40}\bsame\b|same (scene|background|place|location|environment)|returns? to the (same|start)"),
    ),
    Family(
        "camera_subject_distance",
        "range between camera and subject, ordinally",
        question=_rx(r"distance between the camera|camera.{0,30}\bdistance\b|\bdistance\b.{0,30}camera"),
    ),
    # Broad on purpose: Dyn-Bench phrases this a dozen ways ("primary trajectory
    # movement of the camera", "the camera's net orbital motion", "does the
    # camera remain fixed"). The three narrower camera families above claim their
    # own questions first, so this one only sweeps up what is left.
    Family(
        "camera_trajectory",
        "where the camera went, in its own frame",
        question=_rx(
            r"\bcamera\b.{0,40}\b(trajectory|movement|motion|moves?|moved|pans?|tilts?|dollies|orbit\w*|rotat\w*|travel\w*|remain\w*|fixed|static|still)\b"
            r"|\b(trajectory|movement|motion|path|pan|tilt|dolly|orbit\w*)\b.{0,40}\bcamera\b"
        ),
    ),
    Family(
        "motion_continuity",
        "whether the subject moves at all, and how continuously",
        question=_rx(
            r"continuous\w*|intermittent\w*|only (once )?briefly|remains? stationary"
            r"|stationary or move|interrupted by pause|\bpauses?\b"
        ),
    ),
    Family(
        "object_object_relation",
        "one scene object's position relative to another",
        question=_rx(
            r"relative to the|with respect to|in relation to|spatial relationship"
            r"|between the .{2,40} and the |closer to (each other|one another)"
            r"|(farther|further) apart|closer together"
        ),
    ),
    Family(
        "subject_trajectory",
        "where a scene object went",
        question=_rx(r"(position|movement|direction|trajectory|path|motion)\b.{0,40}(change|shift|evolve)|how does the .{2,40} (move|travel|go)\b|direction of the"),
    ),
    Family(
        "activity_semantic",
        "what is happening, described rather than measured",
        question=_rx(r"best describes|main activity|what is happening|what happens to the|composition|posture|appearance|expression|focus|impression|atmosphere"),
    ),
)
FAMILY_NAMES: tuple[str, ...] = (*(f.name for f in FAMILIES), "unfamilied")
FAMILY_DESCRIPTIONS: dict[str, str] = {
    **{f.name: f.description for f in FAMILIES},
    "unfamilied": "matched no family regex",
}


def family_of(question: str, options: Mapping[str, str]) -> str:
    """The primary family for one question -- first match in ``FAMILIES`` order."""

    options_text = " ".join(options[letter] for letter in sorted(options))
    for family in FAMILIES:
        if family.matches(question, options_text):
            return family.name
    return "unfamilied"


def all_families_of(question: str, options: Mapping[str, str]) -> list[str]:
    """Every family a question matches, not just the primary one.

    Reported alongside the primary label because the overlap is itself a finding:
    a question that is simultaneously ``timestamped`` and ``metric_distance``
    needs both a time axis and a metric scale, and the size of that intersection
    is what decides whether AgentV5 can ship them separately.
    """

    options_text = " ".join(options[letter] for letter in sorted(options))
    matched = [f.name for f in FAMILIES if f.matches(question, options_text)]
    return matched or ["unfamilied"]


# --------------------------------------------------------------------------- #
# Template scan
# --------------------------------------------------------------------------- #

_POSSESSIVE = re.compile(r"\bthe\s+[a-z][a-z '_-]{0,30}?'s\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(\.\d+)?")


def normalize_template(question: str) -> str:
    """Collapse a question to a template skeleton.

    Weaker than DSI-Bench's noun-list substitution on purpose: Dyn-Bench names
    far too many subjects for a fixed list, so this only folds the three things
    that are reliably incidental -- numbers, possessive subjects, and
    case/punctuation. Anything that still fails to collapse is genuinely
    distinct phrasing, which is the point of running it.
    """

    text = question.replace("‘", "'").replace("’", "'")
    text = _POSSESSIVE.sub("the SUBJ's", text)
    text = _NUMBER.sub("N", text)
    text = re.sub(r"[^A-Za-z' ]", " ", text).lower()
    return " ".join(text.split())


def distinct_question_templates(rows: Sequence[DynRow]) -> dict[str, dict[str, Any]]:
    """Every distinct template in ``rows``, with counts and a task breakdown."""

    counts: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = {}
    for row in rows:
        template = normalize_template(row.question)
        counts[template] += 1
        by_task.setdefault(template, Counter())[row.task] += 1
    return {
        template: {"count": count, "by_task": dict(sorted(by_task[template].items()))}
        for template, count in counts.most_common()
    }


# --------------------------------------------------------------------------- #
# Media resolution
# --------------------------------------------------------------------------- #

_FRAME_INDEX = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")
_MASK_NAME = re.compile(r"(?:^|_)frame_?(\d+)_obj(\d+)\.png$", re.IGNORECASE)
_MASK_NAME_BARE = re.compile(r"^(\d+)_obj(\d+)\.png$", re.IGNORECASE)


def _video_candidates(dataset: str, video: str) -> list[str]:
    """Filenames to try under ``videos/<dataset>/``, most likely first.

    Most sources use the bare clip name. SA-V is the exception twice over: its
    mp4s carry the ``sav_`` prefix *and* drop the ``_manual`` suffix that the
    parquet's ``video`` column keeps.
    """

    names = [video, f"{dataset}_{video}"]
    if video.endswith("_manual"):
        stem = video[: -len("_manual")]
        names += [stem, f"{dataset}_{stem}"]
    return [f"{name}.mp4" for name in dict.fromkeys(names)]


def _frame_index_of(name: str) -> int | None:
    match = _FRAME_INDEX.search(name)
    return int(match.group(1)) if match else None


def resolve_media(dyn_root: Path, dataset: str, video: str) -> dict[str, Any]:
    """Locate the mp4, the frame directory and the mask directory for one clip.

    Returns a record even when something is missing -- the missing field is
    ``None`` and lands in the index's ``unresolved`` list. A clip we cannot
    resolve is a fact worth reporting, not a reason to abort a 1000-clip scan.
    """

    root = Path(dyn_root)
    clip = f"{dataset}_{video}"

    video_path: Path | None = None
    for candidate in _video_candidates(dataset, video):
        path = root / "videos" / dataset / candidate
        if path.exists():
            video_path = path
            break

    frames_dir = root / "frames" / dataset / clip
    frame_names = sorted(p.name for p in frames_dir.iterdir()) if frames_dir.is_dir() else []
    frame_indices = [i for i in (_frame_index_of(n) for n in frame_names) if i is not None]

    masks_dir = root / "masks" / dataset / clip
    objects: dict[str, list[int]] = {}
    if masks_dir.is_dir():
        for name in sorted(p.name for p in masks_dir.iterdir()):
            match = _MASK_NAME.search(name) or _MASK_NAME_BARE.match(name)
            if match:
                objects.setdefault(f"object_{int(match.group(2))}", []).append(int(match.group(1)))

    sampling = root / "multi_json" / dataset / f"{clip}_frame_sampling.json"
    sampled: dict[str, Any] | None = None
    if sampling.exists():
        payload = json.loads(sampling.read_text(encoding="utf-8"))
        indices = [int(i) for i in payload.get("frame_indices", [])]
        sampled = {
            "total_frames": payload.get("total_frames"),
            "sampled_frames": payload.get("sampled_frames"),
            "frame_indices": indices,
            # The shipped frame_paths point at the authors' cluster; we keep the
            # indices and re-root the paths onto whatever frames/ we actually have.
            "frame_files": [
                frame_names[i] if 0 <= i < len(frame_names) else None for i in indices
            ],
        }

    return {
        "dataset": dataset,
        "video": video,
        "video_key": f"{dataset}/{video}",
        "video_slug": video_slug(dataset, video),
        "video_path": str(video_path) if video_path else None,
        "frames_dir": str(frames_dir) if frame_names else None,
        "frame_pattern": frame_names[0] if frame_names else None,
        "frame_count": len(frame_names),
        "masks_dir": str(masks_dir) if objects else None,
        "objects": {name: len(idx) for name, idx in sorted(objects.items())},
        "mask_frame_stride": _stride_of(next(iter(objects.values()), [])),
        "frame_sampling": sampled,
    }


def _stride_of(indices: Sequence[int]) -> int | None:
    """Spacing between consecutive mask frames, when it is constant.

    DynamicReplica ships masks every fifth frame against 300 RGB frames; DAVIS
    ships one per frame. A downstream tool that assumes dense masks is wrong on
    one of them, so the index states which it is.
    """

    if len(indices) < 2:
        return None
    steps = {b - a for a, b in zip(sorted(indices), sorted(indices)[1:])}
    return steps.pop() if len(steps) == 1 else None


def build_index(dyn_root: Path = DEFAULT_DYN_ROOT) -> dict[str, Any]:
    """Resolve every clip named by the VQA rows to its media on disk."""

    rows = load_rows(qa_parquet(dyn_root))
    keys = sorted({(row.dataset, row.video) for row in rows})
    clips = [resolve_media(dyn_root, dataset, video) for dataset, video in keys]

    unresolved = [
        {
            "video_key": clip["video_key"],
            "missing": [
                field_name
                for field_name in ("video_path", "frames_dir", "masks_dir", "frame_sampling")
                if not clip[field_name]
            ],
        }
        for clip in clips
        if not all(clip[f] for f in ("video_path", "frames_dir", "masks_dir", "frame_sampling"))
    ]

    return {
        "purpose": "per-clip media resolution across Dyn-Bench's eight source datasets",
        "dyn_root": str(dyn_root),
        "generated_at": _utc_now(),
        "counts": {
            "clips": len(clips),
            "unresolved": len(unresolved),
            "per_dataset": dict(Counter(clip["dataset"] for clip in clips)),
        },
        "unresolved": unresolved,
        "clips": {clip["video_key"]: clip for clip in clips},
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def build_stats(
    rows: Sequence[DynRow], normalisation: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Descriptive facts about the 7000 rows. No model, no sampling, no inference."""

    per_task: dict[str, Any] = {}
    for task in TASKS:
        subset = [row for row in rows if row.task == task]
        lengths = [len(row.options[letter]) for row in subset for letter in row.options]
        per_task[task] = {
            "rows": len(subset),
            "distinct_videos": len({row.video_key for row in subset}),
            "distinct_questions": len({row.question for row in subset}),
            "answer_histogram": dict(sorted(Counter(row.answer for row in subset).items())),
            "option_chars_median": round(statistics.median(lengths), 1) if lengths else 0.0,
            "primary_family": dict(
                Counter(family_of(row.question, row.options) for row in subset).most_common()
            ),
        }

    family_overlap: Counter[str] = Counter()
    for row in rows:
        matched = all_families_of(row.question, row.options)
        if len(matched) > 1:
            family_overlap[" + ".join(matched)] += 1

    # Kept because it changes how every accuracy number on this benchmark should
    # be read, and it costs one pass over the parquet. It is a property of the
    # option text, not a model run.
    shortcut: dict[str, Any] = {}
    for task in (*TASKS, "all"):
        subset = rows if task == "all" else [row for row in rows if row.task == task]
        longest = sum(
            1
            for row in subset
            if max(row.option_letters, key=lambda letter: len(row.options[letter])) == row.answer
        )
        shortcut[task] = {
            "rows": len(subset),
            "answer_is_longest_option": round(100.0 * longest / len(subset), 1) if subset else 0.0,
            "answer_is_A": round(
                100.0 * sum(1 for row in subset if row.answer == "A") / len(subset), 1
            )
            if subset
            else 0.0,
        }

    dirty = [
        {"question_id": row.question_id, "issue": issue}
        for row in rows
        for issue in _row_issues(row)
    ]

    return {
        "purpose": "deterministic descriptive statistics over the whole Dyn-Bench VQA split",
        "generated_at": _utc_now(),
        "total_rows": len(rows),
        "normalisation": dict(normalisation or {"lowercase_answer_rows": 0, "whitespace_answer_rows": 0}),
        "per_dataset": dict(Counter(row.dataset for row in rows).most_common()),
        "per_task": per_task,
        "answer_histogram": dict(sorted(Counter(row.answer for row in rows).items())),
        "family_descriptions": FAMILY_DESCRIPTIONS,
        "family_totals": dict(
            Counter(family_of(row.question, row.options) for row in rows).most_common()
        ),
        "family_overlaps": dict(family_overlap.most_common(20)),
        "shortcut_indicators": {
            "note": (
                "properties of the shipped option text, not model results -- recorded "
                "because they bound how much of this benchmark is a vision task"
            ),
            "by_task": shortcut,
        },
        "data_issues": dirty,
    }


def _row_issues(row: DynRow) -> list[str]:
    issues: list[str] = []
    if len(row.options) != 4:
        issues.append(f"expected 4 options, found {len(row.options)}")
    if row.answer not in row.options:
        issues.append(f"answer {row.answer!r} is not one of {sorted(row.options)}")
    if not row.question:
        issues.append("empty question")
    return issues


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def water_fill(available: Mapping[str, int], total: int) -> dict[str, int]:
    """Spread ``total`` as evenly as possible over strata, capped by availability.

    Equal allocation, not proportional: the point of stratifying by family is
    that a 60-row family says as much about what AgentV5 needs as a 900-row one.
    Strata that cannot fill their share hand the remainder back, and the loop
    repeats until nothing more can be placed. Ties break on the stratum name so
    the result does not depend on dict order.
    """

    quota = {name: 0 for name in available}
    remaining = min(total, sum(available.values()))
    open_strata = {name for name, count in available.items() if count > 0}
    while remaining > 0 and open_strata:
        share, extra = divmod(remaining, len(open_strata))
        placed = 0
        for position, name in enumerate(sorted(open_strata)):
            want = share + (1 if position < extra else 0)
            can = min(want, available[name] - quota[name])
            quota[name] += can
            placed += can
        remaining -= placed
        open_strata = {name for name in open_strata if quota[name] < available[name]}
        if placed == 0:
            break
    return {name: count for name, count in quota.items() if count}


def _draw(pool: Sequence[DynRow], quota: int, rng: random.Random, used_videos: set[str]) -> list[DynRow]:
    """Take ``quota`` rows from ``pool``, rare datasets first, distinct videos.

    ``used_videos`` is shared across the families of one task, so a task's 200
    questions come from 200 different clips wherever the pool allows it; it is
    *not* shared across tasks, because the same clip answering a camera question
    and a scene question is two different measurements.
    """

    rare = [row for row in pool if row.dataset in RARE_DATASETS]
    common = [row for row in pool if row.dataset not in RARE_DATASETS]
    rng.shuffle(rare)
    rng.shuffle(common)
    reserve = min(len(rare), max(1, int(quota * RARE_RESERVE_FRACTION)) if rare else 0)
    ordered = [*rare[:reserve], *common, *rare[reserve:]]

    chosen: list[DynRow] = []
    for row in ordered:
        if row.video_key in used_videos:
            continue
        used_videos.add(row.video_key)
        chosen.append(row)
        if len(chosen) == quota:
            break
    # Only if distinct-video coverage could not fill the quota do we allow a
    # repeat clip -- a thin family is worth sampling twice from one video.
    if len(chosen) < quota:
        for row in ordered:
            if row in chosen:
                continue
            chosen.append(row)
            if len(chosen) == quota:
                break
    return chosen


def stratified_sample(
    rows: Sequence[DynRow], seed: int, task_quotas: Mapping[str, int] = TASK_QUOTAS
) -> list[DynRow]:
    """Draw ``task_quotas[task]`` questions per task, stratified by family."""

    rng = random.Random(seed)
    selected: list[DynRow] = []
    for task in sorted(task_quotas):
        # Sorted explicitly so the draw never depends on parquet row order.
        pool = sorted(
            (row for row in rows if row.task == task),
            key=lambda row: (row.video_key, row.qa_index, row.row_index),
        )
        by_family: dict[str, list[DynRow]] = {}
        for row in pool:
            by_family.setdefault(family_of(row.question, row.options), []).append(row)
        quotas = water_fill({name: len(pool_) for name, pool_ in by_family.items()}, task_quotas[task])
        used_videos: set[str] = set()
        for family in sorted(quotas):
            selected.extend(_draw(by_family[family], quotas[family], rng, used_videos))
    return selected


def build_taxonomy_manifest(
    dyn_root: Path = DEFAULT_DYN_ROOT,
    seed: int = DEFAULT_SEED,
    task_quotas: Mapping[str, int] = TASK_QUOTAS,
) -> dict[str, Any]:
    """Select the taxonomy sample and describe the selection well enough to repeat it."""

    parquet = qa_parquet(dyn_root)
    rows = load_rows(parquet)
    selected = stratified_sample(rows, seed, task_quotas)

    entries: list[dict[str, Any]] = []
    for row in selected:
        record = asdict(row)
        record.update(
            {
                "option_letters": row.option_letters,
                "question_id": row.question_id,
                "video_key": row.video_key,
                "video_slug": row.video_slug,
                "family": family_of(row.question, row.options),
                "families": all_families_of(row.question, row.options),
                "hypothesis_level": TASK_TO_LEVEL[row.task],
            }
        )
        entries.append(record)

    ids = Counter(entry["question_id"] for entry in entries)
    duplicates = [qid for qid, count in ids.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate question_ids in sample: {duplicates[:5]}")

    return {
        "benchmark": "Dyn-Bench",
        "purpose": (
            "dyn_dataset_exploration taxonomy sample -- family-stratified within each "
            "coarse task, not proportional to the benchmark's own skew"
        ),
        "created_at": _utc_now(),
        "dyn_root": str(dyn_root),
        "parquet_path": str(parquet),
        "parquet_sha256": file_sha256(parquet),
        "sampling": {
            "design": "family_stratified",
            "description": (
                "equal allocation over question families within each coarse task, capped "
                "by availability; rare source datasets reserved a share of each family's "
                "quota; distinct videos within a task"
            ),
            "seed": seed,
            "task_quotas": dict(task_quotas),
            "rare_datasets": sorted(RARE_DATASETS),
            "rare_reserve_fraction": RARE_RESERVE_FRACTION,
            "total_source_questions": len(rows),
            "total_selected": len(selected),
        },
        "counts": {
            "per_task": dict(Counter(e["task"] for e in entries).most_common()),
            "per_family": dict(Counter(e["family"] for e in entries).most_common()),
            "per_dataset": dict(Counter(e["dataset"] for e in entries).most_common()),
            "distinct_videos": len({e["video_key"] for e in entries}),
        },
        "caveat": (
            "family labels are regex, used as sampling strata and descriptive statistics; "
            "the taxonomy itself is decided by taxonomy_classify.py"
        ),
        "questions": entries,
    }


# --------------------------------------------------------------------------- #
# Paper taxonomy (axis A) -- names verbatim from arXiv 2603.12746
# --------------------------------------------------------------------------- #

PAPER_LEVELS: dict[str, tuple[str, ...]] = {
    "Dynamic Inter-Object Perception": (
        "Act. & Obj. Desc.",
        "Move. & Temp. Dyn.",
        "Spatial Rel. & Change",
    ),
    "Dynamic Object-Scene Tracking": (
        "Mov. Patterns & Traj.",
        "Spatial Rel. & Comp.",
        "Scene Focus & Dyn.",
    ),
    "Dynamic Camera-Object Reasoning": (
        "Cam. Motion & Orient.",
        "Cam-Obj. Interaction",
        "Temp. & Visual Change",
    ),
}
PAPER_TASKS: tuple[str, ...] = tuple(
    name for names in PAPER_LEVELS.values() for name in names
)
LEVEL_OF_PAPER_TASK: dict[str, str] = {
    name: level for level, names in PAPER_LEVELS.items() for name in names
}
# Hypothesis, not fact: the paper names three levels and ships three coarse task
# labels, in the same order. Confirming or breaking this mapping is one of the
# two things the LLM pass is for.
TASK_TO_LEVEL: dict[str, str] = {
    "qa": "Dynamic Inter-Object Perception",
    "sceneqa": "Dynamic Object-Scene Tracking",
    "cameraqa": "Dynamic Camera-Object Reasoning",
}


# --------------------------------------------------------------------------- #
# IO helpers (same contract as dsi_bench_data's)
# --------------------------------------------------------------------------- #

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(payload: Mapping[str, Any], path: Path) -> Path:
    """Write JSON atomically: a SLURM kill mid-write must not leave half a file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cmd_index(args: argparse.Namespace) -> int:
    payload = build_index(args.dyn_root)
    out_path = write_manifest(payload, Path(args.results_dir) / "index.json")
    counts = payload["counts"]
    print(f"{counts['clips']} clips, {counts['unresolved']} unresolved -> {out_path}")
    for clip in payload["unresolved"][:10]:
        print(f"  {clip['video_key']}: missing {', '.join(clip['missing'])}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    parquet = qa_parquet(args.dyn_root)
    rows = load_rows(parquet)
    payload = build_stats(rows, answer_normalisations(parquet))
    out_path = write_manifest(payload, Path(args.results_dir) / "stats.json")
    print(f"{payload['total_rows']} rows -> {out_path}")
    for task, info in payload["per_task"].items():
        print(
            f"  {task}: {info['rows']} rows, {info['distinct_questions']} distinct questions, "
            f"answers {info['answer_histogram']}"
        )
    print("  families:")
    for name, count in payload["family_totals"].items():
        print(f"    {count:5d}  {name}")
    print(f"  normalised on load: {payload['normalisation']}")
    if payload["data_issues"]:
        print(f"  {len(payload['data_issues'])} data issues (see stats.json)")
    return 0


def _cmd_templates(args: argparse.Namespace) -> int:
    rows = load_rows(qa_parquet(args.dyn_root))
    per_task: dict[str, Any] = {}
    for task in TASKS:
        subset = [row for row in rows if row.task == task]
        templates = distinct_question_templates(subset)
        top = list(templates.items())[:25]
        per_task[task] = {
            "rows": len(subset),
            "distinct_templates": len(templates),
            "top25_coverage": sum(info["count"] for _, info in top),
            "top25": {template: info["count"] for template, info in top},
        }
    payload = {
        "purpose": "exhaustive template scan over every VQA row (no LLM, no sampling)",
        "generated_at": _utc_now(),
        "total_rows": len(rows),
        "distinct_templates": len(distinct_question_templates(rows)),
        "per_task": per_task,
        "finding": (
            "the DSI-Bench template method transfers only to cameraqa; qa and sceneqa are "
            "free-form generated text and do not collapse, which is why the taxonomy pass "
            "has to do open coding rather than template matching"
        ),
    }
    out_path = write_manifest(payload, Path(args.results_dir) / "templates.json")
    print(f"{len(rows)} rows -> {payload['distinct_templates']} distinct templates ({out_path})")
    for task, info in per_task.items():
        pct = 100.0 * info["top25_coverage"] / info["rows"]
        print(
            f"  {task}: {info['rows']} rows -> {info['distinct_templates']} templates, "
            f"top-25 cover {info['top25_coverage']} ({pct:.0f}%)"
        )
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    manifest = build_taxonomy_manifest(dyn_root=args.dyn_root, seed=args.seed)
    out_path = write_manifest(manifest, Path(args.results_dir) / "manifest.json")
    print(f"wrote {manifest['sampling']['total_selected']} questions to {out_path}")
    print(f"  distinct videos: {manifest['counts']['distinct_videos']}")
    for task, count in manifest["counts"]["per_task"].items():
        print(f"  {task}: {count}")
    print("  families:")
    for name, count in manifest["counts"]["per_family"].items():
        print(f"    {count:4d}  {name}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dyn-root", type=Path, default=DEFAULT_DYN_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="resolve every clip to its media on disk")
    p_index.set_defaults(func=_cmd_index)

    p_stats = sub.add_parser("stats", help="descriptive statistics over all 7000 rows")
    p_stats.set_defaults(func=_cmd_stats)

    p_templates = sub.add_parser("templates", help="exhaustive question-template scan")
    p_templates.set_defaults(func=_cmd_templates)

    p_sample = sub.add_parser("sample", help="draw the family-stratified 600-question sample")
    p_sample.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_sample.set_defaults(func=_cmd_sample)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
