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
