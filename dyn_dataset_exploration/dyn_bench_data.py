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
