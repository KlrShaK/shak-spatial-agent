"""Offline tests for the Dyn-Bench loader, family scan and sampler.

CPU-only, no network, and -- apart from the two tests marked as needing the real
dataset -- no filesystem beyond ``tmp_path``. What is under test is the parts
that silently produce wrong numbers rather than raising: option parsing, the
family precedence order, the water-fill allocation, and the reproducibility of
the draw.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dyn_dataset_exploration.dyn_bench_data import (
    DEFAULT_DYN_ROOT,
    FAMILY_NAMES,
    PAPER_TASKS,
    TASKS,
    TASK_TO_LEVEL,
    DynRow,
    all_families_of,
    answer_normalisations,
    build_stats,
    distinct_question_templates,
    family_of,
    file_sha256,
    format_question_prompt,
    load_rows,
    normalize_template,
    parse_options,
    qa_parquet,
    read_manifest,
    stratified_sample,
    video_slug,
    water_fill,
    write_manifest,
)

def _dataset_available() -> bool:
    """The real-dataset tests need both the mount and pyarrow.

    pyarrow is imported inside ``load_rows`` rather than at module scope so that
    every other test in this file -- the parsing, family and sampling logic that
    is the actual risk -- runs in any environment that has pytest.
    """

    if not qa_parquet(DEFAULT_DYN_ROOT).exists():
        return False
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


needs_dataset = pytest.mark.skipif(
    not _dataset_available(), reason="Dyn-Bench parquet or pyarrow unavailable"
)


def make_row(**overrides) -> DynRow:
    defaults = dict(
        dataset="davis",
        video="bike-packing",
        task="qa",
        qa_index=0,
        question="How does the bicycle move?",
        options={"A": "left", "B": "right", "C": "up", "D": "down"},
        answer="A",
        source_file="multi_json/davis/davis_bike-packing_qa.json",
        row_index=0,
    )
    defaults.update(overrides)
    return DynRow(**defaults)


# --------------------------------------------------------------------------- #
# Option parsing
# --------------------------------------------------------------------------- #

def test_parse_options_strips_the_embedded_letter():
    parsed = parse_options(["A. Pans from left to right", "B. Tilts up"])
    assert parsed == {"A": "Pans from left to right", "B": "Tilts up"}


def test_parse_options_accepts_the_other_separators():
    assert parse_options(["A) one", "B: two"]) == {"A": "one", "B": "two"}


def test_parse_options_falls_back_to_position_when_unprefixed():
    """A malformed row must still yield four answerable choices."""

    assert parse_options(["one", "two"]) == {"A": "one", "B": "two"}


def test_parse_options_does_not_let_a_repeated_letter_overwrite():
    parsed = parse_options(["A. first", "A. second"])
    assert parsed["A"] == "first"
    assert parsed["B"] == "A. second"


# --------------------------------------------------------------------------- #
# Ids
# --------------------------------------------------------------------------- #

def test_video_slug_separates_clips_that_share_a_tail():
    long_a = "x" * 80 + "_shared_tail"
    long_b = "y" * 80 + "_shared_tail"
    assert video_slug("sav", long_a) != video_slug("sav", long_b)


def test_question_id_is_unique_per_task_and_index():
    ids = {
        make_row(task=task, qa_index=index).question_id
        for task in TASKS
        for index in (0, 1, 2)
    }
    assert len(ids) == 9


# --------------------------------------------------------------------------- #
# Blindness
# --------------------------------------------------------------------------- #
