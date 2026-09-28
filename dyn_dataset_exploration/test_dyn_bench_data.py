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

def test_format_question_prompt_shows_only_question_and_options():
    row = make_row(question="Which way?", answer="C", dataset="sav", task="sceneqa")
    prompt = format_question_prompt(
        {"question": row.question, "options": row.options, "task": row.task}
    )
    assert prompt.splitlines()[0] == "Which way?"
    assert "sceneqa" not in prompt
    assert "sav" not in prompt
    for letter in "ABCD":
        assert f"{letter}: " in prompt


# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "question, options, expected",
    [
        (
            "What is the average speed of the snowboarder?",
            ["A. 5.2 m/s", "B. 8.7 m/s", "C. 12.1 m/s", "D. 15.3 m/s"],
            "metric_speed",
        ),
        (
            "At the moment the dog lies down, what is the distance to the tree?",
            ["A. 5 meters", "B. 10 meters", "C. 15 meters", "D. 20 meters"],
            "metric_distance",
        ),
        ("How many distinct people appear in the video?", ["A. One", "B. Two"], "instance_count"),
        (
            "Which happens first: the camera pans, or the child enters the frame?",
            ["A. Camera", "B. Child"],
            "event_order",
        ),
        (
            "At approximately 2.67 seconds, where is the skateboarder?",
            ["A. left", "B. right"],
            "timestamped",
        ),
        (
            "What is the primary trajectory movement of the camera in the video?",
            ["A. Pans left", "B. Tilts up"],
            "camera_trajectory",
        ),
        (
            "Does the deer's grazing action occur continuously, intermittently, or briefly?",
            ["A. continuously", "B. briefly"],
            "motion_continuity",
        ),
        ("How does the boat's orientation change as it travels?", ["A. left", "B. right"], "subject_orientation"),
        (
            "Is the background scene the same at the start and end of the video?",
            ["A. yes", "B. no"],
            "scene_revisit",
        ),
        (
            "Do the two men get closer to each other or farther apart?",
            ["A. closer", "B. farther"],
            "object_object_relation",
        ),
        ("What best describes the main activity in the video?", ["A. surfing"], "activity_semantic"),
        ("Is this a question about nothing in particular?", ["A. yes"], "unfamilied"),
    ],
)
def test_family_of_assigns_the_expected_primary_family(question, options, expected):
    assert family_of(question, parse_options(options)) == expected


def test_apparent_size_outranks_camera_zoom():
    """Zoom named as the confound must not steal a question about projected size."""

    question = (
        "Over time, does the main object visually appear larger or smaller? "
        "(Considering combined effects of zoom and physical motion)"
    )
    assert family_of(question, parse_options(["A. larger", "B. smaller"])) == "apparent_size"


def test_metric_units_outrank_the_question_stem():
    """The same stem is a different measurement when the options carry units."""

    stem = "How far apart are the dog and the tree?"
    ordinal = parse_options(["A. closer", "B. farther", "C. unchanged", "D. unclear"])
    metric = parse_options(["A. 5 meters", "B. 10 meters", "C. 15 meters", "D. 20 meters"])
    assert family_of(stem, ordinal) != "metric_distance"
    assert family_of(stem, metric) == "metric_distance"


def test_all_families_reports_every_match_not_just_the_primary():
    question = "At approximately 5.0 seconds, what is the distance between the dog and the tree?"
    options = parse_options(["A. 1 meters", "B. 2 meters", "C. 3 meters", "D. 4 meters"])
    matched = all_families_of(question, options)
    assert "metric_distance" in matched
    assert "timestamped" in matched
    assert family_of(question, options) == "metric_distance"


def test_every_family_name_has_a_description():
    from dyn_dataset_exploration.dyn_bench_data import FAMILY_DESCRIPTIONS

    assert set(FAMILY_NAMES) == set(FAMILY_DESCRIPTIONS)


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

def test_normalize_template_folds_numbers_and_possessives():
    a = normalize_template("At 5.33 seconds, how does the cat's position change?")
    b = normalize_template("At 12 seconds, how does the dog's position change?")
    assert a == b


def test_distinct_question_templates_counts_per_task():
    rows = [
        make_row(task="qa", qa_index=0, question="How far did the cat's paw move?"),
        make_row(task="sceneqa", qa_index=1, question="How far did the dog's paw move?"),
    ]
    templates = distinct_question_templates(rows)
    assert len(templates) == 1
    only = next(iter(templates.values()))
    assert only["count"] == 2
    assert only["by_task"] == {"qa": 1, "sceneqa": 1}


# --------------------------------------------------------------------------- #
