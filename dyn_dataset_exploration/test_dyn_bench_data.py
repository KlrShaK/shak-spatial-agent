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
# Water-fill allocation
# --------------------------------------------------------------------------- #

def test_water_fill_is_equal_when_every_stratum_is_deep_enough():
    assert water_fill({"a": 100, "b": 100, "c": 100}, 30) == {"a": 10, "b": 10, "c": 10}


def test_water_fill_redistributes_what_a_thin_stratum_cannot_take():
    quota = water_fill({"thin": 2, "fat": 100, "fatter": 100}, 30)
    assert quota["thin"] == 2
    assert sum(quota.values()) == 30


def test_water_fill_never_exceeds_availability():
    quota = water_fill({"a": 3, "b": 4}, 100)
    assert quota == {"a": 3, "b": 4}


def test_water_fill_is_independent_of_dict_order():
    forward = water_fill({"a": 5, "b": 100, "c": 100}, 31)
    backward = water_fill({"c": 100, "b": 100, "a": 5}, 31)
    assert forward == backward


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def _synthetic_rows() -> list[DynRow]:
    rows: list[DynRow] = []
    index = 0
    for task in TASKS:
        for clip in range(40):
            for qa_index, (question, options) in enumerate(
                [
                    ("How many distinct people appear?", ["A. One", "B. Two"]),
                    ("What best describes the main activity?", ["A. surfing", "B. skiing"]),
                ]
            ):
                rows.append(
                    make_row(
                        task=task,
                        video=f"clip{clip:03d}",
                        qa_index=qa_index,
                        question=question,
                        options=parse_options(options),
                        row_index=index,
                    )
                )
                index += 1
    return rows


def test_stratified_sample_is_reproducible_for_one_seed():
    rows = _synthetic_rows()
    quotas = {task: 20 for task in TASKS}
    first = [r.question_id for r in stratified_sample(rows, 7, quotas)]
    second = [r.question_id for r in stratified_sample(rows, 7, quotas)]
    assert first == second


def test_stratified_sample_does_not_depend_on_input_order():
    rows = _synthetic_rows()
    quotas = {task: 20 for task in TASKS}
    forward = [r.question_id for r in stratified_sample(rows, 7, quotas)]
    backward = [r.question_id for r in stratified_sample(list(reversed(rows)), 7, quotas)]
    assert forward == backward


def test_stratified_sample_honours_the_per_task_quota():
    rows = _synthetic_rows()
    quotas = {task: 20 for task in TASKS}
    sample = stratified_sample(rows, 7, quotas)
    for task in TASKS:
        assert sum(1 for r in sample if r.task == task) == 20


def test_stratified_sample_covers_every_family_present():
    rows = _synthetic_rows()
    sample = stratified_sample(rows, 7, {task: 20 for task in TASKS})
    families = {family_of(r.question, r.options) for r in sample}
    assert families == {"instance_count", "activity_semantic"}


def test_stratified_sample_prefers_distinct_videos_within_a_task():
    rows = _synthetic_rows()
    sample = stratified_sample(rows, 7, {"qa": 20})
    keys = [r.video_key for r in sample if r.task == "qa"]
    assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------------- #
# Stats and IO
# --------------------------------------------------------------------------- #

def test_build_stats_flags_a_broken_answer_key():
    rows = [make_row(answer="E")]
    issues = build_stats(rows)["data_issues"]
    assert issues and "not one of" in issues[0]["issue"]


def test_build_stats_counts_the_shortcut_indicators():
    rows = [
        make_row(
            qa_index=i,
            options=parse_options(["A. short", "B. a considerably longer option"]),
            answer="B",
        )
        for i in range(4)
    ]
    shortcut = build_stats(rows)["shortcut_indicators"]["by_task"]["all"]
    assert shortcut["answer_is_longest_option"] == 100.0
    assert shortcut["answer_is_A"] == 0.0


def test_write_manifest_round_trips(tmp_path: Path):
    payload = {"a": 1, "b": ["c"]}
    path = write_manifest(payload, tmp_path / "m.json")
    assert read_manifest(path) == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_file_sha256_matches_hashlib(tmp_path: Path):
    import hashlib

    path = tmp_path / "f.bin"
    path.write_bytes(b"dyn-bench")
    assert file_sha256(path) == hashlib.sha256(b"dyn-bench").hexdigest()


def test_task_to_level_covers_every_task_with_a_real_paper_level():
    assert set(TASK_TO_LEVEL) == set(TASKS)
    from dyn_dataset_exploration.dyn_bench_data import PAPER_LEVELS

    assert set(TASK_TO_LEVEL.values()) == set(PAPER_LEVELS)
    assert len(PAPER_TASKS) == 9


# --------------------------------------------------------------------------- #
# Against the real dataset
# --------------------------------------------------------------------------- #

@needs_dataset
def test_real_dataset_shape():
    rows = load_rows(qa_parquet(DEFAULT_DYN_ROOT))
    assert len(rows) == 7000
    assert {r.task for r in rows} == set(TASKS)
    assert len({r.video_key for r in rows}) == 1000
    # The export ships four lowercase answers; load_rows must have folded them.
    assert {r.answer for r in rows} == {"A", "B", "C", "D"}


@needs_dataset
def test_real_dataset_has_four_options_and_a_valid_answer_everywhere():
    rows = load_rows(qa_parquet(DEFAULT_DYN_ROOT))
    assert all(len(r.options) == 4 for r in rows)
    assert all(r.answer in r.options for r in rows)


def test_build_stats_carries_the_normalisation_record():
    """What load_rows silently fixed must survive into the artifact."""

    stats = build_stats([make_row()], {"lowercase_answer_rows": 4, "whitespace_answer_rows": 0})
    assert stats["normalisation"]["lowercase_answer_rows"] == 4


def test_build_stats_defaults_the_normalisation_record_to_zero():
    assert build_stats([make_row()])["normalisation"]["lowercase_answer_rows"] == 0


@needs_dataset
def test_real_dataset_ships_four_lowercase_answers():
    counts = answer_normalisations(qa_parquet(DEFAULT_DYN_ROOT))
    assert counts["lowercase_answer_rows"] == 4
    assert counts["whitespace_answer_rows"] == 0
