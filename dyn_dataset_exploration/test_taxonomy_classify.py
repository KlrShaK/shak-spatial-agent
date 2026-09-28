"""Offline tests for the dual-axis classifier's request construction and parsing.

No network: ``call_chat_completion`` is never exercised here. What is under test
is everything around it -- the blindness of the prompt, the enums the response
format pins, the defensive parse, and the rule that decides what lands in the
review queue.

The blindness test is the load-bearing one. The counts this pass produces are
only worth something if the classifier could not see the coarse task label, the
source dataset, or the answer key, so that property is asserted by exact
reconstruction of the message rather than by substring search -- several bucket
names legitimately contain family names as substrings, and a substring test
would either false-positive on those or be weakened until it tested nothing.
"""

from __future__ import annotations

import pytest

from dyn_dataset_exploration.buckets import (
    BUCKET_KEYS,
    BUCKET_NAMES,
    OTHER_BUCKET,
    V4_BUCKETS,
    bucket_glossary,
)
from dyn_dataset_exploration.dyn_bench_data import (
    PAPER_TASKS,
    format_question_prompt,
)
from dyn_dataset_exploration.taxonomy_classify import (
    PAPER_TASK_KEYS,
    classify_messages,
    classify_response_format,
    extract_labels,
    paper_task_glossary,
    review_reasons,
)

SYSTEM = "SYSTEM PROMPT"
ENTRY = {
    "question_id": "qa_davis_bike_packing_abcd1234_0",
    "question": "How does the bicycle move relative to the staircase?",
    "options": {"A": "closer", "B": "farther", "C": "unchanged", "D": "past it"},
    # Everything below must never reach the model.
    "task": "qa",
    "dataset": "davis",
    "video": "bike-packing",
    "answer": "C",
    "family": "object_object_relation",
    "hypothesis_level": "Dynamic Inter-Object Perception",
}


# --------------------------------------------------------------------------- #
# Blindness
# --------------------------------------------------------------------------- #

def test_classify_messages_is_exactly_system_plus_question_and_glossaries():
    messages = classify_messages(ENTRY, SYSTEM)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM
    assert messages[1]["content"] == (
        f"Question:\n{format_question_prompt(ENTRY)}\n\n"
        f"Axis 1 -- paper_task, choose one:\n{paper_task_glossary()}\n\n"
        f"Axis 2 -- bucket, choose one:\n{bucket_glossary()}"
    )


def test_classify_messages_never_carries_the_scoring_side_fields():
    """The source dataset, the clip name and the family label are all withheld.

    The hypothesised *level* is not on this list: all three levels appear in the
    Axis-1 glossary by design, so there is nothing about ours to leak.
    """

    blob = "".join(m["content"] for m in classify_messages(ENTRY, SYSTEM))
    for withheld in ("davis", "bike-packing", "object_object_relation"):
        assert withheld not in blob


def test_bucket_glossary_hides_which_buckets_agent_v4_already_implements():
    """Naming the implemented ones would bias the classifier toward them."""

    glossary = bucket_glossary()
    assert "traj3d_buckets" not in glossary
    assert "AgentV4" not in glossary


# --------------------------------------------------------------------------- #
# Response format
# --------------------------------------------------------------------------- #

def test_response_format_pins_both_enums_and_forbids_extra_keys():
    schema = classify_response_format()["json_schema"]["schema"]
    assert schema["properties"]["paper_task"]["enum"] == list(PAPER_TASK_KEYS)
    assert schema["properties"]["bucket"]["enum"] == list(BUCKET_KEYS)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "paper_task",
        "paper_rationale",
        "bucket",
        "bucket_rationale",
    }


def test_enums_contain_every_taxonomy_key_plus_other():
    assert set(PAPER_TASK_KEYS) == set(PAPER_TASKS) | {OTHER_BUCKET}
    assert set(BUCKET_KEYS) == set(BUCKET_NAMES) | {OTHER_BUCKET}
    assert set(V4_BUCKETS) < set(BUCKET_NAMES)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def test_extract_labels_reads_a_well_formed_reply():
    parsed = extract_labels(
        '{"paper_task": "Spatial Rel. & Change", "paper_rationale": "r1", '
        '"bucket": "subject_subject_relation", "bucket_rationale": "r2"}'
    )
    assert parsed["paper_task"] == "Spatial Rel. & Change"
    assert parsed["bucket"] == "subject_subject_relation"
    assert parsed["paper_rationale"] == "r1"
    assert parsed["bucket_rationale"] == "r2"


@pytest.mark.parametrize(
    "content",
    ["", "not json at all", "{}", '{"bucket": "invented_bucket", "paper_task": "Nope"}', "null"],
)
def test_extract_labels_never_invents_a_label(content):
    """A model that bypasses strict mode must show up as OTHER, not as a new key."""

    parsed = extract_labels(content)
    assert parsed["paper_task"] == OTHER_BUCKET
    assert parsed["bucket"] == OTHER_BUCKET


def test_extract_labels_keeps_a_valid_axis_when_the_other_is_junk():
    parsed = extract_labels(
        '{"paper_task": "Cam. Motion & Orient.", "bucket": "nonsense", '
        '"paper_rationale": "", "bucket_rationale": ""}'
    )
    assert parsed["paper_task"] == "Cam. Motion & Orient."
    assert parsed["bucket"] == OTHER_BUCKET


# --------------------------------------------------------------------------- #
# Review queue rules
# --------------------------------------------------------------------------- #

def _record(paper_task: str, bucket: str, hypothesis_level: str) -> dict:
    return {
        "question_id": "q",
        "task": "qa",
        "family": "f",
        "hypothesis_level": hypothesis_level,
        "parsed": {
            "paper_task": paper_task,
            "bucket": bucket,
            "paper_rationale": "",
            "bucket_rationale": "",
        },
    }


def test_an_implemented_bucket_in_the_hypothesised_level_is_not_queued():
    record = _record(
        "Spatial Rel. & Change", "traj_subject_own_frame", "Dynamic Inter-Object Perception"
    )
    assert review_reasons(record) == []


def test_a_candidate_bucket_is_always_queued():
    """Only a bucket AgentV4 lacks can create AgentV5 work, so every one is read."""

    record = _record(
        "Spatial Rel. & Change", "subject_subject_relation", "Dynamic Inter-Object Perception"
    )
    assert any("not implemented in AgentV4" in r for r in review_reasons(record))


def test_a_level_mismatch_is_queued():
    record = _record(
        "Cam. Motion & Orient.", "traj_camera_own_frame", "Dynamic Inter-Object Perception"
    )
    assert any(r.startswith("level mismatch") for r in review_reasons(record))


def test_other_on_either_axis_is_queued():
    both = _record(OTHER_BUCKET, OTHER_BUCKET, "Dynamic Inter-Object Perception")
    reasons = review_reasons(both)
    assert "paper_task is OTHER" in reasons
    assert "bucket is OTHER" in reasons
