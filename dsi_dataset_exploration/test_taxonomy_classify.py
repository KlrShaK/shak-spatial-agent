"""CPU-only tests for the taxonomy bucket classifier's pure functions.

No network, no model, no API key -- message/schema builders and the defensive
result parser are pure functions, exercised directly.
"""

from __future__ import annotations

import json
import unittest

from dsi_dataset_exploration.taxonomy_classify import (
    bucket_messages,
    bucket_response_format,
    extract_bucket,
)
from dsi_dataset_exploration.taxonomy_data import BUCKET_KEYS, OTHER_BUCKET


def _entry(**overrides) -> dict:
    entry = {
        "question_id": "internet_c1_x",
        "cate": 1,
        "category_name": "Obj:moving cam",
        "question": "How is the car's location moving?",
        "options": {"A": "Moving forward", "B": "Moving backward"},
        "option_letters": ["A", "B"],
        "gt": "A",
        "others": "some private annotator note",
        "hypothesis_bucket": "subject_own_frame",
    }
    entry.update(overrides)
    return entry


class RequestBuildingTest(unittest.TestCase):
    def test_messages_never_leak_ground_truth_or_category(self) -> None:
        messages = bucket_messages(_entry(), system_prompt="SYSTEM")
        user_content = messages[1]["content"]
        self.assertNotIn('"cate"', user_content)
        self.assertNotIn("Obj:moving cam", user_content)
        self.assertNotIn("some private annotator note", user_content)

    def test_messages_include_question_and_options(self) -> None:
        messages = bucket_messages(_entry(), system_prompt="SYSTEM")
        self.assertEqual(messages[0], {"role": "system", "content": "SYSTEM"})
        self.assertIn("How is the car's location moving?", messages[1]["content"])
        self.assertIn("Moving forward", messages[1]["content"])

    def test_response_format_enum_is_every_bucket_key(self) -> None:
        schema = bucket_response_format()["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]["bucket"]["enum"]), set(BUCKET_KEYS))
        self.assertEqual(schema["required"], ["bucket", "rationale"])
        self.assertFalse(schema["additionalProperties"])


class ResultParsingTest(unittest.TestCase):
    def test_extracts_structured_bucket(self) -> None:
        content = json.dumps({"bucket": "distance_change", "rationale": "scalar range"})
        bucket, rationale = extract_bucket(content)
        self.assertEqual(bucket, "distance_change")
        self.assertEqual(rationale, "scalar range")

    def test_unknown_bucket_falls_back_to_other(self) -> None:
        content = json.dumps({"bucket": "not_a_real_bucket", "rationale": "??"})
        bucket, _ = extract_bucket(content)
        self.assertEqual(bucket, OTHER_BUCKET)

    def test_unparseable_content_falls_back_to_other(self) -> None:
        bucket, rationale = extract_bucket("not json at all")
        self.assertEqual(bucket, OTHER_BUCKET)
        self.assertEqual(rationale, "")

    def test_other_is_a_valid_direct_choice(self) -> None:
        content = json.dumps({"bucket": "OTHER", "rationale": "doesn't fit"})
        bucket, _ = extract_bucket(content)
        self.assertEqual(bucket, "OTHER")


if __name__ == "__main__":
    unittest.main()
