"""CPU-only tests for the taxonomy seed data: template scan, hypothesis mapping,
and stratified sampling. No network, no dataset download required -- the
fixture rows below stand in for a slice of std.csv.
"""

from __future__ import annotations

import unittest

from d4rt_agent.dsi_bench_data import DSIRow

from dsi_dataset_exploration.taxonomy_data import (
    BUCKET_KEYS,
    BUCKETS,
    CATE_TO_BUCKET_HYPOTHESIS,
    OTHER_BUCKET,
    distinct_question_templates,
    stratified_sample,
)


def _row(cate: int, path: str, question: str, csv_row_index: int, options=None, gt="A") -> DSIRow:
    return DSIRow(
        cate=cate,
        dataset="internet",
        relative_path=path,
        video_type="1.0",
        question=question,
        options=options or {"A": "Moving forward", "B": "Moving backward"},
        gt=gt,
        others="",
        csv_row_index=csv_row_index,
    )


class TemplateScanTest(unittest.TestCase):
    def test_collapses_subject_noun_variants_to_one_template(self) -> None:
        rows = [
            _row(
                0,
                "a.mp4",
                "In this video clip, how is the character's location moving in the 3D "
                "scene relative to his/her/its own starting orientation and location?",
                0,
            ),
            _row(
                1,
                "b.mp4",
                "In this video clip, how is the car's location moving in the 3D scene "
                "relative to his/her/its own starting orientation and location?",
                1,
            ),
        ]
        templates = distinct_question_templates(rows)
        self.assertEqual(len(templates), 1)
        (info,) = templates.values()
        self.assertEqual(info["count"], 2)
        self.assertEqual(info["by_cate"], {0: 1, 1: 1})

    def test_distinct_wording_gives_distinct_templates(self) -> None:
        rows = [
            _row(4, "a.mp4", "How does the distance between the observer and the car change in this video?", 0),
            _row(
                5,
                "b.mp4",
                "In this video, how does the observer's position change relative to the car's own orientation?",
                1,
            ),
        ]
        templates = distinct_question_templates(rows)
        self.assertEqual(len(templates), 2)

    def test_rare_possessive_noun_still_normalizes(self) -> None:
        # "raccoon" and "dragon" are not in the fixed noun list, but the
        # possessive-phrase fallback still collapses "the X's" regardless.
        rows = [
            _row(0, "a.mp4", "In this video clip, how is the raccoon's location moving relative to its own starting orientation and location?", 0),
            _row(0, "b.mp4", "In this video clip, how is the dragon's location moving relative to its own starting orientation and location?", 1),
        ]
        templates = distinct_question_templates(rows)
        self.assertEqual(len(templates), 1)

    def test_rare_bare_noun_without_possessive_is_not_collapsed(self) -> None:
        # cate 4's template never uses a possessive, so a rare noun outside the
        # fixed list (matches real std.csv: ~30 singleton distance questions)
        # is left un-normalized -- this is expected, not a defect.
        rows = [
            _row(4, "a.mp4", "How does the distance between the observer and the raccoon change in this video?", 0),
            _row(4, "b.mp4", "How does the distance between the observer and the dragon change in this video?", 1),
        ]
        templates = distinct_question_templates(rows)
        self.assertEqual(len(templates), 2)


class BucketHypothesisTest(unittest.TestCase):
    def test_every_cate_has_a_hypothesis(self) -> None:
        self.assertEqual(set(CATE_TO_BUCKET_HYPOTHESIS), {0, 1, 2, 3, 4, 5})

    def test_hypotheses_are_known_buckets(self) -> None:
        for bucket in CATE_TO_BUCKET_HYPOTHESIS.values():
            self.assertIn(bucket, BUCKETS)

    def test_static_and_moving_cam_share_a_bucket(self) -> None:
        self.assertEqual(CATE_TO_BUCKET_HYPOTHESIS[0], CATE_TO_BUCKET_HYPOTHESIS[1])

    def test_static_and_dynamic_scene_share_a_bucket(self) -> None:
        self.assertEqual(CATE_TO_BUCKET_HYPOTHESIS[2], CATE_TO_BUCKET_HYPOTHESIS[3])

    def test_bucket_keys_includes_other_last(self) -> None:
        self.assertEqual(BUCKET_KEYS[-1], OTHER_BUCKET)
        self.assertEqual(len(BUCKET_KEYS), len(BUCKETS) + 1)


class StratifiedSampleTest(unittest.TestCase):
    def _pool(self, cate: int, n: int) -> list[DSIRow]:
        return [_row(cate, f"{cate}_{i}.mp4", f"question {cate}", i) for i in range(n)]

    def test_respects_quotas(self) -> None:
        rows = self._pool(0, 10) + self._pool(1, 10)
        quotas = {0: 3, 1: 4}
        selected = stratified_sample(rows, seed=1, quotas=quotas)
        self.assertEqual(sum(1 for r in selected if r.cate == 0), 3)
        self.assertEqual(sum(1 for r in selected if r.cate == 1), 4)

    def test_deterministic_for_a_fixed_seed(self) -> None:
        rows = self._pool(0, 20)
        quotas = {0: 5}
        first = stratified_sample(rows, seed=42, quotas=quotas)
        second = stratified_sample(rows, seed=42, quotas=quotas)
        self.assertEqual([r.relative_path for r in first], [r.relative_path for r in second])

    def test_distinct_videos_within_a_category(self) -> None:
        # Two rows share a video within cate 0 -- the draw must not pick both.
        rows = self._pool(0, 5)
        rows.append(_row(0, "0_0.mp4", "duplicate video question", 99))
        selected = stratified_sample(rows, seed=7, quotas={0: 5})
        paths = [r.relative_path for r in selected]
        self.assertEqual(len(paths), len(set(paths)))

    def test_raises_when_quota_exceeds_distinct_videos(self) -> None:
        rows = self._pool(0, 2)
        with self.assertRaises(ValueError):
            stratified_sample(rows, seed=1, quotas={0: 5})


if __name__ == "__main__":
    unittest.main()
