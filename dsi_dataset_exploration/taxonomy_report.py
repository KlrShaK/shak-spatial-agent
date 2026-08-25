"""Report generator for the DSI-Bench taxonomy exploration.

Reads the artifacts written by ``taxonomy_data.py`` (``templates.json``,
``manifest.json``) and ``taxonomy_classify.py`` (``classify/*.json``,
``classify/review_queue.json``) and renders the final nested-list taxonomy as
Markdown. No network access.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .taxonomy_data import BUCKET_KEYS, BUCKETS, CATE_TO_BUCKET_HYPOTHESIS, OTHER_BUCKET

DEFAULT_RESULTS_DIR = Path("dsi_dataset_exploration/logs")

CATEGORY_NAMES: tuple[str, ...] = (
    "Obj:static cam",
    "Obj:moving cam",
    "Cam:static scene",
    "Cam:dynamic scene",
    "Obj-Cam distance",
    "Obj-Cam orientation",
)


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_classifications(classify_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not classify_dir.exists():
        return records
    for path in sorted(classify_dir.glob("*.json")):
        if path.stem == "review_queue":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(record["question_id"])] = record
    return records


def _per_cate_bucket_counts(
    classifications: Mapping[str, dict[str, Any]],
) -> dict[int, Counter[str]]:
    per_cate: dict[int, Counter[str]] = {}
    for record in classifications.values():
        per_cate.setdefault(record["cate"], Counter())[record["parsed"]["bucket"]] += 1
    return per_cate


def _category_name(cate: int) -> str:
    return CATEGORY_NAMES[cate] if cate < len(CATEGORY_NAMES) else f"cate {cate}"


def build_bucket_agreement_table(classifications: Mapping[str, dict[str, Any]]) -> str:
    per_cate = _per_cate_bucket_counts(classifications)
    lines = [
        "| cate | category | hypothesis bucket | n | agree | OTHER | agreement % |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cate in sorted(per_cate):
        counts = per_cate[cate]
        total = sum(counts.values())
        hypothesis = CATE_TO_BUCKET_HYPOTHESIS.get(cate, "?")
        agree = counts.get(hypothesis, 0)
        other = counts.get(OTHER_BUCKET, 0)
        pct = 100.0 * agree / total if total else 0.0
        lines.append(
            f"| {cate} | {_category_name(cate)} | {hypothesis} | {total} | {agree} | "
            f"{other} | {pct:.0f}% |"
        )
    return "\n".join(lines)


def build_bucket_frequency_table(classifications: Mapping[str, dict[str, Any]]) -> str:
    frequency: Counter[str] = Counter(
        record["parsed"]["bucket"] for record in classifications.values()
    )
    lines = ["| bucket | description | times chosen |", "| --- | --- | --- |"]
    for key in BUCKET_KEYS:
        description = BUCKETS.get(key, "none of the above fit")
        lines.append(f"| {key} | {description} | {frequency.get(key, 0)} |")
    return "\n".join(lines)


def _cate_status(agreement_pct: float, n: int) -> str:
    if n == 0:
        return "not sampled"
    if agreement_pct >= 70:
        return "confirmed present"
    return "disputed -- see review queue"


def build_taxonomy_markdown(
    templates: Mapping[str, Any],
    manifest: Mapping[str, Any],
    classifications: Mapping[str, dict[str, Any]],
    review_queue: Mapping[str, Any] | None,
) -> str:
    per_cate = _per_cate_bucket_counts(classifications)

    bucket_to_cates: dict[str, list[int]] = {}
    for cate, bucket in CATE_TO_BUCKET_HYPOTHESIS.items():
        bucket_to_cates.setdefault(bucket, []).append(cate)

    lines: list[str] = ["# DSI-Bench task-type taxonomy", ""]
    total_selected = manifest.get("sampling", {}).get("total_selected")
    lines.append(
        f"Exhaustive template scan: {templates.get('total_rows')} rows -> "
        f"{templates.get('distinct_templates')} distinct question templates "
        f"(no LLM involved -- plain regex over every row). "
        f"Blind LLM classification: {len(classifications)} of {total_selected} "
        "sampled questions classified."
    )
    lines.append("")
    lines.append("## Taxonomy")
    lines.append("")
    for key, description in BUCKETS.items():
        cates = bucket_to_cates.get(key, [])
        lines.append(f"- **{key}** — {description}")
        if not cates:
            lines.append(
                "  - status: absent from DSI-Bench as a standalone question type "
                "(no `cate` maps here by hypothesis; whether it appears as an implicit "
                "sub-quantity inside another category's distractor options is open -- "
                "see \"Open questions\")"
            )
            continue
        for cate in cates:
            counts = per_cate.get(cate, Counter())
            n = sum(counts.values())
            agree = counts.get(key, 0)
            pct = 100.0 * agree / n if n else 0.0
            lines.append(
                f"  - cate {cate} ({_category_name(cate)}): {agree}/{n} blind LLM "
                f"agreement ({pct:.0f}%) — {_cate_status(pct, n)}"
            )
    lines.append("")
    lines.append("## cate ↔ bucket mapping")
    lines.append("")
    lines.append(build_bucket_agreement_table(classifications))
    lines.append("")
    lines.append("## Bucket frequency across the sample")
    lines.append("")
    lines.append(build_bucket_frequency_table(classifications))
    lines.append("")
    lines.append("## Open questions")
    lines.append("")
    lines.append(
        "- cate 5 (`Obj-Cam orientation`) is hypothesized as `camera_subject_frame` "
        "(observer position expressed in the subject's facing-defined frame), not "
        "`observer_orient_rel_subject` -- see its agreement row above and the review "
        "queue for any cate-5 disagreements."
    )
    lines.append(
        "- Buckets `subject_observer_frame`, `observer_orient_rel_subject`, "
        "`subject_orient_over_time`, `observer_orient_over_time`, and "
        "`subject_orient_rel_observer` have no standalone DSI-Bench question template. "
        "Whether any of them are tested implicitly (e.g. the rotating/orbiting distractor "
        "options in cate 0-3 may require knowing orientation-through-time to rule out) is "
        "not resolved by this pass and needs a dedicated look at the raw option text."
    )
    lines.append(
        "- `d4rt_agent/prompts/dsi_bench_system.md`'s existing \"Question -> measurement\" "
        "table has no row for cate 5's side-to-side transition phrasing -- a gap this "
        "taxonomy confirms independently of the LLM pass."
    )
    lines.append("")
    lines.append("## Review queue")
    lines.append("")
    if review_queue is None:
        lines.append("Not yet generated -- run `python -m dsi_dataset_exploration.taxonomy_classify review`.")
    else:
        total = review_queue.get("total_disagreements", 0)
        pending = sum(
            1 for item in review_queue.get("queue", []) if item.get("human_verdict") is None
        )
        lines.append(f"{total} disagreements logged, {pending} still awaiting `human_verdict`.")
    lines.append("")
    lines.append("## Future work (explicitly out of scope for this pass)")
    lines.append("")
    lines.append(
        "Wiring this taxonomy into a conditional/task-adaptive system prompt for the D4RT "
        "agent is not designed or implemented here -- this report is the identification "
        "step only."
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    templates = _load_json(results_dir / "templates.json")
    manifest = _load_json(results_dir / "manifest.json")
    classify_dir = results_dir / "classify"
    classifications = _load_classifications(classify_dir)
    review_path = classify_dir / "review_queue.json"
    review_queue = _load_json(review_path) if review_path.exists() else None

    report = build_taxonomy_markdown(templates, manifest, classifications, review_queue)
    out_path = results_dir / "TAXONOMY_REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
