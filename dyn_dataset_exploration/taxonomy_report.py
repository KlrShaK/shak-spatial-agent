"""Report generator for the Dyn-Bench taxonomy exploration.

Reads the artifacts written by ``dyn_bench_data.py`` (``index.json``,
``stats.json``, ``templates.json``, ``manifest.json``) and
``taxonomy_classify.py`` (``classify/*.json``, ``classify/review_queue.json``)
and renders ``TAXONOMY_REPORT.md``. No network access.

The report has one job the DSI-Bench report did not: say, per measurement
bucket, whether ``d4rt_agent/agent_v4`` already produces it. That column is a
lookup on ``buckets.Bucket.v4_builder``, not a judgement made here, so it cannot
drift from the bucket definitions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .buckets import BUCKET_BY_NAME, BUCKET_KEYS, BUCKETS, OTHER_BUCKET
from .dyn_bench_data import (
    DEFAULT_RESULTS_DIR,
    FAMILY_DESCRIPTIONS,
    LEVEL_OF_PAPER_TASK,
    PAPER_LEVELS,
    PAPER_TASKS,
    TASKS,
    TASK_TO_LEVEL,
)
from .taxonomy_classify import load_classifications

# How many example questions to quote per bucket. Enough to make a bucket
# recognisable, few enough that the report stays readable.
EXAMPLES_PER_BUCKET = 2


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [f"| {' | '.join(str(h) for h in header)} |"]
    lines.append(f"| {' | '.join('---' for _ in header)} |")
    for row in rows:
        lines.append(f"| {' | '.join(str(cell) for cell in row)} |")
    return "\n".join(lines)


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "-"


def _coverage_label(name: str) -> str:
    """How to render one bucket's AgentV4 coverage.

    ``semantic_only`` gets its own label rather than "new work": nothing has to
    be *measured* for it, only recognised, so counting it as a missing
    measurement would overstate the pipeline work AgentV5 needs. It is still not
    covered -- AgentV4 has no way to decide a question needs no measurement, and
    runs the GPU pipeline anyway.
    """

    if name == OTHER_BUCKET:
        return "n/a"
    bucket = BUCKET_BY_NAME[name]
    if bucket.covered_by_v4:
        return f"`{bucket.v4_builder}`"
    if name == "semantic_only":
        return "**new routing** (no measurement)"
    return "**new work**"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def bucket_section(
    classifications: Mapping[str, Mapping[str, Any]],
    questions: Mapping[str, Mapping[str, Any]],
) -> str:
    """Per-bucket counts, AgentV4 coverage, and a couple of real questions.

    Buckets with zero hits are kept in the table, not dropped: a candidate that
    no sampled question lands in is a candidate AgentV5 does not have to build,
    and that is a result worth stating explicitly.
    """

    counts: Counter[str] = Counter(r["parsed"]["bucket"] for r in classifications.values())
    examples: dict[str, list[str]] = {}
    for qid, record in sorted(classifications.items()):
        bucket = record["parsed"]["bucket"]
        question = questions.get(qid, {}).get("question", "")
        chosen = examples.setdefault(bucket, [])
        # Distinct text only: `cameraqa` repeats a handful of stems hundreds of
        # times, and two identical example lines illustrate nothing.
        if question and question not in chosen and len(chosen) < EXAMPLES_PER_BUCKET:
            chosen.append(question)

    total = sum(counts.values())
    rows = []
    for key in BUCKET_KEYS:
        bucket = BUCKET_BY_NAME.get(key)
        count = counts.get(key, 0)
        description = bucket.description if bucket else "none of the above fit"
        rows.append([f"`{key}`", count, _pct(count, total), _coverage_label(key), description])

    lines = [
        _table(["bucket", "n", "share", "AgentV4 coverage", "quantity measured"], rows),
        "",
        "### What each bucket looks like in the data",
        "",
    ]
    for key in BUCKET_KEYS:
        if not counts.get(key):
            continue
        lines.append(f"- **`{key}`** ({counts[key]})")
        for question in examples.get(key, []):
            lines.append(f"  - *{question}*")
    return "\n".join(lines)


def population_estimate(
    classifications: Mapping[str, Mapping[str, Any]], stats: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Reweight the stratified sample back onto all 7000 rows.

    The draw was equal-allocation across question families, deliberately *not*
    proportional to the benchmark's own skew -- so a share read straight off the
    sample is a share of the sample, not of Dyn-Bench. Post-stratification fixes
    that: within each (coarse task, family) cell the sampled questions are a
    uniform draw, so the cell's bucket distribution scales up by the cell's true
    population size, which ``stats.json`` already counted exhaustively.

    A cell the sample missed entirely contributes nothing and is reported in
    ``uncovered_rows``, rather than being silently spread over the buckets -- an
    unsampled stratum is missing information, not evidence for the average.
    """

    sampled: dict[tuple[str, str], Counter[str]] = {}
    for record in classifications.values():
        cell = (record["task"], record["family"])
        sampled.setdefault(cell, Counter())[record["parsed"]["bucket"]] += 1

    estimate: Counter[str] = Counter()
    uncovered = 0
    covered = 0
    for task, info in stats["per_task"].items():
        for family, population in info["primary_family"].items():
            counts = sampled.get((task, family))
            if not counts:
                uncovered += population
                continue
            covered += population
            drawn = sum(counts.values())
            for bucket, count in counts.items():
                estimate[bucket] += population * count / drawn

    total = sum(estimate.values())
    return {
        "covered_rows": covered,
        "uncovered_rows": uncovered,
        "by_bucket": {
            bucket: {"rows": round(value), "share": _pct(round(value), round(total))}
            for bucket, value in sorted(estimate.items(), key=lambda kv: -kv[1])
        },
    }


def population_table(estimate: Mapping[str, Any]) -> str:
    rows = [
        [f"`{bucket}`", _coverage_label(bucket), info["rows"], info["share"]]
        for bucket, info in estimate["by_bucket"].items()
    ]
    return _table(["bucket", "AgentV4 coverage", "estimated rows (of 7000)", "share"], rows)


def paper_task_section(classifications: Mapping[str, Mapping[str, Any]]) -> str:
    counts: Counter[str] = Counter(r["parsed"]["paper_task"] for r in classifications.values())
    total = sum(counts.values())
    lines: list[str] = []
    for level, names in PAPER_LEVELS.items():
        level_total = sum(counts.get(name, 0) for name in names)
        lines.append(f"**{level}** — {level_total} of {total} sampled questions")
        lines.append("")
        lines.append(
            _table(
                ["sub-task", "n", "share"],
                [[f"`{name}`", counts.get(name, 0), _pct(counts.get(name, 0), total)] for name in names],
            )
        )
        lines.append("")
    if counts.get(OTHER_BUCKET):
        lines.append(f"`OTHER`: {counts[OTHER_BUCKET]} questions fit none of the nine.")
        lines.append("")
    return "\n".join(lines)


def level_agreement_table(classifications: Mapping[str, Mapping[str, Any]]) -> str:
    """Does the shipped coarse task predict the paper's level?

    This is the Dyn-Bench equivalent of the DSI pass's ``cate`` ↔ bucket
    agreement table. ``TASK_TO_LEVEL`` is the hypothesis; this counts how often
    the blind classifier lands in the hypothesised level.
    """

    rows = []
    for task in TASKS:
        subset = [r for r in classifications.values() if r["task"] == task]
        hypothesis = TASK_TO_LEVEL[task]
        agree = sum(
            1
            for r in subset
            if LEVEL_OF_PAPER_TASK.get(r["parsed"]["paper_task"]) == hypothesis
        )
        other = sum(1 for r in subset if r["parsed"]["paper_task"] == OTHER_BUCKET)
        rows.append([f"`{task}`", hypothesis, len(subset), agree, other, _pct(agree, len(subset))])
    return _table(
        ["coarse task", "hypothesised level", "n", "agree", "OTHER", "agreement"], rows
    )


def cross_tab(classifications: Mapping[str, Mapping[str, Any]]) -> str:
    """Paper sub-task (rows) × measurement bucket (columns), non-empty cells only.

    Rendered as a list rather than a 19-column matrix: at 600 questions the
    matrix is mostly zeros, and the list makes the few dense cells -- the ones
    that say which published sub-task a new AgentV5 bucket would actually move --
    readable without horizontal scrolling.
    """

    pairs: Counter[tuple[str, str]] = Counter(
        (r["parsed"]["paper_task"], r["parsed"]["bucket"]) for r in classifications.values()
    )
    by_task: dict[str, list[tuple[str, int]]] = {}
    for (paper_task, bucket), count in pairs.items():
        by_task.setdefault(paper_task, []).append((bucket, count))

    lines: list[str] = []
    for paper_task in (*PAPER_TASKS, OTHER_BUCKET):
        entries = sorted(by_task.get(paper_task, []), key=lambda kv: -kv[1])
        if not entries:
            continue
        total = sum(count for _, count in entries)
        rendered = ", ".join(f"`{bucket}` {count}" for bucket, count in entries)
        lines.append(f"- **`{paper_task}`** ({total}) → {rendered}")
    return "\n".join(lines)


def family_to_bucket_table(classifications: Mapping[str, Mapping[str, Any]]) -> str:
    """Regex family (what we guessed) vs LLM bucket (what it turned out to be).

    The families exist to stratify the sample, not to classify it. This table is
    how much they happened to agree -- and where they did not, the family regex
    is the thing that was wrong, not the bucket.
    """

    by_family: dict[str, Counter[str]] = {}
    for record in classifications.values():
        by_family.setdefault(record["family"], Counter())[record["parsed"]["bucket"]] += 1
    rows = []
    for family in sorted(by_family, key=lambda f: -sum(by_family[f].values())):
        counts = by_family[family]
        total = sum(counts.values())
        top = ", ".join(f"`{b}` {n}" for b, n in counts.most_common(3))
        rows.append([f"`{family}`", FAMILY_DESCRIPTIONS.get(family, ""), total, top])
    return _table(["sampling family", "regex intent", "n", "buckets the LLM assigned"], rows)


# --------------------------------------------------------------------------- #
# Whole report
# --------------------------------------------------------------------------- #

def build_report(
    index: Mapping[str, Any],
    stats: Mapping[str, Any],
    templates: Mapping[str, Any],
    manifest: Mapping[str, Any],
    classifications: Mapping[str, Mapping[str, Any]],
    review_queue: Mapping[str, Any] | None,
) -> str:
    questions = {str(q["question_id"]): q for q in manifest["questions"]}
    total = len(classifications)
    selected = manifest["sampling"]["total_selected"]
    counts: Counter[str] = Counter(r["parsed"]["bucket"] for r in classifications.values())
    estimate = population_estimate(classifications, stats)
    uncovered = sum(
        count
        for bucket, count in counts.items()
        if bucket == OTHER_BUCKET or not BUCKET_BY_NAME[bucket].covered_by_v4
    )
    semantic = counts.get("semantic_only", 0)
    unbucketed = counts.get(OTHER_BUCKET, 0)
    new_measurements = uncovered - semantic - unbucketed

    lines: list[str] = [
        "# Dyn-Bench task-type taxonomy",
        "",
        f"Deterministic scans over all {stats['total_rows']} VQA rows; blind dual-axis LLM "
        f"classification of {total} of {selected} stratified-sampled questions. "
        f"{index['counts']['clips']} clips resolved to media, "
        f"{index['counts']['unresolved']} unresolved.",
        "",
        f"**Headline:** {uncovered} of {total} sampled questions ({_pct(uncovered, total)}) fall "
        "outside the four measurement buckets `d4rt_agent/agent_v4` implements. "
        f"{new_measurements} of those name a specific quantity AgentV4 does not produce, "
        f"{semantic} need no measurement at all (`semantic_only`), and {unbucketed} fit no "
        "bucket in the list.",
        "",
        "## 1. What the dataset ships",
        "",
        _table(
            ["coarse task", "rows", "distinct videos", "distinct questions", "answer histogram"],
            [
                [
                    f"`{task}`",
                    info["rows"],
                    info["distinct_videos"],
                    info["distinct_questions"],
                    info["answer_histogram"],
                ]
                for task, info in stats["per_task"].items()
            ],
        ),
        "",
        _table(
            ["source dataset", "rows"],
            [[f"`{name}`", count] for name, count in stats["per_dataset"].items()],
        ),
        "",
        "## 2. The template scan, and why it is not enough",
        "",
        "The DSI-Bench pass settled its taxonomy with an exhaustive regex template scan: "
        "1769 rows collapsed to 34 templates, so every question shape was a countable "
        "fact. Rerunning that scan here is the first thing this pass did, and the result "
        "is the reason the rest of it looks different.",
        "",
        _table(
            ["coarse task", "rows", "distinct templates", "top-25 coverage"],
            [
                [
                    f"`{task}`",
                    info["rows"],
                    info["distinct_templates"],
                    f"{info['top25_coverage']} ({_pct(info['top25_coverage'], info['rows'])})",
                ]
                for task, info in templates["per_task"].items()
            ],
        ),
        "",
        "`cameraqa` is semi-templated and the scan works on it. `qa` and `sceneqa` are "
        "free-form generated text — 1578 and 1925 distinct templates for 2000 rows each — "
        "and no amount of regex tuning will collapse them. Hence the LLM open-coding pass "
        "below, and hence the regex *families* in `dyn_bench_data.FAMILIES`, which are a "
        "sampling stratum rather than a taxonomy.",
        "",
        "## 3. Sampling",
        "",
        f"{selected} questions, {manifest['counts']['distinct_videos']} distinct videos, "
        f"{manifest['sampling']['task_quotas']} per coarse task, equal allocation across "
        "question families capped by availability, with "
        f"{int(manifest['sampling']['rare_reserve_fraction'] * 100)}% of each family's quota "
        f"reserved for the four rare source datasets ({', '.join(manifest['sampling']['rare_datasets'])}).",
        "",
        _table(
            ["source dataset", "in sample", "in population"],
            [
                [f"`{name}`", count, stats["per_dataset"].get(name, 0)]
                for name, count in manifest["counts"]["per_dataset"].items()
            ],
        ),
        "",
        "## 4. Axis A — the paper's nine sub-tasks",
        "",
        "The paper defines these; the released parquet does not ship them. Recovered here "
        "by blind classification.",
        "",
        paper_task_section(classifications),
        "### Does the shipped coarse label predict the paper's level?",
        "",
        level_agreement_table(classifications),
        "",
        "The three shipped task files are not a clean partition of the paper's three "
        "levels. `cameraqa` is close to one, `qa` is mostly one, and `sceneqa` is not: "
        "fewer than half of its questions read as object-scene tracking, and most of the "
        "rest read as inter-object perception. Anyone reporting per-level Dyn-Bench "
        "numbers by grouping on the shipped `task` column is not reporting the paper's "
        "levels.",
        "",
        "## 5. Axis B — measurement buckets",
        "",
        "What physical quantity, in what reference frame, would have to be measured for "
        "the question to be decidable. The first four buckets are what AgentV4 implements "
        "today; the rest are candidates this pass was run to confirm or rule out.",
        "",
        "*Method note.* The codebook was revised once. The first pass left 21 of 600 "
        "questions in `OTHER`, and reading them showed one missing distinction stated "
        "repeatedly: a scene object's trajectory expressed in the **camera's** frame, or "
        "in a frame fixed to the **static scene**, as opposed to the object's own axes. "
        "`traj_subject_camera_frame` and `traj_subject_scene_frame` were added and all 600 "
        "questions were reclassified under the extended list, so every count below comes "
        "from one codebook. `OTHER` fell from 21 to 6.",
        "",
        bucket_section(classifications, questions),
        "",
        "## 6. Scaled back to all 7000 rows",
        "",
        "The draw was equal-allocation across question families, so the shares above are "
        "shares of the sample. Post-stratifying each (coarse task, family) cell by its "
        "true population size, counted exhaustively in `stats.json`, gives the estimate "
        "for the benchmark as a whole.",
        "",
        population_table(estimate),
        "",
        f"Estimated from {estimate['covered_rows']} of {stats['total_rows']} rows; "
        f"{estimate['uncovered_rows']} rows fall in (task, family) cells the sample did "
        "not reach and are left out rather than averaged in.",
        "",
        "## 7. Cross-tab — paper sub-task × measurement bucket",
        "",
        cross_tab(classifications),
        "",
        "## 8. Sampling families vs assigned buckets",
        "",
        family_to_bucket_table(classifications),
        "",
        "## 9. Review queue",
        "",
    ]

    if review_queue is None:
        lines.append(
            "Not yet generated — run "
            "`python -m dyn_dataset_exploration.taxonomy_classify review`."
        )
    else:
        pending = sum(
            1 for item in review_queue.get("queue", []) if item.get("human_verdict") is None
        )
        lines.append(
            f"{review_queue['total_queued']} of {review_queue['total_classified']} "
            f"classifications queued for review, {pending} still awaiting `human_verdict`."
        )
        lines.append("")
        lines.append(
            _table(
                ["reason queued", "n"],
                [[reason, count] for reason, count in review_queue["queued_by_reason"].items()],
            )
        )

    lines += [
        "",
        "## 10. Data facts worth carrying into any Dyn-Bench evaluation",
        "",
    ]
    shortcut = stats["shortcut_indicators"]["by_task"]
    lines.append(
        _table(
            ["coarse task", "rows", "answer is the longest option", "answer is A"],
            [
                [
                    f"`{task}`",
                    info["rows"],
                    f"{info['answer_is_longest_option']}%",
                    f"{info['answer_is_A']}%",
                ]
                for task, info in shortcut.items()
            ],
        )
    )
    lines += [
        "",
        f"These are {stats['shortcut_indicators']['note']}.",
        "",
        f"{stats['normalisation']['lowercase_answer_rows']} rows ship a lowercase `answer` "
        f"letter and are folded to upper case on load; {len(stats['data_issues'])} rows have "
        "a structural issue that survives loading (see `stats.json`).",
        "",
        "## 11. Scope and next step",
        "",
        "This pass covers the 7000-row VQA track only. The 3000 dynamic-object-grounding "
        "rows (`cameramask`/`objmask`/`scenemask`, scored by J&F over per-frame masks) are "
        "untouched, and are the largest single capability gap: AgentV4's `final_answer` "
        "action is text-only (`kind` must be `\"text\"`), so it has no way to emit geometry "
        "at all.",
        "",
        "Turning this taxonomy into AgentV5 buckets and tools is not designed here. See "
        "`BUCKET_REQUIREMENTS_V5.md` for the per-bucket input/output specs and "
        "`AGENT_V5_GAPS.md` for what has to change in `d4rt_agent/`.",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    classify_dir = results_dir / "classify"
    review_path = classify_dir / "review_queue.json"

    report = build_report(
        index=_load_json(results_dir / "index.json"),
        stats=_load_json(results_dir / "stats.json"),
        templates=_load_json(results_dir / "templates.json"),
        manifest=_load_json(results_dir / "manifest.json"),
        classifications=load_classifications(classify_dir),
        review_queue=_load_json(review_path) if review_path.exists() else None,
    )
    out_path = results_dir / "TAXONOMY_REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
