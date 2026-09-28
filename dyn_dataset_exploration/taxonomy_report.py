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
