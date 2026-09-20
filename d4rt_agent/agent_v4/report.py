"""Score an agent-v4 run and say where it went wrong.

Accuracy alone cannot tell a good measurement from a lucky guess, and this
design has two separable failure modes that a single number hides:

  * the agent picked the wrong BUCKET -- a classification failure
  * the bucket was right but the measurement was unavailable or misread

So the bucket-vs-cate matrix and the measurement-availability rate are reported
beside the score. A question answered with `measured=False` was answered from
the frames alone: it may still be right, but the pipeline contributed nothing to
it, and counting those as wins would overstate what this architecture does.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Sequence

from d4rt_agent.agent_v4.buckets import CATE_TO_BUCKET, NAMES
from d4rt_agent.dsi_bench_data import parse_choice_letter


def load(results_dir: Path) -> list[dict[str, Any]]:
    answers = sorted((results_dir / "answers").glob("*.json"))
    return [json.loads(p.read_text()) for p in answers]


def chosen_letter(record: dict[str, Any]) -> str | None:
    text = record.get("answer_text")
    if not isinstance(text, str) or not text.strip():
        return None
    letters = list((record.get("reference") or {}).get("options") or {})
    return parse_choice_letter(text, letters or None)


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})" if denominator else "0/0"


def summarise(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    errored = [r for r in records if r.get("status") == "error"]
    answered = [r for r in records if r.get("answered")]
    scored = [(r, chosen_letter(r)) for r in answered]
    unparsed = [r for r, letter in scored if letter is None]
    correct = [r for r, letter in scored
               if letter and letter == (r.get("reference") or {}).get("gt")]

    measured = [r for r in answered if r.get("measurement_available")]
    unmeasured = [r for r in answered if not r.get("measurement_available")]
    correct_measured = [r for r in correct if r.get("measurement_available")]

    by_cate: dict[int, list[Any]] = collections.defaultdict(list)
    confusion: collections.Counter = collections.Counter()
    for record, letter in scored:
        ref = record.get("reference") or {}
        cate = ref.get("cate")
        by_cate[cate].append(letter == ref.get("gt"))
        confusion[(ref.get("expected_bucket"), record.get("chosen_bucket"))] += 1

    reasons = collections.Counter(
        (r.get("measurement_unavailable_reason") or "?").split(":")[0][:60] for r in unmeasured
    )
    return {
        "total": total, "errored": errored, "answered": answered, "unparsed": unparsed,
        "correct": correct, "measured": measured, "unmeasured": unmeasured,
        "correct_measured": correct_measured, "by_cate": by_cate,
        "confusion": confusion, "unavailable_reasons": reasons,
    }


def render(results_dir: Path) -> str:
    records = load(results_dir)
    if not records:
        return f"no answers under {results_dir}"
    s = summarise(records)
    out: list[str] = [f"# agent v4 — {results_dir}", ""]

    out.append(f"answered            {_pct(len(s['answered']), s['total'])}")
    if s["errored"]:
        out.append(f"errored             {len(s['errored'])}")
    if s["unparsed"]:
        out.append(f"no option letter    {len(s['unparsed'])}")
    out.append(f"ACCURACY            {_pct(len(s['correct']), len(s['answered']))}")
    out.append("")
    out.append("Measurement is the point of the design, so score it separately:")
    out.append(f"  measured          {_pct(len(s['measured']), len(s['answered']))}")
    out.append(f"  accuracy WHERE measured   {_pct(len(s['correct_measured']), len(s['measured']))}")
    guessed = len(s["correct"]) - len(s["correct_measured"])
    out.append(f"  correct WITHOUT a measurement  {guessed}"
               f"  <- right from the frames alone, not evidence the pipeline works")
    if s["unavailable_reasons"]:
        out.append("  why unavailable:")
        for reason, count in s["unavailable_reasons"].most_common():
            out.append(f"     {count:3d}  {reason}")

    out += ["", "## accuracy by category", ""]
    out.append(f"{'cate':>5s}  {'expected bucket':<30s} {'n':>4s}  correct")
    for cate in sorted(c for c in s["by_cate"] if c is not None):
        hits = s["by_cate"][cate]
        out.append(f"{cate:5d}  {CATE_TO_BUCKET.get(cate, '?'):<30s} {len(hits):4d}  "
                   f"{_pct(sum(hits), len(hits))}")

    out += ["", "## did the agent pick the bucket the taxonomy expects?", ""]
    agreed = sum(n for (exp, got), n in s["confusion"].items() if exp == got)
    total_c = sum(s["confusion"].values())
    out.append(f"agreement {_pct(agreed, total_c)}")
    disagreements = {k: n for k, n in s["confusion"].items() if k[0] != k[1]}
    if disagreements:
        out.append("")
        out.append(f"{'expected':<30s} {'chosen':<30s} n")
        for (expected, got), count in sorted(disagreements.items(), key=lambda kv: -kv[1]):
            out.append(f"{str(expected):<30s} {str(got):<30s} {count}")

    out += ["", "## pipeline health", ""]
    ambiguous = sum(1 for r in s["answered"] if r.get("grounding_ambiguous"))
    retried = sum(1 for r in s["answered"] if (r.get("subject_attempts") or 0) > 1)
    out.append(f"grounding ambiguous, agent asked to choose   {_pct(ambiguous, len(s['answered']))}")
    out.append(f"subject re-named after a failed grounding     {_pct(retried, len(s['answered']))}")
    steps = [r.get("steps_used") or 0 for r in s["answered"]]
    walls = [r.get("wall_seconds") or 0.0 for r in s["answered"]]
    if steps:
        out.append(f"steps per question   mean {sum(steps)/len(steps):.1f}  max {max(steps)}")
    if walls:
        out.append(f"seconds per question mean {sum(walls)/len(walls):.0f}  max {max(walls):.0f}"
                   f"  total {sum(walls)/3600:.1f} h")

    wrong = [(r, chosen_letter(r)) for r in s["answered"]
             if chosen_letter(r) != (r.get("reference") or {}).get("gt")]
    if wrong:
        out += ["", "## every miss", ""]
        for record, letter in wrong[:40]:
            ref = record.get("reference") or {}
            gt = ref.get("gt")
            out.append(f"- cate {ref.get('cate')}  {record.get('question_id','?')[:52]}")
            out.append(f"    bucket={record.get('chosen_bucket')} measured={record.get('measurement_available')}")
            out.append(f"    said {letter}: {ref.get('options', {}).get(letter, '?')}"
                       f"   |  GT {gt}: {ref.get('options', {}).get(gt, '?')}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, nargs="?",
                        default=Path("d4rt_agent/results/agent_v4"))
    args = parser.parse_args(argv)
    print(render(args.results_dir))


if __name__ == "__main__":
    main()
