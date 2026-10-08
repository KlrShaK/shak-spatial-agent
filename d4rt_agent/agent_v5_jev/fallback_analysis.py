"""Offline: does handing low-confidence answers to an existing baseline help?

    python -m d4rt_agent.agent_v5_jev.fallback_analysis <results-dir> [--out FALLBACK.md]

Baselines come from the decomposition study's eval_rows.json (Qwen letters, the decision flow), which exist
for every question, so this needs no new compute. The confidence threshold is the one free parameter: it is
chosen on one half of the CLIPS and scored on the other half (and the other way round), never on the
questions it is scored on.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

from d4rt_agent.agent_v5.report import DEFAULT_BASELINES, accuracy, bootstrap_ci, load_answers

THRESHOLDS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01)


def rows_with_baselines(results_dir: Path, baselines_path: Path) -> list[dict[str, Any]]:
    base = {b["qid"]: b for b in json.loads(Path(baselines_path).read_text())}
    out = []
    for p in sorted((Path(results_dir) / "answers").glob("*.json")):
        a = json.loads(p.read_text())
        b = base.get(a["question_id"])
        if b is None or not a.get("letter"):
            continue
        out.append({"qid": a["question_id"], "clip": a["question_id"].rsplit("_", 1)[0], "gt": b["gt"],
                    "luna": a["letter"], "conf": a.get("answer_confidence") or 0.0,
                    "letters": b["letters"], "flow": b["pipeline"], "frames": b["frames"],
                    "kind": a.get("kind"), "dataset": a["reference"].get("dataset")})
    return out


def gated(rows: Sequence[dict], fallback: str, t: float) -> list[dict]:
    return [dict(r, correct=float((r["luna"] if r["conf"] >= t else r[fallback]) == r["gt"])) for r in rows]


def cross_fitted(rows: Sequence[dict], fallback: str, seed: int = 0) -> tuple[float, list[float]]:
    """Accuracy of the gate when its threshold is chosen on the other half of the clips (2 folds, 5 repeats)."""

    clips = sorted({r["clip"] for r in rows})
    scores, chosen = [], []
    for rep in range(5):
        rng = random.Random(seed + rep)
        rng.shuffle(clips)
        half = set(clips[: len(clips) // 2])
        for train, test in ((half, None), (None, half)):
            tr = [r for r in rows if (r["clip"] in half) == (train is not None)]
            te = [r for r in rows if (r["clip"] in half) != (train is not None)]
            best = max(THRESHOLDS, key=lambda t: sum(x["correct"] for x in gated(tr, fallback, t)))
            chosen.append(best)
            scores.append(sum(x["correct"] for x in gated(te, fallback, best)) / max(len(te), 1))
    return sum(scores) / len(scores), chosen


def report(results_dir: Path, baselines_path: Path) -> str:
    rows = rows_with_baselines(results_dir, baselines_path)
    n = len(rows)
    if not n:
        return "no answers with baselines yet"
    out = [f"# confidence-gated fallback: {results_dir}", "", f"{n} answered questions that have baselines.", "",
           "| answering | accuracy | 95% CI (clips resampled) |", "|---|---|---|"]
    for name, key in (("Luna (jev-flow)", "luna"), ("Qwen letters", "letters"), ("decision flow (D7)", "flow"),
                      ("Qwen frames-only", "frames")):
        rs = [dict(r, correct=float(r[key] == r["gt"])) for r in rows]
        lo, hi = bootstrap_ci(rs)
        out.append(f"| {name} | {accuracy(rs)[0]:.1%} | {lo:.1%}..{hi:.1%} |")
    agree = [r for r in rows if r["luna"] == r["letters"]]
    out += ["", f"Luna and Qwen letters agree on {len(agree) / n:.1%} of questions; when they agree they are right "
            f"{sum(r['luna'] == r['gt'] for r in agree) / max(len(agree), 1):.1%} of the time, and when they disagree "
            f"Luna is right {sum(r['luna'] == r['gt'] for r in rows if r not in agree) / max(n - len(agree), 1):.1%}, "
            f"Qwen letters {sum(r['letters'] == r['gt'] for r in rows if r not in agree) / max(n - len(agree), 1):.1%}.",
            "", "## accuracy by Luna's confidence", "", "| confidence | n | Luna | Qwen letters | flow |", "|---|---|---|---|---|"]
    for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.01)):
        rs = [r for r in rows if lo <= r["conf"] < hi]
        if rs:
            out.append(f"| {lo:.2f}-{min(hi, 1):.2f} | {len(rs)} | {sum(r['luna'] == r['gt'] for r in rs) / len(rs):.1%} | "
                       f"{sum(r['letters'] == r['gt'] for r in rs) / len(rs):.1%} | {sum(r['flow'] == r['gt'] for r in rs) / len(rs):.1%} |")
    for fb, label in (("letters", "Qwen letters"), ("flow", "the decision flow")):
        out += ["", f"## Luna if confident, else {label}", "", "| threshold | accuracy (all questions) |", "|---|---|"]
        for t in THRESHOLDS:
            gs = gated(rows, fb, t)
            out.append(f"| {t:.2f} | {accuracy(gs)[0]:.1%} |")
        cf, chosen = cross_fitted(rows, fb)
        out.append(f"\nthreshold chosen on one half of the clips, scored on the other: **{cf:.1%}** "
                   f"(thresholds picked: {sorted(set(chosen))}).")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = report(args.results_dir, args.baselines)
    (args.out or args.results_dir / "FALLBACK.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
