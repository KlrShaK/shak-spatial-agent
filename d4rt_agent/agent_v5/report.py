"""Score an AgentV5 run, by kind / task / dataset, with a clip-bootstrap interval, and pair it with
the research pipeline's baselines on the same questions.

    python -m d4rt_agent.agent_v5.report <results-dir> [--baselines <eval_rows.json>] [--out report.md]

``--baselines`` defaults to the decomposition study's ``eval_rows.json`` (Qwen from frames,
Qwen letter log-probs, Qwen with evidence, and the calibrated decision flow, per question), so a
v5 run on the 2800-question eval split is compared question for question.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_BASELINES = Path("/cluster/work/igp_psr/spanwar/shak-spatial-agent/dyn_dataset_exploration/outputs/"
                         "decomposition_eval/eval_rows.json")
BASELINE_COLUMNS = {"frames": "Qwen frames-only", "letters": "Qwen letters", "evidence": "Qwen + evidence",
                    "pipeline": "decision flow (D7)"}


def load_answers(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((Path(results_dir) / "answers").glob("*.json")):
        a = json.loads(path.read_text())
        ref = a.get("reference", {})
        rows.append({"qid": a["question_id"], "letter": a.get("letter"), "gt": ref.get("answer"),
                     "correct": bool(a.get("correct")), "kind": a.get("kind") or "none",
                     "family": a.get("family") or "none", "group": str(ref.get("group")), "task": ref.get("task"),
                     "dataset": ref.get("dataset"), "status": a.get("status"), "steps": a.get("measuring_steps", 0),
                     "tools": [t["tool"] for t in a.get("tools", [])], "peak_tokens": a.get("peak_tokens", 0),
                     "clip": a["question_id"].rsplit("_", 1)[0], "seconds": a.get("wall_seconds")})
    return rows


def bootstrap_ci(rows: Sequence[Mapping[str, Any]], key: str = "correct", n: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """95% interval of the accuracy, resampling clips (questions on one clip are not independent)."""

    by_clip: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(float(r[key]))
    clips = list(by_clip)
    if not clips:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [v for c in (rng.choice(clips) for _ in clips) for v in by_clip[c]]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def accuracy(rows: Iterable[Mapping[str, Any]], key: str = "correct") -> tuple[float, int]:
    rows = list(rows)
    return (sum(float(r[key]) for r in rows) / len(rows) if rows else float("nan")), len(rows)


def breakdown(rows: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r[field])].append(r)
    lines = [f"| {field} | n | accuracy |", "|---|---|---|"]
    for name, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        acc, n = accuracy(rs)
        lines.append(f"| {name} | {n} | {acc:.1%} |")
    return lines


def paired(rows: Sequence[Mapping[str, Any]], baselines: Mapping[str, Mapping[str, Any]]) -> list[str]:
    common = [r for r in rows if r["qid"] in baselines]
    if not common:
        return ["(no question of this run is in the baseline file)"]
    lines = [f"Paired on {len(common)} questions:", "", "| method | accuracy | v5 minus method (95% CI) |",
             "|---|---|---|"]
    v5, _ = accuracy(common)
    lines.append(f"| AgentV5 | {v5:.1%} | |")
    for column, label in BASELINE_COLUMNS.items():
        pairs = [dict(r, base=float(baselines[r["qid"]].get(column) == baselines[r["qid"]]["gt"])) for r in common]
        base, _ = accuracy(pairs, "base")
        diff = [dict(p, d=float(p["correct"]) - p["base"]) for p in pairs]
        lo, hi = bootstrap_ci(diff, "d")
        lines.append(f"| {label} | {base:.1%} | {v5 - base:+.1%} ({lo:+.1%}..{hi:+.1%}) |")
    return lines


def report(results_dir: Path, baselines_path: Path | None) -> str:
    rows = load_answers(results_dir)
    if not rows:
        return "no answers yet"
    acc, n = accuracy(rows)
    lo, hi = bootstrap_ci(rows)
    answered = sum(1 for r in rows if r["letter"])
    out = [f"# AgentV5 report: {results_dir}", "",
           f"**{acc:.1%}** over {n} questions (clip bootstrap 95% CI {lo:.1%}..{hi:.1%}); "
           f"{answered} answered, {n - answered} unanswered or errored.",
           f"Mean measuring steps {sum(r['steps'] for r in rows) / n:.1f}; "
           f"peak context {max(r['peak_tokens'] for r in rows)} tokens.", ""]
    for field in ("kind", "family", "group", "task", "dataset"):
        out += ["", f"## by {field}", ""] + breakdown(rows, field)
    if baselines_path and Path(baselines_path).exists():
        baselines = {b["qid"]: b for b in json.loads(Path(baselines_path).read_text())}
        out += ["", "## against the decomposition study's baselines", ""] + paired(rows, baselines)
    tool_use: dict[str, int] = defaultdict(int)
    for r in rows:
        for t in r["tools"]:
            tool_use[t] += 1
    out += ["", "## tool calls", "", ", ".join(f"{k} {v}" for k, v in sorted(tool_use.items(), key=lambda kv: -kv[1]))]
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = report(args.results_dir, args.baselines)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
