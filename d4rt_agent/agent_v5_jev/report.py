"""Score a jev-flow run and put it next to the runs it must be compared with.

    python -m d4rt_agent.agent_v5_jev.report <results-dir> [--v5-run <AgentV5 run dir>] [--out report.md]

Paired, question for question: AgentV5 with Qwen3-VL-8B (same phases, same tools, the model that
this arm replaces), then the decomposition study's baselines (Qwen frames, Qwen letters, the
calibrated decision flow). Also the caption-only ablation (the same final question without the
measurements), per-kind deltas against V5, and the API usage.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from d4rt_agent.agent_v5.report import (BASELINE_COLUMNS, DEFAULT_BASELINES, accuracy, bootstrap_ci, breakdown,
                                        load_answers)

DEFAULT_V5_RUN = Path("/cluster/work/igp_psr/spanwar/shak-spatial-agent/d4rt_agent/results/"
                      "agent_v5_eval2800_point4d_20261005T183754Z")


def ablation_rows(results_dir: Path) -> dict[str, dict[str, Any]]:
    """qid -> {"caption_only": bool correct} from the answer files that carry the ablation."""

    out: dict[str, dict[str, Any]] = {}
    for path in sorted((Path(results_dir) / "answers").glob("*.json")):
        a = json.loads(path.read_text())
        if a.get("caption_only_letter") is not None:
            out[a["question_id"]] = {"caption_only": a["caption_only_letter"] == a["reference"]["answer"]}
    return out


def diff_line(label: str, rows: Sequence[Mapping[str, Any]], other: Mapping[str, bool]) -> str:
    pairs = [dict(r, base=float(other[r["qid"]]), d=float(r["correct"]) - float(other[r["qid"]]))
             for r in rows if r["qid"] in other]
    if not pairs:
        return f"| {label} | n/a | |"
    base, _ = accuracy(pairs, "base")
    lo, hi = bootstrap_ci(pairs, "d")
    mine, _ = accuracy(pairs)
    return f"| {label} | {base:.1%} | {mine - base:+.1%} ({lo:+.1%}..{hi:+.1%}) |"


def per_kind_delta(rows: Sequence[Mapping[str, Any]], v5: Mapping[str, bool]) -> list[str]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["qid"] in v5:
            groups[str(r["kind"])].append(r)
    lines = ["| kind (chosen by the jev agent) | n | jev-flow | V5 + Qwen | delta |", "|---|---|---|---|---|"]
    for kind, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        mine, n = accuracy(rs)
        theirs = sum(float(v5[r["qid"]]) for r in rs) / n
        lines.append(f"| {kind} | {n} | {mine:.1%} | {theirs:.1%} | {mine - theirs:+.1%} |")
    return lines


def usage(results_dir: Path) -> str:
    path = Path(results_dir) / "decisions" / "usage.jsonl"
    if not path.exists():
        return "no API usage recorded"
    calls, tokens = 0, 0
    for line in path.read_text().splitlines():
        calls += 1
        tokens += int(json.loads(line).get("input_tokens") or 0)
    return f"{calls} API calls, {tokens:,} input tokens (about ${tokens * 0.10 / 1e6:.2f} at $0.10 per 1M)"


def report(results_dir: Path, v5_run: Path | None, baselines_path: Path | None) -> str:
    rows = load_answers(results_dir)
    if not rows:
        return "no answers yet"
    acc, n = accuracy(rows)
    lo, hi = bootstrap_ci(rows)
    answered = sum(1 for r in rows if r["letter"])
    out = [f"# jev-flow report: {results_dir}", "",
           f"**{acc:.1%}** over {n} questions (clip bootstrap 95% CI {lo:.1%}..{hi:.1%}); "
           f"{answered} answered, {n - answered} unanswered or errored.",
           f"Mean measuring steps {sum(r['steps'] for r in rows) / n:.1f}. {usage(results_dir)}.", ""]
    table = ["| compared with (same questions) | its accuracy | jev-flow minus it (95% CI) |", "|---|---|---|"]
    if v5_run and (Path(v5_run) / "answers").exists():
        v5 = {r["qid"]: bool(r["correct"]) for r in load_answers(v5_run)}
        table.append(diff_line("AgentV5 + Qwen3-VL-8B (same phases and tools)", rows, v5))
    else:
        v5 = {}
    if baselines_path and Path(baselines_path).exists():
        base = {b["qid"]: b for b in json.loads(Path(baselines_path).read_text())}
        for column, label in BASELINE_COLUMNS.items():
            table.append(diff_line(label, rows, {q: b.get(column) == b["gt"] for q, b in base.items()}))
    ablation = ablation_rows(results_dir)
    if ablation:
        table.append(diff_line("same final question, caption only (no measurements)", rows,
                               {q: a["caption_only"] for q, a in ablation.items()}))
    out += ["## paired comparison", ""] + table
    if v5:
        out += ["", "## per kind against AgentV5 + Qwen", ""] + per_kind_delta(rows, v5)
    for field in ("family", "group", "task", "dataset"):
        out += ["", f"## by {field}", ""] + breakdown(rows, field)
    out += ["", "## decomposition", "",
            f"decomposition_ok on {sum(1 for r in rows if r['status'] != 'error')} non-error rows; "
            f"tool use: " + ", ".join(f"{k} {v}" for k, v in _tool_counts(rows).items())]
    return "\n".join(out) + "\n"


def _tool_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for t in r["tools"]:
            counts[t] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--v5-run", type=Path, default=DEFAULT_V5_RUN)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = report(args.results_dir, args.v5_run, args.baselines)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
