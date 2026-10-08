"""Where does a jev-flow run lose? Answer-letter bias, questions that need pixels, tool failures, routing.

    python -m d4rt_agent.agent_v5_jev.analysis <results-dir> [--v5-run <AgentV5 run>] [--out ANALYSIS.md]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from d4rt_agent.agent_v5_jev.report import DEFAULT_V5_RUN

BAD = ("unavailable", "not found", "could not", "failed", "rejected", "no instance", "nothing grounded")


def load(run: Path, only: set[str] | None = None) -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted((Path(run) / "answers").glob("*.json")):
        if only is not None and p.stem not in only:
            continue
        a = json.loads(p.read_text())
        out[a["question_id"]] = a
    return out


def pct(x: int, n: int) -> str:
    return f"{x / n:.1%} ({x}/{n})" if n else "n/a"


def analyse(results_dir: Path, v5_run: Path | None) -> str:
    mine = load(results_dir)
    v5 = load(v5_run, set(mine)) if v5_run and (Path(v5_run) / "answers").exists() else {}
    n = len(mine)
    acc = lambda rs: sum(bool(r.get("correct")) for r in rs)      # noqa: E731
    out = [f"# jev-flow error analysis: {results_dir}", "", f"{n} questions, {pct(acc(mine.values()), n)} correct.", ""]

    gt, pred = Counter(a["reference"]["answer"] for a in mine.values()), Counter(a.get("letter") for a in mine.values())
    out += ["## answer-letter bias", "", "| letter | true | jev-flow |" + (" V5+Qwen |" if v5 else ""), "|---|---|---|" + ("---|" if v5 else "")]
    v5p = Counter(a.get("letter") for a in v5.values())
    for letter in sorted(set(gt) | set(pred) - {None}):
        out.append(f"| {letter} | {gt[letter]} | {pred[letter]} |" + (f" {v5p[letter]} |" if v5 else ""))

    routes = defaultdict(list)
    for a in mine.values():
        routes["vlm_only (nothing to measure)" if (a.get("program") or {}).get("route") == "vlm_only" or not a.get("tools")
               else "measured"].append(a)
    out += ["", "## by route", "", "| route | n | jev-flow | V5+Qwen on the same |", "|---|---|---|---|"]
    for name, rs in routes.items():
        theirs = pct(acc([v5[r["question_id"]] for r in rs if r["question_id"] in v5]), len(rs)) if v5 else "n/a"
        out.append(f"| {name} | {len(rs)} | {pct(acc(rs), len(rs))} | {theirs} |")

    bad = [a for a in mine.values() if any(any(w in t["summary"].lower() for w in BAD) for t in a.get("tools", []))]
    out += ["", "## tool failures", "",
            f"{len(bad)} of {n} questions had a tool report unavailable / not found / failed; accuracy on them "
            f"{pct(acc(bad), len(bad))} vs {pct(acc([a for a in mine.values() if a not in bad]), n - len(bad))} on the rest.",
            f"decomposition not ok: {sum(not a.get('decomposition_ok', True) for a in mine.values())}; "
            f"errors: {sum(a.get('status') == 'error' for a in mine.values())}; "
            f"windows refused: {sum(any('refused' in x for x in a.get('notes', [])) for a in mine.values())}."]

    if v5:
        both = [q for q in mine if q in v5]
        same_kind = sum(mine[q].get("kind") == v5[q].get("kind") for q in both)
        win = [q for q in both if mine[q].get("correct") and not v5[q].get("correct")]
        lose = [q for q in both if v5[q].get("correct") and not mine[q].get("correct")]
        out += ["", "## against AgentV5 + Qwen", "",
                f"same kind chosen: {pct(same_kind, len(both))}. jev-flow right and V5 wrong: {len(win)}; "
                f"V5 right and jev-flow wrong: {len(lose)}.",
                f"accuracy when the kinds agree: jev-flow {pct(acc([mine[q] for q in both if mine[q].get('kind') == v5[q].get('kind')]), same_kind)}; "
                f"V5 {pct(acc([v5[q] for q in both if mine[q].get('kind') == v5[q].get('kind')]), same_kind)}."]

    out += ["", "## ablation: the final question without the measurements", ""]
    ab = [a for a in mine.values() if a.get("caption_only_letter")]
    with_tools = [a for a in ab if a.get("tools")]
    out.append(f"caption only: {pct(sum(a['caption_only_letter'] == a['reference']['answer'] for a in ab), len(ab))}; "
               f"with the measurements: {pct(acc(ab), len(ab))}. On the {len(with_tools)} questions that had measurements: "
               f"caption only {pct(sum(a['caption_only_letter'] == a['reference']['answer'] for a in with_tools), len(with_tools))} vs "
               f"{pct(acc(with_tools), len(with_tools))}.")
    conf = [(a["answer_confidence"] or 0, bool(a["correct"])) for a in mine.values() if a.get("answer_confidence") is not None]
    out += ["", "## does the model's confidence say anything?", "", "| confidence | n | accuracy |", "|---|---|---|"]
    for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        rs = [c for c in conf if lo <= c[0] < hi]
        out.append(f"| {lo:.2f}-{min(hi, 1):.2f} | {len(rs)} | {pct(sum(c[1] for c in rs), len(rs))} |")

    t = [a["timing"] for a in mine.values() if a.get("timing")]
    if t:
        out += ["", "## time", "", f"per question: API {sum(x['api_s'] for x in t) / len(t):.1f} s over "
                f"{sum(x['api_calls'] for x in t) / len(t):.1f} calls, tools (scene load + SAM3 + measurement) "
                f"{sum(x['tool_s'] for x in t) / len(t):.1f} s."]
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--v5-run", type=Path, default=DEFAULT_V5_RUN)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = analyse(args.results_dir, args.v5_run)
    (args.out or args.results_dir / "ANALYSIS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
