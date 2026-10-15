"""Score an AgentV5 run on VLM4D the way the benchmark's own scripts do, next to the paper's models.

    python -m d4rt_agent.agent_v5.vlm4d_report <results-dir> [--vlm4d-repo /cluster/work/igp_psr/spanwar/VLM4D] [--out vlm4d_report.md]

Reused from the original repo (github ShijieZhou-UCLA/VLM4D, ``acc_final_statistics.py``): the breakdown and the
aggregation. Real questions split by id into exo-centric (id <= 922: davis + youtube-vos) and ego-centric (> 922: ego4d);
synthetic ones into directional and false-positive (the gold answer is "no ..."); the overall figure is weighted over
all 1816 questions. Their stored, already graded outputs (``processed_outputs_paper_results``, 22 models, CoT and direct
prompts) give a comparison on the identical questions.

The paper grades free-text responses with an LLM judge (o4-mini). AgentV5 answers with a letter, so a letter match is
the same decision without the judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from d4rt_agent.agent_v5.report import bootstrap_ci, load_answers

DEFAULT_REPO = Path("/cluster/work/igp_psr/spanwar/VLM4D")
PAPER = "processed_outputs_paper_results"
EXO_MAX_ID = 922
COMPARE = ("Qwen2.5-VL-7B-Instruct", "Qwen2-VL-7B-Instruct", "InternVL2_5-8B", "VideoLLaMA3-7B",
           "Qwen2.5-VL-72B-Instruct", "gpt-4o", "gemini-2.5-pro-preview-06-05")


def is_fp(answer: str) -> bool:
    """A synthetic question whose gold answer is "no ..." (the video has no such object): ``is_fp`` of the repo."""

    a = str(answer).strip().lower()
    return a == "no" or a.startswith("no ")


def breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """The repo's table. ``rows``: split ('real'|'synthetic'), n (id number), gold (answer text), correct."""

    def acc(rs: Sequence[Mapping[str, Any]]) -> float:
        return 100 * sum(bool(r["correct"]) for r in rs) / len(rs) if rs else float("nan")

    real = [r for r in rows if r["split"] == "real"]
    syn = [r for r in rows if r["split"] == "synthetic"]
    return {
        "n": len(rows), "ego": acc([r for r in real if r["n"] > EXO_MAX_ID]),
        "exo": acc([r for r in real if r["n"] <= EXO_MAX_ID]), "real": acc(real),
        "synth_directional": acc([r for r in syn if not is_fp(r["gold"])]),
        "synth_fp": acc([r for r in syn if is_fp(r["gold"])]), "synth": acc(syn), "overall": acc(list(rows)),
    }


def agent_rows(results_dir: Path) -> list[dict[str, Any]]:
    out = []
    for a in load_answers(results_dir):
        parts = a["qid"].split("_")                     # vlm4d_<split>_<video>_<n>
        ref = json.loads((Path(results_dir) / "answers" / f"{a['qid']}.json").read_text())["reference"]
        out.append({"qid": a["qid"], "split": parts[1], "n": int(parts[-1]), "gold": ref["options"][ref["answer"]],
                    "correct": a["correct"], "clip": a["clip"], "group": a["group"]})
    return out


def paper_rows(repo: Path, prompt: str, model_file: str) -> dict[tuple[str, int], dict[str, Any]]:
    """{(split, n): row} of one stored paper run (all 1816 questions)."""

    out = {}
    for split in ("real", "synthetic"):
        for x in json.loads((Path(repo) / PAPER / f"{split}_mc_{prompt}" / model_file).read_text()):
            out[(split, int(x["id"].split("_")[-1]))] = {"split": split, "n": int(x["id"].split("_")[-1]),
                                                           "gold": x["ground_truth_answer"], "correct": bool(x["correct"])}
    return out


def paper_models(repo: Path, prompt: str) -> list[str]:
    real = {p.name for p in (Path(repo) / PAPER / f"real_mc_{prompt}").glob("*.json")}
    syn = {p.name for p in (Path(repo) / PAPER / f"synthetic_mc_{prompt}").glob("*.json")}
    return sorted(real & syn)


def _row(name: str, b: Mapping[str, float]) -> str:
    return (f"| {name} | {b['overall']:.1f} | {b['exo']:.1f} | {b['ego']:.1f} | {b['real']:.1f} | "
            f"{b['synth_directional']:.1f} | {b['synth_fp']:.1f} | {b['synth']:.1f} |")


HEADER = ["| model | overall | real exo | real ego | real | synth directional | synth FP | synth |",
          "|---|---|---|---|---|---|---|---|"]


def report(results_dir: Path, repo: Path) -> str:
    rows = agent_rows(results_dir)
    if not rows:
        return "no answers yet"
    mine = breakdown(rows)
    lo, hi = bootstrap_ci(rows)
    out = [f"# AgentV5 on VLM4D: {results_dir}", "",
           f"**{mine['overall']:.1f}%** over {len(rows)} questions (video bootstrap 95% CI {100 * lo:.1f}..{100 * hi:.1f}); "
           "accuracy in percent, aggregated as in the benchmark's own `acc_final_statistics.py`.", ""]
    if len(rows) != 1816:
        out += [f"(partial run: {len(rows)} of 1816 questions; the paper's models are compared on these questions only)", ""]
    mine_by_key = {(r["split"], r["n"]): r for r in rows}
    for prompt, label in (("cot", "chain-of-thought prompt (the paper's Table 1)"), ("direct-output", "direct-answer prompt")):
        if not (Path(repo) / PAPER).exists():
            break
        table = []
        for model in paper_models(repo, prompt):
            theirs = paper_rows(repo, prompt, model)
            common = [r for k, r in theirs.items() if k in mine_by_key]
            table.append((model.rsplit(".", 1)[0], breakdown(common), theirs))
        table.sort(key=lambda t: -t[1]["overall"])
        out += [f"## AgentV5 against the paper's models, {label}", "", *HEADER, _row("**AgentV5**", mine)]
        out += [_row(name, b) for name, b, _ in table]
        out += ["", "Paired on the same questions (AgentV5 minus the model, accuracy points; 95% CI over videos):", "",
                "| model | model | AgentV5 minus model |", "|---|---|---|"]
        for name, b, theirs in table:
            if not any(name.startswith(c) for c in COMPARE):
                continue
            pairs = [dict(r, d=float(r["correct"]) - float(theirs[(r["split"], r["n"])]["correct"]))
                     for r in rows if (r["split"], r["n"]) in theirs]
            dlo, dhi = bootstrap_ci(pairs, "d")
            out.append(f"| {name} | {b['overall']:.1f} | {100 * sum(p['d'] for p in pairs) / len(pairs):+.1f} "
                       f"({100 * dlo:+.1f}..{100 * dhi:+.1f}) |")
        out.append("")
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r["correct"])
    out += ["## AgentV5 by video source", "", "| source | n | accuracy |", "|---|---|---|"]
    out += [f"| {g} | {len(v)} | {100 * sum(v) / len(v):.1f} |" for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--vlm4d-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = report(args.results_dir, args.vlm4d_repo)
    (args.out or args.results_dir / "vlm4d_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
