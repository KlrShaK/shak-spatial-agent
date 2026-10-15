"""Render ``logs/TAXONOMY_REPORT.md`` from the labelled questions (no network, no GPU, a few seconds).

    python -m vlm4d_dataset_exploration.taxonomy_report [--root ...] [--repo ...] [--out ...]

Besides the counts, it scores the paper's own graded runs (``processed_outputs_paper_results``: 22-23
models, judged by o3/o4-mini and hand-checked) per paper type and per kind, and a text-only baseline that
picks the option whose wording is most often the gold answer elsewhere in the benchmark (leave-one-out),
which measures how much of each kind the answer priors alone decide.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence

from d4rt_agent.agent_v5.program import singular
from vlm4d_dataset_exploration import taxonomy as T
from vlm4d_dataset_exploration.vlm4d_data import DEFAULT_REPO, DEFAULT_RESULTS_DIR, DEFAULT_ROOT, labelled_rows, write_json

MODELS = {"Qwen2.5-VL-7B": "Qwen2.5-VL-7B-Instruct_1fps.json", "Qwen2.5-VL-72B": "Qwen2.5-VL-72B-Instruct_1fps.json",
          "GPT-4o": "gpt-4o_32frame.json", "Gemini-2.5-Pro": "gemini-2.5-pro-preview-06-05_1fps.json"}
PAPER_FIG2 = {"translational": 55, "rotational": 19, "counting": 17, "false_positive": 9}   # % (Fig. 2)
PREVIEW = Path("data/preview/train.parquet")


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.1f}" if b else "-"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def prior_correct(rows: Sequence[Mapping[str, Any]]) -> list[bool]:
    """Pick the option whose normalised text is most often gold over the OTHER questions offering it."""

    offered, gold = Counter(), Counter()
    for r in rows:
        for o in r["options"].values():
            offered[T.norm(o)] += 1
        gold[T.norm(r["gold"])] += 1
    out = []
    for r in rows:
        g = T.norm(r["gold"])

        def rate(text: str) -> float:
            t = T.norm(text)
            n = offered[t] - 1
            return (gold[t] - (t == g)) / n if n > 0 else 0.25
        letter = max(r["options"], key=lambda k: rate(r["options"][k]))
        out.append(letter in (r["answer"], *r["answer_also"]))
    return out


def paper_correct(repo: Path, prompt: str = "cot") -> dict[str, dict[tuple[str, int], bool]]:
    """{model: {(split, n): correct}} for every model with both splits graded."""

    from d4rt_agent.agent_v5 import vlm4d_report as VR

    if not (Path(repo) / VR.PAPER).exists():
        return {}
    return {m.rsplit(".", 1)[0]: {k: v["correct"] for k, v in VR.paper_rows(repo, prompt, m).items()}
            for m in VR.paper_models(repo, prompt)}


def absence_noun_mismatch(row: Mapping[str, Any]) -> bool:
    """An absence option that names something the question does not ask about ('no buses' on a kayak question)."""

    q = T.norm(row["question"])
    for o in row["options"].values():
        if not T.is_absent(o):
            continue
        m = re.search(r"^(?:there (?:is|are) no|no such|no)\s+(.+?)(?:\s+(?:there|in (?:the|this) (?:video|scene)|on \w+))?\.?$", T.norm(o))
        noun = singular(m.group(1).split()[-1]) if m else ""
        words = re.findall(r"[a-z'-]+", q)
        if noun and noun not in {*words, *(singular(w) for w in words)}:
            return True
    return False


def by(rows: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Any]) -> dict[Any, list[Mapping[str, Any]]]:
    out: dict[Any, list] = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def section_dataset(rows: Sequence[Mapping[str, Any]], videos: Mapping[str, Any] | None) -> list[str]:
    out = ["## 1. What the dataset ships", ""]
    groups = by(rows, lambda r: (r["view"], r["source"]))
    out += table(["view", "source", "questions", "videos", "questions/video", "distinct question strings"],
                 [(v, s, len(g), len({r['video'] for r in g}), f"{len(g) / len({r['video'] for r in g}):.2f}",
                   len({r['question'] for r in g})) for (v, s), g in sorted(groups.items())])
    real = [r for r in rows if r["split"] == "real"]
    exo_ids = {r["n"] for r in real if r["view"] == "exo"}
    out += [f"Real ids 1..{max(exo_ids)} are exo-centric and {max(exo_ids) + 1}..{max(r['n'] for r in real)} ego-centric, "
            f"which is the split the benchmark's `acc_final_statistics.py` hard-codes (id <= 922).", ""]
    numeric = sum(T.option_form(r["gold"]) == "numeric" for r in rows)
    dup = sum(bool(r["answer_also"]) for r in rows)
    mismatch = [r for r in rows if absence_noun_mismatch(r)]
    out += ["Data quirks an agent has to survive:", "",
            f"- The answer is stored as option TEXT ({numeric} numeric), mapped to a letter by AgentV5's loader; "
            f"{dup} question has two options with the same text (either letter counts).",
            f"- {len(mismatch)} questions offer an absence option naming a different object or a typo "
            f"(e.g. \"{mismatch[0]['question']}\" offers \"{next(o for o in mismatch[0]['options'].values() if T.is_absent(o))}\"): "
            "the existence check must read the option's noun, not assume it is the question's." if mismatch else "",
            "- Typos in questions and options are frequent (\"bicyclec\", \"mototrbikes\", \"clckswise\", \"plances\"); "
            "slot validation against the question's words (AgentV5 D3) must tolerate them.", ""]
    if videos:
        out += ["### Videos", ""]
        rows_v = []
        for src, vs in sorted(by(list(videos.values()), lambda v: v["source"]).items()):
            secs = [v["seconds"] for v in vs if v["seconds"]]
            res = Counter(f"{v['width']}x{v['height']}" for v in vs).most_common(2)
            rows_v.append((src, len(vs), f"{statistics.median(secs):.1f}", f"{min(secs):.1f}-{max(secs):.1f}",
                           f"{statistics.median(v['fps'] for v in vs):.0f}", sum(v["height"] > v["width"] for v in vs),
                           ", ".join(f"{k} ({n})" for k, n in res)))
        out += table(["source", "videos", "median s", "range s", "median fps", "portrait", "top resolutions"], rows_v)
    return out


def section_paper_type(rows: Sequence[Mapping[str, Any]], root: Path) -> list[str]:
    out = ["## 2. Axis A: the paper's annotation types", "",
           "Recovered by rule (`taxonomy.paper_type`); the QA files do not ship them. False positive is read from the "
           "gold answer (object or event absent), the others from the question text in the order counting > "
           "rotational > translational > action.", ""]
    real = [r for r in rows if r["split"] == "real"]
    hdr = ["paper type", "real", "real %", "paper Fig. 2 %", "synthetic", "all", "of which name their frame"]
    syn = [r for r in rows if r["split"] == "synthetic"]
    out += table(hdr, [(t, sum(r["paper_type"] == t for r in real), pct(sum(r["paper_type"] == t for r in real), len(real)),
                        PAPER_FIG2.get(t, "- (Fig. A only)"), sum(r["paper_type"] == t for r in syn),
                        sum(r["paper_type"] == t for r in rows),
                        sum(r["paper_type"] == t and r["perspective"] for r in rows)) for t in T.PAPER_TYPES])
    fp_real = [r for r in real if r["paper_type"] == "false_positive"]
    offered = [r for r in real if r["absence_option"]]
    out += [f"Translational, rotational and counting land within 2 points of the paper. False positives do not: "
            f"{len(fp_real)} real questions ({pct(len(fp_real), len(real))}%) have a gold answer that denies the object "
            f"or event, against the paper's 9%; {len(offered)} real questions *offer* an absence option. The synthetic "
            f"split matches the benchmark exactly ({sum(r['paper_type'] == 'false_positive' for r in syn)} = its `is_fp`).", ""]
    try:
        import pandas as pd

        prev = pd.read_parquet(Path(root) / PREVIEW)
        out += ["Check against the 9 preview rows the dataset labels itself (`reasoning_focus`):", ""]
        out += table(["question", "reasoning_focus", "rule"],
                     [(q, f, T.paper_type(q, a)) for q, f, a in zip(prev["question"], prev["reasoning_focus"], prev["answer"])])
    except Exception as exc:          # pandas/pyarrow missing or the preview not downloaded
        out += [f"(preview check skipped: {type(exc).__name__})", ""]
    return out


def section_kinds(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    out = ["## 3. Axis B: AgentV5 family and kind", "",
           "Kinds marked **new** do not exist in `d4rt_agent/agent_v5/kinds.py`. `wearer` = the question is about "
           "the Ego4D camera wearer (\"which way does the person turn\"), so the camera is the subject.", ""]
    views = ("exo", "ego", "synthetic")
    rows_k = []
    for k in T.KINDS:
        g = [r for r in rows if r["kind"] == k.name]
        if not g:
            continue
        rows_k.append((k.family, f"`{k.name}`" + (" **new**" if k.new else ""), len(g), pct(len(g), len(rows)),
                       *[sum(r["view"] == v for r in g) for v in views], sum(r["ego_wearer"] for r in g),
                       sum(r["absence_option"] for r in g)))
    out += table(["family", "kind", "n", "%", *views, "wearer", "offers absence"], rows_k)
    new = sum(r["new_kind"] for r in rows)
    out += [f"**{new} of {len(rows)} questions ({pct(new, len(rows))}%) need a kind AgentV5 does not have.**", ""]
    out += ["### Paper type x kind", ""]
    kinds = [k.name for k in T.KINDS if any(r["kind"] == k.name for r in rows)]
    out += table(["kind", *T.PAPER_TYPES], [(k, *[sum(r["kind"] == k and r["paper_type"] == t for r in rows) or "" for t in T.PAPER_TYPES])
                                           for k in kinds])
    out += ["### Frame of reference x kind", ""]
    frames = sorted({r["frame"] for r in rows})
    out += table(["kind", *frames], [(k, *[sum(r["kind"] == k and r["frame"] == f for r in rows) or "" for f in frames]) for k in kinds])
    windows = Counter(r["window"] for r in rows)
    out += [f"Time window named by the question: {dict(windows)}. VLM4D clips are short and centred on one action, so "
            "most questions are about the whole clip.", ""]
    return out


def section_programs(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    out = ["## 4. Sub-task programs", "",
           "Every question becomes an ordered program: per-question steps first (`resolve_actor` for the ego wearer, "
           "`exists` when an absence option is offered, `window`), then the kind's own steps, then the answer turn.", ""]
    rows_p = []
    for k in T.KINDS:
        g = [r for r in rows if r["kind"] == k.name]
        if g:
            rows_p.append((f"`{k.name}`", len(g), " -> ".join(T.SUBTASKS[k.name]), pct(sum(r["absence_option"] for r in g), len(g)),
                           pct(sum(r["window"] != "whole" for r in g), len(g))))
    out += table(["kind", "n", "kind's steps", "% with exists()", "% with window()"], rows_p)
    example = next(r for r in rows if r["ego_wearer"] and r["absence_option"]) if any(r["ego_wearer"] and r["absence_option"] for r in rows) \
        else next(r for r in rows if r["absence_option"])
    out += [f"Example ({example['qid']}): \"{example['question']}\" -> " + " -> ".join(example["program"]), ""]
    return out


def section_options(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    out = ["## 5. Answer options: forms and priors", "",
           "Each option is given one coarse form. `P(gold | offered)` is how often an option of that form is the "
           "answer when it is offered; 25% would be no prior.", ""]
    rows_o = []
    for split in ("real", "synthetic"):
        g = [r for r in rows if r["split"] == split]
        for form in ("absent", "static", "unsure", "lateral", "depth", "vertical", "rotation", "own", "numeric", "other"):
            offered = [r for r in g if form in r["option_forms"]]
            won = sum(T.option_form(r["gold"]) == form for r in offered)
            if offered:
                rows_o.append((split, form, len(offered), won, pct(won, len(offered))))
    out += table(["split", "form", "questions offering it", "gold", "P(gold | offered) %"], rows_o)
    return out


def section_models(rows: Sequence[Mapping[str, Any]], repo: Path) -> list[str]:
    models = paper_correct(repo)
    out = ["## 6. How the paper's models do, per type and per kind (chain-of-thought runs)", ""]
    if not models:
        return out + ["(the paper's repo is not on this machine)", ""]
    prior = dict(zip((r["qid"] for r in rows), prior_correct(rows)))
    key = {r["qid"]: (r["split"], r["n"]) for r in rows}
    pick = {name: models[f.rsplit(".", 1)[0]] for name, f in MODELS.items() if f.rsplit(".", 1)[0] in models}

    def accs(g: Sequence[Mapping[str, Any]]) -> list[str]:
        each = [sum(m[key[r["qid"]]] for r in g) / len(g) for m in models.values()]
        return [pct(sum(prior[r["qid"]] for r in g), len(g)), *[pct(sum(m[key[r["qid"]]] for r in g), len(g)) for m in pick.values()],
                f"{100 * statistics.mean(each):.1f}", f"{100 * max(each):.1f}"]

    hdr = ["n", "text prior", *pick, f"mean of {len(models)}", f"best of {len(models)}"]
    out += table(["paper type", *hdr], [(t, len(g), *accs(g)) for t in T.PAPER_TYPES if (g := [r for r in rows if r["paper_type"] == t])]
                 + [("**all**", len(rows), *accs(rows))])
    out += table(["kind", *hdr], [(f"`{k.name}`", len(g), *accs(g)) for k in T.KINDS
                                  if len(g := [r for r in rows if r["kind"] == k.name]) >= 5])
    out += ["Kinds with fewer than 5 questions are left out. \"text prior\" never sees the video.", ""]
    return out


def section_other(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    other = [r for r in rows if r["kind"] == "other"]
    return ["## 7. Residual: no rule reads these", ""] + [f"- {r['qid']}: {r['question']}" for r in other] + [""]


def report(rows: Sequence[Mapping[str, Any]], root: Path, repo: Path, videos: Mapping[str, Any] | None) -> str:
    out = ["# VLM4D question taxonomy", "",
           f"Generated by `python -m vlm4d_dataset_exploration.taxonomy_report` over all {len(rows)} questions "
           "(deterministic rules, no sampling, no LLM). Hand-written analysis: `VLM4D_DATASET.md`, "
           "`BUCKET_REQUIREMENTS_V5.md`.", ""]
    for part in (section_dataset(rows, videos), section_paper_type(rows, root), section_kinds(rows), section_programs(rows),
                 section_options(rows), section_models(rows, repo), section_other(rows)):
        out += part
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR)
    args = p.parse_args(argv)
    rows = labelled_rows(args.root)
    videos_path = args.out / "videos.json"
    videos = json.loads(videos_path.read_text()) if videos_path.exists() else None
    write_json(args.out / "manifest.json", {"source": str(args.root), "questions": rows})
    (args.out / "TAXONOMY_REPORT.md").write_text(report(rows, args.root, args.repo, videos), encoding="utf-8")
    print(f"{len(rows)} questions -> {args.out / 'TAXONOMY_REPORT.md'}")


if __name__ == "__main__":
    main()
