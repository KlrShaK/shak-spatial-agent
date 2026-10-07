"""How much of Dyn-Bench does the decision graph reach? A census of all 7000 questions by kind.

The calibration catalog holds only rule-parsable questions on a sample of clips, so it cannot say
what share of the benchmark the graph covers. This does, in three steps::

    python -m dyn_dataset_exploration.decomposition.census rules    # CPU (parquet): rule-parse every row
    python -m dyn_dataset_exploration.decomposition.census qwen     # GPU: Qwen names the kind of every
                                                                    #   row the rules did not read (text only)
    python -m dyn_dataset_exploration.decomposition.census report   # CPU: the tables

``qwen`` also re-reads a random sample of rule-parsed rows, so the two readers can be compared.
Nothing here measures anything: a kind is what S1 would decide the question needs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

from dyn_dataset_exploration.decomposition.program import KIND_BY_NAME, KINDS, rule_slots

OUT = Path(__file__).resolve().parents[1] / "outputs" / "decomposition_census"
AGREEMENT_SAMPLE = 400
SEED = 20260929


def stage_rules(out: Path) -> None:
    from dyn_dataset_exploration.dyn_bench_data import DEFAULT_DYN_ROOT, load_rows, qa_parquet

    rows = load_rows(qa_parquet(DEFAULT_DYN_ROOT))
    records = []
    for row in rows:
        found = rule_slots(row.question)
        records.append({
            "qid": row.question_id, "dataset": row.dataset, "task": row.task, "question": row.question,
            "options": row.options, "rule_kind": None if found is None else found[0],
        })
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(records))
    print(f"{len(records)} rows; rules read {sum(r['rule_kind'] is not None for r in records)}")


def _todo(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    residual = [r for r in records if r["rule_kind"] is None]
    rng = random.Random(SEED)
    sample = rng.sample([r for r in records if r["rule_kind"] is not None], AGREEMENT_SAMPLE)
    return residual + sample


def stage_qwen(out: Path, model: str, shard: str) -> None:
    from dyn_dataset_exploration.decomposition.prep_qwen_sam3 import PROMPTS, Asker, _options_block
    from d4rt_agent.simple_v2 import OfflineQwen

    records = json.loads((out / "rows.json").read_text())
    todo = sorted(_todo(records), key=lambda r: r["qid"])
    index, count = (int(v) for v in shard.split("/"))
    todo = todo[index::count]
    path = out / f"kinds.{index}of{count}.jsonl"
    done = {json.loads(line)["qid"] for line in path.read_text().splitlines()} if path.exists() else set()
    kinds = [k for k in KINDS if k.iteration <= 2]
    kinds_text = "\n".join(f"- `{k.name}`: {k.description}" for k in kinds) + "\n- `other`: none of the above"
    prompt = (PROMPTS / "decompose_kind.md").read_text().replace("{kinds}", kinds_text)
    # a census names 4000+ kinds: the answer alone, without the reasoning sentence (a few tokens, not a paragraph)
    prompt = prompt.replace("Reply with one or two sentences of reasoning, then exactly one JSON object and nothing after it:",
                            "Reply with exactly one JSON object and nothing else:")
    names = {k.name for k in kinds}
    asker = Asker(OfflineQwen(model, max_new_tokens=60))

    def check(reply: dict[str, Any]) -> str:
        kind = str(reply["kind"]).strip().strip("`")
        if kind != "other" and kind not in names:
            raise ValueError(f"{kind!r} is not one of the listed kinds")
        return kind

    started = time.time()
    with path.open("a") as sink:
        for n, r in enumerate(todo):
            if r["qid"] in done:
                continue
            stem = f"Question: {r['question']}\nOptions:\n{_options_block(r['options'])}\n\n"
            kind = asker.ask([{"type": "text", "text": stem + prompt}], "kind", check, [], "kind", max_new_tokens=60)
            sink.write(json.dumps({"qid": r["qid"], "qwen_kind": kind}) + "\n")
            sink.flush()
            if n % 50 == 0:
                print(f"{n}/{len(todo)} ({time.time() - started:.0f}s)", flush=True)


def _pct(a: int, n: int) -> str:
    return f"{a} ({100 * a / max(n, 1):.0f}%)"


def stage_report(out: Path) -> None:
    records = json.loads((out / "rows.json").read_text())
    qwen = {}
    for file in sorted(out.glob("kinds.*.jsonl")):
        for line in file.read_text().splitlines():
            d = json.loads(line)
            qwen[d["qid"]] = d["qwen_kind"]
    n = len(records)
    final: dict[str, str] = {}
    for r in records:
        if r["rule_kind"] is not None:
            final[r["qid"]] = r["rule_kind"]
        elif r["qid"] in qwen:
            final[r["qid"]] = qwen[r["qid"]] or "unparsed"
        else:
            final[r["qid"]] = "unread"
    print(f"rows {n}; rules read {sum(r['rule_kind'] is not None for r in records)}; "
          f"Qwen read {sum(1 for r in records if r['rule_kind'] is None and r['qid'] in qwen)} of the rest")

    counts = Counter(final.values())
    route = {k.name: k.route for k in KINDS}
    print("\n| kind | route | rows | share | by rules | by Qwen |\n|---|---|---|---|---|---|")
    for kind, c in counts.most_common():
        by_rule = sum(1 for r in records if r["rule_kind"] == kind)
        print(f"| {kind} | {route.get(kind, '-')} | {c} | {100 * c / n:.1f}% | {by_rule} | {c - by_rule} |")
    measured = sum(c for k, c in counts.items() if route.get(k) == "measure")
    print(f"\nrouted to a measurement: {_pct(measured, n)}; answered from the frames (semantic, other or unread): {_pct(n - measured, n)}")

    print("\n| dataset | rows | measured | share |\n|---|---|---|---|")
    per_ds: dict[str, list[str]] = defaultdict(list)
    for r in records:
        per_ds[r["dataset"]].append(final[r["qid"]])
    for ds, kinds in sorted(per_ds.items()):
        m = sum(1 for k in kinds if route.get(k) == "measure")
        print(f"| {ds} | {len(kinds)} | {m} | {100 * m / len(kinds):.0f}% |")

    both = [(r["rule_kind"], qwen[r["qid"]]) for r in records if r["rule_kind"] is not None and r["qid"] in qwen]
    if both:
        agree = sum(a == b for a, b in both)
        print(f"\nrules vs Qwen on {len(both)} rule-parsed rows: agree {_pct(agree, len(both))}")
        disagreements = Counter((a, b) for a, b in both if a != b)
        for (a, b), c in disagreements.most_common(10):
            print(f"   rules {a} / Qwen {b}: {c}")

    others = [r for r in records if final[r["qid"]] == "other"]
    rng = random.Random(SEED)
    print(f"\nexamples of `other` ({len(others)} rows):")
    for r in rng.sample(others, min(12, len(others))):
        print(f"   [{r['task']}] {r['question'][:130]}")
    (out / "census.json").write_text(json.dumps({"counts": counts, "measured": measured, "n": n}, indent=1))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["rules", "qwen", "report"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args(argv)
    {"rules": lambda: stage_rules(args.out), "qwen": lambda: stage_qwen(args.out, args.qwen_model, args.shard),
     "report": lambda: stage_report(args.out)}[args.stage]()


if __name__ == "__main__":
    main()
