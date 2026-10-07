"""Does the Decisions model route a question to the right family and kind from its text alone?

No caption, no frames, no options: the question stem only. Compared with the research pipeline's
rule-read kind (``catalog.questions[].rule_kind``, reporting only) and with the kind AgentV5's Qwen
chose. Cheap (about two short calls per question) and runs from anywhere the API is reachable.

    python -m d4rt_agent.agent_v5_jev.routing_check --out <dir> [--limit N] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from d4rt_agent.agent_v5 import kinds as K
from d4rt_agent.agent_v5_jev import decisions
from d4rt_agent.agent_v5_jev.decisions import choice_question

CATALOG = Path("/cluster/work/igp_psr/spanwar/shak-spatial-agent/dyn_dataset_exploration/outputs/"
               "decomposition_eval/catalog.json")


def route(question: str, cache: Path) -> tuple[str, str]:
    text = f"## Question\n{question}"
    family = decisions.decide(text, [choice_question(
        "family", "Whose state does the question ask about, and relative to what?", list(K.FAMILIES.items()))],
        cache)["family"]["choice"]
    kinds = K.kinds_of(family)
    if len(kinds) == 1:
        return family, kinds[0].name
    kind = decisions.decide(text, [choice_question(
        "kind", "Which quantity decides the question?", [(k.name, k.description) for k in kinds])], cache)["kind"]["choice"]
    return family, kind


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--v5-table", type=Path, default=None, help="json rows with qid, kind (AgentV5's Qwen choice)")
    args = parser.parse_args(argv)
    questions = json.loads(CATALOG.read_text())["questions"][: args.limit]
    v5 = {r["qid"]: r["kind"] for r in json.loads(args.v5_table.read_text())} if args.v5_table else {}
    cache = args.out / "decisions"

    def one(q: dict[str, Any]) -> dict[str, Any]:
        family, kind = route(q["question"], cache)
        return {"qid": q["qid"], "rule_kind": q.get("rule_kind"), "branch": q.get("branch"), "luna_family": family,
                "luna_kind": kind, "v5_kind": v5.get(q["qid"])}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, questions))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "routing.json").write_text(json.dumps(rows))
    read = [r for r in rows if r["rule_kind"]]
    print(f"{len(rows)} questions, {len(read)} read by the rules")
    if read:
        print(f"luna kind == rule kind: {sum(r['luna_kind'] == r['rule_kind'] for r in read) / len(read):.1%}")
        if v5:
            print(f"v5 qwen kind == rule kind: {sum(r['v5_kind'] == r['rule_kind'] for r in read) / len(read):.1%}")
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in read:
        if r["luna_kind"] != r["rule_kind"]:
            confusion[r["rule_kind"]][r["luna_kind"]] += 1
    for kind, c in sorted(confusion.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
        print(f"  rule {kind}: {sum(c.values())} differ -> {dict(c.most_common(3))}")


if __name__ == "__main__":
    main()
