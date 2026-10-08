"""Does the grid-choice window agree with the regex window, and where it differs, who is right?

    python -m d4rt_agent.agent_v5_jev.window_check --out <dir> [--workers 8]

Runs over every catalog question with a time hint (the regex found a window, or the stem has time
words). Question text only: no caption, no frames, no options. Prints agreement on the questions the
regex reads, and lists disagreements and the questions the regex ignores, for reading.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from d4rt_agent.agent_v5.program import window_from_question
from d4rt_agent.agent_v5_jev import decisions
from d4rt_agent.agent_v5_jev.windows import TIME_HINT, choose_window, frame_grid, snap_to_explicit

CATALOG = Path("/cluster/work/igp_psr/spanwar/shak-spatial-agent/dyn_dataset_exploration/outputs/"
               "decomposition_eval/catalog.json")
SEED = Path("/cluster/work/igp_psr/spanwar/shak-spatial-agent/d4rt_agent/results/"
            "agent_v5_eval2800_point4d_20261005T183754Z/cache")
HINT = TIME_HINT


def regex_window(question: str) -> tuple[str, float | None, float | None]:
    w = window_from_question(question)
    if w.kind in ("interval",):
        return "interval", w.t0, w.t1
    if w.kind == "instant":
        return "instant", w.t0, None
    return w.kind, None, None


def main(argv: Sequence[str] | None = None) -> None:
    from dyn_dataset_exploration.dyn_bench_data import video_slug

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    cache = args.out / "decisions"
    todo = []
    for q in json.loads(CATALOG.read_text())["questions"]:
        rx = regex_window(q["question"])
        if rx[0] != "whole" or HINT.search(q["question"]):
            clip = json.loads((SEED / "clips" / video_slug(q["dataset"], q["video"]) / "clip.json").read_text())
            if clip.get("fps_gf"):
                todo.append((q, rx, frame_grid(clip["sampled16"], clip["fps_gf"])))

    def one(item: tuple) -> dict[str, Any]:
        q, rx, grid = item
        ask = lambda step, qs: decisions.decide(f"## Question\n{q['question']}", qs, cache)  # noqa: E731
        try:
            kind, t0, t1 = choose_window(ask, grid, "(Read the time span from the question text.)")
        except decisions.Refused:
            kind, t0, t1 = "refused", None, None
        return {"qid": q["qid"], "question": q["question"], "regex": rx, "model": (kind, t0, t1)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, todo))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "windows.json").write_text(json.dumps(rows))

    def near(a: float | None, b: float | None, tol: float = 0.35) -> bool:
        return a is not None and b is not None and abs(a - b) <= tol

    def same(r: dict) -> bool:
        (rk, r0, r1), (mk, m0, m1) = r["regex"], r["model"]
        if rk != mk:
            return False
        return rk == "whole" or (near(r0, m0) and (rk == "instant" or near(r1, m1)))

    read = [r for r in rows if r["regex"][0] != "whole"]
    ignored = [r for r in rows if r["regex"][0] == "whole"]
    print(f"{len(rows)} questions with a time hint; regex reads {len(read)}, ignores {len(ignored)}")
    print(f"model refused on {sum(r['model'][0] == 'refused' for r in rows)} of {len(rows)}")
    print(f"agreement on the regex-read ones: {sum(map(same, read)) / max(len(read), 1):.1%}")
    print("model kind on the questions the regex ignores:", dict(Counter(r["model"][0] for r in ignored)))
    print("\nDISAGREEMENTS on regex-read questions (regex vs model):")
    for r in [r for r in read if not same(r)][:14]:
        print(f"  {r['regex']}  vs  {r['model']}  | {r['question'][:140]}")
    print("\nIGNORED by the regex but the model sees a span (sample):")
    for r in [r for r in ignored if r["model"][0] != "whole"][:14]:
        print(f"  {r['model']} | {r['question'][:140]}")


if __name__ == "__main__":
    main()
