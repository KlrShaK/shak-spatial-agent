"""Write ``results/<qid>.json`` for every question, and ``.evidence.txt`` for those that go to leaf B.

The GPU ``evidence`` stage (``prep_qwen_sam3 --stages evidence``) reads the ``.evidence.txt``
files: Qwen sees the frames and the measured evidence for exactly the questions the measurement
could not settle. CPU only; run it after the constants are calibrated::

    python -m dyn_dataset_exploration.decomposition.write_evidence --out <catalog dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dyn_dataset_exploration.decomposition import interpret, pipeline, scene4d


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("dyn_dataset_exploration/outputs/decomposition"))
    parser.add_argument("--all", action="store_true", help="write evidence for every measured question, not only leaf B")
    parser.add_argument("--kinds", nargs="+", default=None, help="only questions of these kinds (a fix to one signal)")
    parser.add_argument("--split", default=None, help="only questions of this split (dev | test | heldout)")
    parser.add_argument("--backend", default=scene4d.BACKEND, choices=scene4d.BACKENDS, help="which 3D scene to measure on")
    args = parser.parse_args(argv)
    scene4d.BACKEND = args.backend
    ctx = pipeline.Context(args.out, max_clips=6)      # a pass over the catalog: bounded memory
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    counts = {"A": 0, "B": 0, "C": 0}
    for qid in ctx.catalog:
        if args.kinds and ctx.catalog[qid]["branch"] not in args.kinds:
            continue
        if args.split and ctx.catalog[qid].get("split") != args.split:
            continue
        s = pipeline.solve(ctx, qid)
        counts[s.decision.leaf] += 1
        (args.out / "results" / f"{qid}.json").write_text(json.dumps(s.to_dict(), indent=1))
        if s.program.route == "measure" and (s.decision.leaf == "B" or args.all):
            (args.out / "results" / f"{qid}.evidence.txt").write_text(
                interpret.evidence_text(s.program, s.results, s.decision))
    print("leaves:", counts, "->", args.out / "results")


if __name__ == "__main__":
    main()
