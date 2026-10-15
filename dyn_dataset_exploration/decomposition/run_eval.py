"""Score the finished pipeline on the representative evaluation set (``build_eval_catalog``).

Nothing is fitted here. The state applied is whatever ``DYN_CALIBRATED`` names (the evaluation uses
``calibrated.eval.json``: measurement constants from dev clips, the D7 policy refitted on dev rows only).
Every question goes through ``pipeline.solve`` -- rules or Qwen to read it, SAM3 and Point4D to measure
it, the kind's gate and fallback to answer it -- and is compared with the same questions answered by
Qwen alone (frames with reasoning, direct option probabilities, frames plus evidence).

    DYN_CALIBRATED=<state> sbatch scripts/run_cpu_task.slurm -m dyn_dataset_exploration.decomposition.run_eval
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Sequence

from dyn_dataset_exploration.decomposition import pipeline, scene4d

OUT = Path(__file__).resolve().parents[1] / "outputs" / "decomposition_eval"


def collect(out: Path) -> list[dict[str, Any]]:
    ctx = pipeline.Context(out, max_clips=6)
    rows = []
    for qid, entry in ctx.catalog.items():
        try:
            s = pipeline.solve(ctx, qid)
            kind, leaf, final, source = s.program.kind, s.decision.leaf, s.final, s.final_source
            frames, letters, evidence = s.frames_only, (max(s.letters, key=s.letters.get) if s.letters else None), s.evidence_answer
            error = None
        except Exception as exc:  # noqa: BLE001 -- one broken question must not lose the run
            kind, leaf, final, source, frames, letters, evidence, error = "error", "-", None, "none", None, None, None, repr(exc)[:200]
        rows.append({"qid": qid, "clip": entry["clip_slug"], "dataset": entry["dataset"], "task": entry["task"],
                     "read_by": "rules" if entry.get("rule_kind") else "qwen", "kind": kind, "leaf": leaf,
                     "gt": entry["answer"], "pipeline": final, "via": source, "frames": frames, "letters": letters,
                     "evidence": evidence, "error": error})
    return rows


def acc(rows: Sequence[dict[str, Any]], key: str) -> tuple[int, int]:
    """Hits and the number of questions; a question the method could not answer counts as wrong."""

    return sum(r[key] == r["gt"] for r in rows), len(rows)


def fmt(rows: Sequence[dict[str, Any]], key: str) -> str:
    hit, n = acc(rows, key)
    return f"{hit}/{n} ({100 * hit / n:.1f}%)" if n else "-"


def bootstrap(rows: Sequence[dict[str, Any]], keys: Sequence[str], draws: int = 2000, seed: int = 1) -> dict[str, tuple[float, float]]:
    """95% intervals by resampling whole clips (the questions of one clip are not independent)."""

    by_clip: dict[str, list] = defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(r)
    clips = list(by_clip)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(draws):
        pick = [r for c in (rng.choice(clips) for _ in clips) for r in by_clip[c]]
        for k in keys:
            samples[k].append(sum(r[k] == r["gt"] for r in pick) / len(pick))
    lo_hi = {}
    for k, v in samples.items():
        v.sort()
        lo_hi[k] = (v[int(0.025 * draws)], v[int(0.975 * draws)])
    # paired: how much better than Qwen-alone, on the same resampled clips
    return lo_hi


def paired(rows: Sequence[dict[str, Any]], a: str, b: str, draws: int = 2000, seed: int = 2) -> tuple[float, float, float]:
    by_clip: dict[str, list] = defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(r)
    clips = list(by_clip)
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        pick = [r for c in (rng.choice(clips) for _ in clips) for r in by_clip[c]]
        diffs.append(sum(r[a] == r["gt"] for r in pick) / len(pick) - sum(r[b] == r["gt"] for r in pick) / len(pick))
    diffs.sort()
    point = sum(r[a] == r["gt"] for r in rows) / len(rows) - sum(r[b] == r["gt"] for r in rows) / len(rows)
    return point, diffs[int(0.025 * draws)], diffs[int(0.975 * draws)]


def table(rows: Sequence[dict[str, Any]], by: str) -> list[dict[str, Any]]:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[str(r[by])].append(r)
    out = []
    for name, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        out.append({by: name, "n": len(g), "pipeline": fmt(g, "pipeline"), "qwen frames": fmt(g, "frames"),
                    "qwen letters": fmt(g, "letters"), "qwen+evidence": fmt(g, "evidence"),
                    "measurement alone": f"{sum(r['leaf'] == 'A' for r in g)}"})
    return out


def md(rows: Sequence[dict[str, Any]]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--backend", default=scene4d.BACKEND, choices=scene4d.BACKENDS, help="which 3D scene to measure on")
    args = parser.parse_args(argv)
    scene4d.BACKEND = args.backend
    print("state:", pipeline.CALIBRATED, "backend:", scene4d.BACKEND, flush=True)
    rows = collect(args.out)
    (args.out / "eval_rows.json").write_text(json.dumps(rows))
    keys = ["pipeline", "frames", "letters", "evidence"]
    ci = bootstrap(rows, keys)
    n = len(rows)
    print(f"\n{n} questions on {len({r['clip'] for r in rows})} clips; errors: {sum(r['error'] is not None for r in rows)}")
    print("\n| method | accuracy | 95% CI (clips resampled) |\n|---|---|---|")
    for k, label in (("frames", "Qwen3-VL-8B from the frames (reasoning)"), ("letters", "Qwen option probabilities (direct)"),
                     ("evidence", "Qwen with frames + measured evidence"), ("pipeline", "**the decision flow**")):
        print(f"| {label} | {fmt(rows, k)} | {100 * ci[k][0]:.1f}-{100 * ci[k][1]:.1f}% |")
    for other in ("frames", "letters", "evidence"):
        d, lo, hi = paired(rows, "pipeline", other)
        print(f"\npipeline minus Qwen {other}: {100 * d:+.1f} points (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})")
    for by in ("task", "dataset", "read_by", "leaf", "via", "kind"):
        print(f"\n### by {by}\n")
        print(md(table(rows, by)))
    (args.out / "eval_summary.json").write_text(json.dumps({"n": n, "ci": ci}, indent=1))


if __name__ == "__main__":
    main()
