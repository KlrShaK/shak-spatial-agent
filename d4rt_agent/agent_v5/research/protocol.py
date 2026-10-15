"""Answer-blind diagnostic selection and paired clip-cluster evaluation.

Selection reads only the audited question/options corpus, never scoring files.
Comparison is a separate offline operation over recorded predictions/references.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


BENCH_NAMES = {"DSI": "dsi", "Dyn": "dynbench", "VLM4D": "vlm4d"}


def select_diagnostics(rows, n=100, seed=0):
    rng = random.Random(seed)
    selected, review = {}, []
    for benchmark, key in BENCH_NAMES.items():
        strata = defaultdict(list)
        for row in rows:
            if row["benchmark"] == benchmark:
                # Retrieval strata are not capability gold or benchmark-specific routing.
                strata[(row["source"], tuple(sorted(row.get("hits", []))))].append(row)
        for pool in strata.values():
            pool.sort(key=lambda r: r["qid"])
            rng.shuffle(pool)
        keys = sorted(strata)
        rng.shuffle(keys)
        chosen = []
        while len(chosen) < n and any(strata.values()):
            for stratum in keys:
                if strata[stratum] and len(chosen) < n:
                    chosen.append(strata[stratum].pop())
        if len(chosen) != n:
            raise ValueError(f"{benchmark}: only {len(chosen)} eligible questions")
        selected[key] = [r["qid"] for r in chosen]
        review.extend({k: r[k] for k in ("benchmark", "qid", "source", "video", "question", "options", "hits")} |
                      {"reviewed": False, "requirements": []} for r in chosen)
    return {"schema_version": 2, "seed": seed, "per_benchmark": n, "question_ids": selected,
            "selection": "round-robin source and lexical retrieval strata; no answers used", "review": review}


def prediction_rows(directories):
    if isinstance(directories, (str, Path)):
        directories = [directories]
    rows = {}
    benchmarks = set()
    for directory in directories:
        directory = Path(directory)
        config = json.loads((directory / "run_config.json").read_text())
        bench = config["bench"]
        benchmarks.add(bench)
        split = config.get("split", "std") if bench == "dsi" else "all"
        paths = sorted((directory / "answers").glob("*.json"))
        if not paths:
            raise ValueError(f"no predictions in {directory}")
        for path in paths:
            row = json.loads(path.read_text())
            key = f"{bench}:{split}:{row['question_id']}"
            if key in rows:
                raise ValueError(f"duplicate prediction {key}; provide each shard/split only once")
            rows[key] = row
    if len(benchmarks) != 1:
        raise ValueError("compare one benchmark at a time; DSI augmentations may be combined")
    return rows


def score(row):
    if isinstance(row.get("correct"), bool):
        return int(row["correct"])
    ref = row.get("reference", {})
    gold = ref.get("answer", ref.get("gt", ref.get("letter")))
    if gold is None:
        raise ValueError("a recorded reference answer is required for offline scoring")
    valid = {str(gold), *[str(x) for x in ref.get("answer_also", [])]}
    return int(bool(row.get("answered")) and str(row.get("letter")) in valid)


def clip_key(row):
    ref = row.get("reference", {})
    clip = row.get("clip", {})
    value = ref.get("relative_path") or clip.get("relative_path") or ref.get("video") or clip.get("slug")
    if not value:
        raise ValueError("source clip identity is required for clustered comparison")
    # DSI augmentations are views of the same source clip.
    variants = {"std", "hflip", "reverse", "reverse_hflip"}
    return "/".join(part for part in str(value).split("/") if part not in variants)


def paired_compare(baseline, candidate, seed=0, draws=10000):
    import numpy as np
    if set(baseline) != set(candidate):
        raise ValueError("paired comparison requires exactly the same question IDs, including unanswered records")
    if not baseline:
        raise ValueError("no recorded predictions")
    clusters = defaultdict(list)
    for qid in sorted(baseline):
        if clip_key(baseline[qid]) != clip_key(candidate[qid]):
            raise ValueError(f"clip identity mismatch for {qid}")
        clusters[clip_key(baseline[qid])].append((score(baseline[qid]), score(candidate[qid])))
    values = list(clusters.values())
    sizes = np.array([len(v) for v in values])
    changes = np.array([sum(b - a for a, b in v) for v in values])
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(draws):
        sample = rng.integers(0, len(values), len(values))
        bootstrap.append(float(changes[sample].sum() / sizes[sample].sum()))
    lo, hi = np.percentile(bootstrap, [2.5, 97.5])
    return {"questions": len(baseline), "source_clusters": len(clusters),
            "baseline_accuracy": sum(score(r) for r in baseline.values()) / len(baseline),
            "candidate_accuracy": sum(score(r) for r in candidate.values()) / len(candidate),
            "paired_gain": float(changes.sum() / sizes.sum()), "paired_95_ci": [float(lo), float(hi)],
            "positive_interval": bool(lo > 0), "seed": seed, "draws": draws}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--corpus", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    select.add_argument("--per-benchmark", type=int, default=100)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", type=Path, nargs="+", required=True)
    compare.add_argument("--candidate", type=Path, nargs="+", required=True)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        rows = json.loads(args.corpus.read_text())
        payload = select_diagnostics(rows, args.per_benchmark)
        payload["corpus_sha256"] = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    else:
        payload = paired_compare(prediction_rows(args.baseline), prediction_rows(args.candidate))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
