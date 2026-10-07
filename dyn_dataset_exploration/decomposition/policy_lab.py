"""Offline lab for the D7 policy: how should "measurement alone, or who answers otherwise" be chosen?

Works only on ``rows.dev.json`` / ``rows.test.json`` (written by ``run_gates --dump-rows-only``): no scene
is loaded, so an experiment takes seconds. Each method fits a policy on dev rows and is scored three ways:
on the dev rows it was fitted on (optimistic), by 5-fold cross-validation over dev *clips* (the number to
choose a method by), and on the test rows (read once, to report).

    python -m dyn_dataset_exploration.decomposition.policy_lab [--out <catalog dir>]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dyn_dataset_exploration.decomposition import interpret, pipeline
from dyn_dataset_exploration.decomposition.run_gates import (
    FUSION_GRID, MIN_GAIN, OFF, answers_alone, candidates, fallback_answer,
)

DEFAULT = dict(interpret.DECISION)
Policy = dict[str, Any]          # {"gate": {...}, "mode": str, "fusion": {...}|None}


def vote_answer(row: Mapping[str, Any], fusion: Mapping[str, float]) -> str | None:
    """Soft fusion of every cached expert: Qwen's letter log-probabilities, the measurement's support, and
    the frames-only and evidence answers as votes (a hard answer is worth ``cf`` / ``ce`` nats)."""

    if not row.get("letters"):
        return row.get("frames_only")
    total = {}
    for letter, lp in row["letters"].items():
        term = 0.0 if letter in row["unverifiable"] else fusion["b"] * math.log(row["scores"].get(letter, 0.0) + fusion["eps"])
        total[letter] = (fusion["a"] * lp + term + fusion.get("cf", 0.0) * (row.get("frames_only") == letter)
                         + fusion.get("ce", 0.0) * (row.get("evidence") == letter))
    return max(total, key=lambda l: (total[l], -ord(l)))


def answer(row: Mapping[str, Any], policy: Policy, fused_kind: bool) -> str | None:
    if answers_alone(row, policy["gate"], fused_kind):
        return row["free_chosen"]
    if policy["mode"] == "vote":
        return vote_answer(row, policy["fusion"])
    return fallback_answer(row, policy["mode"], policy.get("fusion"))


def hits(rows: Sequence[Mapping[str, Any]], policy: Policy, fused: Callable[[str], bool]) -> int:
    return sum(answer(r, policy, fused(r["kind"])) == r["gt"] for r in rows)


SIMPLE = [
    {"gate": {**DEFAULT, **OFF}, "mode": "frames", "fusion": None},                      # Qwen from the frames
    {"gate": {**DEFAULT, **OFF}, "mode": "evidence", "fusion": None},                    # Qwen + measured evidence
    {"gate": {**DEFAULT, **OFF}, "mode": "fused", "fusion": {"a": 1.0, "b": 0.0, "eps": 1.0}},   # Qwen letters
]
BASELINES = SIMPLE + [{"gate": dict(DEFAULT), "mode": m, "fusion": f} for m, f in
                      (("evidence", None), ("frames", None), ("fused", {"a": 1.0, "b": 2.0, "eps": 0.05}))]


VOTE_GRID = [{"a": 1.0, "b": b, "eps": eps, "cf": cf, "ce": ce}
             for b in (0.0, 0.5, 1.0, 2.0, 4.0) for eps in (0.05, 0.3) for cf in (0.0, 2.0, 5.0) for ce in (0.0, 2.0, 5.0)]
USE_VOTES = False


def space(fused: bool) -> list[Policy]:
    out = []
    if USE_VOTES:
        out += [{"gate": g, "mode": "vote", "fusion": v} for v in VOTE_GRID for g in candidates(fused)]
    for mode in ("evidence", "frames", "fused"):
        for fusion in (FUSION_GRID if mode == "fused" else [None]):
            for gate in candidates(fused):
                out.append({"gate": gate, "mode": mode, "fusion": fusion})
    return out


def best_of(rows, fused_fn, kinds_fused: bool, start: Policy, needed: int, pool: Sequence[Policy]) -> Policy:
    """The best policy of ``pool``, but only if it beats ``start`` by ``needed`` questions."""

    base = hits(rows, start, fused_fn)
    best, best_hits = start, base
    for policy in pool:
        h = hits(rows, policy, fused_fn)
        if h > best_hits and h - base >= needed:
            best, best_hits = policy, h
    return best


def fit(method: str, rows: Sequence[Mapping[str, Any]], fused_fn: Callable[[str], bool], gain: float) -> dict[str, Policy]:
    """A policy per kind."""

    by_kind: dict[str, list] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    if method == "frames":
        return {k: SIMPLE[0] for k in by_kind}
    if method == "letters":
        return {k: SIMPLE[2] for k in by_kind}
    if method == "evidence":
        return {k: SIMPLE[1] for k in by_kind}
    need = lambda n: max(MIN_GAIN, round(gain * n))          # noqa: E731
    if method == "per_kind":                                   # what run_gates did: tuned per kind vs evidence+default
        start = BASELINES[3]
        return {k: best_of(g, fused_fn, False, start, need(len(g)), space(fused_fn(k))) for k, g in by_kind.items()}
    if method == "per_kind_vs_simple":                         # ... but must beat the best untuned choice
        out = {}
        for k, g in by_kind.items():
            start = max(BASELINES, key=lambda p: hits(g, p, fused_fn))
            out[k] = best_of(g, fused_fn, False, start, need(len(g)), space(fused_fn(k)))
        return out
    # pooled: one policy for every kind, fitted on all the rows (gates on the non-fused rule)
    pool_space = space(False)
    start = max(BASELINES, key=lambda p: hits(rows, p, lambda _k: False))
    global_policy = best_of(rows, lambda _k: False, False, start, need(len(rows)) if method != "pooled" else 1, pool_space)
    if method == "pooled":
        return {k: global_policy for k in by_kind}
    if method == "partial":                                    # global policy, per-kind override only on strong evidence
        out = {}
        for k, g in by_kind.items():
            out[k] = best_of(g, fused_fn, False, global_policy, max(need(len(g)), 3), space(fused_fn(k)))
        return out
    raise KeyError(method)


def score(policies: Mapping[str, Policy], rows, fused_fn) -> tuple[int, int]:
    total = 0
    for r in rows:
        p = policies.get(r["kind"], SIMPLE[0])
        total += answer(r, p, fused_fn(r["kind"])) == r["gt"]
    return total, len(rows)


def folds(rows, clip_of: Mapping[str, str], k: int = 5):
    def fold(row):
        return int(hashlib.sha1(clip_of[row["qid"]].encode()).hexdigest()[:6], 16) % k
    for i in range(k):
        yield [r for r in rows if fold(r) != i], [r for r in rows if fold(r) == i]


def pooled_cv(out: Path) -> None:
    """All the labelled rows (dev + test + held-out clips) with clip-grouped 5-fold cross-validation:
    the estimate to quote for a method, since every question is scored by a policy fitted without its clip."""

    rows = []
    for name in ("dev", "test", "heldout"):
        rows += json.loads((out / f"rows.{name}.json").read_text())
    catalog = json.loads((out / "catalog.json").read_text())["questions"]
    clip_of = {e["qid"]: e["clip_slug"] for e in catalog}
    fused_fn = lambda k: k in interpret.FUSE_KINDS and bool(interpret.PRIORS.get(k))          # noqa: E731
    print(f"{len(rows)} questions; 5-fold CV over clips")
    print(f"{'method':<24}{'gain':>5} {'CV accuracy':>13}")
    for method, gain in (("frames", 0), ("letters", 0), ("evidence", 0), ("per_kind", 0.06), ("per_kind_vs_simple", 0.06),
                         ("pooled", 0), ("partial", 0.06), ("partial", 0.12), ("partial", 0.2)):
        hit = n = 0
        per_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for train, held in folds(rows, clip_of):
            fitted = fit(method, train, fused_fn, gain)
            for r in held:
                ok = answer(r, fitted.get(r["kind"], SIMPLE[0]), fused_fn(r["kind"])) == r["gt"]
                hit += ok
                n += 1
                per_kind[r["kind"]][0] += ok
                per_kind[r["kind"]][1] += 1
        print(f"{method:<24}{gain:>5} {hit / n:>12.1%}", flush=True)
        if method in ("partial",) and gain == 0.06:
            print("   per kind:", {k: f"{a}/{b}" for k, (a, b) in sorted(per_kind.items())})


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("dyn_dataset_exploration/outputs/decomposition_dev"))
    parser.add_argument("--pooled", action="store_true",
                        help="cross-validate over dev + test + heldout rows together (5 folds by clip)")
    args = parser.parse_args(argv)
    if args.pooled:
        return pooled_cv(args.out)
    dev = json.loads((args.out / "rows.dev.json").read_text())
    test = json.loads((args.out / "rows.test.json").read_text())
    catalog = json.loads((args.out / "catalog.json").read_text())["questions"]
    clip_of = {e["qid"]: e["clip_slug"] for e in catalog}
    fused_fn = lambda k: k in interpret.FUSE_KINDS and bool(interpret.PRIORS.get(k))          # noqa: E731

    global USE_VOTES
    print(f"{'method':<24}{'gain':>5}  {'dev (fit)':>12} {'dev 5-fold CV':>15} {'test':>12}")
    plan = (("pooled", 0, False), ("partial", 0.06, False), ("pooled", 0, True), ("partial", 0.06, True), ("partial", 0.12, True))
    for method, gain, votes in plan:
        USE_VOTES = votes
        method_label = method + ("+votes" if votes else "")
        fitted = fit(method, dev, fused_fn, gain)
        d, n = score(fitted, dev, fused_fn)
        cv_hit = cv_n = 0
        for train, held in folds(dev, clip_of):
            h, m = score(fit(method, train, fused_fn, gain), held, fused_fn)
            cv_hit += h
            cv_n += m
        t, tn = score(fitted, test, fused_fn)
        print(f"{method_label:<24}{gain:>5}  {d / n:>11.1%} {cv_hit / cv_n:>15.1%} {t / tn:>12.1%}")


if __name__ == "__main__":
    main()
