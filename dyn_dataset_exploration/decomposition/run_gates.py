"""Decision node D7, fitted per kind: when may the measurement answer alone, and who answers otherwise?

    python -m dyn_dataset_exploration.decomposition.run_gates [--out <catalog dir>] [--no-write]

Run after ``run_calibration`` (constants fixed) and after the GPU ``evidence`` stage (Qwen's answers
with the measured evidence exist). For each kind, on **dev clips only**, it searches

* the gate -- how well supported (``min_score``) and how far clear of the runner-up (``min_margin``;
  for a base-rate-fused kind, the posterior margin) the measurement's top option must be to answer
  by itself, or "off" (never), and
* the fallback -- when it does not answer, Qwen with the frames and the measured evidence, or Qwen
  from the frames alone,

by the accuracy of the whole policy. A setting replaces the shared default only if it gains at least
``MIN_GAIN`` questions on dev. The test clips are read once, here, to report. The result is written
into ``calibrated.json`` beside the other constants.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dyn_dataset_exploration.decomposition import calibrate as C
from dyn_dataset_exploration.decomposition import interpret, pipeline
from dyn_dataset_exploration.decomposition.run_calibration import MIN_GAIN, table

OFF = {"min_score": 9.0, "min_margin": 9.0, "fused_margin": 9.0}
SCORE_GRID = (0.3, 0.6, 0.85, 1.0)
MARGIN_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FUSED_GRID = (0.0, 0.1, 0.25, 0.4, 0.6, 0.8)
MODES = ("evidence", "frames", "fused")
GAIN_FRACTION = 0.06
FUSION_GRID = [{"a": 1.0, "b": b, "eps": eps} for b in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0) for eps in (0.05, 0.3)]


def fallback_answer(row: Mapping[str, Any], mode: str, fusion: Mapping[str, float] | None = None) -> str | None:
    if mode == "evidence" and row.get("evidence"):
        return row["evidence"]
    if mode == "fused" and fusion is not None and row.get("letters"):
        return interpret.fuse_scores(row["scores"], row["unverifiable"], row["letters"],
                                     fusion["a"], fusion["b"], fusion["eps"])[0]
    return row.get("frames_only")


def answers_alone(row: Mapping[str, Any], gate: Mapping[str, float], fused: bool) -> bool:
    if row["free_leaf"] != "A" or row["free_chosen"] is None:
        return False
    if fused:
        return row["free_margin"] >= gate["fused_margin"]
    return (row["free_top"] or 0.0) >= gate["min_score"] and row["free_margin"] >= gate["min_margin"]


def policy_hits(rows: Sequence[Mapping[str, Any]], gate: Mapping[str, float], fused: bool, mode: str,
                fusion: Mapping[str, float] | None = None) -> int:
    hits = 0
    for r in rows:
        answer = r["free_chosen"] if answers_alone(r, gate, fused) else fallback_answer(r, mode, fusion)
        hits += answer is not None and answer == r["gt"]
    return hits


def candidates(fused: bool) -> list[dict[str, float]]:
    base = dict(interpret.DECISION)
    if fused:
        grid = [{**base, "fused_margin": m} for m in FUSED_GRID]
    else:
        grid = [{**base, "min_score": s, "min_margin": m} for s in SCORE_GRID for m in MARGIN_GRID]
    return grid + [{**base, **OFF}]


def search(rows: Sequence[Mapping[str, Any]], fused: bool) -> dict[str, Any]:
    """``fused`` here is the base-rate posterior gate (a different thing from the soft fusion *mode*)."""

    default = dict(interpret.DECISION)
    baseline = policy_hits(rows, default, fused, "evidence")
    best = (baseline, default, "evidence", None)
    # about 300 settings compete per kind, so the best of them is optimistic; a change has to gain a
    # margin that grows with the sample (2 questions on a small kind, ~6% of it on a large one)
    needed = max(MIN_GAIN, round(GAIN_FRACTION * len(rows)))
    for mode in MODES:
        for fusion in (FUSION_GRID if mode == "fused" else [None]):
            for gate in candidates(fused):
                hits = policy_hits(rows, gate, fused, mode, fusion)
                # a change must earn its keep; ties keep the default gate, then the evidence fallback
                if hits > best[0] and hits - baseline >= needed:
                    best = (hits, gate, mode, fusion)
    hits, gate, mode, fusion = best
    off = gate.get("min_margin", 0) >= 9.0
    return {"gate": None if gate == default else {k: gate[k] for k in ("min_score", "min_margin", "fused_margin")},
            "mode": mode, "fusion": fusion, "off": off, "hits": hits, "baseline_hits": baseline, "n": len(rows),
            "never_alone": policy_hits(rows, {**default, **OFF}, fused, mode, fusion),
            "frames": sum(r["frames_only"] == r["gt"] for r in rows if r["frames_only"]),
            "evidence": sum(r["evidence"] == r["gt"] for r in rows if r["evidence"]),
            "letters": sum(fallback_answer(r, "fused", {"a": 1.0, "b": 0.0, "eps": 1.0}) == r["gt"] for r in rows if r["letters"]),
            "fused_best": max(policy_hits(rows, {**default, **OFF}, fused, "fused", f) for f in FUSION_GRID)}


def offline_table(rows: Sequence[Mapping[str, Any]], policies: Mapping[str, Mapping[str, Any]],
                  fused_fn) -> list[dict[str, Any]]:
    """The per-kind table under the chosen policies, computed from the dumped rows alone."""

    from dyn_dataset_exploration.decomposition.policy_lab import answer

    def pct(hit: int, n: int) -> str:
        return f"{hit}/{n} ({100 * hit / n:.0f}%)" if n else "-"

    out, total = [], defaultdict(int)
    by_kind: dict[str, list] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    for kind, group in sorted(by_kind.items()):
        policy = policies[kind]
        alone = [r for r in group if answers_alone(r, policy["gate"], fused_fn(kind))]
        final = sum(answer(r, policy, fused_fn(kind)) == r["gt"] for r in group)
        frames = sum(r["frames_only"] == r["gt"] for r in group if r["frames_only"])
        evidence = sum(r["evidence"] == r["gt"] for r in group if r["evidence"])
        letters = sum(fallback_answer(r, "fused", {"a": 1.0, "b": 0.0, "eps": 1.0}) == r["gt"] for r in group if r["letters"])
        ok_alone = sum(r["free_chosen"] == r["gt"] for r in alone)
        out.append({"kind": kind, "n": len(group), "pipeline": pct(final, len(group)), "frames-only": pct(frames, len(group)),
                    "qwen letters": pct(letters, len(group)), "evidence": pct(evidence, len(group)),
                    "answers alone": str(len(alone)), "alone acc": pct(ok_alone, len(alone))})
        for k, v in (("n", len(group)), ("final", final), ("frames", frames), ("letters", letters), ("evidence", evidence),
                     ("alone", len(alone)), ("alone_ok", ok_alone)):
            total[k] += v
    out.append({"kind": "**all**", "n": total["n"], "pipeline": pct(total["final"], total["n"]),
                "frames-only": pct(total["frames"], total["n"]), "qwen letters": pct(total["letters"], total["n"]),
                "evidence": pct(total["evidence"], total["n"]), "answers alone": str(total["alone"]),
                "alone acc": f"{total['alone_ok']}/{total['alone']}"})
    return out


def policies_from_state(kinds: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The per-kind policy that calibrated.json (as loaded) makes the pipeline apply."""

    out = {}
    for kind in kinds:
        gate = interpret.gate(kind)
        mode = pipeline.FALLBACK_BY_KIND.get(kind, "evidence")
        out[kind] = {"gate": gate, "mode": mode, "fusion": interpret.FUSION.get(kind) if mode == "fused" else None}
    return out


def score_heldout(out_dir: Path) -> None:
    """Rows for the held-out clips under the fitted state, and the table: nothing here is fitted."""

    path = out_dir / "rows.heldout.json"
    if not path.exists():
        ctx = pipeline.Context(out_dir, max_clips=6)
        qids = C.split_qids(ctx).get("heldout", [])
        pipeline.FALLBACK_BY_KIND.clear()
        interpret.DECISION_BY_KIND.clear()
        interpret.FUSION.clear()
        rows = C.evaluate(ctx, qids, dict(pipeline.TEMPORAL_BY_KIND), trusted=set(C.scene_kinds()))
        path.write_text(json.dumps(rows, default=str))
        pipeline.apply_calibrated()          # the free decisions were made ungated; put the fitted state back
        print("wrote", path, flush=True)
    rows = json.loads(path.read_text())
    fused_fn = lambda k: k in interpret.FUSE_KINDS and bool(interpret.PRIORS.get(k))          # noqa: E731
    policies = policies_from_state(sorted({r["kind"] for r in rows}))
    result = offline_table(rows, policies, fused_fn)
    print(table(result, list(result[0])))
    (out_dir / "heldout.json").write_text(json.dumps({"table": result}, indent=1, default=str))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("dyn_dataset_exploration/outputs/decomposition_dev"))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--dump-rows-only", action="store_true",
                        help="only write rows.dev.json / rows.test.json (the per-question data the search uses)")
    parser.add_argument("--heldout", action="store_true",
                        help="score the fitted pipeline (calibrated.json as it stands) on the held-out clips; fits nothing")
    parser.add_argument("--fit-all", action="store_true",
                        help="fit on dev + test + heldout rows together (the tables are then in-sample; quote the CV from policy_lab --pooled)")
    parser.add_argument("--method", default="partial", help="policy_lab method: partial | per_kind_vs_simple | pooled | ...")
    args = parser.parse_args(argv)
    from dyn_dataset_exploration.decomposition import policy_lab

    if args.heldout:
        return score_heldout(args.out)
    dev_path, test_path = args.out / "rows.dev.json", args.out / "rows.test.json"
    if args.dump_rows_only or not (dev_path.exists() and test_path.exists()):
        # the slow part (it loads every scene once): a bounded cache keeps memory flat
        ctx = pipeline.Context(args.out, max_clips=6)
        split = C.split_qids(ctx)
        temporal = dict(pipeline.TEMPORAL_BY_KIND)
        everything = set(C.scene_kinds())
        pipeline.FALLBACK_BY_KIND.clear()
        interpret.DECISION_BY_KIND.clear()
        interpret.FUSION.clear()
        dev_path.write_text(json.dumps(C.evaluate(ctx, split["dev"], temporal, trusted=everything), default=str))
        test_path.write_text(json.dumps(C.evaluate(ctx, split["test"], temporal, trusted=everything), default=str))
        print("wrote", dev_path, "and", test_path, flush=True)
        if args.dump_rows_only:
            return

    dev, test = json.loads(dev_path.read_text()), json.loads(test_path.read_text())
    held_path = args.out / "rows.heldout.json"
    held = json.loads(held_path.read_text()) if held_path.exists() else []
    fused_fn = lambda k: k in interpret.FUSE_KINDS and bool(interpret.PRIORS.get(k))          # noqa: E731
    policies = policy_lab.fit(args.method, dev + test + held if args.fit_all else dev, fused_fn, GAIN_FRACTION)

    default = dict(interpret.DECISION)
    pipeline.FALLBACK_BY_KIND.clear()
    interpret.DECISION_BY_KIND.clear()
    interpret.FUSION.clear()
    for kind, policy in sorted(policies.items()):
        gate = {k: policy["gate"][k] for k in ("min_score", "min_margin", "fused_margin")}
        if any(gate[k] != default[k] for k in gate):
            interpret.DECISION_BY_KIND[kind] = gate
        pipeline.FALLBACK_BY_KIND[kind] = "fused" if policy["mode"] == "fused" else policy["mode"]
        if policy["fusion"]:
            interpret.FUSION[kind] = {k: policy["fusion"][k] for k in ("a", "b", "eps")}
        off = policy["gate"]["min_margin"] >= 9.0
        print(f"[{kind}] gate={'OFF' if off else gate} fallback={policy['mode']} fusion={policy['fusion']}")
    interpret.TRUSTED_KINDS.clear()
    interpret.TRUSTED_KINDS.update(k for k, p in policies.items() if p["gate"]["min_margin"] < 9.0)

    reports = {"dev": offline_table(dev, policies, fused_fn), "test": offline_table(test, policies, fused_fn)}
    if held:
        reports["heldout"] = offline_table(held, policies, fused_fn)
    for name in reports:
        print(name)
        print(table(reports[name], list(reports[name][0])), flush=True)
    (args.out / "gates.json").write_text(json.dumps({"method": args.method, "fit_all": args.fit_all, "gain_fraction": GAIN_FRACTION,
                                                     "tables": reports}, indent=1, default=str))
    if not args.no_write:
        payload = json.loads(pipeline.CALIBRATED.read_text())
        payload["constants"] = pipeline.snapshot()
        payload["provenance"]["gates"] = f"D7 gate, fallback and fusion per kind ({args.method}, fitted on dev by run_gates)"
        pipeline.CALIBRATED.write_text(json.dumps(payload, indent=1))
        print("wrote", pipeline.CALIBRATED)


if __name__ == "__main__":
    main()
