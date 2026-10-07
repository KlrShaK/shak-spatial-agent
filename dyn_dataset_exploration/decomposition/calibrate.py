"""Calibrate the named constants on dev clips, report on test clips.

Every threshold in ``measure.TUNING`` / ``interpret.DECISION`` and the direction convention is
a convention of the benchmark's question generator, not physics. This module varies them
one group at a time and scores each setting by how often the *top-scored option* is the
ground truth (independent of the leaf gate, so a threshold is judged on its measurement,
not on how cautious the gate is). Constants are chosen on the dev split only; the test split
is read once, at the end, to report.

Splits are by clip, never by question: the questions of one clip share its geometry.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import itertools
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from dyn_dataset_exploration.decomposition import interpret, measure, pipeline, scene4d
from dyn_dataset_exploration.decomposition.program import class_key


@contextmanager
def tuned(**overrides: Any) -> Iterator[None]:
    """Temporarily set constants. Keys: ``t.<name>`` (measure.TUNING), ``d.<name>``
    (interpret.DECISION), ``s.<NAME>`` (scene4d module constant), ``conv`` (direction convention)."""

    saved: list[tuple[Callable[[Any], None], Any]] = []
    for key, value in overrides.items():
        if key.startswith("t."):
            name = key[2:]
            saved.append((lambda v, n=name: measure.TUNING.__setitem__(n, v), measure.TUNING[name]))
            measure.TUNING[name] = value
        elif key.startswith("d."):
            name = key[2:]
            saved.append((lambda v, n=name: interpret.DECISION.__setitem__(n, v), interpret.DECISION[name]))
            interpret.DECISION[name] = value
        elif key.startswith("s."):
            name = key[2:]
            saved.append((lambda v, n=name: setattr(scene4d, n, v), getattr(scene4d, name)))
            setattr(scene4d, name, value)
        elif key == "conv":
            old = dict(interpret.DIRECTION_CONVENTION)
            saved.append((lambda v: (interpret.DIRECTION_CONVENTION.clear(), interpret.DIRECTION_CONVENTION.update(v)), old))
            interpret.DIRECTION_CONVENTION.clear()
            interpret.DIRECTION_CONVENTION.update(value)
        else:
            raise KeyError(key)
    try:
        yield
    finally:
        for restore, value in reversed(saved):
            restore(value)


def evaluate(ctx: pipeline.Context, qids: Sequence[str], temporal: str | Mapping[str, str] = "all",
             trusted: set[str] | None = None) -> list[dict[str, Any]]:
    """One row per question: what was measured, the top option, and whether it is right.

    ``temporal`` is one frame set for every question, or a ``{kind: frame set}`` mapping.
    """

    rows = []
    for qid in qids:
        kind = ctx.catalog[qid]["branch"]
        mode = temporal if isinstance(temporal, str) else temporal.get(kind, "all")
        s = pipeline.solve(ctx, qid, temporal=mode, trusted=trusted if trusted is not None else set(interpret.TRUSTED_KINDS) | set(scene_kinds()))
        d = s.decision
        gt = s.entry["answer"]
        free = _ungated(s)
        rows.append({
            "free_leaf": free.leaf, "free_chosen": free.chosen, "free_margin": free.margin,
            "free_top": next((sc.score for sc in free.scores if sc.letter == free.chosen), None),
            "letters": s.letters, "unverifiable": [sc.letter for sc in d.scores if sc.verdict == "unverifiable"],
            "evidence": s.evidence_answer, "evidence_correct": None if s.evidence_answer is None else s.evidence_answer == gt,
            "final": s.final, "final_correct": None if s.final is None else s.final == gt,
            "scores": {sc.letter: sc.score for sc in d.scores},
            "classes": {o.letter: class_key(o.atoms) for o in s.program.options},
            "gt_class": next((class_key(o.atoms) for o in s.program.options if o.letter == gt), None),
            "duplicates": s.program.duplicates,
            # the top-scored option ignoring any base-rate fusion, so the strategies can be compared
            "raw_chosen": (min(d.scores, key=lambda sc: (-sc.score, sc.letter)).letter
                           if d.scores and any(sc.verdict != "unverifiable" for sc in d.scores) and d.measured.get("available") else None),
            "qid": qid, "kind": s.program.kind, "temporal": mode, "split": s.entry.get("split"), "dataset": s.entry["dataset"],
            "leaf": d.leaf, "chosen": d.chosen, "margin": d.margin, "gt": gt,
            "available": d.chosen is not None, "correct": None if d.chosen is None else d.chosen == gt,
            "frames_only": s.frames_only, "frames_correct": None if s.frames_only is None else s.frames_only == gt,
            "longest": max(s.entry["options"], key=lambda k: len(s.entry["options"][k])) == gt,
            "letter_A": gt == "A",
        })
    return rows


def _ungated(s: "pipeline.Solution") -> interpret.Decision:
    """The decision this question would get with no D7 threshold (every other check still on):
    the gate search then asks what each threshold would have answered."""

    kind = s.program.kind
    saved = interpret.DECISION_BY_KIND.get(kind)
    interpret.DECISION_BY_KIND[kind] = {"min_score": -1.0, "min_margin": -1.0, "fused_margin": -1.0}
    try:
        return interpret.decide(s.program, s.results, {kind})
    finally:
        if saved is None:
            interpret.DECISION_BY_KIND.pop(kind, None)
        else:
            interpret.DECISION_BY_KIND[kind] = saved


def priors_from(rows: Iterable[Mapping[str, Any]]) -> dict[str, Counter]:
    """Per kind, how often each class of claim is the right one (counted on dev only)."""

    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["gt_class"] is not None:
            counts[r["kind"]][r["gt_class"]] += 1
    return counts


def choose(row: Mapping[str, Any], strategy: str, priors: Mapping[str, Counter], eps: float = 0.15) -> str | None:
    """The letter a strategy answers with, or None when it cannot answer.

    ``measure``: the top-scored option. ``prior``: the option whose class is most often right for
    this kind. ``measure*prior``: score (+eps, so a zero score is not a veto) times the class prior.
    Duplicated options are one claim: the earliest letter is used.
    """

    letters = sorted(row["classes"])
    counts = priors.get(row["kind"], Counter())
    total = max(sum(counts.values()), 1)

    def prior(letter: str) -> float:
        return (counts.get(row["classes"][letter], 0) + 0.5) / (total + 0.5 * max(len(counts), 1))

    if strategy == "measure":
        return row["raw_chosen"]
    if strategy == "prior":
        readable = [l for l in letters if row["classes"][l] is not None]
        return max(readable, key=lambda l: (prior(l), -ord(l))) if readable else None
    if strategy == "measure*prior":
        if not row["available"]:
            return None
        readable = [l for l in letters if row["classes"][l] is not None]
        if not readable:
            return None
        return max(readable, key=lambda l: ((row["scores"].get(l, 0.0) + eps) * prior(l), -ord(l)))
    if strategy == "frames":
        return row["frames_only"]
    raise KeyError(strategy)


STRATEGIES = ("measure", "measure*prior", "prior", "frames")


def strategy_table(rows: Sequence[Mapping[str, Any]], priors: Mapping[str, Counter]) -> list[dict[str, Any]]:
    out = []
    for kind, group in sorted(by_kind(rows).items()):
        row = {"kind": kind, "n": len(group)}
        for st in STRATEGIES:
            answered = [(choose(r, st, priors), r["gt"]) for r in group]
            answered = [(a, g) for a, g in answered if a is not None]
            row[st] = f"{sum(a == g for a, g in answered)}/{len(answered)}" if answered else "-"
        out.append(row)
    return out


def scene_kinds() -> list[str]:
    from dyn_dataset_exploration.decomposition.program import KINDS
    return [k.name for k in KINDS]


def acc(rows: Iterable[Mapping[str, Any]], key: str = "correct") -> tuple[int, int]:
    known = [r for r in rows if r.get(key) is not None]
    return sum(bool(r[key]) for r in known), len(known)


def fmt(rows: Iterable[Mapping[str, Any]], key: str = "correct") -> str:
    hit, n = acc(rows, key)
    return f"{hit}/{n} ({100 * hit / n:.0f}%)" if n else "-"


def by_kind(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
    return groups


def report_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for kind, group in sorted(by_kind(rows).items()):
        a = [r for r in group if r["leaf"] == "A"]
        out.append({
            "kind": kind, "n": len(group), "measured": fmt(group), "coverage": f"{sum(r['available'] for r in group)}/{len(group)}",
            "leaf A": f"{len(a)}", "leaf A acc": fmt(a), "frames-only": fmt(group, "frames_correct"),
            "longest option": fmt(group, "longest"), "always A": fmt(group, "letter_A"),
        })
    return out


def sweep(ctx: pipeline.Context, qids: Sequence[str], grid: Mapping[str, Sequence[Any]], *,
          temporal: str = "all", clear_tracks: bool = False, base: Mapping[str, Any] | None = None
          ) -> list[tuple[dict[str, Any], tuple[int, int]]]:
    """Accuracy of the top-scored option for every combination of ``grid`` on ``qids``."""

    results = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        setting = {**(base or {}), **dict(zip(keys, values))}
        if clear_tracks:
            ctx._tracks.clear()
        with tuned(**setting):
            results.append((dict(zip(keys, values)), acc(evaluate(ctx, qids, temporal))))
    return results


def best_of(results: Sequence[tuple[dict[str, Any], tuple[int, int]]], default: Mapping[str, Any]) -> dict[str, Any]:
    """The best setting; ties go to the default, then to the first listed (grids run
    smallest-change-first), so calibration does not wander on noise."""

    def key(item):
        setting, (hit, n) = item
        return (-hit, 0 if dict(setting) == dict(default) else 1)
    return dict(min(results, key=key)[0])


def split_qids(ctx: pipeline.Context, kinds: Iterable[str] | None = None) -> dict[str, list[str]]:
    wanted = set(kinds) if kinds is not None else None
    out: dict[str, list[str]] = {"dev": [], "test": []}
    for qid, e in ctx.catalog.items():
        if wanted is not None and e["branch"] not in wanted:
            continue
        out.setdefault(e.get("split", "dev"), []).append(qid)
    return out
