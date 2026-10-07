"""The calibration protocol, end to end (CPU; needs the dev catalog's GPU artifacts).

    python -m dyn_dataset_exploration.decomposition.run_calibration [--kinds camera_path ...]

1. score every question with the constants as they stand (dev and test, per kind), next to
   the baselines: Qwen from the frames alone, the longest option, always-A;
2. per kind, coordinate ascent over the kind's constants **on dev only**: a constant moves
   off its default only if that gains at least ``MIN_GAIN`` questions, so a change has to be
   a signal, not noise;
3. report dev and test again with the chosen constants. Test is read here and nowhere else.

Writes ``calibration.json`` (chosen constants + both tables) beside the catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dyn_dataset_exploration.decomposition import calibrate as C
from dyn_dataset_exploration.decomposition import interpret, measure, pipeline, scene4d

MIN_GAIN = 2
MIN_TRUST_N = 8
FUSE_EPS_GRID = (0.05, 0.15, 0.3, 0.6, 1.0)

# kind -> {constant key: candidate values}; the first value listed is not special, the
# current value is always included so "no change" competes.
GROUPS: dict[str, dict[str, Sequence[Any]]] = {
    "camera_path": {
        "t.cam_pan_deg": [3.0, 4.0, 6.0, 8.0, 12.0], "t.cam_tilt_deg": [3.0, 4.0, 6.0, 8.0, 12.0],
        "t.cam_trans_scales": [0.04, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.5], "t.cam_complex_ratio": [0.5, 0.6, 0.8, 1.0, 9.0],
        "t.cam_rot_peak": [1, 0], "t.cam_trans_peak": [0, 1],
        "t.cam_orbit_ratio": [0.4, 0.6, 0.8, 1.0, 9.0], "t.cam_trans_cap_scales": [1.5, 3.0, 6.0, 1e9],
    },
    "camera_shake": {
        "t.jitter_slight_deg": [0.03, 0.05, 0.07, 0.1, 0.15, 0.25, 0.4], "t.jitter_shaky_deg": [0.1, 0.15, 0.25, 0.4, 0.8, 1.2, 1.8],
        "t.jitter_severe_deg": [0.4, 0.8, 1.5, 2.0, 3.0, 4.5],
    },
    "cam_subject_range": {"t.range_rel_tol": [0.05, 0.10, 0.15, 0.20, 0.28, 0.35, 0.5],
                          "t.range_noise_mult": [1.0, 3.0, 5.0]},
    "pair_relation": {"t.pair_rel_tol": [0.05, 0.10, 0.15, 0.20, 0.28, 0.35, 0.5],
                      "t.range_noise_mult": [1.0, 3.0, 5.0]},
    "subject_direction": {"conv": [{"frame": f, "z_toward_camera": z}
                                   for f in ("camera_rel", "world0", "levelled0", "camera_end", "image")
                                   for z in (False, True)],
                          "t.axis_dominance": [0.25, 0.35, 0.5], "t.move_floor_scales": [0.01, 0.03, 0.06],
                          "d.dir_w_z": [0.0, 0.15, 0.3, 0.6, 1.0]},
    "metric_distance": {"s.SCALE_SOURCE": ["da3", "d4rt", "mean"], "t.metric_scale_mult": [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.6, 2.0, 2.5]},
    "landmark_relation": {"t.region_rel_tol": [0.05, 0.10, 0.15, 0.25, 0.40, 0.60]},
    "camera_translation": {"t.cam_dir_floor_scales": [0.02, 0.05, 0.1, 0.2, 0.4], "t.cam_dir_dominance": [0.25, 0.35, 0.5]},
    "metric_speed": {"s.SCALE_SOURCE": ["da3", "d4rt", "mean"], "t.speed_mode": ["net", "path"], "t.speed_noise_mult": [0.0, 1.0, 2.0, 3.0],
                     "t.speed_scale_mult": [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.6, 2.0, 2.5]},
    "event_order": {"t.onset_pan_deg": [2.0, 3.0, 4.0, 6.0, 8.0], "t.tie_frames": [0.0, 1.0, 2.0]},
    "instance_count": {"s.MIN_LIFESPAN_S": [0.2, 0.5, 1.0], "s.GAP_MERGE_S": [1.0, 2.0, 4.0, 6.0],
                       "s.JUMP_MERGE_SCALES": [0.2, 0.35, 0.6]},
    "camera_zoom": {"t.zoom_rel_tol": [0.03, 0.05, 0.08, 0.10, 0.15, 0.25, 0.35, 0.5, 0.8]},
    "apparent_size": {"t.size_rel_tol": [0.1, 0.2, 0.3, 0.45, 0.7]},
    "subject_visibility": {"t.vis_out_frac": [0.2, 0.35, 0.5], "t.vis_partial_border": [0.3, 0.5, 0.8],
                           "t.vis_partial_area": [0.3, 0.5, 0.7]},
    "cam_subject_elevation": {"t.elev_eye_deg": [5.0, 8.0, 10.0, 15.0, 20.0], "t.elev_change_deg": [8.0, 12.0, 20.0]},
    "camera_rhythm": {"t.rhythm_trans_floor_per_s": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4], "t.rhythm_rot_floor_deg_s": [0.5, 1.5, 3.0, 6.0, 10.0, 16.0],
                      "t.rhythm_burst_ratio": [1.5, 2.2, 3.5], "t.rhythm_cv": [0.4, 0.6, 0.9]},
    "scene_revisit": {"t.revisit_same_frac": [0.3, 0.45, 0.6, 0.75, 0.85, 0.92], "t.revisit_diff_frac": [0.05, 0.1, 0.15, 0.25, 0.35, 0.5]},
    "subject_motion_profile": {"t.speed_floor_per_s": [0.005, 0.01, 0.02, 0.05], "t.motion_continuous_frac": [0.6, 0.75, 0.9]},
}
# keys that change what an entity track is, so cached tracks must be dropped between values
TRACK_KEYS = ("s.MIN_LIFESPAN_S", "s.GAP_MERGE_S", "s.JUMP_MERGE_SCALES")


def current(key: str) -> Any:
    if key.startswith("t."):
        return measure.TUNING[key[2:]]
    if key.startswith("d."):
        return interpret.DECISION[key[2:]]
    if key.startswith("s."):
        return getattr(scene4d, key[2:])
    return dict(interpret.DIRECTION_CONVENTION)


def default(key: str) -> Any:
    """The uncalibrated value of one calibration key."""

    D = pipeline.DEFAULTS
    if key.startswith("t."):
        return D["t"][key[2:]]
    if key.startswith("d."):
        return D["d"][key[2:]]
    if key.startswith("s."):
        return D["s"][key[2:]]
    return dict(D["conv"])


def apply(setting: Mapping[str, Any]) -> None:
    for key, value in setting.items():
        if key.startswith("t."):
            measure.TUNING[key[2:]] = value
        elif key.startswith("d."):
            interpret.DECISION[key[2:]] = value
        elif key.startswith("s."):
            setattr(scene4d, key[2:], value)
        else:
            interpret.DIRECTION_CONVENTION.clear()
            interpret.DIRECTION_CONVENTION.update(value)


def ascend(ctx: pipeline.Context, qids: Sequence[str], grid: Mapping[str, Sequence[Any]], rounds: int = 2,
           temporal: str = "all") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Coordinate ascent on ``qids``; returns the chosen constants and the trace."""

    chosen = {k: current(k) for k in grid}
    trace = []
    for r in range(rounds):
        moved = False
        for key, values in grid.items():
            options = list(values)
            if chosen[key] not in options:
                options.insert(0, chosen[key])
            scored = []
            for v in options:
                if key in TRACK_KEYS:
                    ctx._tracks.clear()
                with C.tuned(**{**chosen, key: v}):
                    scored.append((v, C.acc(C.evaluate(ctx, qids, temporal))))
            base = next(hit for v, (hit, n) in scored if v == chosen[key])
            best_v, (best_hit, n) = max(scored, key=lambda item: (item[1][0], item[0] == chosen[key]))
            trace.append({"round": r, "key": key, "current": chosen[key], "base_hits": base, "best": best_v,
                          "best_hits": best_hit, "n": n, "all": [(v, h) for v, (h, _) in scored]})
            if best_hit - base >= MIN_GAIN and best_v != chosen[key]:
                chosen[key] = best_v
                moved = True
        if not moved:
            break
    return chosen, trace


def table(rows: Sequence[dict[str, Any]], cols: Sequence[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("dyn_dataset_exploration/outputs/decomposition_dev"))
    parser.add_argument("--kinds", nargs="+", default=list(GROUPS))
    parser.add_argument("--no-write", action="store_true", help="do not update calibrated.json")
    args = parser.parse_args(argv)

    # Calibrate from the uncalibrated defaults, not from the last run. A run over only some kinds
    # keeps what is already written for the others and starts just these from their defaults.
    partial = set(args.kinds) != set(GROUPS)
    pipeline.reset()
    if partial:
        pipeline.apply_calibrated()
        for kind in args.kinds:
            apply({key: default(key) for key in GROUPS[kind]})
    ctx = pipeline.Context(args.out)
    split = C.split_qids(ctx)
    by_kind: dict[str, dict[str, list[str]]] = {k: {"dev": [], "test": []} for k in args.kinds}
    for name, qids in split.items():
        if name not in ("dev", "test"):          # the held-out clips are never used to fit anything
            continue
        for qid in qids:
            kind = ctx.catalog[qid]["branch"]
            if kind in by_kind:
                by_kind[kind][name].append(qid)

    before = {"dev": C.evaluate(ctx, [q for k in args.kinds for q in by_kind[k]["dev"]], "all")}
    print("== defaults (dev only; test is read once, after the constants are fixed) ==")
    print(table(C.report_table(before["dev"]), list(C.report_table(before["dev"])[0])))

    chosen_all: dict[str, Any] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    for kind in args.kinds:
        dev = by_kind[kind]["dev"]
        if not dev:
            continue
        candidates = {}
        for mode in ("all", "sampled16"):
            chosen_m, trace_m = ascend(ctx, dev, GROUPS[kind], temporal=mode)
            with C.tuned(**chosen_m):
                hits = C.acc(C.evaluate(ctx, dev, mode))[0]
            candidates[mode] = (hits, chosen_m, trace_m)
        best_mode = max(candidates, key=lambda m_: (candidates[m_][0], m_ == "all"))
        print(f"\n[{kind}] frame set on dev: " + ", ".join(f"{m_}={candidates[m_][0]}" for m_ in candidates) + f" -> {best_mode}")
        _, chosen, trace = candidates[best_mode]
        pipeline.TEMPORAL_BY_KIND[kind] = best_mode
        traces[kind] = trace
        chosen_all[kind] = chosen
        apply(chosen)
        print(f"[{kind}] dev n={len(dev)} chosen: {chosen}")
        for t in trace:
            if t["best"] != t["current"] or t["best_hits"] > t["base_hits"]:
                print(f"   {t['key']}: {t['current']} -> {t['best']}  ({t['base_hits']} -> {t['best_hits']} of {t['n']})")

    # ---- the base-rate branch -----------------------------------------------------------
    dev_ids = [q for k in args.kinds for q in by_kind[k]["dev"]]
    test_ids = [q for k in args.kinds for q in by_kind[k]["test"]]
    temporal = dict(pipeline.TEMPORAL_BY_KIND)
    measured_dev = C.evaluate(ctx, dev_ids, temporal)
    priors = C.priors_from(measured_dev)
    interpret.PRIORS.update({k: dict(v) for k, v in priors.items()})
    interpret.FUSE_KINDS.difference_update(args.kinds)
    fuse_eps: dict[str, float] = {}
    for kind, group in C.by_kind(measured_dev).items():
        m_hit, m_n = C.acc(group)
        best_eps, best_hit = None, -1
        for eps in FUSE_EPS_GRID:
            hit = sum(1 for r in group if (c := C.choose(r, "measure*prior", priors, eps)) is not None and c == r["gt"])
            if hit > best_hit:
                best_eps, best_hit = eps, hit
        prior_hit = sum(1 for r in group if (c := C.choose(r, "prior", priors)) is not None and c == r["gt"])
        fuse = best_hit - m_hit >= MIN_GAIN
        if fuse:
            interpret.FUSE_KINDS.add(kind)
            fuse_eps[kind] = best_eps
        print(f"   base rate {kind}: measure {m_hit}/{m_n}, measure*prior {best_hit} (eps {best_eps}), prior alone {prior_hit}"
              f" -> {'FUSED' if fuse else 'measurement alone'}")
    for kind in args.kinds:
        interpret.FUSE_EPS.pop(kind, None)
    interpret.FUSE_EPS.update(fuse_eps)

    # ---- which kinds may answer deterministically -----------------------------------------
    # On dev, the kind (as it will actually run, fused or not) must be at least as accurate as Qwen
    # from the frames alone, on a meaningful number of questions.
    after = {"dev": C.evaluate(ctx, dev_ids, temporal)}
    trusted = set()
    for kind, group in C.by_kind(after["dev"]).items():
        hit, n = C.acc(group)
        f_hit, f_n = C.acc(group, "frames_correct")
        if n >= MIN_TRUST_N and f_n and hit / n >= f_hit / f_n:
            trusted.add(kind)
        print(f"   trust {kind}: measured {hit}/{n} vs frames {f_hit}/{f_n} -> {'TRUSTED' if kind in trusted else 'not trusted'}")
    interpret.TRUSTED_KINDS.difference_update(args.kinds)
    interpret.TRUSTED_KINDS.update(trusted)
    after["test"] = C.evaluate(ctx, test_ids, temporal)
    print("\n== calibrated (the pipeline as it will run) ==")
    for name in ("dev", "test"):
        print(name); print(table(C.report_table(after[name]), list(C.report_table(after[name])[0])))
    print("\n== strategies (class priors counted on dev; scored against ground truth) ==")
    for name in ("dev", "test"):
        print(name)
        tab = C.strategy_table(after[name], priors)
        print(table(tab, list(tab[0])))
    report = "calibration.json" if not partial else "calibration." + "+".join(args.kinds) + ".json"
    (args.out / report).write_text(json.dumps({
        "priors": {k: dict(v) for k, v in priors.items()},
        "strategies": {n: C.strategy_table(after[n], priors) for n in ("dev", "test")},
        "chosen": chosen_all, "trace": traces,
        "before_dev": C.report_table(before["dev"]),
        "after": {n: C.report_table(r) for n, r in after.items()},
    }, indent=1, default=str))
    print("wrote", args.out / report)
    if not args.no_write:
        pipeline.CALIBRATED.write_text(json.dumps({
            "constants": pipeline.snapshot(),
            "provenance": {"catalog": str(args.out), "dev_questions": len(split["dev"]),
                           "test_questions": len(split["test"]), "min_gain": MIN_GAIN, "min_trust_n": MIN_TRUST_N,
                           "note": "chosen on dev clips only; test clips are read once for the report"},
        }, indent=1))
        print("wrote", pipeline.CALIBRATED)


if __name__ == "__main__":
    main()
