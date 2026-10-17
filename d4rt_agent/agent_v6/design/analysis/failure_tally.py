"""Tally where V5 (and V4) questions fail: model output, host-refused arguments, or measurement.

Reads saved answer JSONs only. Writes failure_tally.json and failure_tally.md next to this script.
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path

R = Path("/cluster/work/igp_psr/spanwar/shak-spatial-agent/d4rt_agent/results")
OUT = Path(__file__).resolve().parent

RUNS = {
    "v5_dyn2800": [R / "agent_v5_eval2800_point4d_20261005T183754Z/answers"],
    "v5_dsi800": [R / "agent_v5_dsi800_point4d_20261010T012544Z/answers"],
    "v5_vlm4d": [R / "agent_v5_vlm4d_point4d_20261010T012544Z/answers"],
    "v5comp_dsi": sorted((R / "agent_v5_subset_20261014T021547Z_retry/dsi").glob("*/answers")),
    "v5comp_vlm4d": [R / "agent_v5_subset_20261014T021547Z_retry/vlm4d/answers"],
}
V4_RUNS = {"v4_census_std": R / "census/std/answers"}

ADMIN = {"ground", "select", "show", "inspect", "done_measuring", "switch_kind", "add_entity", "add_requirement",
         "merge_identity", "event_window", "observe", "record_event"}


def reject_kind(err: str) -> str:
    e = err.lower()
    if "no json" in e or "json" in e and ("invalid" in e or "decode" in e or "parse" in e or "expecting" in e):
        return "malformed output (no/invalid JSON)"
    if "does not use any noun" in e or "noun" in e:
        return "slot word check (host lexical rule)"
    if "this step needs" in e or "not offered" in e or "unknown action" in e or "not one of the actions" in e \
            or "use only" in e or "action must be" in e:
        return "wrong action for the step"
    if "must be one of" in e or "choose one of" in e or "one of [" in e:
        return "value outside the allowed menu"
    if "missing" in e or "required" in e:
        return "missing argument"
    return "other argument validation"


def unavailable_kind(reason: str) -> str:
    r = reason.lower()
    if re.search(r"could not (find|locate)|not found|no instance|found nothing|segment(er|ation) (found|failed)|no mask", r):
        return "grounding: entity not found"
    if "orientation" in r or "facing" in r or "front" in r:
        return "orientation unresolved"
    if re.search(r"fewer than|too few|at least (two|2)|only \d+ frame|not enough", r):
        return "too few usable frames"
    if "frame rate" in r or "fps" in r:
        return "no frame rate"
    if "failed (" in r or "traceback" in r or "error" in r:
        return "tool crashed"
    if "scale" in r or "metric" in r:
        return "metric scale unavailable"
    if "pose" in r or "camera" in r:
        return "camera/pose unavailable"
    return "other"


def tally_v5(name, dirs):
    files = [f for d in dirs for f in glob.glob(str(d / "*.json"))]
    s = collections.Counter()
    rej = collections.Counter()
    rej_ex = {}
    tool_err = collections.Counter()
    unav = collections.Counter()
    unav_ex = {}
    per_tool = collections.defaultdict(lambda: [0, 0])  # tool -> [calls, unavailable]
    acc = collections.defaultdict(lambda: [0, 0])        # bucket -> [correct, n]
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            s["unreadable"] += 1
            continue
        s["questions"] += 1
        ok = bool(d.get("correct"))
        s["correct"] += ok
        s["unanswered"] += not d.get("answered", d.get("letter") is not None)
        q_rej = q_terr = q_unav = 0
        measured = 0
        measured_unav = 0
        for t in d.get("trace", []):
            if t.get("status") != "accepted":
                k = reject_kind(str(t.get("error") or ""))
                rej[(t.get("phase", "").split(":")[0], k)] += 1
                rej_ex.setdefault(k, str(t.get("error"))[:160])
                q_rej += 1
            if t.get("tool_error"):
                tool_err[str(t["tool_error"])[:90]] += 1
                q_terr += 1
        for rec in d.get("tools", []) or []:
            tool = rec.get("tool")
            data = rec.get("data") or {}
            summ = str(rec.get("summary") or "")
            bad = data.get("available") is False or bool(re.search(r"unavailable|could not|failed \(", summ))
            per_tool[tool][0] += 1
            if tool not in ADMIN:
                measured += 1
            if bad:
                per_tool[tool][1] += 1
                reason = str(data.get("reason") or summ)
                k = unavailable_kind(reason)
                unav[k] += 1
                unav_ex.setdefault(k, reason[:160])
                q_unav += 1
                if tool not in ADMIN:
                    measured_unav += 1
        s["q_with_rejection"] += q_rej > 0
        s["q_with_tool_error"] += q_terr > 0
        s["q_with_unavailable"] += q_unav > 0
        all_unav = measured > 0 and measured_unav == measured
        s["q_all_measurements_unavailable"] += all_unav
        s["q_no_measurement_call"] += measured == 0
        for bucket, flag in (("all", True), ("any_rejection", q_rej > 0), ("any_tool_error", q_terr > 0),
                             ("any_unavailable", q_unav > 0), ("all_measurements_unavailable", all_unav),
                             ("no_measurement_call", measured == 0),
                             ("measured_ok", measured > 0 and measured_unav == 0)):
            if flag:
                acc[bucket][0] += ok
                acc[bucket][1] += 1
    return {
        "summary": dict(s),
        "rejections_by_phase_kind": {f"{p} | {k}": v for (p, k), v in rej.most_common()},
        "rejection_examples": rej_ex,
        "tool_errors_top": dict(tool_err.most_common(12)),
        "tool_errors_total": sum(tool_err.values()),
        "unavailable_by_reason": dict(unav.most_common()),
        "unavailable_examples": unav_ex,
        "per_tool_calls_unavailable": {k: v for k, v in sorted(per_tool.items(), key=lambda kv: -kv[1][0])},
        "accuracy_by_bucket": {k: [v[0], v[1], round(100 * v[0] / v[1], 1) if v[1] else None] for k, v in acc.items()},
    }


def tally_v4(name, d):
    files = glob.glob(str(d / "*.json"))
    s = collections.Counter()
    rej = collections.Counter()
    unav = collections.Counter()
    unav_ex = {}
    for f in files:
        try:
            a = json.load(open(f))
        except Exception:
            s["unreadable"] += 1
            continue
        s["questions"] += 1
        s["unanswered"] += not a.get("answered")
        q_rej = 0
        for t in a.get("trace", []):
            if t.get("status") != "accepted":
                rej[reject_kind(str(t.get("error") or ""))] += 1
                q_rej += 1
        s["q_with_rejection"] += q_rej > 0
        if not a.get("measurement_available"):
            s["q_measurement_unavailable"] += 1
            reason = str(a.get("measurement_unavailable_reason") or "")
            k = unavailable_kind(reason)
            unav[k] += 1
            unav_ex.setdefault(k, reason[:160])
        s["model_calls"] += len(a.get("trace", []))
    return {"summary": dict(s), "rejections": dict(rej.most_common()), "unavailable_by_reason": dict(unav.most_common()),
            "unavailable_examples": unav_ex}


def main():
    out = {}
    for name, dirs in RUNS.items():
        out[name] = tally_v5(name, dirs)
        print(name, out[name]["summary"], flush=True)
    for name, d in V4_RUNS.items():
        out[name] = tally_v4(name, d)
        print(name, out[name]["summary"], flush=True)
    (OUT / "failure_tally.json").write_text(json.dumps(out, indent=1))
    lines = []
    for name, r in out.items():
        lines.append(f"## {name}\n")
        lines.append("```\n" + json.dumps(r, indent=1)[:6000] + "\n```\n")
    (OUT / "failure_tally.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
