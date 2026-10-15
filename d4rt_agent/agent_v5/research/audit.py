"""Answer-blind corpus screens and descriptive joins to existing AgentV5 traces.

Run on SLURM: this scans several thousand saved answer files, but loads no models
or videos. No fitting, option selection, or production-agent modification.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[3]
DATA = REPO.parent / "datasets"
OUT = REPO / "d4rt_agent/results/agent_v5_capability_research_20261013"
INPUTS = {
    "DSI": DATA / "DSI-Bench/metadatas/std.csv",
    "Dyn": DATA / "Dyn-Bench/qa_tasks.parquet",
    "VLM4D": REPO / "vlm4d_dataset_exploration/logs/manifest.json",
}
RUNS = {
    "DSI": "agent_v5_dsi800_point4d_20261010T012544Z",
    "Dyn": "agent_v5_eval2800_point4d_20261005T183754Z",
    "VLM4D": "agent_v5_vlm4d_point4d_20261010T012544Z",
}
# Independent, overlapping screens. Options are needed even when the question
# stem does not mention the distinguishing quantity. These are not a router.
PATTERNS = {
    "absence": r"\b(?:there (?:is|are) no\b|not (?:present|visible)\b|absent\b|does not appear\b)",
    "own_frame": r"\b(?:own (?:orientation|perspective|frame)|starting orientation|its (?:own )?(?:left|right)|their (?:left|right))\b",
    "camera_reference": r"\b(?:relative to (?:the )?camera|towards? (?:the )?camera|away from (?:the )?camera|camera.s (?:perspective|view))\b",
    "rotation": r"\b(?:turn(?:s|ing|ed)?|rotat\w*|spin\w*|clockwise|counterclockwise|anticlockwise|orbit\w*)\b",
    "count": r"\b(?:how many|number of|count(?:ed|ing)?)\b",
    "repetition": r"\b(?:how many times|number of times|repetitions?|cycles?|laps?)\b",
    "anatomical_hand": r"\b(?:(?:left|right)[ -]hand|which hand|both hands)\b",
    "signed_relation": r"\b(?:left of|right of|in front of|behind|ahead of|above|below|beneath)\b",
    "region_path": r"\b(?:along|across|cross(?:es|ing|ed)?|inside|outside|through|enter(?:s|ing|ed)?)\b",
    "group_configuration": r"\b(?:cluster\w*|dispers\w*|formation|group of|flock|herd|spread(?:ing)? (?:out|apart))\b",
    "event_order": r"\b(?:happens? first|occurs? first|before|after|simultaneous\w*|earlier|later|preced\w*)\b",
    "acceleration": r"\b(?:accelerat\w*|decelerat\w*|speed(?:s|ing)? up|slow(?:s|ing)? down|faster|slower)\b",
    "phase_comparison": r"\b(?:first 25|middle 50|last 25|first half|second half|lowest|highest|variance)\b",
    "metric": r"\b(?:metres?|meters?|m/s|km/h|mph)\b",
}
SCREENS = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source_hashes() -> dict:
    paths = list(INPUTS.values()) + list(Path(__file__).parent.glob("*.py"))
    paths += list((REPO / "d4rt_agent/agent_v5").glob("*.py"))
    paths.append(REPO / "d4rt_agent/agent_v5/thresholds.json")
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def load_corpus() -> list[dict]:
    from d4rt_agent.dsi_bench_data import parse_options, video_slug
    from dyn_dataset_exploration.dyn_bench_data import parse_options as dyn_options
    from dyn_dataset_exploration.dyn_bench_data import video_slug as dyn_slug
    import pyarrow.parquet as pq

    rows = []
    with INPUTS["DSI"].open() as f:
        for r in csv.DictReader(f):
            source = r["relative_path"].split("/")[0]
            rows.append(dict(benchmark="DSI", qid=f"{source}_c{r['cate']}_{video_slug(r['relative_path'])}",
                             source=source, video=r["relative_path"], group=r["cate"],
                             question=r["question"], options=parse_options(r["options"])))
    columns = ["dataset", "video", "task", "qa_index", "question", "options"]
    for r in pq.read_table(INPUTS["Dyn"], columns=columns).to_pylist():
        rows.append(dict(benchmark="Dyn", qid=f"{r['task']}_{dyn_slug(r['dataset'], r['video'])}_{r['qa_index']}",
                         source=r["dataset"], video=r["video"], group=r["task"],
                         question=r["question"], options=dyn_options(r["options"])))
    for r in json.loads(INPUTS["VLM4D"].read_text())["questions"]:
        # Deliberately exclude gold, paper_type (partly gold-derived), taxonomy,
        # program and expected kind. Use only original question and options.
        rows.append(dict(benchmark="VLM4D", qid=r["qid"], source=r["source"], video=r["video"],
                         group=r["split"], question=r["question"], options=r["options"]))
    assert Counter(r["benchmark"] for r in rows) == {"DSI": 1769, "Dyn": 7000, "VLM4D": 1816}
    assert len({(r["benchmark"], r["qid"]) for r in rows}) == len(rows)
    for r in rows:
        r["question_hits"] = [k for k, p in SCREENS.items() if p.search(r["question"])]
        opt = "\n".join(r["options"].values())
        r["option_hits"] = [k for k, p in SCREENS.items() if p.search(opt)]
        r["hits"] = sorted(set(r["question_hits"] + r["option_hits"]))
    return rows


def census(rows: list[dict]) -> dict:
    result = {}
    for bench in INPUTS:
        subset = [r for r in rows if r["benchmark"] == bench]
        result[bench] = {"questions": len(subset), "clips": len({r["video"] for r in subset}),
                         "groups": dict(Counter(r["group"] for r in subset)), "screens": {}}
        for flag in SCREENS:
            hits = [r for r in subset if flag in r["hits"]]
            result[bench]["screens"][flag] = {
                "either": len(hits), "question": sum(flag in r["question_hits"] for r in hits),
                "options_only": sum(flag not in r["question_hits"] for r in hits),
            }
    return result


def review_sample(rows: list[dict], n: int = 5) -> list[dict]:
    """Diverse, deterministic retrieval sample; no gold or prediction lookup."""
    selected = {}
    for bench in INPUTS:
        for flag in SCREENS:
            candidates = [r for r in rows if r["benchmark"] == bench and flag in r["hits"]]
            seen_questions = set()
            for r in sorted(candidates, key=lambda r: hashlib.sha256(r["qid"].encode()).hexdigest()):
                stem = r["question"].lower()
                if stem in seen_questions:
                    continue
                seen_questions.add(stem)
                key = (bench, r["qid"])
                selected.setdefault(key, {**r, "retrieved_for": []})["retrieved_for"].append(flag)
                if len(seen_questions) >= n:
                    break
    return list(selected.values())


def trace_audit(rows: list[dict], out: Path) -> dict:
    records, summaries = [], {}
    for bench, run in RUNS.items():
        lookup = {r["qid"]: r for r in rows if r["benchmark"] == bench}
        paths = sorted((REPO / "d4rt_agent/results" / run / "answers").glob("*.json"))
        assert paths, run
        for path in paths:
            a = json.loads(path.read_text())
            q = lookup[a["question_id"]]
            measurements = [t for t in a.get("tools", []) if t["tool"] not in
                            {"ground", "select", "show", "switch_kind", "done_measuring"}]
            unavailable = [t for t in measurements if t.get("data", {}).get("available") is False]
            events = (a.get("program") or {}).get("events", [])
            records.append({"benchmark": bench, "qid": q["qid"], "source": q["source"], "video": q["video"],
                            "hits": q["hits"], "kind": a.get("kind"), "family": a.get("family"),
                            "correct": bool(a["correct"]), "answered": bool(a.get("letter")),
                            "decomposition_ok": a.get("decomposition_ok"),
                            "measurement_calls": len(measurements), "unavailable_calls": len(unavailable),
                            "unavailable_reasons": [t["data"].get("reason") for t in unavailable],
                            "tools": [t["tool"] for t in measurements],
                            "track_frames": [t.get("arguments", {}).get("frame") or "scene"
                                             for t in measurements if t["tool"] == "entity_track"],
                            "event_predicates": [e["predicate"] for e in events]})
        rs = [r for r in records if r["benchmark"] == bench]
        def aggregate(subset):
            return {"n": len(subset), "correct": sum(r["correct"] for r in subset),
                    "unanswered": sum(not r["answered"] for r in subset),
                    "no_measurement": sum(r["measurement_calls"] == 0 for r in subset),
                    "any_unavailable": sum(r["unavailable_calls"] > 0 for r in subset),
                    "all_attempted_unavailable": sum(0 < r["measurement_calls"] == r["unavailable_calls"] for r in subset),
                    "kinds": dict(Counter(r["kind"] for r in subset)),
                    "track_frames": dict(Counter(f for r in subset for f in r["track_frames"]))}
        summaries[bench] = {"run": run, "overall": aggregate(rs),
                            "by_screen": {flag: aggregate([r for r in rs if flag in r["hits"]]) for flag in SCREENS},
                            "by_kind": {kind: aggregate([r for r in rs if r["kind"] == kind])
                                        for kind in sorted({r["kind"] for r in rs}, key=str)},
                            "event_predicates": dict(Counter(p for r in rs for p in r["event_predicates"]))}
        print(f"{bench}: joined {len(rs)} traces", flush=True)
    dump(out / "trace_rows.json", records)
    return summaries


def write_tables(out: Path, counts: dict, traces: dict) -> None:
    lines = ["# Automatically generated corpus screens", "",
             "Overlapping lexical retrieval flags; not semantic coverage labels. Cells: total (options-only).", "",
             "| Screen | DSI / 1769 | Dyn / 7000 | VLM4D / 1816 |", "|---|---:|---:|---:|"]
    for flag in SCREENS:
        values = [counts[b]["screens"][flag] for b in INPUTS]
        lines.append("| " + flag + " | " + " | ".join(f"{v['either']} ({v['options_only']})" for v in values) + " |")
    lines += ["", "## Existing run join", "", "Availability here means an explicit `available=False` in a saved tool record.",
              "Absence of that flag does not establish correctness, adequate support, or full coverage.", "",
              "| Benchmark | Scored | Correct | Unanswered | No measurement | Any unavailable | All attempted unavailable |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for b, summary in traces.items():
        r = summary["overall"]
        lines.append(f"| {b} | {r['n']} | {r['correct']} | {r['unanswered']} | {r['no_measurement']} | "
                     f"{r['any_unavailable']} | {r['all_attempted_unavailable']} |")
    (out / "census_tables.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_corpus()
    counts = census(rows)
    dump(args.out / "corpus_blind.json", rows)
    dump(args.out / "census.json", counts)
    dump(args.out / "review_candidates.json", review_sample(rows))
    dump(args.out / "screens.json", PATTERNS)
    dump(args.out / "source_hashes.json", source_hashes())
    dump(args.out / "provenance.json", {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                                       "scope": "all three datasets inform algorithm design; no training or fitting"})
    print(f"Census: {len(rows)} questions", flush=True)
    traces = trace_audit(rows, args.out)
    dump(args.out / "trace_summary.json", traces)
    write_tables(args.out, counts, traces)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
