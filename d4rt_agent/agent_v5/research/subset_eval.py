"""Bounded, answer-blind AgentV5 subset evaluation and durable SLURM monitoring.

prepare freezes the candidate order; workers count pilots toward final samples;
control runs at most two benchmark workers and persists 30-minute status reports.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from d4rt_agent.agent_v5.run import parse_args, source_digest, write_json

ROOT = Path(__file__).resolve().parents[3]
MAIN_REPO = Path(os.environ.get("MAIN_REPO", "/cluster/work/igp_psr/spanwar/shak-spatial-agent"))
SPLITS = ("std", "hflip", "reverse", "reverse_hflip")
SEED = 20261014
CONFIG = ["--backbone", "point4d", "--stack", "compositional", "--answer-policy", "rated_repair",
          "--answer-views", "targeted", "--max-steps", "32", "--repair-tool-limit", "4",
          "--repair-cycles", "1", "--token-budget", "16000"]
EXPECTED = dict(backbone="point4d", stack="compositional", answer_policy="rated_repair",
                answer_views="targeted", max_steps=32, repair_tool_limit=4, repair_cycles=1, token_budget=16000)


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(argv, **kwargs):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=45, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr.strip()} {result.stdout.strip()}")
    return result.stdout.strip()


def stratified_order(rows, seed):
    """Nested random draw: cover kinds/sources/cells, then proportional deficits.

    Only id/source/kind/stratum fields are inspected. Prefixes are final sample
    candidates, so a runtime-dependent cutoff never changes already-run IDs.
    """
    rng = random.Random(seed)
    pools = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        pools[row["stratum"]].append(row)
    sizes = {k: len(v) for k, v in pools.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    chosen, counts = [], Counter()

    def take(cell):
        chosen.append(pools[cell].pop())
        counts[cell] += 1

    # Rare kinds first; the pilot retains kinds even if there are more cells
    # than pilot questions. Sources and all cells follow before proportional fill.
    for field in ("kind", "source"):
        totals = Counter(r[field] for r in rows)
        for label in sorted(totals, key=lambda k: (totals[k], k)):
            if any(r[field] == label for r in chosen):
                continue
            cells = [k for k, v in pools.items() if v and v[-1][field] == label]
            take(rng.choices(cells, weights=[len(pools[k]) for k in cells])[0])
    for cell in sorted(pools, key=lambda k: (sizes[k], k)):
        if not counts[cell]:
            take(cell)
    total = len(rows)
    while len(chosen) < total:
        cell = max((k for k, v in pools.items() if v),
                   key=lambda k: ((len(chosen) + 1) * sizes[k] / total - counts[k], k))
        take(cell)
    return chosen


def metadata(entry, bench):
    from vlm4d_dataset_exploration.taxonomy import classify

    source = str(entry.get("group")) if bench == "vlm4d" else str(entry.get("dataset", ""))
    kind = classify(entry["question"], [], source)["kind"] if bench == "vlm4d" else str(entry.get("group"))
    clip = entry.get("relative_path") or f"{entry['dataset']}/{entry['video']}"
    return dict(id=entry["id"], source=source, kind=kind, stratum=f"{source}:{kind}", clip=clip)


def validate_media(entries, bench, args):
    """Check selected media paths without decoding videos on the login node."""
    seen = set()
    for entry in entries:
        if bench != "dynbench":
            path = Path(entry["video_path"])
            if not path.is_file():
                raise ValueError(f"missing media: {path}")
        else:
            key = (entry["dataset"], entry["video"])
            if key in seen:
                continue
            seen.add(key)
            clip = f"{key[0]}_{key[1]}"
            folder = args.dyn_root / "frames" / key[0] / clip
            sampling = args.dyn_root / "multi_json" / key[0] / f"{clip}_frame_sampling.json"
            if not folder.is_dir() or not sampling.is_file():
                raise ValueError(f"missing Dyn media or sampling: {folder}, {sampling}")


def prepare(destination, source):
    from d4rt_agent.agent_v5.run import dsi_entries, dynbench_entries, vlm4d_entries

    destination = Path(destination).resolve()
    if (destination / "plan.json").exists():
        raise ValueError("use a new campaign directory; selection plans are immutable")
    args = parse_args(["--results-dir", str(destination), "--all"] + CONFIG)
    manifests = {}
    base = None
    for split, name in zip(SPLITS, ("agent_v4_random200", "agent_v4_hflip", "agent_v4_reverse", "agent_v4_reverse_hflip")):
        path = ROOT / "d4rt_agent/results" / name / "manifest.json"
        manifest = read(path)
        entries = manifest["questions"]
        ids = {r["question_id"] for r in entries}
        if len(entries) != 200 or len(ids) != 200 or len({r["relative_path"] for r in entries}) != 190:
            raise ValueError(f"unexpected matched DSI manifest: {path}")
        if base is not None and ids != base:
            raise ValueError("DSI augmentation IDs differ")
        base = ids
        copied = destination / "manifests" / f"dsi_{split}.json"
        write_json(copied, manifest)
        manifests[split] = str(copied)
        args.manifest, args.split = copied, split
        validate_media(dsi_entries(args), "dsi", args)
    seeds = {read(p)["sampling"]["seed"] for p in manifests.values()}
    if len(seeds) != 1:
        raise ValueError("DSI seeds differ")
    args.manifest, args.split = Path(manifests["std"]), "std"
    dsi = dsi_entries(args)
    # Restore source metadata, omitted by the common DSI loader.
    sources = {r["question_id"]: r["dataset"] for r in read(args.manifest)["questions"]}
    for entry in dsi:
        entry["dataset"] = sources[entry["id"]]
    vlm = vlm4d_entries(args)
    dyn = dynbench_entries(args)
    clips = defaultdict(list)
    for entry in dyn:
        clips[f"{entry['dataset']}/{entry['video']}"].append(entry)
    if len(clips) != 1000 or any(Counter(e["task"] for e in v) != Counter(cameraqa=3, qa=2, sceneqa=2) for v in clips.values()):
        raise ValueError("Dyn population does not contain 1000 clips with seven questions each")
    units = [dict(id=k, clip=k, source=v[0]["dataset"], kind=v[0]["dataset"], stratum=v[0]["dataset"],
                  question_ids=sorted(e["id"] for e in v)) for k, v in sorted(clips.items())]
    populations = {"dsi": [metadata(e, "dsi") for e in dsi],
                   "vlm4d": [metadata(e, "vlm4d") for e in vlm], "dynbench": units}
    if len({u["source"] for u in units}) != 8:
        raise ValueError("Dyn must retain eight sources")
    specs = {}
    for bench, ceiling, pilot in (("dsi", 200, 8), ("vlm4d", 800, 24), ("dynbench", 120, 8)):
        pool = populations[bench]
        order = stratified_order(pool, next(iter(seeds)) if bench == "dsi" else SEED)[:ceiling]
        ids = {qid for u in order for qid in u.get("question_ids", [u["id"]])}
        entries = {"dsi": dsi, "vlm4d": vlm, "dynbench": dyn}[bench]
        validate_media([e for e in entries if e["id"] in ids], bench, args)
        specs[bench] = dict(order=order, ceiling=ceiling, pilot=pilot,
                            population=dict(Counter(u["stratum"] for u in pool)),
                            seed=next(iter(seeds)) if bench == "dsi" else SEED)
    manifest = read(Path(source) / "source_manifest.json")
    if manifest["baseline"]:
        raise ValueError("evaluation requires latest candidate source")
    import pyarrow

    dependencies = Path(pyarrow.__file__).resolve().parent.parent
    plan = dict(created_at=utc(), source=str(Path(source).resolve()), source_sha256=source_digest(Path(source)),
                config=EXPECTED, config_args=CONFIG, manifests=manifests, benchmarks=specs,
                dependencies=str(dependencies), pyarrow_version=pyarrow.__version__,
                results=str(destination), budget_seconds=9.5 * 3600, overhead_seconds=1800, uncertainty=1.25,
                budget_scope="per benchmark worker; pilot GPU time included; queue excluded")
    write_json(destination / "plan.json", plan)
    return plan


def selection(spec, n):
    return [qid for u in spec["order"][:n] for qid in u.get("question_ids", [u["id"]])]


def sized_units(spec, elapsed, pilot_seconds, plan):
    """Largest candidate prefix that fits observed stratified pilot rates."""
    rates = defaultdict(list)
    for unit in spec["order"][:spec["pilot"]]:
        rates[unit["stratum"]].append(pilot_seconds[unit["id"]])
    global_rate = sum(pilot_seconds.values()) / len(pilot_seconds)
    means = {s: sum(v) / len(v) for s, v in rates.items()}
    minimum = max(spec["pilot"], len(spec["population"]))
    chosen, projected = None, None
    remaining = 0
    for n, unit in enumerate(spec["order"], 1):
        if n > spec["pilot"]:
            remaining += means.get(unit["stratum"], global_rate)
        estimate = elapsed + plan["uncertainty"] * remaining + plan["overhead_seconds"]
        if estimate <= plan["budget_seconds"] and n >= minimum:
            chosen, projected = n, estimate
    if chosen is None:
        raise RuntimeError("runtime budget cannot retain pilot and required stratum coverage")
    return chosen, projected


def extend(plan_path, bench, budget_seconds, max_units=None, load_seconds=300):
    """Grow a completed benchmark's final IDs along its frozen order (a nested prefix).

    Same sizing rule as the pilot, with rates from this campaign's completed answers
    (same source and configuration) and one model load per run.py call. The frozen
    worker then runs only the new IDs; earlier answers are reused.
    """
    plan = read(plan_path)
    spec = plan["benchmarks"][bench]
    directory = Path(plan["results"]) / bench
    state, final = read(directory / "worker_state.json"), read(directory / "final_selection.json")
    if state.get("phase") != "complete" or final["source_sha256"] != plan["source_sha256"]:
        raise RuntimeError(f"{bench}: extend only a completed worker of this plan")
    splits = SPLITS if bench == "dsi" else ("all",)
    done = final["units"]
    seconds = Counter()
    for unit in spec["order"][:done]:
        for split in splits:
            answers = directory / split / "answers" if bench == "dsi" else directory / "answers"
            seconds[unit["id"]] += sum(read(answers / f"{q}.json")["wall_seconds"] for q in unit.get("question_ids", [unit["id"]]))
    rates = defaultdict(list)
    for unit in spec["order"][:done]:
        rates[unit["stratum"]].append(seconds[unit["id"]])
    means = {s: sum(v) / len(v) for s, v in rates.items()}
    pooled = sum(seconds.values()) / len(seconds)
    chosen, projected, remaining = None, None, len(splits) * load_seconds
    for n, unit in enumerate(spec["order"][:min(max_units or spec["ceiling"], spec["ceiling"])], 1):
        if n > done:
            remaining += means.get(unit["stratum"], pooled)
            estimate = plan["uncertainty"] * remaining + plan["overhead_seconds"]
            if estimate <= budget_seconds:
                chosen, projected = n, estimate
    if chosen is None:
        raise RuntimeError(f"{bench}: budget {budget_seconds:.0f}s cannot fit one more unit")
    archive = directory / f"final_selection_{done}units.json"
    write_json(archive, final)
    write_json(directory / f"worker_state_{state.get('job_id')}.json", state)
    write_json(directory / "final_selection.json", dict(
        question_ids={bench: selection(spec, chosen)}, units=chosen, frozen_at=utc(), source_sha256=plan["source_sha256"],
        projected_total_seconds=projected, budget_seconds=budget_seconds, load_seconds=load_seconds,
        extended_from=dict(units=done, selection=str(archive), unit_seconds=dict(seconds))))
    return chosen, projected


def runner_args(plan, bench, split, selection_path):
    directory = Path(plan["results"]) / bench
    if bench == "dsi":
        directory /= split
    argv = ["--bench", bench, "--results-dir", str(directory), "--selection-file", str(selection_path),
            "--shuffle-seed", str(plan["benchmarks"][bench]["seed"]), "--qwen-model", os.environ["MODEL_DIR"]] + plan["config_args"]
    if bench == "dsi":
        argv += ["--split", split, "--manifest", plan["manifests"][split]]
    if bench == "dynbench":
        argv += ["--all"]
    return argv, directory


def checked_answers(directory, wanted, plan):
    config = read(directory / "run_config.json")
    for key, value in plan["config"].items():
        if config.get(key) != value:
            raise RuntimeError(f"parsed configuration mismatch: {key}")
    if config["source_sha256"] != plan["source_sha256"] or config["questions"] != len(wanted):
        raise RuntimeError("source hash or question count mismatch")
    rows = [read(directory / "answers" / f"{qid}.json") for qid in wanted]
    for row in rows:
        if row.get("status") == "error" or row.get("answer_policy") != "rated_repair" or row.get("answer_protocol") != "freeform_assessment_v1":
            raise RuntimeError(f"pilot/final answer failed workflow check: {row.get('question_id')} {row.get('error', row.get('status'))}")
        if any(token in json.dumps(row).lower() for token in ("out of memory", "cuda error", "traceback (most recent")):
            raise RuntimeError(f"fatal failure recorded in {row['question_id']}")
    return rows


def worker(plan_path, bench):
    plan = read(plan_path)
    directory = Path(plan["results"]) / bench
    state_path = directory / "worker_state.json"
    started = time.monotonic()
    state = dict(bench=bench, started_at=utc(), job_id=os.environ.get("SLURM_JOB_ID"), phase="pilot",
                 config=plan["config"], source_sha256=plan["source_sha256"])
    state["gpu"] = command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    write_json(state_path, state)
    try:
        import pyarrow

        if pyarrow.__version__ != plan["pyarrow_version"]:
            raise RuntimeError("staged PyArrow version differs from evaluation plan")
        if source_digest() != plan["source_sha256"]:
            raise RuntimeError("worker source differs from frozen plan")
        source_manifest = read(ROOT / "source_manifest.json")
        for name, digest in source_manifest["files"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"frozen source changed: {name}")
        spec = plan["benchmarks"][bench]
        final_path = directory / "final_selection.json"
        pilot_path = directory / "pilot_selection.json"
        pilot_ids = selection(spec, spec["pilot"])
        write_json(pilot_path, dict(question_ids={bench: pilot_ids}, seed=spec["seed"]))
        splits = SPLITS if bench == "dsi" else ("all",)
        if not final_path.exists():
            times = Counter()
            for split in splits:
                argv, output = runner_args(plan, bench, split, pilot_path)
                # Decode pilot media and validate Dyn GF before loading any model.
                subprocess.run([sys.executable, "-u", "-m", "d4rt_agent.agent_v5.run", *argv, "--dry-run"], check=True)
                subprocess.run([sys.executable, "-u", "-m", "d4rt_agent.agent_v5.run", *argv], check=True)
                rows = checked_answers(output, pilot_ids, plan)
                for unit in spec["order"][:spec["pilot"]]:
                    unit_ids = set(unit.get("question_ids", [unit["id"]]))
                    times[unit["id"]] += sum(r["wall_seconds"] for r in rows if r["question_id"] in unit_ids)
            elapsed = time.monotonic() - started
            n, projected = sized_units(spec, elapsed, times, plan)
            write_json(final_path, dict(question_ids={bench: selection(spec, n)}, units=n, frozen_at=utc(),
                                       pilot_elapsed_seconds=elapsed, pilot_unit_seconds=dict(times),
                                       projected_total_seconds=projected, source_sha256=plan["source_sha256"]))
        final = read(final_path)
        if final["source_sha256"] != plan["source_sha256"]:
            raise RuntimeError("final IDs belong to another source")
        state.update(phase="evaluation", final_units=final["units"], planned_questions=len(final["question_ids"][bench]) * len(splits),
                     projected_total_seconds=final["projected_total_seconds"], pilot_passed=True)
        write_json(state_path, state)
        for split in splits:
            argv, output = runner_args(plan, bench, split, final_path)
            subprocess.run([sys.executable, "-u", "-m", "d4rt_agent.agent_v5.run", *argv], check=True)
            checked_answers(output, final["question_ids"][bench], plan)
        state.update(phase="complete", finished_at=utc(), elapsed_seconds=time.monotonic() - started)
        write_json(state_path, state)
    except Exception as error:
        state.update(phase="failed", error=str(error), finished_at=utc())
        write_json(state_path, state)
        raise


def earliest_platform(source, time_limit):
    """One job per piece of work: the partition `sbatch --test-only` expects to start first."""
    starts = {}
    for platform in ("a100", "blackwell"):
        result = subprocess.run(["sbatch", "--test-only", f"--time={time_limit}",
                                 str(Path(source) / "scripts" / f"run_agent_v5_{platform}.slurm")],
                                cwd=source, text=True, capture_output=True, timeout=45)
        match = re.search(r"to start at (\S+)", result.stderr + result.stdout)
        if match:
            starts[platform] = match.group(1)
    if not starts:
        raise RuntimeError("sbatch --test-only gave no start estimate for either partition")
    return min(starts, key=starts.get), starts


def submit_pair(plan_path, bench, platforms=("a100", "blackwell"), time_limit="10:00:00"):
    plan = read(plan_path)
    source = Path(plan["source"])
    result = Path(plan["results"]) / bench
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{bench}-{os.getpid()}"
    race = result / ".race"
    race.mkdir(parents=True, exist_ok=True)
    record = dict(bench=bench, attempt=attempt, submitted_at=utc(), jobs={}, visibility={}, phase="submitting",
                  time_limit=time_limit)
    path = result / "submission.json"
    write_json(path, record)
    try:
        for platform in platforms:
            exports = f"ALL,REPO_ROOT={source},MAIN_REPO={MAIN_REPO},RUN_DIR={result},ATTEMPT_ID={attempt},BENCH={bench},BACKBONE=point4d,EVALUATION_PLAN={Path(plan_path).resolve()},EVALUATION_DEPENDENCIES={plan['dependencies']}"
            job = command(["sbatch", "--parsable", f"--time={time_limit}", f"--export={exports}",
                           f"--job-name=v5-subset-{bench}-{platform}", str(source / "scripts" / f"run_agent_v5_{platform}.slurm")], cwd=source).split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"missing or invalid job ID: {job!r}")
            record["jobs"][platform] = job
            write_json(path, record)
        write_json(race / f"{attempt}.json", record)
        (race / f"{attempt}.jobs").write_text("\n".join(record["jobs"].values()) + "\n")
        for platform, job in record["jobs"].items():
            visible = command(["scontrol", "show", "job", job])
            if f"JobId={job}" not in visible:
                raise RuntimeError(f"job {job} not visible to scheduler")
            record["visibility"][platform] = visible
        record["phase"] = "submitted"
        write_json(path, record)
        return record
    except Exception as error:
        record.update(phase="submission_failed", error=str(error))
        for job in record["jobs"].values():
            try:
                command(["scancel", job])
            except Exception as cancel_error:
                record.setdefault("cancellation_errors", []).append(str(cancel_error))
        write_json(path, record)
        raise


def benchmark_status(plan, bench):
    directory = Path(plan["results"]) / bench
    submission_path = directory / "submission.json"
    if not submission_path.exists():
        return dict(phase="waiting_for_slot", completed=0)
    submission = read(submission_path)
    jobs = list(submission["jobs"].values())
    status = dict(jobs=submission["jobs"], phase=submission["phase"], completed=0)
    if jobs:
        try:
            status["scheduler"] = command(["sacct", "-n", "-P", "-j", ",".join(jobs), "--format=JobIDRaw,State,Elapsed,NodeList"])
            status["scheduler_source"] = "sacct"
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            status["scheduler_error"] = str(error)
            try:
                status["scheduler"] = command(["squeue", "--noheader", "--jobs", ",".join(jobs), "--format=%i|%T|%M|%N"])
                status["scheduler_source"] = "squeue"
            except (RuntimeError, subprocess.TimeoutExpired) as queue_error:
                status["scheduler"] = ""
                status["scheduler_error"] += f"; queue query also failed: {queue_error}"
                status["scheduler_source"] = "unavailable"
    claim = directory / ".race" / f"{submission['attempt']}.claim"
    if (claim / "job").exists():
        status["winner"] = (claim / "job").read_text().strip()
        sibling = [j for j in jobs if j != status["winner"]]
        records = dict(line.split("|", 2)[:2] for line in status.get("scheduler", "").splitlines() if "|" in line)
        status["sibling_states"] = {j: records.get(j, "not visible") for j in sibling}
        # Reconcile late submission/cancellation rather than leave a sibling queued.
        for job in sibling:
            if records.get(job) in ("PENDING", "RUNNING", "CONFIGURING"):
                command(["scancel", job])
        status["sibling_cancelled"] = all(records.get(j, "").startswith(("CANCELLED", "COMPLETED")) for j in sibling)
    state_path = directory / "worker_state.json"
    if state_path.exists():
        status.update(read(state_path))
    paths = list(directory.glob("*/answers/*.json")) if bench == "dsi" else list((directory / "answers").glob("*.json"))
    rows = [read(p) for p in paths]
    status.update(completed=len(rows), planned=status.get("planned_questions", (800 if bench != "dynbench" else 840)),
                  errors=sum(r.get("status") == "error" for r in rows),
                  recoveries=dict(Counter(r.get("schema_recovery", "not_needed") for r in rows)),
                  repair_rounds=sum(len(r.get("repair_history", [])) for r in rows))
    fatal = [r for r in rows if r.get("status") == "error" or any(
        token in json.dumps(r).lower() for token in ("out of memory", "cuda error", "traceback (most recent"))]
    if fatal and status.get("phase") != "complete":
        if status.get("winner"):
            command(["scancel", status["winner"]])
        status.update(phase="failed", error=f"fatal answer failure: {fatal[0]['question_id']}")
    # Model-loading failures can precede the first answer or worker-state update.
    for job in jobs:
        platform = "a100" if submission["jobs"].get("a100") == job else "bw"
        log = MAIN_REPO / "logs/d4rt_agent" / f"agentv5_{platform}_{job}.out"
        if log.exists():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 16384))
                tail = handle.read().decode(errors="replace")
            if any(token in tail.lower() for token in ("out of memory", "traceback (most recent call last)", "cuda error")):
                if status.get("winner") == job and status.get("phase") != "complete":
                    command(["scancel", job])
                    status.update(phase="failed", error=f"fatal log failure: {log}", log_tail=tail[-4000:])
    seconds = sum(r.get("wall_seconds", 0) for r in rows)
    if seconds:
        status.update(questions_per_hour=3600 * len(rows) / seconds,
                      eta_seconds=(status["planned"] - len(rows)) * seconds / len(rows))
    if paths and status.get("phase") in ("pilot", "evaluation"):
        # An extended worker starts with earlier answers on disk: count from its own start.
        started = datetime.fromisoformat(status["started_at"]).timestamp() if status.get("started_at") else 0
        status["last_progress_age_seconds"] = time.time() - max(started, max(p.stat().st_mtime for p in paths))
        if status["last_progress_age_seconds"] > 3600:
            raise RuntimeError(f"{bench}: no completed question for over one hour; investigate worker log")
    if not paths and status.get("started_at") and status.get("phase") in ("pilot", "evaluation"):
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(status["started_at"])).total_seconds()
        if age > 3600:
            raise RuntimeError(f"{bench}: no pilot answer within one hour of startup; investigate worker log")
    top = [line.split("|")[1] for line in status.get("scheduler", "").splitlines() if line.split("|")[0] in jobs]
    terminal = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "COMPLETED", "BOOT_FAIL")
    if top and len(top) == len(jobs) and all(s.startswith(terminal) for s in top) and status.get("phase") != "complete":
        status.update(phase="failed", error="all submitted jobs ended without complete worker state")
    return status


def weighted_summary(rows, population, seed=SEED, draws=2000):
    """Population weights within strata, bootstrap shared source-video clusters."""
    import numpy as np

    if not rows:
        return dict(completed=0)
    counts = Counter(r["stratum"] for r in rows)
    total_population = sum(population.values())
    clusters = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        weight = population[row["stratum"]] / counts[row["stratum"]]
        clusters[row["clip"]][0] += weight * bool(row["correct"])
        clusters[row["clip"]][1] += weight
    values = np.array(list(clusters.values()))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        summed = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        samples.append(summed[0] / summed[1])
    return dict(completed=len(rows), accuracy=sum(bool(r["correct"]) for r in rows) / len(rows),
                answered=sum(bool(r.get("answered", r.get("status") not in ("error", "unanswered"))) for r in rows),
                status_counts=dict(Counter(r.get("status", "unknown") for r in rows)),
                population_weighted_accuracy=float(values[:, 0].sum() / values[:, 1].sum()),
                weighted_source_video_bootstrap_95_ci=np.percentile(samples, [2.5, 97.5]).tolist(),
                source_videos=len(values), represented_population_fraction=sum(population[s] for s in counts) / total_population,
                errors=sum(r.get("status") == "error" for r in rows),
                strata={s: dict(n=counts[s], accuracy=sum(bool(r["correct"]) for r in rows if r["stratum"] == s) / counts[s],
                                population=population[s]) for s in sorted(counts)})


def save_report(plan):
    results = Path(plan["results"])
    report = dict(updated_at=utc(), source_sha256=plan["source_sha256"], config=plan["config"], benchmarks={})
    for bench, spec in plan["benchmarks"].items():
        lookup = {qid: unit for unit in spec["order"] for qid in unit.get("question_ids", [unit["id"]])}
        splits = SPLITS if bench == "dsi" else ("all",)
        per_split, combined = {}, []
        for split in splits:
            directory = results / bench / split if bench == "dsi" else results / bench
            rows = []
            for path in sorted((directory / "answers").glob("*.json")):
                answer = read(path)
                unit = lookup[answer["question_id"]]
                rows.append(dict(correct=answer.get("correct", False), status=answer.get("status"),
                                 answered=bool(answer.get("letter")),
                                 stratum=unit["stratum"], clip=unit["clip"]))
            per_split[split] = weighted_summary(rows, spec["population"])
            combined.extend(rows)
        report["benchmarks"][bench] = dict(splits=per_split, combined=weighted_summary(combined, spec["population"]))
    write_json(results / "report.json", report)
    lines = ["# AgentV5 subset evaluation", "", f"Updated: {report['updated_at']}", "",
             "Accuracy includes recorded errors. Population weights use sampling strata; intervals resample source videos.",
             "Partial results cover completed questions only; see represented population fractions in report.json.", "",
             "| benchmark / split | completed | answered | accuracy | weighted accuracy | weighted 95% CI | errors |", "|---|---:|---:|---:|---:|---|---:|"]
    for bench, data in report["benchmarks"].items():
        for split, result in {**data["splits"], "combined": data["combined"]}.items():
            if result["completed"]:
                lo, hi = result["weighted_source_video_bootstrap_95_ci"]
                lines.append(f"| {bench} / {split} | {result['completed']} | {result['answered']} | {result['accuracy']:.1%} | {result['population_weighted_accuracy']:.1%} | {lo:.1%}–{hi:.1%} | {result['errors']} |")
    (results / "report.md").write_text("\n".join(lines) + "\n")


def control(plan_path):
    plan = read(plan_path)
    directory = Path(plan["results"])
    state = dict(started_at=utc(), phase="running", controller_job=os.environ.get("SLURM_JOB_ID"))
    try:
        for bench in ("dsi", "vlm4d"):
            if not (directory / bench / "submission.json").exists():
                submit_pair(plan_path, bench)
        next_report = 0
        while True:
            statuses = {bench: benchmark_status(plan, bench) for bench in plan["benchmarks"]}
            if any(s.get("phase") == "failed" or s.get("phase") == "submission_failed" for s in statuses.values()):
                raise RuntimeError(f"benchmark failure: {statuses}")
            if statuses["dynbench"]["phase"] == "waiting_for_slot" and any(statuses[b]["phase"] == "complete" for b in ("dsi", "vlm4d")):
                submit_pair(plan_path, "dynbench")
                statuses["dynbench"] = benchmark_status(plan, "dynbench")
            state.update(updated_at=utc(), benchmarks=statuses)
            write_json(directory / "status.json", state)
            if time.monotonic() >= next_report:
                write_json(directory / "monitor" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"), state)
                save_report(plan)
                print(json.dumps(state), flush=True)
                next_report = time.monotonic() + 1800
            if all(s["phase"] == "complete" for s in statuses.values()):
                state.update(phase="complete", finished_at=utc())
                write_json(directory / "status.json", state)
                save_report(plan)
                return
            time.sleep(30)
    except Exception as error:
        state.update(phase="failed", error=str(error), updated_at=utc())
        write_json(directory / "status.json", state)
        print(json.dumps(state), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "control", "status", "extend", "submit"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--bench", choices=("dsi", "vlm4d", "dynbench"))
    parser.add_argument("--budget-seconds", type=float, help="extend: GPU budget of the extension job")
    parser.add_argument("--max-units", type=int, help="extend: cap on the extended prefix")
    parser.add_argument("--time", default="10:00:00", help="submit: SLURM time limit")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.source)
    elif args.command == "worker":
        worker(args.plan, args.bench)
    elif args.command == "control":
        control(args.plan)
    elif args.command == "extend":
        print(json.dumps(dict(zip(("units", "projected_seconds"), extend(args.plan, args.bench, args.budget_seconds, args.max_units)))))
    elif args.command == "submit":
        platform, starts = earliest_platform(read(args.plan)["source"], args.time)
        record = submit_pair(args.plan, args.bench, (platform,), args.time)
        print(json.dumps(dict(test_only_starts=starts, jobs=record["jobs"], phase=record["phase"])))
    else:
        plan = read(args.plan)

        def checked(bench):  # a stall is an alert to investigate, not a reason to drop the other benchmarks
            try:
                return benchmark_status(plan, bench)
            except RuntimeError as error:
                return dict(phase="alert", error=str(error))

        state = dict(updated_at=utc(), phase="monitoring", benchmarks={b: checked(b) for b in plan["benchmarks"]})
        directory = Path(plan["results"])
        write_json(directory / "status.json", state)
        write_json(directory / "monitor" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"), state)
        print(json.dumps(state, indent=2))
        save_report(plan)


if __name__ == "__main__":
    main()
