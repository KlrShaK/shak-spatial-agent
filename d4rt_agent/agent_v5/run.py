"""Run AgentV5 on Dyn-Bench or DSI-Bench.

    python -m d4rt_agent.agent_v5.run --bench dynbench --backbone point4d --results-dir <dir> --qwen-model <path>
    python -m d4rt_agent.agent_v5.run --bench dsi --split std --random-sample 200 --backbone d4rt ...
    python -m d4rt_agent.agent_v5.run --bench vlm4d --backbone point4d --results-dir <dir> ...

Dyn-Bench questions come from a catalog (default: the decomposition study's 2800-question eval
split, ``dyn_dataset_exploration/outputs/decomposition_eval/catalog.json``) or from the whole
parquet with ``--all``. DSI-Bench uses v4's manifest builder. One ``answers/<qid>.json`` per
question, written atomically; a rerun skips what exists (``--force`` redoes it). The ground truth
is kept in the answer file's ``reference`` block and is never passed to the agent.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = Path("/cluster/work/igp_psr/spanwar/shak-spatial-agent/dyn_dataset_exploration/outputs/"
                      "decomposition_eval/catalog.json")
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", choices=["dynbench", "dsi", "vlm4d"], default="dynbench")
    p.add_argument("--backbone", choices=["point4d", "d4rt"], default="point4d")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None, help="clips, scenes, groundings (default: <results>/cache)")
    p.add_argument("--qwen-model", default=os.environ.get("MODEL_DIR", DEFAULT_QWEN_MODEL))
    # Dyn-Bench
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--all", action="store_true", help="every Dyn-Bench VQA row instead of a catalog")
    p.add_argument("--dyn-root", type=Path, default=Path("/cluster/work/igp_psr/spanwar/datasets/Dyn-Bench"))
    # VLM4D (real + synthetic multiple-choice QA, offline copy; videos are plain mp4)
    p.add_argument("--vlm4d-root", type=Path, default=Path("/cluster/work/igp_psr/spanwar/datasets/VLM4D"))
    # DSI-Bench
    p.add_argument("--dsi-root", type=Path, default=Path("/cluster/work/igp_psr/spanwar/datasets/DSI-Bench"))
    p.add_argument("--split", default="std")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--random-sample", type=int, default=200)
    # selection and budgets
    p.add_argument("--only", nargs="*", default=None, help="question ids")
    p.add_argument("--selection-file", type=Path, default=None, help="answer-blind benchmark -> question ID manifest")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke", type=int, default=None, metavar="N",
                   help="smoke test: N questions, one per question group (catalog branch / DSI category) first, "
                        "the rest from the largest groups in proportion; drawn with a fixed seed")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="process the clips in a seeded random order (a clip's questions stay together), so a run "
                        "cut short at any point is still an unbiased sample; default: sorted by clip")
    p.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th clip")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stack", choices=["legacy", "compositional"], default="legacy",
                   help="candidate compositional stack is opt-in until matched evaluation")
    p.add_argument("--legacy-menu", action="store_true", help="ablation: compositional workflow with legacy family/kind menus")
    p.add_argument("--answer-policy", choices=["plain", "rated", "rated_repair"], default=None)
    p.add_argument("--repair-tool-limit", type=int, choices=range(5), default=4)
    p.add_argument("--repair-cycles", type=int, default=None,
                   help="plain-policy review cycles; 0 disables repair, positive values enable one rated repair round")
    p.add_argument("--answer-views", choices=["targeted", "overview", "standard"], default="targeted")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--token-budget", type=int, default=16000)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--dry-run", action="store_true", help="resolve questions and clips and print the first prompt; "
                                                          "no model is loaded")
    args = p.parse_args(argv)
    args.answer_policy = args.answer_policy or ("rated_repair" if args.stack == "compositional" else "plain")
    args.repair_cycles = args.repair_cycles if args.repair_cycles is not None else (2 if args.answer_policy == "plain" else 1)
    if args.stack == "legacy" and args.answer_policy != "plain":
        p.error("rated policies require --stack compositional")
    return args


# --------------------------------------------------------------------------- #
# questions
# --------------------------------------------------------------------------- #

def dynbench_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all:
        from dyn_dataset_exploration.dyn_bench_data import load_rows, qa_parquet

        return [{"id": r.question_id, "question": r.question, "options": dict(r.options), "answer": r.answer,
                 "dataset": r.dataset, "video": r.video, "task": r.task, "group": r.task}
                for r in load_rows(qa_parquet(args.dyn_root))]
    catalog = json.loads(Path(args.catalog).read_text())
    return [{"id": q["qid"], "question": q["question"], "options": dict(q["options"]), "answer": q["answer"],
             "dataset": q["dataset"], "video": q["video"], "task": q["task"],
             # the research pipeline's rule label: reporting only, never shown to the agent
             "group": q.get("branch") or q.get("rule_kind")}
            for q in catalog["questions"]]


def vlm4d_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Every VLM4D question (real 1371 + synthetic 445), from the offline ``QA_local`` files.

    The benchmark stores the answer as the option's TEXT (sometimes a number), so it is mapped to a letter
    here; if two options carry the same text, either letter counts (``answer_also``). The id is unique only
    within one file, so the question id is ``vlm4d_<split>_<video>_<n>``: its last ``_`` separates the video,
    which is the unit the report resamples (several questions share a video)."""

    import re

    root = Path(args.vlm4d_root)
    out: list[dict[str, Any]] = []
    for split, name in (("real", "real_mc.json"), ("synthetic", "synthetic_mc.json")):
        for row in json.loads((root / "QA_local" / name).read_text()):
            options = {str(k): str(v).strip() for k, v in sorted(row["choices"].items())}
            letters = [k for k, v in options.items() if v == str(row["answer"]).strip()]
            if not letters:
                raise ValueError(f"VLM4D {split} {row['id']}: answer {row['answer']!r} matches no option {options}")
            video = Path(row["video"])
            relative = str(video.relative_to(root)) if video.is_relative_to(root) else video.name
            stem = re.sub(r"[^A-Za-z0-9-]+", "-", relative.rsplit(".", 1)[0].split("/", 1)[-1])
            source = relative.split("/")[1] if split == "real" else "synthetic"
            out.append({"id": f"vlm4d_{split}_{stem}_{row['id'].rsplit('_', 1)[-1]}", "question": row["question"],
                        "options": options, "answer": letters[0], "answer_also": letters[1:],
                        "relative_path": relative, "video_path": str(video), "task": split, "group": source,
                        "dataset": f"vlm4d_{split}"})
    return out


def dsi_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    from d4rt_agent.dsi_bench_data import DEFAULT_SEED, build_manifest, read_manifest, write_manifest

    path = args.manifest or (args.results_dir / "manifest.json")
    if path.exists():
        manifest = read_manifest(path)
    else:
        manifest = build_manifest(dsi_root=args.dsi_root, split=args.split, seed=DEFAULT_SEED,
                                  random_size=args.random_sample)
        write_manifest(manifest, path)
    out = []
    for e in manifest["questions"]:
        out.append({"id": e["question_id"], "question": e["question"], "options": dict(e["options"]), "answer": e["gt"],
                    "relative_path": e["relative_path"], "video_path": e["video_path"], "task": e.get("category_name"),
                    "group": e.get("cate")})
    return out


def select(entries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if getattr(args, "selection_file", None):
        manifest = json.loads(args.selection_file.read_text())
        wanted = set(manifest["question_ids"][args.bench])
        missing = wanted - {e["id"] for e in entries}
        if missing:
            raise ValueError(f"selection IDs absent from loaded benchmark ({len(missing)}); use --all / full DSI manifest")
        entries = [e for e in entries if e["id"] in wanted]
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["id"] in wanted]
    if args.smoke:
        entries = smoke_sample(entries, args.smoke)
    key = (lambda e: e.get("video_path") or f"{e['dataset']}/{e['video']}")  # noqa: E731
    clips = sorted({key(e) for e in entries})
    index, count = (int(v) for v in args.shard.split("/"))
    mine = set(clips[index::count])
    order = sorted(mine)
    if args.shuffle_seed is not None:
        import random

        random.Random(args.shuffle_seed).shuffle(order)
    rank = {clip: i for i, clip in enumerate(order)}
    entries = sorted((e for e in entries if key(e) in mine), key=lambda e: (rank[key(e)], e["id"]))  # a clip's questions together
    return entries[: args.limit] if args.limit is not None else entries


def smoke_sample(entries: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """``n`` questions: one per group, then the remaining slots to groups by size (largest remainder)."""

    import random

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in sorted(entries, key=lambda e: e["id"]):
        groups.setdefault(str(e.get("group")), []).append(e)
    for members in groups.values():
        rng.shuffle(members)
    quota = {g: 1 for g in groups}
    extra = n - len(groups)
    if extra > 0:
        total = sum(len(m) - 1 for m in groups.values())
        share = {g: extra * (len(m) - 1) / max(total, 1) for g, m in groups.items()}
        for g in groups:
            quota[g] += int(share[g])
        for g in sorted(groups, key=lambda g: -(share[g] - int(share[g])))[: n - sum(quota.values())]:
            quota[g] += 1
    picked = [e for g, members in sorted(groups.items()) for e in members[: quota[g]]]
    return picked[:n]


def open_clip(entry: Mapping[str, Any], args: argparse.Namespace, cache: Path) -> Any:
    from d4rt_agent.agent_v5 import clips

    if args.bench in ("dsi", "vlm4d"):                      # both are plain videos decoded to a frame folder
        return clips.open_dsi(Path(entry["video_path"]), entry["relative_path"], cache)
    return clips.open_dynbench(args.dyn_root, entry["dataset"], entry["video"], cache)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> int:
    from d4rt_agent.agent_v5.models import Models, gpu_memory_report
    from d4rt_agent.agent_v5.orchestrator import AgentV5
    from d4rt_agent.agent_v5.tools import Workspace

    entries = select({"dsi": dsi_entries, "vlm4d": vlm4d_entries}.get(args.bench, dynbench_entries)(args), args)
    cache = args.cache_dir or (args.results_dir / "cache")
    answers = args.results_dir / "answers"
    pending = [e for e in entries if args.force or not (answers / f"{e['id']}.json").exists()]
    write_json(args.results_dir / "run_config.json", {
        "bench": args.bench, "backbone": args.backbone, "qwen_model": args.qwen_model, "catalog": str(args.catalog),
        "all": args.all, "split": args.split,
        "max_steps": args.max_steps if args.max_steps is not None else (32 if args.stack == "compositional" else 16),
        "token_budget": args.token_budget,
        "shard": args.shard, "questions": len(entries), "pending": len(pending),
        "code_sha": os.environ.get("CODE_SHA"), "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    print(f"{len(entries)} questions, {len(pending)} pending ({args.bench}, backbone {args.backbone})", flush=True)
    if args.dry_run:
        return dry_run(pending, args, cache)

    models = Models(qwen_model=args.qwen_model, backbone=args.backbone, cache=cache,
                    max_new_tokens=args.max_new_tokens)

    def workspace(clip: Any, program: Any) -> Workspace:
        return Workspace(clip, program, scene_fn=models.scenes().scene, ground_fn=models.ground,
                         orient_fn=models.orient, **({"point_fn": models.track_points} if hasattr(models, "track_points") else {}))

    agent_class = AgentV5
    extra = {}
    if args.stack == "compositional":
        from d4rt_agent.agent_v5.composition import CompositionalAgentV5
        agent_class = CompositionalAgentV5
        extra = dict(legacy_menu=args.legacy_menu, repair_cycles=args.repair_cycles, answer_views=args.answer_views,
                     answer_policy=args.answer_policy, repair_tool_limit=args.repair_tool_limit)
    agent = agent_class(qwen=models.qwen(), workspace_factory=workspace, backbone=args.backbone,
                        max_steps=args.max_steps if args.max_steps is not None else (32 if extra else 16),
                        token_budget=args.token_budget, **extra)
    tally: Counter[str] = Counter()
    for n, entry in enumerate(pending, 1):
        started = time.time()
        try:
            clip = open_clip(entry, args, cache)
        except Exception as error:                              # noqa: BLE001 -- recorded, the run goes on
            solution = {"question_id": entry["id"], "status": "error", "answered": False, "letter": None,
                        "error": f"clip: {error}"}
        else:
            task = {"id": entry["id"], "question": entry["question"], "options": entry["options"], "clip": clip}
            solution = agent.solve(task)
        solution["reference"] = {"answer": entry["answer"], "options": entry["options"], "task": entry.get("task"),
                                 "group": entry.get("group"), "dataset": entry.get("dataset")}
        solution["correct"] = solution.get("letter") in (entry["answer"], *entry.get("answer_also", ()))
        solution["wall_seconds"] = round(time.time() - started, 1)
        solution.update(gpu_memory_report())
        write_json(answers / f"{entry['id']}.json", solution)
        tally["correct" if solution["correct"] else solution.get("status", "wrong")] += 1
        print(f"[{n}/{len(pending)}] {entry['id']}: {solution.get('kind')} -> {solution.get('letter')} "
              f"(gt {entry['answer']}) {solution['wall_seconds']}s  {dict(tally)}", flush=True)
    return 0


def dry_run(entries: Sequence[Mapping[str, Any]], args: argparse.Namespace, cache: Path) -> int:
    from d4rt_agent.agent_v5 import kinds
    from d4rt_agent.agent_v5.orchestrator import prompt

    failures = 0
    for entry in entries:
        try:
            clip = open_clip(entry, args, cache)
            print(f"ok   {entry['id']}: {clip.length} GF frames, fps {clip.fps}, answer frames {len(clip.answer_frames)}",
                  flush=True)
        except Exception as error:                              # noqa: BLE001
            failures += 1
            print(f"FAIL {entry['id']}: {error}", flush=True)
    if entries:
        print("\n--- first decomposition prompt ---\n" + prompt("system.md") + "\n\n"
              + prompt("choose_family.md", question=entries[0]["question"] + ("\nOptions: " + str(entries[0]["options"])
                        if args.stack == "compositional" else ""), menu=(kinds.family_menu(
                            legacy=args.stack == "legacy" or args.legacy_menu) if hasattr(kinds, "LEGACY_FAMILIES")
                            else kinds.family_menu())))
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
