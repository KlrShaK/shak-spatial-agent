"""Run the jev agent on the Dyn-Bench eval split.

    python -m d4rt_agent.agent_v5_jev.run --results-dir <run> --backbone point4d --seed-cache <V5 run>/cache \
        [--api-mode live|collect|cached] [--smoke N | --limit N | --only ID ...] [--shard i/n]

Questions, clips, scene caches and SAM3 grounding are AgentV5's (``agent_v5.run``, ``agent_v5.models``).
What differs: no Qwen is loaded. Every decision is a Decisions call (``decisions.py``, cached on disk), made
on the clip's caption (``caption.py``, written beforehand). One ``answers/<qid>.json`` per question; a rerun
skips what exists. In ``collect`` mode a question that needs an API answer it does not have yet is skipped
(its request waits in ``decisions/pending/``); after ``decisions.py`` sends them, rerun to continue.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from d4rt_agent.agent_v5.run import dynbench_entries, open_clip, select, write_json
from d4rt_agent.agent_v5_jev import decisions
from d4rt_agent.agent_v5_jev.args import parse
from d4rt_agent.agent_v5_jev.caption import load_caption


def seed_clip_cache(seed: Path, cache: Path, slug: str, backbone: str) -> None:
    """Link a finished run's scene and SAM3 files for one clip into this run's cache.

    The directories are real (new groundings are written here) and the files are symlinks, so the
    seed run's cache is read but never written."""

    for sub in (backbone, "sam3"):
        source, target = Path(seed) / "clips" / slug / sub, Path(cache) / "clips" / slug / sub
        if not source.is_dir():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            link = target / path.name
            if not link.exists() and not link.is_symlink():
                os.symlink(path, link)


def run(args: Any) -> int:
    from d4rt_agent.agent_v5.models import Models, gpu_memory_report
    from d4rt_agent.agent_v5.tools import Workspace
    from d4rt_agent.agent_v5_jev.agent import JevAgent

    entries = select(dynbench_entries(args), args)
    cache = args.cache_dir or (args.results_dir / "cache")
    answers = args.results_dir / "answers"
    pending = [e for e in entries if args.force or not (answers / f"{e['id']}.json").exists()]
    tag = "" if args.shard == "0/1" else "_shard_" + args.shard.replace("/", "_of_")   # shard processes share RUN_DIR
    write_json(args.results_dir / f"run_config{tag}.json", {
        "arm": "jev-flow", "bench": args.bench, "backbone": args.backbone, "decisions_model": decisions.DEFAULT_MODEL,
        "caption_model": str(args.caption_model), "api_mode": args.api_mode, "seed_cache": str(args.seed_cache),
        "catalog": str(args.catalog), "max_tool_steps": args.max_tool_steps, "window_mode": args.window_mode, "shard": args.shard,
        "questions": len(entries), "pending": len(pending), "code_sha": os.environ.get("CODE_SHA"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    print(f"{len(entries)} questions, {len(pending)} pending (jev-flow, backbone {args.backbone}, "
          f"api {args.api_mode})", flush=True)

    models = Models(qwen_model=None, backbone=args.backbone, cache=cache)

    def workspace(clip: Any, program: Any) -> Workspace:
        return Workspace(clip, program, scene_fn=models.scenes().scene, ground_fn=models.ground,
                         orient_fn=models.orient)

    agent = JevAgent(
        decide=lambda text, questions: decisions.decide(text, questions, args.decisions_cache, mode=args.api_mode),
        workspace_factory=workspace, captions=lambda slug: load_caption(args.captions_dir, slug),
        backbone=args.backbone, max_tool_steps=args.max_tool_steps, window_mode=args.window_mode)
    tally: Counter[str] = Counter()
    for n, entry in enumerate(pending, 1):
        started = time.time()
        try:
            if args.seed_cache:
                from dyn_dataset_exploration.dyn_bench_data import video_slug

                seed_clip_cache(args.seed_cache, cache, video_slug(entry["dataset"], entry["video"]), args.backbone)
            clip = open_clip(entry, args, cache)
        except Exception as error:                              # noqa: BLE001 -- recorded, the run goes on
            solution = {"question_id": entry["id"], "status": "error", "answered": False, "letter": None,
                        "error": f"clip: {error}"}
        else:
            task = {"id": entry["id"], "question": entry["question"], "options": entry["options"], "clip": clip}
            try:
                solution = agent.solve(task)
            except decisions.NeedRequest:
                tally["waiting"] += 1
                continue
            except FileNotFoundError as error:                  # no caption yet: no answer file, a rerun fills it in
                tally["no_caption"] += 1
                print(f"[{n}/{len(pending)}] {entry['id']}: skipped, {error}", flush=True)
                continue
        solution["reference"] = {"answer": entry["answer"], "options": entry["options"], "task": entry.get("task"),
                                 "group": entry.get("group"), "dataset": entry.get("dataset")}
        solution["correct"] = solution.get("letter") == entry["answer"]
        solution["wall_seconds"] = round(time.time() - started, 1)
        solution.update(gpu_memory_report())
        write_json(answers / f"{entry['id']}.json", solution)
        tally["correct" if solution["correct"] else solution.get("status", "wrong")] += 1
        print(f"[{n}/{len(pending)}] {entry['id']}: {solution.get('kind')} -> {solution.get('letter')} "
              f"(gt {entry['answer']}) {solution['wall_seconds']}s  {dict(tally)}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run(parse(argv)))


if __name__ == "__main__":
    main()
