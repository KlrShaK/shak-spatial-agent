"""Which way of asking the 2B captioner works? Variants on a few clips, scored on usefulness.

    python -m d4rt_agent.agent_v5_jev.caption_experiment run   --results-dir R --caption-model M [--clips 12] [--smoke 30]
    python -m d4rt_agent.agent_v5_jev.caption_experiment score --results-dir R --v5-table T

``run`` (GPU) writes ``R/caption_exp/<variant>/<slug>.json`` with the text plus input and output token
counts and whether generation stopped by itself. ``score`` (CPU) reports per variant: entities parsed,
repeated-line share, how often the entity list was reached, tokens, and the share of the subjects
AgentV5's Qwen named on that clip's questions that the caption's entities also name.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from d4rt_agent.agent_v5_jev.caption import generate, messages, parse_entities, pick_frames, timeline_spans

# name: frames, long side, prompt(s), repetition penalty, max new tokens
VARIANTS: dict[str, dict[str, Any]] = {
    "f16_long": dict(frames=16, side=448, prompts=["caption.md"], rp=1.0, tokens=700),
    "f8_long": dict(frames=8, side=448, prompts=["caption.md"], rp=1.0, tokens=700),
    "f8_short": dict(frames=8, side=448, prompts=["caption_short.md"], rp=1.0, tokens=400),
    "f4_short": dict(frames=4, side=448, prompts=["caption_short.md"], rp=1.0, tokens=400),
    "f16_small_short": dict(frames=16, side=256, prompts=["caption_short.md"], rp=1.0, tokens=400),
    "f8_short_rp": dict(frames=8, side=448, prompts=["caption_short.md"], rp=1.15, tokens=400),
    "f8_two_step": dict(frames=8, side=448, prompts=["caption_entities.md", "caption_timeline.md"], rp=1.0, tokens=250),
    # round 2: build on the two-step that worked
    "two_rp_f8": dict(frames=8, side=448, prompts=["caption_entities.md", "caption_timeline.md"], rp=1.15, tokens=250),
    "two_rp_f4": dict(frames=4, side=448, prompts=["caption_entities.md", "caption_timeline.md"], rp=1.15, tokens=250),
    "two_rp_f16": dict(frames=16, side=448, prompts=["caption_entities.md", "caption_timeline.md"], rp=1.15, tokens=250),
    "two_rp_f16_small": dict(frames=16, side=256, prompts=["caption_entities.md", "caption_timeline.md"], rp=1.15, tokens=250),
}


def run(args: Any) -> None:
    from d4rt_agent.agent_v5.run import dynbench_entries, open_clip, select
    from d4rt_agent.simple_v2 import OfflineQwen

    entries = select(dynbench_entries(args), args)
    seen: dict[str, Any] = {}
    for e in entries:
        seen.setdefault(f"{e['dataset']}/{e['video']}", e)
    chosen = list(seen.values())[::2][: args.clips]
    cache = args.cache_dir or (args.results_dir / "cache")
    qwen = OfflineQwen(args.caption_model, max_new_tokens=700, seed=42)
    out_root = args.results_dir / "caption_exp"
    for entry in chosen:
        clip = open_clip(entry, args, cache)
        for name, v in VARIANTS.items():
            target = out_root / name / f"{clip.slug}.json"
            if target.exists():
                continue
            started = time.time()
            idx = pick_frames(clip, v["frames"])
            texts, n_in, n_out, finished = [], 0, 0, True
            for prompt in v["prompts"]:
                text, a, b = generate(qwen, messages(clip, idx, v["side"], prompt), v["tokens"], v["rp"])
                texts.append(text)
                n_in, n_out, finished = n_in + a, n_out + b, finished and b < v["tokens"]
            joined = "\n\nTimeline:\n".join(texts) if len(texts) == 2 else texts[0]
            if len(texts) == 2:
                joined = "Entities:\n" + joined
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"slug": clip.slug, "variant": name, "text": joined, "input_tokens": n_in,
                                          "output_tokens": n_out, "stopped_by_itself": finished,
                                          "seconds": round(time.time() - started, 1)}))
            print(f"{clip.slug} {name}: in {n_in} out {n_out} stopped={finished} {time.time() - started:.1f}s", flush=True)


_STOP = {"the", "a", "an", "of", "in", "on", "with", "and", "at", "to"}


def heads(phrase: str) -> set[str]:
    from d4rt_agent.agent_v5.program import singular

    return {singular(w) for w in re.findall(r"[a-z]+", phrase.lower()) if w not in _STOP}


def score(args: Any) -> None:
    from dyn_dataset_exploration.dyn_bench_data import video_slug

    catalog = json.loads(Path(args.catalog).read_text())["questions"]
    slug_of = {q["qid"]: video_slug(q["dataset"], q["video"]) for q in catalog}
    v5 = defaultdict(list)       # slug -> subject phrases V5's Qwen named on that clip's questions
    for r in json.loads(Path(args.v5_table).read_text()):
        for k, val in (r.get("slots") or {}).items():
            if k in ("subject", "reference", "landmark", "category") and isinstance(val, str) and val != "camera":
                v5[slug_of[r["qid"]]].append(val)
    root = Path(args.results_dir) / "caption_exp"
    print(f"{'variant':18s} {'clips':>5s} {'ents':>5s} {'ent>0':>6s} {'spans':>5s} {'rep%':>5s} {'stop%':>6s} {'in_tok':>7s} {'out_tok':>7s} {'cover%':>7s}")
    for vdir in sorted(root.iterdir()):
        rows = [json.loads(p.read_text()) for p in vdir.glob("*.json")]
        if not rows:
            continue
        ents = [parse_entities(r["text"]) for r in rows]
        lines = [[l.strip() for l in r["text"].splitlines() if l.strip()] for r in rows]
        rep = [1 - len(set(ls)) / max(len(ls), 1) for ls in lines]
        cover, total = 0, 0
        for r, es in zip(rows, ents):
            cap_words = set().union(*(heads(e["name"]) for e in es)) if es else set()
            for subj in v5.get(r["slug"], []):
                total += 1
                cover += bool(heads(subj) & cap_words)
        n = len(rows)
        print(f"{vdir.name:18s} {n:5d} {sum(map(len, ents)) / n:5.1f} {sum(bool(e) for e in ents) / n:6.0%} "
              f"{sum(len(timeline_spans(r['text'])) for r in rows) / n:5.1f} {100 * sum(rep) / n:5.1f} "
              f"{sum(r['stopped_by_itself'] for r in rows) / n:6.0%} {sum(r['input_tokens'] for r in rows) / n:7.0f} "
              f"{sum(r['output_tokens'] for r in rows) / n:7.0f} {cover / max(total, 1):7.0%}")


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode, rest = argv[0], argv[1:]
    import argparse

    from d4rt_agent.agent_v5_jev.args import parse

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--clips", type=int, default=12)
    extra.add_argument("--v5-table", type=Path, default=None)
    ns, rest = extra.parse_known_args(rest)
    args = parse(rest)
    args.clips, args.v5_table = ns.clips, ns.v5_table
    {"run": run, "score": score}[mode](args)


if __name__ == "__main__":
    main()
