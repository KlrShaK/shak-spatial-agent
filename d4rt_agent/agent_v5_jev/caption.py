"""The caption stage: Qwen3-VL-2B describes a clip's 16 answer frames once, as text.

The Decisions model never sees pixels; this text is its only view of the video. Output per clip:
``<captions-dir>/<slug>.json`` with the raw text and the parsed entity list (name, first and last
seen, behaviour). Questions are not shown to the captioner, so one caption serves all of a clip's
questions and cannot leak anything about them.

    python -m d4rt_agent.agent_v5_jev.caption --results-dir <run> --captions-dir <dir> --caption-model <path> [--shard i/n]
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

PROMPTS = Path(__file__).resolve().parent / "prompts"
CAPTION_LONG_SIDE = 448
CAPTION_TOKENS = 700


def caption_messages(clip: Any) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index in clip.answer_frames:
        content.append({"type": "text", "text": clip.label(index) + ":"})
        content.append({"type": "image", "image": clip.pil(index, CAPTION_LONG_SIDE)})
    content.append({"type": "text", "text": (PROMPTS / "caption.md").read_text(encoding="utf-8")})
    return [{"role": "user", "content": content}]


_ENTITY = re.compile(r"^\s*[-*]\s*(?P<name>[^|]+?)\s*(?:\|(?P<rest>.*))?$")
_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*s?")


def parse_entities(text: str) -> list[dict[str, Any]]:
    """Entities from the caption's ``Entities:`` list; tolerant of a missing time or behaviour."""

    match = re.search(r"entities\s*:\s*\n?(.*)", text, re.I | re.S)
    if not match:
        return []
    out: list[dict[str, Any]] = []
    for line in match.group(1).splitlines():
        m = _ENTITY.match(line)
        if not m:
            continue
        name = re.sub(r"[`*_]", "", m.group("name")).strip().strip(".")
        if not name or len(name.split()) > 6:
            continue
        parts = [p.strip() for p in (m.group("rest") or "").split("|")]
        times = [float(_TIME.search(p).group(1)) if _TIME.search(p) else None for p in parts[:2]]
        times += [None] * (2 - len(times))
        out.append({"name": name, "first_s": times[0], "last_s": times[1],
                    "behaviour": parts[2] if len(parts) > 2 else (parts[-1] if parts and not _TIME.search(parts[-1]) else "")})
    return out


def timeline_spans(text: str) -> list[tuple[float, float]]:
    """The ``<start>-<end> s`` spans of the Timeline lines (candidate time windows)."""

    spans = []
    for m in re.finditer(r"^\s*[-*]\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*s", text, re.M):
        a, b = float(m.group(1)), float(m.group(2))
        if b > a:
            spans.append((a, b))
    return spans


def caption_path(captions_dir: Path, slug: str) -> Path:
    return Path(captions_dir) / f"{slug}.json"


def load_caption(captions_dir: Path, slug: str) -> dict[str, Any]:
    return json.loads(caption_path(captions_dir, slug).read_text())


def write_caption(captions_dir: Path, slug: str, text: str, model: str) -> dict[str, Any]:
    record = {"slug": slug, "model": model, "text": text.strip(), "entities": parse_entities(text),
              "spans": timeline_spans(text)}
    path = caption_path(captions_dir, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, path)
    return record


def main(argv: Sequence[str] | None = None) -> None:
    from d4rt_agent.agent_v5_jev.args import parse
    from d4rt_agent.agent_v5.run import dynbench_entries, open_clip, select

    args = parse(argv)
    entries = select(dynbench_entries(args), args)
    cache = args.cache_dir or (args.results_dir / "cache")
    slugs: dict[str, Any] = {}
    for entry in entries:
        key = f"{entry['dataset']}/{entry['video']}"
        slugs.setdefault(key, entry)
    todo = [e for e in slugs.values()
            if not caption_path(args.captions_dir, _slug(e)).exists() or args.force]
    print(f"{len(slugs)} clips, {len(todo)} to caption", flush=True)
    if args.dry_run or not todo:
        return
    from d4rt_agent.simple_v2 import OfflineQwen

    qwen = OfflineQwen(args.caption_model, max_new_tokens=CAPTION_TOKENS, seed=42)
    for n, entry in enumerate(todo, 1):
        started = time.time()
        try:
            clip = open_clip(entry, args, cache)
            text = qwen.generate(caption_messages(clip), max_new_tokens=CAPTION_TOKENS)
            record = write_caption(args.captions_dir, clip.slug, text, str(args.caption_model))
            note = f"{len(record['entities'])} entities, {len(record['spans'])} spans"
        except Exception as error:                      # noqa: BLE001 -- recorded, the run goes on
            note = f"FAILED {type(error).__name__}: {error}"
        print(f"[{n}/{len(todo)}] {_slug(entry)} {note} {time.time() - started:.1f}s", flush=True)


def _slug(entry: Any) -> str:
    from dyn_dataset_exploration.dyn_bench_data import video_slug

    return video_slug(entry["dataset"], entry["video"])


if __name__ == "__main__":
    main()
