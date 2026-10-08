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
# The recipe chosen from the caption experiment (caption_experiment.py, 12 clips): one small question per
# generation (entities, then timeline), 8 frames, repetition penalty. The single long prompt the 2B model
# could not follow (no section headers, endless repeated lines) is gone.
FRAMES = 8
LONG_SIDE = 448
REPETITION_PENALTY = 1.15
STEP_TOKENS = 250
STEP_PROMPTS = ("caption_entities.md", "caption_timeline.md")


def pick_frames(clip: Any, n: int) -> list[int]:
    import numpy as np

    frames = clip.answer_frames
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, n)).astype(int))
    return [frames[i] for i in idx]


def messages(clip: Any, indices: Sequence[int], side: int, prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index in indices:
        content.append({"type": "text", "text": clip.label(index) + ":"})
        content.append({"type": "image", "image": clip.pil(index, side)})
    content.append({"type": "text", "text": (PROMPTS / prompt).read_text(encoding="utf-8")})
    return [{"role": "user", "content": content}]


def generate(qwen: Any, msgs: list[dict[str, Any]], max_new_tokens: int, rp: float) -> tuple[str, int, int]:
    """OfflineQwen.generate with a repetition penalty and token counts: (text, input tokens, output tokens)."""

    qwen.torch.manual_seed(qwen.seed)
    inputs = qwen.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                return_tensors="pt").to(qwen.model.device)
    with qwen.torch.inference_mode():
        out = qwen.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                  repetition_penalty=rp)
    new = out[:, inputs.input_ids.shape[1]:]
    text = qwen.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return text, int(inputs.input_ids.shape[1]), int(new.shape[1])


def caption_clip(qwen: Any, clip: Any, gen=generate) -> str:
    """The caption text of one clip: ``Entities:`` list, then ``Timeline:``."""

    indices = pick_frames(clip, FRAMES)
    entities, timeline = (gen(qwen, messages(clip, indices, LONG_SIDE, p), STEP_TOKENS, REPETITION_PENALTY)[0]
                          for p in STEP_PROMPTS)
    return f"Entities:\n{entities}\n\nTimeline:\n{timeline}"


_ENTITY = re.compile(r"^\s*[-*]\s*(?P<name>[^|]+?)\s*(?:\|(?P<rest>.*))?$")
_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*s?")


def parse_entities(text: str) -> list[dict[str, Any]]:
    """Entities from the caption's ``Entities:`` list (it ends at the ``Timeline:`` header or the
    end); tolerant of a missing time or behaviour."""

    match = re.search(r"entities\s*:\s*\n?(.*?)(?=^\s*\W*timeline\s*:|\Z)", text, re.I | re.S | re.M)
    if not match:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in match.group(1).splitlines():
        m = _ENTITY.match(line)
        if not m:
            continue
        name = re.sub(r"[`*_]", "", m.group("name")).strip().strip(".")
        if not name or len(name.split()) > 6 or name.lower() in seen:
            continue
        seen.add(name.lower())
        parts = [p.strip() for p in (m.group("rest") or "").split("|")]
        times = [float(_TIME.search(p).group(1)) if _TIME.search(p) else None for p in parts[:2]]
        times += [None] * (2 - len(times))
        out.append({"name": name, "first_s": times[0], "last_s": times[1],
                    "behaviour": parts[2] if len(parts) > 2 else (parts[-1] if parts and not _TIME.search(parts[-1]) else "")})
    return out


def timeline_spans(text: str) -> list[tuple[float, float]]:
    """Time spans of the Timeline lines (candidate event windows).

    ``- 0.5-1.5 s: ...`` gives (0.5, 1.5). A small model often writes one timestamp per line
    (``- 0.5 s: ...``); then each line runs to the next one's time."""

    spans = []
    for m in re.finditer(r"^\s*[-*]?\s*(\d+(?:\.\d+)?)\s*(?:-|\u2013|to)\s*(\d+(?:\.\d+)?)\s*s", text, re.M):
        a, b = float(m.group(1)), float(m.group(2))
        if b > a:
            spans.append((a, b))
    if spans:
        return spans
    times = [float(m.group(1)) for m in re.finditer(r"^\s*[-*]\s*(\d+(?:\.\d+)?)\s*s\s*:", text, re.M)]
    return [(a, b) for a, b in zip(times, times[1:]) if b > a]


def caption_path(captions_dir: Path, slug: str) -> Path:
    return Path(captions_dir) / f"{slug}.json"


def load_caption(captions_dir: Path, slug: str) -> dict[str, Any]:
    """The stored caption; entities and spans are re-read from its text, so a parser fix applies to
    captions written earlier."""

    record = json.loads(caption_path(captions_dir, slug).read_text())
    record["entities"], record["spans"] = parse_entities(record["text"]), timeline_spans(record["text"])
    return record


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

    qwen = OfflineQwen(args.caption_model, max_new_tokens=STEP_TOKENS, seed=42)
    for n, entry in enumerate(todo, 1):
        started = time.time()
        try:
            clip = open_clip(entry, args, cache)
            text = caption_clip(qwen, clip)
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
