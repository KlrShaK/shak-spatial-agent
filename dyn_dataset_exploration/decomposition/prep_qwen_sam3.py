"""GPU prep, part 1: Qwen3-VL-8B reads each question, SAM3 grounds what it names.

Runs in ``envs/d4rt_bw_third_party`` with the staged transformers tree, exactly as
``scripts/agent_v4_body.sh`` sets it up (``scripts/run_decomposition_prep_a100.slurm``
does this). Stages, in order::

    python -m dyn_dataset_exploration.decomposition.prep_qwen_sam3 --stages parse name ground answer

- ``parse``: Qwen picks the kind (text only), then fills that kind's slots (frames shown);
  every reply is validated and a rejected one is sent back with the reason. Writes
  ``programs/<qid>.qwen.json`` and ``programs/<qid>.pipeline.json`` -- the program S1
  actually uses: the rule parse where a rule matches, Qwen's otherwise.
- ``name``: an implicit "main subject" is named from the frames alone, once per clip (never
  from sibling questions), and filled into every program that needs it.
- ``ground``: one SAM3 text session per (clip, phrase) over the GF frames, keeping every
  instance: ``clips/<slug>/sam3/<phrase_key>.npz``.
- ``answer``: Qwen answers every question from the 16 sampled frames with no tools -- the
  baseline each leaf has to beat -- and is the leaf-C answer for semantic questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dyn_dataset_exploration.dyn_bench_data import read_manifest, write_manifest
from dyn_dataset_exploration.decomposition.build_catalog import DEFAULT_OUT
from dyn_dataset_exploration.decomposition.program import (
    KIND_BY_NAME,
    KINDS,
    SLOT_SCHEMAS,
    QueryProgram,
    assemble,
    phrase_ladder,
    sam3_phrase,
    validate,
)

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
MAX_ATTEMPTS = 3
QWEN_LONG_SIDE = 640          # 16 full-resolution frames would dominate the context
MASK_LONG_SIDE = 512
SYSTEM = ("You are a careful assistant that decomposes questions about videos. Answer only "
          "what each step asks, and end every reply with exactly one JSON object.")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def first_json(text: str, key: str) -> dict[str, Any]:
    """The first JSON object in ``text`` that carries ``key``."""

    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and key in value:
            return value
    raise ValueError(f"no JSON object with a `{key}` field in the reply")


def phrase_key(phrase: str) -> str:
    """A filesystem-safe key for one SAM3 prompt (hashed: phrases are model-authored)."""

    stem = "".join(ch if ch.isalnum() else "_" for ch in phrase.strip().lower())[:32].strip("_")
    return f"{stem}_{hashlib.sha1(phrase.strip().lower().encode()).hexdigest()[:8]}"


def load_clip(out: Path, slug: str) -> dict[str, Any]:
    return json.loads((out / "clips" / slug / "clip.json").read_text())


def frame_content(out: Path, clip: Mapping[str, Any], indices: Sequence[int]) -> list[dict[str, Any]]:
    """Frames for Qwen, each labelled with its GF index and time."""

    from PIL import Image

    content: list[dict[str, Any]] = []
    for index in indices:
        with Image.open(out / "clips" / clip["clip_slug"] / "frames" / f"{index:06d}.jpg") as image:
            image = image.convert("RGB")
            scale = QWEN_LONG_SIDE / max(image.size)
            if scale < 1:
                image = image.resize((round(image.width * scale), round(image.height * scale)))
        t = clip["t_s"][index] if clip.get("t_s") else None
        label = f"Frame {index}" + (f" (t={t:.2f} s)" if t is not None else "")
        content += [{"type": "text", "text": label + ":"}, {"type": "image", "image": image}]
    return content


class Asker:
    """One validated question to Qwen: reject with the reason, retry, keep the trace."""

    def __init__(self, qwen: Any) -> None:
        self.qwen = qwen

    def ask(self, content: list[dict[str, Any]], key: str,
            check: Callable[[dict[str, Any]], Any], trace: list[dict[str, Any]],
            step: str, max_new_tokens: int = 400) -> Any:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.time()
            raw = self.qwen.generate(messages, max_new_tokens=max_new_tokens)
            record = {"step": step, "attempt": attempt, "raw": raw,
                      "seconds": round(time.time() - started, 2)}
            try:
                value = check(first_json(raw, key))
            except (ValueError, KeyError, TypeError) as error:
                record["rejected"] = str(error)
                trace.append(record)
                messages += [
                    {"role": "assistant", "content": [{"type": "text", "text": raw}]},
                    {"role": "user", "content": [{"type": "text", "text":
                        f"Rejected: {error}. Reply again, ending with one corrected JSON object."}]},
                ]
                continue
            record["accepted"] = True
            trace.append(record)
            return value
        return None


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def _options_block(options: Mapping[str, str]) -> str:
    return "\n".join(f"{letter}. {options[letter]}" for letter in sorted(options))


def stage_parse(out: Path, catalog: list[dict[str, Any]], asker: Asker) -> None:
    kinds = [k for k in KINDS if k.iteration <= 2]
    kinds_text = "\n".join(f"- `{k.name}`: {k.description}" for k in kinds) + \
        "\n- `other`: none of the above"
    kind_prompt = (PROMPTS / "decompose_kind.md").read_text().replace("{kinds}", kinds_text)
    slot_prompt = (PROMPTS / "decompose_slots.md").read_text()

    for entry in catalog:
        qid, question, options = entry["qid"], entry["question"], entry["options"]
        if (out / "programs" / f"{qid}.pipeline.json").exists():
            continue                      # a rule read it, or an earlier run did
        clip = load_clip(out, entry["clip_slug"])
        trace: list[dict[str, Any]] = []
        stem = f"Question: {question}\nOptions:\n{_options_block(options)}\n\n"

        def check_kind(reply: dict[str, Any]) -> str:
            kind = str(reply["kind"]).strip().strip("`")
            if kind != "other" and kind not in {k.name for k in kinds}:
                raise ValueError(f"{kind!r} is not one of the listed kinds")
            return kind

        kind = asker.ask([{"type": "text", "text": stem + kind_prompt}], "kind", check_kind,
                         trace, "kind")
        program: QueryProgram | None = None
        if kind in SLOT_SCHEMAS and SLOT_SCHEMAS[kind]:
            schema = "\n".join(f"- `{name}`: {text}" for name, text in SLOT_SCHEMAS[kind].items())
            text = slot_prompt.replace("{kind}", kind).replace(
                "{kind_description}", KIND_BY_NAME[kind].description).replace("{slots}", schema)
            sampled = clip["sampled16"][::2]

            def check_slots(reply: dict[str, Any]) -> QueryProgram:
                slots = reply["slots"]
                if not isinstance(slots, dict):
                    raise ValueError("`slots` must be a JSON object")
                return assemble(qid, kind, slots, question, options, source="qwen")

            program = asker.ask(frame_content(out, clip, sampled) +
                                [{"type": "text", "text": stem + text}],
                                "slots", check_slots, trace, "slots")
        elif kind in SLOT_SCHEMAS:
            program = assemble(qid, kind, {}, question, options, source="qwen")

        payload = program.to_dict() if program else {
            "qid": qid, "kind": kind, "unsupported": True,
            "reason": "no kind chosen" if kind is None else f"kind {kind!r} has no template"}
        write_manifest(payload, out / "programs" / f"{qid}.qwen.json")
        write_manifest({"qid": qid, "trace": trace}, out / "traces" / f"{qid}.parse.json")

        rules_path = out / "programs" / f"{qid}.rules.json"
        pipeline = json.loads(rules_path.read_text()) if rules_path.exists() else payload
        write_manifest(pipeline, out / "programs" / f"{qid}.pipeline.json")
        print(f"parse {qid}: qwen kind={kind} rules={'yes' if rules_path.exists() else 'no'}", flush=True)


def stage_name(out: Path, catalog: list[dict[str, Any]], asker: Asker) -> None:
    prompt = (PROMPTS / "name_main_subject.md").read_text()
    named: dict[str, str | None] = {}
    for entry in catalog:
        for source in ("pipeline", "qwen"):
            path = out / "programs" / f"{entry['qid']}.{source}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            if data.get("unsupported"):
                continue
            program = QueryProgram.from_dict(data)
            missing = [e for e in program.entities if e.kind == "main_subject" and not e.phrase]
            if not missing:
                continue
            slug = entry["clip_slug"]
            if slug not in named:
                clip = load_clip(out, slug)
                trace: list[dict[str, Any]] = []

                def check(reply: dict[str, Any]) -> str:
                    phrase, _ = sam3_phrase(str(reply["subject"]))
                    if not phrase or phrase in {"camera", "the camera"}:
                        raise ValueError("name a thing in the scene, not the camera")
                    return phrase

                named[slug] = asker.ask(frame_content(out, clip, clip["sampled16"]) +
                                        [{"type": "text", "text": prompt}], "subject", check,
                                        trace, "main_subject")
                write_manifest({"clip_slug": slug, "subject": named[slug], "trace": trace},
                               out / "traces" / f"{slug}.main_subject.json")
            if named[slug] is None:
                continue
            for entity in missing:
                entity.phrase = named[slug]
                entity.note = "named by Qwen from the frames"
            program.open_slots = [s for s in program.open_slots
                                  if s.split(".", 1)[0] not in {e.id for e in missing}]
            validate(program)
            write_manifest(program.to_dict(), path)
            print(f"name {entry['qid']} ({source}): {named[slug]!r}", flush=True)


def _grounding_phrases(out: Path, catalog: list[dict[str, Any]]) -> dict[str, set[str]]:
    phrases: dict[str, set[str]] = {}
    for entry in catalog:
        for source in ("pipeline", "qwen", "reference"):
            path = out / "programs" / f"{entry['qid']}.{source}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for entity in data.get("entities", []):
                if entity["kind"] != "camera" and entity.get("phrase"):
                    phrases.setdefault(entry["clip_slug"], set()).add(entity["phrase"])
    return phrases


def ground_phrase(predictor: Any, frames_dir: Path, phrase: str, n_frames: int,
                  source_hw: Sequence[int]) -> dict[str, Any]:
    """Every instance SAM3 finds for ``phrase``, tracked across the GF frames."""

    import cv2
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import _to_numpy, bf16_autocast

    per_frame: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
    with bf16_autocast():
        session = predictor.handle_request({"type": "start_session", "resource_path": str(frames_dir)})
        sid = session["session_id"]
        predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                  "text": phrase})
        for response in predictor.handle_stream_request(
                {"type": "propagate_in_video", "session_id": sid}):
            outputs = response["outputs"]
            ids = np.asarray(_to_numpy(outputs["out_obj_ids"])).reshape(-1)
            probs = np.asarray(_to_numpy(outputs["out_probs"])).reshape(-1)
            masks = np.asarray(_to_numpy(outputs["out_binary_masks"]))
            per_frame[int(response["frame_index"])] = {
                int(obj): (np.asarray(masks[k]).squeeze().astype(bool), float(probs[k]))
                for k, obj in enumerate(ids)
            }
        predictor.handle_request({"type": "close_session", "session_id": sid})

    height, width = int(source_hw[0]), int(source_hw[1])
    scale = MASK_LONG_SIDE / max(height, width)
    mh, mw = max(1, round(height * scale)), max(1, round(width * scale))
    obj_ids = sorted({obj for frame in per_frame.values() for obj in frame})
    masks = np.zeros((len(obj_ids), n_frames, mh, mw), dtype=bool)
    probs = np.full((len(obj_ids), n_frames), np.nan, dtype=np.float32)
    boxes = np.full((len(obj_ids), n_frames, 4), np.nan, dtype=np.float32)
    area = np.zeros((len(obj_ids), n_frames), dtype=np.float32)
    border = np.zeros((len(obj_ids), n_frames), dtype=bool)
    for t, frame in per_frame.items():
        if not 0 <= t < n_frames:
            continue
        for k, obj in enumerate(obj_ids):
            if obj not in frame:
                continue
            mask, prob = frame[obj]
            small = cv2.resize(mask.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
            if not small.any():
                continue
            masks[k, t], probs[k, t] = small, prob
            ys, xs = np.nonzero(mask)
            mask_h, mask_w = mask.shape
            boxes[k, t] = (xs.min() / mask_w, ys.min() / mask_h, (xs.max() + 1) / mask_w, (ys.max() + 1) / mask_h)
            area[k, t] = mask.mean()
            border[k, t] = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return {"obj_ids": np.asarray(obj_ids, dtype=np.int64), "masks": masks, "probs": probs,
            "boxes": boxes, "area_frac": area, "border": border}


def _ground_one(predictor: Any, out: Path, slug: str, clip: Mapping[str, Any], phrase: str) -> int:
    """SAM3 on one (clip, phrase), unless already done; the number of instances found."""

    frames_dir = out / "clips" / slug / "frames"
    target = out / "clips" / slug / "sam3" / f"{phrase_key(phrase)}.npz"
    if target.exists():
        return int(json.loads(target.with_suffix(".json").read_text())["instances"])
    started = time.time()
    result = ground_phrase(predictor, frames_dir, phrase, clip["gf"]["length"], clip["source_hw"])
    target.parent.mkdir(parents=True, exist_ok=True)
    masks = result.pop("masks")
    np.savez_compressed(target, masks_packed=np.packbits(masks, axis=-1),
                        masks_shape=np.asarray(masks.shape), **result)
    present = (~np.isnan(result["probs"])).sum(axis=1)
    write_manifest({
        "clip_slug": slug, "phrase": phrase, "prompt_frame": 0,
        "instances": len(result["obj_ids"]),
        "frames_present": {int(o): int(n) for o, n in zip(result["obj_ids"], present)},
        "seconds": round(time.time() - started, 2),
    }, target.with_suffix(".json"))
    print(f"ground {slug} {phrase!r}: {len(result['obj_ids'])} instances "
          f"({time.time() - started:.1f}s)", flush=True)
    return len(result["obj_ids"])


def stage_ground(out: Path, catalog: list[dict[str, Any]]) -> None:
    """S2: SAM3 on every phrase. A phrase with no instance climbs the phrase ladder (the noun
    phrase before its first preposition or participle, then the head noun) until one finds some;
    the pipeline reads the first rung that has instances."""

    import torch
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(gpus_to_use=range(max(torch.cuda.device_count(), 1)))
    for slug, phrases in sorted(_grounding_phrases(out, catalog).items()):
        clip = load_clip(out, slug)
        for phrase in sorted(phrases):
            if _ground_one(predictor, out, slug, clip, phrase) > 0:
                continue
            for rung in phrase_ladder(phrase)[1:]:
                if _ground_one(predictor, out, slug, clip, rung) > 0:
                    print(f"ground {slug} {phrase!r}: found with the simpler prompt {rung!r}", flush=True)
                    break


def stage_answer(out: Path, catalog: list[dict[str, Any]], asker: Asker) -> None:
    template = (PROMPTS / "answer_from_frames.md").read_text()
    for entry in catalog:
        if (out / "answers" / f"{entry['qid']}.frames_only.json").exists():
            continue
        clip = load_clip(out, entry["clip_slug"])
        trace: list[dict[str, Any]] = []
        letters = sorted(entry["options"])

        def check(reply: dict[str, Any]) -> str:
            letter = str(reply["answer"]).strip().upper()[:1]
            if letter not in letters:
                raise ValueError(f"answer must be one of {letters}")
            return letter

        text = template.replace("{question}", entry["question"]).replace(
            "{options}", _options_block(entry["options"]))
        letter = asker.ask(frame_content(out, clip, clip["sampled16"]) +
                           [{"type": "text", "text": text}], "answer", check, trace, "frames_only")
        write_manifest({"qid": entry["qid"], "mode": "frames_only", "answer": letter,
                        "trace": trace}, out / "answers" / f"{entry['qid']}.frames_only.json")
        print(f"answer {entry['qid']}: {letter}", flush=True)


def letter_logprobs(qwen: Any, messages: list[dict[str, Any]], letters: Sequence[str]) -> dict[str, float]:
    """Qwen's probability of each option letter as the first token of its reply (log, renormalised
    over the letters): the model as one soft expert rather than one hard answer."""

    torch = qwen.torch
    inputs = qwen.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(qwen.model.device)
    tokenizer = qwen.processor.tokenizer
    ids = []
    for letter in letters:
        pieces = tokenizer.encode(letter, add_special_tokens=False)
        if len(pieces) != 1:
            raise ValueError(f"option letter {letter!r} is not a single token: {pieces}")
        ids.append(pieces[0])
    with torch.inference_mode():
        logits = qwen.model(**inputs).logits[0, -1].float()
    logp = torch.log_softmax(logits[ids], dim=0)
    return {letter: float(v) for letter, v in zip(letters, logp)}


def stage_letters(out: Path, catalog: list[dict[str, Any]], qwen: Any) -> None:
    """Every question: the model's soft answer (letter log-probabilities) from the 16 frames alone.
    The fusion leaf multiplies it with the measurement's support."""

    template = (PROMPTS / "answer_letter_only.md").read_text()
    system = "You answer multiple-choice questions about videos. Reply with only the option letter."
    for entry in catalog:
        target = out / "answers" / f"{entry['qid']}.letters.json"
        if target.exists():
            continue
        clip = load_clip(out, entry["clip_slug"])
        letters = sorted(entry["options"])
        text = template.replace("{question}", entry["question"]).replace("{options}", _options_block(entry["options"]))
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": frame_content(out, clip, clip["sampled16"]) + [{"type": "text", "text": text}]}]
        logprobs = letter_logprobs(qwen, messages, letters)
        write_manifest({"qid": entry["qid"], "mode": "letters", "logprobs": logprobs,
                        "top": max(logprobs, key=logprobs.get)}, target)
        print(f"letters {entry['qid']}: {max(logprobs, key=logprobs.get)}", flush=True)


def stage_evidence(out: Path, catalog: list[dict[str, Any]], asker: Asker) -> None:
    """Leaf B: Qwen sees the frames AND the measured evidence (``results/<qid>.evidence.txt``,
    written by the CPU pipeline), and answers."""

    template = (PROMPTS / "answer_with_evidence.md").read_text()
    for entry in catalog:
        path = out / "results" / f"{entry['qid']}.evidence.txt"
        if not path.exists():
            continue
        target = out / "answers" / f"{entry['qid']}.evidence.json"
        if target.exists() and json.loads(target.read_text()).get("evidence") == path.read_text():
            continue                      # the same evidence was already answered
        clip = load_clip(out, entry["clip_slug"])
        trace: list[dict[str, Any]] = []
        letters = sorted(entry["options"])

        def check(reply: dict[str, Any]) -> str:
            letter = str(reply["answer"]).strip().upper()[:1]
            if letter not in letters:
                raise ValueError(f"answer must be one of {letters}")
            return letter

        text = template.replace("{question}", entry["question"]).replace(
            "{options}", _options_block(entry["options"])).replace("{evidence}", path.read_text())
        letter = asker.ask(frame_content(out, clip, clip["sampled16"]) +
                           [{"type": "text", "text": text}], "answer", check, trace, "with_evidence")
        write_manifest({"qid": entry["qid"], "mode": "evidence", "answer": letter, "trace": trace,
                        "evidence": path.read_text()}, target)
        print(f"evidence {entry['qid']}: {letter}", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stages", nargs="+", default=["parse", "name", "ground", "answer"],
                        choices=["parse", "name", "ground", "answer", "evidence", "letters"])
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--only", nargs="*", default=None, help="question ids")
    parser.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th clip")
    args = parser.parse_args(argv)

    catalog = read_manifest(args.out / "catalog.json")["questions"]
    if args.only:
        catalog = [e for e in catalog if e["qid"] in set(args.only)]
    index, count = (int(v) for v in args.shard.split("/"))
    slugs = sorted({e["clip_slug"] for e in catalog})
    mine = set(slugs[index::count])
    catalog = [e for e in catalog if e["clip_slug"] in mine]
    asker = qwen = None
    if {"parse", "name", "answer", "evidence", "letters"} & set(args.stages):
        from d4rt_agent.simple_v2 import OfflineQwen
        qwen = OfflineQwen(args.qwen_model, max_new_tokens=400)
        asker = Asker(qwen)
    for stage in args.stages:
        started = time.time()
        if stage == "parse":
            stage_parse(args.out, catalog, asker)
        elif stage == "name":
            stage_name(args.out, catalog, asker)
        elif stage == "ground":
            stage_ground(args.out, catalog)
        elif stage == "answer":
            stage_answer(args.out, catalog, asker)
        elif stage == "evidence":
            stage_evidence(args.out, catalog, asker)
        elif stage == "letters":
            stage_letters(args.out, catalog, qwen)
        print(f"stage {stage} done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
