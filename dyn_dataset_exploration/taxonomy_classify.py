"""Blind dual-axis LLM classification of the Dyn-Bench taxonomy sample.

Each sampled question (text + options only -- never ``task``, ``dataset``,
``answer``, or the regex family label, all of which would hand over one or both
answers) is sent to a small OpenAI-compatible model, which labels it twice in one
call:

* ``paper_task`` -- one of the nine sub-tasks named in arXiv 2603.12746. The
  paper defines them but the released parquet does not ship them, so recovering
  them is the only way to compare our findings to the published tables.
* ``bucket`` -- one of the measurement buckets in ``buckets.py``: what physical
  quantity, in what reference frame, would actually decide the question.

Both labels come from one call rather than two, for the same reason the DSI pass
used synchronous ``chat/completions`` rather than the Batch API used by
``d4rt_agent/dsi_bench_judge.py``: the workload is small (600 calls) and the
process is meant to be iterative -- classify, review disagreements, refine the
bucket list, reclassify -- which the Batch API's submit/poll/collect ceremony
would only slow down. Keeping the two axes in one call also lets each label
inform the other, which is what a human coder would do.

Reads ``OPENAI_API_KEY`` (and optionally ``OPENAI_BASE_URL``) from the
environment and never prints it. Run as::

    source ~/.bashrc && python -m dyn_dataset_exploration.taxonomy_classify classify
    python -m dyn_dataset_exploration.taxonomy_classify review   # no network
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import requests

from .buckets import BUCKET_KEYS, OTHER_BUCKET, V4_BUCKETS, bucket_glossary
from .dyn_bench_data import (
    DEFAULT_RESULTS_DIR,
    LEVEL_OF_PAPER_TASK,
    PAPER_TASKS,
    TASK_TO_LEVEL,
    format_question_prompt,
    read_manifest,
)

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dual_axis_system.md"

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_MAX_COMPLETION_TOKENS = 600
DEFAULT_TIMEOUT = 90
DEFAULT_CONCURRENCY = 8

PAPER_TASK_KEYS: tuple[str, ...] = (*PAPER_TASKS, OTHER_BUCKET)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Run this command as:\n"
            "  source ~/.bashrc && python -m dyn_dataset_exploration.taxonomy_classify ..."
        )
    return key


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}"}


def _write_atomic(path: Path, payload: Any) -> None:
    """Write JSON so an interrupted command never leaves a half-parsed file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _classify_dir(results_dir: Path) -> Path:
    return Path(results_dir) / "classify"


# --------------------------------------------------------------------------- #
# Request construction (pure, no network -- exercised by the offline tests)
# --------------------------------------------------------------------------- #

def paper_task_glossary() -> str:
    lines: list[str] = []
    for name in PAPER_TASKS:
        lines.append(f"- {name}  (level: {LEVEL_OF_PAPER_TASK[name]})")
    lines.append(f"- {OTHER_BUCKET}: none of the above fit")
    return "\n".join(lines)


def classify_messages(entry: Mapping[str, Any], system_prompt: str) -> list[dict[str, str]]:
    """Build the two-message chat for one question.

    The only fields of ``entry`` that reach the model are ``question`` and
    ``options``, via ``format_question_prompt``. ``test_taxonomy_classify.py``
    asserts that directly, because the blindness of this pass is the thing that
    makes its counts worth anything.
    """

    user = (
        f"Question:\n{format_question_prompt(entry)}\n\n"
        f"Axis 1 -- paper_task, choose one:\n{paper_task_glossary()}\n\n"
        f"Axis 2 -- bucket, choose one:\n{bucket_glossary()}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def classify_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dyn_bench_question_labels",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "paper_task": {"type": "string", "enum": list(PAPER_TASK_KEYS)},
                    "paper_rationale": {"type": "string"},
                    "bucket": {"type": "string", "enum": list(BUCKET_KEYS)},
                    "bucket_rationale": {"type": "string"},
                },
                "required": ["paper_task", "paper_rationale", "bucket", "bucket_rationale"],
                "additionalProperties": False,
            },
        },
    }


# --------------------------------------------------------------------------- #
# Result parsing (pure, no network)
# --------------------------------------------------------------------------- #

def extract_labels(content: str) -> dict[str, str]:
    """Recover both labels from one reply, defensively.

    Anything that does not resolve to a known key becomes ``OTHER`` -- the
    classifier is never allowed to invent a label, even if the model bypassed
    strict mode. ``OTHER`` on both axes with empty rationales is how a malformed
    reply shows up in the counts, which is the honest place for it.
    """

    text = (content or "").strip()
    paper_task = bucket = ""
    paper_rationale = bucket_rationale = ""
    try:
        obj = json.loads(text)
        if isinstance(obj, Mapping):
            paper_task = str(obj.get("paper_task", "")).strip()
            bucket = str(obj.get("bucket", "")).strip()
            paper_rationale = str(obj.get("paper_rationale", "")).strip()
            bucket_rationale = str(obj.get("bucket_rationale", "")).strip()
    except (ValueError, TypeError):
        pass

    return {
        "paper_task": paper_task if paper_task in PAPER_TASK_KEYS else OTHER_BUCKET,
        "paper_rationale": paper_rationale,
        "bucket": bucket if bucket in BUCKET_KEYS else OTHER_BUCKET,
        "bucket_rationale": bucket_rationale,
    }


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #

def call_chat_completion(
    messages: list[dict[str, str]],
    response_format: Mapping[str, Any],
    *,
    model: str,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
) -> dict[str, Any]:
    """POST one chat/completions request, retrying transient 5xx errors."""

    url = f"{_base_url()}/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "response_format": dict(response_format),
        "max_completion_tokens": max_completion_tokens,
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=_auth_headers(), json=body, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2**attempt)
            continue
        if response.status_code >= 500:
            last_error = RuntimeError(f"http_{response.status_code}: {response.text[:300]}")
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"chat/completions failed after {max_retries} attempts: {last_error}")


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def _load_manifest_entries(results_dir: Path) -> list[dict[str, Any]]:
    manifest = read_manifest(Path(results_dir) / "manifest.json")
    return manifest["questions"]


def _select_entries(
    entries: Sequence[Mapping[str, Any]], limit: int | None, only: Sequence[str] | None
) -> list[Mapping[str, Any]]:
    selected = list(entries)
    if only:
        wanted = set(only)
        selected = [e for e in selected if str(e["question_id"]) in wanted]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _already_done(classify_dir: Path) -> set[str]:
    if not classify_dir.exists():
        return set()
    return {path.stem for path in classify_dir.glob("*.json") if path.stem != "review_queue"}


def _classify_one(entry: Mapping[str, Any], *, system_prompt: str, model: str) -> dict[str, Any]:
    messages = classify_messages(entry, system_prompt)
    response_format = classify_response_format()
    started = time.monotonic()
    raw = call_chat_completion(messages, response_format, model=model)
    latency = time.monotonic() - started
    parsed = extract_labels(raw["choices"][0]["message"]["content"])
    return {
        "question_id": entry["question_id"],
        # Scoring-side facts, recorded next to the labels but never sent: the
        # coarse task and family are what the labels get compared against.
        "task": entry["task"],
        "dataset": entry["dataset"],
        "family": entry["family"],
        "families": entry.get("families", []),
        "hypothesis_level": entry.get("hypothesis_level", TASK_TO_LEVEL.get(entry["task"])),
        "model": model,
        "requested_at": _utc_now(),
        "latency_seconds": round(latency, 3),
        "messages": messages,
        "response_format": response_format,
        "raw_response": raw,
        "parsed": parsed,
    }


def classify(args: argparse.Namespace) -> int:
    entries = _load_manifest_entries(args.results_dir)
    entries = _select_entries(entries, args.limit, args.only)
    classify_dir = _classify_dir(args.results_dir)
    classify_dir.mkdir(parents=True, exist_ok=True)

    done = set() if args.force else _already_done(classify_dir)
    pending = [e for e in entries if str(e["question_id"]) not in done]
    print(
        f"{len(entries)} selected, {len(pending)} pending "
        f"({len(entries) - len(pending)} already done)"
    )
    if not pending:
        return 0

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    errors = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _classify_one, entry, system_prompt=system_prompt, model=args.model
            ): entry
            for entry in pending
        }
        for future in as_completed(futures):
            entry = futures[future]
            qid = str(entry["question_id"])
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 -- report and continue, never abort the batch
                errors += 1
                print(f"  ERROR {qid}: {exc}")
                continue
            _write_atomic(classify_dir / f"{qid}.json", record)
            print(f"  {qid}: {record['parsed']['bucket']} / {record['parsed']['paper_task']}")

    print(f"done: {len(pending) - errors} written, {errors} errors")
    return 1 if errors else 0


def load_classifications(classify_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not Path(classify_dir).exists():
        return records
    for path in sorted(Path(classify_dir).glob("*.json")):
        if path.stem == "review_queue":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(record["question_id"])] = record
    return records


def review_reasons(record: Mapping[str, Any]) -> list[str]:
    """Why one classification wants a human look.

    Three reasons, and they are not the same kind of thing. A level mismatch
    tests the paper-taxonomy hypothesis. ``OTHER`` says the answer space is
    incomplete. A *new* bucket -- one AgentV4 does not implement -- is the only
    kind of row that can create work for AgentV5, so every one of those is worth
    reading even when nothing looks wrong with it.
    """

    parsed = record["parsed"]
    reasons: list[str] = []
    level = LEVEL_OF_PAPER_TASK.get(parsed["paper_task"])
    if level is not None and level != record.get("hypothesis_level"):
        reasons.append(f"level mismatch: {record.get('hypothesis_level')} -> {level}")
    if parsed["paper_task"] == OTHER_BUCKET:
        reasons.append("paper_task is OTHER")
    if parsed["bucket"] == OTHER_BUCKET:
        reasons.append("bucket is OTHER")
    elif parsed["bucket"] not in V4_BUCKETS:
        reasons.append(f"candidate bucket not implemented in AgentV4: {parsed['bucket']}")
    return reasons


def review(args: argparse.Namespace) -> int:
    entries = {str(e["question_id"]): e for e in _load_manifest_entries(args.results_dir)}
    classify_dir = _classify_dir(args.results_dir)
    classifications = load_classifications(classify_dir)
    if not classifications:
        raise SystemExit(f"no classifications found in {classify_dir} -- run `classify` first")

    queue: list[dict[str, Any]] = []
    for qid, record in sorted(classifications.items()):
        reasons = review_reasons(record)
        if not reasons:
            continue
        entry = entries.get(qid, {})
        parsed = record["parsed"]
        queue.append(
            {
                "question_id": qid,
                "task": record.get("task"),
                "family": record.get("family"),
                "video_key": entry.get("video_key"),
                "question": entry.get("question"),
                "options": entry.get("options"),
                "hypothesis_level": record.get("hypothesis_level"),
                "llm_paper_task": parsed["paper_task"],
                "llm_paper_rationale": parsed["paper_rationale"],
                "llm_bucket": parsed["bucket"],
                "llm_bucket_rationale": parsed["bucket_rationale"],
                "review_reasons": reasons,
                "human_verdict": None,
                "human_notes": "",
            }
        )

    queue.sort(key=lambda item: (item["task"] or "", item["llm_bucket"], item["question_id"]))
    payload = {
        "generated_at": _utc_now(),
        "total_classified": len(classifications),
        "total_queued": len(queue),
        "queued_by_reason": _count_reasons(queue),
        "queue": queue,
    }
    out_path = classify_dir / "review_queue.json"
    _write_atomic(out_path, payload)
    print(f"{len(classifications)} classified, {len(queue)} queued for review -> {out_path}")
    for reason, count in payload["queued_by_reason"].items():
        print(f"  {count:4d}  {reason}")
    return 0


def _count_reasons(queue: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in queue:
        for reason in item["review_reasons"]:
            key = reason.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p_classify = sub.add_parser(
        "classify", help="blind-classify the sample on both taxonomy axes"
    )
    p_classify.add_argument("--model", default=os.environ.get("TAXONOMY_MODEL", DEFAULT_MODEL))
    p_classify.add_argument(
        "--limit", type=int, default=None, help="classify only the first N questions (smoke test)"
    )
    p_classify.add_argument(
        "--only", nargs="*", default=None, help="classify only these question_ids"
    )
    p_classify.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p_classify.add_argument(
        "--force", action="store_true", help="reclassify even if already written"
    )
    p_classify.set_defaults(func=classify)

    p_review = sub.add_parser("review", help="build the review queue (no network)")
    p_review.set_defaults(func=review)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
