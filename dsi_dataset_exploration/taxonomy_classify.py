"""Blind LLM classification of the taxonomy sample against the nine-bucket seed
taxonomy in ``taxonomy_data.py``.

Each sampled question (text + options only -- no ``cate``, ``GT``, ``others``,
or ``video_type``, the same withholding ``dsi_bench_data.format_question_prompt``
already does for the agent eval, applied here for a different reason: it keeps
this classification a genuine blind test, not a leak) is sent to a small OpenAI
model, which must name the one physical quantity/reference frame it thinks the
question measures, choosing one of the nine described buckets or ``OTHER``.

Calls are synchronous ``chat/completions`` requests, not the Batch API used by
``d4rt_agent/dsi_bench_judge.py`` -- the workload is small (at most 200 calls,
one per sampled question) and the process is meant to be iterative (classify,
review disagreements, refine the taxonomy, reclassify), which the Batch API's
submit/poll/collect ceremony would only slow down.

Reads ``OPENAI_API_KEY`` (and optionally ``OPENAI_BASE_URL``) from the
environment and never prints it. Run as::

    source ~/.bashrc && python -m dsi_dataset_exploration.taxonomy_classify classify
    python -m dsi_dataset_exploration.taxonomy_classify review   # no network
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

from d4rt_agent.dsi_bench_data import format_question_prompt, read_manifest

from .taxonomy_data import BUCKET_KEYS, BUCKETS, CATE_TO_BUCKET_HYPOTHESIS, OTHER_BUCKET

DEFAULT_RESULTS_DIR = Path("dsi_dataset_exploration/logs")
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "bucket_system.md"

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_MAX_COMPLETION_TOKENS = 400
DEFAULT_TIMEOUT = 60
DEFAULT_CONCURRENCY = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Run this command as:\n"
            "  source ~/.bashrc && python -m dsi_dataset_exploration.taxonomy_classify ..."
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

def bucket_glossary() -> str:
    lines = [f"- {key}: {description}" for key, description in BUCKETS.items()]
    lines.append(f"- {OTHER_BUCKET}: none of the above fit")
    return "\n".join(lines)


def bucket_messages(entry: Mapping[str, Any], system_prompt: str) -> list[dict[str, str]]:
    """Build the two-message chat for one question. cate/GT/others never included."""

    user = f"Question:\n{format_question_prompt(entry)}\n\nBuckets:\n{bucket_glossary()}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def bucket_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "measurement_bucket",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "bucket": {"type": "string", "enum": list(BUCKET_KEYS)},
                    "rationale": {"type": "string"},
                },
                "required": ["bucket", "rationale"],
                "additionalProperties": False,
            },
        },
    }


# --------------------------------------------------------------------------- #
# Result parsing (pure, no network)
# --------------------------------------------------------------------------- #

def extract_bucket(content: str) -> tuple[str, str]:
    """Recover ``(bucket, rationale)`` from one reply, defensively.

    Anything that does not resolve to a known bucket key becomes ``OTHER`` --
    the classifier is never allowed to invent a bucket, even if the model
    bypassed strict mode.
    """

    text = (content or "").strip()
    bucket = ""
    rationale = ""
    try:
        obj = json.loads(text)
        if isinstance(obj, Mapping):
            bucket = str(obj.get("bucket", "")).strip()
            rationale = str(obj.get("rationale", "")).strip()
    except (ValueError, TypeError):
        pass

    if bucket in BUCKET_KEYS:
        return bucket, rationale
    return OTHER_BUCKET, rationale


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
    messages = bucket_messages(entry, system_prompt)
    response_format = bucket_response_format()
    started = time.monotonic()
    raw = call_chat_completion(messages, response_format, model=model)
    latency = time.monotonic() - started
    content = raw["choices"][0]["message"]["content"]
    bucket, rationale = extract_bucket(content)
    return {
        "question_id": entry["question_id"],
        "cate": entry["cate"],
        "category_name": entry["category_name"],
        "hypothesis_bucket": entry.get(
            "hypothesis_bucket", CATE_TO_BUCKET_HYPOTHESIS.get(entry["cate"])
        ),
        "model": model,
        "requested_at": _utc_now(),
        "latency_seconds": round(latency, 3),
        "messages": messages,
        "response_format": response_format,
        "raw_response": raw,
        "parsed": {"bucket": bucket, "rationale": rationale},
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
            print(f"  {qid}: {record['parsed']['bucket']}")

    print(f"done: {len(pending) - errors} written, {errors} errors")
    return 1 if errors else 0


def _load_classifications(classify_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not classify_dir.exists():
        return records
    for path in sorted(classify_dir.glob("*.json")):
        if path.stem == "review_queue":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(record["question_id"])] = record
    return records


def review(args: argparse.Namespace) -> int:
    entries = {str(e["question_id"]): e for e in _load_manifest_entries(args.results_dir)}
    classify_dir = _classify_dir(args.results_dir)
    classifications = _load_classifications(classify_dir)
    if not classifications:
        raise SystemExit(f"no classifications found in {classify_dir} -- run `classify` first")

    queue: list[dict[str, Any]] = []
    bucket_frequency: dict[str, int] = {key: 0 for key in BUCKET_KEYS}
    for qid, record in classifications.items():
        llm_bucket = record["parsed"]["bucket"]
        bucket_frequency[llm_bucket] = bucket_frequency.get(llm_bucket, 0) + 1
        hypothesis_bucket = record.get("hypothesis_bucket")
        if llm_bucket == hypothesis_bucket:
            continue
        entry = entries.get(qid, {})
        queue.append(
            {
                "question_id": qid,
                "video_slug": entry.get("video_slug"),
                "question": entry.get("question"),
                "options": entry.get("options"),
                "cate": record.get("cate"),
                "category_name": record.get("category_name"),
                "hypothesis_bucket": hypothesis_bucket,
                "llm_bucket": llm_bucket,
                "llm_rationale": record["parsed"]["rationale"],
                "human_verdict": None,
                "human_notes": "",
            }
        )

    queue.sort(key=lambda item: (item["cate"], item["question_id"]))
    payload = {
        "generated_at": _utc_now(),
        "total_classified": len(classifications),
        "total_disagreements": len(queue),
        "bucket_frequency": bucket_frequency,
        "queue": queue,
    }
    out_path = classify_dir / "review_queue.json"
    _write_atomic(out_path, payload)
    print(f"{len(classifications)} classified, {len(queue)} disagreements -> {out_path}")
    for bucket, count in bucket_frequency.items():
        print(f"  {bucket}: {count}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p_classify = sub.add_parser(
        "classify", help="blind-classify the sample against the nine buckets"
    )
    p_classify.add_argument("--model", default=os.environ.get("TAXONOMY_MODEL", DEFAULT_MODEL))
    p_classify.add_argument(
        "--limit", type=int, default=None, help="classify only the first N questions (smoke test)"
    )
    p_classify.add_argument("--only", nargs="*", default=None, help="classify only these question_ids")
    p_classify.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p_classify.add_argument("--force", action="store_true", help="reclassify even if already written")
    p_classify.set_defaults(func=classify)

    p_review = sub.add_parser("review", help="build the disagreement review queue (no network)")
    p_review.set_defaults(func=review)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
