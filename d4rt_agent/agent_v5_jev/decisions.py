"""Client for OpenAI's Decisions API (`POST /v1/decisions`, text in, typed answers out).

Every call goes through a disk cache keyed by the request body, which also makes the flow work
where the API cannot be reached (compute nodes without a proxy): in `collect` mode `decide`
writes the missing request to `<cache>/pending/` and raises `NeedRequest`; `send_pending`, run
where the network works, fills the cache; the rerun then finds every answer cached.

Requests carry only text built by the caller. This module never sees a catalog row.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_URL = "https://api.openai.com/v1/decisions"
RETRY_STATUS = (429, 500, 502, 503, 504)
_TMP_COUNTER = itertools.count()
REFUSAL_RETRIES = 2


class NeedRequest(RuntimeError):
    """`collect` mode: the request was written to pending/ and has no cached answer yet."""


class Refused(RuntimeError):
    """The model returned a refusal for a question."""


def choice_question(name: str, instructions: str, choices: Sequence[tuple[str, str]]) -> dict:
    """A `choice` question; `choices` is [(value, description), ...]."""

    values = [value for value, _ in choices]
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate choice values in question {name!r}: {values}")
    return {"type": "choice", "name": name, "instructions": instructions,
            "choices": [{"value": value, "description": description}
                        for value, description in choices]}


def request_body(text: str, questions: Sequence[Mapping[str, Any]], model: str = DEFAULT_MODEL) -> dict:
    names = [q["name"] for q in questions]
    if len(set(names)) != len(names):
        raise ValueError(f"question names must be unique: {names}")
    return {"model": model, "input": text, "questions": list(questions)}


def request_key(body: Mapping[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}.{threading.get_ident()}.{next(_TMP_COUNTER)}")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def post(body: Mapping[str, Any], *, url: str = DEFAULT_URL, timeout: float = 120.0,
         attempts: int = 5) -> dict:
    """POST one request; retry rate limits and server errors with backoff."""

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(attempts):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in RETRY_STATUS or attempt == attempts - 1:
                raise RuntimeError(f"decisions HTTP {response.status_code}: {response.text[:300]}")
        time.sleep(min(30.0, 2.0 ** attempt))
    raise RuntimeError("unreachable")


def parse_answers(response: Mapping[str, Any], questions: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    """{question name: {"choice", "probs": {value: p}, "confidence"}}; raises Refused."""

    answers = {a["name"]: a for a in response["answers"]}
    parsed: dict[str, dict] = {}
    for question in questions:
        answer = answers.get(question["name"])
        if answer is None:
            raise KeyError(f"no answer for question {question['name']!r}")
        if answer["type"] == "refusal":
            raise Refused(f"question {question['name']!r} refused")
        if answer["type"] != "choice":
            raise ValueError(f"question {question['name']!r}: expected a choice answer, got {answer['type']}")
        parsed[question["name"]] = {
            "choice": answer["choice"],
            "probs": {p["value"]: p["probability"] for p in answer["probabilities"]},
            "confidence": answer.get("confidence"),
        }
    return parsed


def decide(text: str, questions: Sequence[Mapping[str, Any]], cache: Path, *,
           mode: str = "live", model: str = DEFAULT_MODEL, poster=post) -> dict[str, dict]:
    """Answer `questions` about `text`; cached by request. mode: live | collect | cached."""

    forced = {q["name"]: {"choice": q["choices"][0]["value"], "probs": {q["choices"][0]["value"]: 1.0},
                          "confidence": 1.0, "forced": True}
              for q in questions if len(q["choices"]) == 1}      # the API needs at least two choices
    if forced:
        asked = [q for q in questions if q["name"] not in forced]
        return {**forced, **(decide(text, asked, cache, mode=mode, model=model, poster=poster) if asked else {})}
    body = request_body(text, questions, model)
    key = request_key(body)
    hit = cache / "responses" / f"{key}.json"
    if hit.exists():
        return parse_answers(json.loads(hit.read_text()), questions)
    if mode == "cached":
        raise KeyError(f"no cached answer for request {key}")
    if mode == "collect":
        _write_json(cache / "pending" / f"{key}.json", body)
        raise NeedRequest(key)
    for attempt in range(REFUSAL_RETRIES + 1):
        response = poster(body)
        _log_usage(cache, key, response)
        try:
            parsed = parse_answers(response, questions)
        except Refused:
            if attempt == REFUSAL_RETRIES:
                raise                       # not cached: a refusal is not an answer
            continue
        _write_json(hit, response)
        return parsed
    raise AssertionError("unreachable")


def _log_usage(cache: Path, key: str, response: Mapping[str, Any]) -> None:
    usage = response.get("usage", {})
    line = json.dumps({"key": key, "input_tokens": usage.get("input_tokens"), "t": time.time()})
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "usage.jsonl", "a") as handle:
        handle.write(line + "\n")


def send_pending(cache: Path, *, workers: int = 8, poster=post) -> tuple[int, list[str]]:
    """Answer every pending request (run where the API is reachable). Returns (sent, failed keys)."""

    todo = sorted((cache / "pending").glob("*.json")) if (cache / "pending").exists() else []
    failed: list[str] = []

    def one(path: Path) -> bool:
        hit = cache / "responses" / path.name
        if hit.exists():
            path.unlink()
            return True
        try:
            response = poster(json.loads(path.read_text()))
        except Exception as error:   # keep the batch going; the key is reported
            failed.append(f"{path.stem}: {error}")
            return False
        _write_json(hit, response)
        _log_usage(cache, path.stem, response)
        path.unlink()
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sent = sum(pool.map(one, todo))
    return sent, failed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Answer pending decision requests (login node).")
    parser.add_argument("cache", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sent, failed = send_pending(args.cache, workers=args.workers)
    print(f"sent {sent}, failed {len(failed)}")
    for line in failed[:10]:
        print("  ", line)


if __name__ == "__main__":
    main()
