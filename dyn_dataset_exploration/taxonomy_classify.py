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
