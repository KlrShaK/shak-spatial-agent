"""The four actions the v4 agent may emit, and what the host accepts.

Deliberately much smaller than `simple_v2_contracts`: there is no grounding
geometry to protect, no evidence graph to keep consistent, and no arithmetic to
sandbox, because the agent no longer measures anything itself.  What remains is
making sure each step's answer is well formed before the host spends a GPU on it.

Every action carries a `justification`, kept for the same reason v2 kept it: a
trace where each step says why it was taken is auditable after the fact, and a
model made to state a reason states a better one.
"""

from __future__ import annotations

from typing import Any, Mapping

from d4rt_agent.agent_v4 import buckets as buckets_module

MAX_SUBJECT_WORDS = 5
MAX_FINAL_TEXT_CHARS = 2000

ACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "choose_bucket": {
        "description": "Name the one measurement that decides this question.",
        "arguments": {
            "bucket": f"one of {list(buckets_module.NAMES)}",
            "justification": "one or two lines naming the quantity and its frame",
        },
    },
    "name_subject": {
        "description": "Name the subject the question is about, for segmentation.",
        "arguments": {
            "subject": f"a noun phrase of at most {MAX_SUBJECT_WORDS} words",
            "justification": "one line on why this phrase identifies the right thing",
        },
    },
    "choose_candidate": {
        "description": "Pick which detected candidate is the intended subject.",
        "arguments": {
            "index": "the integer index of the candidate crop you are choosing",
            "justification": "one line on what distinguishes it",
        },
    },
    "final_answer": {
        "description": "Answer the multiple-choice question.",
        "arguments": {
            "kind": '"text"',
            "text": "the chosen option letter, then that option's text, then the measurements",
            "justification": "one line tying the measurement to the option",
            "limitations": "optional: what stayed uncertain",
        },
    },
}


class ActionRejected(ValueError):
    """The host refused an action; the model is told why and tries again."""


def _require_justification(arguments: Mapping[str, Any]) -> str:
    value = arguments.get("justification")
    if not isinstance(value, str) or not value.strip():
        raise ActionRejected("every action needs a non-empty `justification` string")
    return value.strip()


def _reject_unknown_fields(arguments: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ActionRejected(
            f"{name}: unexpected argument(s) {unknown}; allowed are {sorted(allowed)}"
        )


def validate_action(action: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(name, cleaned arguments)`` or raise ActionRejected."""

    if not isinstance(action, Mapping):
        raise ActionRejected("an action must be a JSON object")
    name = action.get("action") or action.get("name")
    if name not in ACTION_SCHEMAS:
        raise ActionRejected(
            f"unknown action {name!r}; use one of {sorted(ACTION_SCHEMAS)}"
        )
    arguments = action.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise ActionRejected(f"{name}: `arguments` must be a JSON object")

    justification = _require_justification(arguments)

    if name == "choose_bucket":
        _reject_unknown_fields(arguments, {"bucket", "justification"}, name)
        bucket = arguments.get("bucket")
        if bucket not in buckets_module.BY_NAME:
            raise ActionRejected(
                f"choose_bucket: {bucket!r} is not a bucket. Choose exactly one of "
                f"{list(buckets_module.NAMES)}, spelled exactly as written."
            )
        return name, {"bucket": bucket, "justification": justification}

    if name == "name_subject":
        _reject_unknown_fields(arguments, {"subject", "justification"}, name)
        subject = arguments.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            raise ActionRejected("name_subject: `subject` must be a non-empty string")
        subject = " ".join(subject.split())
        words = subject.split(" ")
        if len(words) > MAX_SUBJECT_WORDS:
            raise ActionRejected(
                f"name_subject: {subject!r} is {len(words)} words; use at most "
                f"{MAX_SUBJECT_WORDS}. The segmenter takes an open-vocabulary phrase, and a "
                "long one matches worse, not better. Name the thing, not its situation."
            )
        return name, {"subject": subject, "justification": justification}

    if name == "choose_candidate":
        _reject_unknown_fields(arguments, {"index", "justification"}, name)
        index = arguments.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ActionRejected("choose_candidate: `index` must be an integer")
        return name, {"index": index, "justification": justification}

    _reject_unknown_fields(
        arguments, {"kind", "text", "justification", "limitations"}, name
    )
    kind = arguments.get("kind", "text")
    if kind != "text":
        raise ActionRejected(
            'final_answer: these are multiple-choice questions, so `kind` must be "text" '
            "and the text must begin with the option letter you chose."
        )
    text = arguments.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ActionRejected("final_answer: `text` must be a non-empty string")
    if len(text) > MAX_FINAL_TEXT_CHARS:
        raise ActionRejected(
            f"final_answer: `text` is {len(text)} characters; keep it under "
            f"{MAX_FINAL_TEXT_CHARS}."
        )
    limitations = arguments.get("limitations")
    if limitations is not None and not isinstance(limitations, str):
        raise ActionRejected("final_answer: `limitations` must be a string when present")
    return name, {
        "kind": "text",
        "text": text.strip(),
        "justification": justification,
        "limitations": (limitations or "").strip() or None,
    }
