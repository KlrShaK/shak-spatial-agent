"""The actions the v5 agent may emit, and what the host accepts at each step.

Each turn the host offers exactly the actions valid at that point (one in the decomposition phase,
the kind's tool menu while measuring, ``final_answer`` at the end) and lists only their schemas,
so the prompt stays small and an out-of-turn action is refused with a reason the model can act on.
Every action carries a ``justification``, as in v4: a trace where each step says why it was taken
is auditable, and a model made to state a reason states a better one.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from d4rt_agent.agent_v5.tools import TOOL_SPECS

DECOMPOSITION_SPECS: dict[str, dict[str, Any]] = {
    "choose_family": {"description": "Say what the question is about.",
                      "arguments": {"family": "one family name from the list"}, "required": ["family"]},
    "choose_kind": {"description": "Say which quantity decides the question.",
                    "arguments": {"kind": "one kind name from the list"}, "required": ["kind"]},
    "name_entity": {"description": "Answer the one slot question you were asked.",
                    "arguments": {"slot": "the slot name you were asked about", "value": "your answer"},
                    "required": ["slot", "value"]},
    "choose_window": {"description": "Give the time span the question is about.",
                      "arguments": {"start_s": "seconds", "end_s": "seconds"}, "required": ["start_s", "end_s"]},
    "final_answer": {"description": "Answer the multiple-choice question.",
                     "arguments": {"letter": "the option letter", "limitations": "optional: what stayed uncertain",
                                   "evidence_ids": "optional list of current evidence IDs",
                                   "repair_request": "optional {requirement,option,claim,missing,evidence,mode: measure|reroute,could_change_choice}"},
                     "required": ["letter"]},
    "answer_only": {"description": "Recovery: submit from current evidence after formatting failures.",
                    "arguments": {"letter": "actual option label", "limitations": "remaining uncertainty",
                                  "evidence_ids": "optional current reports or included visuals"}, "required": ["letter"]},
}
SPECS: dict[str, dict[str, Any]] = {**DECOMPOSITION_SPECS, **TOOL_SPECS}

COMPOSITION_SPECS = {
    "set_event_count": {"description": "How many distinct event descriptions must be localized? Include option-defined events.",
        "arguments": {"count": "integer from 1 to 16"}, "required": ["count"]},
    "declare_event": {"description": "Name one event to investigate, preserving the original action and participants.",
        "arguments": {"text": "one full event description from the question/options"}, "required": ["text"]},
    "set_frame": {"description": "Choose the requested frame and its anchor explicitly.",
        "arguments": {"frame": "scene|world0|camera|image|body_initial|body_current", "reference": "optional reference entity id",
                      "anchor_s": "optional seconds, default 0", "explicit": "boolean: directly specified in wording"}, "required": ["frame"]},
    "set_count_unit": {"description": "Distinguish physical identities, occurrences and complete repetitions.",
        "arguments": {"unit": "distinct_entities|event_occurrences|repetitions",
                      "aggregation": "unique|at_time|max_simultaneous"}, "required": ["unit"]},
    "set_predicate": {"description": "Preserve the exact qualifying action/attribute, including negation and anatomical roles.",
        "arguments": {"predicate": "predicate text, or 'none'"}, "required": ["predicate"]},
    "add_entity": {"description": "Name an additional participant, region, category or anatomical part.",
        "arguments": {"entity": "new short entity id", "phrase": "short segmentation noun phrase",
                      "kind": "object|category|landmark_region|group", "role": "optional role/referring expression"},
        "required": ["entity", "phrase", "kind"]},
    "add_requirement": {"description": "Add or revisit a complementary recipe without discarding current evidence.",
        "arguments": {"kind": "kind name", "entities": "existing entity IDs", "depends_on": "optional requirement IDs"},
        "required": ["kind", "entities"]},
}
SPECS.update(COMPOSITION_SPECS)
TOOL_SPECS.update({k: COMPOSITION_SPECS[k] for k in ("add_entity", "add_requirement")})


class ActionRejected(ValueError):
    """The host refused an action; the model is told why and tries again."""


def schemas_text(names: Sequence[str]) -> str:
    """The schemas of the actions valid now, one compact JSON line each."""

    lines = []
    for name in names:
        spec = SPECS[name]
        args = dict(spec["arguments"])
        args["justification"] = "one line: why"
        lines.append(f"- `{name}`: {spec['description']} Arguments: {json.dumps(args, separators=(', ', ': '))}")
    return "\n".join(lines)


def validate_action(action: Mapping[str, Any], allowed: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """``(name, arguments)`` of a well-formed action that is valid now, or ActionRejected."""

    if not isinstance(action, Mapping):
        raise ActionRejected("the JSON envelope must be an object")
    name = action.get("action", action.get("name"))
    if not isinstance(name, str):
        raise ActionRejected("the JSON needs an `action` string")
    if name not in allowed:
        raise ActionRejected(f"`{name}` is not available now; use one of {list(allowed)}")
    arguments = action.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise ActionRejected("`arguments` must be a JSON object")
    arguments = dict(arguments)
    spec = SPECS[name]
    justification = arguments.pop("justification", None)
    if not isinstance(justification, str) or not justification.strip():
        raise ActionRejected("every action needs a non-empty `justification` string")
    unknown = sorted(set(arguments) - set(spec["arguments"]))
    if unknown:
        raise ActionRejected(f"`{name}`: unexpected argument(s) {unknown}; allowed are {sorted(spec['arguments'])}")
    missing = [key for key in spec["required"] if arguments.get(key) in (None, "")]
    if missing:
        raise ActionRejected(f"`{name}` needs {missing}")
    arguments = {k: v for k, v in arguments.items() if v is not None}
    arguments["justification"] = justification.strip()
    return name, arguments


def parse_letter(value: Any, letters: Sequence[str]) -> str:
    text = str(value).strip()
    if text in letters:
        return text
    text = text.upper().strip("().:")
    if text[:1] in letters and (len(text) == 1 or not text[1].isalpha()):
        return text[:1]
    raise ActionRejected(f"`letter` must be one of {list(letters)}")


def assessment_prefix(raw: str) -> str:
    """Retain advisory prose even when the following submission is malformed."""
    start = re.search(r'\{\s*"(?:action|name)"\s*:\s*"(?:final_answer|answer_only)"', raw)
    return raw[:start.start()].strip() if start else (raw.strip() if raw.strip() != "{}" else "")


def parse_answer_response(raw: str) -> tuple[str, dict[str, Any]]:
    """Separate free-form prose from one terminal, strictly parsed action envelope."""
    fence = re.search(r"(?:^|\n)[ \t]*```(?:json)?[ \t]*\n(.*?)\n[ \t]*```[ \t]*$", raw.strip(), re.S)
    if fence:
        raw = raw.strip()[:fence.start()] + "\n" + fence.group(1)
    decoder = json.JSONDecoder()
    candidates = []
    offset = 0
    while offset < len(raw):
        start = raw.find("{", offset)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(raw[start:])
        except ValueError:
            offset = start + 1
            continue
        end = start + consumed
        if isinstance(value, dict):
            name = value.get("action", value.get("name"))
            if name in ("final_answer", "answer_only"):
                candidates.append((start, end))
            elif isinstance(name, str) and name in SPECS:
                raise ActionRejected("only a final submission tool may appear in the answer response")
        offset = end  # Skip braces inside already parsed objects and quoted strings.
    if len(candidates) != 1:
        raise ActionRejected("expected exactly one complete final tool-call object")
    start, end = candidates[0]
    if raw[end:].strip():
        raise ActionRejected("nothing may follow the final tool-call object")
    prefix = raw[:start].strip()
    if prefix.endswith("["):
        raise ActionRejected("the final tool call must be an object, not an array")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ActionRejected(f"duplicate key {key}")
            result[key] = value
        return result

    envelope = json.loads(raw[start:end], object_pairs_hook=unique_pairs)
    if set(envelope) != {"action", "arguments"}:
        raise ActionRejected("expected an action/arguments JSON object")
    return prefix, envelope


def validate_answer_submission(args: Mapping[str, Any], options: Mapping[str, str],
                               evidence: Mapping[str, Mapping[str, Any]], *, repair_allowed: bool) -> str:
    """Validate the submission and repair mechanics; do not interpret assessment prose."""
    letter = args.get("letter")
    if not isinstance(letter, str) or letter.strip() not in options:
        raise ActionRejected(f"letter must be an actual option label from {list(options)}")
    if args.get("limitations") is not None and not isinstance(args["limitations"], str):
        raise ActionRejected("limitations must be a string or null")
    repair = args.get("repair_request")
    if repair is not None:
        if not repair_allowed:
            raise ActionRejected("repair is unavailable; omit repair_request and answer with stated limitations")
        if not isinstance(repair, dict) or repair.get("mode") not in ("measure", "reroute"):
            raise ActionRejected("repair_request needs mode measure or reroute")
        fields = {"requirement", "option", "claim", "missing", "evidence", "mode", "could_change_choice"}
        if set(repair) != fields:
            raise ActionRejected(f"repair_request needs exactly {sorted(fields)}")
        if not isinstance(repair["option"], str) or repair["option"] not in options:
            raise ActionRejected("repair must identify an actual option label")
        if not isinstance(repair.get("requirement"), str):
            raise ActionRejected("repair must identify a requirement ID")
        if not all(isinstance(repair.get(k), str) and repair[k].strip()
                   for k in ("claim", "missing", "could_change_choice")):
            raise ActionRejected("repair needs claim text, missing evidence and how resolving it could_change_choice")
        refs = repair["evidence"]
        if not isinstance(refs, list) or any(not isinstance(r, str) or r not in evidence for r in refs):
            raise ActionRejected("repair evidence must list current IDs (empty allowed when none exists)")
    refs = args.get("evidence_ids", [])
    if not isinstance(refs, list) or any(not isinstance(r, str) or r not in evidence for r in refs):
        raise ActionRejected("evidence_ids must cite current reports or included visuals")
    return letter.strip()


def answer_template(options: Mapping[str, str]) -> str:
    return json.dumps({"action": "final_answer", "arguments": {
        "letter": "<one of " + ", ".join(options) + ">", "justification": "<brief reason for this choice>",
        "limitations": "<remaining uncertainty>"}}, separators=(",", ":"))
