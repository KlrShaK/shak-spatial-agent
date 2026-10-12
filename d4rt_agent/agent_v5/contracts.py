"""The actions the v5 agent may emit, and what the host accepts at each step.

Each turn the host offers exactly the actions valid at that point (one in the decomposition phase,
the kind's tool menu while measuring, ``final_answer`` at the end) and lists only their schemas,
so the prompt stays small and an out-of-turn action is refused with a reason the model can act on.
Every action carries a ``justification``, as in v4: a trace where each step says why it was taken
is auditable, and a model made to state a reason states a better one.
"""

from __future__ import annotations

import json
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
    "review_evidence": {"description": "Check evidence once before answering; keep the current kind's small tool menu.",
                        "arguments": {"decision": "'answer', 'measure' with the same kind, or 'reroute' through family/kind",
                                      "missing": "required for repair: the specific missing or conflicting evidence"},
                        "required": ["decision"]},
    "final_answer": {"description": "Answer the multiple-choice question.",
                     "arguments": {"letter": "the option letter", "limitations": "optional: what stayed uncertain"},
                     "required": ["letter"]},
}
SPECS: dict[str, dict[str, Any]] = {**DECOMPOSITION_SPECS, **TOOL_SPECS}


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
    arguments["justification"] = justification.strip()
    return name, arguments


def parse_letter(value: Any, letters: Sequence[str]) -> str:
    text = str(value).strip().upper().strip("().:")
    if text[:1] in letters and (len(text) == 1 or not text[1].isalpha()):
        return text[:1]
    raise ActionRejected(f"`letter` must be one of {list(letters)}")
