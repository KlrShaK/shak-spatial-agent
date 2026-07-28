#!/usr/bin/env python
"""Print an agent run's questions, reasoning, and answers as plain text.

Usage::

    python scripts/show_agent_reasoning.py d4rt_agent/results/.../simple_v2_centroid.json
"""

from __future__ import annotations

import json
import sys
import textwrap


def _wrap(text: str, indent: str = "    ") -> str:
    paragraphs = [p.strip() for p in text.strip().split("\n") if p.strip()]
    return "\n".join(
        textwrap.fill(p, width=96, initial_indent=indent, subsequent_indent=indent)
        for p in paragraphs
    )


def _thinking(raw: str) -> str:
    """The model thinks in prose, then emits one JSON action; keep the prose."""

    cut = raw.find('{"action"')
    return raw[:cut].strip() if cut > 0 else raw.strip()


def main(path: str) -> int:
    run = json.load(open(path))
    print("=" * 100)
    print(f"RUN {path}")
    print(f"status={run.get('status')}  point_mode={run.get('point_mode')}  "
          f"failures={len(run.get('failures', []))}")
    print("=" * 100)

    scores = {s.get("task_id"): s for s in run.get("scores", [])}

    for question in run.get("questions", []):
        task_id = question.get("id")
        print(f"\n\n{'#' * 100}")
        print(f"# {task_id}")
        print(f"# Q: {question.get('question')}")
        print("#" * 100)

        for entry in question.get("trace", []):
            action = entry.get("parsed_action", {})
            name = action.get("action", "?")
            print(f"\n--- step {entry.get('step')}: {name} " + "-" * 60)

            thought = _thinking(entry.get("raw_qwen_response", ""))
            if thought:
                print("\n  REASONING:")
                print(_wrap(thought, "    "))

            arguments = action.get("arguments", {})
            if name == "query_d4rt":
                print(f"\n  QUERY: label={arguments.get('label')!r} "
                      f"bbox={arguments.get('bbox_2d_1000')} t_src={arguments.get('t_src')} "
                      f"t_cam={arguments.get('t_cam')} t_tgt={arguments.get('t_tgt')}")
            elif name == "python_math":
                print("\n  CODE:")
                for line in str(arguments.get("code", "")).split("\n"):
                    print(f"      {line}")
                outputs = entry.get("result", {}).get("outputs")
                if outputs is not None:
                    print(f"  OUTPUTS: {json.dumps(outputs)}")
            elif name == "final_answer":
                kind = arguments.get("kind", "numeric")
                print(f"\n  ANSWER ({kind}):")
                if kind == "text":
                    print(_wrap(str(arguments.get("text", "")), "      "))
                else:
                    print(f"      {arguments.get('value')} {arguments.get('unit')}")
                print(f"  CITES: {arguments.get('evidence_ids')}")
                print("  LIMITATIONS:")
                print(_wrap(str(arguments.get("limitations", "")), "      "))

            if arguments.get("justification"):
                print(f"\n  JUSTIFICATION: {arguments['justification']}")
            if "error" in entry:
                print(f"\n  >>> REJECTED: {entry['error']}")

        score = scores.get(task_id)
        if score:
            print(f"\n  SCORE: ", end="")
            if score.get("scored", True):
                print(f"agent={score.get('agent_final_value')} "
                      f"gt={score.get('gt_value_m')} "
                      f"error={score.get('absolute_error_m')}")
            else:
                print(f"UNSCORED ({score.get('unscored_reason')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else
                          "d4rt_agent/results/basketball_6/simple_v2_centroid_blackwell.json"))
