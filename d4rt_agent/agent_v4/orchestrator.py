"""The v4 loop: a fixed sequence of host-driven steps, not a free-form agent.

`SimpleV2Orchestrator` is built around free-form evidence accumulation -- the
model decides what to call, the host validates and records it, repeat until the
model answers.  None of that structure applies once the host owns the
measurement plan, so this is a separate loop rather than a subclass of it.
Parsing (`extract_first_action`) is shared, because the failure modes of getting
one JSON object out of a chatty model have not changed.

What the host enforces here is the *sequence*.  A model asked to choose a bucket
will sometimes answer the question instead; accepting that would let it skip the
measurement entirely, which is the one thing this design exists to prevent.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from d4rt_agent.agent_v4 import buckets as buckets_module
from d4rt_agent.agent_v4.contracts import ACTION_SCHEMAS, ActionRejected, validate_action
from d4rt_agent.simple_v2 import extract_first_action

PROMPTS = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (PROMPTS / "agent_v4_system.md").read_text(encoding="utf-8")
BUCKET_CHOICE_PROMPT = (PROMPTS / "bucket_choice.md").read_text(encoding="utf-8")

# How many times the agent may re-name a subject the segmenter could not find.
# Beyond this the phrase is not the problem -- the subject is not resolvable in
# this clip -- and further attempts only burn the budget, which is exactly the
# pattern DSIOrchestrator had to add a hard refusal for in v2.
MAX_SUBJECT_ATTEMPTS = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentV4Orchestrator:
    """Drive one question from bucket choice to final answer."""

    def __init__(
        self,
        *,
        qwen: Any,
        pipeline: Any,
        max_steps: int = 10,
        system_prompt: str | None = None,
    ) -> None:
        self.qwen = qwen
        self.pipeline = pipeline
        self.max_steps = max(4, int(max_steps))
        self.system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt

    # -- message construction ----------------------------------------------

    def _opening_content(self, task: Mapping[str, Any], sampled: Any) -> list[dict[str, Any]]:
        from PIL import Image

        content: list[dict[str, Any]] = []
        for index, frame in enumerate(sampled.frames_rgb):
            content.append({"type": "text", "text": f"Sampled frame {index}:"})
            content.append({"type": "image", "image": Image.fromarray(frame)})
        content.append({
            "type": "text",
            "text": (
                f"Question: {task['question']}\n\n"
                f"Available action schemas: {json.dumps(ACTION_SCHEMAS, separators=(',', ':'))}\n\n"
                + BUCKET_CHOICE_PROMPT.replace("{choices}", buckets_module.choices_block())
            ),
        })
        return content

    def _say(self, messages: list[dict[str, Any]], text: str,
             images: list[Any] | None = None) -> None:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for image in images or []:
            content.append({"type": "image", "image": image})
        messages.append({"role": "user", "content": content})

    # -- the loop -----------------------------------------------------------

    def solve(self, task: Mapping[str, Any]) -> dict[str, Any]:
        sampled = self.pipeline.open(task["relative_path"], video_path=task.get("video_path"))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": self._opening_content(task, sampled)},
        ]

        trace: list[dict[str, Any]] = []
        state: dict[str, Any] = {
            "bucket": None, "subject": None, "grounding": None,
            "report": None, "subject_attempts": 0, "measured": False,
        }
        try:
            return self._loop(task, messages, trace, state)
        except Exception:
            # Whatever went wrong, the bucket and subject chosen before it are
            # exactly what a diagnosis needs, and losing them to a bare traceback
            # is how a failed question becomes uninvestigable.
            solution = self._solution(task, state, trace, None, exhausted=False)
            solution["status"] = "error"
            solution["error"] = traceback.format_exc()
            return solution

    def _loop(self, task: Mapping[str, Any], messages: list[dict[str, Any]],
              trace: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
        expected = "choose_bucket"

        for step in range(1, self.max_steps + 1):
            raw = self.qwen.generate(messages)
            attempt: dict[str, Any] = {
                "step": step, "expected_action": expected, "raw_response": raw,
                "parsed_action": None, "status": "rejected", "error": None,
                "at": _utc_now(),
            }

            try:
                extracted = extract_first_action(raw)
            except ValueError as error:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})
                self._reject(attempt, trace, messages, str(error), expected, missing=True)
                continue

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": raw[: extracted.end].strip()}],
            })
            try:
                name, arguments = validate_action(extracted.value)
            except ActionRejected as error:
                self._reject(attempt, trace, messages, str(error), expected)
                continue

            attempt["parsed_action"] = {"action": name, "arguments": arguments}
            if name != expected:
                self._reject(
                    attempt, trace, messages,
                    f"this step needs `{expected}`, not `{name}`. "
                    + self._what_is_needed(expected, state),
                    expected,
                )
                continue

            # Advance BEFORE recording acceptance: a step can still be refused
            # here (an out-of-range candidate index), and a trace that called it
            # accepted first would disagree with what the model was told.
            try:
                expected, done = self._advance(name, arguments, state, messages, task)
            except ActionRejected as error:
                self._reject(attempt, trace, messages, str(error), expected)
                continue

            attempt["status"] = "accepted"
            trace.append(attempt)
            if done:
                return self._solution(task, state, trace, arguments)

        return self._solution(task, state, trace, None, exhausted=True)

    def _what_is_needed(self, expected: str, state: Mapping[str, Any]) -> str:
        if expected == "choose_bucket":
            return "Choose the measurement first; you have not been given any measurement yet."
        if expected == "name_subject":
            return (
                f"The `{state['bucket']}` measurement needs a subject to segment. "
                "Name it in five words or fewer."
            )
        if expected == "choose_candidate":
            return "Pick which of the numbered candidate crops above is the intended subject."
        return "You have the measurement; answer the question now."

    def _reject(self, attempt: dict[str, Any], trace: list[dict[str, Any]],
                messages: list[dict[str, Any]], reason: str, expected: str,
                missing: bool = False) -> None:
        attempt["error"] = reason
        trace.append(attempt)
        correction = (
            "Your next response must end with exactly one JSON action"
            if missing else "Return one corrected JSON action"
        )
        self._say(
            messages,
            f"Action rejected: {reason}\n\n{correction}, using action `{expected}`.",
        )

    # -- state transitions --------------------------------------------------

    def _advance(self, name: str, arguments: Mapping[str, Any], state: dict[str, Any],
                 messages: list[dict[str, Any]], task: Mapping[str, Any]) -> tuple[str, bool]:
        if name == "final_answer":
            return "final_answer", True

        if name == "choose_bucket":
            bucket = buckets_module.get(arguments["bucket"])
            state["bucket"] = bucket.name
            if not bucket.needs_subject:
                self.pipeline.prepare_camera_only()
                self._deliver(bucket, None, messages, task, state)
                return "final_answer", False
            self._say(
                messages,
                "## Step 2 — name the subject\n\n"
                f"The `{bucket.name}` measurement is about a specific thing in the scene. "
                "Name it in **five words or fewer**, as a plain noun phrase an "
                "open-vocabulary segmenter can find — for example `the runner`, "
                "`the red car`, `the person in yellow`.\n\n"
                "Describe the thing, not its situation: `the runner` finds a runner, "
                "`the runner who moves left across the road` does not find one better. "
                "If several similar things are visible, add the one detail that "
                "distinguishes the intended one.",
            )
            return "name_subject", False

        if name == "name_subject":
            return self._handle_subject(arguments["subject"], state, messages, task)

        # choose_candidate
        grounding = state["grounding"]
        valid = [c.index for c in grounding.candidates]
        if arguments["index"] not in valid:
            raise ActionRejected(f"choose_candidate: index must be one of {valid}")
        state["chosen_index"] = arguments["index"]
        bucket = buckets_module.get(state["bucket"])
        self._commit_and_deliver(bucket, grounding, arguments["index"], messages, task, state)
        return "final_answer", False

    def _commit_and_deliver(self, bucket: Any, grounding: Any, index: int,
                            messages: list[dict[str, Any]], task: Mapping[str, Any],
                            state: dict[str, Any]) -> None:
        """Track the chosen instance, then measure -- declining if tracking fails.

        Propagation can fail on a clip where grounding succeeded: SAM3's video
        model is prompted with the same phrase but tracks its own instances, and
        when none of them overlaps the image model's pick it raises. That is a
        measurement outcome for this question, not a reason to lose it -- the
        other 199 in the run are unaffected, and an honest "could not track" lets
        the agent answer from the frames and say so.
        """

        try:
            self.pipeline.commit(grounding, index)
        except Exception as error:
            self._deliver(bucket, state["subject"], messages, task, state, report={
                "bucket": bucket.name, "available": False,
                "unavailable_reason": f"the subject could not be tracked through the clip: {error}",
                "text": (
                    f"The {bucket.name} measurement is UNAVAILABLE: the subject was located but "
                    f"could not be tracked through the clip ({type(error).__name__}: {error}), so "
                    "there is no 3D trajectory to measure."
                ),
                "metrics": {},
            })
            return
        self._deliver(bucket, state["subject"], messages, task, state)

    def _handle_subject(self, subject: str, state: dict[str, Any],
                        messages: list[dict[str, Any]],
                        task: Mapping[str, Any]) -> tuple[str, bool]:
        state["subject"] = subject
        state["subject_attempts"] += 1
        bucket = buckets_module.get(state["bucket"])
        self.pipeline.open(task["relative_path"], subject, video_path=task.get("video_path"))
        try:
            grounding = self.pipeline.ground(subject)
        except Exception as error:
            from d4rt_agent.agent_v4.pipeline import Grounding
            grounding = Grounding(status="not_found", subject=subject,
                                  reason=f"{type(error).__name__}: {error}")
        state["grounding"] = grounding

        if grounding.status != "ok":
            if state["subject_attempts"] < MAX_SUBJECT_ATTEMPTS:
                self._say(
                    messages,
                    f"The segmenter found nothing matching {subject!r} in any of the 32 "
                    f"frames ({grounding.reason}). Name the subject differently — a more "
                    "common, more generic word usually works better than a more specific "
                    "one. This is your last attempt.",
                )
                return "name_subject", False
            # Two phrases have failed; the subject is not resolvable in this clip.
            state["report"] = {
                "bucket": bucket.name, "available": False,
                "unavailable_reason": f"the segmenter could not find {subject!r} in this clip",
                "text": (
                    f"The {bucket.name} measurement is UNAVAILABLE: the segmenter could not "
                    f"locate the subject in any frame, so nothing could be tracked in 3D."
                ),
                "metrics": {},
            }
            self._deliver(bucket, subject, messages, task, state, report=state["report"])
            return "final_answer", False

        if grounding.ambiguous:
            self._say(
                messages,
                f"## Step 3 — which one?\n\nThe segmenter found several things matching "
                f"{subject!r} on frame {grounding.anchor_frame}, and is not confident which "
                f"you meant ({grounding.reason}). Each crop below is one candidate, in "
                "order. Choose the one the question is about by its number.\n\n"
                + "\n".join(
                    f"Candidate #{c.index} (confidence {c.score:.3f}):"
                    for c in grounding.candidates
                ),
                images=[c.crop for c in grounding.candidates if c.crop is not None],
            )
            return "choose_candidate", False

        state["chosen_index"] = grounding.best_index
        self._commit_and_deliver(bucket, grounding, grounding.best_index, messages, task, state)
        return "final_answer", False

    # -- delivering the measurement ----------------------------------------

    def _deliver(self, bucket: Any, subject: str | None, messages: list[dict[str, Any]],
                 task: Mapping[str, Any], state: dict[str, Any],
                 report: dict[str, Any] | None = None) -> None:
        """Run the measurement and compose the final turn.

        This is the whole point of the design: the agent sees the question, its
        options, ONE measurement, and the contract for reading that measurement.
        Not four measurements, and not a menu of tools.
        """

        if report is None:
            try:
                report = self.pipeline.measure(bucket, subject)
            except Exception as error:  # a pipeline failure is a measurement outcome
                report = {
                    "bucket": bucket.name, "available": False,
                    "unavailable_reason": f"the measurement pipeline failed: {error}",
                    "text": (
                        f"The {bucket.name} measurement is UNAVAILABLE: the pipeline failed "
                        f"while computing it ({type(error).__name__}: {error})."
                    ),
                    "metrics": {},
                }
        state["report"] = report
        state["measured"] = report.get("available", False)

        block = [
            "## Step 4 — answer",
            "",
            f"Measurement: `{bucket.name}`",
            "",
            report["text"],
        ]
        rendered = _render_metrics(report.get("metrics") or {})
        if rendered:
            block += ["", rendered]
        if report.get("available"):
            block += ["", bucket.contract]
        block += [
            "",
            "---",
            "",
            "Now answer this question. Choose exactly one option.",
            "",
            task["question"],
            "",
            "Options:",
            *[f"{letter}: {text}" for letter, text in task["options"].items()],
            "",
            "Reply with your reasoning, then a `final_answer` action whose `text` begins "
            "with the option letter, followed by that option's text, followed by the "
            "measurement that decided it.",
        ]
        self._say(messages, "\n".join(block))

    def _solution(self, task: Mapping[str, Any], state: Mapping[str, Any],
                  trace: list[dict[str, Any]], arguments: Mapping[str, Any] | None,
                  exhausted: bool = False) -> dict[str, Any]:
        report = state.get("report") or {}
        return {
            "question_id": task.get("id"),
            "relative_path": task.get("relative_path"),
            "answered": arguments is not None,
            "exhausted": exhausted,
            "answer_text": (arguments or {}).get("text"),
            "limitations": (arguments or {}).get("limitations"),
            "chosen_bucket": state.get("bucket"),
            "subject": state.get("subject"),
            "subject_attempts": state.get("subject_attempts", 0),
            "chosen_candidate_index": state.get("chosen_index"),
            "grounding_ambiguous": bool(
                getattr(state.get("grounding"), "ambiguous", False)
            ),
            "measurement_available": bool(report.get("available")),
            "measurement_unavailable_reason": report.get("unavailable_reason"),
            "report": report,
            "trace": trace,
            "steps_used": len(trace),
            "at": _utc_now(),
        }


def _round(value: Any, places: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: _round(v, places) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, places) for v in value]
    return value


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    widths = {c: max(len(c), *(len(_cell(r.get(c))) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    body = [
        "  ".join(_cell(row.get(c)).ljust(widths[c]) for c in columns) for row in rows
    ]
    return "\n".join([header, "-" * len(header), *body])


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_metrics(metrics: Mapping[str, Any]) -> str:
    """Show the numbers as a table where there is one, as JSON otherwise.

    A ten-row segment table read as JSON is technically complete and practically
    unreadable; the columns are what let a reader see that a pan grows steadily
    across every segment rather than jumping once.
    """

    if not metrics:
        return ""
    parts: list[str] = []

    segments = metrics.get("segments")
    if segments:
        columns = ["segment", "frame_start", "frame_end", "direction", "distance", "above_noise"]
        columns += [k for k in segments[0] if k.endswith("_delta")]
        parts.append("Per-segment measurements:\n" + _table(segments, columns))
        summary = {
            k: metrics[k] for k in
            ("total_distance", "net_displacement", "net_above_noise", "noise_floor",
             "angle_totals", "angle_spans", "angle_noise_deg", "full_span")
            if k in metrics
        }
        parts.append("Totals: " + json.dumps(_round(summary), separators=(", ", ": ")))

    checkpoints = metrics.get("checkpoints")
    if checkpoints:
        columns = [c for c in checkpoints[0] if c != "per_frame"]
        parts.append("Checkpoints:\n" + _table(checkpoints, columns))
        extra = {k: v for k, v in metrics.items()
                 if k not in {"checkpoints", "per_frame"}}
        if extra:
            parts.append("Also: " + json.dumps(_round(extra), separators=(", ", ": ")))

    if not parts:
        return json.dumps(_round(dict(metrics)), separators=(", ", ": "))
    return "\n\n".join(parts)
