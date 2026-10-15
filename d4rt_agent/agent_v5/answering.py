"""Free-form option assessment, bounded submission recovery and one conditional repair."""

from datetime import datetime, timezone
import json
import time

from d4rt_agent.agent_v5.contracts import (
    ActionRejected, answer_template, assessment_prefix, parse_answer_response,
    validate_action, validate_answer_submission,
)
from d4rt_agent.agent_v5.final_context import build_packet, visible_refs


def _instructions(agent, task, state, repair_allowed):
    from d4rt_agent.agent_v5.orchestrator import prompt

    if agent.answer_policy == "plain":
        assessment = "Choose from the current evidence and included visuals."
    else:
        assessment = prompt("assess_answer.md")
    repair = (f"One conditional repair is available, at most {agent.repair_tool_limit} additional calls within the total budget. "
              "Include repair_request only for a specific resolvable claim that could change the choice. "
              "Its fields are requirement (existing r-ID), option, claim, missing, evidence (current IDs), "
              "mode (measure or reroute through complementary requirements), and could_change_choice."
              if repair_allowed else "Repair is unavailable. Omit repair_request and retain remaining limitations.")
    return assessment + "\n" + repair + "\nSubmit exactly one final_answer JSON object at the end, with no trailing text. " + \
           "Keep option assessments outside the JSON.\n" + answer_template(task["options"])


def _record_generation(agent, conv, raw, started, phase, attempt, refs, limit):
    return {"phase": phase, "attempt": attempt, "raw": raw, "at": datetime.now(timezone.utc).isoformat(),
            "status": "rejected", "seconds": time.monotonic() - started, "max_new_tokens": limit,
            "input_tokens_estimate": conv.tokens(), "output_tokens": agent._count_fn()(raw) if agent._count_fn() else None,
            "visual_ids": [k for k in refs if k.startswith(("visual:", "frame:"))],
            "evidence_ids": [k for k in refs if not k.startswith(("visual:", "frame:"))],
            "assessment_text": assessment_prefix(raw)}


def _remember(state, record):
    text = record.get("assessment_text", "")
    if text:
        if state.assessment_text is None:
            state.assessment_text = text
        state.assessment_history.append({"text": text, "attempt": record["attempt"], "phase": record["phase"],
            "visual_ids": record["visual_ids"], "submission_accepted": record["status"] == "accepted"})


def _visual_manifest(state, manifest, turns, omitted):
    state.final_visual_ids = [vid for vid, turn in turns if turn.images]
    state.visual_manifest = {vid: {**record, "included": vid in state.final_visual_ids} for vid, record in manifest.items()}
    state.omitted_visual_ids = sorted(set(omitted) | (set(manifest) - set(state.final_visual_ids)))


def _validate(agent, task, state, args, refs, repair_allowed):
    letter = validate_answer_submission(args, task["options"], refs, repair_allowed=repair_allowed)
    request = args.get("repair_request")
    if request and request["requirement"] not in {r.id for r in state.program.requirements}:
        raise ActionRejected("repair must target an existing requirement ID")
    return letter


def _submit(agent, task, trace, state, repair_allowed):
    from d4rt_agent.agent_v5.orchestrator import ANSWER_TOKENS

    conv, reports, turns, manifest, notes, omitted = build_packet(agent, task, state)
    limits = list(dict.fromkeys(state.notes + agent._gaps(state) + notes))
    conv.add("user", "Unresolved limitations: " + "; ".join(limits) + "\n" +
             _instructions(agent, task, state, repair_allowed), kind="answer")
    state.visual_manifest = manifest
    state.omitted_visual_ids = omitted
    limit = ANSWER_TOKENS if agent.answer_policy == "plain" else ANSWER_TOKENS + 256 * len(task["options"])
    for attempt in range(1, 4):
        started = time.monotonic()
        raw = agent._generate(conv, limit)
        refs = visible_refs(reports, turns)
        _visual_manifest(state, manifest, turns, omitted)
        record = _record_generation(agent, conv, raw, started, "answer", attempt, refs, limit)
        trace.append(record)
        try:
            text, envelope = parse_answer_response(raw)
            record["assessment_text"] = text
            name, args = validate_action(envelope, ["final_answer"])
            letter = _validate(agent, task, state, args, refs, repair_allowed)
        except (ValueError, TypeError, AttributeError) as exc:
            record["error"] = str(exc)
            _remember(state, record)
            conv.add("assistant", raw, kind="reject", summary=record["assessment_text"])
            conv.add("user", "Submission rejected: " + str(exc) + ". Preserve earlier assessment as unverified advisory notes. "
                     "Correct only the final submission; do not repeat assessments.\n" + answer_template(task["options"]), kind="reject")
            continue
        record.update(status="accepted", action=name, arguments=args)
        _remember(state, record)
        conv.add("assistant", raw, kind="step")
        return letter, args, conv
    state.schema_recovery = "attempted"
    # Rebuild the same packet policy, restoring relevant evidence instead of carrying malformed envelopes.
    conv, reports, turns, manifest, notes, omitted = build_packet(agent, task, state)
    if state.assessment_text:
        conv.add("user", "Earlier assessment (unverified advisory notes):\n" + state.assessment_text, kind="tool",
                 summary="Earlier model assessment is not measurement evidence.")
    conv.add("user", "Formatting recovery. Choose from the current reports and included visuals. Return only "
             'one {"action":"answer_only","arguments":{"letter":"<actual option label>",'
             '"justification":"...","limitations":"..."}} object. No repair.', kind="answer")
    started = time.monotonic()
    raw = agent._generate(conv, ANSWER_TOKENS)
    refs = visible_refs(reports, turns)
    _visual_manifest(state, manifest, turns, omitted)
    record = _record_generation(agent, conv, raw, started, "answer_recovery", 1, refs, ANSWER_TOKENS)
    trace.append(record)
    try:
        text, envelope = parse_answer_response(raw)
        record["assessment_text"] = text
        name, args = validate_action(envelope, ["answer_only"])
        letter = _validate(agent, task, state, args, refs, False)
    except (ValueError, TypeError, AttributeError) as exc:
        record["error"] = str(exc)
        state.schema_recovery = "failed"
        _remember(state, record)
        return None, None, conv
    record.update(status="accepted", action=name, arguments=args)
    state.schema_recovery = "answer_only"
    _remember(state, record)
    return letter, args, conv


def answer(agent, task, conv, trace, state):
    repair_allowed = (agent.answer_policy == "rated_repair" and agent.repair_cycles > 0 and agent.repair_tool_limit > 0
                      and not state.repair_history and state.measuring_steps < agent.max_steps)
    letter, args, final = _submit(agent, task, trace, state, repair_allowed)
    if args and args.get("repair_request"):
        request = args["repair_request"]
        before = state.measuring_steps
        state.repair_history.append({"request": request, "before_letter": letter,
            "before_assessment_text": state.assessment_text, "steps_before": before})
        agent._repair(task, conv, trace, state, request)
        # Provisional conclusions are archived, never added to the repaired evidence packet.
        state.assessment_text = None
        repaired_letter, repaired_args, repaired = _submit(agent, task, trace, state, False)
        if repaired_letter is None:
            state.notes.append("Post-repair submission failed; retained the previously accepted answer from before repair.")
            args = {**args, "repair_request": None, "limitations": "; ".join(filter(None, [
                args.get("limitations"), "Post-repair submission failed; this answer uses pre-repair evidence."]))}
        else:
            letter, args = repaired_letter, repaired_args
        state.repair_history[-1].update(after_letter=letter, steps_after=state.measuring_steps,
                                       calls_consumed=state.measuring_steps - before,
                                       submission_failed=repaired_letter is None)
        final.trims.extend(repaired.trims)
        final.peak_tokens = max(final.peak_tokens, repaired.peak_tokens)
        final.turns = repaired.turns
    conv.turns = final.turns
    conv.trims.extend(final.trims)
    conv.peak_tokens = max(conv.peak_tokens, final.peak_tokens)
    return letter, args
