"""Guided requirement composition over the shared measurement layer.

The legacy loop remains available for matched routing comparisons. This loop reads
options early, makes one small decision at a time and answers from current evidence.
"""

from __future__ import annotations

from dataclasses import asdict
import math
import re

from d4rt_agent.agent_v5 import kinds
from d4rt_agent.agent_v5.capabilities import serial
from d4rt_agent.agent_v5.evidence import attach, current_components
from d4rt_agent.agent_v5.final_context import register
from d4rt_agent.agent_v5.contracts import ActionRejected, schemas_text
from d4rt_agent.agent_v5.orchestrator import AgentV5, Exhausted, DECISION_TOKENS, TOOL_TOKENS, PROMPTS, prompt
from d4rt_agent.agent_v5.program import Entity, Event, FrameSpec, QueryProgram, Requirement, window_from_question
from d4rt_agent.agent_v5.tools import ToolError


FRAME_KINDS = {"trajectory", "relative_position", "subject_direction", "traj_subject_own_frame", "pair_relation"}
COUNT_KINDS = {"count_entities", "select_entities", "count_events", "instance_count"}
INITIAL_BODY = re.compile(r"(?:initial|starting|original).{0,40}(?:orientation|facing)|own (?:reference )?frame", re.I)


class CompositionalAgentV5(AgentV5):
    def __init__(self, *, legacy_menu=False, repair_cycles=2, answer_views="targeted", max_steps=32,
                 answer_policy="rated_repair", repair_tool_limit=4, **kwargs):
        super().__init__(max_steps=max_steps, **kwargs)
        self.legacy_menu = legacy_menu
        self.repair_cycles = int(repair_cycles)
        self.answer_views = answer_views
        if answer_policy not in ("plain", "rated", "rated_repair"):
            raise ValueError("unknown answer_policy")
        if not 0 <= int(repair_tool_limit) <= 4:
            raise ValueError("repair_tool_limit must be between 0 and 4")
        self.answer_policy = answer_policy
        self.repair_tool_limit = int(repair_tool_limit)

    def _decompose(self, task, clip, conv, trace, state):
        text = str(task["question"]) + "\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in task["options"].items())
        frames = [(clip.label(i), clip.pil(i)) for i in clip.keyframes()]
        conv.add("user", prompt("choose_family.md", question=text, menu=kinds.family_menu(self.legacy_menu)),
                 images=frames, kind="opening")
        families = kinds.LEGACY_FAMILIES if self.legacy_menu else kinds.FAMILIES
        try:
            _, _, state.family = self._decide(conv, trace, ["choose_family"], "family",
                lambda n, a: self._member(a["family"], families), DECISION_TOKENS)
            options = kinds.kinds_of(state.family, self.legacy_menu)
            if len(options) == 1:
                state.kind = options[0].name
            else:
                conv.add("user", prompt("choose_kind.md", family=state.family,
                                        menu=kinds.kind_menu(state.family, self.legacy_menu)), kind="step")
                _, _, state.kind = self._decide(conv, trace, ["choose_kind"], "kind",
                    lambda n, a: self._member(a["kind"], {k.name for k in options}), DECISION_TOKENS)
            self._fill_slots(text, conv, trace, state, kinds.KIND_BY_NAME[state.kind].slots)
        except Exhausted as exc:
            state.notes.append(str(exc))
            state.decomposition_ok = False
            state.kind, state.family, state.slots = "visual_state", "visual", {}
        entities, events = kinds.entities_from_slots(state.kind, state.slots, text)
        state.program = QueryProgram(str(task["id"]), state.kind, "measure", "", entities=entities, events=events,
                                     window=window_from_question(str(task["question"])))
        req = Requirement("r1", state.kind, [e.id for e in entities if e.kind != "camera"] or ["cam"],
                          window=state.program.window)
        state.program.requirements.append(req)
        state.active_requirement = req
        self._requirement_fields(task, conv, trace, state, req)

    @staticmethod
    def _member(value, choices):
        value = str(value).strip().strip("`")
        if value not in choices:
            raise ActionRejected(f"choose one of {list(choices)}")
        return value

    def _requirement_fields(self, task, conv, trace, state, req):
        conv.add("user", "Participants: " + ", ".join(f"{e.id} = {e.phrase or e.kind}" for e in state.program.entities), kind="step")
        try:
            if req.kind in FRAME_KINDS:
                conv.add("user", "Which reference frame does this requirement request? Preserve explicit initial orientation; "
                         "never replace it with scene axes. Name the reference entity and anchor.\n" + schemas_text(["set_frame"]), kind="step")

                def check_frame(name, args):
                    frame = self._member(args["frame"], ("scene", "world0", "camera", "image", "body_initial", "body_current"))
                    if INITIAL_BODY.search(str(task["question"])) and req.kind in ("trajectory", "subject_direction", "traj_subject_own_frame") \
                            and frame != "body_initial":
                        raise ActionRejected("the question explicitly specifies initial body orientation")
                    reference = str(args.get("reference") or req.entities[0])
                    if state.program.entity(reference.split("[")[0]) is None:
                        raise ActionRejected("frame reference must be a named entity")
                    anchor = float(args.get("anchor_s", 0))
                    if not math.isfinite(anchor) or anchor < 0 or \
                            (task["clip"].fps and anchor > (task["clip"].length - 1) / task["clip"].fps):
                        raise ActionRejected("frame anchor is outside the clip")
                    return FrameSpec(frame, reference, anchor, bool(args.get("explicit", False)))

                _, _, req.frame = self._decide(conv, trace, ["set_frame"], "frame", check_frame, DECISION_TOKENS)
            if req.kind in COUNT_KINDS:
                conv.add("user", "What is the counting unit and aggregation? Objects, occurrences and complete cycles differ.\n" +
                         schemas_text(["set_count_unit"]), kind="step")
                def check_count(name, args):
                    return (self._member(args["unit"], ("distinct_entities", "event_occurrences", "repetitions")),
                            self._member(args.get("aggregation", "unique"), ("unique", "at_time", "max_simultaneous")))
                _, args, (unit, aggregation) = self._decide(conv, trace, ["set_count_unit"], "count_unit", check_count, DECISION_TOKENS)
                req.count_unit, req.reduction = unit, aggregation
                if unit != "distinct_entities":
                    req.kind = "count_events"
                conv.add("user", "What exact predicate qualifies the counted entities/events? Preserve negation, action and hand/part roles.\n" +
                         schemas_text(["set_predicate"]), kind="step")
                _, args, text = self._decide(conv, trace, ["set_predicate"], "predicate",
                    lambda n, a: self._text(a["predicate"]), DECISION_TOKENS)
                req.predicate = "" if text.lower() == "none" else text
            if req.kind in ("order_events", "locate_events") and not req.events:
                conv.add("user", "Declare the events whose timing is required, including those named only in options.\n" +
                         schemas_text(["set_event_count"]), kind="step")
                def check_number(name, args):
                    number = int(args["count"])
                    if not 1 <= number <= 16:
                        raise ActionRejected("declare between 1 and 16 events")
                    return number
                _, _, number = self._decide(conv, trace, ["set_event_count"], "event_count", check_number, DECISION_TOKENS)
                for index in range(number):
                    conv.add("user", f"Name required event {index + 1}/{number}.\n" + schemas_text(["declare_event"]), kind="step")
                    _, _, text = self._decide(conv, trace, ["declare_event"], "declare_event",
                        lambda n, a: self._text(a["text"]), DECISION_TOKENS)
                    eid = f"ev{len(state.program.events) + 1}"
                    state.program.events.append(Event(eid, "unresolved", "other", text))
                    req.events.append(eid)
        except Exhausted as exc:
            state.notes.append(str(exc))
            # An unresolved frame is not a license to silently measure in scene coordinates.
            if req.kind in FRAME_KINDS:
                req.frame.basis = "unresolved"

    @staticmethod
    def _text(value):
        if not isinstance(value, str) or not value.strip():
            raise ActionRejected("provide a nonempty string")
        return value.strip()

    def _intro(self, state):
        req = state.active_requirement
        registry = ", ".join(k.name for k in kinds.KINDS)
        entities = ", ".join(f"{e.id}: {e.phrase or e.kind} ({e.role})" for e in state.program.entities)
        playbook_path = PROMPTS / "playbooks" / f"{req.kind}.md"
        playbook = playbook_path.read_text() if playbook_path.exists() else kinds.KIND_BY_NAME[req.kind].description
        events = "; ".join(f"{e.id}: {e.text}" for e in state.program.events)
        return prompt("compose.md", requirement=asdict(req), entities=entities, events=events,
                      kinds=registry, playbook=playbook, schemas=schemas_text(self._allowed(state)))

    def _allowed(self, state):
        req = state.active_requirement
        names = list(kinds.tools_for(req.kind))
        if req.kind not in kinds.COMPOSITIONAL_NAMES:
            names = list(dict.fromkeys(names + list(kinds.COMPOSITION_TOOLS)))
        names = [n for n in names if n != "switch_kind"]
        # Keep common actions relevant to the current state and recipe.
        if not state.workspace.inspections:
            names = [n for n in names if n not in ("observe", "merge_identity", "record_event")]
        if not state.workspace.occurrences:
            names = [n for n in names if n not in ("event_reduce", "event_window")]
        if req.kind in ("count_entities", "select_entities", "visibility", "framing", "visual_state", "interaction"):
            names = [n for n in names if n != "track_quality"]
        return names

    def _measure(self, task, conv, trace, state, tool_limit=None):
        if state.workspace is None:
            state.workspace = self.workspace_factory(task["clip"], state.program)
        repairs, executions = 0, 0
        while state.measuring_steps < self.max_steps and (tool_limit is None or executions < tool_limit):
            state.program.window = state.active_requirement.window
            conv.add("user", self._intro(state), kind="tool", summary=f"active recipe {state.active_requirement.kind}")
            try:
                name, args, _ = self._decide(conv, trace, self._allowed(state), "measure", lambda n, a: None, TOOL_TOKENS)
            except Exhausted as exc:
                state.notes.append(str(exc))
                break
            state.measuring_steps += 1
            if name == "done_measuring":
                gaps = self._gaps(state)
                if self.answer_policy == "plain" and tool_limit is None and gaps and repairs < self.repair_cycles and state.measuring_steps < self.max_steps:
                    repairs += 1
                    conv.add("user", f"Evidence review {repairs}/{self.repair_cycles}: {gaps}. Request a targeted observation or "
                             "complementary requirement. If still unresolved, retain that limitation.", kind="step")
                    continue
                if self.answer_policy == "plain":
                    state.notes.extend(gaps)
                break
            executions += 1
            if name in ("add_entity", "add_requirement"):
                try:
                    self._compose(task, conv, trace, state, name, args)
                except (ValueError, TypeError, ActionRejected) as exc:
                    conv.add("user", str(exc), kind="reject")
                    trace[-1]["tool_error"] = str(exc)
                continue
            req = state.active_requirement
            args = dict(args)
            if name in ("entity_track", "relative_position"):
                if req.frame.basis == "unresolved":
                    conv.add("user", "The requested frame remains unresolved; add/revisit the requirement or inspect it.", kind="reject")
                    continue
                frame = "own" if req.frame.basis == "body_initial" and name == "entity_track" else req.frame.basis
                args["frame"] = frame
                if req.frame.anchor_s is not None:
                    args["anchor_s"] = req.frame.anchor_s
                args["reference"] = req.frame.reference
            if name in ("count", "count_entities"):
                if req.predicate and not args.get("predicate") and not args.get("predicates"):
                    current = {o["predicate"] for o in state.workspace.observations.values()
                               if o["revision"] == state.workspace.revisions.get(o["entity"].split("[")[0], 0)}
                    if req.predicate not in current:
                        message = f"Bind the full qualifier {req.predicate!r} to observed predicates before counting. " \
                                  f"Supply predicate/predicates explicitly, or observe the full qualifier. Current names: {sorted(current)}."
                        conv.add("user", message, kind="reject")
                        trace[-1]["tool_error"] = message
                        continue
                    args["predicate"] = req.predicate
                args.setdefault("aggregation", req.reduction if req.reduction != "profile" else "unique")
            if name == "event_reduce" and req.count_unit == "repetitions":
                args.setdefault("operation", "repetitions")
            elif name == "event_reduce" and req.kind == "order_events":
                args.setdefault("operation", "order")
            if name == "event_reduce" and req.predicate:
                args.setdefault("predicate", req.predicate)
            self._run_tool(conv, trace, state, name, args)
            if name == "event_window":
                req.window = state.program.window
        if state.measuring_steps >= self.max_steps:
            state.notes.append(f"measuring stopped after {self.max_steps} steps")

    def _compose(self, task, conv, trace, state, name, args):
        if name == "add_entity":
            eid = str(args["entity"])
            if not re.fullmatch(r"[a-z][a-z0-9_]*", eid) or state.program.entity(eid):
                raise ActionRejected("use a new lowercase entity ID; existing identities are changed with ground/select")
            kind = self._member(args["kind"], ("object", "category", "landmark_region", "group"))
            phrase = str(args["phrase"]).strip()
            if not phrase or len(phrase.split()) > 8:
                raise ActionRejected("name a short segmentation noun phrase and retain the full role separately")
            state.program.entities.append(Entity(eid, kind, phrase, role=str(args.get("role", "participant"))))
            conv.add("user", f"Added {eid}: {phrase}", kind="step")
            return
        kind = self._member(args["kind"], kinds.COMPOSITIONAL_NAMES)
        entities = args["entities"]
        if not isinstance(entities, list) or not entities or any(not isinstance(e, str) for e in entities):
            raise ActionRejected("entities must be a nonempty list of entity IDs")
        if any(state.program.entity(e.split("[")[0]) is None for e in entities):
            raise ActionRejected("add missing entities before requesting measurements on them")
        dependencies = args.get("depends_on")
        if dependencies is None:
            dependencies = []
        if not isinstance(dependencies, list) or any(not isinstance(d, str) for d in dependencies):
            raise ActionRejected("depends_on must be a list of requirement IDs or null")
        if any(d not in {r.id for r in state.program.requirements} for d in dependencies):
            raise ActionRejected("dependencies must name existing requirements")
        req = next((r for r in state.program.requirements if r.kind == kind and r.entities == entities), None)
        if req is None:
            req = Requirement(f"r{len(state.program.requirements) + 1}", kind, entities,
                              window=state.program.window, depends_on=dependencies)
            state.program.requirements.append(req)
        else:
            for evidence in state.workspace.evidence.values():
                if evidence.requirement == req.id:
                    evidence.superseded = True
        state.active_requirement = req
        self._requirement_fields(task, conv, trace, state, req)

    def _run_tool(self, conv, trace, state, name, args):
        before = len(state.results)
        super()._run_tool(conv, trace, state, name, args)
        if len(state.results) == before:
            return
        result = state.results[-1][2]
        ws, req = state.workspace, state.active_requirement
        if name == "entity_track" and args.get("frame") != "image" and args.get("entity") not in ("cam", "camera") \
                and result.data.get("available") is not False:
            from d4rt_agent.agent_v5.spatial_tools import track_quality
            try:
                diagnostic = track_quality(ws, {"entity": args["entity"], **{k: args[k] for k in ("window", "anchor_s") if k in args}})
                result.data["track_diagnostics"] = diagnostic.data
                result.data["status"] = "conflicting" if diagnostic.data.get("status") == "conflicting" else "partial"
                result.text += "\n" + diagnostic.text
            except (ValueError, RuntimeError) as exc:
                result.data["status"] = "partial"
                result.text += f"\nCorrespondence check unresolved: {exc}. Visible-surface displacement is only a proxy."
        if name in ("record_event", "observe") and result.data.get("available"):
            value = result.data.get("value", {})
            keys = ("id",) if name == "record_event" else ("entity", "predicate", "window")
            for old_name, _, previous in state.results[:-1]:
                old_value = previous.data.get("value", {})
                if old_name == name and all(old_value.get(k) == value.get(k) for k in keys):
                    for evidence in previous.evidence:
                        evidence.superseded = True
        attach(ws, req, name, args, result)
        visual_ids = result.data.get("visual_ids", [])
        for vid in visual_ids:
            ws.visuals[vid]["requirement"] = req.id
        if result.images and not visual_ids:
            frames = ws.clip.keyframes(4)
            for i, item in enumerate(result.images):
                pic = item[1] if isinstance(item, tuple) else item
                vid = f"visual:{result.evidence[0].id}:{i}"
                target = args.get("entity") or args.get("target")
                if target in ("entities", "camera"):
                    target = None
                register(ws, vid, pic, frames, entity=target, requirement=req.id,
                         evidence_ids=[r.id for r in result.evidence], annotation="box")
                result.evidence[0].visual_ids.append(vid)
        result.data["evidence_records"] = [serial(asdict(e)) for e in result.evidence]
        conv.turns[-1].text = result.text + "\nNext: one more tool, complementary requirement, or done_measuring."

    def _gaps(self, state):
        ws = state.workspace
        administrative = {"ground", "select", "inspect", "show", "merge_identity", "event_window"}
        records = [e for e in ws.evidence.values() if not e.superseded and e.status in ("supported", "partial")
                   and e.observable not in administrative]
        required_tools = {"count_entities": {"count_entities", "count"}, "select_entities": {"count_entities", "observe"},
                          "count_events": {"event_reduce"}, "order_events": {"event_reduce"}}
        gaps = []
        for req in state.program.requirements:
            matching = [e for e in records if e.requirement == req.id and
                        (req.kind not in required_tools or e.observable in required_tools[req.kind])]
            if not matching:
                gaps.append(f"{req.id} ({req.kind}) has no current supporting reduction/observation")
            for eid in req.events:
                if not any(entry["occurrence"].event_id == eid and entry["occurrence"].status == "supported" and
                           all(ws.revisions.get(e, 0) == revision for e, revision in entry["revisions"].items())
                           for entry in ws.occurrences.values()):
                    gaps.append(f"required event {eid} has not been localized")
        for _, _, result in state.results:
            metadata = result.data.get("evidence")
            if metadata and not ws.evidence[metadata["id"]].superseded and isinstance(result.data.get("value"), dict):
                unresolved = result.data["value"].get("unresolved")
                if unresolved:
                    gaps.append(f"{metadata['id']}: unresolved predicate/identity candidates {unresolved}")
        return gaps

    def _answer(self, task, clip, conv, trace, state):
        from d4rt_agent.agent_v5.answering import answer
        return answer(self, task, conv, trace, state)

    def _repair(self, task, conv, trace, state, request):
        state.active_requirement = next(r for r in state.program.requirements if r.id == request["requirement"])
        state.program.window = state.active_requirement.window
        conv.turns = [t for t in conv.turns if t.role != "assistant"]
        conv.add("user", "Repair only this decision-relevant gap, preserving valid evidence: " + str(request) +
                 f". At most {self.repair_tool_limit} additional calls within the remaining total budget. "
                 "Use add_requirement to revise a frame or compose a complementary observable, and ground/select to revise bindings.", kind="step")
        self._measure(task, conv, trace, state, tool_limit=self.repair_tool_limit)

    def _solution(self, task, state, conv, trace, letter, args):
        result = super()._solution(task, state, conv, trace, letter, args)
        result.update(stack="compositional", schema_version=3, legacy_menu=self.legacy_menu,
                      repair_cycles=self.repair_cycles, answer_views=self.answer_views, answer_policy=self.answer_policy,
                      answer_protocol="plain_v1" if self.answer_policy == "plain" else "freeform_assessment_v1",
                      repair_tool_limit=self.repair_tool_limit, assessment_text=state.assessment_text,
                      assessment_present=bool(state.assessment_text), assessment_complete=None,
                      assessment_format="freeform", option_assessments=None,
                      assessment_history=state.assessment_history, repair_history=state.repair_history,
                      schema_recovery=state.schema_recovery, final_visual_ids=state.final_visual_ids,
                      visual_manifest=state.visual_manifest, omitted_visual_ids=state.omitted_visual_ids)
        if state.workspace:
            result["evidence"] = [serial(asdict(e)) for e in state.workspace.evidence.values()]
            result["notes"] = list(dict.fromkeys(state.notes + self._gaps(state)))
        result["generation_costs"] = {
            "calls": len(trace), "seconds": sum(r.get("seconds", 0) for r in trace),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in trace) if self._count_fn() else None,
            "input_tokens_estimate": sum(r.get("input_tokens_estimate", 0) for r in trace)}
        return result
