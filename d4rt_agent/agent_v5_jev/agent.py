"""AgentV5's three phases, with every decision taken by a Decisions call on text.

Phase 1 (options hidden): family -> kind -> the kind's slots -> the window when an event bounds it.
Phase 2: per step, which tool to run (and on which entity / frame); the host runs it through V5's
         ``Workspace`` and appends its table to the dossier. Tools that need images or instance
         numbers (ground, select, show, switch_kind) are not offered: the model is text-only.
Phase 3: one choice over the option letters, given the caption, the dossier and the history.

The model never receives frames, the ground truth, or (before phase 3) the options. Everything it
chose is recorded in ``trace`` with its probabilities, so a wrong answer can be traced to a step.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from d4rt_agent.agent_v5 import kinds as kinds_module
from d4rt_agent.agent_v5.kinds import FAMILIES, KIND_BY_NAME, Slot
from d4rt_agent.agent_v5.program import QueryProgram, Window, window_from_question
from d4rt_agent.agent_v5.tools import TOOL_SPECS, TOOLS, ToolError, ToolResult, Workspace
from d4rt_agent.agent_v5_jev.candidates import entity_candidates
from d4rt_agent.agent_v5_jev.decisions import Refused, choice_question
from d4rt_agent.agent_v5_jev.windows import TIME_HINT, choose_window, frame_grid, snap_to_explicit, window_frames

V5_PROMPTS = Path(__file__).resolve().parents[1] / "agent_v5" / "prompts"
PROMPTS = Path(__file__).resolve().parent / "prompts"
NOT_OFFERED = {"ground", "select", "show", "switch_kind"}
DONE = "done_measuring"
FRAME_CHOICES = (("scene", "first camera's axes, camera motion removed (default)"),
                 ("camera", "as the moving camera sees it"), ("image", "in the picture (left/right, up/down, size)"),
                 ("own", "relative to the entity's own facing direction"))

Decide = Callable[[str, Sequence[Mapping[str, Any]]], dict[str, dict]]


class JevAgent:
    def __init__(self, *, decide: Decide, workspace_factory: Callable[[Any, QueryProgram], Workspace],
                 captions: Callable[[str], Mapping[str, Any]], backbone: str, max_tool_steps: int = 6,
                 window_mode: str = "regex") -> None:
        self.decide = decide
        self.workspace_factory = workspace_factory
        self.captions = captions
        self.backbone = backbone
        self.max_tool_steps = int(max_tool_steps)
        self.window_mode = window_mode          # regex: V5's reading of the question | hybrid: see _hybrid_window

    # -- one question ---------------------------------------------------------------

    def solve(self, task: Mapping[str, Any]) -> dict[str, Any]:
        run = _Run(task, self.captions(task["clip"].slug))
        try:
            self._decompose(run)
            if KIND_BY_NAME[run.kind].route == "measure":
                self._measure(run)
            self._answer(run)
        except Exception as error:      # noqa: BLE001 -- keep what was decided (NeedRequest is re-raised)
            if type(error).__name__ == "NeedRequest":
                raise
            run.error = traceback.format_exc()
        return run.solution(self.backbone)

    # -- calls ------------------------------------------------------------------------

    def _ask(self, run: "_Run", step: str, text: str, questions: list[dict]) -> dict[str, dict]:
        answers = self.decide(text, questions)
        for name, answer in answers.items():
            run.trace.append({"step": step, "question": name, "choice": answer["choice"],
                              "confidence": answer["confidence"], "probs": answer["probs"]})
        return answers

    def _context(self, run: "_Run", *extra: str) -> str:
        parts = ["## Description of the video (from 16 sampled frames)", run.caption["text"], "",
                 f"## Question\n{run.question}"]
        parts += [e for e in extra if e]
        return "\n".join(parts)

    # -- phase 1 ----------------------------------------------------------------------

    def _decompose(self, run: "_Run") -> None:
        text = self._context(run)
        family = self._ask(run, "family", text, [choice_question(
            "family", "Whose state does the question ask about, and relative to what?",
            list(FAMILIES.items()))])["family"]["choice"]
        run.family = family
        run.history.append(f"family: {family}")
        if family == "semantic":
            run.kind = "semantic"
        else:
            options = kinds_module.kinds_of(family)
            if len(options) == 1:
                run.kind = options[0].name
            else:
                run.kind = self._ask(run, "kind", text, [choice_question(
                    "kind", "Which quantity decides the question?",
                    [(k.name, k.description) for k in options])])["kind"]["choice"]
                run.history.append(f"kind: {run.kind}")
        self._fill_slots(run)
        run.program = self._program(run)

    def _fill_slots(self, run: "_Run") -> None:
        slots = KIND_BY_NAME[run.kind].slots
        if not slots:
            return
        questions: list[dict] = []
        for slot in slots:
            choices = _slot_choices(slot, run.question, run.caption)
            if not choices:
                run.decomposition_ok = False
                run.notes.append(f"no candidate for slot {slot.name}")
                run.kind = "semantic"
                run.slots.clear()
                return
            if len(choices) == 1:
                run.slots[slot.name] = choices[0][0]
                run.history.append(f"{slot.name}: {choices[0][0]} (only candidate)")
            else:
                questions.append(choice_question(slot.name, slot.ask, choices))
        if questions:
            answers = self._ask(run, "slots", self._context(run, f"\n(The question is about {run.kind}: "
                                                                f"{KIND_BY_NAME[run.kind].description})"), questions)
            for slot in slots:
                if slot.name in answers:
                    run.slots[slot.name] = answers[slot.name]["choice"]
                    run.history.append(f"{slot.name}: {run.slots[slot.name]}")

    def _program(self, run: "_Run") -> QueryProgram:
        spec = KIND_BY_NAME[run.kind]
        entities, events = kinds_module.entities_from_slots(run.kind, run.slots, run.question)
        window = window_from_question(run.question)
        clip = run.task["clip"]
        if (self.window_mode == "hybrid" and spec.route == "measure" and clip.fps
                and (window.kind != "whole" or TIME_HINT.search(run.question))):
            window = self._hybrid_window(run, window, clip)
        elif window.kind == "event" and spec.route == "measure":
            window = self._event_window(run, window)
        elif window.kind == "event":
            window = Window("whole")
        return QueryProgram(str(run.task["id"]), run.kind, spec.route, "", entities=entities, events=events,
                            window=window)

    def _hybrid_window(self, run: "_Run", regex_window: Window, clip: Any) -> Window:
        """The model reads the structure of the span (whole / one moment / a stretch, and the anchors
        start, middle, end) from the clip's frame grid; a time the question states explicitly replaces
        the nearest grid time. A refusal keeps the regex window."""

        grid = frame_grid(window_frames(clip.length), clip.fps)
        clue = f"(Event bounding the span, if any: '{regex_window.event}'.) " if regex_window.event else ""
        context = self._context(run)
        try:
            kind, t0, t1 = choose_window(lambda step, qs: self._ask(run, step, context, qs), grid, clue)
        except Refused:
            run.notes.append("window refused; kept the regex window")
            return regex_window
        t0, t1 = snap_to_explicit(t0, run.question, grid), snap_to_explicit(t1, run.question, grid)
        run.history.append(f"window: {kind} {t0}-{t1}")
        if kind == "whole":
            return Window("whole")
        if kind == "interval" and t0 is not None and t1 is not None and t1 <= t0:
            return regex_window
        return Window(kind, t0, t1, text=run.question)

    def _event_window(self, run: "_Run", window: Window) -> Window:
        """A window bounded by an event ("until the frame where the gate is fully open"), chosen
        from the clip's own frame times (``windows.choose_window``)."""

        clip = run.task["clip"]
        if not clip.fps:
            return Window("whole")
        context = self._context(run)
        try:
            kind, t0, t1 = choose_window(
                lambda step, qs: self._ask(run, step, context, qs), frame_grid(window_frames(clip.length), clip.fps),
                f"(The span is bounded by this event: '{window.event}'. Use the description of the video to say when it happens.)")
        except Refused:                      # the model could not place it: fall back to the whole clip
            run.notes.append(f"window refused for event '{window.event}'; using the whole video")
            return Window("whole")
        run.history.append(f"window: {kind} {t0}-{t1}")
        if kind == "whole":
            return Window("whole")
        return Window(kind, t0, t1, text=window.event)

    # -- phase 2 ----------------------------------------------------------------------

    def _measure(self, run: "_Run") -> None:
        ws = self.workspace_factory(run.task["clip"], run.program)
        run.workspace = ws
        done: set[str] = set()
        for _ in range(self.max_tool_steps):
            menu = [t for t in kinds_module.tools_for(run.kind) if t not in NOT_OFFERED]
            calls = _candidate_calls(ws, run.kind, menu, done)
            if not calls:
                return
            questions = [choice_question("tool", "Which measurement is most useful next?",
                                         [(c.key, c.describe()) for c in calls] +
                                         ([(DONE, "stop measuring: the measurements are enough")] if run.results else []))]
            pick = self._ask(run, f"tool{len(run.results) + 1}", self._measure_text(run), questions)["tool"]["choice"]
            if pick == DONE:
                run.history.append("measuring: done")
                return
            call = next(c for c in calls if c.key == pick)
            done.add(call.key)
            run.history.append(f"measure: {call.key}")
            self._run_tool(run, ws, call)

    def _measure_text(self, run: "_Run") -> str:
        playbook = V5_PROMPTS / "playbooks" / f"{run.kind}.md"
        program = run.program
        lines = [f"Kind: {run.kind} -- {KIND_BY_NAME[run.kind].description}",
                 "Entities: " + (", ".join(f"{e.id} = {e.phrase or e.kind}" for e in program.entities) or "none"),
                 f"Time span: {_window_text(program.window)}"]
        if playbook.exists():
            lines += ["", "How to measure this kind:", playbook.read_text(encoding="utf-8").strip()]
        if run.results:
            lines += ["", "## Measurements so far", *(r.text for _, _, r in run.results)]
        lines += ["", "## Decisions so far", *run.history]
        return self._context(run, "\n" + "\n".join(lines))

    def _run_tool(self, run: "_Run", ws: Workspace, call: "_Call") -> None:
        try:
            result = TOOLS[call.tool](ws, call.args)
        except ToolError as error:
            result = ToolResult(f"## {call.tool}\nThe tool could not run: {error}.", f"{call.tool}: rejected ({error})")
        except Exception as error:      # noqa: BLE001 -- a broken tool is evidence too
            result = ToolResult(f"## {call.tool}\nThe tool failed ({type(error).__name__}: {error}). Treat this "
                                "measurement as unavailable.", f"{call.tool}: failed",
                                data={"error": traceback.format_exc()})
        run.results.append((call.tool, call.args, result))
        run.trace[-1]["tool_summary"] = result.summary

    # -- phase 3 ----------------------------------------------------------------------

    def _answer(self, run: "_Run") -> None:
        options: Mapping[str, str] = run.task["options"]
        letters = sorted(options)
        if run.results:
            dossier = "\n\n".join(r.text for _, _, r in run.results)
        elif KIND_BY_NAME[run.kind].route == "vlm_only":
            dossier = "(No measurement: this question is decided by what happens in the video.)"
        else:
            dossier = "(No measurement could be made; decide from the description.)"
        guidance = (PROMPTS / "answer_context.md").read_text(encoding="utf-8").strip()
        question = [choice_question("answer", "Which option is correct?", [(k, options[k]) for k in letters])]
        history = "\n".join(run.history)
        full = self._context(run, f"\n## Measurements\n{dossier}", f"\n## How the analysis went\n{history}",
                             "\n" + guidance)
        answer = self._ask(run, "answer", full, question)["answer"]
        run.letter, run.answer = answer["choice"], answer
        # ablation: the same question without the measurements, to separate what the caption alone gives
        bare = self._ask(run, "answer_caption_only", self._context(run, "\n" + guidance), question)["answer"]
        run.caption_only = bare


class _Run:
    def __init__(self, task: Mapping[str, Any], caption: Mapping[str, Any]) -> None:
        self.task = task
        self.question = str(task["question"])
        self.caption = caption
        self.family: str | None = None
        self.kind: str = "semantic"
        self.slots: dict[str, str] = {}
        self.program: QueryProgram | None = None
        self.workspace: Workspace | None = None
        self.results: list[tuple[str, dict[str, Any], ToolResult]] = []
        self.history: list[str] = []
        self.trace: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.decomposition_ok = True
        self.letter: str | None = None
        self.answer: dict[str, Any] | None = None
        self.caption_only: dict[str, Any] | None = None
        self.error: str | None = None

    def solution(self, backbone: str) -> dict[str, Any]:
        from d4rt_agent.agent_v5.orchestrator import _jsonable

        return {
            "question_id": self.task["id"],
            "status": "error" if self.error else ("answered" if self.letter else "unanswered"),
            "answered": self.letter is not None,
            "letter": self.letter,
            "family": self.family, "kind": self.kind, "slots": dict(self.slots),
            "decomposition_ok": self.decomposition_ok,
            "program": None if self.program is None else self.program.to_dict(),
            "backbone": backbone,
            "tools": [{"tool": n, "arguments": a, "summary": r.summary, "data": _jsonable(r.data)}
                      for n, a, r in self.results],
            "measuring_steps": len(self.results), "kind_switches": 0, "notes": self.notes,
            "peak_tokens": 0, "trims": [],
            "history": self.history, "trace": self.trace,
            "answer_probs": None if self.answer is None else self.answer["probs"],
            "answer_confidence": None if self.answer is None else self.answer["confidence"],
            "caption_only_letter": None if self.caption_only is None else self.caption_only["choice"],
            "caption_only_probs": None if self.caption_only is None else self.caption_only["probs"],
            **({"error": self.error} if self.error else {}),
        }


class _Call:
    def __init__(self, tool: str, args: dict[str, Any], note: str = "") -> None:
        self.tool, self.args, self.note = tool, args, note
        self.key = tool if not args else tool + "(" + ", ".join(f"{k}={v}" for k, v in args.items()) + ")"

    def describe(self) -> str:
        return TOOL_SPECS[self.tool]["description"] + (f" -- {self.note}" if self.note else "")


def _slot_choices(slot: Slot, question: str, caption: Mapping[str, Any]) -> list[tuple[str, str]]:
    if slot.type == "choice":
        return [(c, c) for c in slot.choices]
    names = [e["name"] for e in caption.get("entities", [])]
    cands = entity_candidates(slot, question, names)
    described = {e["name"].lower(): e.get("behaviour") or "" for e in caption.get("entities", [])}
    return [(c, described.get(c.lower()) or "a phrase from the question or the description") for c in cands]


def _window_text(w: Window) -> str:
    return {"whole": "the whole video", "instant": f"the moment {w.t0} s",
            "interval": f"{w.t0} s to {w.t1} s"}.get(w.kind, f"{w.phase or w.kind}")


def _entity_args(ws: Workspace) -> list[str]:
    program = ws.program
    skip = {e.id for e in program.entities if e.kind in ("landmark_object", "landmark_region", "category", "camera")}
    return [i for i in ws.entity_choices(include_camera=False) if i.split("[")[0] not in skip]


def _candidate_calls(ws: Workspace, kind: str, menu: Sequence[str], done: set[str]) -> list[_Call]:
    """Every (tool, arguments) the model may pick next, minus those already run.

    Arguments are enumerated from the program (its entities and events) so the model chooses among
    concrete calls, as it would among tool names; a tool with one sensible argument set appears once."""

    entities = _entity_args(ws)
    names = {e.id: (e.phrase or e.kind) for e in ws.program.entities}

    def who(eid: str) -> str:
        return f"{eid} = {names.get(eid.split('[')[0], eid)}"

    frames = dict(FRAME_CHOICES)
    calls: list[_Call] = []
    for tool in menu:
        if tool == DONE:
            continue
        if tool in ("camera_motion", "camera_profile", "camera_zoom", "scene_revisit", "count"):
            calls.append(_Call(tool, {}))
        elif tool == "entity_track":
            calls += [_Call(tool, {"entity": e, "frame": f}, f"{who(e)}, {frames[f]}")
                      for e in entities for f in frames]
        elif tool == "distance":
            if entities:
                calls.append(_Call(tool, {"a": entities[0]}, f"between {who(entities[0])} and the camera"))
            if len(entities) >= 2:
                calls.append(_Call(tool, {"a": entities[0], "b": entities[1]},
                                   f"between {who(entities[0])} and {who(entities[1])}"))
        elif tool == "event_onset":
            calls += [_Call(tool, {"event": e.id}, f"{e.id}: {e.text}") for e in ws.program.events]
        else:
            calls += [_Call(tool, {"entity": e}, who(e)) for e in entities]
    return [c for c in calls if c.key not in done]
