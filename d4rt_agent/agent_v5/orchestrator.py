"""The v5 loop: decompose one decision at a time, measure with tools the model chooses, then answer.

Phase 1 (host-ordered, as v4 learned it has to be): family -> kind within the family -> one turn per
slot of the kind -> the window (regex; one turn only when the question bounds it by an event).
Every turn offers exactly one action and is validated before the next is asked.

Phase 2 (model-driven): the kind's playbook and tool menu are revealed; the model calls tools one
per turn, may switch kind once, and ends with ``done_measuring``. Measurement never decides the
answer: tools return numbers, noise floors and descriptive labels only.

Phase 3: the benchmark's frames, every tool result in full, and -- for the first time -- the
answer options; the model reasons and gives one letter.

The whole exchange is one conversation kept under a token budget (``context.Conversation``), and
everything is recorded in the trace so a wrong answer can be diagnosed afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import traceback
from typing import Any, Callable, Mapping

from d4rt_agent.agent_v5 import kinds as kinds_module
from d4rt_agent.agent_v5.clips import KEYFRAME_LONG_SIDE, QWEN_LONG_SIDE, Clip
from d4rt_agent.agent_v5.context import Conversation
from d4rt_agent.agent_v5.contracts import ActionRejected, parse_letter, schemas_text, validate_action
from d4rt_agent.agent_v5.kinds import KIND_BY_NAME, LEGACY_FAMILIES as FAMILIES, Slot
from d4rt_agent.agent_v5.program import QueryProgram, Window, window_from_question
from d4rt_agent.agent_v5.tools import TOOLS, ToolError, ToolResult, Workspace
from d4rt_agent.simple_v2 import extract_first_action

PROMPTS = Path(__file__).resolve().parent / "prompts"
MAX_REJECTIONS = 3
DECISION_TOKENS = 256
TOOL_TOKENS = 384
ANSWER_TOKENS = 512


def prompt(name: str, **values: Any) -> str:
    """A prompt file with ``{key}`` placeholders filled (``str.format`` would trip on the JSON)."""

    text = (PROMPTS / name).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class State:
    family: str | None = None
    kind: str | None = None
    slots: dict[str, str] = field(default_factory=dict)
    program: QueryProgram | None = None
    workspace: Workspace | None = None
    results: list[tuple[str, dict[str, Any], ToolResult]] = field(default_factory=list)
    kind_switches: int = 0
    measuring_steps: int = 0
    decomposition_ok: bool = True
    notes: list[str] = field(default_factory=list)


class Exhausted(RuntimeError):
    """A decision was rejected too many times; the run moves on with what it has."""


class AgentV5:
    def __init__(self, *, qwen: Any, workspace_factory: Callable[[Clip, QueryProgram], Workspace],
                 backbone: str, max_steps: int = 16, token_budget: int = 16000) -> None:
        self.qwen = qwen
        self.workspace_factory = workspace_factory
        self.backbone = backbone
        self.max_steps = int(max_steps)
        self.token_budget = int(token_budget)

    # -- plumbing -----------------------------------------------------------------

    def _count_fn(self) -> Callable[[str], int] | None:
        tokenizer = getattr(getattr(self.qwen, "processor", None), "tokenizer", None)
        if tokenizer is None:
            return None
        return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def _generate(self, conv: Conversation, max_new_tokens: int) -> str:
        conv.fit()
        try:
            return self.qwen.generate(conv.messages(), max_new_tokens=max_new_tokens)
        except TypeError:                                   # test doubles without the keyword
            return self.qwen.generate(conv.messages())

    def _decide(self, conv: Conversation, trace: list[dict[str, Any]], allowed: list[str], phase: str,
                check: Callable[[str, dict[str, Any]], Any], max_new_tokens: int) -> tuple[str, dict[str, Any], Any]:
        """Generate until one valid action from ``allowed`` passes ``check``; ``(name, args, check's value)``."""

        for attempt in range(1, MAX_REJECTIONS + 1):
            raw = self._generate(conv, max_new_tokens)
            record: dict[str, Any] = {"phase": phase, "attempt": attempt, "allowed": allowed, "raw": raw,
                                      "at": _utc_now(), "status": "rejected"}
            trace.append(record)
            try:
                extracted = extract_first_action(raw)
            except ValueError:
                conv.add("assistant", raw, kind="reject")
                self._reject(conv, "your reply must end with exactly one JSON action", allowed)
                record["error"] = "no JSON action"
                continue
            conv.add("assistant", raw[: extracted.end].strip(), kind="step")
            try:
                name, args = validate_action(extracted.value, allowed)
                value = check(name, args)
            except (ActionRejected, ToolError, ValueError) as error:
                conv.turns[-1].kind = "reject"
                self._reject(conv, str(error), allowed)
                record["error"] = str(error)
                continue
            record.update({"status": "accepted", "action": name, "arguments": args})
            return name, args, value
        raise Exhausted(f"{phase}: no valid action after {MAX_REJECTIONS} attempts")

    def _reject(self, conv: Conversation, reason: str, allowed: list[str]) -> None:
        conv.add("user", f"Action rejected: {reason}\nReply again with brief reasoning and ONE JSON action, "
                         f"using one of {allowed}.", kind="reject", summary="(a rejected action)")

    # -- the question -------------------------------------------------------------

    def solve(self, task: Mapping[str, Any]) -> dict[str, Any]:
        clip: Clip = task["clip"]
        conv = Conversation(prompt("system.md"), self.token_budget, self._count_fn())
        trace: list[dict[str, Any]] = []
        state = State()
        try:
            self._decompose(task, clip, conv, trace, state)
            spec = KIND_BY_NAME[state.kind]
            if spec.route == "measure":
                self._measure(task, conv, trace, state)
            letter, args = self._answer(task, clip, conv, trace, state)
            return self._solution(task, state, conv, trace, letter, args)
        except Exception:                                   # noqa: BLE001 -- keep what was decided
            solution = self._solution(task, state, conv, trace, None, None)
            solution["status"] = "error"
            solution["error"] = traceback.format_exc()
            return solution

    # -- phase 1 ----------------------------------------------------------------------

    def _decompose(self, task: Mapping[str, Any], clip: Clip, conv: Conversation, trace: list[dict[str, Any]],
                   state: State) -> None:
        question = str(task["question"])
        keyframes = [(clip.label(i) + ":", clip.pil(i, KEYFRAME_LONG_SIDE)) for i in clip.keyframes()]
        conv.add("user", prompt("choose_family.md", question=question, menu=kinds_module.family_menu(legacy=True)),
                 images=keyframes, kind="opening")
        try:
            _, _, state.family = self._decide(conv, trace, ["choose_family"], "family",
                                              lambda n, a: _member(a["family"], FAMILIES, "family"), DECISION_TOKENS)
            if state.family == "semantic":
                state.kind = "semantic"
            else:
                options = kinds_module.kinds_of(state.family, legacy=True)
                if len(options) == 1:
                    state.kind = options[0].name
                else:
                    conv.add("user", prompt("choose_kind.md", family=state.family,
                                            menu=kinds_module.kind_menu(state.family, legacy=True)), kind="step")
                    _, _, state.kind = self._decide(
                        conv, trace, ["choose_kind"], "kind",
                        lambda n, a: _member(a["kind"], {k.name for k in options}, "kind"), DECISION_TOKENS)
            self._fill_slots(question, conv, trace, state, kinds_module.LEGACY_KIND_BY_NAME[state.kind].slots)
        except Exhausted as error:
            state.decomposition_ok = False
            state.notes.append(str(error))
            state.kind = state.kind if state.kind and not kinds_module.LEGACY_KIND_BY_NAME[state.kind].slots else "semantic"
        state.program = self._program(task, clip, conv, trace, state)

    def _fill_slots(self, question: str, conv: Conversation, trace: list[dict[str, Any]], state: State,
                    slots: tuple[Slot, ...]) -> None:
        for slot in slots:
            if slot.name in state.slots:
                continue
            choices = f"Choose one of: {', '.join(slot.choices)}\n" if slot.choices else ""
            conv.add("user", prompt("name_entity.md", step=len(state.slots) + 3, ask=slot.ask, choices=choices,
                                    question=question, slot=slot.name), kind="step")

            def check(name: str, args: dict[str, Any], slot: Slot = slot) -> str:
                if str(args["slot"]).strip() != slot.name:
                    raise ActionRejected(f"this turn asks for slot `{slot.name}`")
                return kinds_module.validate_slot(slot, args["value"], question)

            _, _, value = self._decide(conv, trace, ["name_entity"], f"slot:{slot.name}", check, DECISION_TOKENS)
            state.slots[slot.name] = value

    def _program(self, task: Mapping[str, Any], clip: Clip, conv: Conversation, trace: list[dict[str, Any]],
                 state: State) -> QueryProgram:
        question = str(task["question"])
        spec = kinds_module.LEGACY_KIND_BY_NAME[state.kind]
        entities, events = kinds_module.entities_from_slots(state.kind, state.slots, question, legacy=True)
        window = window_from_question(question)
        if window.kind == "event" and spec.route == "measure" and clip.fps:
            duration = round((clip.length - 1) / clip.fps, 2)
            conv.add("user", prompt("choose_window.md", step=len(state.slots) + 3, event=window.event,
                                    duration=duration), kind="step")

            def check(name: str, args: dict[str, Any]) -> Window:
                a, b = float(args["start_s"]), float(args["end_s"])
                if not (0 <= min(a, b) and max(a, b) <= duration + 1e-6):
                    raise ActionRejected(f"the span must lie within 0..{duration} s")
                return Window("interval", min(a, b), max(a, b), text=window.event)

            try:
                _, _, window = self._decide(conv, trace, ["choose_window"], "window", check, DECISION_TOKENS)
            except Exhausted as error:
                state.notes.append(str(error))
                window = Window("whole")
        elif window.kind == "event":
            window = Window("whole")
        return QueryProgram(str(task["id"]), state.kind, spec.route, "", entities=entities, events=events,
                            window=window)

    # -- phase 2 ----------------------------------------------------------------------

    def _measure(self, task: Mapping[str, Any], conv: Conversation, trace: list[dict[str, Any]], state: State) -> None:
        state.workspace = self.workspace_factory(task["clip"], state.program)
        conv.add("user", self._measure_intro(state), kind="step")
        while state.measuring_steps < self.max_steps:
            allowed = list(kinds_module.tools_for(state.kind, legacy=True))
            if state.kind_switches >= 1 and "switch_kind" in allowed:
                allowed.remove("switch_kind")
            try:
                name, args, _ = self._decide(conv, trace, allowed, "measure", lambda n, a: None, TOOL_TOKENS)
            except Exhausted as error:
                state.notes.append(str(error))
                return
            state.measuring_steps += 1
            if name == "done_measuring":
                return
            if name == "switch_kind":
                self._switch(task, conv, trace, state, str(args["kind"]).strip().strip("`"))
                continue
            self._run_tool(conv, trace, state, name, args)
        state.notes.append(f"measuring stopped after {self.max_steps} steps")

    def _measure_intro(self, state: State) -> str:
        spec = kinds_module.LEGACY_KIND_BY_NAME[state.kind]
        program = state.program
        entities = ", ".join(
            "`cam` (the camera)" if e.kind == "camera" else
            (", ".join(f"`{e.id}[{i}]` ({e.phrase} #{i + 1})" for i in range(e.count or 2)) if e.kind == "group"
             else f"`{e.id}` ({e.phrase})") for e in program.entities)
        events = "".join(f"- Event `{ev.id}`: {ev.text}\n" for ev in program.events)
        w = program.window
        window = {"whole": "the whole clip", "instant": f"the moment {w.t0} s",
                  "interval": f"{w.t0} s to {w.t1} s"}.get(w.kind, f"{w.phase or w.kind}")
        playbook_path = PROMPTS / "playbooks" / f"{spec.name}.md"
        playbook = playbook_path.read_text() if playbook_path.exists() else "- Use the tools that measure the quantity."
        return prompt("measure.md", kind=spec.name, description=spec.description, entities=entities, events=events,
                      window=window, playbook=playbook.strip(), steps=self.max_steps,
                      schemas=schemas_text(kinds_module.tools_for(spec.name, legacy=True)))

    def _switch(self, task: Mapping[str, Any], conv: Conversation, trace: list[dict[str, Any]], state: State,
                kind: str) -> None:
        if kind not in kinds_module.LEGACY_KIND_BY_NAME or kind == state.kind:
            conv.add("user", f"`{kind}` is not a different kind from the list; continue measuring.", kind="reject")
            return
        state.kind_switches += 1
        state.notes.append(f"switched kind {state.kind} -> {kind}")
        state.kind = kind
        try:
            self._fill_slots(str(task["question"]), conv, trace, state, kinds_module.LEGACY_KIND_BY_NAME[kind].slots)
        except Exhausted as error:
            state.notes.append(str(error))
        entities, events = kinds_module.entities_from_slots(kind, state.slots, str(task["question"]))
        known = {e.id for e in state.program.entities}
        state.program.entities += [e for e in entities if e.id not in known]
        state.program.events = events or state.program.events
        state.program.kind = kind
        state.program.route = kinds_module.LEGACY_KIND_BY_NAME[kind].route
        if state.program.route == "vlm_only":
            state.measuring_steps = self.max_steps          # nothing to measure: go to the answer
            return
        conv.add("user", self._measure_intro(state), kind="step")

    def _run_tool(self, conv: Conversation, trace: list[dict[str, Any]], state: State, name: str,
                  args: dict[str, Any]) -> None:
        call = {k: v for k, v in args.items() if k != "justification"}
        try:
            result = TOOLS[name](state.workspace, call)
        except ToolError as error:
            conv.add("user", f"`{name}` could not run: {error}. Fix the arguments or use another tool.",
                     kind="reject", summary=f"{name}: rejected ({error})")
            trace[-1]["tool_error"] = str(error)
            return
        except Exception as error:                              # noqa: BLE001 -- a broken tool is evidence too
            result = ToolResult(f"## {name}\nThe tool failed ({type(error).__name__}: {error}). Treat this "
                                "measurement as unavailable.", f"{name}: failed", data={"error": traceback.format_exc()})
        state.results.append((name, call, result))
        kind = "show" if result.images else "tool"
        conv.add("user", result.text + "\n\nNext: one more tool, or `done_measuring`.", images=result.images,
                 kind=kind, summary=result.summary)
        trace[-1]["tool_summary"] = result.summary

    # -- phase 3 ----------------------------------------------------------------------

    def _answer(self, task: Mapping[str, Any], clip: Clip, conv: Conversation, trace: list[dict[str, Any]],
                state: State) -> tuple[str | None, dict[str, Any] | None]:
        options: Mapping[str, str] = task["options"]
        letters = sorted(options)
        if state.results:
            dossier = "\n\n".join(r.text for _, _, r in state.results)
        elif KIND_BY_NAME[state.kind].route == "vlm_only":
            dossier = "(No measurement: this question is decided by what the frames show.)"
        else:
            dossier = "(No measurement could be made; decide from the frames.)"
        frames = [(clip.label(i) + ":", clip.pil(i, QWEN_LONG_SIDE)) for i in clip.answer_frames]
        # the keyframes of the opening turn are superseded by the full frame set
        for turn in conv.turns:
            if turn.kind == "opening":
                turn.images = []
                turn.text = "[the first keyframes were shown here]\n" + turn.text
        conv.add("user", prompt("answer.md", n_frames=len(frames), dossier=dossier, question=task["question"],
                                options="\n".join(f"{k}. {options[k]}" for k in letters)),
                 images=frames, kind="answer")
        try:
            _, args, letter = self._decide(conv, trace, ["final_answer"], "answer",
                                           lambda n, a: parse_letter(a["letter"], letters), ANSWER_TOKENS)
        except Exhausted as error:
            state.notes.append(str(error))
            return None, None
        return letter, args

    # -- the record -------------------------------------------------------------------

    def _solution(self, task: Mapping[str, Any], state: State, conv: Conversation, trace: list[dict[str, Any]],
                  letter: str | None, args: Mapping[str, Any] | None) -> dict[str, Any]:
        program = state.program
        return {
            "question_id": task["id"],
            "status": "answered" if letter else "unanswered",
            "answered": letter is not None,
            "letter": letter,
            "justification": None if args is None else args.get("justification"),
            "limitations": None if args is None else args.get("limitations"),
            "family": state.family,
            "kind": state.kind,
            "slots": dict(state.slots),
            "decomposition_ok": state.decomposition_ok,
            "program": None if program is None else program.to_dict(),
            "backbone": self.backbone,
            "tools": [{"tool": name, "arguments": call, "summary": r.summary,
                       "data": _jsonable(r.data)} for name, call, r in state.results],
            "measuring_steps": state.measuring_steps,
            "kind_switches": state.kind_switches,
            "notes": state.notes,
            "peak_tokens": conv.peak_tokens,
            "trims": conv.trims,
            "trace": trace,
        }


def _member(value: Any, allowed: Any, what: str) -> str:
    text = str(value).strip().strip("`")
    if text not in allowed:
        raise ActionRejected(f"`{what}` must be one of {sorted(allowed)}")
    return text


def _jsonable(x: Any) -> Any:
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x
