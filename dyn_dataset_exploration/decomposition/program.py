"""S1 -- turn one question and its options into a typed Query Program.

A Dyn-Bench question is a composition of six choices: WHO is involved, WHEN, WHAT signal,
relative to which FRAME, how it is REDUCED, and how the OPTIONS encode the answer. A
**kind** fixes most of them (it is a program template, the successor of AgentV4's
buckets); the question fills the rest (entity phrases, the window, the events).

The order is host-enforced, as AgentV4 learned it has to be:

1. regex surface facts: timestamps, the window, duplicate options;
2. the kind, by rule for templated questions (``rule_parse``) and otherwise by Qwen
   choosing from ``KINDS`` (M2);
3. the kind's empty slots, one small validated question each (M2);
4. option atoms, by the kind's option parser -- each option becomes the claim it makes
   about the measurement, so S4 can check it (CLEVRER's "every option is a program").

Everything here is pure Python: the notebook kernel and the GPU prep job both import it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Callable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Vocabularies of the schema
# --------------------------------------------------------------------------- #

ENTITY_KINDS = (
    "camera", "object", "group", "category", "main_subject",
    "landmark_object", "landmark_region",
)
WINDOW_KINDS = ("whole", "interval", "instant", "phase", "event")
PHASES = ("first_25", "middle_50", "last_25", "first_half", "second_half", "start_to_middle", "end")
# "convention": the frame is a generator convention, decided by calibration, not wording.
FRAMES = ("world0", "camera_t", "levelled", "image", "convention")
SIGNALS = (
    "camera_motion", "camera_jitter", "range", "region_distance", "displacement",
    "instances", "onset", "order", "camera_speed", "focal", "covisibility", "elevation",
    "presence", "area", "speed", "speed_metric", "camera_displacement",
)
REDUCTIONS = ("dominant_motion", "level", "trend", "direction", "value_at", "count", "onset", "order",
               "profile", "shape", "verdict", "value")
ANSWER_FORMS = (
    "camera_motion", "shake_level", "trend", "direction", "metric", "count", "order",
    "narrative", "label", "rhythm", "zoom", "revisit", "elevation", "visibility", "size",
    "motion_profile",
)
# kinds whose subject may be the implicit "main subject/object"
IMPLICIT_SUBJECT_KINDS = ("cam_subject_range", "cam_subject_elevation", "apparent_size", "subject_visibility")
EVENT_PREDICATES = (
    "pan_left", "pan_right", "pan_any", "tilt_up", "tilt_down", "tilt_any", "starts_moving", "stops_moving",
    "enters_frame", "leaves_frame", "becomes_visible", "other",
)
ROUTES = ("measure", "vlm_only")


# --------------------------------------------------------------------------- #
# The kinds: program templates, and the list Qwen chooses from
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Kind:
    name: str
    # One line for Qwen: the QUANTITY and its FRAME, never the wording of any option.
    description: str
    route: str
    answer_form: str
    default_leaf: str     # "A" deterministic | "B" Qwen with evidence | "C" Qwen only
    iteration: int        # 1 = built in the first notebook iteration


KINDS: tuple[Kind, ...] = (
    Kind("camera_path", "How the camera itself moved or turned over the clip: pan, tilt, "
         "dolly, truck, pedestal, stationary, or a mix.", "measure", "camera_motion", "A", 1),
    Kind("camera_shake", "How much high-frequency jitter the camera has: steady footage "
         "versus handheld shake.", "measure", "shake_level", "A", 1),
    Kind("cam_subject_range", "Whether the straight-line distance between the camera and "
         "one subject grows, shrinks, holds, or reverses.", "measure", "trend", "A", 1),
    Kind("subject_direction", "Which way one subject's position moved (left/right, up/down, "
         "forward/backward) between two moments.", "measure", "direction", "A", 1),
    Kind("pair_relation", "Whether the distance between two scene entities, neither of them "
         "the camera, grows, shrinks, or holds.", "measure", "trend", "A", 1),
    Kind("metric_distance", "The distance in metres between two entities, or an entity and "
         "the camera, at a named moment.", "measure", "metric", "A", 1),
    Kind("instance_count", "How many distinct instances of a named category appear in the "
         "clip.", "measure", "count", "A", 1),
    Kind("event_order", "Which of two named events starts first; an event is a camera "
         "motion or something an entity does.", "measure", "order", "A", 1),
    Kind("landmark_relation", "How one entity's position changes relative to a static "
         "landmark or region of the scene.", "measure", "trend", "B", 1),
    Kind("semantic", "What is happening or what something is: decided by appearance, not "
         "by any measurement.", "vlm_only", "label", "C", 1),
    # Later iterations. Registered so the graph is complete; no parser or measurement yet.
    Kind("camera_rhythm", "The camera's speed over time: steady, speeding up, stop-start, "
         "or fixed.", "measure", "rhythm", "A", 2),
    Kind("camera_zoom", "Whether the focal length changed, separated from the camera moving "
         "forward or backward.", "measure", "zoom", "A", 2),
    Kind("scene_revisit", "Whether the last view shows the same place as the first.",
         "measure", "revisit", "A", 2),
    Kind("cam_subject_elevation", "The camera's height relative to a subject: above, level, "
         "or below.", "measure", "elevation", "A", 2),
    Kind("orbit_direction", "Which way the camera circles a subject, seen from above.",
         "measure", "label", "A", 2),
    Kind("side_filmed", "Which side of a subject faces the camera; needs the subject's "
         "facing.", "measure", "label", "B", 2),
    Kind("subject_visibility", "Whether and where a subject is inside the image frame, and "
         "when it enters or leaves.", "measure", "visibility", "A", 2),
    Kind("apparent_size", "How large a subject appears in the image over time.",
         "measure", "size", "A", 2),
    Kind("subject_motion_profile", "Whether a subject moves at all, and whether "
         "continuously, intermittently, or briefly.", "measure", "motion_profile", "A", 2),
    Kind("metric_speed", "The average speed in m/s of an entity over a named interval.",
         "measure", "metric", "A", 2),
    Kind("camera_translation", "Which way the camera's position moved between two moments: "
         "left/right, up/down, forward/backward.", "measure", "direction", "A", 2),
    Kind("tracked_objects_count", "How many objects the benchmark's own scene description "
         "tracks: not a vision count.", "vlm_only", "count", "B", 2),
)
KIND_BY_NAME: dict[str, Kind] = {kind.name: kind for kind in KINDS}


def kinds_block(iteration: int | None = None) -> str:
    """The kinds as Qwen sees them when choosing one."""

    chosen = [k for k in KINDS if iteration is None or k.iteration <= iteration]
    return "\n".join(f"{i}. `{k.name}` -- {k.description}" for i, k in enumerate(chosen, 1))


# --------------------------------------------------------------------------- #
# The Query Program
# --------------------------------------------------------------------------- #

@dataclass
class Entity:
    id: str
    kind: str
    phrase: str | None = None      # the SAM3 prompt, at most 5 words; None for the camera
    count: int | None = None       # group size ("the two men" -> 2)
    role: str = "subject"
    note: str | None = None        # the full referring expression, when it was shortened


@dataclass
class Window:
    kind: str = "whole"
    t0: float | None = None
    t1: float | None = None
    phase: str | None = None
    event: str | None = None
    text: str | None = None        # the words the window was read from


@dataclass
class Event:
    id: str
    actor: str                     # "camera" or an entity id
    predicate: str
    text: str


@dataclass
class Measurement:
    id: str
    signal: str
    of: list[str]                  # "camera", entity ids, "e1[0]" (group member), event or measurement ids
    reduce: str
    frame: str | None = None
    metric: bool = False
    window: Window | None = None    # None: the program's window


@dataclass
class OptionSpec:
    letter: str
    text: str
    atoms: dict[str, Any] | None   # the claim this option makes; None if nothing parsed
    parser: str                    # which parser produced the atoms, or "none"


@dataclass
class QueryProgram:
    qid: str
    kind: str
    route: str
    answer_form: str
    entities: list[Entity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    window: Window = field(default_factory=Window)
    measurements: list[Measurement] = field(default_factory=list)
    options: list[OptionSpec] = field(default_factory=list)
    duplicates: list[list[str]] = field(default_factory=list)
    open_slots: list[str] = field(default_factory=list)   # what S1 still has to ask Qwen
    source: str = "rules"          # "rules" | "qwen" | "reference"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QueryProgram":
        payload = dict(data)
        payload["entities"] = [Entity(**e) for e in payload.get("entities", [])]
        payload["events"] = [Event(**e) for e in payload.get("events", [])]
        payload["window"] = Window(**payload.get("window", {}))
        payload["measurements"] = [
            Measurement(**{**m, "window": Window(**m["window"]) if m.get("window") else None})
            for m in payload.get("measurements", [])
        ]
        payload["options"] = [OptionSpec(**o) for o in payload.get("options", [])]
        return cls(**payload)


_MEMBER = re.compile(r"^(?P<id>[A-Za-z_]\w*)\[(?P<k>\d+)\]$")


def validate(program: QueryProgram) -> None:
    """Raise ValueError naming every problem; a program that passes can be executed."""

    problems: list[str] = []
    kind = KIND_BY_NAME.get(program.kind)
    if kind is None:
        problems.append(f"unknown kind {program.kind!r}")
    elif program.route != kind.route:
        problems.append(f"route {program.route!r} does not match kind route {kind.route!r}")
    if program.route not in ROUTES:
        problems.append(f"unknown route {program.route!r}")
    if program.answer_form not in ANSWER_FORMS:
        problems.append(f"unknown answer_form {program.answer_form!r}")

    entities: dict[str, Entity] = {}
    for entity in program.entities:
        if entity.id in entities:
            problems.append(f"duplicate entity id {entity.id!r}")
        entities[entity.id] = entity
        if entity.kind not in ENTITY_KINDS:
            problems.append(f"{entity.id}: unknown entity kind {entity.kind!r}")
        if entity.kind == "camera" and entity.phrase:
            problems.append(f"{entity.id}: the camera takes no phrase")
        if entity.kind == "group" and (entity.count or 0) < 2:
            problems.append(f"{entity.id}: a group needs count >= 2")
        if entity.kind in {"object", "group", "category", "landmark_object", "landmark_region"} \
                and not entity.phrase:
            problems.append(f"{entity.id}: a {entity.kind} needs a phrase")
        if entity.phrase and len(entity.phrase.split()) > 5:
            problems.append(f"{entity.id}: phrase {entity.phrase!r} is longer than 5 words")

    events: dict[str, Event] = {}
    for event in program.events:
        if event.id in events or event.id in entities:
            problems.append(f"duplicate id {event.id!r}")
        events[event.id] = event
        if event.actor != "camera" and event.actor not in entities:
            problems.append(f"{event.id}: actor {event.actor!r} is neither camera nor an entity")
        if event.predicate not in EVENT_PREDICATES:
            problems.append(f"{event.id}: unknown predicate {event.predicate!r}")

    problems += _window_problems(program.window, "window")
    seen: set[str] = set()
    for m in program.measurements:
        if m.id in seen or m.id in entities or m.id in events:
            problems.append(f"duplicate id {m.id!r}")
        if m.signal not in SIGNALS:
            problems.append(f"{m.id}: unknown signal {m.signal!r}")
        if m.reduce not in REDUCTIONS:
            problems.append(f"{m.id}: unknown reduction {m.reduce!r}")
        if m.frame is not None and m.frame not in FRAMES:
            problems.append(f"{m.id}: unknown frame {m.frame!r}")
        for operand in m.of:
            if not _operand_ok(operand, entities, events, seen):
                problems.append(f"{m.id}: operand {operand!r} is not defined before it")
        if m.window is not None:
            problems += _window_problems(m.window, f"{m.id}.window")
        seen.add(m.id)
    if program.route == "vlm_only" and program.measurements:
        problems.append("a vlm_only program takes no measurements")
    if program.route == "measure" and not program.measurements and not program.open_slots:
        problems.append("a measure program needs at least one measurement")

    letters = [o.letter for o in program.options]
    if len(set(letters)) != len(letters):
        problems.append(f"duplicate option letters {letters}")
    if problems:
        raise ValueError(f"{program.qid}: " + "; ".join(problems))


def _window_problems(window: Window, where: str) -> list[str]:
    problems = []
    if window.kind not in WINDOW_KINDS:
        problems.append(f"{where}: unknown kind {window.kind!r}")
    if window.kind == "instant" and window.t0 is None:
        problems.append(f"{where}: an instant needs t0")
    if window.kind == "interval":
        if window.t0 is None or window.t1 is None:
            problems.append(f"{where}: an interval needs t0 and t1")
        elif window.t0 > window.t1:
            problems.append(f"{where}: t0 {window.t0} is after t1 {window.t1}")
    if window.kind == "phase" and window.phase not in PHASES:
        problems.append(f"{where}: unknown phase {window.phase!r}")
    return problems


def _operand_ok(operand: str, entities: Mapping[str, Entity], events: Mapping[str, Event],
                measurements: set[str]) -> bool:
    if operand == "camera" or operand in entities or operand in events or operand in measurements:
        return True
    member = _MEMBER.match(operand)
    if member and member.group("id") in entities:
        count = entities[member.group("id")].count or 0
        return int(member.group("k")) < count
    return False


def signature(program: QueryProgram) -> dict[str, Any]:
    """The parts of a program that decide its answer -- what two parses must agree on."""

    def _window(w: Window | None) -> Any:
        return None if w is None else (w.kind, w.t0, w.t1, w.phase)

    return {
        "kind": program.kind,
        "entities": sorted((e.id, e.kind, e.phrase, e.count) for e in program.entities),
        "events": sorted((e.id, e.actor, e.predicate) for e in program.events),
        "window": _window(program.window),
        "measurements": [
            (m.id, m.signal, tuple(m.of), m.reduce, m.frame, m.metric, _window(m.window))
            for m in program.measurements
        ],
        "options": {o.letter: o.atoms for o in program.options},
    }


def class_key(atoms: Mapping[str, Any] | None) -> str | None:
    """A hashable name for the class of claim an option makes (what an answer prior is over)."""

    return None if atoms is None else json.dumps(atoms, sort_keys=True, default=str)


def diff_programs(parsed: QueryProgram, reference: QueryProgram) -> list[str]:
    """Where a parse disagrees with the reference, one line per disagreement.

    A phrase the parse left as an open slot (the main subject before Qwen names it) is
    reported as open, not as wrong: it is the next sub-problem, not a parse error.
    """

    a, b = signature(parsed), signature(reference)
    open_ids = {slot.split(".", 1)[0] for slot in parsed.open_slots}
    lines: list[str] = []
    for key in ("kind", "window", "events", "measurements"):
        if a[key] != b[key]:
            lines.append(f"{key}: parsed {a[key]} != reference {b[key]}")
    parsed_entities = {e[0]: e for e in a["entities"]}
    for eid, kind, phrase, count in b["entities"]:
        mine = parsed_entities.pop(eid, None)
        if mine is None:
            lines.append(f"entity {eid}: missing from the parse")
        elif eid in open_ids and mine[2] is None:
            lines.append(f"entity {eid}: phrase is an open slot (reference: {phrase!r})")
        elif mine[1:] != (kind, phrase, count):
            lines.append(f"entity {eid}: parsed {mine[1:]} != reference {(kind, phrase, count)}")
    lines += [f"entity {eid}: not in the reference" for eid in parsed_entities]
    for letter in sorted(set(a["options"]) | set(b["options"])):
        if a["options"].get(letter) != b["options"].get(letter):
            lines.append(f"option {letter}: parsed {a['options'].get(letter)} "
                         f"!= reference {b['options'].get(letter)}")
    return lines


# --------------------------------------------------------------------------- #
# Surface facts: timestamps, windows, duplicate options, entity phrases
# --------------------------------------------------------------------------- #

_NUM = r"(\d+(?:\.\d+)?)"
_TIME_PATTERNS = (
    re.compile(_NUM + r"\s*s?\s*[-–]\s*" + _NUM + r"\s*s\b"),            # "0-3s", "0s–1.3s"
    re.compile(r"\bt\s*[=≈~]\s*" + _NUM),                               # "t≈2.0", "t=0.0s"
    re.compile(_NUM + r"\s*-?\s*(?:s|secs?|seconds?)\b(?!\s*/)"),        # "5.83 seconds", "5-second"
)


def timestamps(text: str) -> list[float]:
    """Every time value in the text, in order of appearance, without double counting."""

    hits: list[tuple[int, int, list[float]]] = []
    for pattern in _TIME_PATTERNS:
        for match in pattern.finditer(text):
            if any(s <= match.start() < e or s < match.end() <= e for s, e, _ in hits):
                continue
            hits.append((match.start(), match.end(), [float(g) for g in match.groups() if g]))
    return [value for _, _, values in sorted(hits) for value in values]


_PHASE_WORDS = (
    (re.compile(r"\bfirst 25\s*%"), "first_25"),
    (re.compile(r"\bmiddle 50\s*%"), "middle_50"),
    (re.compile(r"\blast 25\s*%"), "last_25"),
    (re.compile(r"\bfirst half\b"), "first_half"),
    (re.compile(r"\bsecond half\b"), "second_half"),
    (re.compile(r"\b(start|beginning)( of the video)? to the middle\b"), "start_to_middle"),
    (re.compile(r"\bat the end of the video\b"), "end"),
)
_FROM_START = re.compile(r"\bfrom (the )?(start|beginning|first frame)\b")
_EVENT_ANCHOR = re.compile(r"\b(?:the )?(?:frame|moment|point|time) (?:where|when) (?P<event>[^,?]+)")


def window_from_question(question: str) -> Window:
    """The question's time window, from its words alone.

    Two named times make an interval; one named time is an instant, unless the question
    runs "from the start" to it. A window bounded by something happening ("to the frame
    where the gate is fully open") is an event window: finding that frame is a separate
    sub-problem, not a regex.
    """

    lowered = question.lower()
    times = timestamps(question)
    for pattern, phase in _PHASE_WORDS:
        if pattern.search(lowered):
            span = (min(times), max(times)) if len(times) >= 2 else (None, None)
            return Window("phase", span[0], span[1], phase=phase, text=question)
    if len(times) >= 2:
        return Window("interval", min(times), max(times), text=question)
    if len(times) == 1:
        if _FROM_START.search(lowered):
            return Window("interval", 0.0, times[0], text=question)
        return Window("instant", times[0], None, text=question)
    anchor = _EVENT_ANCHOR.search(lowered)
    if anchor:
        return Window("event", event=anchor.group("event").strip(), text=question)
    return Window("whole")


def duplicate_options(options: Mapping[str, str]) -> list[list[str]]:
    """Groups of letters whose options say the same thing (tie -> the earlier letter)."""

    groups: dict[str, list[str]] = {}
    for letter in sorted(options):
        key = re.sub(r"[^a-z0-9]+", " ", options[letter].lower()).strip()
        groups.setdefault(key, []).append(letter)
    return [letters for letters in groups.values() if len(letters) > 1]


_IRREGULAR_PLURALS = {
    "people": "person", "persons": "person", "men": "man", "women": "woman",
    "children": "child", "kids": "kid", "geese": "goose", "mice": "mouse",
    "sheep": "sheep", "fish": "fish", "deer": "deer", "individuals": "person",
}


def singular(noun: str) -> str:
    noun = noun.lower()
    if noun in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[noun]
    if noun.endswith("ies"):
        return noun[:-3] + "y"
    if noun.endswith("sses") or noun in {"buses", "gases", "bonuses"}:
        return noun[:-2]                                   # glasses -> glass, buses -> bus
    if noun.endswith("ses"):
        return noun[:-1]                                   # horses -> horse, houses -> house
    if noun.endswith(("xes", "ches", "shes")):
        return noun[:-2]
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def sam3_phrase(text: str, max_words: int = 5) -> tuple[str, str | None]:
    """An open-vocabulary prompt of at most ``max_words`` words, and the original if cut.

    Articles are dropped (SAM3 scores "person in dark jacket" and "the person in the dark
    jacket" alike, and the words are better spent on attributes); the head noun and the
    first attributes are kept, since a long phrase matches worse, not better.
    """

    cleaned = re.sub(r"['’]s\b", "", text.strip().strip("?.,;:"))
    words = [w for w in cleaned.split() if w.lower() not in {"the", "a", "an"}]
    phrase = " ".join(words[:max_words]).lower()
    return phrase, (text.strip() if len(words) > max_words or phrase != text.strip().lower() else None)


_CLAUSE_WORDS = {"with", "on", "in", "at", "near", "next", "beside", "behind", "under", "over", "from", "to", "of",
                 "and", "by", "for", "while", "ahead", "down", "up", "against", "leaning", "into", "across"}
_NOT_PARTICIPLES = {"building", "ceiling", "painting", "clothing", "morning", "evening", "wedding", "parking", "railing",
                    "spring", "king", "string", "ring", "wing", "swing", "thing", "sibling", "pudding", "icing", "bling",
                    "flooring", "siding", "awning", "landing", "heading", "framing", "seating", "lighting"}
_ORDINAL_WORDS = {"first", "second", "third", "fourth", "other", "another", "main", "larger", "smaller", "left", "right",
                  "leftmost", "rightmost", "lead", "leading", "nearest", "farthest", "closest", "next", "last", "small",
                  "large", "big", "little", "tall", "short"}


def phrase_ladder(phrase: str) -> list[str]:
    """The prompts to try, in order, when the segmenter finds nothing for ``phrase``.

    The phrase itself first; then the noun phrase before the first preposition or participle
    ("camel walking near fence" -> "camel"; "brown horse with rider" -> "brown horse"); then the
    bare head noun ("horse"). A simpler prompt can match more than the subject meant (every
    "person"), which is why it is a fallback and why the track says it was used.
    """

    text = " ".join(phrase.lower().split())
    rungs = [text]
    words = text.split()
    cut = words
    for i, w in enumerate(words[1:], start=1):
        participle = len(w) >= 5 and w.endswith("ing") and w not in _NOT_PARTICIPLES
        if participle or w in _CLAUSE_WORDS:
            cut = words[:i]
            break
    if cut != words:
        rungs.append(" ".join(cut))
    core = [w for w in cut if w not in _ORDINAL_WORDS]
    head = core[-1] if core else None
    if head and head not in rungs and head not in {"it", "they", "them", "one", "thing", "video", "scene"}:
        rungs.append(head)
    return rungs


# --------------------------------------------------------------------------- #
# Option parsers: each option -> the claim it makes about the measurement
# --------------------------------------------------------------------------- #

def camera_motion_atoms(text: str) -> dict[str, Any] | None:
    """``{"motion": ..., "direction": ...}`` for camera-trajectory options."""

    t = text.lower()
    if re.search(r"\bcomplex\b|\brandom|multiple directions|irregular", t):
        return {"motion": "complex", "direction": None}
    if re.search(r"\b(fixed|stationary|static|still)\b|remains? (relatively )?(stable|still)"
                 r"|no (noticeable |intentional )?movement|does not move", t):
        return {"motion": "static", "direction": None, "shake": bool(re.search(r"shak", t))}
    if re.search(r"\barcs?\b", t):
        turn = "ccw" if "counterclockwise" in t or "counter-clockwise" in t else (
            "cw" if "clockwise" in t else None)
        return {"motion": "orbit", "direction": turn}
    if re.search(r"\bpan|scann|sweep", t):
        dest = re.search(r"from (left|right) to (left|right)", t)
        side = dest.group(2) if dest else _one_of(t, ("left", "right"))
        return {"motion": "pan", "direction": side}
    if re.search(r"\bvertical\b", t):
        return {"motion": "vertical", "direction": None}
    if re.search(r"\btilt", t):
        return {"motion": "tilt", "direction": _one_of(t, ("up", "down"))}
    if re.search(r"\bdoll|\bpush|\bpull|\bdepth\b", t):
        if re.search(r"forward|push(es)? in|\bin\b", t) and not re.search(r"backward|pull|\bout\b", t):
            return {"motion": "dolly", "direction": "forward"}
        if re.search(r"backward|pull|\bout\b", t) and not re.search(r"forward|push", t):
            return {"motion": "dolly", "direction": "backward"}
        return {"motion": "dolly", "direction": None}
    if re.search(r"\btruck|\btrack", t):
        return {"motion": "truck", "direction": _one_of(t, ("left", "right"))}
    if re.search(r"\borbit|circl|around", t):
        return {"motion": "orbit", "direction": None}
    if re.search(r"\bzoom", t):
        return {"motion": "zoom", "direction": _one_of(t, ("in", "out"))}
    if re.search(r"\b(pedestal|crane|boom|rises|descends)\b|moves? (up|down)", t):
        return {"motion": "pedestal", "direction": _one_of(t, ("up", "down"))}
    if re.search(r"\bmoves? (forward|backward)", t):
        return {"motion": "dolly", "direction": _one_of(t, ("forward", "backward"))}
    if re.search(r"\bmoves? (left|right)", t):
        return {"motion": "truck", "direction": _one_of(t, ("left", "right"))}
    return None


def _one_of(text: str, words: Sequence[str]) -> str | None:
    found = [w for w in words if re.search(rf"\b{w}\w*", text)]
    return found[0] if len(found) == 1 else None


_SHAKE_LEVELS = (("stable", 0), ("slight", 1), ("shaky", 2), ("severe", 3))


def shake_atoms(text: str) -> dict[str, Any] | None:
    """``{"level": 0..3}`` from the label the benchmark puts first."""

    t = text.lower().strip()
    for word, level in _SHAKE_LEVELS:
        if t.startswith(word):
            return {"level": level}
    for word, level in _SHAKE_LEVELS:
        if re.search(rf"\b{word}", t):
            return {"level": level}
    return None


# Ordered: compound shapes before the plain directions they contain.
_TREND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fluctuating", re.compile(r"fluctuat|back and forth|irregular|repeatedly|in and out|varies"
                               r"|multiple times|twice")),
    ("closer_then_farther", re.compile(
        r"(decreas\w*|closer|approach\w*|meet|touch|cross\w* paths)[^.;]*?\bthen\b[^.;]*?"
        r"(increas\w*|farther|further|away|apart|back out|drifts? back|separat\w*)")),
    ("farther_then_closer", re.compile(
        r"(increas\w*|farther|further|away|apart)[^.;]*?\bthen\b[^.;]*?"
        r"(decreas\w*|closer|approach\w*|return\w*|back together|come back)")),
    ("closer_then_constant", re.compile(
        r"(closer|decreas\w*|approach\w*)[^.;]*?\bthen\b[^.;]*?(stop\w*|stay\w*|remain\w*|hold\w*)")),
    ("farther_then_constant", re.compile(
        r"(farther|further|increas\w*|away)[^.;]*?\bthen\b[^.;]*?(stop\w*|stay\w*|remain\w*|hold\w*)")),
    ("constant", re.compile(
        r"(remain\w*|stay\w*|keep\w*|maintain\w*)[^.;]*?"
        r"(stable|constant|same|consistent|similar|unchanged)"
        r"|\bconstant\b|same distance|fixed distance|unchanged|\bstable\b|no significant change")),
    ("closer", re.compile(r"closer|approach\w*|grows? larger|becomes? larger|decreas\w*|nearer"
                          r"|toward\w* the camera")),
    ("farther", re.compile(r"farther|further|retreat\w*|smaller|increas\w*|away|apart|reced\w*")),
)


def trend_atoms(text: str) -> dict[str, Any] | None:
    """``{"trend": ...}`` for any option about a distance getting larger or smaller.

    Circling and "impossible to tell" are claims a range alone cannot check; they get their
    own atoms so S4 can see the option is unverifiable rather than silently mis-scored.
    """

    t = text.lower()
    if re.search(r"impossible to (determine|tell)|cannot be determined|unclear", t):
        return {"undeterminable": True}
    if re.search(r"\bcircl\w*", t) and not re.search(r"then", t):
        return {"circling": True}
    for name, pattern in _TREND_PATTERNS:
        if pattern.search(t):
            return {"trend": name}
    return None


_AXIS_WORDS = {
    "x": (("right", r"right\w*"), ("left", r"left\w*")),
    "y": (("up", r"up(ward|wards)?\b"), ("down", r"down(ward|wards)?\b")),
    "z": (("forward", r"forward\w*"), ("backward", r"backward\w*")),
}


def direction_atoms(text: str) -> dict[str, Any] | None:
    """``{"x": "right", "z": "forward", "slight": ["z"]}``; an axis not named is don't-care."""

    t = text.lower()
    if re.search(r"stationary|remains? (still|in place|in the same position)|does not move"
                 r"|in the same spot|no movement", t):
        return {"stationary": True}
    if re.search(r"random|erratic", t):
        return {"random": True}
    if re.search(r"sideways|laterally|across the frame", t):
        return {"lateral": True}
    if re.search(r"(closer|farther|further|away|distance|toward)", t) and not re.search(
            r"\b(left|right|up|down|upward|downward|forward|backward)\b", t):
        # "It starts far away and moves closer": a claim about range, not about an axis.
        return trend_atoms(t)
    atoms: dict[str, Any] = {}
    slight: list[str] = []
    for axis, choices in _AXIS_WORDS.items():
        for value, pattern in choices:
            match = re.search(r"(slightly |a little )?\b" + pattern, t)
            if match:
                atoms[axis] = value
                if match.group(1):
                    slight.append(axis)
                break
    if not atoms:
        return None
    if slight:
        atoms["slight"] = slight
    return atoms


_METRIC = re.compile(
    _NUM + r"\s*(m/s|meters? per second|metres? per second|km/h|cm|centimet\w+|m\b|meters?|metres?)")


def metric_atoms(text: str) -> dict[str, Any] | None:
    """``{"value": 1.6, "unit": "m"}`` (speeds in m/s), converted to SI."""

    match = _METRIC.search(text.lower())
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2)
    if unit.startswith("km/h"):
        return {"value": value / 3.6, "unit": "m/s"}
    if "/s" in unit or "per second" in unit:
        return {"value": value, "unit": "m/s"}
    if unit.startswith(("cm", "centimet")):
        return {"value": value / 100.0, "unit": "m"}
    return {"value": value, "unit": "m"}


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "single": 1,
}


def count_atoms(text: str) -> dict[str, Any] | None:
    t = text.lower()
    digits = re.search(r"\b(\d+)\b", t)
    if digits:
        return {"count": int(digits.group(1))}
    for word in re.findall(r"[a-z]+", t):
        if word in _NUMBER_WORDS:
            return {"count": _NUMBER_WORDS[word]}
    return None


_STOP = frozenset(
    "the a an of in on at to or and first begins begin starts start noticeable distinct "
    "clearly its their his her into from with by is are be both occur occurs happen happens "
    "frame scene video one them these events event".split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)][:6]
    return word[:6]


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def _best_event(text: str, events: Sequence[Event]) -> str | None:
    scores = [len(_stems(text) & _stems(ev.text + " " + ev.predicate.replace("_", " ")))
              for ev in events]
    if not scores or max(scores) == 0 or scores.count(max(scores)) > 1:
        return None
    return events[scores.index(max(scores))].id


def order_atoms(text: str, events: Sequence[Event]) -> dict[str, Any] | None:
    """``{"first": "ev1" | "ev2" | "tie" | "none" | "one_only", "only": ...}``."""

    t = text.lower()
    if re.search(r"same (frame|time|moment)|simultaneous|both (occur|happen)", t):
        return {"first": "tie"}
    if re.search(r"\bneither\b", t):
        return {"first": "none"}
    only = re.search(r"\bonly\b(?P<what>[^;.]*?)(\b(occurs|happens)\b|;|$)", t)
    if only:
        what = only.group("what")
        if re.fullmatch(r"\s*(one( of (them|these|these events|the events|the two))?)?\s*", what):
            return {"first": "one_only"}
        best = _best_event(what, events)
        return {"first": best, "only": best} if best else None
    for word, earlier_side in ((" before ", 0), (" after ", 1)):
        if word in t:
            sides = t.split(word, 1)
            best = _best_event(sides[earlier_side], events)
            return {"first": best} if best else None
    best = _best_event(t, events)
    return {"first": best} if best else None


def rhythm_atoms(text: str) -> dict[str, Any] | None:
    t = text.lower()
    if re.search(r"handheld|noticeable shaking", t):
        return {"rhythm": "shaky"}
    if re.search(r"initially remains stable|starts? (still|stable)", t):
        return {"rhythm": "still_then_burst"}
    if re.search(r"suddenly stop|sudden stop|stopped moving|then suddenly stops", t):
        return {"rhythm": "moving_then_stop"}
    if re.search(r"speeds? up|sudden acceleration|whips|starts slow, then suddenly", t):
        return {"rhythm": "speeds_up"}
    if re.search(r"\bfixed\b|no intentional movement|stationary", t):
        return {"rhythm": "fixed"}
    if re.search(r"irregular|chaotic|erratic", t):
        return {"rhythm": "irregular"}
    if re.search(r"consistent|uniform|steady|smooth", t):
        return {"rhythm": "steady"}
    return None


def zoom_atoms(text: str) -> dict[str, Any] | None:
    t = text.lower().strip()
    for prefix, name in (("none", "none"), ("no zoom", "none"), ("zoom in", "in"), ("zooming in and out", "mixed"),
                         ("zooming in", "in"), ("zoom out", "out"), ("zooming out", "out"), ("mixed", "mixed")):
        if t.startswith(prefix):
            return {"zoom": name}
    return None


def revisit_atoms(text: str) -> dict[str, Any] | None:
    t = text.lower().strip()
    for prefix, name in (("same", "same"), ("slight", "slight"), ("different", "different"), ("obscured", "obscured")):
        if t.startswith(prefix):
            return {"revisit": name}
    return None


def elevation_atoms(text: str) -> dict[str, Any] | None:
    t = text.lower().strip()
    for prefix, name in (("eye", "eye"), ("high", "high"), ("low", "low"), ("changing", "changing")):
        if t.startswith(prefix):
            return {"angle": name}
    return None


def visibility_atoms(text: str) -> dict[str, Any] | None:
    t = text.lower().strip()
    if re.search(r"unclear|unable to determine|not clear", t):
        return {"visible": "unclear"}
    if re.search(r"partially|partly|cut off|only briefly", t):
        return {"visible": "partial"}
    if re.search(r"out of (the )?frame|moved out|^no\b", t):
        return {"visible": "out"}
    if re.search(r"fully visible|^yes\b", t):
        return {"visible": "full"}
    return None


def size_atoms(text: str) -> dict[str, Any] | None:
    t = text.lower().strip()
    for prefix, name in (("larger", "larger"), ("smaller", "smaller"), ("unchanged", "unchanged"),
                         ("fluctuating", "fluctuating")):
        if t.startswith(prefix):
            return {"size": name}
    return None


def motion_profile_atoms(text: str) -> dict[str, Any] | None:
    t = text.lower().strip()
    if re.search(r"^not at all|completely stationary|^it remains .*stationary", t):
        return {"motion": "none"}
    if re.search(r"^only once|only once briefly", t):
        return {"motion": "once"}
    if re.search(r"^intermittent|moves intermittently", t):
        return {"motion": "intermittent"}
    if re.search(r"^continuous|moves continuously", t):
        return {"motion": "continuous"}
    return None


OPTION_PARSERS: dict[str, Callable[..., dict[str, Any] | None]] = {
    "rhythm": rhythm_atoms, "zoom": zoom_atoms, "revisit": revisit_atoms, "elevation": elevation_atoms,
    "visibility": visibility_atoms, "size": size_atoms, "motion_profile": motion_profile_atoms,
    "camera_motion": camera_motion_atoms,
    "shake_level": shake_atoms,
    "trend": trend_atoms,
    "direction": direction_atoms,
    "metric": metric_atoms,
    "count": count_atoms,
}


def parse_options(options: Mapping[str, str], answer_form: str,
                  events: Sequence[Event] = ()) -> list[OptionSpec]:
    specs = []
    for letter in sorted(options):
        text = options[letter]
        if answer_form == "order":
            atoms, parser = order_atoms(text, events), "order"
        elif answer_form in OPTION_PARSERS:
            atoms, parser = OPTION_PARSERS[answer_form](text), answer_form
        else:
            atoms, parser = None, "none"
        specs.append(OptionSpec(letter, text, atoms, parser if atoms is not None else "none"))
    return specs


# --------------------------------------------------------------------------- #
# Events (for event_order)
# --------------------------------------------------------------------------- #

_ENTITY_VERB = re.compile(
    r"^(?:the\s+)?(?P<x>.+?)\s+(?:enters|entering|walks|walking|leaves|leaving|exits|exiting|"
    r"becomes|appears|begins|starts|begin|start|beginning|is|gets|moves|turns|bends|lowers|steps|stepping|runs|"
    r"jumps|emerges|sits|stands|comes|disappears|shows|swims|flies|drives|rides)\b")


_MOVE_VERBS = (r"walk\w*|run\w*|mov\w*|swim\w*|fly\w*|driv\w*|ride\w*|riding|skat\w*|climb\w*|danc\w*|"
               r"roll\w*|slid\w*|step\w*|head\w*|approach\w*|chas\w*|paddl\w*|sail\w*|surf\w*|crawl\w*|"
               r"accelerat\w*|overtak\w*|drift\w*|travel\w*|glid\w*|dash\w*|sprint\w*|jog\w*|hop\w*|leap\w*|"
               r"trot\w*|gallop\w*|wander\w*|advanc\w*|retreat\w*|back\w* away")


def event_slot(text: str) -> dict[str, str]:
    """One side of "which happens first" as a slot: who acts, and which typed predicate it is."""

    t = text.lower().strip().strip("?.,")
    # the camera has to be the one acting: "the white tiger walks toward the camera" is the tiger's event
    if re.match(r"^(?:the\s+)?(?:(?:moving|handheld|main)\s+)?camera\s+(?!focus\b)\w+", t):
        pan = re.search(r"\b(left|right)ward pan|\bpan(?:s|ning)?\s+(?:slightly\s+)?(?:to the\s+)?(left|right)", t)
        tilt = re.search(r"\b(up|down)ward tilt|\btilt(?:s|ing)?\s+(?:slightly\s+)?(up|down)", t)
        if pan:
            predicate = f"pan_{pan.group(1) or pan.group(2)}"
        elif re.search(r"\bpan", t):
            predicate = "pan_any"
        elif tilt:
            predicate = f"tilt_{tilt.group(1) or tilt.group(2)}"
        elif re.search(r"\btilt", t):
            predicate = "tilt_any"
        elif re.search(r"\b(begins?|starts?|starting|beginning)\b.*\b(mov|shift|translat|track|follow|sweep|drift|dolly|push|pull|rise|rising|descend)", t):
            predicate = "starts_moving"
        elif re.search(r"\bstops\b", t):
            predicate = "stops_moving"
        else:
            predicate = "other"
        return {"actor": "camera", "predicate": predicate, "text": text.strip()}

    if re.search(r"\b(enters|entering|walks into|comes into (view|frame)|appears|shows up|becomes (?:\w+ )?visible|is first seen|comes into (view|frame))\b", t):
        predicate = "becomes_visible" if re.search(r"visible|first seen", t) else "enters_frame"
    elif re.search(r"\b(leaves|leaving|exits|exiting|walks out|disappears?|out of (the )?(frame|view))\b", t):
        predicate = "leaves_frame"
    elif re.search(rf"\b(begins|starts|starting|begin|start|beginning)\s+(to\s+)?({_MOVE_VERBS})", t):
        predicate = "starts_moving"
    else:
        predicate = "other"
    match = _ENTITY_VERB.match(t)
    return {"actor": match.group("x") if match else t, "predicate": predicate, "text": text.strip()}


def _is_camera(actor: str) -> bool:
    return str(actor).strip().lower() in ("camera", "the camera", "cam")


def normalise_event_slots(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The host's own reading of each event sentence outranks the model's where they disagree.

    The model sometimes files "the white tiger walks toward the camera" under the camera, and
    answers ``other`` for a plain "starts walking"; the sentence itself settles both.
    """

    fixed = []
    for slot in events:
        slot = dict(slot)
        text = slot.get("text") or ""
        if text:
            host = event_slot(text)
            if _is_camera(host["actor"]) != _is_camera(slot.get("actor", "")):
                slot["actor"], slot["predicate"] = host["actor"], host["predicate"]
            elif slot.get("predicate") in (None, "other") and host["predicate"] != "other":
                slot["predicate"] = host["predicate"]
        fixed.append(slot)
    return fixed


def upgrade_events(program: "QueryProgram") -> "QueryProgram":
    """The same host reading for a program read from disk: ``other`` becomes a checkable
    predicate where the event's own words allow it (the actor is not changed, because the
    entity it names was grounded when the program was written)."""

    for event in program.events:
        if event.predicate == "other":
            host = event_slot(event.text)
            if host["predicate"] != "other" and _is_camera(host["actor"]) == _is_camera(event.actor):
                event.predicate = host["predicate"]
    return program


# --------------------------------------------------------------------------- #
# Kind templates: a kind plus its slots is a program
# --------------------------------------------------------------------------- #

_REGION_WORDS = frozenset(
    "shore shoreline beach sand road street path trail sidewalk lane track field grass lawn "
    "water river lake sea ocean pool wave waves slope hill snow ground floor court sky "
    "room kitchen yard garden forest".split()
)

# What each kind leaves for the question to fill -- the only thing Qwen is asked for.
SLOT_SCHEMAS: dict[str, dict[str, str]] = {
    "camera_path": {}, "camera_shake": {}, "semantic": {}, "tracked_objects_count": {},
    "camera_rhythm": {}, "camera_zoom": {}, "scene_revisit": {}, "camera_translation": {},
    "cam_subject_range": {
        "subject": "the subject, at most 5 words; null when the question only says "
                   "'the main subject'",
        "implicit": "true when the question says 'the main subject/object' without naming it",
    },
    "cam_subject_elevation": {
        "subject": "the subject, at most 5 words; null when the question only says 'the main object'",
        "implicit": "true when the question says 'the main subject/object' without naming it",
    },
    "apparent_size": {
        "subject": "the subject, at most 5 words; null when the question only says 'the main object'",
        "implicit": "true when the question says 'the main subject/object' without naming it",
    },
    "subject_visibility": {
        "subject": "the subject, at most 5 words; null when the question only says 'the main object'",
        "implicit": "true when the question says 'the main subject/object' without naming it",
    },
    "subject_motion_profile": {"subject": "the entity whose motion is asked about, at most 5 words"},
    "subject_direction": {"subject": "the entity whose position changes, at most 5 words"},
    "metric_speed": {"subject": "the entity whose average speed is asked, at most 5 words"},
    "pair_relation": {
        "category": "when the question says 'the two X': X as a singular noun; else null",
        "a": "first entity, at most 5 words (null when category is set)",
        "b": "second entity, at most 5 words (null when category is set)",
    },
    "metric_distance": {
        "category": "when the question says 'the two X': X as a singular noun; else null",
        "a": "first entity, at most 5 words (null when category is set)",
        "b": "second entity, at most 5 words, or 'camera' (null when category is set)",
    },
    "instance_count": {"category": "the counted category as a singular noun"},
    "event_order": {
        "events": "the two events in the order the question names them; each "
                  "{actor: 'camera' or an entity of at most 5 words, predicate: one of "
                  + ", ".join(EVENT_PREDICATES) + ", text: the event as written}",
    },
    "landmark_relation": {
        "subject": "the moving entity, at most 5 words",
        "landmark": "what its position is compared to, at most 5 words",
        "landmark_kind": "'object' for a compact thing (door, lamp, car), 'region' for an "
                         "extended area or surface (road, grass, shoreline, kitchen)",
    },
}


def validate_slots(kind_name: str, slots: Mapping[str, Any]) -> None:
    """Raise ValueError with a message a model can act on; used to reject and retry."""

    if kind_name not in SLOT_SCHEMAS:
        raise ValueError(f"kind {kind_name!r} has no slot template yet")
    problems: list[str] = []
    unknown = sorted(set(slots) - set(SLOT_SCHEMAS[kind_name]))
    if unknown:
        problems.append(f"unexpected slots {unknown}; this kind takes {sorted(SLOT_SCHEMAS[kind_name])}")

    def _phrase(name: str, required: bool = True) -> None:
        value = slots.get(name)
        if value is None:
            if required:
                problems.append(f"`{name}` is required")
            return
        if not isinstance(value, str) or not value.strip():
            problems.append(f"`{name}` must be a non-empty string")

    if kind_name in ("subject_direction", "metric_speed"):
        _phrase("subject")
    elif kind_name in IMPLICIT_SUBJECT_KINDS:
        _phrase("subject", required=not slots.get("implicit"))
    elif kind_name == "subject_motion_profile":
        _phrase("subject")
    elif kind_name in ("pair_relation", "metric_distance"):
        if slots.get("category"):
            _phrase("category")
        else:
            _phrase("a")
            _phrase("b")
    elif kind_name == "instance_count":
        _phrase("category")
    elif kind_name == "landmark_relation":
        _phrase("subject")
        _phrase("landmark")
        if slots.get("landmark_kind") not in (None, "object", "region"):
            problems.append("`landmark_kind` must be 'object' or 'region'")
    elif kind_name == "event_order":
        events = slots.get("events")
        if not isinstance(events, list) or len(events) != 2:
            problems.append("`events` must list exactly two events")
        else:
            for index, event in enumerate(events, 1):
                if not isinstance(event, Mapping) or not event.get("actor"):
                    problems.append(f"event {index} needs an `actor`")
                elif event.get("predicate") not in EVENT_PREDICATES:
                    problems.append(f"event {index}: predicate must be one of {list(EVENT_PREDICATES)}")
    if problems:
        raise ValueError("; ".join(problems))


def _entity(eid: str, kind: str, text: str, role: str = "subject") -> Entity:
    phrase, note = sam3_phrase(text)
    return Entity(eid, kind, phrase, role=role, note=note)


def _camera() -> Entity:
    return Entity("cam", "camera", role="observer")


def _pair(slots: Mapping[str, Any], reduce: str, metric: bool) -> dict[str, Any]:
    if slots.get("category"):
        entities = [Entity("g1", "group", singular(sam3_phrase(slots["category"])[0]), count=2, role="pair")]
        operands = ["g1[0]", "g1[1]"]
    elif re.fullmatch(r"(the )?camera", str(slots["b"]).strip().lower()):
        entities = [_entity("e1", "object", slots["a"]), _camera()]
        operands = ["e1", "cam"]
    else:
        entities = [_entity("e1", "object", slots["a"]), _entity("e2", "object", slots["b"])]
        operands = ["e1", "e2"]
    return {"entities": entities,
            "measurements": [Measurement("m1", "range", operands, reduce, metric=metric)]}


def _template(kind_name: str, slots: Mapping[str, Any], question: str) -> dict[str, Any]:
    """Entities, events and measurements for one kind, from its filled slots."""

    if kind_name == "camera_path":
        return {"entities": [_camera()],
                "measurements": [Measurement("m1", "camera_motion", ["cam"], "dominant_motion",
                                             frame="levelled")]}
    if kind_name == "camera_shake":
        return {"entities": [_camera()],
                "measurements": [Measurement("m1", "camera_jitter", ["cam"], "level")]}
    if kind_name in IMPLICIT_SUBJECT_KINDS:
        # Whether the subject is implicit is a fact about the question's words, so the
        # host reads it there rather than trusting the model's flag.
        implicit = bool(slots.get("implicit")) or bool(
            re.search(r"\bmain (subject|object)\b", question, re.I))
        subject = slots.get("subject")
        if subject and str(subject).strip().lower() in ("main object", "main subject", "the main object", "the main subject"):
            subject = None
        if subject:
            e1 = _entity("e1", "main_subject" if implicit else "object", subject)
        else:
            e1 = Entity("e1", "main_subject")
        entities = [_camera(), e1] if kind_name in ("cam_subject_range", "cam_subject_elevation") else [e1]
        measurement = {
            "cam_subject_range": Measurement("m1", "range", ["e1", "cam"], "trend"),
            "cam_subject_elevation": Measurement("m1", "elevation", ["e1", "cam"], "verdict", frame="levelled"),
            "apparent_size": Measurement("m1", "area", ["e1"], "shape"),
            "subject_visibility": Measurement("m1", "presence", ["e1"], "verdict"),
        }[kind_name]
        return {"entities": entities, "measurements": [measurement],
                "open_slots": [] if subject else ["e1.phrase: name the main subject from the frames"]}
    if kind_name == "camera_rhythm":
        return {"entities": [_camera()], "measurements": [Measurement("m1", "camera_speed", ["cam"], "profile")]}
    if kind_name == "camera_translation":
        return {"entities": [_camera()],
                "measurements": [Measurement("m1", "camera_displacement", ["cam"], "direction", frame="levelled")]}
    if kind_name == "camera_zoom":
        return {"entities": [_camera()], "measurements": [Measurement("m1", "focal", ["cam"], "verdict")]}
    if kind_name == "scene_revisit":
        return {"entities": [_camera()], "measurements": [Measurement("m1", "covisibility", ["cam"], "verdict")]}
    if kind_name == "subject_motion_profile":
        return {"entities": [_entity("e1", "object", slots["subject"])],
                "measurements": [Measurement("m1", "speed", ["e1"], "profile")]}
    if kind_name == "subject_direction":
        # Only an explicit "relative to the camera" fixes the frame; otherwise it is the
        # generator's convention, decided by calibration.
        frame = "camera_t" if "relative to the camera" in question.lower() else "convention"
        return {"entities": [_entity("e1", "object", slots["subject"])],
                "measurements": [Measurement("m1", "displacement", ["e1"], "direction", frame=frame)]}
    if kind_name == "metric_speed":
        # the subject's own motion through the scene, in metres per second: the camera's motion
        # is not the subject's, so the position track is in the world frame
        return {"entities": [_entity("e1", "object", slots["subject"])],
                "measurements": [Measurement("m1", "speed_metric", ["e1"], "value", metric=True)]}
    if kind_name == "pair_relation":
        return _pair(slots, "trend", metric=False)
    if kind_name == "metric_distance":
        return _pair(slots, "value_at", metric=True)
    if kind_name == "instance_count":
        category = singular(sam3_phrase(slots["category"])[0])
        return {"entities": [Entity("c1", "category", category, role="counted")],
                "measurements": [Measurement("m1", "instances", ["c1"], "count")]}
    if kind_name == "event_order":
        entities: list[Entity] = []
        events: list[Event] = []
        for index, slot in enumerate(normalise_event_slots(slots["events"]), start=1):
            if str(slot["actor"]).strip().lower() in ("camera", "the camera"):
                if not any(e.id == "cam" for e in entities):
                    entities.insert(0, _camera())
                actor = "cam"
            else:
                actor = f"e{index}"
                entities.append(_entity(actor, "object", slot["actor"], role="event_actor"))
            # Options are matched to events by their words; a model that leaves `text` out
            # would make "Only the camera motion occurs" unmatchable.
            text = slot.get("text") or f"{slot['actor']} {slot['predicate'].replace('_', ' ')}"
            events.append(Event(f"ev{index}", actor, slot["predicate"], text))
        return {"entities": entities, "events": events,
                "measurements": [Measurement("m1", "onset", ["ev1"], "onset"),
                                 Measurement("m2", "onset", ["ev2"], "onset"),
                                 Measurement("m3", "order", ["m1", "m2"], "order")]}
    if kind_name == "landmark_relation":
        landmark = sam3_phrase(slots["landmark"])[0]
        region = slots.get("landmark_kind") == "region" or (
            slots.get("landmark_kind") is None and bool(set(landmark.split()) & _REGION_WORDS))
        return {"entities": [_entity("e1", "object", slots["subject"]),
                             _entity("l1", "landmark_region" if region else "landmark_object",
                                     slots["landmark"], role="reference")],
                "measurements": [Measurement("m1", "region_distance" if region else "range",
                                             ["e1", "l1"], "trend")]}
    return {}


def _check_category_in_question(slots: Mapping[str, Any], question: str) -> None:
    """A counted or paired category is the question's word, not the model's paraphrase.

    Counting "child" when the question asks for "people" answers a different question
    whenever an adult is in view, so it is rejected (and the model asked again).
    """

    category = slots.get("category")
    if not category:
        return
    words = {singular(w) for w in re.findall(r"[a-z]+", question.lower())}
    wanted = {singular(w) for w in re.findall(r"[a-z]+", str(category).lower())}
    if not wanted & words:
        raise ValueError(f"category {category!r} is not a word from the question; use the "
                         "question's own noun in singular form (e.g. 'people' -> 'person')")


def assemble(qid: str, kind_name: str, slots: Mapping[str, Any], question: str,
             options: Mapping[str, str], source: str) -> QueryProgram:
    """Kind + slots + the question's own words -> a validated Query Program.

    Rules and Qwen both end here, so a program never depends on who read the question --
    only on what they read out of it.
    """

    validate_slots(kind_name, slots)
    kind = KIND_BY_NAME[kind_name]
    text = " ".join(question.split())
    _check_category_in_question(slots, text)
    parts = _template(kind_name, slots, text)
    program = QueryProgram(
        qid=qid,
        kind=kind.name,
        route=kind.route,
        answer_form=kind.answer_form,
        entities=parts.get("entities", []),
        events=parts.get("events", []),
        window=window_from_question(text),
        measurements=parts.get("measurements", []),
        duplicates=duplicate_options(options),
        open_slots=parts.get("open_slots", []),
        source=source,
    )
    program.options = parse_options(options, program.answer_form, program.events)
    unparsed = [o.letter for o in program.options if o.atoms is None]
    if unparsed and program.route == "measure":
        program.notes.append(f"options without atoms: {unparsed}")
    validate(program)
    return program


# --------------------------------------------------------------------------- #
# The rule parser: templated questions need no model
# --------------------------------------------------------------------------- #

SlotReader = Callable[[re.Match[str]], dict[str, Any]]


def _pair_slots(match: re.Match[str]) -> dict[str, Any]:
    groups = match.groupdict()
    if groups.get("cat"):
        return {"category": singular(groups["cat"])}
    return {"a": groups["a"], "b": groups["b"]}


_POSSESSIVE = r"['’]s"
RULES: tuple[tuple[str, re.Pattern[str], SlotReader], ...] = (
    ("camera_path", re.compile(
        r"^what is the (primary|main) (trajectory movement|trajectory|movement trajectory) "
        r"of the camera( movement)?( in the video)?\??$", re.I),
     lambda m: {}),
    ("camera_shake", re.compile(r"^is there noticeable camera shake( in the video)?\??$", re.I),
     lambda m: {}),
    ("cam_subject_range", re.compile(
        r"^how does the distance between the camera and the main subject change\??$", re.I),
     lambda m: {"subject": None, "implicit": True}),
    ("cam_subject_range", re.compile(
        # "How does the distance between the camera and X change?", "Over the entire video, does the
        # camera's distance to X change significantly?", "Over the interval from t=0s to t=5s, does the
        # camera get closer to X?": the same question about one subject, worded three ways
        r"^(?:(?:over|during|throughout) [^,]*, )?(?:how does|does) (?:the )?(?:"
        r"distance between the camera and|camera[’']?s? distance to|camera (?:get|move|come)s? (?:closer to|farther from|further from|toward|away from)) "
        r"(?:the )?(?P<x>.+?)(?: (?:change|remain|get|stay|increase|decrease|over|during|throughout|between|from)\b.*|,.*)?\??$",
        re.I),
     lambda m: {"subject": None, "implicit": True} if re.fullmatch(r"main (subject|object)", m.group("x").lower())
     else {"subject": m.group("x"), "implicit": False}),
    ("cam_subject_range", re.compile(
        r"^(?:over the (?:course of the|entire) video, )?does (?:the )?(?P<x>(?!two\b).+?) get closer\b[^?]*\bthe camera\??$", re.I),
     lambda m: {"subject": m.group("x"), "implicit": False}),
    ("pair_relation", re.compile(
        r"^(?:over the (?:course of the|entire) video, )?do (?:the )?two (?P<cat>[a-z ]+?) get closer\b(?![^?]*\bthe camera\b)[^?]*\??$", re.I),
     _pair_slots),
    ("pair_relation", re.compile(
        r"^(?:over the (?:course of the|entire) video, )?do (?:the )?(?P<a>(?!two\b).+?) and (?:the )?(?P<b>.+?) get closer\b[^?]*\??$", re.I),
     _pair_slots),
    ("subject_direction", re.compile(
        r"^(from|between) (the )?(start|beginning|first frame)( of the video)? (to|and) "
        r"(the )?(end|middle|final frame|last frame|frame at [\d.]+ seconds|[\d.]+-second mark)"
        r"( of the video)?,? how does (the )?(?P<x>.+?)" + _POSSESSIVE +
        r" position change( relative to the camera)?\??$", re.I),
     lambda m: {"subject": m.group("x")}),
    ("pair_relation", re.compile(
        r"^(over the course of the video, )?do (the )?two (?P<cat>[a-z]+) get closer"
        r"( to each other)?,? (or )?farther apart", re.I),
     _pair_slots),
    ("pair_relation", re.compile(
        r"^(over the course of the video, )?do (the )?(?P<a>.+?) and (the )?(?P<b>.+?) get "
        r"closer( to each other| to or farther from each other)?,? or farther", re.I),
     _pair_slots),
    ("metric_distance", re.compile(
        r"^at (approximately |about |around )?(the )?[\d.]+[- ]?(second|seconds|s)( mark)?"
        r"( into the video)?,? what is the (approximate )?distance between the two "
        r"(?P<cat>[a-z]+)\??$", re.I),
     _pair_slots),
    ("metric_distance", re.compile(
        r"^at (approximately |about |around )?(the )?[\d.]+[- ]?(second|seconds|s)( mark)?"
        r"( into the video)?,? what is the (approximate )?distance between (the )?(?P<a>.+?) "
        r"and (the )?(?P<b>.+?)\??$", re.I),
     _pair_slots),
    ("metric_speed", re.compile(
        r"^what is the (?:approximate |estimated |average |mean )*(?:average |mean )?speed of (?:the )?(?P<x>.+?)"
        r"(?: (?:between|from|over|during|throughout|across|in the (?:video|clip))\b.*)?\??$", re.I),
     lambda m: {"subject": m.group("x")}),
    ("tracked_objects_count", re.compile(
        r"^how many distinct [a-z ]*objects\b.*\b(tracked|described|scene context)", re.I),
     lambda m: {}),
    ("instance_count", re.compile(
        r"^how many distinct (?P<cat>[a-z]+) (are |were )?(visible|appear|appearing|present|"
        r"shown|seen|in)\b", re.I),
     lambda m: {"category": singular(m.group("cat"))}),
    ("event_order", re.compile(
        r"^which (event )?happens first( in the video)?:\s*(?P<a>.+?),?\s+or\s+(?P<b>.+?)\??$",
        re.I),
     lambda m: {"events": [event_slot(m.group("a")), event_slot(m.group("b"))]}),
    ("subject_direction", re.compile(
        r"^(?:.*?,\s*)?how does the (?P<x>.+?)" + _POSSESSIVE +
        r" position (?:change )?relative to the camera(?: change)?\b", re.I),
     lambda m: {"subject": m.group("x")}),
    ("landmark_relation", re.compile(
        r"^how does the (?P<x>.+?)" + _POSSESSIVE + r" position change relative to the "
        r"(?P<y>.+?)( as the (scene|video|sequence) (progresses|unfolds|evolves))?\??$", re.I),
     lambda m: {"subject": m.group("x"), "landmark": m.group("y")}),
    ("semantic", re.compile(r"^what best describes the main activity( in the video)?\??$", re.I),
     lambda m: {}),
    ("camera_rhythm", re.compile(
        r"^what is the (movement speed|speed or pacing)\b[^?]*\bcamera\b[^?]*\??$", re.I), lambda m: {}),
    ("camera_translation", re.compile(
        r"^(?:between|over|during|from) [^,?]+, what is (?:the )?(?:camera[’']?s? dominant translation(?: direction)?|"
        r"dominant translation(?: direction)? of the camera)\??$", re.I), lambda m: {}),
    ("camera_zoom", re.compile(r"^was zoom(ing)?\b[^?]*\?$", re.I), lambda m: {}),
    ("scene_revisit", re.compile(
        r"^(are the scenes|is the background scene)\b[^?]*\bsame\b[^?]*\?$", re.I), lambda m: {}),
    ("cam_subject_elevation", re.compile(
        r"^what is the (shooting angle|camera angle) relative to (?:the )?(?P<x>.+?)\?$", re.I),
     lambda m: {"subject": m.group("x"), "implicit": bool(re.fullmatch(r"main (object|subject)", m.group("x").lower()))}),
    ("apparent_size", re.compile(
        r"^(?:over time, |visually, )?does (?:the )?(?P<x>.+?) (?:visually )?(?:appear |get )(?:visually )?"
        r"larger or smaller\b", re.I),
     lambda m: {"subject": m.group("x"), "implicit": bool(re.fullmatch(r"main (object|subject)", m.group("x").lower()))}),
    ("subject_visibility", re.compile(
        r"^(?:at the end of the video, )?is (?:the )?(?P<x>.+?) still in (?:the )?frame"
        r"(?: at the end of the video)?\?$", re.I),
     lambda m: {"subject": m.group("x"), "implicit": bool(re.fullmatch(r"main (object|subject)", m.group("x").lower()))}),
    # "Does the camera remain stationary or move ...?" asks how much and when the *camera* moves: not a subject's motion profile
    ("subject_motion_profile", re.compile(
        r"^does (?!(?:the )?camera\b)(?:the )?(?P<x>[a-z ]+?)(?:['’]s [a-z ]+?)? (?:remain stationary or move|"
        r"(?:[a-z]+ )?(?:movement|motion) occur continuously)", re.I),
     lambda m: {"subject": m.group("x")}),
)


def rule_slots(question: str) -> tuple[str, dict[str, Any]] | None:
    """The kind and slots a rule reads off a templated question, or None."""

    text = " ".join(question.split())
    for kind_name, pattern, read in RULES:
        match = pattern.match(text)
        if match:
            return kind_name, read(match)
    return None


def rule_parse(qid: str, question: str, options: Mapping[str, str]) -> QueryProgram | None:
    """A validated program for a templated question, or None when no rule matches."""

    found = rule_slots(question)
    if found is None:
        return None
    kind_name, slots = found
    return assemble(qid, kind_name, slots, question, options, source="rules")
