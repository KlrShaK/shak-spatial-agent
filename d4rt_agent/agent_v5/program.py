"""The typed state a question is decomposed into, and the templates that turn it into measurements.

Promoted from ``dyn_dataset_exploration/decomposition/program.py`` (the research record, which stays
frozen). What is kept: the schema vocabularies, the Query Program dataclasses the measurement code
reads, the per-kind slot schemas and templates, and the regex surface facts (timestamps, windows,
phrase shortening). What is left behind on purpose: the rule parser that decided the kind without the
model, and every option parser -- v5's model reads the question and chooses the answer itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

# --------------------------------------------------------------------------- #
# Vocabularies of the schema
# --------------------------------------------------------------------------- #

ENTITY_KINDS = (
    "camera", "object", "group", "category", "main_subject",
    "landmark_object", "landmark_region",
)
WINDOW_KINDS = ("whole", "interval", "instant", "phase", "event", "frames")
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
class FrameSpec:
    basis: str = "scene"
    reference: str | None = None
    anchor_s: float | None = None
    explicit: bool = False


@dataclass
class Requirement:
    id: str
    kind: str
    entities: list[str] = field(default_factory=list)
    frame: FrameSpec = field(default_factory=FrameSpec)
    window: Window = field(default_factory=Window)
    reduction: str = "profile"
    units: str = "relative"
    predicate: str = ""
    count_unit: str | None = None
    depends_on: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


@dataclass
class InstanceSet:
    entity: str
    revision: int
    instances: list[dict[str, Any]] = field(default_factory=list)
    coverage: str = "detected candidates only; missed instances remain possible"


@dataclass
class EventOccurrence:
    id: str
    predicate: str
    participants: list[str]
    start_s: float | None
    end_s: float | None
    evidence_ids: list[str] = field(default_factory=list)
    status: str = "supported"
    complete: bool = True
    onset_bracket_s: list[float] | None = None
    end_bracket_s: list[float] | None = None
    event_id: str | None = None
    start_frame: float | None = None
    end_frame: float | None = None
    onset_bracket_frames: list[float] | None = None
    end_bracket_frames: list[float] | None = None


@dataclass
class EvidenceRecord:
    id: str
    observable: str
    entities: list[str]
    revisions: dict[str, int]
    status: str
    frame: str | None = None
    window: dict[str, Any] | None = None
    units: str | dict[str, str] | None = None
    source: str = "host_measurement"
    requirement: str | None = None
    superseded: bool = False
    component: str | None = None
    source_ids: list[str] = field(default_factory=list)
    reduction: str | None = None
    selected_instances: dict[str, list[int]] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    visual_ids: list[str] = field(default_factory=list)
    available: bool = True
    quantity: str | None = None
    axes: str | None = None


@dataclass
class QueryProgram:
    """What the agent's decomposition turns produced: the kind, the entities and events it named,
    and the window. ``measurements`` holds the tool calls made so far, in the order they ran."""

    qid: str
    kind: str
    route: str
    answer_form: str
    entities: list[Entity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    window: Window = field(default_factory=Window)
    measurements: list[Measurement] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    schema_version: int = 3
    requirements: list[Requirement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def entity(self, entity_id: str) -> Entity | None:
        return next((e for e in self.entities if e.id == entity_id), None)


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


def phrase_in_question(phrase: str, question: str) -> bool:
    """Whether a named entity is the question's own words, not the model's paraphrase.

    Counting "child" when the question asks for "people" answers a different question whenever
    an adult is in view. At least one content word of the phrase (singularised) must appear in
    the question; attributes the model adds from the frames ("red", "left") are allowed.
    """

    words = {singular(w) for w in re.findall(r"[a-z]+", question.lower())}
    wanted = {singular(w) for w in re.findall(r"[a-z]+", phrase.lower())} - {"the", "a", "an"}
    return bool(wanted & words)
