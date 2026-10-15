"""Deterministic two-axis taxonomy of the VLM4D questions (no network, no model).

Axis A -- ``paper_type``: the five annotation types of the VLM4D paper (arXiv 2508.02095, Sec. 3.2,
Fig. 2 and supplementary Fig. A): translational, rotational, counting, false_positive, action. The
released QA files do not ship them; the preview parquet labels 9 demo rows with them (plus
"perspective-aware rotation", which is a rotational question asked in the subject's own frame).
``false_positive`` is decided by the GOLD answer (the named object is absent), as the benchmark's own
``acc_final_statistics.is_fp`` does; ``motion_type`` is the same label read from the text alone, i.e.
what an agent can know before answering.

Axis B -- ``family`` / ``kind``: where the question lands in AgentV5's decision tree
(``d4rt_agent/agent_v5/kinds.py``). Kinds AgentV5 already has keep their names; the ones VLM4D needs
and AgentV5 lacks are in ``NEW_KINDS``. Each kind has an ordered sub-task program (``SUBTASKS``) in the
vocabulary of AgentV5's tools; ``program`` adds the per-question steps (existence check, actor
resolution, time window).

Why rules and not an LLM pass (as the Dyn-Bench and DSI explorations used): VLM4D's 1816 questions are
short, human-written and close to templates (1371 real rows have 1161 distinct strings but a few dozen
stems), and the paper publishes the type proportions to check against. Every rule is listed in order
in ``classify``; the residual that no rule reads is ``other`` and is printed in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

PAPER_TYPES = ("translational", "rotational", "counting", "false_positive", "action")

# --------------------------------------------------------------------------- #
# option forms
# --------------------------------------------------------------------------- #

_ABSENT = re.compile(
    r"^(there (is|are) no\b|no such\b|no\b.*\b(there|in the video|in the scene)\.?$"
    r"|no (?!rotation|turn|turning|movement|hands?\b|change)[a-z' -]+$)")
_STATIC = re.compile(
    r"\b(not (moving|spinning|rotating|turning|pedalling)|staying (in place|still)|stationary|standing still"
    r"|no (rotation|turn|turning|movement)|moving straight and not turning|did not move|doesn.t move|did not fall"
    r"|remained in the same|stays? (in )?the same|staying the same|not at all|stand(s|ing) still)\b")
_UNSURE = re.compile(r"\b(not sure|not enough information|neither|unclear|possibly|maybe|both ways|random)\b")
_ROT = re.compile(r"\b(clockwise|counter-?clockwise|anti-?clockwise)\b")
_LATERAL = re.compile(r"\b(left|right)\b")
_DEPTH = re.compile(r"\b(towards?|away|closer|further|farther|nearer)\b")
_VERTICAL = re.compile(r"\b(up|down|upwards?|downwards?|uphill|downhill|above|below|higher|lower)\b")
_OWN = re.compile(r"\b(forwards?|backwards?)\b")
_NUMBER = re.compile(r"^\d+$|^(once|twice|thrice|three times|not at all|\d+ (times|\w+))\.?$")


_EVENT_ABSENT = re.compile(r"\bdid not (touch|use)\b|^never\b|^nothing\b|^none of them\b|^no objects? (are|is) moving\b"
                           r"|^no tape used\b|^did nothing\b")


def is_absent(option: str) -> bool:
    """The option says the asked-about object is not in the video ('no dog there', 'there are no cars...')."""

    return bool(_ABSENT.match(norm(option)))


def is_false_positive(gold: str) -> bool:
    """The gold answer denies that the object exists or that the asked event happened at all (paper: 'whether an
    object or event actually occurred'). 'Not moving' is not one: the object is there and the answer is a motion."""

    g = norm(gold)
    return is_absent(g) or bool(_EVENT_ABSENT.search(g))


def option_form(option: str) -> str:
    """One coarse form per option, in this order: absent, static, unsure, numeric, rotation, lateral, depth,
    vertical, own (forward/backward), other."""

    o = norm(option)
    for name, pat in (("absent", None), ("static", _STATIC), ("unsure", _UNSURE), ("numeric", _NUMBER),
                      ("rotation", _ROT), ("lateral", _LATERAL), ("depth", _DEPTH), ("vertical", _VERTICAL),
                      ("own", _OWN)):
        if (is_absent(o) if pat is None else pat.search(o)):
            return name
    return "other"


def norm(text: Any) -> str:
    return " ".join(str(text).replace("’", "'").replace("‘", "'").lower().split()).strip()


# --------------------------------------------------------------------------- #
# axis A: the paper's annotation type
# --------------------------------------------------------------------------- #

_COUNT_Q = re.compile(r"\bhow many\b(?! degrees)|\bmore than (once|one)\b")
_ROT_Q = re.compile(r"\b(clockwise|counter-?clockwise|anti-?clockwise|clckswise|spin\w*|rotat\w*|revolv\w*|revolutions?"
                    r"|flips?|frontflip|backflip|somersault\w*|turn\w*|trun)\b|\bpan(s|ning)? (his|her|their)\b|\bcamera (view )?pan")
_DIR_Q = re.compile(
    r"\b(left|right|towards?|away|closer|further|farther|furhter|forwards?|backwards?|up|down|upwards?|downwards?"
    r"|uphill|downhill|direction|way|side|where|position|front|behind|above|below|north|south|east|west|sideways"
    r"|vertical\w*|lateral\w*|longitudinal\w*|cross court|down the line|higher|lower|perpendicular|parallel"
    r"|facing|face|leading|faster|slower|speed|quadrant|moving|move|moved|movement|coming|going|heading|end to"
    r"|lean|swipe|sweep|fly|flying|falling|rising)\b")
_HAND_Q = re.compile(r"\bwhich (hand|arm|sleeve)\b|\bleft or right hand\b|\bwith which hand\b|\bwhat (is|does) the (left|right) hand\b"
                     r"|\bto which hand\b|\bswap .* hands\b|\bbetween hands\b")


def motion_type(question: str) -> str:
    """The paper's type from the question text alone (FP cannot be told from the text)."""

    q = norm(question)
    if _COUNT_Q.search(q):
        return "counting"
    if _HAND_Q.search(q):
        return "action"
    if _ROT_Q.search(q):
        return "rotational"
    if _DIR_Q.search(q):
        return "translational"
    return "action"


def paper_type(question: str, gold: str) -> str:
    """False positive when the gold answer says the object or event is absent; otherwise ``motion_type``."""

    return "false_positive" if is_false_positive(gold) else motion_type(question)


def perspective(question: str) -> bool:
    """The paper's 'perspective-aware' flag: the question names the frame it is asked in."""

    return bool(re.search(r"\bperspectives?\b|\bpoint of view\b|\bwith respect to\b|\brespect(ive)? to\b|\brelative to\b"
                          r"|\bcamera frame\b", norm(question)))


# --------------------------------------------------------------------------- #
# axis B: AgentV5 family / kind
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Kind:
    name: str
    family: str
    description: str                # the quantity and its frame
    new: bool = False               # not in d4rt_agent/agent_v5/kinds.py


KINDS: tuple[Kind, ...] = (
    # camera (in Ego4D clips "the person" who turns or walks is usually the camera wearer)
    Kind("camera_path", "camera", "The camera's own turning: pan/yaw left or right, head turns of an ego wearer, "
         "clockwise seen from above = turning right."),
    Kind("camera_translation", "camera", "Which way the camera (ego wearer) moved: left/right/forward/backward."),
    Kind("orbit_direction", "camera", "Which way the camera circles around a subject."),
    # subject_motion
    Kind("subject_direction", "subject_motion", "Which way one entity's position moved: left/right in the image, "
         "toward/away, up/down, uphill/downhill; includes objects moved by hands."),
    Kind("traj_subject_own_frame", "subject_motion", "Forward/backward or left/right turning in the entity's OWN frame."),
    Kind("subject_turn", "subject_motion", "Whether an entity's heading turns left, right or not at all, from the camera's "
         "view or with no frame named.", new=True),
    Kind("rotation_sense", "subject_motion", "Clockwise / counter-clockwise / not spinning: an entity spinning about "
         "its own axis, a hand or tool stroke, or circulation around a course.", new=True),
    # subject_vs_camera
    Kind("cam_subject_range", "subject_vs_camera", "Whether one entity gets closer to or farther from the camera "
         "(in ego video: the wearer approaching a landmark)."),
    Kind("camera_viewpoint", "subject_vs_camera", "Which way an entity faces relative to the camera, or which side the "
         "camera films it from."),
    # two_entities
    Kind("pair_relation", "two_entities", "Whether the distance between two entities (neither the camera) grows or shrinks."),
    Kind("landmark_relation", "two_entities", "Which way an entity moves relative to another entity or landmark "
         "('with respect to the sink', 'towards the cook')."),
    Kind("relative_position", "two_entities", "On which side of B an entity A is, is placed, or passes "
         "(left/right/front/behind/above), at a moment or over a pass.", new=True),
    # count
    Kind("instance_count", "count", "How many distinct instances of a category appear."),
    Kind("count_with_predicate", "count", "How many instances satisfy a per-instance predicate: moving left, facing "
         "the camera, standing still, spinning, dancing.", new=True),
    Kind("repetition_count", "count", "How many times an event happens: jumps, spins, steps, hits, scoops, pick-ups; "
         "often filtered by direction or hand.", new=True),
    # events
    Kind("event_order", "events", "Which of several events or actors comes first, or in which order."),
    # semantic
    Kind("hand_attribution", "semantic", "Which hand (left/right/both) performs an action, or what a hand does.", new=True),
    Kind("semantic", "semantic", "What happens or what something is: decided by appearance, not geometry."),
    Kind("other", "other", "No rule reads it: listed in the report for hand review."),
)
KIND_BY_NAME: dict[str, Kind] = {k.name: k for k in KINDS}
NEW_KINDS: tuple[str, ...] = tuple(k.name for k in KINDS if k.new)

# The ordered sub-task program of each kind, in AgentV5 tool vocabulary. ``program`` prefixes the
# per-question steps; the last step is always read by Qwen against the options.
SUBTASKS: dict[str, tuple[str, ...]] = {
    "camera_path": ("camera_motion(yaw, pitch)", "label(pan left/right, clockwise from above)"),
    "camera_translation": ("camera_motion(translation, wearer frame)", "label(left/right/forward/backward)"),
    "orbit_direction": ("ground(subject)", "camera_motion(around subject, seen from above)", "label(cw/ccw)"),
    "subject_direction": ("ground(subject)", "select(instance)", "entity_track(frame)", "label(lateral, depth, vertical, static)"),
    "traj_subject_own_frame": ("ground(subject)", "entity_track(world)", "orientation(subject)",
                               "label(forward/backward, turn left/right in own frame)"),
    "subject_turn": ("ground(subject)", "entity_track(image or world)", "heading_change(subject)",
                     "label(left/right/straight)"),
    "rotation_sense": ("ground(subject or part)", "rotation_sense(subject: flow curl or orientation yaw, or path area)",
                       "label(cw/ccw/none)"),
    "cam_subject_range": ("ground(subject)", "distance(subject, camera)", "label(closer/farther/unchanged)"),
    "camera_viewpoint": ("ground(subject)", "orientation(subject)", "label(facing left/right/toward/away)"),
    "pair_relation": ("ground(A)", "ground(B)", "distance(A, B)", "label(closer/farther/unchanged)"),
    "landmark_relation": ("ground(A)", "ground(reference)", "entity_track(A, frame=reference)",
                          "label(toward/away/left/right of reference)"),
    "relative_position": ("ground(A)", "ground(reference)", "relative_side(A, reference, moment)",
                          "label(left/right/front/behind/above/below)"),
    "instance_count": ("ground(category, all instances)", "count(distinct tracks)"),
    "count_with_predicate": ("ground(category, all instances)", "per-instance measure(predicate)",
                             "count(instances satisfying it)"),
    "repetition_count": ("ground(actor or object)", "event_series(actor, predicate)", "count(onsets, filtered)"),
    "event_order": ("ground(actors)", "event_onset(each)", "order"),
    "hand_attribution": ("ground(left hand, right hand)", "vlm_read(which hand acts)"),
    "semantic": ("vlm_read(frames)",),
    "other": ("vlm_read(frames)",),
}

_WEARER = r"(the )?(person|he|she|they|man|woman|artist|cook|worker|barber|baker|cashier|driver|this person)"
_EGO_TURN = re.compile(rf"\b(which|what|in which|in what|to which|to what) (way|direction|rotational direction|rotational directions|side)"
                       rf" (does|did|do|is|was) {_WEARER} (turn|go|move|walk|reach|lean|rotate)\b(?! (the|a|his|her|their|its|it|them)\b)"
                       rf"|\b(does|did) {_WEARER} (turn|make a turn)\b|\bhow does the \w+ rotate their body\b"
                       rf"|\b(which|what) (way|direction) (does|did) {_WEARER} turn\b|\bwhich direction does the man turn\b"
                       rf"|\bhow did the person turn\b|\bwhich way (does|did) the person turn\b|\bwhich rotational direction (does|did) the person\b"
                       rf"|\bperson turn\b|\bperson with the camera\b|\bdid the person move to the\b")
_HEAD = re.compile(r"\b(pan|pans|turn|turns) (his|her|their) head\b|\bhead movement\b|\blook (to the|down|at)\b|\bturn his gaze\b|\bturned his head\b")
_EGO_RANGE = re.compile(rf"\b(does|did) {_WEARER} (move|get) (closer|farther|further)\b|\bperson move farther\b")
_CAMERA = re.compile(r"\bcamera (view )?(pan|rotat)\w*|\bdid the camera pan\b")
_ORBIT = re.compile(r"\bcamera rotating .* around\b")
_TAKEN_FROM = re.compile(r"\bvideo (is )?taken from which side\b")
_OWN_FRAME = re.compile(r"\bown perspectives?\b|\bfrom (its|their|his|her) perspectives?\b|\bfrom (the )?(car|cars|driver|drivers|bike rider)'?s'? perspective\b"
                        r"|\bforwards? or backwards?\b")
_TURN = re.compile(r"\b(turn|turning|turned|turining|trun)\b")
_FACING = re.compile(r"\bfacing\b|\bface the same way\b")
_RANGE = re.compile(r"\b(coming closer|closer or|moving closer|getting closer|further away|farther away|furhter away"
                    r"|towards or away|toward the camera|towards the camera|away from the camera|distance between)\b"
                    r"|\bgo toward the camera\b|\bcloser to the camera\b|\bcloser\?$|\bdriving closer\b")
_PAIR = re.compile(r"\beach other\b|\bcloser to the (?!camera)\w+|\bmove closer to the person\b|\btoward the mirror\b")
_REFERENCE = re.compile(r"\b(with )?respect(ive)? to\b|\brelative to\b|\bside of\b|\bwhich side\b|\bwhat side\b|\bon which side\b"
                        r"|\bwhere (is|was|were|did) .*\b(placed|place|pick|with respect)\b|\bplaced\b|\bgo past\b|\bwalk past\b"
                        r"|\bpass by\b|\bwhich side\b|\bwhat position\b|\bquadrant\b|\bwhere is the\b|\bon which stack\b"
                        r"|\bwhich direction was the stack\b|\bfrom which side\b"
                        r"|\bwhich slice\b|\bwhat side\b|\bto which side\b|\bsitting\?$")
_MOVE_VERB = re.compile(r"\b(mov\w*|go|goes|going|went|travel\w*|fly|flies|sweep|scoop\w*|pour\w*|slid\w*|peeled|toss|"
                        r"brought|bring|sliding)\b")
_ORDER = re.compile(r"\b(in which order|sequence|leading|curl up first)\b|(?<!at )\bfirst( move)?\?$")
_SPEED = re.compile(r"\b(faster|slower|same speed)\b")
_IDENTITY = re.compile(r"^(what|which) (object|objects|thing|car|color)\b|\bwhat is heading\b")


def _ego_wearer(q: str, source: str) -> bool:
    """In an Ego4D clip the asked-about person is the camera wearer (heuristic: a person noun that turns or walks)."""

    return source == "ego4d" and bool(_EGO_TURN.search(q) or _HEAD.search(q) or _EGO_RANGE.search(q))


def _count_kind(q: str) -> str:
    if re.search(r"\bhow many (times|full|steps|flips|spins|clockwise|counter-clockwise|keepy|revolutions|spoons|spoonfuls|scoops)\b"
                 r"|\bmore than (once|one)\b|\band how many\b|\bhow many \w+( \w+)? (did|does|do|were|was) (?!there)"
                 r"|\bhow many \w+( \w+)? (went|go)\b|\bhow many [\w ]+ (placed|poured|plucked|thrown|shot)\b", q):
        return "repetition_count"
    if re.search(r"\bhow many [\w -]+? (are|is|were|was|never|moving|going|driving|coming|\w+ed)\b", q):
        return "count_with_predicate"
    return "instance_count"


def classify(question: str, options: Sequence[str], source: str) -> dict[str, Any]:
    """Family, kind and the per-question flags. ``source``: davis | youtube-vos | ego4d | synthetic.

    The rules are ordered; the first that matches decides the kind. Gold answers are never read here."""

    q = norm(question)
    wearer = _ego_wearer(q, source)
    if _COUNT_Q.search(q):
        kind = _count_kind(q)
    elif _HAND_Q.search(q):
        kind = "hand_attribution"
    elif _ORBIT.search(q):
        kind = "orbit_direction"
    elif _CAMERA.search(q) or (wearer and (_HEAD.search(q) or _ROT_Q.search(q))):
        kind = "camera_path"
    elif _EGO_RANGE.search(q) and source == "ego4d":
        kind = "cam_subject_range"
    elif wearer:
        kind = "camera_translation"
    elif _TAKEN_FROM.search(q):
        kind = "camera_viewpoint"
    elif _OWN_FRAME.search(q):
        kind = "traj_subject_own_frame"
    elif _ROT.search(q) or re.search(r"\b(spin\w*|rotat\w*|revolv\w*|clckswise)\b", q):
        kind = "rotation_sense"
    elif _ORDER.search(q):
        kind = "event_order"
    elif _PAIR.search(q):
        kind = "pair_relation"
    elif _RANGE.search(q):
        kind = "cam_subject_range"
    elif _FACING.search(q):
        kind = "camera_viewpoint"
    elif _REFERENCE.search(re.sub(r"\b(with )?respect(ive)? to (the camera|(it'?s|its) original position|the start of the video)\b|\bside of the (camera )?frame\b", "", q)):
        kind = "landmark_relation" if (_MOVE_VERB.search(q) and not re.search(r"\b(placed?|past|pass|sitting)\b", q)) else "relative_position"
    elif _TURN.search(q):
        kind = "subject_turn"
    elif _SPEED.search(q) or _IDENTITY.search(q):
        kind = "other"
    elif re.search(r"\b(frontflip|backflip|flip)\b|\bhow many degrees\b", q):
        kind = "rotation_sense"
    elif motion_type(q) in ("translational", "rotational"):
        kind = "subject_direction"
    else:
        kind = "semantic"
    forms = [option_form(o) for o in options]
    if kind == "subject_turn" and forms.count("rotation") >= 2:      # 'which way did the insect turn?' cw / ccw
        kind = "rotation_sense"
    return {
        "family": KIND_BY_NAME[kind].family,
        "kind": kind,
        "new_kind": KIND_BY_NAME[kind].new,
        "frame": frame_of(q, kind, wearer),
        "window": window_of(q),
        "ego_wearer": wearer,
        "absence_option": "absent" in forms,
        "option_forms": forms,
    }


def frame_of(question: str, kind: str, wearer: bool = False) -> str:
    """The reference frame the question names (or the one its kind implies)."""

    q = norm(question)
    if kind in ("traj_subject_own_frame",) or re.search(r"\bown perspectives?\b", q):
        return "own"
    if wearer or kind in ("camera_path", "camera_translation"):
        return "wearer/camera"
    if re.search(r"\bcamera\b", q) or kind in ("cam_subject_range", "camera_viewpoint", "orbit_direction"):
        return "camera"
    if kind in ("landmark_relation", "relative_position", "pair_relation"):
        return "reference"
    if re.search(r"\b(uphill|downhill|north|south|east|vertical\w*|higher|lower|up the stairs|down the stairs)\b", q):
        return "world"
    if kind in ("subject_direction", "subject_turn", "rotation_sense", "count_with_predicate", "repetition_count"):
        return "unspecified"
    return "none"


def window_of(question: str) -> str:
    q = norm(question)
    if re.search(r"\b(at the start|at first|initially|at the beginning|before)\b", q):
        return "start"
    if re.search(r"\b(at the end|towards the end|after|afterwards|last|finally)\b", q):
        return "end"
    return "whole"


def program(labels: Mapping[str, Any]) -> list[str]:
    """The ordered sub-tasks of one question: actor resolution, existence check, window, then the kind's own."""

    steps: list[str] = []
    if labels["ego_wearer"]:
        steps.append("resolve_actor(person = camera wearer)")
    if labels["absence_option"]:
        steps.append("exists(named object) -> the 'no X' option")
    if labels["window"] != "whole":
        steps.append(f"window({labels['window']})")
    return steps + list(SUBTASKS[labels["kind"]]) + ["final_answer(options)"]
