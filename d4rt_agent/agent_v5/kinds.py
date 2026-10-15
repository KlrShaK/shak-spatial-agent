"""The decision tree as the agent walks it: family -> kind -> one slot at a time -> tools.

The research pipeline's kinds (``decomposition/program.py``) are grouped into seven families, so
each choice the model makes is between 2-7 short lines instead of ~20 (one decision per turn: an
8B model asked to do several things at once does each of them worse). A kind fixes which
entities must be named (its *slots*, asked one per turn), the tools the agent may call, and the
playbook it is shown. Two DSI-Bench kinds (own-frame trajectory, camera viewpoint) are added so
AgentV4's benchmark stays covered.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from d4rt_agent.agent_v5.program import (
    EVENT_PREDICATES,
    Entity,
    Event,
    phrase_in_question,
    sam3_phrase,
    singular,
)

COMMON_TOOLS = ("ground", "select", "show", "switch_kind", "done_measuring")

LEGACY_FAMILIES: dict[str, str] = {
    "camera": "Only the camera itself: how it moved, turned, zoomed or shook, or whether the background at the end "
              "is the same place as at the start (same / slight shift / different scene).",
    "subject_motion": "How one entity moves: its direction of travel, whether and how it moves, its speed, or its turning in its own frame.",
    "subject_vs_camera": "One entity relative to the camera: their distance, the camera's height or viewing side, "
                         "whether/where it is in the frame, how large it looks.",
    "two_entities": "The relation between two entities, or an entity and a landmark/region: closer, farther, distance in metres.",
    "count": "How many of something there are.",
    "events": "Which of two named events happens first (camera actions or things an entity does).",
    "semantic": "What is happening or what something is: decided by appearance, not by geometry.",
}


@dataclass(frozen=True)
class Slot:
    name: str
    ask: str                       # the one question this turn asks
    type: str                      # "entity" | "entity_or_camera" | "category" | "choice"
    choices: tuple[str, ...] = ()
    implicit_ok: bool = False      # 'the main subject' may be named from the frames


@dataclass(frozen=True)
class KindSpec:
    name: str
    family: str
    description: str               # the QUANTITY and its FRAME, never an option's wording
    slots: tuple[Slot, ...]
    tools: tuple[str, ...]
    route: str = "measure"


_SUBJECT = Slot("subject", "Name the entity the question is about, in at most 5 words, using the question's own noun "
                "(you may add a colour or position word to tell it apart).", "entity")
_MAIN_SUBJECT = Slot("subject", "Name the entity the question is about, in at most 5 words. If the question only says "
                     "'the main subject/object', name what that is in the frames.", "entity", implicit_ok=True)
_FIRST = Slot("subject", "Name the FIRST entity of the pair, in at most 5 words, using the question's own noun. "
              "If the question says 'the two X', name X.", "entity")
_SECOND = Slot("reference", "Name the SECOND entity of the pair, in at most 5 words, or answer 'camera' if it is the "
               "camera. If the question says 'the two X', name X again.", "entity_or_camera")


def _event_slots() -> tuple[Slot, ...]:
    slots: list[Slot] = []
    for n, which in ((1, "FIRST"), (2, "SECOND")):
        slots.append(Slot(f"event{n}_actor", f"Who performs the {which} event the question names? Answer 'camera', "
                          "or name the entity in at most 5 words.", "entity_or_camera"))
        slots.append(Slot(f"event{n}_predicate", f"Which kind of action is the {which} event?", "choice",
                          choices=EVENT_PREDICATES))
    return tuple(slots)


LEGACY_KINDS: tuple[KindSpec, ...] = (
    # camera
    KindSpec("camera_path", "camera", "The camera's own movement or turning over the clip: pan, tilt, dolly, truck, "
             "pedestal, orbit, static, or a mix.", (), ("camera_motion",)),
    KindSpec("camera_translation", "camera", "Which way the camera's position moved between two moments: left/right, "
             "up/down, forward/backward.", (), ("camera_motion",)),
    KindSpec("orbit_direction", "camera", "Which way the camera circles around a subject, seen from above.",
             (_SUBJECT,), ("camera_motion", "orientation")),
    KindSpec("camera_shake", "camera", "How much high-frequency jitter the camera has: steady versus handheld shake.",
             (), ("camera_profile",)),
    KindSpec("camera_rhythm", "camera", "The camera's speed over time: steady, speeding up, stop-start, or fixed.",
             (), ("camera_profile", "camera_motion")),
    KindSpec("camera_zoom", "camera", "Whether the focal length changed (zoom), as opposed to the camera moving "
             "forward or backward.", (), ("camera_zoom", "camera_motion")),
    KindSpec("scene_revisit", "camera", "Whether the background in the last frames is the same place as in the first "
             "(same, slightly shifted, or a different scene).", (),
             ("scene_revisit", "camera_motion")),
    # subject_motion
    KindSpec("subject_direction", "subject_motion", "Which way one entity's position moved through the scene "
             "(left/right, up/down, forward/backward), between two moments.", (_SUBJECT,), ("entity_track",)),
    KindSpec("traj_subject_own_frame", "subject_motion", "How one entity moved or turned relative to its OWN facing "
             "direction (its forward, its left).", (_SUBJECT,), ("entity_track", "orientation")),
    KindSpec("subject_motion_profile", "subject_motion", "Whether an entity moves at all, and whether continuously, "
             "intermittently, or once.", (_SUBJECT,), ("motion_profile", "entity_track")),
    KindSpec("metric_speed", "subject_motion", "An entity's average speed in metres per second.", (_SUBJECT,),
             ("speed", "motion_profile")),
    # subject_vs_camera
    KindSpec("cam_subject_range", "subject_vs_camera", "Whether the straight-line distance between the camera and "
             "one entity grows, shrinks, holds, or reverses.", (_MAIN_SUBJECT,), ("distance",)),
    KindSpec("cam_subject_elevation", "subject_vs_camera", "The camera's height relative to an entity: above, level, "
             "or below (camera angle).", (_MAIN_SUBJECT,), ("elevation",)),
    KindSpec("camera_viewpoint", "subject_vs_camera", "Which side of an entity (front, back, left, right) the camera "
             "sees or views it from, and how that changes.", (_SUBJECT,), ("orientation", "entity_track")),
    KindSpec("subject_visibility", "subject_vs_camera", "Whether and where an entity is inside the image frame, and "
             "when it enters or leaves.", (_MAIN_SUBJECT,), ("visibility",)),
    KindSpec("apparent_size", "subject_vs_camera", "How large an entity appears in the image over time.",
             (_MAIN_SUBJECT,), ("apparent_size", "distance")),
    # two_entities
    KindSpec("pair_relation", "two_entities", "Whether the distance between two entities, neither of them the "
             "camera, grows, shrinks, or holds.", (_FIRST, _SECOND), ("distance",)),
    KindSpec("metric_distance", "two_entities", "The distance in metres between two entities (or an entity and the "
             "camera) at a moment.", (_FIRST, _SECOND), ("distance",)),
    KindSpec("landmark_relation", "two_entities", "How one entity's position changes relative to a static landmark "
             "or region of the scene (a door, the road, the shoreline).",
             (_SUBJECT,
              Slot("landmark", "Name the landmark or region the entity is compared with, in at most 5 words.", "entity"),
              Slot("landmark_kind", "Is that landmark a compact object (door, lamp, car) or an extended region or "
                   "surface (road, grass, shoreline, room)?", "choice", choices=("object", "region"))),
             ("region_relation", "distance", "entity_track")),
    # count
    KindSpec("instance_count", "count", "How many distinct instances of a named category appear in the clip.",
             (Slot("category", "Name the counted category as a singular noun taken from the question "
                   "(e.g. 'people' -> 'person').", "category"),), ("count",)),
    KindSpec("tracked_objects_count", "count", "How many objects the benchmark's own scene description tracks: "
             "not a visual count.", (), (), route="vlm_only"),
    # events
    KindSpec("event_order", "events", "Which of two named events starts first; an event is a camera motion or "
             "something an entity does.", _event_slots(), ("event_onset", "camera_motion", "visibility")),
    # semantic
    KindSpec("semantic", "semantic", "What is happening or what something is: appearance, not measurement.", (), (),
             route="vlm_only"),
)
FAMILIES = {
    "motion": "An actor's trajectory, body or projected rotation, kinematics or stability.",
    "spatial": "Range, signed relative position, viewpoint, region relation or group configuration.",
    "view": "Visibility, framing, lens zoom or revisiting a view.",
    "entity_sets": "Count or identify physical entities satisfying a predicate.",
    "events": "Locate, order or count event occurrences and repetitions.",
    "visual": "Appearance/activity or actor-object-part interaction, using targeted inspection.",
}
_ACTOR = Slot("subject", "Name the actor (or 'camera'). Use the question/options or visually identify a generic subject.", "entity_or_camera", implicit_ok=True)
_CATEGORY = Slot("category", "Name the category whose instances should be considered.", "category")
_RELATION = (_ACTOR, _SECOND)
KINDS = (
    KindSpec("trajectory", "motion", "Position change or path, including a relative orbit; specify frame and anchor.",
             (_ACTOR,), ("entity_track", "camera_motion", "orbit")),
    KindSpec("body_rotation", "motion", "Body-facing or camera orientation change; not path turning or wheel spin.",
             (_ACTOR,), ("orientation", "camera_motion")),
    KindSpec("projected_rotation", "motion", "Image-plane spin of tracked material points, not body-facing.",
             (_SUBJECT,), ("projected_rotation", "camera_motion")),
    KindSpec("kinematics", "motion", "Path speed, velocity, acceleration or comparison across actors/windows.",
             (_ACTOR,), ("kinematics", "motion_profile")),
    KindSpec("stability", "motion", "Camera pose jitter with estimation quality; low net motion does not imply stability.",
             (), ("camera_profile", "camera_motion")),
    KindSpec("range", "spatial", "Scalar separation or its trend; metric only with scale provenance.",
             _RELATION, ("distance",)),
    KindSpec("relative_position", "spatial", "Signed offsets: left/right, above/below, ahead/behind in an explicit frame.",
             _RELATION, ("relative_position",)),
    KindSpec("viewpoint", "spatial", "Viewing side relative to body orientation, and viewing elevation.",
             (_SUBJECT,), ("orientation", "elevation", "relative_position")),
    KindSpec("region_relation", "spatial", "Along/across/inside a region, distinguishing projection from physical geometry.",
             next(k.slots for k in LEGACY_KINDS if k.name == "landmark_relation"), ("region_geometry", "region_relation")),
    KindSpec("group_configuration", "spatial", "Member separation, dispersion and alignment; not just centroid motion.",
             (_CATEGORY,), ("census", "group_geometry")),
    KindSpec("visibility", "view", "Visibility intervals, image entry/exit and occlusion uncertainty.",
             (_MAIN_SUBJECT,), ("view_state",)),
    KindSpec("framing", "view", "Image location and apparent extent, distinct from physical range.",
             (_MAIN_SUBJECT,), ("view_state", "distance")),
    KindSpec("zoom", "view", "Focal change versus approach or changing subject size.",
             (), ("camera_zoom", "camera_motion")),
    KindSpec("revisit", "view", "Background correspondence between views.", (), ("scene_revisit", "camera_motion")),
    KindSpec("count_entities", "entity_sets", "Distinct identities at a time, in a window, or satisfying a predicate.",
             (_CATEGORY,), ("census", "evaluate_predicate", "count_entities", "encounters")),
    KindSpec("select_entities", "entity_sets", "Identify instances by supported predicates; unknown is not false.",
             (_CATEGORY,), ("census", "evaluate_predicate", "count_entities")),
    KindSpec("locate_events", "events", "Localize events with original predicates and participants.", (), ("locate_event", "record_event", "event_reduce")),
    KindSpec("order_events", "events", "Order any number of localized occurrences, including option-defined events.", (), ("locate_event", "record_event", "event_reduce")),
    KindSpec("count_events", "events", "Count occurrences or completed action cycles; not physical objects.", (), ("locate_event", "record_event", "event_reduce", "encounters")),
    KindSpec("visual_state", "visual", "Appearance, activity or absence with explicit inspected evidence.", (), ("observe",)),
    KindSpec("interaction", "visual", "Actor-object-part relationships, including anatomical roles.", (), ("observe", "record_event")),
)
LEGACY_KIND_BY_NAME = {k.name: k for k in LEGACY_KINDS}
KIND_BY_NAME = {k.name: k for k in (*LEGACY_KINDS, *KINDS)}
COMPOSITIONAL_NAMES = {k.name for k in KINDS}
COMPOSITION_TOOLS = ("ground", "select", "inspect", "observe", "merge_identity", "track_quality",
                     "add_entity", "add_requirement", "event_window", "done_measuring")


def kinds_of(family: str, legacy: bool = False) -> list[KindSpec]:
    return [k for k in (LEGACY_KINDS if legacy else KINDS) if k.family == family]


def family_menu(legacy: bool = False) -> str:
    return "\n".join(f"- `{name}`: {text}" for name, text in (LEGACY_FAMILIES if legacy else FAMILIES).items())


def kind_menu(family: str, legacy: bool = False) -> str:
    return "\n".join(f"- `{k.name}`: {k.description}" for k in kinds_of(family, legacy))


def tools_for(kind: str, legacy: bool = False) -> tuple[str, ...]:
    spec = (LEGACY_KIND_BY_NAME if legacy else KIND_BY_NAME)[kind]
    if not legacy and kind in COMPOSITIONAL_NAMES:
        return tuple(dict.fromkeys(spec.tools + COMPOSITION_TOOLS))
    if spec.route == "vlm_only":
        return ("done_measuring",)
    return spec.tools + COMMON_TOOLS


# --------------------------------------------------------------------------- #
# slot answers -> entities and events
# --------------------------------------------------------------------------- #

_MAIN = re.compile(r"\b(?:main (?:subject|object|character|person)|character|subject|(?:the|this) object)\b", re.I)
_CAMERA = re.compile(r"^(the )?(camera|observer|viewer|cameraman|camera operator)$", re.I)


def is_camera(text: str) -> bool:
    return bool(_CAMERA.match(str(text).strip()))


def validate_slot(slot: Slot, value: Any, question: str) -> str:
    """The normalised slot value, or ValueError with a message the model can act on."""

    if slot.type == "choice":
        text = str(value).strip().lower()
        if text not in slot.choices:
            raise ValueError(f"`value` must be one of {list(slot.choices)}")
        return text
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`value` must be a non-empty phrase")
    text = " ".join(value.strip().split())
    if slot.type == "entity_or_camera" and is_camera(text):
        return "camera"
    if len(text.split()) > 8:
        raise ValueError("name it in at most 5 words")
    if slot.type == "category":
        if not phrase_in_question(text, question):
            raise ValueError(f"{text!r} is not a word from the question; use the question's own noun in singular "
                             "form (e.g. 'people' -> 'person')")
        return singular(sam3_phrase(text)[0])
    implicit = slot.implicit_ok and bool(_MAIN.search(question))
    if not implicit and not phrase_in_question(text, question):
        raise ValueError(f"{text!r} does not use any noun from the question; name the entity with the question's "
                         "own words (you may add a colour or position word)")
    return text


def entities_from_slots(kind: str, slots: Mapping[str, str], question: str, *, legacy: bool = False) -> tuple[list[Entity], list[Event]]:
    """The entities (always including the camera, ``cam``) and events a kind's filled slots define."""

    entities = [Entity("cam", "camera", role="observer")]
    events: list[Event] = []

    def add(eid: str, text: str, kind_name: str, role: str) -> str:
        phrase, note = sam3_phrase(text)
        entities.append(Entity(eid, kind_name, phrase, role=role, note=note))
        return eid

    if "subject" in slots:
        spec = (LEGACY_KIND_BY_NAME if legacy else KIND_BY_NAME)[kind]
        main = spec.slots and any(s.implicit_ok for s in spec.slots) \
            and bool(_MAIN.search(question))
        first = slots["subject"]
        second = slots.get("reference")
        if second is not None and second != "camera" \
                and sam3_phrase(first)[0] == sam3_phrase(second)[0]:
            # 'the two X': one group of two instances of the same category
            phrase = singular(sam3_phrase(first)[0])
            entities.append(Entity("g1", "group", phrase, count=2, role="pair"))
        else:
            if first != "camera":
                add("e1", first, "main_subject" if main else "object", "subject")
            if second == "camera":
                pass                                    # the camera is already an entity
            elif second is not None:
                add("e2", second, "object", "reference")
    if "landmark" in slots:
        region = slots.get("landmark_kind") == "region"
        add("l1", slots["landmark"], "landmark_region" if region else "landmark_object", "reference")
    if "category" in slots:
        entities.append(Entity("c1", "category", slots["category"], role="counted"))
    for n in (1, 2):
        actor = slots.get(f"event{n}_actor")
        if actor is None:
            continue
        actor_id = "cam" if actor == "camera" else add(f"e{n}", actor, "object", "event_actor")
        predicate = slots.get(f"event{n}_predicate", "other")
        events.append(Event(f"ev{n}", actor_id, predicate, f"{actor} {predicate.replace('_', ' ')}"))
    return entities, events


def entity_ids(entities: list[Entity]) -> list[str]:
    """What a tool may be pointed at: 'cam', 'e1', 'e2', 'l1', 'c1', 'g1[0]', 'g1[1]'."""

    ids: list[str] = []
    for e in entities:
        if e.kind == "group":
            ids += [f"{e.id}[{i}]" for i in range(e.count or 2)]
        else:
            ids.append(e.id)
    return ids
