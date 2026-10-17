"""Per-clip geometry, named entities, and time labels shared by the V6 tools.

Models are lazy V5 adapters. All interfaces can instead be supplied with small CPU
fixtures; importing this module never loads a model or opens a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import numpy as np

from d4rt_agent.agent_v5 import orientation
from d4rt_agent.agent_v5.scene import (
    EntityTrack, Grounding, Scene, lift_instances, metric_scale, smooth_track,
)


class WorkspaceError(ValueError):
    """A tool can report this recoverable problem to the agent."""


class TimeUnavailable(WorkspaceError):
    pass


class EntityUnavailable(WorkspaceError):
    pass


@dataclass(frozen=True)
class Entity:
    label: str
    category: str
    instances: tuple[int, ...] = ()
    is_camera: bool = False


@dataclass(frozen=True)
class EntityGroup:
    name: str
    grounding: Grounding
    entities: tuple[Entity, ...]
    revision: int


@dataclass(frozen=True)
class Candidate:
    colour: str
    label: str
    instances: tuple[int, ...]


@dataclass(frozen=True)
class CandidateSelection:
    name: str
    revision: int
    choices: tuple[Candidate, ...]


class CandidateRequired(WorkspaceError):
    def __init__(self, selection: CandidateSelection) -> None:
        self.selection = selection
        super().__init__(f"Choose the intended {selection.name} by its colour in the segmented image.")


@dataclass(frozen=True)
class MetricScale:
    metres_per_unit: float | None
    source: str | None
    relative_spread: float | None = None

    @property
    def available(self) -> bool:
        return self.metres_per_unit is not None

    @property
    def units(self) -> str:
        return "metres" if self.available else "scene-relative units"

    def convert(self, values: Any) -> np.ndarray:
        return np.asarray(values) * (self.metres_per_unit if self.available else 1.0)


@dataclass(frozen=True)
class FrameView:
    label: str
    image: np.ndarray = field(repr=False)
    seconds: float | None = None
    index: int = field(default=0, repr=False)


@dataclass
class _Cached:
    revisions: tuple[tuple[str, int], ...]
    value: Any


_COLOURS = ("red", "blue", "green", "yellow", "magenta", "cyan", "orange", "purple",
            "lime", "pink", "teal", "brown")
_CAMERA_NAMES = frozenset({"camera", "cam", "observer", "viewer"})


def normalise_name(name: str) -> str:
    name = " ".join(str(name).strip().casefold().split())
    name = re.sub(r"^(?:the|a|an)\s+", "", name)
    return "camera" if name in _CAMERA_NAMES else name


def _letter_label(number: int) -> str:
    result = ""
    while number >= 0:
        result = chr(65 + number % 26) + result
        number = number // 26 - 1
    return result


class Workspace:
    def __init__(self, clip: Any, models: Any = None, *, scene: Scene | None = None,
                 scene_cache: Any = None, grounder: Callable | None = None,
                 orientation_provider: Callable | None = None) -> None:
        self.clip = clip
        self.models = models
        self._scene = scene
        self._scene_cache = scene_cache
        self._grounder = grounder
        self._orientation_provider = orientation_provider
        self._registered: set[str] = set()
        self._groups: dict[str, EntityGroup] = {}
        self._labels: dict[str, Entity] = {}
        self._selected: dict[str, str] = {}
        self._next_label: dict[str, int] = {}
        self.revisions: dict[str, int] = {}
        self._cache: dict[Any, _Cached] = {}
        self._shown: dict[str, int] = {}
        self._frame_labels: dict[int, str] = {}
        self.pending_candidates: CandidateSelection | None = None
        self.grounding_misses = 0

    @classmethod
    def from_runtime(cls, clip: Any, cache: Path, *, backbone: str = "point4d",
                     qwen_model: str | None = None, **model_options: Any) -> Workspace:
        from d4rt_agent.agent_v5.models import Models

        return cls(clip, Models(qwen_model=qwen_model, backbone=backbone, cache=cache, **model_options))

    def begin_question(self) -> None:
        """Reset conversational choices; retain the clip's expensive observations."""
        for name in tuple(self._selected):
            self.invalidate(name)
        self._selected.clear()
        self.pending_candidates = None
        self.grounding_misses = 0

    def begin_tool_call(self, name: str) -> None:
        if name != "choose_segmentation_candidate":
            self.pending_candidates = None

    def scene(self) -> Scene:
        if self._scene is None:
            cache = self._scene_cache
            if cache is None and self.models is not None:
                cache = self.models.scenes()
            if cache is None:
                raise WorkspaceError("Scene geometry is unavailable.")
            self._scene = cache.scene(self.clip)
        return self._scene

    @property
    def metric(self) -> MetricScale:
        scene = self.scene()
        scale, spread = metric_scale(scene)
        source = scene.metric_source
        # Old D4RT caches claimed native metres. A scale needs an actual estimator.
        if source in (None, "d4rt-native", "d4rt-raw", "point4d-raw") or scale is None \
                or not np.isfinite(scale) or scale <= 0:
            return MetricScale(None, source, spread)
        return MetricScale(float(scale), source, spread)

    @property
    def known_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(("camera", *sorted(self._registered), *self._groups, *self._labels)))

    def register(self, name: str) -> str:
        name = normalise_name(name)
        if not name:
            raise WorkspaceError("An entity name is required.")
        self._registered.add(name)
        return name

    def canonical_name(self, name: str) -> str:
        return normalise_name(name)

    def invalidate(self, name: str) -> None:
        name = normalise_name(name)
        entity = self._labels.get(name)
        category = entity.category if entity is not None else name
        self.revisions[category] = self.revisions.get(category, 0) + 1
        if category in self._groups:
            self._groups[category] = replace(self._groups[category], revision=self.revisions[category])
        self._cache = {key: entry for key, entry in self._cache.items()
                       if category not in dict(entry.revisions)}
        if self.pending_candidates is not None and self.pending_candidates.name == category:
            self.pending_candidates = None

    def cached(self, key: Any, dependencies: Sequence[str], build: Callable[[], Any]) -> Any:
        """Cache derived measurements with all entity dependencies, including pairs."""
        categories = {self._labels.get(normalise_name(name), Entity("", normalise_name(name))).category
                      for name in dependencies}
        revisions = tuple(sorted((name, self.revisions.get(name, 0)) for name in categories))
        existing = self._cache.get(key)
        if existing is None or existing.revisions != revisions:
            self._cache[key] = _Cached(revisions, build())
        return self._cache[key].value

    def ground(self, name: str, *, refresh: bool = False,
               at: float | str | None = None, phrase: str | None = None) -> EntityGroup:
        name = normalise_name(name)
        if name == "camera":
            raise WorkspaceError("The camera is already available; it needs no segmentation.")
        if not name:
            raise WorkspaceError("An entity name is required.")
        if name in self._labels:
            name = self._labels[name].category
        if name in self._groups and not refresh:
            return self._groups[name]
        grounder = self._grounder
        if grounder is None and self.models is not None:
            grounder = self.models.ground
        if grounder is None:
            raise EntityUnavailable(f"Segmentation is unavailable for {name}.")
        prompt_frame = 0 if at is None else self.frame_index(at)
        grounding = grounder(self.clip, phrase or name, prompt_frame)
        if not isinstance(grounding, Grounding):
            raise TypeError("grounder must return a V5 Grounding")
        if grounding.masks.shape[1] != self.clip.length:
            raise ValueError("grounding must cover the clip's complete frame sequence")
        previous = self._groups.get(name)
        entities = self._label_instances(name, grounding, previous)
        self.invalidate(name)
        self._selected.pop(name, None)
        for label, entity in tuple(self._labels.items()):
            if entity.category == name:
                del self._labels[label]
        self._labels.update((entity.label, entity) for entity in entities)
        group = EntityGroup(name, grounding, entities, self.revisions[name])
        self._groups[name] = group
        if not entities:
            self.grounding_misses += 1
        return group

    def reground(self, name: str, *, phrase: str | None = None,
                 at: float | str | None = None) -> EntityGroup:
        return self.ground(name, refresh=True, at=at, phrase=phrase)

    def _label_instances(self, name: str, grounding: Grounding,
                         previous: EntityGroup | None) -> tuple[Entity, ...]:
        active = [i for i in range(grounding.count) if grounding.masks[i].any()]
        matched: dict[int, str] = {}
        # SAM identifiers are session-local. Reuse a label only when the masks agree,
        # never merely because a new SAM session reused the same integer.
        if previous is not None:
            old_masks = previous.grounding.masks
            if old_masks.shape[1:] == grounding.masks.shape[1:]:
                scores = []
                for i in active:
                    for old in previous.entities:
                        mask = old_masks[list(old.instances)].any(axis=0)
                        union = np.count_nonzero(mask | grounding.masks[i])
                        iou = np.count_nonzero(mask & grounding.masks[i]) / max(union, 1)
                        if iou > 0.5:
                            scores.append((iou, i, old.label))
                used = set()
                for _, i, label in sorted(scores, reverse=True):
                    if i not in matched and label not in used:
                        matched[i] = label
                        used.add(label)
        # Initial labels follow appearance, then image position, not confidence rank.
        def appearance(i: int) -> tuple:
            present = np.flatnonzero(grounding.masks[i].any(axis=(1, 2)))
            first = int(present[0])
            xs = np.nonzero(grounding.masks[i, first])[1]
            return first, float(xs.mean()), i

        result = []
        for i in sorted(active, key=appearance):
            label = matched.get(i)
            if label is None:
                number = self._next_label.get(name, 1)
                self._next_label[name] = number + 1
                label = f"{name} {number}"
            result.append(Entity(label, name, (i,)))
        return tuple(result)

    def entities(self, category: str) -> tuple[Entity, ...]:
        return self.ground(category).entities

    def missing_message(self, name: str) -> str:
        if self.grounding_misses <= 1:
            return (f"No object found with the name {normalise_name(name)}. "
                    "Suggest an alternative name: singular, short, descriptive words.")
        return "Still not found. This is not proof that it is absent: check with get_frames or ask_video."

    def resolve(self, name: str) -> Entity:
        name = normalise_name(name)
        if name == "camera":
            return Entity("camera", "camera", is_camera=True)
        if name in self._labels:
            return self._labels[name]
        # A retired/unknown numbered label must not silently become a fresh noun.
        match = re.fullmatch(r"(.+) (\d+)", name)
        if match and match[1] in self._groups:
            raise EntityUnavailable(f"The label {name} is unavailable; inspect {match[1]} again.")
        group = self.ground(name)
        if not group.entities:
            raise EntityUnavailable(self.missing_message(name))
        if len(group.entities) == 1:
            return group.entities[0]
        if name in self._selected:
            return self._labels[self._selected[name]]
        choices = tuple(Candidate(_COLOURS[i] if i < len(_COLOURS) else f"colour {_letter_label(i)}",
                                  entity.label, entity.instances)
                        for i, entity in enumerate(group.entities))
        self.pending_candidates = CandidateSelection(name, self.revisions[name], choices)
        raise CandidateRequired(self.pending_candidates)

    ensure_selection = resolve

    def choose(self, colour: str) -> Entity:
        pending = self.pending_candidates
        if pending is None or self.revisions[pending.name] != pending.revision:
            raise WorkspaceError("There is no current segmentation choice.")
        matches = [choice for choice in pending.choices if choice.colour == colour.strip().casefold()]
        if not matches:
            raise WorkspaceError("Choose one of these colours: " + ", ".join(c.colour for c in pending.choices))
        entity = self._labels[matches[0].label]
        self.invalidate(pending.name)
        self._selected[pending.name] = entity.label
        self.pending_candidates = None
        return entity
