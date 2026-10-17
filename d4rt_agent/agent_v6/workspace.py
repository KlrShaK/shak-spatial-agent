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
