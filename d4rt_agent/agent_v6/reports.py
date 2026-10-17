"""Shared, option-blind reports with chronological phases and majority-first prose."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from d4rt_agent.agent_v5.render import named_axes, table
from d4rt_agent.sam_orientany_d4rt_test.traj3d_buckets import rank_components

MAX_PHASE_ROWS = 6
FINE_STEPS = 20
MIN_PHASE_FRACTION = 0.075
STRONG_NOISE_MULTIPLE = 3.0
AXES = (("right", "left"), ("down", "up"), ("forward", "backward"))


@dataclass
class Report:
    title: str
    headline: str = ""
    rows: list[Sequence[Any]] = field(default_factory=list)
    columns: Sequence[str] = field(default_factory=tuple)
    reading_full: str = ""
    reading_short: str = ""
    status: str = "ok"
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    images: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in {"ok", "partial", "unavailable"}:
            raise ValueError(f"unknown report status: {self.status}")
        if self.status == "unavailable" and not self.reason:
            raise ValueError("an unavailable report needs a reason")

    def render(self, full: bool = True) -> str:
        """Only the reading, never raw series or image objects, enters the prompt."""
        if self.status == "unavailable":
            return f"{self.title} — unavailable: {self.reason}"
        lines = [self.title, self.headline]
        if self.status == "partial" and self.reason:
            lines.append(f"Partial: {self.reason}")
        if self.rows:
            rendered = table(self.columns, self.rows).splitlines()
            lines.extend(line.rstrip() for i, line in enumerate(rendered) if i != 1)
        notes = self.reading_full if full else self.reading_short
        if notes:
            lines.append("Reading: " + notes.strip())
        return "\n".join(line for line in lines if line)

    @classmethod
    def unavailable(cls, title: str, reason: str, **kwargs: Any) -> "Report":
        return cls(title=title, status="unavailable", reason=reason, **kwargs)


@dataclass
class Phase:
    start: float
    end: float
    label: str
    displacement: list[float]
    distance: float
    start_label: str
    end_label: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def span(self) -> str:
        if self.start_label.endswith(" s") and self.end_label.endswith(" s"):
            return self.start_label[:-2] + "–" + self.end_label
        return self.start_label + "–" + self.end_label


def time_label(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0")
    return (text + "0" if text.endswith(".") else text) + " s"


def load_contract(name: str) -> str:
    """V6 adaptations of V4's geometry contracts, without answer-choice wording."""
    return (Path(__file__).parent / "prompts" / f"contract_{name}.md").read_text().strip()


def _direction(delta: np.ndarray, noise: float, axes: Sequence[Sequence[str]]) -> str:
    # Noise is a Euclidean displacement floor, not a separate floor per axis.
    if float(np.linalg.norm(delta)) <= noise:
        return "still"
    words = named_axes(delta, np.nextafter(0.0, np.inf), dominance=0.5)
    labels = [pair[0] if delta[i] > 0 else pair[1]
              for i, (word, pair) in enumerate(zip(words.values(), axes)) if word]
    return "+".join(labels)


def _motion_knots(clock: np.ndarray, points: np.ndarray, noise: float) -> list[int]:
    """Retain changes whose deviation from a time-linear path exceeds the floor.

    The full input is checked before the equal-time grid is built. Consequently a
    brief excursion cannot fall between grid samples, while slow coherent motion
    survives even when each individual step is smaller than the position noise.
    """
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        first, last = pending.pop()
        if last - first < 2:
            continue
        fraction = (clock[first:last + 1] - clock[first]) / (clock[last] - clock[first])
        line = points[first] + fraction[:, None] * (points[last] - points[first])
        deviation = np.linalg.norm(points[first:last + 1] - line, axis=1)
        middle = first + int(np.argmax(deviation))
        if first < middle < last and deviation[middle - first] > noise:
            keep.add(middle)
            pending.extend(((first, middle), (middle, last)))
    return sorted(keep)


def _merge(phases: Sequence[Phase]) -> list[Phase]:
    merged: list[Phase] = []
    for phase in phases:
        if merged and phase.label == merged[-1].label:
            previous = merged[-1]
            previous.end = phase.end
            previous.end_label = phase.end_label
            previous.displacement = (np.asarray(previous.displacement) + phase.displacement).tolist()
            previous.distance += phase.distance
        else:
            merged.append(Phase(**asdict(phase)))
    return merged


def segment_motion(
    times: Sequence[float] | None,
    positions: Sequence[Sequence[float]],
    noise: float,
    *,
    time_labels: Sequence[str] | None = None,
    n_steps: int = FINE_STEPS,
    min_phase_fraction: float = MIN_PHASE_FRACTION,
    strong_noise_multiple: float = STRONG_NOISE_MULTIPLE,
    axes: Sequence[Sequence[str]] = AXES,
) -> list[Phase]:
    """Label equal-time steps, merge runs, and absorb brief weak excursions.

    Unknown-fps tracks use ordered shown-frame labels and sample order. Strong
    brief motion survives; display packing, rather than deletion, caps rows.
    """
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("motion needs at least two three-dimensional positions")
    if not np.isfinite(points).all():
        raise ValueError("motion positions must be finite")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if times is None:
        if time_labels is None:
            raise ValueError("unknown frame rate requires labels of shown frames")
        clock = np.arange(len(points), dtype=float)
    else:
        clock = np.asarray(times, dtype=float)
    if clock.shape != (len(points),) or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError("times must be finite, strictly increasing, and match positions")
    if time_labels is not None and len(time_labels) != len(points):
        raise ValueError("time labels must match positions")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    knots = _motion_knots(clock, points, noise)
    edges = np.unique(np.concatenate((
        np.linspace(clock[0], clock[-1], min(n_steps, len(points) - 1) + 1), clock[knots])))
    sampled = np.column_stack([np.interp(edges, clock[knots], points[knots, i]) for i in range(3)])
    if times is None:
        labels = [str(time_labels[int(np.argmin(abs(clock - t)))]) for t in edges]
    else:
        labels = [time_label(t) for t in edges]
    steps = np.diff(sampled, axis=0)
    phases = _merge([
        Phase(float(a), float(b), _direction(delta, 0.0, axes), delta.tolist(),
              float(np.linalg.norm(delta)), labels[i], labels[i + 1])
        for i, (a, b, delta) in enumerate(zip(edges[:-1], edges[1:], steps))
    ])
    for phase in phases:
        phase.label = _direction(np.asarray(phase.displacement), noise, axes)
    phases = _merge(phases)
    minimum = float(clock[-1] - clock[0]) * min_phase_fraction
    while len(phases) > 1:
        weak = next((i for i, p in enumerate(phases)
                     if p.duration < minimum and p.distance < strong_noise_multiple * noise), None)
        if weak is None:
            break
        neighbours = [i for i in (weak - 1, weak + 1) if 0 <= i < len(phases)]
        target = max(neighbours, key=lambda i: phases[i].duration)
        phases[weak].label = phases[target].label
        phases = _merge(phases)
    return phases
