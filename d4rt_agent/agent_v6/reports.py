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


def _relabel_and_merge(phases: Sequence[Phase], noise: float, axes: Sequence[Sequence[str]]) -> list[Phase]:
    """Every combined phase gets its direction from its combined displacement."""
    while True:
        for phase in phases:
            phase.label = _direction(np.asarray(phase.displacement), noise, axes)
        merged = _merge(phases)
        if len(merged) == len(phases):
            return merged
        phases = merged


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
    phases = _relabel_and_merge(phases, noise, axes)
    minimum = float(clock[-1] - clock[0]) * min_phase_fraction
    while len(phases) > 1:
        weak = next((i for i, p in enumerate(phases)
                     if p.duration < minimum and p.distance < strong_noise_multiple * noise), None)
        if weak is None:
            break
        neighbours = [i for i in (weak - 1, weak + 1) if 0 <= i < len(phases)]
        target = max(neighbours, key=lambda i: phases[i].duration)
        phases[weak].label = phases[target].label
        phases = _relabel_and_merge(_merge(phases), noise, axes)
    for phase in phases:
        if phase.label == "still":
            # Discarded weak excursions are not resolved travel. Retain only
            # endpoint drift below the displacement floor in a still phase.
            phase.distance = float(np.linalg.norm(phase.displacement))
    return phases


def majority_direction(phases: Sequence[Phase]) -> str:
    """Largest total duration; distance and first occurrence resolve a tie."""
    totals: dict[str, list[float]] = {}
    for phase in phases:
        item = totals.setdefault(phase.label, [0.0, 0.0])
        item[0] += phase.duration
        item[1] += phase.distance
    return max(totals, key=lambda label: totals[label])


def phase_rows(phases: Sequence[Phase], max_rows: int = MAX_PHASE_ROWS) -> list[tuple[str, str, str]]:
    """Pack long profiles without losing a strong phase or its time span."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if not phases:
        return []
    groups = np.array_split(np.arange(len(phases)), min(max_rows, len(phases)))
    rows = []
    for indices in groups:
        group = [phases[int(i)] for i in indices]
        first, last = group[0], group[-1]
        span = Phase(first.start, last.end, "", [], 0, first.start_label, last.end_label).span
        direction = group[0].label if len(group) == 1 else "; ".join(
            f"{p.label} at {p.span} ({p.distance:.2f})" for p in group)
        rows.append((span, direction, f"{sum(p.distance for p in group):.2f}"))
    return rows


def motion_headline(
    phases: Sequence[Phase], noise: float, *, axes: Sequence[Sequence[str]] = AXES,
) -> str:
    majority = majority_direction(phases)
    net = np.sum([p.displacement for p in phases], axis=0)
    if axes == AXES:
        components = rank_components(net)
    else:
        components = ", ".join(f"{abs(value):.3f} {pair[0] if value >= 0 else pair[1]}"
                               for value, pair in sorted(zip(net, axes), key=lambda x: -abs(x[0])))
    duration = sum(p.duration for p in phases)
    share = sum(p.duration for p in phases if p.label == majority) / duration
    if majority == "still":
        lead = ("STILL throughout" if len(phases) == 1 else "Mostly STILL" if share > 0.5
                else f"STILL for the largest time share ({share:.0%})")
    else:
        when = "most of the clip" if share > 0.5 else f"the largest time share ({share:.0%})"
        lead = f"Moves {majority.upper()} for {when}"
    moved = sum(p.distance for p in phases if p.label == majority)
    magnitude = f"moved {moved:.3f}; " if majority != "still" else ""
    text = f"{lead}: {magnitude}net {components} (noise {noise:.3f})."
    minority = [f"{p.label.upper()} at {p.span} (moved {p.distance:.3f}; noise {noise:.3f})"
                for p in phases if p.label != majority]
    return text + (" " + "; ".join(minority) + "." if minority else "")


def motion_report(
    title: str,
    times: Sequence[float] | None,
    positions: Sequence[Sequence[float]],
    noise: float,
    *,
    time_labels: Sequence[str] | None = None,
    units: str = "scene-relative units",
    metric_source: str | None = None,
    reading_full: str = "",
    reading_short: str = "",
    axes: Sequence[Sequence[str]] = AXES,
    max_rows: int = MAX_PHASE_ROWS,
    **segmentation: Any,
) -> Report:
    phases = segment_motion(times, positions, noise, time_labels=time_labels, axes=axes, **segmentation)
    if units in {"m", "metres", "meters"}:
        if not metric_source:
            raise ValueError("metres require the metric scale source")
        unit_note = f"Distances are metres; scale source: {metric_source}."
    else:
        unit_note = f"Distances are {units}, comparable within this clip only."
    still_note = ('The noise floor applies to Euclidean displacement over a coherent phase. '
                  '"Still" means no sustained motion above that floor; brief weak phases may be absorbed. '
                  'Distances in still phases are unresolved endpoint drift below the floor.')
    if times is None:
        still_note += " Time spans use shown-frame labels; duration is compared by sample order."
    return Report(
        title=title, headline=motion_headline(phases, noise, axes=axes),
        rows=phase_rows(phases, max_rows), columns=("time", "direction", "moved"),
        reading_full=" ".join(filter(None, [reading_full, still_note, unit_note])),
        reading_short=" ".join(filter(None, [reading_short, still_note, unit_note])),
        raw={"times": None if times is None else np.asarray(times).tolist(),
             "positions": np.asarray(positions).tolist(), "noise": float(noise),
             "phases": [asdict(p) for p in phases], "majority": majority_direction(phases),
             "units": units, "metric_source": metric_source},
    )
