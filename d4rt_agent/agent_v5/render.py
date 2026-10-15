"""How measurements read to a language model: small time-ordered tables, a noise floor on every
change, and one chronological line of motion primitives.

Design rules, from the research pipeline and the v5 draft:

* **Seconds, in time order.** Every row is a time span; rows are never sorted by size. A U-shaped
  path reads "forward -> left -> backward" in the order it happened (draft rule 3).
* **A change below its floor is not a motion.** Every change is printed with ``*`` when it is above
  its noise floor, so "stayed put" is readable off the table instead of off a near-zero number.
* **Units are explicit.** Model-relative distances are in *scene scales* (1.0 = the scene's typical
  depth extent); metres appear only when a metric source exists, with that source named.
* **Labels describe, never decide.** A descriptive label (a thresholded reading of the series) is
  printed after the numbers, marked as such; no option text ever appears in a tool's output.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

AXIS_WORDS = (("right", "left"), ("down", "up"), ("forward", "backward"))
MAX_SEGMENTS = 8


def segment_bounds(lo: int, hi: int, n: int = MAX_SEGMENTS) -> list[tuple[int, int]]:
    """``n`` consecutive spans of scene positions covering ``lo..hi`` (fewer on short windows)."""

    n = max(1, min(n, hi - lo))
    edges = np.unique(np.round(np.linspace(lo, hi, n + 1)).astype(int))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def time_label(t_s: np.ndarray | None, pos: int, gf_index: np.ndarray | None = None) -> str:
    return f"{float(t_s[pos]):.2f}s" if t_s is not None else f"f{int(gf_index[pos]) if gf_index is not None else pos}"


def span_label(t_s: np.ndarray | None, a: int, b: int, gf_index: np.ndarray | None = None) -> str:
    return f"{time_label(t_s, a, gf_index)}-{time_label(t_s, b, gf_index)}"


def named_axes(comp: Sequence[float], floor: float, dominance: float) -> dict[str, str | None]:
    """Each axis's direction word when it is above ``floor`` and at least ``dominance`` of the
    largest component; None otherwise. Components are [right, down, forward]."""

    comp = np.asarray(comp, dtype=float)
    out: dict[str, str | None] = {"x": None, "y": None, "z": None}
    if not np.isfinite(comp).all():
        return out
    biggest = float(np.max(np.abs(comp)))
    if biggest < floor:
        return out
    for axis, value, (pos, neg) in zip("xyz", comp, AXIS_WORDS):
        if abs(value) >= dominance * biggest and abs(value) >= floor:
            out[axis] = pos if value >= 0 else neg
    return out


def primitive(words: Mapping[str, str | None]) -> str:
    """'right+forward', or 'still' when no axis is named. Ordered left/right, up/down, forward/backward."""

    named = [w for w in (words.get("x"), words.get("y"), words.get("z")) if w]
    return "+".join(named) if named else "still"


def chronological(spans: Iterable[tuple[str, str, str, float]]) -> str:
    """'0.00s-0.83s forward (0.21) -> 0.83s-1.67s left (0.15) -> ...' with consecutive equal
    primitives merged. Each item is (start label, end label, primitive, magnitude)."""

    merged: list[list[Any]] = []
    for start, end, prim, mag in spans:
        if merged and merged[-1][2] == prim:
            merged[-1][1] = end
            merged[-1][3] += mag
        else:
            merged.append([start, end, prim, mag])
    if not merged:
        return "(no usable span)"
    biggest = max(m[3] for m in merged)
    parts = []
    for start, end, prim, mag in merged:
        flag = " [largest]" if len(merged) > 1 and mag == biggest and prim != "still" else ""
        parts.append(f"{start}-{end} {prim} ({mag:.2f}){flag}")
    return " -> ".join(parts)


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A fixed-width text table."""

    cells = [[_cell(v) for v in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "-"
        return f"{value:+.3f}" if abs(value) < 10 else f"{value:+.1f}"
    return str(value)


def signed(value: float, floor: float) -> str:
    """'+0.123*' when above the floor, '+0.004' when not."""

    if value is None or not np.isfinite(value):
        return "-"
    text = f"{value:+.3f}" if abs(value) < 10 else f"{value:+.1f}"
    return text + ("*" if abs(value) >= floor else "")


def block(title: str, lines: Iterable[str]) -> str:
    body = "\n".join(line for line in lines if line is not None and line != "")
    return f"## {title}\n{body}"


def reliability(confidence: str, notes: Sequence[str]) -> str:
    text = f"Reliability: {confidence}."
    if notes:
        text += " Notes: " + "; ".join(n.rstrip(". ") for n in notes) + "."
    return text
