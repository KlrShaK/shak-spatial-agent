"""The question's time span, chosen by the decision model from a preset grid.

The grid is the clip's own sampled frame times, so every answer is a valid time and no free text is
parsed. Three small choices, each a quick judgement: what kind of span (whole video, one moment, a
stretch), then the times. ``ask(step, questions)`` is how the caller sends them (and records them).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from d4rt_agent.agent_v5.program import timestamps

from d4rt_agent.agent_v5_jev.decisions import choice_question

KINDS = (("whole", "the whole video, or no particular time"),
         ("instant", "one moment (for example 'at 2 s', 'at the start', 'at the end', 'in the first frame')"),
         ("interval", "a stretch from one time to another (for example 'between 1 s and 3 s', "
                      "'from the start to the middle', 'until the frame where ...')"))

Ask = Callable[[str, list[dict]], dict[str, dict]]


GRID_POINTS = 32


def window_frames(length: int, n: int = GRID_POINTS) -> list[int]:
    """``n`` frame indices spread evenly over a clip of ``length`` frames, the first and last included."""

    if length <= n:
        return list(range(length))
    return sorted({round(i * (length - 1) / (n - 1)) for i in range(n)})


def frame_grid(frames: Sequence[int], fps: float) -> list[tuple[str, str]]:
    """[(value, description)] for the clip's sampled frames: the value is the time in seconds. The
    first, middle and last times say what they are, so "at the start" and "by the end" have a time to pick."""

    out = []
    for i, f in enumerate(frames):
        note = (" -- the start of the video" if i == 0 else " -- the end of the video" if i == len(frames) - 1
                else " -- the middle of the video" if i == len(frames) // 2 else "")
        out.append((f"{f / fps:.2f}", f"frame {f}, at {f / fps:.2f} s{note}"))
    return out


def choose_window(ask: Ask, grid: Sequence[tuple[str, str]], clue: str) -> tuple[str, float | None, float | None]:
    """``(kind, t0, t1)``: ("whole", None, None), ("instant", t, None) or ("interval", t0, t1).

    ``clue`` says what to look for (the question stem, or the event that bounds the span)."""

    clue = f"{clue} The video runs from {grid[0][0]} s to {grid[-1][0]} s.".strip()
    kind = ask("window_kind", [choice_question(
        "kind", f"What time span does this question refer to?  {clue}", list(KINDS))])["kind"]["choice"]
    if kind == "whole":
        return "whole", None, None
    if kind == "instant":
        at = ask("window_at", [choice_question("at", f"At which of these times? {clue}", list(grid))])["at"]["choice"]
        return "instant", float(at), None
    start = ask("window_start", [choice_question(
        "start", f"At which of these times does the span begin? {clue}", list(grid))])["start"]["choice"]
    later = [g for g in grid if float(g[0]) > float(start)]
    end = (ask("window_end", [choice_question("end", f"At which of these times does the span end? {clue}", later)])
           ["end"]["choice"] if later else grid[-1][0])
    return "interval", float(start), float(end)


# Words that say the question is about a particular time (used to decide when to ask at all).
TIME_HINT = re.compile(r"\b(seconds?|secs?|\d+(\.\d+)?\s*s\b|start|beginning|end\b|halfway|midway|middle|first half|"
                       r"second half|initially|finally|first frame|last frame|frame where|moment)", re.I)


def snap_to_explicit(t: float | None, question: str, grid: Sequence[tuple[str, str]]) -> float | None:
    """A time the model picked from the coarse grid, replaced by the time the question states
    explicitly when one lies within a grid step of it ("at 10.0 s" stays 10.0, not 10.67)."""

    if t is None:
        return None
    times = [float(g[0]) for g in grid]
    step = max((b - a for a, b in zip(times, times[1:])), default=0.0)
    near = [x for x in timestamps(question) if abs(x - t) <= step]
    return min(near, key=lambda x: abs(x - t)) if near else t
