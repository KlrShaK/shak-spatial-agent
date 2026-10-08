"""The question's time span, chosen by the decision model from a preset grid.

The grid is the clip's own sampled frame times, so every answer is a valid time and no free text is
parsed. Three small choices, each a quick judgement: what kind of span (whole video, one moment, a
stretch), then the times. ``ask(step, questions)`` is how the caller sends them (and records them).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from d4rt_agent.agent_v5_jev.decisions import choice_question

KINDS = (("whole", "the whole video, or no particular time"),
         ("instant", "one moment (for example 'at 2 s', 'at the start', 'at the end', 'in the first frame')"),
         ("interval", "a stretch from one time to another (for example 'between 1 s and 3 s', "
                      "'from the start to the middle', 'until the frame where ...')"))

Ask = Callable[[str, list[dict]], dict[str, dict]]


def frame_grid(frames: Sequence[int], fps: float) -> list[tuple[str, str]]:
    """[(value, description)] for the clip's sampled frames: the value is the time in seconds."""

    return [(f"{f / fps:.2f}", f"frame {f}, at {f / fps:.2f} s") for f in frames]


def choose_window(ask: Ask, grid: Sequence[tuple[str, str]], clue: str) -> tuple[str, float | None, float | None]:
    """``(kind, t0, t1)``: ("whole", None, None), ("instant", t, None) or ("interval", t0, t1).

    ``clue`` says what to look for (the question stem, or the event that bounds the span)."""

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
