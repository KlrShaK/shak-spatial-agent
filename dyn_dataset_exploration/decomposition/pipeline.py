"""Run S1 -> S4 for one catalog question from the artifacts on disk (CPU only).

``solve(out, qid)`` is what the notebook calls step by step and what the batch view calls
in a loop; nothing in it needs a GPU because every model output it depends on is an
artifact written by the prep stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping

from dyn_dataset_exploration.decomposition import interpret, measure, scene4d
from dyn_dataset_exploration.decomposition.program import QueryProgram, phrase_ladder, upgrade_events
from dyn_dataset_exploration.decomposition.scene4d import (
    EntityTrack,
    Grounding,
    Scene,
    load_grounding,
    load_scene,
    resolve_entities,
    subsample_scene,
)

PROGRAM_SOURCES = ("pipeline", "qwen", "rules", "reference")

# --- calibrated constants ---------------------------------------------------------------- #
# ``calibrated.json`` (written by run_calibration, committed) holds the constants chosen on the
# dev clips. The pipeline applies it on import, so the notebook, the evidence writer and any
# agent built on this package all run the same numbers; ``reset()`` returns to the uncalibrated
# defaults, which is where a new calibration starts from.

# DYN_CALIBRATED points at another state file (the evaluation uses one whose policy was fitted on dev clips only)
CALIBRATED = Path(os.environ.get("DYN_CALIBRATED") or Path(__file__).with_name("calibrated.json"))
# what answers a kind the policy was never fitted on (semantic questions, kinds with no measurement):
# Qwen's own option probabilities, the best single Qwen mode in the cross-validation
DEFAULT_FALLBACK = "letters"
# Which frames a kind is measured on: "all" of the clip, or only "sampled16", the 16 the benchmark's
# question generator saw. Chosen per kind by calibration; "all" until then.
TEMPORAL_BY_KIND: dict[str, str] = {}
# What answers when the measurement does not: Qwen with the frames and the measured evidence
# ("evidence"), Qwen from the frames alone ("frames"), or the soft fusion of Qwen's option
# probabilities with the measurement's support ("fused"). Chosen per kind on dev (a kind whose
# measurements mislead the model is better left to the frames).
FALLBACK_BY_KIND: dict[str, str] = {}
SCENE_CONSTANTS = ("SCALE_SOURCE", "MIN_LIFESPAN_S", "GAP_MERGE_S", "JUMP_MERGE_SCALES", "SMOOTH_WINDOW_S", "MIN_CORE_CELLS", "DEPTH_BAND")


def snapshot() -> dict[str, Any]:
    from dyn_dataset_exploration.decomposition import scene4d as _scene

    return {
        "t": dict(measure.TUNING), "d": dict(interpret.DECISION),
        "s": {k: getattr(_scene, k) for k in SCENE_CONSTANTS},
        "conv": dict(interpret.DIRECTION_CONVENTION), "trusted": sorted(interpret.TRUSTED_KINDS),
        "gate": {k: dict(v) for k, v in interpret.DECISION_BY_KIND.items()},
        "fusion": {k: dict(v) for k, v in interpret.FUSION.items()},
        "temporal": dict(TEMPORAL_BY_KIND), "fallback": dict(FALLBACK_BY_KIND),
        "fuse": {"kinds": sorted(interpret.FUSE_KINDS), "eps": dict(interpret.FUSE_EPS),
                 "priors": {k: dict(v) for k, v in interpret.PRIORS.items()}},
    }


def apply_state(state: Mapping[str, Any]) -> None:
    from dyn_dataset_exploration.decomposition import scene4d as _scene

    measure.TUNING.update(state.get("t", {}))
    interpret.DECISION.update(state.get("d", {}))
    for k, v in state.get("s", {}).items():
        setattr(_scene, k, v)
    if "conv" in state:
        interpret.DIRECTION_CONVENTION.clear()
        interpret.DIRECTION_CONVENTION.update(state["conv"])
    if "fusion" in state:
        interpret.FUSION.clear()
        interpret.FUSION.update({k: dict(v) for k, v in state["fusion"].items()})
    if "gate" in state:
        interpret.DECISION_BY_KIND.clear()
        interpret.DECISION_BY_KIND.update({k: dict(v) for k, v in state["gate"].items()})
    if "trusted" in state:
        interpret.TRUSTED_KINDS.clear()
        interpret.TRUSTED_KINDS.update(state["trusted"])
    if "fallback" in state:
        FALLBACK_BY_KIND.clear()
        FALLBACK_BY_KIND.update(state["fallback"])
    if "temporal" in state:
        TEMPORAL_BY_KIND.clear()
        TEMPORAL_BY_KIND.update(state["temporal"])
    if "fuse" in state:
        interpret.FUSE_KINDS.clear()
        interpret.FUSE_KINDS.update(state["fuse"]["kinds"])
        interpret.FUSE_EPS.clear()
        interpret.FUSE_EPS.update(state["fuse"]["eps"])
        interpret.PRIORS.clear()
        interpret.PRIORS.update({k: dict(v) for k, v in state["fuse"]["priors"].items()})


DEFAULTS = snapshot()


def reset() -> None:
    apply_state(DEFAULTS)


def apply_calibrated(path: Path | None = None) -> bool:
    path = Path(path or CALIBRATED)
    if not path.exists():
        return False
    apply_state(json.loads(path.read_text())["constants"])
    return True


CALIBRATION_LOADED = apply_calibrated()


@dataclass
class Solution:
    qid: str
    entry: dict[str, Any]
    program: QueryProgram
    scene: Scene | None
    groundings: dict[str, Grounding | None]
    tracks: dict[str, list[EntityTrack]]
    results: dict[str, measure.Result]
    decision: interpret.Decision
    final: str | None                    # the letter this pipeline answers
    final_source: str                    # measurement | evidence | frames | none
    frames_only: str | None
    evidence_answer: str | None
    letters: dict[str, float] | None = None        # Qwen's log-probability of each option letter
    notes: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool | None:
        return None if self.final is None else self.final == self.entry["answer"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid, "kind": self.program.kind, "program_source": self.program.source,
            "leaf": self.decision.leaf, "final": self.final, "final_source": self.final_source,
            "answer": self.entry["answer"], "correct": self.correct,
            "frames_only": self.frames_only, "evidence_answer": self.evidence_answer,
            "decision": self.decision.to_dict(),
            "results": {k: r.to_dict() for k, r in self.results.items()},
            "notes": self.notes,
        }


def load_catalog(out: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((Path(out) / "catalog.json").read_text())
    return {e["qid"]: e for e in payload["questions"]}


class Context:
    """Everything loaded from disk, remembered: a calibration pass re-runs the measurements
    thousands of times and must not re-read a scene or re-lift a mask each time."""

    def __init__(self, out: Path, max_clips: int | None = None) -> None:
        """``max_clips`` bounds memory for a one-pass job: only the most recently used clips keep their
        scene, groundings and tracks (a catalog is ordered clip by clip). None keeps everything, which is
        what a calibration wants because it re-measures the same clips hundreds of times."""

        self.out = Path(out)
        self.max_clips = max_clips
        self._recent: list[str] = []
        self.catalog = load_catalog(self.out)
        self._scenes: dict[tuple[str, str, str], Scene | None] = {}
        self._groundings: dict[tuple[str, str], Grounding | None] = {}
        self._tracks: dict[tuple, dict[str, list[EntityTrack]]] = {}
        self._programs: dict[tuple[str, str], QueryProgram] = {}

    def program(self, qid: str, source: str) -> QueryProgram:
        key = (qid, source)
        if key not in self._programs:
            self._programs[key] = load_program(self.out, qid, source)
        return self._programs[key]

    def _touch(self, slug: str) -> None:
        if self.max_clips is None:
            return
        if slug in self._recent:
            self._recent.remove(slug)
        self._recent.append(slug)
        while len(self._recent) > self.max_clips:
            old = self._recent.pop(0)
            for cache in (self._scenes, self._groundings, self._tracks):
                for key in [k for k in cache if k[0] == old]:
                    del cache[key]

    def scene(self, slug: str, temporal: str = "all") -> Scene | None:
        self._touch(slug)
        key = (slug, temporal, scene4d.BACKEND)        # switching scene4d.BACKEND must not serve a stale scene
        if key not in self._scenes:
            try:
                scene = load_scene(self.out, slug)
            except FileNotFoundError:
                scene = None
            if scene is not None and temporal == "sampled16":
                scene = subsample_scene(scene, scene.clip["sampled16"])
            self._scenes[key] = scene
        return self._scenes[key]

    def grounding(self, slug: str, phrase: str) -> Grounding | None:
        """The instances SAM3 found for ``phrase``; when it found none, for the first simpler
        prompt of the phrase ladder that found some (the grounding says which it used)."""

        key = (slug, phrase)
        if key not in self._groundings:
            found = primary = None
            for rung in phrase_ladder(phrase):
                g = load_grounding(self.out, slug, rung)
                if rung == phrase:
                    primary = g
                if g is not None and g.count > 0:
                    g.used_phrase = rung
                    found = g
                    break
            self._groundings[key] = found or primary
        return self._groundings[key]

    def tracks(self, slug: str, temporal: str, scene: Scene, program: QueryProgram,
               groundings: Mapping[str, Grounding | None]) -> dict[str, list[EntityTrack]]:
        key = (slug, temporal, scene4d.BACKEND, tuple((e.id, e.kind, e.phrase, e.count) for e in program.entities))
        if key not in self._tracks:
            self._tracks[key] = resolve_entities(scene, groundings, program)
        return self._tracks[key]


def load_program(out: Path, qid: str, source: str = "pipeline") -> QueryProgram:
    path = Path(out) / "programs" / f"{qid}.{source}.json"
    data = json.loads(path.read_text())
    if data.get("unsupported"):
        # no kind fits (or no template): nothing to measure, so the model answers from the frames
        return QueryProgram(qid=qid, kind="semantic", route="vlm_only", answer_form="narrative",
                            source=source, notes=[data.get("reason", "no measurable kind")])
    return upgrade_events(QueryProgram.from_dict(data))


def _letters(out: Path, qid: str) -> dict[str, float] | None:
    path = Path(out) / "answers" / f"{qid}.letters.json"
    return json.loads(path.read_text())["logprobs"] if path.exists() else None


def _answer(out: Path, qid: str, mode: str) -> str | None:
    path = Path(out) / "answers" / f"{qid}.{mode}.json"
    return json.loads(path.read_text()).get("answer") if path.exists() else None


def solve(out: Path | Context, qid: str, *, source: str = "pipeline", trusted: set[str] | None = None,
          temporal: str | None = None) -> Solution:
    ctx = out if isinstance(out, Context) else Context(out)
    out = ctx.out
    entry = ctx.catalog[qid]
    program = ctx.program(qid, source)
    if temporal is None:
        temporal = TEMPORAL_BY_KIND.get(program.kind, "all")
    frames_only = _answer(out, qid, "frames_only")
    evidence_answer = _answer(out, qid, "evidence")
    letters = _letters(out, qid)
    notes: list[str] = []
    scene: Scene | None = None
    groundings: dict[str, Grounding | None] = {}
    tracks: dict[str, list[EntityTrack]] = {}
    results: dict[str, measure.Result] = {}

    if program.route == "measure":
        scene = ctx.scene(entry["clip_slug"], temporal)
        if scene is None:
            notes.append("no Point4D scene for this clip")
        else:
            for entity in program.entities:
                if entity.phrase and entity.kind != "camera":
                    groundings[entity.phrase] = ctx.grounding(entry["clip_slug"], entity.phrase)
            tracks = ctx.tracks(entry["clip_slug"], temporal, scene, program, groundings)
            results, tracks = measure.run_program(scene, program, groundings, tracks)
    decision = interpret.decide(program, results, trusted)

    if decision.leaf == "A":
        final, final_source = decision.chosen, "measurement"
    elif decision.leaf == "B":
        mode = FALLBACK_BY_KIND.get(program.kind, DEFAULT_FALLBACK)
        fused = interpret.fuse_for_kind(program.kind, decision.scores, letters) if mode in ("fused", "letters") and letters else None
        if fused is not None:
            final, final_source = fused[0], "fused"
        elif mode == "evidence" and evidence_answer:
            final, final_source = evidence_answer, "evidence"
        else:
            final, final_source = frames_only, "frames"
    else:                                              # leaf C: nothing to measure
        final, final_source = (max(letters, key=letters.get), "letters") if letters else (frames_only, "frames")
    return Solution(qid, entry, program, scene, groundings, tracks, results, decision, final,
                    final_source if final else "none", frames_only, evidence_answer, letters, notes)


def solve_all(out: Path | Context, **kwargs: Any) -> list[Solution]:  # noqa: D103
    ctx = out if isinstance(out, Context) else Context(out)
    return [solve(ctx, qid, **kwargs) for qid in ctx.catalog]


def decision_path(s: Solution) -> list[dict[str, str]]:
    """The decision nodes D0-D7 this question passed through, as a table."""

    p, d = s.program, s.decision
    rows = [
        {"node": "D0 option form", "asks": "how do the options encode the answer?", "decided": p.answer_form},
        {"node": "D1 route", "asks": "does geometry, time or visibility decide it?",
         "decided": "yes -> measure" if p.route == "measure" else "no -> answer from the frames (leaf C)"},
    ]
    if p.route == "vlm_only":
        return rows
    who = ", ".join(f"{e.id}={e.kind}" + (f"({e.phrase!r})" if e.phrase else "") for e in p.entities)
    w = p.window
    when = w.kind + (f" {w.phase}" if w.phase else "") + (f" {w.t0}-{w.t1}s" if w.t0 is not None and w.t1 is not None
                                                         else f" {w.t0}s" if w.t0 is not None else "")
    rows += [
        {"node": "D2 who", "asks": "which entities does the question involve?", "decided": who},
        {"node": "D3 when", "asks": "which part of the clip?", "decided": when},
        {"node": "D4 what + frame", "asks": "which signal, in which reference frame?",
         "decided": "; ".join(f"{m.signal}" + (f" in {m.frame}" if m.frame else "") + (" (metric)" if m.metric else "")
                              for m in p.measurements)},
        {"node": "D5 reduce", "asks": "how is the signal reduced to an answer?",
         "decided": "; ".join(f"{m.id}: {m.reduce}" for m in p.measurements)},
        {"node": "D6 options", "asks": "what claim does each option make?",
         "decided": f"{sum(o.atoms is not None for o in p.options)}/{len(p.options)} options read into claims"},
        {"node": "D7 decide", "asks": "clear margin and every claim checkable?",
         "decided": f"leaf {d.leaf}: {d.reason}"},
    ]
    g = interpret.gate(p.kind)
    if p.kind not in interpret.TRUSTED_KINDS and interpret.TRUSTED_KINDS:
        gate_text = "never alone for this kind (the measurement did not beat the model on dev)"
    elif p.kind in interpret.FUSE_KINDS:
        gate_text = f"posterior margin >= {g['fused_margin']:g} (measurement x this kind's base rate)"
    else:
        gate_text = f"support >= {g['min_score']:g} and margin >= {g['min_margin']:g}"
    fallback = {"evidence": "Qwen with the frames and the measured evidence", "frames": "Qwen from the frames only",
                "fused": "Qwen's option probabilities x the measurement's support",
                "letters": "Qwen's option probabilities (direct answer)"}[FALLBACK_BY_KIND.get(p.kind, DEFAULT_FALLBACK)]
    rows.append({"node": "D7 gate (this kind)", "asks": "how sure must the measurement be to answer alone; who answers otherwise?",
                 "decided": f"{gate_text}; otherwise {fallback}"})
    return rows


def summary_rows(solutions: list[Solution]) -> list[dict[str, Any]]:
    rows = []
    for s in solutions:
        rows.append({
            "qid": s.qid[:46], "kind": s.program.kind, "leaf": s.decision.leaf, "margin": round(s.decision.margin, 2),
            "measured": s.decision.chosen, "final": s.final, "src": s.final_source,
            "gt": s.entry["answer"], "ok": s.correct, "frames_only": s.frames_only,
            "why": s.decision.reason[:70],
        })
    return rows
