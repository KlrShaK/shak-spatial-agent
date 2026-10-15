"""The tools the agent calls, and the per-question workspace they share.

Every tool takes the workspace and the model's validated arguments and returns a ``ToolResult``:
a text block for the conversation (tables in seconds, noise floors, a descriptive label, caveats),
a one-line summary that replaces it when the history is trimmed, optional images (``show`` only)
and the raw data for the trace. Tools are kind-agnostic -- which ones the agent may call is decided
by the kind's menu (``kinds.tools_for``) -- and none of them ever sees or mentions an answer option.

Entities are named by the ids the decomposition created: ``cam`` (the camera), ``e1``/``e2``,
``g1[0]``/``g1[1]`` (two instances of one category), ``l1`` (a landmark), ``c1`` (a counted
category). A tool pointed at an entity that has not been grounded grounds it first, with the
phrase the agent named, and says what SAM3 found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from d4rt_agent.agent_v5 import measure, orientation, render
from d4rt_agent.agent_v5.measure import TUNING, Result
from d4rt_agent.agent_v5.program import Entity, Measurement, QueryProgram, Window, phrase_ladder
from d4rt_agent.agent_v5.scene import (
    EntityTrack,
    Grounding,
    Scene,
    WindowUnresolved,
    effective_fps,
    lift_instances,
    rank_instances,
    resolve_entities,
    resolve_window,
)


class ToolError(ValueError):
    """The arguments cannot be executed as given; the model is told why and may retry."""


@dataclass
class ToolResult:
    text: str
    summary: str
    images: list[Any] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# the workspace: one question's scene, groundings and tracks
# --------------------------------------------------------------------------- #

class Workspace:
    """What the tools of one question share. Providers are injected so tests run without models:
    ``scene_fn(clip) -> Scene``, ``ground_fn(clip, phrase, prompt_frame) -> Grounding``,
    ``orient_fn() -> OAV2 model``."""

    def __init__(self, clip: Any, program: QueryProgram, *, scene_fn: Callable[[Any], Scene],
                 ground_fn: Callable[[Any, str, int], Grounding], orient_fn: Callable[[], Any] | None = None) -> None:
        self.clip = clip
        self.program = program
        self._scene_fn = scene_fn
        self._ground_fn = ground_fn
        self._orient_fn = orient_fn
        self._scene: Scene | None = None
        self.groundings: dict[str, Grounding] = {}        # entity id -> grounding
        self.used_phrase: dict[str, str] = {}             # entity id -> the prompt that found it
        self.selected: dict[str, list[int]] = {}          # entity id -> instance indices chosen by the agent
        self._tracks: dict[str, list[EntityTrack]] = {}
        self._facing: dict[str, list[dict[str, Any]]] = {}
        self.counter = 0

    # -- scene -------------------------------------------------------------------

    def scene(self) -> Scene:
        if self._scene is None:
            self._scene = self._scene_fn(self.clip)
        return self._scene

    def next_id(self) -> str:
        self.counter += 1
        return f"t{self.counter}"

    # -- entities ----------------------------------------------------------------

    def entity(self, eid: str) -> Entity:
        base = eid.split("[")[0]
        entity = self.program.entity(base)
        if entity is None or entity.kind == "camera":
            raise ToolError(f"unknown entity {eid!r}; use one of {self.entity_choices(include_camera=False)}")
        return entity

    def entity_choices(self, include_camera: bool = True) -> list[str]:
        from d4rt_agent.agent_v5.kinds import entity_ids

        ids = entity_ids(self.program.entities)
        return ids if include_camera else [i for i in ids if i != "cam"]

    def grounding(self, base: str) -> Grounding:
        """The entity's grounding, made on first use with its phrase (then the simpler rungs of
        ``phrase_ladder`` if SAM3 finds nothing)."""

        if base not in self.groundings:
            entity = self.entity(base)
            if not entity.phrase:
                raise ToolError(f"{base} has no phrase to ground; name it with `ground`")
            self._ground(base, entity.phrase, 0)
        return self.groundings[base]

    def _ground(self, base: str, phrase: str, prompt_frame: int) -> Grounding:
        found = None
        for rung in phrase_ladder(phrase):
            g = self._ground_fn(self.clip, rung, prompt_frame)
            if g.count > 0:
                found = g
                self.used_phrase[base] = rung
                break
        if found is None:
            found = Grounding(phrase, np.zeros(0, dtype=int), np.zeros((0, self.clip.length, 1, 1), bool),
                              np.zeros((0, self.clip.length)), np.zeros((0, self.clip.length, 4)),
                              np.zeros((0, self.clip.length)), np.zeros((0, self.clip.length), bool))
            self.used_phrase[base] = phrase
        else:
            found.used_phrase = self.used_phrase[base] if self.used_phrase[base] != phrase else None
            found.phrase = phrase
        self.groundings[base] = found
        self._tracks.pop(base, None)
        self._facing = {k: v for k, v in self._facing.items() if k.split("[")[0] != base}
        return found

    def reground(self, base: str, phrase: str, prompt_frame: int) -> Grounding:
        entity = self.entity(base)
        entity.phrase = phrase
        self.selected.pop(base, None)
        return self._ground(base, phrase, prompt_frame)

    def tracks_of(self, base: str) -> list[EntityTrack]:
        if base in self._tracks:
            return self._tracks[base]
        entity = self.entity(base)
        g = self.grounding(base)
        scene = self.scene()
        if g.count == 0:
            tracks: list[EntityTrack] = []
        elif base in self.selected:
            ks = self.selected[base]
            if entity.kind == "group":
                tracks = [lift_instances(scene, g, [k], entity_id=f"{base}[{i}]", policy="selected")
                          for i, k in enumerate(ks)]
            else:
                tracks = [lift_instances(scene, g, ks, entity_id=base, policy="selected")]
        else:
            stub = QueryProgram(self.program.qid, self.program.kind, "measure", "", entities=[entity])
            tracks = resolve_entities(scene, {entity.phrase: g}, stub).get(base, [])
        for t in tracks:
            if g.used_phrase:
                t.notes.append(f"found only with the simpler prompt {g.used_phrase!r}: it may be another instance")
        self._tracks[base] = tracks
        return tracks

    def track(self, eid: str) -> EntityTrack | str | None:
        """'cam' -> 'camera'; 'g1[1]' -> that member; 'e1' -> its track; None when not found."""

        if eid in ("cam", "camera"):
            return "camera"
        if "[" in eid:
            base, k = eid[:-1].split("[")
            lst = self.tracks_of(base)
            return lst[int(k)] if int(k) < len(lst) else None
        lst = self.tracks_of(eid)
        return lst[0] if lst else None

    def tracks_map(self, *eids: str) -> dict[str, list[EntityTrack]]:
        out: dict[str, list[EntityTrack]] = {}
        for eid in eids:
            if eid in ("cam", "camera"):
                continue
            base = eid.split("[")[0]
            out[base] = self.tracks_of(base)
        return out

    def masks_at(self, eid: str) -> Callable[[int], np.ndarray | None]:
        base = eid.split("[")[0]
        g = self.grounding(base)
        track = self.track(eid)
        instances = track.instances if isinstance(track, EntityTrack) else []
        scene = self.scene()

        def at(pos: int) -> np.ndarray | None:
            if not instances:
                return None
            gf = int(scene.gf_index[pos])
            mask = np.zeros(g.masks.shape[2:], dtype=bool)
            for k in instances:
                mask |= g.masks[k, gf]
            return mask if mask.any() else None

        return at

    def facing(self, eid: str, positions: Sequence[int]) -> list[dict[str, Any]]:
        if self._orient_fn is None:
            raise ToolError("orientation is not available in this run")
        key = f"{eid}@{','.join(map(str, positions))}"
        if key not in self._facing:
            self._facing[key] = orientation.facing_rows(self._orient_fn(), self.clip, self.scene(),
                                                        self.masks_at(eid), positions)
        return self._facing[key]

    # -- windows -------------------------------------------------------------------

    def window(self, args: Mapping[str, Any]) -> Window:
        """The tool's window: ``window: [start_s, end_s]`` or ``at_s``, else the question's own."""

        if args.get("at_s") is not None:
            return Window("instant", float(args["at_s"]))
        if args.get("window") not in (None, "", []):
            w = args["window"]
            # The model often writes the list as a string ("[3.0, 6.0]", "3-6 s"): read the numbers.
            values = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", w)] if isinstance(w, str) else list(w) \
                if isinstance(w, (list, tuple)) else []
            if len(values) != 2:
                raise ToolError("`window` must be two times in seconds, e.g. [3.0, 6.0]; or leave it out to use "
                                "the question's own time span")
            a, b = float(values[0]), float(values[1])
            return Window("interval", min(a, b), max(a, b))
        return self.program.window

    def span(self, window: Window) -> tuple[int, int, str]:
        try:
            return resolve_window(self.scene(), window)
        except WindowUnresolved as error:
            raise ToolError(f"{error}; call it without a window to use the whole clip") from error


# --------------------------------------------------------------------------- #
# helpers shared by the tools
# --------------------------------------------------------------------------- #

def _who(ws: Workspace, eid: str) -> str:
    if eid in ("cam", "camera"):
        return "the camera"
    entity = ws.program.entity(eid.split("[")[0])
    label = entity.phrase if entity and entity.phrase else eid
    if "[" in eid:
        label += f" #{int(eid.split('[')[1][:-1]) + 1}"
    return f"{eid} ({label})"


def _units_line(scene: Scene) -> str:
    text = "Distances are in scene scales (1.0 = the scene's typical depth extent; model-relative)."
    if scene.metres_per_unit:
        metres = scene.metres_per_unit * scene.scale
        spread = f", spread {scene.metric_iqr:.0%}" if scene.metric_iqr else ""
        text += f" 1 scene scale is about {metres:.2f} m (metric source: {scene.metric_source}{spread})."
    return text


def _unavailable(title: str, reason: str, result: Result | None = None) -> ToolResult:
    text = render.block(title, [f"Could NOT be measured: {reason}.",
                                "Answer this part from the frames, and say the measurement was unavailable."])
    return ToolResult(text, f"{title}: unavailable ({reason})", data=result.to_dict() if result else {"available": False, "reason": reason})


def _not_found(ws: Workspace, eid: str, title: str) -> ToolResult | None:
    track = ws.track(eid)
    if track is None:
        base = eid.split("[")[0]
        g = ws.groundings.get(base)
        if g is None or g.count == 0:
            return _unavailable(title, f"the segmenter found no {ws.entity(base).phrase!r}; "
                                       "try `ground` with another phrase or a time where it is visible")
        return _unavailable(title, f"{eid} has no usable track")
    return None


def _measure_id(ws: Workspace) -> str:
    return ws.next_id()


def _label(text: str) -> str:
    return f"Descriptive label (a thresholded reading of the numbers above; check it against them): {text}"


def _common_notes(r: Result) -> list[str]:
    notes = list(r.notes)
    if r.window_note and r.window_note != "whole clip":
        notes.insert(0, f"window: {r.window_note}")
    return notes


def _first_present(track: EntityTrack, lo: int, hi: int) -> np.ndarray:
    idx = np.flatnonzero(track.present & np.isfinite(track.smooth).all(1))
    return idx[(idx >= lo) & (idx <= hi)]


# --------------------------------------------------------------------------- #
# grounding tools
# --------------------------------------------------------------------------- #

def grounding_text(ws: Workspace, base: str) -> str:
    g = ws.grounding(base)
    scene = ws.scene()
    entity = ws.entity(base)
    if g.count == 0:
        return f"{base}: SAM3 found nothing for {entity.phrase!r} (also tried {phrase_ladder(entity.phrase)[1:]})."
    rows = []
    for row in rank_instances(scene, g, "salient" if entity.kind == "main_subject" else "best"):
        k = row["k"]
        if row["frames"] == 0:
            continue
        a, b = row["span"]
        boxes = g.boxes[k][scene.gf_index][g.present(k)[scene.gf_index]]
        cx = float(np.nanmean((boxes[:, 0] + boxes[:, 2]) / 2)) if len(boxes) else float("nan")
        cy = float(np.nanmean((boxes[:, 1] + boxes[:, 3]) / 2)) if len(boxes) else float("nan")
        rows.append([k, render.span_label(scene.t_s, a, b, gf_index=scene.gf_index), f"{row['presence']:.0%}", f"{row['score']:.2f}",
                     f"{row['area']:.1%}", _region_name(cx, cy)])
    used = ws.used_phrase.get(base, entity.phrase)
    head = f"{base}: SAM3 found {g.count} instance(s) for {used!r}" + \
        (f" (the full phrase {entity.phrase!r} found none)" if used != entity.phrase else "") + "."
    chosen = ws.selected.get(base)
    tracks = ws.tracks_of(base)
    using = chosen if chosen is not None else sorted({k for t in tracks for k in t.instances})
    tail = (f"Measurements use instance(s) {using}" + (" (your selection)." if chosen is not None else
            " (chosen automatically: longest-lived and most confident; fragments of one entity merged). "
            "Use `select` to choose others, `show` to see them."))
    return "\n".join([head, render.table(["inst", "visible", "of frames", "score", "area", "where"], rows), tail])


def _region_name(cx: float, cy: float) -> str:
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return "-"
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    return "centre" if (row, col) == ("middle", "centre") else f"{row}-{col}"


def tool_ground(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    base = str(args["entity"]).split("[")[0]
    ws.entity(base)
    phrase = str(args.get("phrase") or ws.entity(base).phrase or "").strip()
    if not phrase:
        raise ToolError("give a `phrase` to ground")
    frame = 0
    if args.get("at_s") is not None:
        if ws.clip.fps is None:
            raise ToolError("this clip has no established frame rate; omit `at_s`")
        frame = int(min(max(round(float(args["at_s"]) * ws.clip.fps), 0), ws.clip.length - 1))
    ws.reground(base, phrase, frame)
    text = grounding_text(ws, base)
    g = ws.groundings[base]
    return ToolResult(render.block(f"ground {base}", [text]), f"ground {base} {phrase!r}: {g.count} instance(s)",
                      data={"entity": base, "phrase": phrase, "instances": int(g.count)})


def tool_select(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    base = str(args["entity"]).split("[")[0]
    entity = ws.entity(base)
    g = ws.grounding(base)
    ks = [int(k) for k in args["instances"]]
    bad = [k for k in ks if not 0 <= k < g.count]
    if bad or not ks:
        raise ToolError(f"instances must be among 0..{g.count - 1}")
    if entity.kind == "group" and len(ks) != (entity.count or 2):
        raise ToolError(f"{base} is a pair: select exactly {entity.count or 2} instances, one per member")
    ws.selected[base] = ks
    ws._tracks.pop(base, None)                     # noqa: SLF001 -- re-lift with the selection
    return ToolResult(render.block(f"select {base}", [grounding_text(ws, base)]), f"select {base} {ks}",
                      data={"entity": base, "instances": ks})


# --------------------------------------------------------------------------- #
# camera tools
# --------------------------------------------------------------------------- #

def tool_camera_motion(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    title = "camera_motion"
    scene = ws.scene()
    window = ws.window(args)
    m = Measurement(_measure_id(ws), "camera_motion", ["cam"], "dominant_motion", window=window)
    r = measure.measure_camera_motion(scene, window, m)
    if not r.available:
        return _unavailable(title, r.reason, r)
    lo, hi = r.window
    cam = measure.camera_state(scene)
    t = scene.t_s
    rot_noise = max(1.0, _residual_noise(cam["pan"], cam["pan_smooth"]))
    trans_noise = _residual_noise3(cam["trans"], cam["trans_smooth"]) / scene.scale
    trans_floor = max(0.02, 3 * trans_noise)
    rows, spans = [], []
    for a, b in render.segment_bounds(lo, hi):
        dpan = float(cam["pan_smooth"][b] - cam["pan_smooth"][a])
        dtilt = float(cam["tilt_smooth"][b] - cam["tilt_smooth"][a])
        droll = float(cam["roll_smooth"][b] - cam["roll_smooth"][a])
        d = (cam["trans_smooth"][b] - cam["trans_smooth"][a]) / scene.scale     # [right, down, forward]
        rows.append([render.span_label(t, a, b, gf_index=scene.gf_index), render.signed(dpan, rot_noise), render.signed(dtilt, rot_noise),
                     render.signed(droll, rot_noise), render.signed(d[0], trans_floor),
                     render.signed(-d[1], trans_floor), render.signed(d[2], trans_floor)])
        spans.append((render.time_label(t, a, gf_index=scene.gf_index), render.time_label(t, b, gf_index=scene.gf_index),
                      _camera_primitive(dpan, dtilt, d, rot_noise, trans_floor),
                      max(abs(dpan) / 10.0, float(np.linalg.norm(d)) / 0.1)))
    v = r.value
    disp = (cam["trans_smooth"][hi] - cam["trans_smooth"][lo]) / scene.scale
    words = render.named_axes(disp, TUNING["cam_dir_floor_scales"], TUNING["cam_dir_dominance"])
    label = f"dominant motion '{v['motion']}'" + (f" {v['direction']}" if v.get("direction") else "")
    if v["motion"] == "static":
        label += (" -- no measured component exceeded the descriptive floor; inspect the values and pose quality "
                  "before concluding the camera is fixed")
    if v.get("orbit"):
        label += " (orbit: lateral translation while turning back toward the centre; cw = clockwise seen from above)"
    lines = [
        "Camera rotation is relative to its start; pan + = turning right, tilt + = up, roll + = rotating clockwise "
        "(the image content turns counter-clockwise). Translation is in the gravity-levelled frame of the first "
        "camera: right/left, up/down, forward/backward. " + _units_line(scene),
        f"Changes marked * exceed the noise floor (rotation {rot_noise:.1f} deg, translation {trans_floor:.3f}).",
        render.table(["time", "pan deg", "tilt deg", "roll deg", "right", "up", "forward"], rows),
        f"Over the window: peak pan {v['pan_deg']:+.1f} deg, peak tilt {v['tilt_deg']:+.1f} deg, net translation "
        f"right {disp[0]:+.3f}, up {-disp[1]:+.3f}, forward {disp[2]:+.3f} (net direction: {render.primitive(words)}).",
        "In time order: " + render.chronological(spans),
        _label(label + f". Thresholds: a pan/tilt counts from {TUNING['cam_pan_deg']:.0f}/{TUNING['cam_tilt_deg']:.0f} "
               f"deg, a translation from {TUNING['cam_trans_scales']:.2f} scene scales."),
        render.reliability(r.confidence, _common_notes(r)),
    ]
    return ToolResult(render.block(title, lines), f"camera_motion: {label}; peak pan {v['pan_deg']:+.0f} deg, "
                      f"net move {render.primitive(words)}", data={**r.to_dict(),
                      "status": "unresolved" if v["motion"] == "unresolved" else "partial"})


def _camera_primitive(dpan: float, dtilt: float, d: np.ndarray, rot_floor: float, trans_floor: float) -> str:
    parts = []
    if abs(dpan) >= max(rot_floor, 2.0):
        parts.append("pan " + ("right" if dpan > 0 else "left"))
    if abs(dtilt) >= max(rot_floor, 2.0):
        parts.append("tilt " + ("up" if dtilt > 0 else "down"))
    named = render.named_axes(d, trans_floor, TUNING["cam_dir_dominance"])
    if any(named.values()):
        parts.append("move " + render.primitive(named))
    return " & ".join(parts) if parts else "still"


def _residual_noise(raw: np.ndarray, smooth: np.ndarray) -> float:
    x = np.asarray(raw, float)
    ok = np.isfinite(x) & np.isfinite(smooth)
    if ok.sum() < 3:
        return 0.0
    resid = measure._unwrapped_deg(x)[ok] - smooth[ok]                       # noqa: SLF001
    return float(1.4826 * np.median(np.abs(resid - np.median(resid))))


def _residual_noise3(raw: np.ndarray, smooth: np.ndarray) -> float:
    resid = np.linalg.norm(raw - smooth, axis=1)
    resid = resid[np.isfinite(resid)]
    return float(1.4826 * np.median(np.abs(resid - np.median(resid)))) if len(resid) > 2 else 0.0


def tool_camera_profile(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    title = "camera_profile"
    scene = ws.scene()
    window = ws.window(args)
    m = Measurement(_measure_id(ws), "camera_speed", ["cam"], "profile", window=window)
    rhythm = measure.measure_rhythm(scene, window, m)
    jitter = measure.measure_camera_jitter(scene, window, m)
    if not rhythm.available and not jitter.available:
        return _unavailable(title, rhythm.reason or jitter.reason)
    lines = []
    if rhythm.available:
        lo, hi = rhythm.window
        cam = measure.camera_state(scene)
        t = scene.t_s
        fps = effective_fps(scene)
        rows = []
        for a, b in render.segment_bounds(lo, hi):
            dt = max(float(t[b] - t[a]) if t is not None else (b - a) / fps, 1e-6)
            trans = float(np.linalg.norm(cam["trans_smooth"][b] - cam["trans_smooth"][a])) / scene.scale / dt
            rot = float(np.hypot(cam["pan_smooth"][b] - cam["pan_smooth"][a],
                                 cam["tilt_smooth"][b] - cam["tilt_smooth"][a])) / dt
            energy = trans / TUNING["rhythm_trans_floor_per_s"] + rot / TUNING["rhythm_rot_floor_deg_s"]
            rows.append([render.span_label(t, a, b, gf_index=scene.gf_index), f"{trans:.3f}", f"{rot:.1f}", f"{energy:.1f}",
                         "moving" if energy >= 1.0 else "still"])
        v = rhythm.value
        lines += [
            "Camera speed per time span: translation in scene scales per second, rotation in degrees per second, and "
            "their combined 'motion energy' where 1.0 is the threshold for 'moving'.",
            render.table(["time", "move/s", "turn deg/s", "energy", "state"], rows),
            f"Mean energy by thirds of the window: {np.round(v['thirds'], 2).tolist()}; variation (cv) {v['cv']:.2f}.",
            _label(f"speed profile '{v['rhythm'].replace('_', ' ')}'"),
        ]
    if jitter.available:
        v = jitter.value
        names = ["stable", "slight shake", "shaky", "severe shake"]
        lines += [
            f"Shake: rotation jitter {v['rotation_residual_deg']:.2f} deg rms and translation jitter "
            f"{v['translation_residual_scales']:.4f} scene scales rms after removing the smooth path "
            f"(levels from {TUNING['jitter_slight_deg']}/{TUNING['jitter_shaky_deg']}/{TUNING['jitter_severe_deg']} deg).",
            _label(f"shake '{names[v['level']]}'"),
        ]
    notes = (_common_notes(rhythm) if rhythm.available else []) + (jitter.notes if jitter.available else [])
    conf = rhythm.confidence if rhythm.available else jitter.confidence
    lines.append(render.reliability(conf, notes))
    summary = "camera_profile: " + ", ".join(
        x for x in (f"rhythm {rhythm.value['rhythm']}" if rhythm.available else "",
                    f"jitter {jitter.value['rotation_residual_deg']:.2f} deg" if jitter.available else "") if x)
    return ToolResult(render.block(title, lines), summary,
                      data={"rhythm": rhythm.to_dict(), "jitter": jitter.to_dict()})


def tool_camera_zoom(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    title = "camera_zoom"
    scene = ws.scene()
    window = ws.window(args)
    r = measure.measure_zoom(scene, window, Measurement(_measure_id(ws), "focal", ["cam"], "verdict", window=window))
    if not r.available:
        return _unavailable(title, r.reason, r)
    lo, hi = r.window
    cam = measure.camera_state(scene)
    fx = scene.K[:, 0, 0]
    t = scene.t_s
    tol = TUNING["zoom_rel_tol"]
    rows = []
    for a, b in render.segment_bounds(lo, hi):
        ratio = float(fx[b] / fx[lo])
        fwd = float(cam["trans_smooth"][b, 2] - cam["trans_smooth"][a, 2]) / scene.scale
        rows.append([render.span_label(t, a, b, gf_index=scene.gf_index), f"x{ratio:.3f}", render.signed(fwd, 0.05)])
    v = r.value
    lines = [
        "Focal length relative to the window's start (x1.10 = 10% longer = zoomed in), and the camera's forward "
        "movement in each span (a dolly forward also enlarges the view but leaves the focal length unchanged).",
        render.table(["time", "focal", "forward"], rows),
        f"Focal ratio at the end x{v['focal_ratio_end']:.3f} (range x{v['focal_ratio_min']:.3f}..x{v['focal_ratio_max']:.3f}); "
        f"net forward movement {v['forward_over_scale']:+.3f} scene scales.",
        _label(f"zoom '{v['zoom']}' (a change beyond {tol:.0%} counts)"),
        render.reliability(r.confidence, _common_notes(r) + ["focal length is estimated from the reconstruction; "
                                                              "small changes are not reliable"]),
    ]
    return ToolResult(render.block(title, lines), f"camera_zoom: {v['zoom']} (focal x{v['focal_ratio_end']:.2f})",
                      data=r.to_dict())


def tool_scene_revisit(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    title = "scene_revisit"
    scene = ws.scene()
    window = ws.window(args)
    r = measure.measure_revisit(scene, window, Measurement(_measure_id(ws), "covisibility", ["cam"], "verdict",
                                                           window=window))
    if not r.available:
        return _unavailable(title, r.reason, r)
    v = r.value
    lines = [
        f"{v['shared_fraction']:.0%} of the first view's 3D points are seen again, at a consistent depth, from the last "
        f"camera; {v['in_view_fraction']:.0%} fall inside the last view at all. Between first and last frame the camera "
        f"turned {v['pan_deg']:+.0f} deg and moved {v['translation_over_scale']:.2f} scene scales.",
        _label(f"view '{v['revisit']}' (same from {TUNING['revisit_same_frac']:.0%} shared, "
               f"different below {TUNING['revisit_diff_frac']:.0%})"),
        render.reliability(r.confidence, _common_notes(r)),
    ]
    return ToolResult(render.block(title, lines), f"scene_revisit: {v['revisit']} ({v['shared_fraction']:.0%} shared)",
                      data=r.to_dict())


# --------------------------------------------------------------------------- #
# entity tools
# --------------------------------------------------------------------------- #

FRAME_TEXT = {
    "scene": "in the scene's frame: the FIRST camera's axes with gravity as vertical (right/left, up/down, "
             "forward = away from where the camera first looked). Camera motion is removed.",
    "camera": "as the moving camera sees it (right/left in the picture, forward = away from the camera). Camera "
              "motion is NOT removed: a still object looks like it moves when the camera does.",
    "image": "in the picture: right/left and up/down as fractions of the image width/height; forward from the "
             "depth to the camera.",
    "own": "in the entity's OWN frame: forward = the way it faces at its first usable frame, right = its right "
           "(facing from Orient-Anything-V2).",
}
FRAME_CAVEAT = {
    "scene": "left/right in this frame matches how the benchmark words directions about 9 times in 10; "
             "forward/backward from a single camera's depth is close to a guess -- decide that part from the frames.",
    "camera": "forward/backward (toward/away from the camera) is the depth estimate; left/right is reliable.",
    "image": "image left/right/up/down is reliable; depth is an estimate.",
    "own": "the entity's facing is estimated per frame; a symmetric object has no reliable front.",
}


def tool_entity_track(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    eid = str(args["entity"])
    frame = str(args.get("frame") or "scene")
    if frame not in FRAME_TEXT:
        raise ToolError(f"`frame` must be one of {list(FRAME_TEXT)}")
    title = f"entity_track {eid} ({frame} frame)"
    ws.entity(eid)
    missing = _not_found(ws, eid, title)
    if missing:
        return missing
    track = ws.track(eid)
    scene = ws.scene()
    lo, hi, note = ws.span(ws.window(args))
    idx = _first_present(track, lo, hi)
    if len(idx) < 2:
        return _unavailable(title, "the entity has a 3D position on fewer than 2 frames of the window")
    basis = None
    if frame == "own":
        rows_f = ws.facing(eid, list(idx[np.unique(np.linspace(0, len(idx) - 1, min(8, len(idx))).astype(int))]))
        usable = [r for r in rows_f if r.get("usable")]
        if not usable:
            return _unavailable(title, "Orient-Anything never resolved a unique front for this entity")
        basis = orientation.subject_basis(usable[0])

    def comp(p0: int, p1: int) -> np.ndarray:
        if frame == "scene":
            return scene.to_level(track.smooth[p1] - track.smooth[p0]) / scene.scale
        if frame == "camera":
            return (scene.to_camera(track.smooth[p1], p1) - scene.to_camera(track.smooth[p0], p0)) / scene.scale
        if frame == "image":
            z = (scene.to_camera(track.smooth[p1], p1) - scene.to_camera(track.smooth[p0], p0))[2] / scene.scale
            c0, c1 = track.centroid[p0], track.centroid[p1]
            return np.array([c1[0] - c0[0], c1[1] - c0[1], z])
        return ((track.smooth[p1] - track.smooth[p0]) @ basis) / scene.scale

    floor = max(TUNING["move_floor_scales"], 3 * track.noise / scene.scale)
    floor_xy = TUNING["image_move_floor"] if frame == "image" else floor
    t = scene.t_s
    rows, spans = [], []
    for a, b in render.segment_bounds(int(idx[0]), int(idx[-1])):
        inside = idx[(idx >= a) & (idx <= b)]
        if len(inside) < 2:
            rows.append([render.span_label(t, a, b, gf_index=scene.gf_index), "-", "-", "-", "-", "not visible"])
            continue
        d = comp(int(inside[0]), int(inside[-1]))
        words = render.named_axes(np.array([d[0], d[1], d[2]]), min(floor_xy, floor), TUNING["axis_dominance"]) \
            if frame != "image" else _image_words(d, floor_xy, floor)
        prim = render.primitive(words)
        size = float(np.linalg.norm(d))
        rows.append([render.span_label(t, int(inside[0]), int(inside[-1]), gf_index=scene.gf_index), render.signed(d[0], floor_xy),
                     render.signed(-d[1], floor_xy), render.signed(d[2], floor), f"{size:.3f}", prim])
        spans.append((render.time_label(t, int(inside[0]), gf_index=scene.gf_index), render.time_label(t, int(inside[-1]), gf_index=scene.gf_index), prim, size))
    net = comp(int(idx[0]), int(idx[-1]))
    net_words = render.named_axes(net, floor, TUNING["axis_dominance"]) if frame != "image" \
        else _image_words(net, floor_xy, floor)
    path_len = sum(s[3] for s in spans)
    xy_unit = "image fraction" if frame == "image" else "scene scales"
    lines = [
        f"{_who(ws, eid)}, positions {FRAME_TEXT[frame]}",
        f"Columns: right (+) / left (-), up (+) / down (-) in {xy_unit}; forward (+) / backward (-) in scene scales. "
        + _units_line(scene),
        f"Changes marked * exceed the noise floor ({floor_xy:.3f} sideways/vertical, {floor:.3f} depth).",
        render.table(["time", "right", "up", "forward", "moved", "direction"], rows),
        f"Net from first to last visible frame: right {net[0]:+.3f}, up {-net[1]:+.3f}, forward {net[2]:+.3f} "
        f"(net direction: {render.primitive(net_words)}); summed path {path_len:.3f}.",
        "In time order: " + render.chronological(spans),
        _label(f"net '{render.primitive(net_words)}'; describe the motion in the order above, not by the largest part"),
        render.reliability("high" if track.confident[idx].mean() > 0.6 else "medium",
                           ([f"window: {note}"] if note != "whole clip" else [])
                           + ([f"visible only {render.span_label(t, int(idx[0]), int(idx[-1]), gf_index=scene.gf_index)} of the window"]
                              if (idx[0] - lo) > 0.25 * max(hi - lo, 1) or (hi - idx[-1]) > 0.25 * max(hi - lo, 1)
                              else []) + track.notes + [FRAME_CAVEAT[frame]]),
    ]
    summary = f"entity_track {eid} ({frame}): " + render.chronological(spans)[:160]
    return ToolResult(render.block(title, lines), summary,
                      data={"entity": eid, "frame": frame, "net": [float(x) for x in net], "series_used": "smoothed",
                            "rows": rows, "noise_floor": floor})


def _image_words(d: np.ndarray, floor_xy: float, floor_z: float) -> dict[str, str | None]:
    words = render.named_axes(np.array([d[0], d[1], 0.0]), floor_xy, TUNING["axis_dominance"])
    words["z"] = ("forward" if d[2] > 0 else "backward") if abs(d[2]) >= floor_z else None
    return words


def tool_distance(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    a = str(args["a"])
    b = str(args.get("b") or "cam")
    title = f"distance {a} to {b}"
    for op in (a, b):
        if op not in ("cam", "camera"):
            ws.entity(op)
            missing = _not_found(ws, op, title)
            if missing:
                return missing
    scene = ws.scene()
    window = ws.window(args)
    at = args.get("at_s") is not None
    m = Measurement(_measure_id(ws), "range", [a, "cam" if b == "camera" else b], "value_at" if at else "trend",
                    window=window, metric=at)
    r = measure.measure_range(scene, ws.program, m, ws.tracks_map(a, b))
    if not r.available:
        return _unavailable(title, r.reason, r)
    rng = np.asarray(r.series["range"], float)
    t = scene.t_s
    mpu = scene.metres_per_unit
    v = r.value
    if at:
        lines = [
            f"Distance between {_who(ws, a)} and {_who(ws, b)} at {render.time_label(t, v['frame'], gf_index=scene.gf_index)}: "
            f"{v['units'] / scene.scale:.3f} scene scales" + (f" = about {v['metres']:.2f} m" if "metres" in v else
                                                               " (no metric scale for this clip)") + ".",
            _units_line(scene),
            "Metric depth from one camera is uncertain: expect up to a factor of about 2 either way, so use it to "
            "rule out values several times too large or too small, not to split close ones.",
            render.reliability(r.confidence, _common_notes(r)),
        ]
        summary = f"distance {a}-{b} at {render.time_label(t, v['frame'], gf_index=scene.gf_index)}: " + \
            (f"{v['metres']:.2f} m" if "metres" in v else f"{v['units'] / scene.scale:.3f} scales")
        return ToolResult(render.block(title, lines), summary, data=r.to_dict())
    lo, hi = r.window
    ok = np.flatnonzero(np.isfinite(rng))
    ok = ok[(ok >= lo) & (ok <= hi)]
    tol_rel = float(np.exp(r.floor["noise_log"] * TUNING["range_noise_mult"]) - 1) if r.floor.get("noise_log") else 0.0
    tol_rel = max(tol_rel, TUNING["pair_rel_tol"] if ws.program.kind == "pair_relation" else TUNING["range_rel_tol"])
    smooth = np.asarray(r.series["smooth"], float)
    rows, spans = [], []
    for s, e in render.segment_bounds(int(ok[0]), int(ok[-1])):
        inside = ok[(ok >= s) & (ok <= e)]
        if len(inside) < 2:
            rows.append([render.span_label(t, s, e, gf_index=scene.gf_index), "-", "-", "-", "not both visible"])
            continue
        r0, r1 = float(smooth[inside[0]]), float(smooth[inside[-1]])
        change = r1 / r0 - 1.0
        word = "still" if abs(change) < tol_rel / 2 else ("closer" if change < 0 else "farther")
        rows.append([render.span_label(t, int(inside[0]), int(inside[-1]), gf_index=scene.gf_index),
                     f"{r0 / scene.scale:.3f}" + (f" ({r0 * mpu:.1f} m)" if mpu else ""),
                     f"{r1 / scene.scale:.3f}" + (f" ({r1 * mpu:.1f} m)" if mpu else ""),
                     f"{change:+.0%}", word])
        spans.append((render.time_label(t, int(inside[0]), gf_index=scene.gf_index), render.time_label(t, int(inside[-1]), gf_index=scene.gf_index), word, abs(change)))
    lines = [
        f"Straight-line distance between {_who(ws, a)} and {_who(ws, b)} over time (smoothed). " + _units_line(scene),
        f"A change counts when it exceeds about {tol_rel:.0%} of the distance (per span, half that is shown as a move).",
        render.table(["time", "from", "to", "change", "reading"], rows),
        f"Start {v['start_units'] / scene.scale:.3f} -> end {v['end_units'] / scene.scale:.3f} scene scales "
        f"(end/start x{v['ratio_end_over_start']:.2f}).",
        "In time order: " + render.chronological(spans),
        _label(f"trend '{v['trend']}'" + (f", levelling off after {v['plateau_from']:.0%} of the window"
                                           if v.get("plateau_from") is not None else "")),
        render.reliability(r.confidence, _common_notes(r)),
    ]
    return ToolResult(render.block(title, lines), f"distance {a}-{b}: {v['trend']} (x{v['ratio_end_over_start']:.2f})",
                      data=r.to_dict())


def tool_region_relation(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    eid, lid = str(args["entity"]), str(args.get("landmark") or "l1")
    title = f"region_relation {eid} to {lid}"
    for op in (eid, lid):
        ws.entity(op)
    missing = _not_found(ws, eid, title)
    if missing:
        return missing
    landmark = ws.entity(lid)
    g = ws.grounding(lid)
    if g.count == 0:
        return _unavailable(title, f"the segmenter found no {landmark.phrase!r}")
    scene = ws.scene()
    window = ws.window(args)
    m = Measurement(_measure_id(ws), "region_distance", [eid, lid], "trend", window=window)
    stub = QueryProgram(ws.program.qid, ws.program.kind, "measure", "", entities=[landmark])
    r = measure.measure_region_distance(scene, stub, m, ws.tracks_map(eid), {landmark.phrase: g})
    if not r.available:
        return _unavailable(title, r.reason, r)
    dist = np.asarray(r.series["smooth"], float)
    ok = np.flatnonzero(np.isfinite(dist))
    t = scene.t_s
    rows = []
    for s, e in render.segment_bounds(int(ok[0]), int(ok[-1])):
        inside = ok[(ok >= s) & (ok <= e)]
        if len(inside) < 2:
            continue
        d0, d1 = dist[inside[0]] / scene.scale, dist[inside[-1]] / scene.scale
        rows.append([render.span_label(t, int(inside[0]), int(inside[-1]), gf_index=scene.gf_index), f"{d0:.3f}", f"{d1:.3f}",
                     render.signed(d1 - d0, r.value["tolerance_units"] / scene.scale)])
    v = r.value
    lines = [
        f"Distance from {_who(ws, eid)} to the nearest visible surface point of {_who(ws, lid)} (not its centre). "
        + _units_line(scene),
        render.table(["time", "from", "to", "change"], rows),
        _label(f"trend '{v['trend']}' (changes under {v['tolerance_units'] / scene.scale:.3f} scales are not counted)"),
        render.reliability(r.confidence, _common_notes(r)),
    ]
    return ToolResult(render.block(title, lines), f"region_relation {eid}-{lid}: {v['trend']}", data=r.to_dict())


def tool_elevation(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    eid = str(args["entity"])
    title = f"elevation of the camera over {eid}"
    ws.entity(eid)
    missing = _not_found(ws, eid, title)
    if missing:
        return missing
    scene = ws.scene()
    window = ws.window(args)
    m = Measurement(_measure_id(ws), "elevation", [eid], "verdict", window=window)
    r = measure.measure_elevation(scene, ws.program, m, ws.tracks_map(eid))
    if not r.available:
        return _unavailable(title, r.reason, r)
    elev = np.asarray(r.series["elevation"], float)
    ok = np.flatnonzero(np.isfinite(elev))
    t = scene.t_s
    rows = [[render.span_label(t, s, e, gf_index=scene.gf_index), f"{np.nanmedian(elev[s:e + 1]):+.0f}"]
            for s, e in render.segment_bounds(int(ok[0]), int(ok[-1])) if np.isfinite(elev[s:e + 1]).any()]
    v = r.value
    lines = [
        f"Angle of the camera above (+) or below (-) the horizon as seen from {_who(ws, eid)}, against gravity.",
        render.table(["time", "elevation deg"], rows),
        f"Median {v['median_deg']:+.0f} deg, range {v['min_deg']:+.0f}..{v['max_deg']:+.0f} deg.",
        _label(f"camera '{v['angle']}' (eye level within +-{TUNING['elev_eye_deg']:.0f} deg)"),
        render.reliability(r.confidence, _common_notes(r)),
    ]
    return ToolResult(render.block(title, lines), f"elevation {eid}: {v['angle']} ({v['median_deg']:+.0f} deg)",
                      data=r.to_dict())


def tool_visibility(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    eid = str(args["entity"])
    title = f"visibility of {eid}"
    ws.entity(eid)
    track = ws.track(eid)
    if track is None:
        g = ws.groundings.get(eid.split("[")[0])
        lines = [f"The segmenter found no {ws.entity(eid).phrase!r} on any frame." if g is None or g.count == 0
                 else f"{eid} has no usable track.",
                 "If the frames show it, it may be too small or too unusual for the segmenter; try `ground` with "
                 "another phrase or a time where it is visible."]
        return ToolResult(render.block(title, lines), f"visibility {eid}: not found", data={"found": False})
    scene = ws.scene()
    window = ws.window(args)
    lo, hi, note = ws.span(window)
    m = Measurement(_measure_id(ws), "presence", [eid], "verdict", window=window)
    r = measure.measure_visibility(scene, ws.program, m, ws.tracks_map(eid))
    t = scene.t_s
    rows = []
    for s, e in render.segment_bounds(lo, hi):
        seg = slice(s, e + 1)
        pres = track.present[seg]
        if not pres.any():
            rows.append([render.span_label(t, s, e, gf_index=scene.gf_index), "0%", "-", "-", "-"])
            continue
        c = track.centroid[seg][pres]
        rows.append([render.span_label(t, s, e, gf_index=scene.gf_index), f"{pres.mean():.0%}", f"{track.border[seg][pres].mean():.0%}",
                     f"{np.median(track.area_frac[seg][pres]):.1%}",
                     _region_name(float(np.mean(c[:, 0])), float(np.mean(c[:, 1])))])
    first, last = track.first, track.last
    enter = "in view from the first frame" if first == 0 else (
        f"first visible at {render.time_label(t, first, gf_index=scene.gf_index)}" if first is not None else "never visible")
    leave = "still in view on the last frame" if last == scene.length - 1 else (
        f"last visible at {render.time_label(t, last, gf_index=scene.gf_index)}" if last is not None else "")
    gaps = _gaps(track.present)
    lines = [
        f"Where {_who(ws, eid)} is in the picture: share of frames it is visible, share of those where it touches the "
        "image border (cut off), its median size (share of the image), and where in the picture it is.",
        render.table(["time", "visible", "at border", "size", "where"], rows),
        f"{enter}; {leave}." + (f" Not visible during: {', '.join(render.span_label(t, a, b, gf_index=scene.gf_index) for a, b in gaps)}."
                                if gaps else ""),
    ]
    if r.available:
        v = r.value
        lines.append(_label(f"in the asked window it is '{v['visible']}' (visible in {v['present_fraction']:.0%} of "
                            "frames; 'partial' = mostly cut off by the border or much smaller than usual)"))
    lines.append(render.reliability(r.confidence if r.available else "medium",
                                    ([f"window: {note}"] if note != "whole clip" else []) + track.notes
                                    + ["the segmenter can miss an entity on a few frames (blur, occlusion)"]))
    return ToolResult(render.block(title, lines), f"visibility {eid}: {enter}; {leave}",
                      data=r.to_dict() if r.available else {"available": False})


def _gaps(present: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(present)
    if len(idx) < 2:
        return []
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 2:
            out.append((int(a + 1), int(b - 1)))
    return out


def tool_apparent_size(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    eid = str(args["entity"])
    title = f"apparent_size of {eid}"
    ws.entity(eid)
    missing = _not_found(ws, eid, title)
    if missing:
        return missing
    scene = ws.scene()
    window = ws.window(args)
    m = Measurement(_measure_id(ws), "area", [eid], "shape", window=window)
    r = measure.measure_apparent_size(scene, ws.program, m, ws.tracks_map(eid))
    if not r.available:
        return _unavailable(title, r.reason, r)
    area = np.asarray(r.series["area"], float)
    ok = np.flatnonzero(np.isfinite(area))
    t = scene.t_s
    rows = [[render.span_label(t, s, e, gf_index=scene.gf_index), f"{np.nanmedian(area[s:e + 1]):.2%}"]
            for s, e in render.segment_bounds(int(ok[0]), int(ok[-1])) if np.isfinite(area[s:e + 1]).any()]
    v = r.value
    lines = [
        f"How much of the picture {_who(ws, eid)} covers over time (frames where it is cut off by the border are "
        "left out when possible).",
        render.table(["time", "share of image"], rows),
        f"From {v['area_start']:.2%} to {v['area_end']:.2%} (end/start x{v['area_ratio_end']:.2f}).",
        _label(f"looks '{v['size']}' (a change beyond {TUNING['size_rel_tol']:.0%} counts)"),
        render.reliability(r.confidence, _common_notes(r)),
    ]
    return ToolResult(render.block(title, lines), f"apparent_size {eid}: {v['size']} (x{v['area_ratio_end']:.2f})",
                      data=r.to_dict())


def tool_motion_profile(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    eid = str(args["entity"])
    title = f"motion_profile of {eid}"
    ws.entity(eid)
    missing = _not_found(ws, eid, title)
    if missing:
        return missing
    scene = ws.scene()
    window = ws.window(args)
    m = Measurement(_measure_id(ws), "speed", [eid], "profile", window=window)
    r = measure.measure_motion_profile(scene, ws.program, m, ws.tracks_map(eid))
    if not r.available:
        return _unavailable(title, r.reason, r)
    speed, moving, at = (np.asarray(r.series[k]) for k in ("speed", "moving", "frames"))
    floor = r.floor["speed_per_s"]
    t = scene.t_s
    rows = []
    for s, e in render.segment_bounds(int(np.floor(at[0])), int(np.ceil(at[-1]))):
        sel = (at >= s) & (at <= e)
        if not sel.any():
            continue
        rows.append([render.span_label(t, s, e, gf_index=scene.gf_index), f"{np.mean(speed[sel]):.3f}", f"{np.mean(moving[sel]):.0%}"])
    v = r.value
    lines = [
        f"Speed of {_who(ws, eid)} through the scene (camera motion removed), in scene scales per second; "
        f"'moving' = above {floor:.3f}/s.",
        render.table(["time", "speed/s", "moving"], rows),
        f"Moving {v['moving_fraction']:.0%} of the time in {v['runs']} run(s); visible on {v['present_fraction']:.0%} "
        "of frames.",
        _label(f"motion '{v['motion']}'"),
        render.reliability(r.confidence, _common_notes(r)),
    ]
    return ToolResult(render.block(title, lines), f"motion_profile {eid}: {v['motion']} ({v['runs']} run(s))",
                      data=r.to_dict())


def tool_speed(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    eid = str(args["entity"])
    title = f"speed of {eid}"
    ws.entity(eid)
    missing = _not_found(ws, eid, title)
    if missing:
        return missing
    scene = ws.scene()
    window = ws.window(args)
    m = Measurement(_measure_id(ws), "speed_metric", [eid], "value", metric=True, window=window)
    r = measure.measure_metric_speed(scene, ws.program, m, ws.tracks_map(eid))
    if not r.available:
        return _unavailable(title, r.reason, r)
    v = r.value
    if "mps" in v:
        head = (f"{_who(ws, eid)} averages about {v['mps']:.2f} m/s over {v['duration_s']:.1f} s: start-to-end "
                f"displacement {v['net_mps']:.2f} m/s, distance actually travelled {v['path_mps']:.2f} m/s "
                f"(metric source: {scene.metric_source}).")
    else:
        head = (f"{_who(ws, eid)} averages {v['units_per_s'] / scene.scale:.3f} scene scales per second over "
                f"{v['duration_s']:.1f} s; there is no metric scale for this clip.")
    lines = [head, "The camera's own motion is not counted. " + _units_line(scene),
             "Metric speed from one camera is uncertain: expect up to a factor of about 2; use it to rule out options "
             "several times too fast or too slow, and check against what the entity is (a walking person is ~1.4 m/s).",
             render.reliability(r.confidence, _common_notes(r))]
    summary = f"speed {eid}: " + (f"{v['mps']:.2f} m/s" if "mps" in v else f"{v['units_per_s'] / scene.scale:.3f} scales/s")
    return ToolResult(render.block(title, lines), summary, data=r.to_dict())


def tool_count(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    eid = str(args.get("entity") or "c1")
    entity = ws.entity(eid)
    title = f"count of {entity.phrase}"
    g = ws.grounding(eid)
    scene = ws.scene()
    tracks = ws.tracks_of(eid)
    m = Measurement(_measure_id(ws), "instances", [eid], "count")
    stub = QueryProgram(ws.program.qid, ws.program.kind, "measure", "", entities=[entity])
    r = measure.measure_instances(scene, m, {eid: tracks}, {entity.phrase: g}, stub)
    if not r.available:
        return _unavailable(title, r.reason)
    t = scene.t_s
    rows = [[i + 1, render.span_label(t, tr.first, tr.last), f"{tr.present.mean():.0%}",
             f"{np.median(tr.area_frac[tr.present]):.1%}"] for i, tr in enumerate(tracks)]
    v = r.value
    lines = [
        f"Distinct instances of {entity.phrase!r} the segmenter tracked (fragments of one instance that disappears and "
        f"comes back are merged; tracklets shorter than about half a second are not counted).",
        render.table(["#", "visible", "of frames", "size"], rows),
        f"Count by minimum lifespan: {v['by_min_lifespan']}.",
        _label(f"{v['count']} instance(s)" + ("" if v["stable"] else " (unstable: depends on the minimum lifespan)")),
        render.reliability(r.confidence, _common_notes(r) + [
            "the segmenter can miss small, distant or occluded instances, and can split one instance in two"]),
    ]
    return ToolResult(render.block(title, lines), f"count {entity.phrase}: {v['count']}", data=r.to_dict())


def tool_event_onset(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    evid = str(args["event"])
    ev = next((e for e in ws.program.events if e.id == evid), None)
    if ev is None:
        raise ToolError(f"unknown event {evid!r}; use one of {[e.id for e in ws.program.events]}")
    title = f"event_onset {evid} ({ev.text})"
    if ev.actor != "cam":
        missing = _not_found(ws, ev.actor, title)
        if missing:
            return missing
    if ev.predicate in ("other", "stops_moving"):
        return _unavailable(title, f"'{ev.predicate}' has no geometric signal; judge its timing from the frames")
    scene = ws.scene()
    m = Measurement(_measure_id(ws), "onset", [evid], "onset")
    r = measure.measure_event_onset(scene, ws.program, m, ws.tracks_map(ev.actor))
    if not r.available:
        return _unavailable(title, r.reason, r)
    v = r.value
    t = scene.t_s
    when = "was not found in the clip" if v["onset"] is None else f"starts at {render.time_label(t, v['onset'])}"
    lines = [f"Event {evid} ({ev.text}): {when}.", render.reliability(r.confidence, r.notes + [
        "an onset is where the measured signal first clearly changes; compare the two events' times yourself"])]
    return ToolResult(render.block(title, lines), f"event_onset {evid}: {when}", data=r.to_dict())


def tool_orientation(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    eid = str(args["entity"])
    title = f"orientation of {eid}"
    ws.entity(eid)
    missing = _not_found(ws, eid, title)
    if missing:
        return missing
    track = ws.track(eid)
    scene = ws.scene()
    lo, hi, note = ws.span(ws.window(args))
    present = np.flatnonzero(track.present)
    present = present[(present >= lo) & (present <= hi)]
    if not len(present):
        return _unavailable(title, "the entity is not visible in the window")
    positions = list(present[np.unique(np.linspace(0, len(present) - 1, min(8, len(present))).astype(int))])
    rows_f = ws.facing(eid, positions)
    t = scene.t_s
    rows, bearings = [], []
    first_heading = None
    for row in rows_f:
        p = row["pos"]
        if not row.get("usable"):
            rows.append([render.time_label(t, p, gf_index=scene.gf_index), row.get("status", "no unique front") if row.get("status") != "ok"
                         else "no unique front", "-", "-", "-"])
            continue
        heading = orientation.heading_deg(row["forward_level"])
        first_heading = heading if first_heading is None else first_heading
        turn = ((heading - first_heading + 180) % 360) - 180
        basis = orientation.subject_basis(row)
        side = "-"
        if basis is not None and np.isfinite(track.smooth[p]).all():
            bearing = orientation.camera_bearing(scene, track.smooth[p], p, basis)
            quadrant, octant, margin = orientation.sector(bearing)
            bearings.append((render.time_label(t, p), quadrant))
            side = f"{quadrant} ({octant}, {bearing:+.0f} deg{', near a boundary' if margin < 10 else ''})"
        rows.append([render.time_label(t, p, gf_index=scene.gf_index), orientation.facing_words(row["forward_cam"]), f"{turn:+.0f}", side,
                     f"{row['azimuth_deg']:.0f}"])
    if not bearings and first_heading is None:
        return _unavailable(title, "Orient-Anything never resolved a unique front for this entity")
    lines = [
        f"Which way {_who(ws, eid)} faces (Orient-Anything-V2, frames where it resolves a unique front): as the camera "
        "sees it; its heading change since the first such frame (+ = turned to its right, about gravity); and the "
        "camera's position seen from the entity (front = the camera is in front of it, i.e. sees its front side).",
        render.table(["time", "faces", "turned deg", "camera on its", "oav2 az"], rows),
    ]
    if bearings:
        lines.append("Camera side over time: " + " -> ".join(f"{a} {q}" for a, q in bearings))
    lines.append(render.reliability("medium", ([f"window: {note}"] if note != "whole clip" else []) + [
        "facing is estimated per frame from appearance; check a surprising reading against the frames"]))
    return ToolResult(render.block(title, lines),
                      f"orientation {eid}: camera side " + (" -> ".join(q for _, q in bearings) or "unknown"),
                      data={"rows": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                                     for r in rows_f]})


def tool_show(ws: Workspace, args: Mapping[str, Any]) -> ToolResult:
    from d4rt_agent.agent_v5 import viz

    target = str(args.get("target") or "entities")
    if target not in ("entities", "camera") and target.split("[")[0] not in [e.id for e in ws.program.entities]:
        raise ToolError(f"`target` must be 'entities', 'camera' or an entity id ({ws.entity_choices(False)})")
    image, caption = viz.render(ws, target)
    return ToolResult(render.block(f"show {target}", [caption]), f"show {target} (image shown)", images=[image],
                      data={"target": target})


TOOLS: dict[str, Callable[[Workspace, Mapping[str, Any]], ToolResult]] = {
    "ground": tool_ground,
    "select": tool_select,
    "camera_motion": tool_camera_motion,
    "camera_profile": tool_camera_profile,
    "camera_zoom": tool_camera_zoom,
    "scene_revisit": tool_scene_revisit,
    "entity_track": tool_entity_track,
    "distance": tool_distance,
    "region_relation": tool_region_relation,
    "elevation": tool_elevation,
    "visibility": tool_visibility,
    "apparent_size": tool_apparent_size,
    "motion_profile": tool_motion_profile,
    "speed": tool_speed,
    "count": tool_count,
    "event_onset": tool_event_onset,
    "orientation": tool_orientation,
    "show": tool_show,
}

_WINDOW = "optional: [start_s, end_s], two numbers in seconds; omit it to use the question's own time span"
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "ground": {"description": "Find an entity again with a different phrase, or prompting the segmenter at a time "
                              "where it is clearly visible. Short phrases (the plain noun) find more instances.",
               "arguments": {"entity": "entity id", "phrase": "at most 5 words", "at_s": "optional seconds"},
               "required": ["entity"]},
    "select": {"description": "Choose which of the found instances is the intended entity (see `ground` / `show`).",
               "arguments": {"entity": "entity id", "instances": "list of instance numbers"},
               "required": ["entity", "instances"]},
    "camera_motion": {"description": "The camera's rotation (pan/tilt/roll) and translation over time.",
                      "arguments": {"window": _WINDOW}, "required": []},
    "camera_profile": {"description": "The camera's speed over time and its shake.",
                       "arguments": {"window": _WINDOW}, "required": []},
    "camera_zoom": {"description": "The focal length over time, next to the camera's forward movement.",
                    "arguments": {"window": _WINDOW}, "required": []},
    "scene_revisit": {"description": "How much of the first view is seen again at the end.",
                      "arguments": {"window": _WINDOW}, "required": []},
    "entity_track": {"description": "An entity's 3D movement over time, in time order.",
                     "arguments": {"entity": "entity id",
                                   "frame": "'scene' (default; first camera's axes, camera motion removed), 'camera' "
                                            "(as the moving camera sees it), 'image' (in the picture) or 'own' "
                                            "(relative to the entity's facing)",
                                   "window": _WINDOW},
                     "required": ["entity"]},
    "distance": {"description": "The distance between two entities, or an entity and the camera, over time "
                                "(or at one moment with at_s, in metres when available).",
                 "arguments": {"a": "entity id", "b": "entity id or 'cam' (default)", "window": _WINDOW,
                               "at_s": "optional: one moment, in seconds"},
                 "required": ["a"]},
    "region_relation": {"description": "An entity's distance to the nearest surface of a landmark region over time.",
                        "arguments": {"entity": "entity id", "landmark": "landmark id (default l1)", "window": _WINDOW},
                        "required": ["entity"]},
    "elevation": {"description": "The camera's height angle above or below an entity.",
                  "arguments": {"entity": "entity id", "window": _WINDOW}, "required": ["entity"]},
    "visibility": {"description": "When and where an entity is in the picture, cut off or not.",
                   "arguments": {"entity": "entity id", "window": _WINDOW}, "required": ["entity"]},
    "apparent_size": {"description": "How much of the picture an entity covers over time.",
                      "arguments": {"entity": "entity id", "window": _WINDOW}, "required": ["entity"]},
    "motion_profile": {"description": "An entity's speed over time: still, continuous, intermittent.",
                       "arguments": {"entity": "entity id", "window": _WINDOW}, "required": ["entity"]},
    "speed": {"description": "An entity's average speed, in m/s when a metric scale exists.",
              "arguments": {"entity": "entity id", "window": _WINDOW}, "required": ["entity"]},
    "count": {"description": "How many distinct instances of the counted category appear.",
              "arguments": {"entity": "category id (default c1)"}, "required": []},
    "event_onset": {"description": "When one of the named events starts.",
                    "arguments": {"event": "ev1 or ev2"}, "required": ["event"]},
    "orientation": {"description": "Which way an entity faces over time, and which of its sides the camera sees.",
                    "arguments": {"entity": "entity id", "window": _WINDOW}, "required": ["entity"]},
    "show": {"description": "An image of keyframes with the grounded entities outlined, or an entity's 3D track "
                            "drawn on them; use it to check that the right thing was measured.",
             "arguments": {"target": "'entities', 'camera' or an entity id"}, "required": []},
    "switch_kind": {"description": "Change the question kind once, if the measurements show you chose the wrong one.",
                    "arguments": {"kind": "a kind from the menu you were shown"}, "required": ["kind"]},
    "done_measuring": {"description": "Stop measuring and move on to answering.", "arguments": {}, "required": []},
}
