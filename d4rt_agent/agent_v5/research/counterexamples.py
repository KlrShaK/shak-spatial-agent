"""Small, exact counterexamples replayed through current V5 measurement code.

Inputs are synthetic and contain no perception noise unless explicitly stated.
These are diagnostic experiments, not a benchmark score or production fixes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from d4rt_agent.agent_v5 import measure, synthetic
from d4rt_agent.agent_v5.program import Entity, Event, Measurement, QueryProgram, Window
from d4rt_agent.agent_v5.scene import EntityTrack, Grounding, effective_fps, lift_instances, resolve_window
from d4rt_agent.agent_v5.tools import (
    Workspace, tool_camera_motion, tool_count, tool_entity_track, tool_event_onset, tool_orientation, tool_speed,
)

OUT = Path(__file__).resolve().parents[2] / "results/agent_v5_capability_research_20261013"


def track(xyz, eid="e1", present=None) -> EntityTrack:
    xyz = np.asarray(xyz, dtype=float)
    n = len(xyz)
    present = np.ones(n, bool) if present is None else np.asarray(present, bool)
    return EntityTrack(eid, eid, [0], present, present.copy(), xyz.copy(), xyz.copy(),
                       np.full(n, 20), present * 0.1, np.zeros(n, bool),
                       np.column_stack([np.linspace(0.2, 0.8, n), np.full(n, 0.5)]), 0.0)


def workspace(scene, tracks=None, kind="subject_direction", events=(), window=None):
    tracks = tracks or {}
    entities = [Entity("cam", "camera")] + [Entity(e, "object", e) for e in tracks]
    program = QueryProgram("diagnostic", kind, "measure", "", entities=entities, events=list(events),
                           window=window or Window())
    clip = SimpleNamespace(length=scene.length, fps=scene.fps, slug="synthetic")
    ws = Workspace(clip, program, scene_fn=lambda c: scene, ground_fn=lambda c, p, f: None)
    ws._tracks.update(tracks)  # Exact tracks intentionally bypass detector/tracker errors.
    return ws


def result(case, invariant, observed, reference, reproduced, production_function):
    return dict(case=case, invariant=invariant, observed=observed, reference=reference,
                reproduced=bool(reproduced), production_function=production_function)


def experiments() -> tuple[list[dict], dict]:
    out = []
    scene, _ = synthetic.build_scene(frames=61, fps=10, metres_per_unit=1.0)
    scene.metric_source = "exact-synthetic"
    t = scene.t_s

    # 1. Reflection preserves range and changes signed relative position.
    ref = track(np.tile([0, 0, 4], (61, 1)), "e2")
    values = []
    for sign in (-1, 1):
        a = track(np.tile([sign, 0, 4], (61, 1)))
        p = QueryProgram("range", "pair_relation", "measure", "")
        r = measure.measure_range(scene, p, Measurement("m", "range", ["e1", "e2"], "trend"),
                                  {"e1": [a], "e2": [ref]})
        values.append(r.value)
    out.append(result("range_cannot_identify_side", "Left/right reflection must reverse signed side.", values,
                      {"signed_right_offsets": [-1, 1]}, values[0] == values[1], "measure_range"))

    # 2. A round trip has positive mean speed and zero net displacement.
    x = 2 * np.sin(np.pi * t / t[-1])
    a = track(np.column_stack([x, np.zeros(61), np.full(61, 4)]))
    ws = workspace(scene, {"e1": [a]}, kind="metric_speed")
    speed = tool_speed(ws, {"entity": "e1"})
    exact_speed = float(np.linalg.norm(np.diff(a.xyz, axis=0), axis=1).sum() / t[-1])
    out.append(result("net_displacement_is_not_speed", "Mean speed is distance travelled / duration.",
                      {"value": speed.data["value"], "report": speed.text}, {"mean_speed": exact_speed},
                      speed.data["value"]["mps"] < exact_speed * 0.05, "tool_speed / measure_metric_speed"))

    # 3. The same camera round trip must not be described as fixed.
    moving_scene, _ = synthetic.build_scene(frames=61, fps=10, camera_t=lambda i: (x[i], 0, 0))
    camera = tool_camera_motion(workspace(moving_scene, kind="camera_path"), {})
    out.append(result("camera_return_is_not_static", "Returning to the starting pose does not erase the path.",
                      {"value": camera.data["value"], "report": camera.text}, {"excursion": float(x.max()),
                      "path_length": float(np.abs(np.diff(x)).sum())},
                      camera.data["value"]["motion"] == "static", "tool_camera_motion / measure_camera_motion"))

    # 4. Image evidence exists even if depth cannot be obtained.
    image_track = track(np.full((61, 3), np.nan))
    ws = workspace(scene, {"e1": [image_track]})
    image_result = tool_entity_track(ws, {"entity": "e1", "frame": "image"})
    out.append(result("image_track_requires_no_depth", "A valid 2D track supports image displacement without 3D.",
                      image_result.data, {"image_dx": 0.6}, image_result.data.get("available") is False,
                      "tool_entity_track / _first_present"))

    # 5/6. The event tool returns first-ever appearance despite a later window,
    # and last disappearance instead of the first leave event.
    pres = np.zeros(61, bool)
    pres[5:15] = True
    pres[35:45] = True
    a = track(np.tile([0, 0, 4], (61, 1)), present=pres)
    events = [Event("ev1", "e1", "enters_frame", "entity enters"),
              Event("ev2", "e1", "leaves_frame", "entity leaves")]
    ws = workspace(scene, {"e1": [a]}, kind="event_order", events=events, window=Window("interval", 3, 5))
    entry = tool_event_onset(ws, {"event": "ev1", "window": [3, 5]})
    out.append(result("event_window_ignored", "An entry requested during [3,5] seconds must be in that interval.",
                      entry.data, {"entry_s": 3.5}, entry.data["value"]["onset_s"] == 0.5,
                      "tool_event_onset / measure_event_onset"))
    ws.program.window = Window()
    first_leave = tool_event_onset(ws, {"event": "ev2"})
    out.append(result("last_leave_is_not_first_leave", "Ordering first events needs first transition, not last presence.",
                      first_leave.data, {"all_leave_frames": [15, 45], "first_leave_s": 1.5},
                      first_leave.data["value"]["onset"] == 45, "measure_event_onset"))

    # 7. Stops-moving is in the schema but explicitly unavailable in the tool.
    ws.program.events.append(Event("ev3", "e1", "stops_moving", "entity stops"))
    stop = tool_event_onset(ws, {"event": "ev3"})
    out.append(result("declared_stop_predicate_has_no_measurement", "A declared geometric predicate needs a usable signal.",
                      stop.data, {"supported_schema_predicate": "stops_moving"},
                      stop.data.get("available") is False, "tool_event_onset"))

    # 8. Count in a phase is distinct from count anywhere in the clip.
    tracks = [track(np.tile([i, 0, 4], (61, 1)), f"c1[{i}]", t < 2 if i == 0 else t > 3) for i in range(3)]
    ws = workspace(scene, {"c1": tracks}, kind="instance_count", window=Window("interval", 0, 1.5))
    ws.program.entities[-1] = Entity("c1", "category", "person")
    ws.groundings["c1"] = Grounding("person", np.arange(3), np.zeros((3, 61, 1, 1), bool),
                                   np.ones((3, 61)), np.zeros((3, 61, 4)), np.ones((3, 61)), np.zeros((3, 61), bool))
    count = tool_count(ws, {"entity": "c1"})
    out.append(result("count_window_ignored", "Count in [0,1.5] seconds excludes entities visible only after 3 seconds.",
                      count.data, {"interval_count": 1, "whole_clip_count": 3}, count.data["value"]["count"] == 3,
                      "tool_count / measure_instances"))

    # 9. A complete turn is not zero cumulative angular motion. The raw rows
    # remain informative; the defect is the rendered 'turned' column.
    a = track(np.tile([0, 0, 4], (61, 1)))
    ws = workspace(scene, {"e1": [a]})
    def facing(eid, positions):
        rows = []
        for p in positions:
            angle = 2 * np.pi * p / 60
            f = np.array([np.sin(angle), 0, np.cos(angle)])
            rows.append(dict(pos=int(p), usable=True, status="ok", forward_cam=f, forward_world=f,
                             forward_level=f, up_world=np.array([0, -1, 0]), azimuth_deg=float(np.degrees(angle))))
        return rows
    ws.facing = facing
    orient = tool_orientation(ws, {"entity": "e1"})
    table_rows = [s.strip() for s in orient.text.splitlines() if re.match(r"\s*\d+\.\d+s\s", s)]
    last_turn = float(re.split(r"\s{2,}", table_rows[-1])[2])
    out.append(result("orientation_wrap_loses_turn_count", "A 360 degree turn requires unwrapped cumulative rotation.",
                      {"final_rendered_turn": last_turn, "report": orient.text}, {"cumulative_turn_deg": 360},
                      abs(last_turn) < 1, "tool_orientation"))

    # 10. Nonuniform sampling: a constant physical velocity stays constant when
    # represented with actual timestamps, but not with mean FPS and row indices.
    irregular, _ = synthetic.build_scene(frames=21, fps=10)
    irregular.gf_index = np.r_[np.arange(11), np.arange(20, 111, 10)]
    xyz = np.column_stack([irregular.t_s, np.zeros(21), np.full(21, 4)])
    speed_values, _ = measure.windowed_speed(xyz, np.arange(21), effective_fps(irregular))
    dt_speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(irregular.t_s)
    out.append(result("irregular_timestamps_need_actual_dt", "Constant velocity is invariant to temporal resampling.",
                      {"speed_min": float(speed_values.min()), "speed_max": float(speed_values.max())},
                      {"timestamp_speed_min": float(dt_speed.min()), "timestamp_speed_max": float(dt_speed.max())},
                      np.ptp(speed_values) > 1, "windowed_speed as called by measure_motion_profile"))

    # 11. Percentage windows should refer to elapsed duration, not row count.
    lo, hi, note = resolve_window(irregular, Window("phase", phase="first_half"))
    out.append(result("phase_uses_row_fraction", "Half the clip means half elapsed time under irregular sampling.",
                      {"lo": lo, "hi": hi, "end_s": float(irregular.t_s[hi]), "note": note},
                      {"target_end_s": float(irregular.t_s[-1] / 2)},
                      abs(irregular.t_s[hi] - irregular.t_s[-1] / 2) > 1, "resolve_window"))

    # 12. A stationary planar surface partly occluded from changing sides has
    # shifting visible-surface medians, even with exact geometry and masks.
    wall, _ = synthetic.build_scene(frames=41, fps=10)
    masks = np.zeros((1, 41, 96, 160), bool)
    for i in range(41):
        x0 = int(round(16 + 64 * i / 40))
        masks[0, i, 16:80, x0:x0 + 48] = True
    g = Grounding("stationary wall", np.array([0]), masks, np.ones((1, 41)),
                  np.zeros((1, 41, 4)), masks.mean(axis=(2, 3)), np.zeros((1, 41), bool))
    lifted = lift_instances(wall, g, [0], entity_id="e1")
    drift = float(np.linalg.norm(lifted.smooth[-1] - lifted.smooth[0]))
    out.append(result("visible_surface_median_is_not_material_track", "Occlusion of a fixed surface must not imply object translation.",
                      {"inferred_displacement": drift, "residual_noise": lifted.noise,
                       "valid_frames": int(np.isfinite(lifted.smooth).all(1).sum())},
                      {"material_displacement": 0.0, "point_cloud_constant": bool(np.all(wall.points == wall.points[0]))},
                      drift > 0.5, "lift_instances"))
    series = {"time": t.tolist(), "return_path_x": x.tolist(), "irregular_time": irregular.t_s.tolist(),
              "irregular_measured_speed": speed_values.tolist(), "surface_x": lifted.smooth[:, 0].tolist()}
    return out, series


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases, series = experiments()
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__), *[Path(__file__).parents[1] / n for n in
                                ("tools.py", "measure.py", "scene.py", "thresholds.json")]]}
    payload = {"source_hashes": hashes, "thresholds_unchanged": dict(measure.TUNING), "cases": cases, "series": series}
    (args.out / "counterexamples.json").write_text(json.dumps(payload, indent=2, default=lambda x: x.tolist()) + "\n")
    lines = ["# Synthetic contract experiments", "", "Exact inputs; no new model inference, no threshold fitting.", "",
             "A reproduced failure demonstrates a possible defect, not its prevalence on benchmark clips.", "",
             "| Case | Failure reproduced | Function |", "|---|---|---|"]
    for r in cases:
        lines.append(f"| {r['case']} | {r['reproduced']} | {r['production_function']} |")
    (args.out / "counterexamples.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cases": len(cases), "reproduced": sum(r["reproduced"] for r in cases), "out": str(args.out)}))


if __name__ == "__main__":
    main()
