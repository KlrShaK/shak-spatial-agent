"""Camera measurements over a whole clip or a host-derived question interval."""

from dataclasses import asdict, replace

import numpy as np

from d4rt_agent.agent_v5.measure import (
    _unwrapped_deg, camera_state, measure_revisit, measure_zoom, windowed_speed,
)
from d4rt_agent.agent_v5.program import Measurement
from d4rt_agent.sam_orientany_d4rt_test.traj3d_buckets import (
    angle_noise_deg, displacement_noise,
)
from ..reports import Report, load_contract, segment_motion, time_label
from ..workspace import WorkspaceError

TRANSLATION_FLOOR_SCALES = 0.01
ZOOM_LOG_FLOOR = float(np.log1p(0.10))
_ROTATIONS = (("pan", "pans right", "pans left"),
              ("tilt", "tilts up", "tilts down"),
              ("roll", "rolls clockwise", "rolls counter-clockwise"))


def _scene(workspace, window):
    scene = workspace.scene()
    if window is None:
        return scene
    if len(window) != 2:
        raise WorkspaceError("A host interval needs its start and end in seconds.")
    start, end = (workspace.frame_index(t) for t in window)
    if end < start:
        raise WorkspaceError("The interval ends before it starts.")
    lo, hi = (int(np.argmin(abs(scene.gf_index - frame))) for frame in (start, end))
    use = slice(lo, hi + 1)
    fields = {key: getattr(scene, key)[use] for key in (
        "gf_index", "cam_R", "cam_t", "K", "points", "grid_traj", "grid_conf")}
    fields["pose_valid"] = None if scene.pose_valid is None else scene.pose_valid[use]
    return replace(scene, **fields)


def _time(workspace, scene):
    times = workspace.times(scene.gf_index)
    labels = workspace.time_labels(scene.gf_index)
    clock = np.arange(scene.length, dtype=float) if times is None else times
    return times, clock, labels


def _span(a, b, clock, labels, known_time):
    if known_time:
        return time_label(a)[:-2] + "–" + time_label(b)
    return labels[int(np.argmin(abs(clock - a)))] + "–" + labels[int(np.argmin(abs(clock - b)))]


def _scalar_phases(times, values, noise, labels, positive, negative):
    points = np.column_stack((values, np.zeros((len(values), 2))))
    return segment_motion(times, points, noise, time_labels=labels,
                          axes=((positive, negative), ("unused", "unused"), ("unused", "unused")))


def _phase_at(phases, at):
    return next(phase for phase in phases if phase.start <= at <= phase.end)


def _delta(values, clock, a, b):
    return np.array([np.interp(b, clock, values[:, i]) - np.interp(a, clock, values[:, i])
                     for i in range(values.shape[1])])


def _pack(rows, max_rows=6):
    """Keep every interval and number when a profile needs more than six rows."""
    if len(rows) <= max_rows:
        return rows
    packed = []
    for indices in np.array_split(np.arange(len(rows)), max_rows):
        group = [rows[int(i)] for i in indices]
        packed.append(tuple("; ".join(row[0] if column == 0 else f"{row[column]} at {row[0]}"
                                     for row in group) for column in range(len(group[0]))))
    return packed


def _unit_note(workspace):
    metric = workspace.metric
    if metric.available:
        return f"Distances are metres; metric scale source: {metric.source}."
    return "Distances are scene-relative units, comparable within this clip only."


def _camera_phases(times, clock, labels, positions, position_noise, angles, angle_noise):
    translation = segment_motion(times, positions, position_noise, time_labels=labels)
    rotations = {name: _scalar_phases(times, angles[:, i], angle_noise[i], labels, positive, negative)
                 for i, (name, positive, negative) in enumerate(_ROTATIONS)
                 if np.isfinite(angles[:, i]).all()}
    channels = [translation, *rotations.values()]
    edges = sorted({p.start for phases in channels for p in phases} |
                   {p.end for phases in channels for p in phases})
    phases = []
    for a, b in zip(edges[:-1], edges[1:]):
        middle = (a + b) / 2
        movement = _phase_at(translation, middle).label
        turns = {name: _phase_at(phases_, middle).label for name, phases_ in rotations.items()}
        words = (["moves " + movement] if movement != "still" else []) + [
            word for word in turns.values() if word != "still"]
        displacement = _delta(positions, clock, a, b)
        turn = _delta(angles, clock, a, b)
        key = " and ".join(words) if words else "still"
        phases.append({"start": a, "end": b, "span": _span(a, b, clock, labels, times is not None),
                       "label": key, "translation": movement, "turn_labels": turns,
                       "displacement": displacement.tolist(), "moved": float(np.linalg.norm(displacement)),
                       "rotation_deg": turn.tolist()})
    return phases, translation, rotations


def _description(phases, noise, angle_noise):
    first = phases[0]
    parts = []
    if first["translation"] != "still":
        parts.append(f"MOVES {first['translation'].upper()} "
                     f"({sum(p['moved'] for p in phases):.3f}; noise {noise:.3f})")
    for i, (name, _, _) in enumerate(_ROTATIONS):
        direction = first["turn_labels"].get(name)
        if direction and direction != "still":
            value = sum(p["rotation_deg"][i] for p in phases)
            parts.append(f"{direction.upper()} ({value:+.1f}°; noise {angle_noise[i]:.1f}°)")
    if not parts:
        return f"STILL (translation noise {noise:.3f}; rotation floors " + ", ".join(
            f"{name} {floor:.1f}°" for (name, _, _), floor in zip(_ROTATIONS, angle_noise)) + ")"
    return " and ".join(parts)


def track_camera(workspace, *, window=None):
    title = "track_camera — camera in its gravity-levelled starting view"
    try:
        scene = _scene(workspace, window)
        if scene.length < 2:
            return Report.unavailable(title, "the interval has too few camera poses")
        if scene.pose_valid is not None and not scene.pose_valid.all():
            return Report.unavailable(title, "camera poses failed in this interval; held poses are not measured stillness")
        if not np.isfinite(scene.scale) or scene.scale <= 0:
            return Report.unavailable(title, "the scene scale is invalid")
        cam = camera_state(scene)
        if not np.isfinite(cam["trans"]).all():
            return Report.unavailable(title, "camera positions are incomplete")
        times, clock, labels = _time(workspace, scene)
        metric = workspace.metric
        positions = metric.convert(cam["trans"])
        noise = float(metric.convert(displacement_noise(cam["trans"], cam["trans_smooth"],
                                                       TRANSLATION_FLOOR_SCALES * scene.scale)))
        angles = np.column_stack([_unwrapped_deg(cam[name]) for name, _, _ in _ROTATIONS])
        angle_noise = [angle_noise_deg(angles[:, i]) for i in range(3)]
        phases, translation, rotations = _camera_phases(times, clock, labels, positions, noise, angles, angle_noise)
        durations = {}
        for phase in phases:
            durations[phase["label"]] = durations.get(phase["label"], 0) + phase["end"] - phase["start"]
        majority = max(durations, key=durations.get)
        main = [p for p in phases if p["label"] == majority]
        share = durations[majority] / (clock[-1] - clock[0])
        timing = "throughout" if len(durations) == 1 else "for most of the interval" if share > .5 else "for the largest time share"
        headline = "Camera " + _description(main, noise, angle_noise) + " " + timing + "."
        minority = [_description([p], noise, angle_noise) + " at " + p["span"]
                    for p in phases if p["label"] != majority]
        if minority:
            headline += " " + "; ".join(minority) + "."
        for phase in phases:
            x, pan = phase["displacement"][0], phase["rotation_deg"][0]
            if abs(x) > noise and abs(pan) > angle_noise[0] and x * pan < 0:
                sense = "CLOCKWISE" if x < 0 else "COUNTER-CLOCKWISE"
                headline += f" {sense} orbit signature at {phase['span']}: opposing lateral translation and pan."
        path = sum(p.distance for p in translation if p.label != "still")
        net = float(np.linalg.norm(positions[-1] - positions[0]))
        tempo = "Tempo unavailable: frame rate is unknown."
        if times is not None:
            duration = float(times[-1] - times[0])
            turn_path = {name: sum(p.distance for p in phases_ if p.label != "still")
                         for name, phases_ in rotations.items()}
            tempo = (f"Tempo: path speed {path / duration:.3f} {metric.units}/s; "
                     f"net displacement rate {net / duration:.3f} {metric.units}/s; " +
                     ", ".join(f"{name} path rate {value / duration:.1f}°/s" for name, value in turn_path.items()) + ".")
            speeds, _ = windowed_speed(metric.convert(cam["trans_smooth"]), np.arange(scene.length),
                                       scene.fps, times=times)
            if len(speeds) >= 3:
                thirds = [float(np.mean(part)) for part in np.array_split(speeds, 3)]
                tempo += " First/middle/last path speed: " + "/".join(f"{value:.3f}" for value in thirds) + f" {metric.units}/s."
        trans_residual = metric.convert(cam["trans"] - cam["trans_smooth"])
        rot_smooth = np.column_stack([cam[name + "_smooth"] for name, _, _ in _ROTATIONS])
        residuals = angles - rot_smooth
        finite_residuals = residuals[np.isfinite(residuals)]
        jitter = {"translation": float(np.sqrt(np.mean(np.sum(trans_residual ** 2, axis=1)))),
                  "rotation_deg": float(np.sqrt(np.mean(finite_residuals ** 2))) if len(finite_residuals) else None}
        headline += " " + tempo + f" Jitter residual: translation {jitter['translation']:.3f}"
        if jitter["rotation_deg"] is not None:
            headline += f", rotation {jitter['rotation_deg']:.2f}° (pan, tilt and roll included)."
        rows = []
        for phase in phases:
            rows.append((phase["span"], phase["translation"], f"{phase['moved']:.3f}", *[
                f"{value:+.1f}°" if np.isfinite(value) else "unavailable" for value in phase["rotation_deg"]]))
        missing = [name for name, _, _ in _ROTATIONS if name not in rotations]
        if missing and majority == "still":
            headline = headline.replace("Camera STILL", "Measured translation STILL; some rotation unavailable", 1)
        if scene.up_source != "plane":
            title = "track_camera — camera in its starting view (gravity unavailable)"
        notes = load_contract("traj_camera_own_frame") + "\n" + _unit_note(workspace)
        notes += " Brief weak phases may be absorbed; strong brief excursions are retained."
        if times is not None and scene.fps is not None and scene.fps <= 8:
            notes += " Sparse frame sampling can understate shake."
        return Report(title, headline, rows=_pack(rows), columns=("time", "translation", "moved", "pan", "tilt", "roll"),
                      reading_full=notes,
                      reading_short="Left translation + right pan gives clockwise orbit; opposite signs give counter-clockwise. " + _unit_note(workspace),
                      status="partial" if missing else "ok", reason="Unmeasurable rotation: " + ", ".join(missing) if missing else "",
                      raw={"times": None if times is None else times.tolist(), "positions": positions.tolist(),
                           "phases": phases, "majority": majority, "translation_phases": [asdict(p) for p in translation],
                           "rotation_deg": angles.tolist(), "rotation_noise_deg": angle_noise,
                           "translation_noise": noise, "path_distance": path, "net_displacement": net,
                           "jitter": jitter, "units": metric.units, "metric_source": metric.source})
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return Report.unavailable(title, str(error))
