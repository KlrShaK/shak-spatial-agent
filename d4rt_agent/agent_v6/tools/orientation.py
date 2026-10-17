"""Facing observations, the coarse visual fallback, and orientation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from d4rt_agent.agent_v5 import orientation as geometry
from d4rt_agent.sam_orientany_d4rt_test.traj3d_geometry import offaxis_correction
from d4rt_agent.agent_v6.reports import Report, load_contract, phase_rows, segment_motion
from d4rt_agent.agent_v6.workspace import WorkspaceError

# Crop-view axes; the crop ray is corrected before lifting into world coordinates.
FACING_BINS = {
    "facing away": 0.0,
    "facing away and image-right": 45.0,
    "facing image-right": 90.0,
    "facing the camera and image-right": 135.0,
    "facing the camera": 180.0,
    "facing the camera and image-left": -135.0,
    "facing image-left": -90.0,
    "facing away and image-left": -45.0,
}
FALLBACK_SOURCE = "facing from the VLM fallback, coarse to ±22.5°"


def window_positions(ws, window=None) -> np.ndarray:
    scene = ws.scene()
    if window is None:
        return np.arange(scene.length)
    if len(window) != 2:
        raise WorkspaceError("A time window needs its start and end in seconds.")
    start, end = (ws.frame_index(t) for t in window)
    if start > end:
        raise WorkspaceError("The time window ends before it starts.")
    first, last = [int(np.argmin(abs(scene.gf_index - index))) for index in (start, end)]
    return np.arange(first, last + 1)


def labels_for(ws, positions: Sequence[int]) -> list[str]:
    return ws.time_labels(ws.scene().gf_index[np.asarray(positions, dtype=int)])


def basis_for(scene, row) -> np.ndarray | None:
    """Subject right/down/forward: measured heading with gravity defining vertical."""
    if not row.get("usable"):
        return None
    forward = np.asarray(row["forward_world"], dtype=float)
    forward = forward - np.dot(forward, scene.up) * scene.up
    if not np.isfinite(forward).all() or np.linalg.norm(forward) < 1e-8:
        return None
    return geometry.subject_basis({**row, "forward_world": forward, "up_world": scene.up})


def _crop(ws, entity, position):
    from PIL import Image

    frame = Image.fromarray(ws.clip.frame(int(ws.scene().gf_index[position])))
    mask = ws.masks_at(entity)(position)
    if mask is None:
        return None
    full = np.asarray(Image.fromarray(mask).resize(frame.size, resample=Image.Resampling.NEAREST))
    ys, xs = np.nonzero(full)
    if not len(xs):
        return None
    margin = max(1, round(0.1 * max(xs.max() - xs.min(), ys.max() - ys.min())))
    box = (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
           min(frame.width, int(xs.max()) + margin + 1), min(frame.height, int(ys.max()) + margin + 1))
    return frame.crop(box), (float(xs.mean()), float(ys.mean()))


def facing_rows(ws, name: str, *, qwen=None, positions=None) -> list[dict[str, Any]]:
    """OAV2 first; if no front resolves, query three crops with greedy local Qwen.

    Sparse fallback observations remain sparse. They are not interpolated into
    an apparently measured continuous heading trajectory.
    """
    entity = ws.resolve(name)
    scene = ws.scene()
    track = ws.track_record(entity.label)
    positions = np.arange(scene.length) if positions is None else np.asarray(positions, dtype=int)
    positions = positions[track.present[positions]]
    if not len(positions):
        return []
    try:
        rows = ws.facing(entity.label, positions)
    except Exception as error:  # Perception failure must not suppress available translation.
        rows = [{"pos": int(p), "usable": False, "status": str(error)} for p in positions]
    if any(basis_for(scene, row) is not None for row in rows):
        return rows
    if qwen is None and ws.models is not None:
        try:
            qwen = ws.models.qwen()
        except Exception:  # Failed fallback loading also leaves translation available.
            return rows
    if qwen is None:
        return rows
    targets = np.linspace(positions[0], positions[-1], 3)
    sampled = tuple(dict.fromkeys(int(positions[np.argmin(abs(positions - target))]) for target in targets))

    def fallback():
        prompt = (Path(__file__).parents[1] / "prompts" / "facing_fallback.md").read_text()
        result = []
        for pos in sampled:
            row = {"pos": pos, "usable": False, "source": FALLBACK_SOURCE}
            cropped = _crop(ws, entity.label, pos)
            if cropped is None:
                result.append({**row, "status": "no visible crop"})
                continue
            image, centre = cropped
            # OfflineQwen.generate is deterministic (do_sample=False, num_beams=1).
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt}, {"type": "image", "image": image},
            ]}]
            try:
                answer = qwen.generate(messages, max_new_tokens=32).strip().casefold().strip(" .\"'")
            except Exception as error:
                result.append({**row, "status": str(error)})
                continue
            if answer not in FACING_BINS:
                result.append({**row, "status": "no resolvable front"})
                continue
            angle = np.deg2rad(FACING_BINS[answer])
            forward_crop = np.array([np.sin(angle), 0.0, np.cos(angle)])
            forward_cam = offaxis_correction(forward_crop, centre, geometry.source_intrinsics(scene, pos))
            world = scene.cam_R[pos] @ forward_cam
            result.append({**row, "usable": True, "status": "ok", "label": answer,
                           "forward_crop": forward_crop, "crop_centre": centre,
                           "forward_cam": forward_cam, "forward_world": world,
                           "up_world": scene.up, "forward_level": scene.to_level(world)})
        return result
    return ws.cached(("facing_fallback", entity.label, sampled), (entity.category,), fallback)


def pack_rows(rows, max_rows=6):
    """Pack temporal rows while retaining every constituent label and reading."""
    if len(rows) <= max_rows:
        return rows
    output = []
    for group in np.array_split(np.arange(len(rows)), max_rows):
        selected = [rows[int(i)] for i in group]
        span = f"{selected[0][0]} … {selected[-1][0]}"
        quantities = " / ".join(dict.fromkeys(str(row[1]) for row in selected))
        reading = "; ".join(f"{row[1]}: {row[2]} at {row[0]}" for row in selected)
        output.append((span, quantities, reading))
    return output


def turn_series(scene, rows) -> dict[str, Any]:
    usable = sorted((row for row in rows if basis_for(scene, row) is not None), key=lambda row: row["pos"])
    if not usable:
        return {"positions": [], "turn_deg": [], "cumulative_turn_deg": None, "angular_path_deg": None, "segments": []}
    positions = np.array([int(row["pos"]) for row in usable])
    headings = np.array([geometry.heading_deg(scene.to_level(row["forward_world"])) for row in usable])
    groups = np.split(np.arange(len(usable)), np.flatnonzero(np.diff(positions) != 1) + 1)
    segments = []
    for group in groups:
        turns = np.degrees(np.unwrap(np.radians(headings[group])))
        turns -= turns[0]
        segments.append({"positions": positions[group].tolist(), "turn_deg": turns.tolist(),
                         "heading_deg": headings[group].tolist(),
                         "cumulative_turn_deg": float(turns[-1]) if len(group) > 1 else None,
                         "angular_path_deg": float(abs(np.diff(turns)).sum()) if len(group) > 1 else None})
    continuous = len(segments) == 1
    return {"positions": positions.tolist(), "turn_deg": [v for segment in segments for v in segment["turn_deg"]],
            "cumulative_turn_deg": segments[0]["cumulative_turn_deg"] if continuous else None,
            "angular_path_deg": segments[0]["angular_path_deg"] if continuous else None,
            "segments": segments, "partial": not continuous or len(usable) != len(rows),
            "source": FALLBACK_SOURCE if any(row.get("source") == FALLBACK_SOURCE for row in usable)
            else "Orient-Anything-V2"}


def heading_report(ws, name, rows) -> Report:
    series = turn_series(ws.scene(), rows)
    if not series["positions"]:
        return Report.unavailable(f"{name} heading", "No resolvable front.", raw=series)
    if len(series["positions"]) < 2:
        return Report.unavailable(f"{name} heading", "One facing observation cannot measure heading change.", raw=series)
    source = series["source"]
    turn, path = series["cumulative_turn_deg"], series["angular_path_deg"]
    if turn is None:
        headline = f"{name}: turning across missing observations is unavailable."
    elif path <= 5:
        headline = f"{name}: heading {turn:+.0f}° over its observed span; no turning above the 5° floor."
    elif abs(turn) <= 5:
        headline = f"{name}: heading {turn:+.0f}° net; total turning {path:.0f}°."
    else:
        headline = f"{name}: heading {turn:+.0f}°, turning to its {'right' if turn > 0 else 'left'}; total turning {path:.0f}°."
    output = []
    for segment in series["segments"]:
        positions = np.asarray(segment["positions"])
        turns = np.asarray(segment["turn_deg"])
        labels = labels_for(ws, positions)
        if len(positions) == 1:
            output.append((labels[0], "isolated facing observation; intervening turn unknown",
                           f"heading {segment['heading_deg'][0]:+.0f}°"))
            continue
        angular_path = np.column_stack([turns, np.zeros((len(turns), 2))])
        phases = segment_motion(ws.times(ws.scene().gf_index[positions]), angular_path, 5.0,
                                time_labels=labels, axes=(("turns right", "turns left"), ("", ""), ("", "")))
        output.extend(phase_rows(phases))
    note = (f"Camera rotation is removed. Positive heading is a turn to the subject's right. Source: {source}. "
            "Turning is accumulated only within consecutive observed runs; gaps carry no measured turn.")
    if source == FALLBACK_SOURCE:
        note += " Three coarse observations cannot resolve turns between observations."
    short = f"Positive heading means turning to the subject's right; missing intervals have no measured turn. Source: {source}."
    return Report(f"{name} heading", headline, pack_rows(output), ("time", "turning", "degrees"), note, short,
                  status="partial" if series["partial"] else "ok",
                  reason="Facing covers separate observations or only part of the requested span." if series["partial"] else "",
                  raw=series)


def sector_phases(ws, positions, labels, values) -> list[tuple[str, str, str]]:
    """Merge adjacent side observations; pack excess runs without dropping them."""
    stamps = labels_for(ws, positions)
    runs = []
    for i, label in enumerate(labels):
        if runs and label == runs[-1][2] and positions[i] == positions[i - 1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i, label])
    packed = []
    for group in np.array_split(np.arange(len(runs)), min(6, len(runs))) if runs else []:
        readings = []
        first, last = runs[int(group[0])][0], runs[int(group[-1])][1]
        for index in group:
            a, b, label = runs[int(index)]
            span = stamps[a] if a == b else f"{stamps[a]}–{stamps[b]}"
            readings.append(f"{label} ({values[a]:+.0f}° → {values[b]:+.0f}°) at {span}")
        packed.append((f"{stamps[first]}–{stamps[last]}", "; ".join(readings), ""))
    return packed


def track_orientation(ws, main, frame_of_reference="camera", *, qwen=None, window=None) -> Report:
    title = f"Facing of {main}, relative to {frame_of_reference}"
    try:
        entity = ws.resolve(main)
        positions = window_positions(ws, window)
        rows = facing_rows(ws, entity.label, qwen=qwen, positions=positions)
        heading = heading_report(ws, entity.label, rows)
        if frame_of_reference == "own":
            heading.raw["window"] = window
            return heading
        scene = ws.scene()
        reference_rows = {}
        if frame_of_reference != "camera":
            reference = ws.resolve(frame_of_reference)
            reference_rows = {row["pos"]: row for row in facing_rows(ws, reference.label, qwen=qwen, positions=positions)}
        track = ws.track_record(entity.label)
        values, labels, used = [], [], []
        for row in rows:
            pos = int(row["pos"])
            basis = basis_for(scene, row)
            if basis is None:
                continue
            if frame_of_reference == "camera":
                if not np.isfinite(track.smooth[pos]).all() or (scene.pose_valid is not None and not scene.pose_valid[pos]):
                    continue
                angle = geometry.camera_bearing(scene, track.smooth[pos], pos, basis)
            else:
                other_basis = basis_for(scene, reference_rows.get(pos, {}))
                if other_basis is None:
                    continue
                vector = row["forward_world"] @ other_basis
                angle = float(np.degrees(np.arctan2(vector[0], vector[2])))
            quadrant, octant, _ = geometry.sector(angle)
            values.append(angle)
            labels.append(octant)
            used.append(pos)
        if not used:
            return Report.unavailable(title, "No common observations with a resolvable front.", raw={"window": window})
        phases = sector_phases(ws, used, labels, values)
        route = " → ".join(label for i, label in enumerate(labels) if i == 0 or label != labels[i - 1])
        if frame_of_reference == "camera":
            headline = f"{entity.label} shows its {route} side to the camera."
        else:
            headline = f"{entity.label} faces {route} in {reference.label}'s current facing."
        partial = len(used) != len(positions) or any(b - a != 1 for a, b in zip(used, used[1:]))
        if partial:
            headline = "At observed times: " + headline + " Facing between missing observations is unavailable."
        source = heading.raw.get("source", "Orient-Anything-V2")
        note = load_contract("orient_camera_subject_frame") + f" Facing source: {source}."
        short = f"Sides follow the subject's current facing; missing observations do not establish a continuous side. Source: {source}."
        return Report(title, headline, phases, ("time", "facing", ""), note, short,
                      status="partial" if partial else "ok",
                      reason="Facing is measured only at the listed observations." if partial else "",
                      raw={"positions": used, "bearing_deg": values, "sides": labels,
                           "heading": heading.raw, "source": source, "window": window})
    except WorkspaceError as error:
        # Ambiguity must reach the dispatcher so it can present candidate colours.
        from d4rt_agent.agent_v6.workspace import CandidateRequired
        if isinstance(error, CandidateRequired):
            raise
        return Report.unavailable(title, str(error), raw={"window": window})
