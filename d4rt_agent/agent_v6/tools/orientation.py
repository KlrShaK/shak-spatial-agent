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
