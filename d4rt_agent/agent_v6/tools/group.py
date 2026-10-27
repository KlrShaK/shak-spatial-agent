"""Instance directions and group spread, retaining changes of membership."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np

from d4rt_agent.agent_v6.reports import AXES, Report, _direction, segment_motion
from d4rt_agent.agent_v6.workspace import WorkspaceError
from .entity import CAMERA_AXES
from .orientation import basis_for, facing_rows, labels_for, pack_rows, window_positions


def track_multiple_entities(ws, category, frame_of_reference="world", *, qwen=None, window=None) -> Report:
    title = f"{category} instances in the {frame_of_reference} frame"
    if frame_of_reference not in ("world", "camera", "own"):
        raise WorkspaceError("frame_of_reference must be world, camera, or own")
    try:
        entities = ws.entities(category)
        if not entities:
            return Report.unavailable(title, ws.missing_message(category), raw={"window": window})
        scene = ws.scene()
        positions = window_positions(ws, window)
        tracks = {entity.label: ws.track_record(entity.label) for entity in entities}
        counts, directions, unresolved, output = Counter(), {}, [], []
        reference_positions = {}
        for entity in entities:
            track = tracks[entity.label]
            valid = track.present & np.isfinite(track.smooth).all(1)
            if frame_of_reference == "camera" and scene.pose_valid is not None:
                valid &= scene.pose_valid
            used = positions[valid[positions]]
            if len(used) < 2:
                unresolved.append(entity.label)
                output.append(("—", entity.label, "motion unavailable: fewer than two measured positions"))
                continue
            points = track.smooth[used]
            axes = AXES
            if frame_of_reference == "world":
                path = scene.to_level(points)
            elif frame_of_reference == "camera":
                path = np.einsum("ti,tij->tj", points - scene.cam_t[used], scene.cam_R[used])
                axes = CAMERA_AXES
            else:
                rows = facing_rows(ws, entity.label, qwen=qwen, positions=positions)
                by_position = {int(row['pos']): row for row in rows}
                anchor = next((int(pos) for pos in used
                               if basis_for(scene, by_position.get(int(pos), {})) is not None), None)
                basis = basis_for(scene, by_position[anchor]) if anchor is not None else None
                if basis is None:
                    unresolved.append(entity.label)
                    output.append(("—", entity.label, "own-frame direction unavailable: no starting front"))
                    continue
                reference_positions[entity.label] = anchor
                path = (points - track.smooth[anchor]) @ basis
            delta = path[-1] - path[0]
            noise = max(3 * track.noise, 0.03 * scene.scale)
            direction = _direction(delta, noise, axes)
            directions[entity.label] = direction
            counts[direction] += 1
            stamps = labels_for(ws, used[[0, -1]])
            reference = f"; axes from facing at {labels_for(ws, [anchor])[0]}" if frame_of_reference == "own" else ""
            output.append((f"{stamps[0]}–{stamps[1]}", entity.label,
                           f"{direction}; net {np.linalg.norm(ws.metric.convert(delta)):.2f} {ws.metric.units}{reference}"))
        spread, members = [], []
        for position in positions:
            present = [name for name, track in tracks.items() if track.present[position]]
            points = np.asarray([track.smooth[position] for track in tracks.values()
                                 if track.present[position] and np.isfinite(track.smooth[position]).all()])
            members.append(present)
            if len(points) < 2:
                spread.append(None)
            else:
                distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
                spread.append(float(ws.metric.convert(np.median(distances[np.triu_indices(len(points), 1)]))))
        intervals = []
        for i, visible in enumerate(members):
            if intervals and visible == intervals[-1][2]:
                intervals[-1][1] = i
            else:
                intervals.append([i, i, visible])
        stamps = labels_for(ws, positions)
        membership, spread_phases, extrema, temporal = [], [], [], []
        floor = float(ws.metric.convert(max(2 * max(track.noise for track in tracks.values()), 1e-8 * scene.scale)))
        for a, b, visible in intervals:
            membership.append({"positions": [int(positions[a]), int(positions[b])], "members": visible})
            finite = np.array([i for i in range(a, b + 1) if spread[i] is not None], dtype=int)
            if not len(finite):
                temporal.append((f"{stamps[a]}–{stamps[b]}", "group spread",
                                 f"{len(visible)} visible; spread unavailable: fewer than two measured members"))
                continue
            low = int(finite[np.argmin([spread[i] for i in finite])])
            high = int(finite[np.argmax([spread[i] for i in finite])])
            extrema.append({"minimum": spread[low], "minimum_position": int(positions[low]),
                            "maximum": spread[high], "maximum_position": int(positions[high]),
                            "members": visible})
            temporal.append((f"{stamps[a]}–{stamps[b]}", "spread extrema",
                             f"{len(visible)} visible; minimum {spread[low]:.2f} at {stamps[low]}; "
                             f"maximum {spread[high]:.2f} at {stamps[high]} {ws.metric.units}"))
            # Missing depths and membership changes are never joined into a phase.
            groups = np.split(finite, np.flatnonzero(np.diff(finite) != 1) + 1)
            for group in groups:
                if len(group) < 2:
                    i = int(group[0])
                    temporal.append((stamps[i], "group spread", f"observed {spread[i]:.2f} {ws.metric.units}"))
                    continue
                values = np.array([spread[i] for i in group])
                times = ws.times(scene.gf_index[positions[group]])
                clock = np.arange(len(group)) if times is None else times
                points = np.column_stack([values, np.zeros((len(values), 2))])
                phases = segment_motion(times, points, floor, time_labels=[stamps[i] for i in group],
                                        axes=(("spreading", "converging"), ("", ""), ("", "")))
                for phase in phases:
                    first = float(np.interp(phase.start, clock, values))
                    last = float(np.interp(phase.end, clock, values))
                    spread_phases.append({**asdict(phase), "members": visible})
                    temporal.append((phase.span, "group spread",
                                     f"{len(visible)} visible; {phase.label}; median pairwise distance "
                                     f"{first:.2f} → {last:.2f} {ws.metric.units}"))
        output.extend(pack_rows(temporal))
        ordered = counts.most_common()
        if ordered:
            lead, number = ordered[0]
            headline = f"{number} of {len(entities)} {category} instances: {lead.upper()}"
            headline += "".join(f"; {number} {direction}" for direction, number in ordered[1:]) + "."
        else:
            headline = "No instance has a measurable direction in the requested frame."
        if unresolved:
            headline += " Unresolved: " + ", ".join(unresolved) + "."
        if len(membership) > 1:
            headline += " Membership changes; spread is reported separately for each membership interval."
        note = ("Directions use net displacement in the requested frame. Own-frame motion uses each instance's "
                "starting facing. Spread is median pairwise distance; compare it within a constant membership interval. "
                "Direction counts summarize detected instances; the VLM decides the answer's count using the masks.")
        if ws.metric.available:
            note += f" Distances are metres; scale source: {ws.metric.source}."
        short = "Directions use net displacement; own axes are fixed at starting facing. Compare spread within each membership interval."
        return Report(title, headline, output, ("time", "instance or group", "reading"), note, short,
                      status="partial" if unresolved else "ok",
                      reason="Some instance directions are unavailable." if unresolved else "",
                      raw={"directions": directions, "counts": dict(counts), "unresolved": unresolved,
                           "reference_positions": reference_positions,
                           "spread": spread, "spread_phases": spread_phases, "spread_extrema": extrema,
                           "membership": membership, "positions": positions.tolist(),
                           "window": window})
    except WorkspaceError as error:
        return Report.unavailable(title, str(error), raw={"window": window})
