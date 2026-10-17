"""Entity motion and pair geometry in explicit world, camera, or subject axes."""

from __future__ import annotations

import numpy as np

from d4rt_agent.agent_v5 import orientation as geometry
from d4rt_agent.agent_v6.reports import AXES, Report, load_contract, motion_report
from d4rt_agent.agent_v6.workspace import CandidateRequired, WorkspaceError
from .orientation import basis_for, facing_rows, heading_report, labels_for, pack_rows, window_positions

CAMERA_AXES = (("right", "left"), ("down", "up"), ("away from camera", "towards camera"))
RANGE_AXES = (("farther apart", "closer together"), ("", ""), ("", ""))


def _speed_summary(times, points, start, end, units):
    """Path and endpoint rates over a real time interval, including its boundaries."""
    knots = np.r_[start, times[(times > start) & (times < end)], end]
    sampled = np.column_stack([np.interp(knots, times, points[:, axis]) for axis in range(3)])
    duration = float(end - start)
    path = float(np.linalg.norm(np.diff(sampled, axis=0), axis=1).sum())
    net = float(np.linalg.norm(sampled[-1] - sampled[0]))
    return {"start_s": float(start), "end_s": float(end), "duration_s": duration,
            "path_length": path, "net_displacement": net,
            "path_speed": path / duration, "net_displacement_rate": net / duration, "units": units}


def _add_speed(report):
    """Add model-visible rates without interpreting sample order as elapsed seconds."""
    report.raw["speed"] = None
    report.raw["phase_speeds"] = []
    if report.raw["times"] is None:
        report.headline += " Speed unavailable: the frame rate is unknown."
        return
    times = np.asarray(report.raw["times"], dtype=float)
    points = np.asarray(report.raw["positions"], dtype=float)
    units = "m/s" if report.raw["units"] == "metres" else "scene-relative units/s"
    overall = _speed_summary(times, points, times[0], times[-1], units)
    report.raw["speed"] = overall
    report.headline += (f" Path speed {overall['path_speed']:.2f} {units}; "
                        f"net displacement rate {overall['net_displacement_rate']:.2f} {units} over the observed span.")
    rows = []
    for phase in report.raw["phases"]:
        speed = _speed_summary(times, points, phase["start"], phase["end"], units)
        report.raw["phase_speeds"].append({**speed, "direction": phase["label"]})
        span = phase["start_label"][:-2] + "–" + phase["end_label"]
        rows.append((span, phase["label"],
                     f"moved {speed['path_length']:.2f} {report.raw['units']}; "
                     f"path speed {speed['path_speed']:.2f} {units}; "
                     f"net displacement rate {speed['net_displacement_rate']:.2f} {units}"))
    report.rows = pack_rows(rows)
    report.columns = ("time", "direction", "movement and speed")
    report.reading_full += (" Path speed divides the measured trajectory length by elapsed seconds; "
                            "net displacement rate divides endpoint separation by the same duration. "
                            "Rates use the report's stated coordinate frame.")


def motion_profile(ws, name, points, positions, noise, *, axes=AXES, reading="", include_speed=True) -> Report:
    """Run the shared phase reporter separately on each observed interval."""
    positions = np.asarray(positions, dtype=int)
    requested = len(positions)
    valid = np.isfinite(points[positions]).all(axis=1)
    positions = positions[valid]
    groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
    scale = ws.metric
    reports = []
    for group in groups:
        if len(group) < 2:
            continue
        reports.append(motion_report(
            name, ws.times(ws.scene().gf_index[group]), scale.convert(points[group]),
            float(scale.convert(max(float(noise), 1e-8 * ws.scene().scale))),
            time_labels=labels_for(ws, group), units=scale.units, metric_source=scale.source,
            axes=axes, reading_full=reading, reading_short="Motion uses the stated axes; missing intervals remain unmeasured.",
        ))
        reports[-1].raw["sample_positions"] = group.tolist()
        if include_speed:
            _add_speed(reports[-1])
    if not reports:
        return Report.unavailable(name, "Fewer than two consecutive measured positions.")
    if len(reports) == 1:
        result = reports[0]
    else:
        result = Report(name, " ".join(r.headline for r in reports),
                        [row for r in reports for row in r.rows], reports[0].columns,
                        reports[0].reading_full, reports[0].reading_short,
                        raw={"segments": [r.raw for r in reports]})
    if len(reports) > 1 or len(positions) != requested:
        result.status = "partial"
        result.reason = "Motion is reported only over consecutive measured positions; missing intervals are not bridged."
    result.rows = pack_rows(result.rows)
    source = f" (scale: {scale.source})" if scale.available else ""
    result.reading_short = f"Motion uses the stated axes; missing intervals remain unmeasured. Distances are {scale.units}{source}."
    return result


def _relative_report(ws, main, other, positions, rows) -> Report:
    scene = ws.scene()
    pa, pb = ws.track(main), ws.track(other)
    valid = np.isfinite(pa[positions]).all(1) & np.isfinite(pb[positions]).all(1)
    positions = positions[valid]
    title = f"{other} relative to {main}"
    if not len(positions):
        return Report.unavailable(title, "No common measured positions.")
    # Distance depends only on position. Sparse facing must never thin this series.
    distances = np.linalg.norm(pb[positions] - pa[positions], axis=1)
    metric_distances = ws.metric.convert(distances)
    distance_path = np.full_like(pa, np.nan)
    distance_path[positions] = np.column_stack([distances, np.zeros((len(positions), 2))])
    distance_report = motion_profile(ws, title + " distance", distance_path, positions,
                                     ws.track_record(main).noise + ws.track_record(other).noise,
                                     axes=RANGE_AXES, reading=load_contract("sub_obs_distance_change"), include_speed=False)
    output = [(row[0], "distance: " + row[1], row[2]) for row in distance_report.rows]
    stamps_all = labels_for(ws, positions)
    low, high = int(np.argmin(metric_distances)), int(np.argmax(metric_distances))
    extrema = (f"Distance ranges from {metric_distances[low]:.2f} at {stamps_all[low]} "
               f"to {metric_distances[high]:.2f} at {stamps_all[high]} {ws.metric.units}.")
    if not output:
        output = [(stamps_all[0], "distance observed", f"{metric_distances[0]:.2f} {ws.metric.units}")]
    by_position = {int(row["pos"]): row for row in rows}
    offsets, bearings, elevations, sides, used = [], [], [], [], []
    for pos in positions:
        basis = basis_for(scene, by_position.get(int(pos), {}))
        if basis is None:
            continue
        vector = (pb[pos] - pa[pos]) @ basis
        bearing = float(np.degrees(np.arctan2(vector[0], vector[2])))
        offsets.append(vector.tolist())
        bearings.append(bearing)
        elevations.append(float(np.degrees(np.arctan2(-vector[1], np.hypot(vector[0], vector[2])))))
        sides.append(geometry.sector(bearing)[1] if np.linalg.norm(vector) > 1e-8 else "coincident")
        used.append(int(pos))
    if used:
        stamps = labels_for(ws, used)
        groups = []
        for i, side in enumerate(sides):
            if groups and side == groups[-1][2] and used[i] == used[i - 1] + 1:
                groups[-1][1] = i
            else:
                groups.append([i, i, side])
        for a, b, side in groups:
            span = stamps[a] if a == b else f"{stamps[a]}–{stamps[b]}"
            output.append((span, f"observed side: {side}",
                           f"bearing {bearings[a]:+.0f}° → {bearings[b]:+.0f}°; "
                           f"vertical angle {elevations[a]:+.0f}° → {elevations[b]:+.0f}°"))
        sequence = [side for i, side in enumerate(sides) if i == 0 or side != sides[i - 1]]
        headline = f"At facing observations, {other} is {' → '.join(sequence)} of {main}. "
    else:
        headline = "The subject's sides are unavailable because no front resolves. "
    headline += extrema
    if distance_report.status != "unavailable":
        headline += " Distance: " + distance_report.headline
    source = f"; scale source: {ws.metric.source}" if ws.metric.available else ""
    note = (load_contract("orient_camera_subject_frame") + " " + load_contract("sub_obs_distance_change")
            + f" Distances are {ws.metric.units}{source}. Positive vertical angle means above. "
              "Distance uses every common position; side and angle observations require facing.")
    short = ("Sides use the subject's current facing; distance uses every common measured position. "
             "Missing facing does not remove distance observations.")
    partial = len(used) != len(positions) or distance_report.status != "ok"
    return Report(title, headline, pack_rows(output), ("time", "relation", "reading"), note, short,
                  status="partial" if partial else "ok",
                  reason="Distance and facing have separate observation coverage." if partial else "",
                  raw={"positions": used, "offsets": ws.metric.convert(offsets).tolist(),
                       "distance_positions": positions.tolist(), "distance": metric_distances.tolist(),
                       "distance_profile": distance_report.raw, "bearing_deg": bearings,
                       "vertical_angle_deg": elevations, "sides": sides, "other": other, "main": main})


def track_entity(ws, main, others=None, frame_of_reference="world", *, qwen=None, window=None) -> Report:
    title = f"Motion of {main} in the {frame_of_reference} frame"
    if frame_of_reference not in ("world", "camera", "subject"):
        raise WorkspaceError("frame_of_reference must be world, camera, or subject")
    try:
        main = ws.resolve(main).label
        others = [ws.resolve(name).label for name in (others or [])]
        positions = window_positions(ws, window)
        scene = ws.scene()
        pieces, motion, headings, relations = [], {}, {}, []
        rows_by_name = {}
        names = list(dict.fromkeys([main, *others]))
        if frame_of_reference != "camera":
            for name in names if frame_of_reference == "world" else [main]:
                rows = facing_rows(ws, name, qwen=qwen, positions=None if frame_of_reference == "subject" else positions)
                rows_by_name[name] = rows
                heading = heading_report(ws, name, [row for row in rows if row["pos"] in positions])
                headings[name] = heading.raw
                pieces.append(heading)
        if frame_of_reference == "subject":
            track = ws.track_record(main)
            good = np.flatnonzero(track.present & np.isfinite(track.smooth).all(1))
            by_position = {row["pos"]: row for row in rows_by_name[main]}
            basis = basis_for(scene, by_position.get(int(good[0]), {})) if len(good) else None
            if basis is None:
                own = Report.unavailable(f"{main} in its starting facing", "No resolvable front at the starting position.")
                pieces.append(own)
                title = f"World motion of {main}; starting-facing motion unavailable"
                path = scene.to_level(track.smooth)
                reading = "World motion uses the gravity-levelled starting view; forward is where the camera first looked."
            else:
                path = (track.smooth - track.smooth[good[0]]) @ basis
                reading = load_contract("traj_subject_own_frame")
            profile = motion_profile(ws, main, path, positions, 3 * track.noise, reading=reading)
            profile.raw["frame_of_reference"] = "world" if basis is None else "subject"
            profile.raw["starting_position"] = int(good[0]) if len(good) else None
            motion[main] = profile.raw
            pieces.insert(0, profile)
            for other in others or ["camera"]:
                relation = _relative_report(ws, main, other, positions, rows_by_name[main])
                pieces.append(relation)
                relations.append(relation.raw)
        else:
            for name in names:
                track = ws.track_record(name)
                if frame_of_reference == "world":
                    path = scene.to_level(track.smooth)
                    reading = "World motion uses the gravity-levelled starting view; forward is where the camera first looked. Camera motion is removed."
                    axes = AXES
                else:
                    path = np.einsum("ti,tij->tj", track.smooth - scene.cam_t, scene.cam_R)
                    if scene.pose_valid is not None:
                        path[~scene.pose_valid] = np.nan
                    reading = "Positions are in the camera's view at each moment, including apparent motion caused by camera movement."
                    axes = CAMERA_AXES
                profile = motion_profile(ws, name, path, positions, 3 * track.noise, axes=axes, reading=reading)
                profile.raw["frame_of_reference"] = frame_of_reference
                motion[name] = profile.raw
                pieces.insert(0, profile)
                if frame_of_reference == "camera":
                    distance = np.linalg.norm(track.smooth - scene.cam_t, axis=1)
                    radial = np.column_stack([distance, np.zeros((scene.length, 2))])
                    radial[~np.isfinite(path).all(1)] = np.nan
                    ranges = motion_profile(ws, f"{name} distance to camera", radial, positions, 3 * track.noise,
                                            axes=RANGE_AXES, reading=load_contract("sub_obs_distance_change"), include_speed=False)
                    pieces.append(ranges)
                    relations.append(ranges.raw)
        available = [piece for piece in pieces if piece.status != "unavailable"]
        if not available:
            return Report.unavailable(title, "; ".join(piece.reason for piece in pieces), raw={"window": window})
        missing = [f"{piece.title}: {piece.reason}" for piece in pieces if piece.status == "unavailable"]
        output = [(row[0], piece.title, "; ".join(str(v) for v in row[1:] if v != ""))
                  for piece in available for row in piece.rows]
        notes = "\n".join(dict.fromkeys(piece.reading_full for piece in available if piece.reading_full))
        if frame_of_reference == "camera":
            short = "Coordinates follow the moving camera; range is straight-line distance. Missing observations are not bridged."
        elif frame_of_reference == "world":
            short = "World motion uses the gravity-levelled starting view; positive heading means turning right. Position and facing have separate coverage."
        elif basis is None:
            short = "Displayed motion uses world axes because starting facing is unavailable. Distance remains available independently of facing."
        else:
            short = "Main motion uses its starting facing; others use its facing at each moment. Missing facing does not erase distance measurements."
        return Report(title, " ".join(f"{piece.title}: {piece.headline}" for piece in available),
                      pack_rows(output), ("time", "quantity", "reading"), notes, short,
                      status="partial" if missing or any(p.status == "partial" for p in available) else "ok",
                      reason="; ".join(missing), raw={"motions": motion, "headings": headings,
                                                    "relations": relations, "window": window})
    except CandidateRequired:
        raise
    except WorkspaceError as error:
        return Report.unavailable(title, str(error), raw={"window": window})
