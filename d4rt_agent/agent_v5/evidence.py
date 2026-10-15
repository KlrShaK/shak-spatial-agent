"""Reporting over existing observations. Never reconstruct or ground to fill metadata."""

from dataclasses import asdict
import json

import numpy as np

from d4rt_agent.agent_v5.capabilities import image_axis, runs, serial
from d4rt_agent.agent_v5.program import EvidenceRecord
from d4rt_agent.agent_v5.scene import resolve_window


# Observable, reduction, coordinate frame, units; compound components have independent records.
DESCRIPTIONS = {
    "ground": ("segmentation candidates", "discovery", "image", "instances"),
    "select": ("instance binding", "selection", "image", "instance_indices"),
    "show": ("annotated inspection", "sampled frames", "image", "pixels"),
    "inspect": ("targeted inspection", "sampled frames", "image", "pixels"),
    "camera_motion": ("camera trajectory", "excursion, net, chronology", "scene", {"rotation": "degrees", "raw_translation": "reconstruction_unit", "normalized_translation": "scene_scales"}),
    "camera_profile": ("camera profile", "compound", "scene", None),
    "rhythm": ("camera motion energy", "temporal profile", "scene", "dimensionless"),
    "jitter": ("camera pose residual", "RMS", "scene", {"rotation": "degrees", "translation": "scene_scales"}),
    "camera_zoom": ("lens and translation", "ratio and net", "camera intrinsics; scene translation", {"focal": "ratio", "translation": "scene_scales"}),
    "scene_revisit": ("view overlap", "first versus last", "world0", "fraction"),
    "entity_track": ("entity trajectory", "net and chronology", "scene", "scene_scales"),
    "distance": ("scalar separation", "value or trend", "world0", {"raw_units": "reconstruction_unit", "relative": "scene_scales", "metric": "metres_if_scaled"}),
    "region_relation": ("nearest surface distance", "trend", "world0", "reconstruction_unit"),
    "elevation": ("view elevation", "median and range", "scene", "degrees"),
    "visibility": ("detected visibility", "intervals", "image", "fraction"),
    "apparent_size": ("projected area", "trend", "image", "image_fraction"),
    "motion_profile": ("entity motion", "temporal profile", "world0", "scene_scales/s_if_known"),
    "speed": ("net displacement and path rates", "calibrated net/path selection", "world0", {"raw_speed": "reconstruction_unit/s", "metric_speed": "metres/s_if_scaled"}),
    "count": ("detected identities", "scoped aggregation", "image", "distinct_entities"),
    "count_entities": ("detected qualifying identities", "scoped aggregation", "image", "distinct_entities"),
    "census": ("detected identity set", "visibility intervals", "image", "instances"),
    "evaluate_predicate": ("geometric qualification", "per instance", None, None),
    "observe": ("visual predicate", "one scoped assertion", "image", None),
    "merge_identity": ("identity reconciliation", "aliases", "image", None),
    "view_state": ("projected position and extent", "visibility intervals and samples", "image", "image_fraction"),
    "relative_position": ("signed relative offset", "samples", "scene", "scene_scales"),
    "kinematics": ("path speed and acceleration", "observed interval statistics", "world0", "scene_scales/s"),
    "orbit": ("relative angular trajectory", "signed turn and angular path", "scene", "degrees"),
    "group_geometry": ("group configuration", "dispersion and alignment", "world0", "scene_scales"),
    "event_onset": ("signal transitions", "first and all onsets", "time", "generator_frames"),
    "locate_event": ("signal transitions", "all onsets", "time", "generator_frames"),
    "record_event": ("visual occurrence", "localized interval", "time", "generator_frames"),
    "event_reduce": ("localized occurrences", "count/order/duration", "time", None),
    "event_window": ("resolved temporal scope", "event anchors", "time", "generator_frames"),
    "orientation": ("body and viewing orientation", "compound", None, "degrees"),
    "facing": ("body facing", "sampled facing and accumulated turn", "camera (facing); gravity-levelled world0 (heading)", "degrees"),
    "camera_bearing": ("viewing side", "sampled bearing", "body_current", "degrees"),
    "track_quality": ("correspondence diagnostics", "cohort/surface comparison", "world0", "scene_scales"),
    "projected_rotation": ("projected point rotation", "accumulated signed turn", "image", "degrees"),
    "region_geometry": ("projected boundary relation", "sampled containment and tangent", "image", "image_fraction"),
    "encounters": ("encounter proposals", "candidate intervals", None, None),
}

IMAGE_TOOLS = {"ground", "select", "show", "inspect", "count", "count_entities", "census", "evaluate_predicate",
               "observe", "merge_identity", "view_state", "region_geometry", "record_event", "event_reduce", "event_window"}
ADMINISTRATIVE = {"ground", "select", "show", "inspect", "merge_identity", "event_window"}


def is_current(ws, record):
    return not record.superseded and all(ws.revisions.get(e, 0) == r for e, r in record.revisions.items())


def _tracks(ws, entities):
    return [t for eid in entities for t in ws._tracks.get(eid.split("[")[0], [])
            if "[" not in eid or t.id == eid]


def _indices(ws, entity, name):
    """Read the cached binding without triggering grounding or reconstruction."""
    base = entity.split("[")[0]
    g = ws.groundings.get(base)
    if g is None:
        return sorted({int(k) for t in _tracks(ws, [entity]) for k in t.instances})
    aliases = ws.identity_aliases.get(base, {})
    if "[" in entity:
        member = int(entity.split("[")[1].rstrip("]"))
        if ws.entity(base).kind == "group":
            return ws.member_indices(base, member)
        root = aliases.get(member, member)
        return [k for k in range(g.count) if aliases.get(k, k) == root]
    if name in ("ground", "count", "count_entities", "census"):
        return list(range(g.count))
    tracks = [t for t in ws._tracks.get(base, []) if t.id == entity]
    if tracks:
        return sorted({int(k) for t in tracks for k in t.instances})
    if base in ws.selected:
        return sorted({k for index in ws.selected[base] for k in ws.instance_indices(base, index)})
    if name in ("inspect", "show"):
        return list(range(g.count))
    if name in ("view_state", "region_geometry", "entity_track") and g.count:
        if ws.program.entity(base).kind == "landmark_region":
            return list(range(g.count))
        return [int(np.argmax(g.area_frac.sum(axis=1)))]
    return []


def _coverage(ws, name, args, data, entities):
    image = name in IMAGE_TOOLS or args.get("frame") == "image" or data.get("frame") == "image"
    axis = image_axis(ws) if image else ws._scene
    window = None
    try:
        window = ws.window({"frames": data["value"]["window"]}) if name == "inspect" and data.get("value", {}).get("window") else ws.window(args)
        lo, hi = resolve_window(axis, window)[:2] if axis is not None else (None, None)
    except (ValueError, TypeError):
        lo = hi = None
    output = {"requested": asdict(window) if window else None, "intervals_gf": None,
              "intervals_s": None, "valid_samples": None, "expected_samples": None,
              "visibility_gaps": {}, "depth_gaps": {}, "detection_completeness": "unknown"}
    if lo is None:
        return output
    output["expected_samples"] = hi - lo + 1
    output["sampling"] = "generator_frames" if image else "sampled_geometry"
    available = data.get("available") is not False and data.get("found") is not False and "error" not in data
    samples = None
    if available:
        if name == "inspect":
            samples = data.get("value", {}).get("frames")
        elif name == "show":
            samples = ws.clip.keyframes(4)
        elif name in ("ground", "select", "census", "count", "count_entities", "view_state", "visibility"):
            samples = list(range(lo, hi + 1))
        elif name in ("facing", "camera_bearing"):
            samples = data.get("valid_frames")
        elif "valid_frames" in data:
            samples = data["valid_frames"]
        else:
            series = data.get("series", {})
            for key in ("range", "area", "elevation", "smooth", "signal"):
                if key in series:
                    values = np.asarray(series[key], dtype=float)
                    if values.ndim == 1:
                        samples = np.flatnonzero(np.isfinite(values)).tolist()
                        break
            if samples is None and "frames" in series:
                samples = series["frames"]
            if samples is None and name in ("camera_motion", "rhythm", "jitter", "camera_zoom"):
                samples = list(range(lo, hi + 1))
            if samples is None and name == "scene_revisit":
                samples = [lo, hi]
            if samples is None and name in ("entity_track", "motion_profile", "kinematics"):
                tracks = _tracks(ws, entities)
                if len(tracks) == 1:
                    samples = np.flatnonzero(tracks[0].present & np.isfinite(tracks[0].smooth).all(1)).tolist()
            if samples is None:
                rows = data.get("value", {}).get("rows", [])
                frames = [r["frame"] for r in rows if isinstance(r, dict) and "frame" in r]
                if frames:
                    samples = [int(np.argmin(abs(axis.gf_index - f))) for f in frames]
    if samples is not None:
        samples = sorted({int(i) for i in samples if lo <= int(i) <= hi})
        if not image and axis.pose_valid is not None and name in ("camera_motion", "rhythm", "jitter", "camera_zoom"):
            samples = [i for i in samples if axis.pose_valid[i]]
        output["valid_samples"] = len(samples)
        output["sample_gfs"] = [int(axis.gf_index[i]) for i in samples]
        intervals = runs(np.isin(np.arange(axis.length), samples))
        output["intervals_gf"] = [[int(axis.gf_index[a]), int(axis.gf_index[b])] for a, b in intervals]
        if axis.t_s is not None:
            output["intervals_s"] = [[float(axis.t_s[a]), float(axis.t_s[b])] for a, b in intervals]
            output["requested_span_s"] = float(axis.t_s[hi] - axis.t_s[lo])
            output["observed_span_s"] = sum(float(axis.t_s[b] - axis.t_s[a]) for a, b in intervals)
    scene = ws._scene
    if scene is not None and not image:
        for tr in _tracks(ws, entities):
            for field, missing in (("visibility_gaps", ~tr.present),
                                   ("depth_gaps", tr.present & ~np.isfinite(tr.xyz).all(1))):
                gaps = runs(missing[lo:hi + 1])
                output[field][tr.id] = [[int(scene.gf_index[a + lo]), int(scene.gf_index[b + lo])] for a, b in gaps]
    if image:
        for eid in entities:
            g = ws.groundings.get(eid.split("[")[0])
            indices = _indices(ws, eid, name)
            if g is not None and indices:
                present = g.masks[indices].any(axis=(0, 2, 3))
                output["visibility_gaps"][eid] = [[a + lo, b + lo] for a, b in runs(~present[lo:hi + 1])]
    return output


def _display_series(ws, component, data, record):
    scene, series = ws._scene, data.get("series", {})
    frames = series.get("frames", [])
    if scene is None or not len(frames):
        return None
    frames = np.asarray(frames, int)
    columns = {k: np.asarray(v, float) for k, v in series.items() if k not in ("frames", "t", "picks", "tol", "series_used")
               and isinstance(v, (list, np.ndarray)) and np.asarray(v).ndim in (1, 2) and len(v) == len(frames)}
    if not columns:
        return None
    valid = (frames >= 0) & (frames < scene.length)
    valid &= np.logical_or.reduce([np.isfinite(v).all(1) if v.ndim == 2 else np.isfinite(v) for v in columns.values()])
    if component in ("camera_motion", "rhythm", "jitter", "camera_zoom") and scene.pose_valid is not None:
        valid &= scene.pose_valid[np.clip(frames, 0, scene.length - 1)]
    observed = record.observations.get("sample_gfs")
    if observed is not None:
        valid &= np.isin(scene.gf_index[np.clip(frames, 0, scene.length - 1)], observed)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return {"rows": [], "note": "No supported samples survived the observation diagnostics."}
    picks = set(indices[np.unique(np.linspace(0, len(indices) - 1, min(8, len(indices))).astype(int))])
    pivots = data.get("value", {}).get("pivots", [])
    picks.update(int(i) for i in indices if int(frames[i]) in pivots)
    units = {k: "reconstruction_unit" if k in ("range", "smooth", "pos", "xyz", "trans") else
             "degrees" if k.startswith(("pan", "tilt", "roll")) or k == "elevation" else
             "dimensionless" if k in ("energy", "moving", "present") else
             "image_fraction" if k == "area" else "pixels" if k == "focal" else
             "log_focal_ratio" if k == "smooth_log" else
             "scene_scales/s" if k == "speed" and component == "motion_profile" else
             "native_estimator_units" for k in columns}
    if "trans" in columns:
        columns["translation_scene_scales"] = columns.pop("trans") / scene.scale
        units.pop("trans")
        units["translation_scene_scales"] = "scene_scales"
    rows = [{"frame": int(scene.gf_index[frames[i]]), "time_s": None if scene.t_s is None else float(scene.t_s[frames[i]]),
             **{k: serial(v[i]) for k, v in columns.items()}} for i in sorted(picks)]
    return {"rows": rows, "units": units, "note": "Display samples and existing turning points; full series remain in the audit."}


def _display_value(ws, component, value):
    """Translate known legacy sample indices for the packet; retain raw values in the audit."""
    if ws._scene is None or not isinstance(value, dict):
        return value
    scene = ws._scene
    value = dict(value)

    def gf(index):
        return None if index is None else int(scene.gf_index[int(index)])

    if component in ("distance", "region_relation", "camera_zoom"):
        if "pivots" in value:
            value["pivots_gf"] = [gf(i) for i in value.pop("pivots")]
        if "plateau_from" in value:
            value["plateau_from_fraction"] = value.pop("plateau_from")
    if component == "distance":
        for key in ("frame", "requested_frame"):
            if key in value:
                value[key] = gf(value[key])
        if "frame_gap" in value:
            value["estimator_sample_gap"] = value.pop("frame_gap")
            value["frame_gap_gf"] = abs(value["frame"] - value["requested_frame"])
    if component == "event_onset":
        value["onset"] = gf(value.get("onset"))
        value["occurrences"] = [{**row, "onset": gf(row["onset"]), "bracket": [gf(i) for i in row["bracket"]]}
                                for row in value.get("occurrences", [])]
    return value


def attach(ws, requirement, name, args, result):
    """Attach independent components to the canonical workspace evidence registry."""
    data = result.data
    entities = data.get("entities") or [str(args[k]) for k in ("entity", "a", "b", "actor", "reference", "landmark") if args.get(k)]
    if name == "show":
        target = str(args.get("target", "entities"))
        entities = [e.id for e in ws.program.entities if e.kind != "camera"] if target == "entities" else ["cam" if target == "camera" else target]
    entities = list(dict.fromkeys(entities))
    explicit_members = {str(args[k]).split("[")[0]: str(args[k]) for k in ("entity", "a", "b", "actor", "reference", "landmark")
                        if args.get(k) and "[" in str(args[k])}
    if name in ("inspect", "observe") and data.get("value", {}).get("entity"):
        target = str(data["value"]["entity"])
        if "[" in target:
            explicit_members[target.split("[")[0]] = target
    entities = [explicit_members.get(e, e) for e in entities]
    if not entities and (name.startswith("camera_") or name == "scene_revisit"):
        entities = ["cam"]
    revisions = {e.split("[")[0]: ws.revisions.get(e.split("[")[0], 0) for e in entities}
    components = {name: data}
    if name == "camera_profile":
        components = {k: data[k] for k in ("rhythm", "jitter", "presence") if isinstance(data.get(k), dict)} or components
    elif name == "orientation" and "rows" in data:
        components = {"facing": {**data, "valid_frames": [r["pos"] for r in data["rows"] if r.get("usable")]},
                      "camera_bearing": {**data, "valid_frames": data.get("bearing_positions", []),
                                         "available": bool(data.get("bearing_positions"))}}
    for component, values in components.items():
        quantity, reduction, frame, units = DESCRIPTIONS.get(component, DESCRIPTIONS[name])
        available = values.get("available") is not False and values.get("found") is not False and "error" not in values
        status = values.get("status", "partial" if available else "unresolved")
        if not available or status == "unresolved":
            available, status = False, "unresolved"
        observed = _coverage(ws, component, args, values, entities)
        limitations = list(dict.fromkeys(values.get("notes", []) + values.get("limitations", []) + [str(values[k]) for k in
            ("reason", "sampling_note", "reference_point", "window_note") if values.get(k)]))
        if observed["visibility_gaps"] and any(observed["visibility_gaps"].values()):
            limitations.append("Nonvisibility intervals are unobserved; segmentation failure and occlusion can mimic absence.")
        if component in ("count", "count_entities", "census"):
            limitations.append("Counts concern detected identities; completeness is not established.")
        if ws.clip.fps is None:
            limitations.append("FPS is unknown; seconds-based timing is unavailable or explicitly provisional.")
        if values.get("value", {}).get("unresolved"):
            limitations.append("Some candidate predicates or identities remain unresolved.")
        diagnostic = values.get("track_diagnostics", {})
        limitations.extend(diagnostic.get("limitations", []))
        if diagnostic.get("status") == "conflicting":
            limitations.append("Persistent-point and visible-surface displacement conflict; the surface centre is not a verified material track.")
        source_ids = []
        if ws._scene is not None and component not in IMAGE_TOOLS:
            source_ids.append(f"{ws.clip.slug}:geometry")
        selected = {}
        for entity in entities:
            base = entity.split("[")[0]
            if base in ws.groundings or base in ws._tracks:
                source_ids.append(f"{ws.clip.slug}:entity:{base}:revision:{ws.revisions.get(base, 0)}")
                selected[entity] = _indices(ws, entity, component)
                if component in ("count", "count_entities", "census") and base in ws.groundings:
                    selected[entity] = sorted({ws.identity_aliases.get(base, {}).get(k, k) for k in range(ws.groundings[base].count)})
        visual_ids = list(values.get("visual_ids", [])) + list(values.get("value", {}).get("evidence_ids", []))
        record = EvidenceRecord(ws.next_id(), quantity, entities, revisions, status,
            frame=values.get("frame", args.get("frame", frame)), window=observed["requested"],
            units=values.get("units", units), source=values.get("source", "host_measurement"),
            requirement=requirement.id, component=component, source_ids=source_ids, reduction=values.get("reduce", reduction),
            selected_instances=selected, observations=observed,
            quality={"label": values.get("confidence") or "unknown",
                     "basis": "heuristic" if values.get("confidence") else "unknown",
                     "reasons": values.get("notes", []), "thresholds": values.get("floor", {})},
            limitations=limitations, visual_ids=visual_ids, available=available, quantity=quantity)
        if record.frame in ("body_initial", "body_current", "own"):
            record.axes = "+x reference-body right, +y body down, +z body forward; anchor/reference are in the requirement"
        elif record.frame == "image":
            record.axes = "+x image right, +y image down; no depth coordinate"
        elif component == "camera_zoom":
            record.axes = "Focal ratios describe camera intrinsics; supporting translation uses gravity-levelled initial-camera axes, +z forward"
        elif component in ("camera_motion", "rhythm", "jitter"):
            record.axes = "+x right, +y down, +z forward; pan +right, tilt +up, roll +clockwise as seen by camera"
        elif record.frame in ("scene", "world0", "camera"):
            record.axes = "+x right, +y down, +z forward; scene is gravity-levelled, world0 uses initial-camera axes"
        if component == "camera_bearing":
            record.axes = "0 degrees body front, +90 body right, +/-180 back, -90 body left"
        # The observable remains the callable name for compatibility with existing requirement checks.
        record.observable = name
        ws.evidence[record.id] = record
        result.evidence.append(record)
        if component in ("distance", "region_relation", "camera_zoom", "event_onset") and isinstance(values.get("value"), dict):
            result.data.setdefault("display_values", {})[component] = _display_value(ws, component, values["value"])
        display = _display_series(ws, component, values, record)
        if display is not None:
            result.data.setdefault("display_series", {})[component] = display
    result.data["evidence"] = serial(asdict(result.evidence[0]))
    result.data["evidence_records"] = [serial(asdict(r)) for r in result.evidence]
    result.text = "\n".join(f"Evidence {r.id} ({r.status}, requirement {r.requirement}, component {r.component})" for r in result.evidence) + "\n" + result.text
    return result


def current_components(ws, result):
    return [r for r in result.evidence if is_current(ws, r)]


def compact_result(result, records):
    """One packet per observable; raw series stay in the saved tool data."""
    data = result.data
    value = data.get("value")
    if value is None:
        value = {k: v for k, v in data.items() if k not in ("series", "evidence", "evidence_records", "error", "track_diagnostics", "display_series", "display_values")}
    if isinstance(value, dict):
        value = dict(value)
        if "candidates" in value:
            value["candidates"] = [{k: r[k] for k in ("id", "identity_unresolved", "intervals_gf") if k in r} for r in value["candidates"]]
        if "instances" in value and isinstance(value["instances"], list):
            value["instances"] = [{k: r[k] for k in ("id", "identity_unresolved", "intervals_gf") if k in r} for r in value["instances"]]
    packets = []
    for record in records:
        component_value = data.get(record.component, {}).get("value") if record.component != record.observable else value
        if record.component in data.get("display_values", {}):
            component_value = data["display_values"][record.component]
        if record.component == "camera_bearing":
            component_value = {"bearings": data.get("bearings", []), "available": record.available}
        elif record.component == "facing":
            component_value = {"cumulative_turn_deg": data.get("cumulative_turn_deg"), "samples": data.get("facing_samples", [])}
        metadata = serial(asdict(record))
        metadata["observations"].pop("sample_gfs", None)  # exact sample IDs remain in the audit record
        packet = {"evidence": metadata, "value": component_value if component_value is not None else value}
        if record.component in data.get("display_series", {}):
            packet["samples"] = data["display_series"][record.component]
        packets.append(packet)
    return json.dumps(packets, ensure_ascii=False, separators=(",", ":"))


def short_result(result, records):
    packets = json.loads(compact_result(result, records))
    for packet in packets:
        evidence = packet["evidence"]
        packet["evidence"] = {k: evidence[k] for k in ("id", "quantity", "component", "requirement", "entities", "status",
            "available", "frame", "axes", "units", "window", "source_ids", "limitations")}
        packet["evidence"]["quality"] = {k: evidence["quality"][k] for k in ("label", "basis")}
        packet["evidence"]["observations"] = {k: evidence["observations"].get(k) for k in
            ("valid_samples", "expected_samples", "sampling", "detection_completeness")}
        packet["context_limit"] = "Coverage details and full series were omitted; they remain in the saved audit."
    return json.dumps(packets, ensure_ascii=False, separators=(",", ":"))
