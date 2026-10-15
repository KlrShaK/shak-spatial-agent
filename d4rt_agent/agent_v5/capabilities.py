"""Composable observations and reductions. No benchmark labels or answers enter this module.

SAM instances and visual observations use the full generator-frame axis, independently
of scene reconstruction. Geometry stays on its own timestamped axis. Visual assertions
retain their inspection provenance and are never relabelled as geometric measurements.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from d4rt_agent.agent_v5.program import EventOccurrence, InstanceSet, Window
from d4rt_agent.agent_v5.scene import EntityTrack, WindowUnresolved, lift_instances, resolve_window


def error(message):
    from d4rt_agent.agent_v5.tools import ToolError
    raise ToolError(message)


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def report(name, value, *, status="supported", note="", images=(), **scope):
    from d4rt_agent.agent_v5.tools import ToolResult
    value = serial(value)
    data = {"available": status != "unresolved", "status": status, "value": value,
            "limitations": [note] if note else [], **serial(scope)}
    text = f"## {name}\n{json.dumps(value, ensure_ascii=False)}\n{note}"
    return ToolResult(text, f"{name}: {status}", images=list(images), data=data)


def image_axis(ws):
    n, fps = ws.clip.length, ws.clip.fps
    return SimpleNamespace(length=n, gf_index=np.arange(n),
                           t_s=None if fps is None else np.arange(n) / fps,
                           clip={"dataset": getattr(ws.clip, "record", {}).get("dataset", "unknown")})


def bounds(ws, args, axis=None):
    axis = image_axis(ws) if axis is None else axis
    try:
        return resolve_window(axis, ws.window(args))[:2]
    except WindowUnresolved as exc:
        error(str(exc))


def frame_stamp(axis, index):
    """Public frame IDs are generator frames, never positions in a sparse reconstruction."""
    return {"frame": int(axis.gf_index[index]),
            "time_s": None if axis.t_s is None else float(axis.t_s[index])}


def runs(present):
    """All inclusive intervals, retaining gaps rather than first/last envelopes."""
    edges = np.diff(np.r_[False, np.asarray(present, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1) - 1).tolist()))


def split_entity(eid):
    if "[" in eid:
        base, index = eid.rstrip("]").split("[")
        return base, int(index)
    return eid, None


def image_series(ws, eid):
    base, index = split_entity(eid)
    # Explicit synthetic/provider tracks can supply image coordinates without depth.
    if base in ws._tracks and base not in ws.groundings:
        tr = ws._tracks[base][index or 0]
        return tr.centroid, tr.present, tr.area_frac, ws.scene()
    g = ws.grounding(base)
    if not g.count:
        error(f"{base}: no segmentation; this does not establish absence")
    if index is None:
        chosen = ws.selected.get(base)
        if chosen:
            indices = sorted({k for index in chosen for k in ws.instance_indices(base, index)})
        elif ws.entity(base).kind == "landmark_region":
            indices = list(range(g.count))
        else:
            indices = [int(np.argmax(g.area_frac.sum(axis=1)))]
    else:
        if not 0 <= index < g.count:
            error(f"{eid}: instance index outside segmentation")
        indices = ws.member_indices(base, index)
    mask = g.masks[indices].any(axis=0)
    centroids = np.full((mask.shape[0], 2), np.nan)
    for i in np.flatnonzero(mask.any(axis=(1, 2))):
        y, x = np.nonzero(mask[i])
        centroids[i] = [x.mean() / mask.shape[2], y.mean() / mask.shape[1]]
    return centroids, mask.any(axis=(1, 2)), mask.mean(axis=(1, 2)), image_axis(ws)


def census(ws, args):
    base = str(args.get("entity", "c1"))
    g = ws.grounding(base)
    lo, hi = bounds(ws, args)
    aliases = ws.identity_aliases.get(base, {})
    groups = {}
    for k in range(g.count):
        root = aliases.get(k, k)
        groups.setdefault(root, []).append(k)
    if ws.entity(base).kind == "group":
        count = len(ws.selected[base]) if base in ws.selected else len(ws.tracks_of(base))
        groups = {i: ws.member_indices(base, i) for i in range(count)}
    # Same-place, disjoint tracklets are identity hypotheses, not automatically two objects.
    from d4rt_agent.agent_v5.scene import GAP_MERGE_S
    fragments = []
    if ws.entity(base).kind == "category":
        positions = {root: np.flatnonzero(g.masks[indices].any(axis=(0, 2, 3)))
                     for root, indices in groups.items()}
        for a, ia in groups.items():
            va = positions[a]
            if not len(va):
                continue
            for b, ib in groups.items():
                vb = positions[b]
                if not len(vb) or va[-1] >= vb[0] or ws.clip.fps is None or \
                        (vb[0] - va[-1]) / ws.clip.fps > GAP_MERGE_S:
                    continue
                ma, mb = g.masks[ia, va[-1]].any(0), g.masks[ib, vb[0]].any(0)
                if (ma & mb).sum() / max(int((ma | mb).sum()), 1) >= 0.5:
                    fragments.append([a, b])
    in_window = {root for root, indices in groups.items()
                 if g.masks[indices, lo:hi + 1].any()}
    fragments = [pair for pair in fragments if all(k in in_window for k in pair)]
    uncertain = {k for pair in fragments for k in pair}
    rows = []
    for root, indices in groups.items():
        visible = g.masks[indices].any(axis=(0, 2, 3))
        if not visible[lo:hi + 1].any():
            continue
        intervals = runs(visible)
        rows.append({"id": f"{base}[{root}]", "indices": indices,
                     "sam_ids": [int(g.obj_ids[k]) for k in indices], "intervals_gf": intervals,
                     "sam_sessions": [int((g.sessions or (0,) * g.count)[k]) for k in indices],
                     "identity_unresolved": root in uncertain or
                         any(k in ws.unverified_instances.get(base, set()) for k in indices),
                     "possible_same_identity": [f"{base}[{b if a == root else a}]" for a, b in fragments if root in (a, b)],
                     "intervals_s": None if ws.clip.fps is None else
                         [[a / ws.clip.fps, b / ws.clip.fps] for a, b in intervals],
                     "frames_in_window": int(visible[lo:hi + 1].sum()),
                     "brief": int(visible.sum()) < 3})
    instance_set = InstanceSet(base, ws.revisions.get(base, 0), rows)
    return report("instance census", asdict(instance_set), status="partial",
                  note="IDs are local to this grounding revision. Category brief tracks are retained. "
                       "Inspect possible fragments, then merge_identity or observe distinct_identity before counting them.",
                  entities=[base], frame="image", window=[lo, hi], units="instances")


def current_observation(ws, entity, predicate, lo, hi):
    base, _ = split_entity(entity)
    for obs in reversed(list(ws.observations.values())):
        if obs["entity"] == entity and obs["predicate"] == predicate and \
                obs["revision"] == ws.revisions.get(base, 0) and obs["window"][0] <= lo and obs["window"][1] >= hi:
            return obs["value"]
    return "unknown"


def count_entities(ws, args):
    base = str(args.get("entity", "c1"))
    result = census(ws, args)
    rows = result.data["value"]["instances"]
    lo, hi = bounds(ws, args)
    predicate = str(args.get("predicate") or "")
    predicates = args.get("predicates") or ([predicate] if predicate else [])
    if not isinstance(predicates, list) or any(not isinstance(p, str) for p in predicates):
        error("predicates must be a list of predicate names to combine with AND")
    aggregation = str(args.get("aggregation", "unique"))
    if aggregation not in ("unique", "at_time", "max_simultaneous"):
        error("aggregation must be unique, at_time or max_simultaneous; event counts use event_reduce")
    if aggregation == "at_time" and lo != hi:
        error("at_time requires at_s or an instant question window")
    confirmed, unresolved = [], []
    occupancy = np.zeros(hi - lo + 1, dtype=int)
    for row in rows:
        values = [current_observation(ws, row["id"], p, lo, hi) for p in predicates]
        if row["identity_unresolved"] and (aggregation == "unique" or
                any(k in ws.unverified_instances.get(base, set()) for k in row["indices"])):
            values.append(current_observation(ws, row["id"], "distinct_identity", lo, hi))
        value = "false" if "false" in values else "unknown" if "unknown" in values else "true"
        if value == "true":
            confirmed.append(row["id"])
            visible = np.zeros(ws.clip.length, bool)
            for a, b in row["intervals_gf"]:
                visible[a:b + 1] = True
            occupancy += visible[lo:hi + 1]
        elif value == "unknown":
            unresolved.append(row["id"])
    count = len(confirmed) if aggregation == "unique" else int(occupancy.max(initial=0))
    return report("entity count", {"count": count, "aggregation": aggregation, "predicates": predicates,
                  "confirmed": confirmed, "unresolved": unresolved, "candidates": rows}, status="partial",
                  note="Count applies to detected identities only. Failed segmentation is not proof of zero objects.",
                  entities=[base], frame="image", window=[lo, hi], units="distinct_entities")


def evaluate_predicate(ws, args):
    """Batch a named geometric predicate; semantic qualifiers remain separate predicates."""
    from d4rt_agent.agent_v5.measure import TUNING
    base, frame, operation = str(args["entity"]), str(args["frame"]), str(args["operation"])
    if operation not in ("left", "right", "up", "down", "forward", "backward", "moving", "stationary"):
        error("unsupported geometric predicate")
    if frame == "image" and operation in ("forward", "backward"):
        error("image coordinates cannot establish forward/backward")
    name = f"{operation}:{frame}"
    candidates = census(ws, {"entity": base, **{k: args[k] for k in WINDOW_ARGS if k in args}}).data["value"]["instances"]
    lo, hi = bounds(ws, args)
    rows = []
    for candidate in candidates:
        eid = candidate["id"]
        if frame == "image":
            p, valid, _, axis = image_series(ws, eid)
            indices = np.flatnonzero(valid & np.isfinite(p).all(1))
            indices = indices[(indices >= lo) & (indices <= hi)]
            floor = TUNING["image_move_floor"]
        else:
            axis = ws.scene()
            a, b = bounds(ws, args, axis)
            p, valid = position(ws, eid)
            indices = np.flatnonzero(valid)
            indices = indices[(indices >= a) & (indices <= b)]
            floor = TUNING["move_floor_scales"]
        value, delta, excursion = "unknown", None, None
        if len(indices) >= 2:
            relative = p[indices] - p[indices[0]]
            if frame != "image":
                # A fixed initial body frame for each member, rather than scene axes.
                relative = frame_vectors(ws, relative, indices, {**args, "reference": eid}) / axis.scale
            if np.isfinite(relative).all():
                delta = relative[-1]
                excursion = float(np.linalg.norm(relative, axis=1).max())
                if operation in ("moving", "stationary"):
                    passed = excursion > floor if operation == "moving" else excursion <= floor
                else:
                    component, sign = {"left": (0, -1), "right": (0, 1), "up": (1, -1), "down": (1, 1),
                                       "forward": (2, 1), "backward": (2, -1)}[operation]
                    passed = float(delta[component]) * sign > floor
                value = "true" if passed else "false"
        ws.observations[ws.next_id()] = {"entity": eid, "predicate": name, "value": value, "window": [lo, hi],
            "revision": ws.revisions.get(base, 0), "evidence_ids": [], "source": "host_geometric_predicate"}
        rows.append({"id": eid, "value": value, "net": delta, "excursion": excursion, "floor": floor})
    ws.supersede({"count", "count_entities"})
    return report("geometric predicate", {"predicate": name, "rows": rows}, status="partial", entities=[base], frame=frame,
                  window=[lo, hi], note="Directions use net displacement during observed visibility; motion uses excursion. "
                  "Combine this predicate with separate semantic qualifiers using count_entities.predicates. Floors are estimated, not truth.")


def inspect(ws, args):
    from d4rt_agent.agent_v5.final_context import register, render_frame
    explicit = args.get("frame_ids")
    if explicit is not None:
        if not isinstance(explicit, list) or not 1 <= len(explicit) <= 8 or any(
                not isinstance(f, int) or isinstance(f, bool) or not 0 <= f < ws.clip.length for f in explicit):
            error("frame_ids must list one to eight valid generator-frame integers")
        frames = np.unique(explicit)
        lo, hi = int(frames[0]), int(frames[-1])
    else:
        lo, hi = bounds(ws, args)
        frames = np.unique(np.round(np.linspace(lo, hi, min(8, hi - lo + 1))).astype(int))
    target = args.get("entity")
    annotation = str(args.get("annotation", "mask" if target else "none"))
    if annotation not in ("mask", "box", "none"):
        error("annotation must be mask, box or none")
    if target:
        ws.grounding(split_entity(str(target))[0])
    iid = ws.next_id()
    images, visual_ids = [], []
    for f in frames:
        image, missing = render_frame(ws, target, int(f), annotation, bool(args.get("crop")))
        vid = f"visual:{iid}:{f}:{annotation}"
        register(ws, vid, image, [int(f)], entity=target, inspection_id=iid, annotation=annotation,
                 crop=bool(args.get("crop")), limitations=[missing] if missing else [])
        visual_ids.append(vid)
        images.append((vid + ": " + ws.clip.label(int(f)), image))
    revisions = {split_entity(str(target))[0]: ws.revisions.get(split_entity(str(target))[0], 0)} if target else {}
    ws.inspections[iid] = {"window": [lo, hi], "frames": frames.tolist(), "entity": target,
                           "revisions": revisions, "visual_ids": visual_ids}
    return report("targeted inspection", {"inspection_id": iid, **ws.inspections[iid]}, images=images,
                  note="Only these sampled frames were inspected; refine the window for short events.",
                  visual_ids=visual_ids, entities=[] if not target else [split_entity(target)[0]],
                  frame="image", window=[lo, hi])


def checked_inspections(ws, args):
    ids = args.get("evidence_ids", [])
    if not isinstance(ids, list) or not ids:
        error("cite at least one inspection ID")
    records = []
    for iid in ids:
        # An image ID from an inspection (visual:t2:52:mask) stands for that inspection (t2).
        visual = ws.visuals.get(iid) if isinstance(iid, str) else None
        if iid not in ws.inspections and visual and visual["inspection_id"] in ws.inspections:
            iid = visual["inspection_id"]
        record = ws.inspections.get(iid)
        if record is None:
            error(f"unknown inspection {iid}")
        if any(ws.revisions.get(e, 0) != r for e, r in record["revisions"].items()):
            error(f"inspection {iid} predates an identity change; inspect again")
        records.append(record)
    return records


def observe(ws, args):
    inspections = checked_inspections(ws, args)
    entity, predicate, value = str(args["entity"]), str(args["predicate"]), str(args["value"])
    base, _ = split_entity(entity)
    ws.entity(base)
    if value not in ("true", "false", "unknown"):
        error("value must be true, false or unknown")
    lo, hi = bounds(ws, args)
    if not any(r["window"][0] <= lo <= hi <= r["window"][1] for r in inspections):
        error("predicate scope must fit within a cited inspection")
    oid = ws.next_id()
    obs = {"entity": entity, "predicate": predicate, "value": value, "window": [lo, hi],
           "revision": ws.revisions.get(base, 0), "evidence_ids": args["evidence_ids"]}
    ws.observations[oid] = obs
    ws.supersede({"count", "count_entities"})
    return report("visual predicate", {"observation_id": oid, **obs}, status="partial" if value == "unknown" else "supported",
                  source="vlm_observation", entities=[base], frame="image", window=[lo, hi],
                  note="A model observation on sampled images, not a geometric or ground-truth label.")


def merge_identity(ws, args):
    checked_inspections(ws, args)
    base = str(args["entity"])
    g = ws.grounding(base)
    indices = list(dict.fromkeys(int(i) for i in args["instances"]))
    if len(indices) < 2 or any(i < 0 or i >= g.count for i in indices):
        error("provide at least two valid instance indices")
    aliases = ws.identity_aliases.setdefault(base, {})
    roots = {aliases.get(k, k) for k in indices}
    root = min(roots)
    for k in range(g.count):
        if aliases.get(k, k) in roots:
            aliases[k] = root
            ws.unverified_instances.get(base, set()).discard(k)
    chosen = ws.selected.get(base, [])
    if ws.entity(base).kind == "group" and len({aliases.get(k, k) for k in chosen}) < len(chosen):
        ws.selected.pop(base)
    ws.invalidate(base)
    return report("identity reconciliation", {"entity": base, "aliases": aliases}, entities=[base],
                  source="vlm_identity_review", note="Merge retains all original SAM IDs; inspect again after this revision.")
