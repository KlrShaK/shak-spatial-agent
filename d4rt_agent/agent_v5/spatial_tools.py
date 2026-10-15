"""Correspondence diagnostics, projected rotation, regions and encounter proposals."""

from __future__ import annotations

import numpy as np

from d4rt_agent.agent_v5.capabilities import (
    bounds, census, error, frame_stamp, image_series, position, report, runs, split_entity, spec, WINDOW_ARGS,
)


def cohort(ws, eid, anchor_s=0.0):
    scene = ws.scene()
    if scene.t_s is None and anchor_s:
        error("a timed query anchor requires established FPS")
    anchor = int(np.argmin(np.abs(scene.t_s - anchor_s))) if scene.t_s is not None else 0
    base, index = split_entity(eid)
    g = ws.grounding(base)
    if not g.count:
        error("no segmented instance for correspondence checking")
    indices = ws.member_indices(base, index) if index is not None else ws.selected.get(base, [int(np.argmax(g.area_frac.sum(1)))])
    if any(k < 0 or k >= g.count for k in indices):
        error("instance index outside segmentation")
    masks = g.masks[indices].any(0)[scene.gf_index]
    tracks = scene.grid_traj.astype(float)
    if getattr(ws, "_point_fn", None) is not None and anchor:
        query = ws._point_fn(ws.clip, scene, g, indices, anchor)
        tracks = query["tracks"]
        ws.point_provenance[(eid, anchor)] = query["diagnostics"]
    local = np.einsum("tni,tij->tnj", tracks - scene.cam_t[:, None], scene.cam_R)
    uvz = np.einsum("tij,tnj->tni", scene.K, local)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvz[..., :2] / uvz[..., 2:3]
    h, w = scene.model_hw
    mh, mw = masks.shape[1:]
    xy = uv * [mw / w, mh / h]
    valid = np.isfinite(xy).all(2) & (local[..., 2] > 0)
    valid &= (xy[..., 0] >= 0) & (xy[..., 0] < mw) & (xy[..., 1] >= 0) & (xy[..., 1] < mh)
    if scene.pose_valid is not None:
        valid &= scene.pose_valid[:, None]
    safe = np.nan_to_num(xy, nan=0, posinf=0, neginf=0)
    ix = np.clip(safe[..., 0], 0, mw - 1).astype(int)
    iy = np.clip(safe[..., 1], 0, mh - 1).astype(int)
    membership = valid & masks[np.arange(scene.length)[:, None], iy, ix]
    seeds = membership[anchor]
    return scene, anchor, tracks[:, seeds], uv[:, seeds], membership[:, seeds], valid[:, seeds]


def track_quality(ws, args):
    eid = str(args["entity"])
    scene, anchor, tracks, uv, membership, valid = cohort(ws, eid, float(args.get("anchor_s", 0)))
    if tracks.shape[1] < 3:
        return report("track quality", {"seeds": tracks.shape[1], "anchor_frame": int(scene.gf_index[anchor])}, status="unresolved",
                      note="Fewer than three anchored correspondences. Inspect/re-ground at a clearer time.")
    centre, centre_valid = position(ws, eid)
    displacement = np.nanmedian(np.where(valid[..., None], tracks - tracks[anchor], np.nan), axis=1)
    surface = centre - centre[anchor]
    lo, hi = bounds(ws, args, scene)
    sample = np.unique(np.linspace(lo, hi, min(8, hi - lo + 1)).astype(int))
    rows = [{**frame_stamp(scene, i), "mask_agreement": float(membership[i].mean()),
             "persistent_delta": displacement[i] / scene.scale,
             "surface_delta": surface[i] / scene.scale,
             "disagreement_scales": float(np.linalg.norm(displacement[i] - surface[i]) / scene.scale)} for i in sample]
    from d4rt_agent.agent_v5.measure import TUNING
    disagreement = np.linalg.norm(displacement[lo:hi + 1] - surface[lo:hi + 1], axis=1) / scene.scale
    conflicting = bool(np.any(disagreement > TUNING["move_floor_scales"]))
    return report("track quality", {"anchor_frame": int(scene.gf_index[anchor]), "seeds": tracks.shape[1], "rows": rows,
                  "reseed_alignment": ws.point_provenance.get((eid, anchor)),
                  "pose_validity_known": scene.pose_valid is not None}, status="conflicting" if conflicting else "partial", entities=[eid],
                  frame="world0", units="scene_scales", window=[int(scene.gf_index[lo]), int(scene.gf_index[hi])],
                  note="Persistent cohort and visible-surface centre are different estimators. Mask disagreement can reflect drift or occlusion. "
                       "Neither small smoothing residuals nor tracker confidence alone establish identity; confidence is not a probability.")


def similarity_angle(a, b):
    """Proper least-squares 2D rotation; reflections are not rotations."""
    a, b = a - a.mean(0), b - b.mean(0)
    if len(a) < 3 or np.linalg.matrix_rank(a) < 2 or np.linalg.matrix_rank(b) < 2:
        return None
    u, _, vt = np.linalg.svd(a.T @ b)
    fix = np.diag([1, np.linalg.det(u @ vt)])
    rotation = u @ fix @ vt
    return float(np.arctan2(rotation[0, 1], rotation[0, 0]))


def projected_rotation(ws, args):
    eid = str(args["entity"])
    scene, anchor, tracks, uv, membership, valid = cohort(ws, eid, float(args.get("anchor_s", 0)))
    lo, hi = bounds(ws, args, scene)
    # A fixed cohort avoids replacing lost points with a different visible surface each frame.
    fixed = membership[lo:hi + 1].all(0)
    if fixed.sum() < 3:
        return report("projected rotation", {"persistent_points_in_all_masks": int(fixed.sum())}, status="unresolved",
                      note="Narrow the inspected interval or reseed. Mask silhouette alone cannot measure a spinning symmetric wheel.")
    angles = [similarity_angle(uv[i, fixed], uv[i + 1, fixed]) for i in range(lo, hi)]
    if not angles or any(a is None for a in angles):
        return report("projected rotation", {}, status="unresolved", note="Degenerate projected point configuration.")
    degrees = np.degrees(np.cumsum([0.0, *angles]))
    sample = np.unique(np.linspace(0, len(degrees) - 1, min(8, len(degrees))).astype(int))
    return report("projected rotation", {"signed_degrees": float(degrees[-1]),
                  "angular_path_degrees": float(np.abs(np.degrees(angles)).sum()),
                  "rows": [{**frame_stamp(scene, lo + i), "degrees": float(degrees[i])} for i in sample]},
                  status="partial", entities=[eid], frame="image", units="degrees",
                  window=[int(scene.gf_index[lo]), int(scene.gf_index[hi])],
                  note="Positive is clockwise in the image (+y down). Includes camera roll and perspective; sampling may alias fast spin.")


def region_geometry(ws, args):
    from scipy.ndimage import distance_transform_edt
    eid, landmark = str(args["entity"]), str(args.get("landmark", "l1"))
    xy, present, _, axis = image_series(ws, eid)
    g = ws.grounding(landmark)
    mask = g.masks.any(0)
    lo, hi = bounds(ws, args, axis)
    rows = []
    previous_tangent = None
    for i in np.unique(np.linspace(lo, hi, min(8, hi - lo + 1)).astype(int)):
        if not present[i] or not mask[i].any():
            continue
        h, w = mask.shape[1:]
        x, y = np.clip((xy[i] * [w, h]).astype(int), [0, 0], [w - 1, h - 1])
        yy, xx = np.nonzero(mask[i])
        pixels = np.column_stack([xx / w, yy / h])
        _, _, vt = np.linalg.svd(pixels - pixels.mean(0), full_matrices=False)
        tangent = vt[0]
        if previous_tangent is not None and tangent @ previous_tangent < 0:
            tangent = -tangent
        previous_tangent = tangent
        inside = bool(mask[i, y, x])
        distance = distance_transform_edt(mask[i] if inside else ~mask[i], sampling=[1 / h, 1 / w])[y, x]
        rows.append({"frame": int(i), "projected_inside": inside, "signed_boundary_distance": float(distance * (1 if inside else -1)),
                     "region_tangent": tangent, "subject_image_position": xy[i]})
    return report("region geometry", {"rows": rows}, status="partial", entities=[eid, landmark],
                  frame="image", units="image_fraction", window=[lo, hi],
                  note="Projection only: image overlap is not physical contact. Tangents of round/irregular masks can be ambiguous; "
                       "use region_relation for 3D surface distances and inspect crossings before declaring an encounter.")


def encounters(ws, args):
    actor, category = str(args["actor"]), str(args["entity"])
    rows = census(ws, {"entity": category, **{k: args[k] for k in WINDOW_ARGS if k in args}}).data["value"]["instances"]
    proposals = [{"target": row["id"], "visibility_intervals_s": row["intervals_s"]} for row in rows]
    if args.get("geometry", False):
        scene = ws.scene()
        p, ok = position(ws, actor)
        # Travel direction, not camera facing: panning alone cannot create a pass.
        delta = np.gradient(p, axis=0)
        norms = np.linalg.norm(delta, axis=1)
        tangent = delta / np.maximum(norms[:, None], 1e-12)
        lo, hi = bounds(ws, args, scene)
        for proposal in proposals:
            q, qok = position(ws, proposal["target"])
            relative = q - p
            longitudinal = (relative * tangent).sum(1)
            valid = ok & qok & (norms > 1e-9)
            crossing = (longitudinal[:-1] > 0) & (longitudinal[1:] <= 0) & valid[:-1] & valid[1:]
            hits = np.flatnonzero(crossing) + 1
            hits = hits[(hits >= lo) & (hits <= hi)]
            proposal["candidate_passes"] = [{**frame_stamp(scene, i),
                                           "range_scales": float(np.linalg.norm(relative[i]) / scene.scale)} for i in hits]
    return report("encounter proposals", {"actor": actor, "proposals": proposals}, status="partial", entities=[actor, category],
                  note="Proposals only. Inspect before/during/after, record supported passes with actor and target IDs, then event_reduce. "
                       "Visibility, occlusion and panning alone do not establish a pass. Unique targets differ from encounter occurrences.")


SPATIAL_TOOLS = {"track_quality": track_quality, "projected_rotation": projected_rotation,
                 "region_geometry": region_geometry, "encounters": encounters}
SPATIAL_SPECS = {
    "track_quality": spec("Compare a fixed material-point cohort with the visible-surface centre and masks.",
        {"entity": "entity/member id", "anchor_s": "optional query anchor seconds", **WINDOW_ARGS}, ("entity",)),
    "projected_rotation": spec("Unwrapped image-plane rotation from persistent correspondences.",
        {"entity": "entity/member id", "anchor_s": "optional query anchor seconds", **WINDOW_ARGS}, ("entity",)),
    "region_geometry": spec("Projected containment, boundary distance and region tangent over time.",
        {"entity": "entity/member id", "landmark": "region id", **WINDOW_ARGS}, ("entity", "landmark")),
    "encounters": spec("Propose encounters from visibility; optionally verify longitudinal crossing geometry before inspection.",
        {"actor": "actor id or cam", "entity": "target category id", "geometry": "optional boolean", **WINDOW_ARGS}, ("actor", "entity")),
}
