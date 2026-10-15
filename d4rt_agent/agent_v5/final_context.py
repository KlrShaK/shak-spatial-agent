"""Revisioned visual artifacts and compact, evidence-specific answer packets."""

from dataclasses import asdict
import json

import numpy as np

from d4rt_agent.agent_v5.capabilities import serial, split_entity
from d4rt_agent.agent_v5.context import Conversation
from d4rt_agent.agent_v5.evidence import ADMINISTRATIVE, compact_result, current_components, short_result


VISUAL_KINDS = {"visual_state", "interaction", "semantic"}
MAX_TARGETED_IMAGES = 8
MAX_COUNT_IMAGES = 4
COLOURS = ((60, 220, 100), (240, 100, 70), (80, 160, 255), (240, 200, 60), (200, 90, 240), (60, 220, 210))


def render_frame(ws, target, gf, annotation="mask", crop=False, *, instances=None):
    """Render cached SAM masks; final packing must never invoke segmentation."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(ws.clip.frame(int(gf))).convert("RGB")
    missing = None
    if not target or annotation == "none":
        image.thumbnail((640, 640))
        return image, missing
    base, member = split_entity(target)
    g = ws.groundings.get(base)
    if g is None or not g.count:
        missing = "SAM masks unavailable; this plain image does not establish absence."
    else:
        aliases = ws.identity_aliases.get(base, {})
        is_group = ws.entity(base).kind == "group"
        members = {}
        if is_group and (member is not None or base in ws._tracks or base in ws.selected):
            count = len(ws.selected[base]) if base in ws.selected else len(ws.tracks_of(base))
            members = {i: ws.member_indices(base, i) for i in range(count)}
        chosen = instances if instances is not None else \
                 ws.member_indices(base, member) if member is not None else \
                 [k for indices in members.values() for k in indices] if members else \
                 ws.selected.get(base, list(range(g.count)))
        groups = {}
        for k in chosen:
            root = aliases.get(k, k)
            groups[root] = ws.instance_indices(base, root)
        if members:
            # A merged group member is one overlay/crop, even when it contains multiple raw SAM IDs.
            groups = {indices[0]: indices for indices in members.values() if set(indices) & set(chosen)}
        boxes = []
        for root, indices in sorted(groups.items()):
            mask = g.masks[indices, int(gf)].any(0)
            if not mask.any():
                continue
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)) > 0
            colour = COLOURS[root % len(COLOURS)]
            if annotation == "mask":
                pixels = np.array(image)
                pixels[mask] = (0.65 * pixels[mask] + 0.35 * np.asarray(colour)).astype(np.uint8)
                image = Image.fromarray(pixels)
            y, x = np.nonzero(mask)
            boxes.append(((int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1), root, colour))
        draw = ImageDraw.Draw(image)
        for box, root, colour in boxes:
            draw.rectangle(box, outline=colour, width=2)
            bound = next((i for i, indices in members.items() if root in indices), None)
            label = f"{base}[{bound}] (SAM {','.join(map(str, members[bound]))})" if bound is not None else \
                    f"SAM[{root}]" if is_group else f"{base}[{root}]"
            draw.text(box[:2], label, fill=colour)
        if crop and len(boxes) == 1:
            image = image.crop(boxes[0][0])
        if not boxes:
            missing = "No candidate mask is visible on this sampled frame; this does not prove physical absence."
    image.thumbnail((640, 640))
    return image, missing


def register(ws, vid, image, frames, *, entity=None, inspection_id=None, evidence_ids=(), requirement=None,
             annotation="none", crop=False, limitations=()):
    revisions = {entity.split("[")[0]: ws.revisions.get(entity.split("[")[0], 0)} if entity else {}
    ws.visuals[vid] = {"id": vid, "image": image, "frames": list(map(int, frames)), "entity": entity,
                       "inspection_id": inspection_id, "evidence_ids": list(evidence_ids), "requirement": requirement,
                       "revisions": revisions, "annotation": annotation, "crop": bool(crop),
                       "limitations": list(limitations)}
    return vid


def current_visual(ws, artifact):
    return all(ws.revisions.get(e, 0) == r for e, r in artifact["revisions"].items()) and all(
        e in ws.evidence and not ws.evidence[e].superseded for e in artifact["evidence_ids"])


def count_frames(ws, data, entity):
    g = ws.groundings.get(entity)
    lo, hi = [int(x) for x in data.get("window", [0, ws.clip.length - 1])]
    if g is None or not g.count:
        return [lo], []
    value = data.get("value", {})
    candidates = value.get("candidates", [])
    presence = {row["id"]: g.masks[row["indices"]].any(axis=(0, 2, 3)) for row in candidates}
    if not presence:
        return [lo], []
    aggregation = value.get("aggregation", "unique")
    if aggregation == "at_time":
        selected = [lo]
    elif aggregation == "max_simultaneous":
        confirmed = set(value.get("confirmed", presence))
        occupancy = sum((p.astype(int) for key, p in presence.items() if key in confirmed), np.zeros(ws.clip.length, int))
        selected = [lo + int(np.argmax(occupancy[lo:hi + 1]))]
    else:
        selected, unseen = [], set(presence)
        while unseen and len(selected) < MAX_COUNT_IMAGES:
            scores = [(sum(bool(presence[e][i]) for e in unseen), -i) for i in range(lo, hi + 1)]
            score, negative = max(scores)
            if not score:
                break
            gf = -negative
            selected.append(gf)
            unseen -= {e for e in unseen if presence[e][gf]}
    covered = {e for e, p in presence.items() if any(p[f] for f in selected)}
    return sorted(selected or [lo]), sorted(set(presence) - covered)


def _count_visuals(ws, result, record):
    entity = result.data.get("entities", record.entities)
    if not entity:
        return [], ["Counting verification has no resolved category binding."]
    entity = entity[0].split("[")[0]
    frames, unseen = count_frames(ws, result.data, entity)
    limitations = [f"Candidates not shown in counting verification: {unseen}"] if unseen else []
    ids = []
    candidates = result.data.get("value", {}).get("candidates", [])
    instances = sorted({i for row in candidates for i in row["indices"]})
    for gf in frames:
        vid = f"visual:{record.id}:{gf}:mask"
        if vid not in ws.visuals:
            image, missing = render_frame(ws, entity, gf, instances=instances)
            register(ws, vid, image, [gf], entity=entity, evidence_ids=[record.id], requirement=record.requirement,
                     annotation="mask", limitations=[missing] if missing else [])
        ids.append(vid)
    return ids, limitations


def build_packet(agent, task, state):
    from d4rt_agent.agent_v5.orchestrator import prompt

    ws, clip = state.workspace, task["clip"]
    system = prompt("system.md") + "\nFor final option assessment, use enough concise text to assess every supplied option before " \
             "the single final-answer JSON. The 1–3 sentence limit applies to routing and measurement decisions."
    conv = Conversation(system, agent.token_budget, agent._count_fn())
    requirements = state.program.requirements
    required = {r.id for r in requirements}
    pure_visual = bool(requirements) and all(r.kind in VISUAL_KINDS for r in requirements)
    conv.add("user", "Question: " + task["question"] + "\nOptions:\n" +
             "\n".join(f"{k}. {v}" for k, v in task["options"].items()) + "\nBindings: " +
             json.dumps([{"id": e.id, "phrase": e.phrase, "role": e.role} for e in state.program.entities]) +
             "\nRequirements: " + json.dumps(serial([asdict(r) for r in requirements])), kind="opening")
    current, dedup = [], set()
    if ws:
        for name, call, result in reversed(state.results):
            records = [r for r in current_components(ws, result) if r.requirement in required]
            selected = []
            for record in records:
                key = (record.requirement, record.component, tuple(record.entities), json.dumps(record.window, sort_keys=True))
                if key not in dedup:
                    dedup.add(key)
                    selected.append(record)
            if selected:
                current.append((name, call, result, selected))
        current.reverse()
    report_refs, dependencies, count_groups, notes = {}, set(), [], []
    for name, _, result, records in current:
        for record in records:
            dependencies.update(record.visual_ids)
        if name not in ADMINISTRATIVE or any(not r.available for r in records):
            report_refs.update({r.id: r for r in records})
            text = compact_result(result, records)
            conv.add("user", text, kind="tool", summary=short_result(result, records))
        if ws and name in ("count", "count_entities"):
            ids, limits = _count_visuals(ws, result, records[0])
            count_groups.append(ids)
            notes.extend(limits)
    groups = list(count_groups)
    if ws:
        needed = []
        optional = {}
        for artifact in ws.visuals.values():
            if not current_visual(ws, artifact):
                continue
            if artifact["inspection_id"] in dependencies or artifact["id"] in dependencies:
                needed.append(artifact)
            elif artifact["requirement"] in required and not artifact["evidence_ids"]:
                key = (artifact["requirement"], artifact["entity"], artifact["inspection_id"])
                optional.setdefault(key, []).append(artifact["id"])
        grouped = {}
        for artifact in needed:
            grouped.setdefault(artifact["inspection_id"] or artifact["id"], []).append(artifact["id"])
        groups.extend(grouped.values())
        # Preserve different scopes that current claims depend on; optional duplicate scopes use their latest inspection.
        latest = {}
        for ids in optional.values():
            a = ws.visuals[ids[0]]
            frames = [f for vid in ids for f in ws.visuals[vid]["frames"]]
            latest[(a["requirement"], a["entity"], min(frames), max(frames))] = ids
        groups.extend(latest.values())
    selected_ids, signatures = [], set()
    ordered = [list(dict.fromkeys(ids)) for ids in groups]
    ordered = [[ids[0], *([ids[-1]] if len(ids) > 1 else []), *ids[1:-1]] for ids in ordered if ids]
    while any(ordered) and len(selected_ids) < MAX_TARGETED_IMAGES:
        for ids in ordered:
            if not ids or len(selected_ids) >= MAX_TARGETED_IMAGES:
                continue
            vid = ids.pop(0)
            a = ws.visuals[vid]
            signature = (tuple(a["frames"]), a["entity"], a["annotation"], a["crop"], tuple(sorted(a["revisions"].items())))
            if signature not in signatures:
                signatures.add(signature)
                selected_ids.append(vid)
    turns = []
    metadata = {vid: {k: v for k, v in ws.visuals[vid].items() if k != "image"}
                for ids in groups for vid in ids}
    for vid in selected_ids:
        a = ws.visuals[vid]
        frame_labels = ", ".join(clip.label(f) for f in a["frames"])
        text = f"{vid}: {frame_labels}; target={a['entity']}; annotation={a['annotation']}. " + " ".join(a["limitations"])
        turn = conv.add("user", text, images=[(vid, a["image"])], kind="show",
                        summary=f"Visual {vid} omitted to fit context; it cannot be cited.")
        turns.append((vid, turn))
        metadata[vid] = {k: v for k, v in a.items() if k != "image"}
    standard = clip.answer_frames if pure_visual or agent.answer_views == "standard" else \
               clip.keyframes() if agent.answer_views == "overview" else []
    for gf in dict.fromkeys(standard):
        vid = f"frame:{gf}"
        turn = conv.add("user", f"{vid}: {clip.label(gf)}", images=[(vid, clip.pil(gf))], kind="show",
                        summary=f"Visual {vid} omitted to fit context; it cannot be cited.")
        turns.append((vid, turn))
        metadata[vid] = {"id": vid, "frames": [int(gf)], "annotation": "none"}
    omitted = sorted({vid for ids in groups for vid in ids} - set(selected_ids))
    if omitted:
        notes.append(f"Targeted visual limit omitted {len(omitted)} artifacts; see saved manifest.")
    return conv, report_refs, turns, metadata, notes, omitted


def visible_refs(report_refs, turns):
    refs = {key: {"available": r.available} for key, r in report_refs.items()}
    refs.update({vid: {"available": True} for vid, turn in turns if turn.images})
    return refs
