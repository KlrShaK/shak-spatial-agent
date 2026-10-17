"""Whole-clip mask tables, labelled overlays, and on-demand noun inventories."""

from __future__ import annotations

import colorsys
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from ..calls import parse_object, valid_entity
from ..reports import Report, segment_motion
from ..workspace import WorkspaceError

COLOURS = ("red", "blue", "green", "yellow", "magenta", "cyan", "orange", "purple",
           "lime", "pink", "teal", "brown")
COLUMNS = ("visible", "object", "size (% of image)", "at image edge")


def _runs(present):
    indices = np.flatnonzero(present)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _span(workspace, start, end):
    a, b = workspace.time_labels([start, end])
    if start == end:
        return a
    return a.removesuffix(" s") + "–" + b if a.endswith(" s") else a + "–" + b


def _spans(workspace, present):
    return "; ".join(_span(workspace, start, end) for start, end in _runs(present)) or "—"


def _percent(value):
    return f"{100 * float(value):.1f}".rstrip("0").rstrip(".") + "%"


def _size_profile(workspace, area, present, pixels):
    """Merge rising/falling runs above mask jitter; retain strong reversals."""
    profiles, raw = [], []
    for start, end in _runs(present):
        values = area[start:end + 1]
        if len(values) == 1:
            profiles.append(_percent(values[0]))
            continue
        jitter = float(np.median(abs(np.diff(values, n=2)))) / 2 if len(values) > 2 else 0.0
        noise = max(0.5 / pixels, 3 * jitter)
        points = np.column_stack((values, np.zeros((len(values), 2))))
        phases = segment_motion(np.arange(start, end + 1), points, noise,
                                axes=(("larger", "smaller"), ("", ""), ("", "")))
        sizes = [float(values[0])]
        for phase in phases:
            last = float(np.interp(phase.end, np.arange(start, end + 1), values))
            if phase.label != "still":
                sizes.append(last)
            raw.append({"start": workspace.time_labels([round(phase.start)])[0],
                        "end": workspace.time_labels([round(phase.end)])[0],
                        "change": "unchanged" if phase.label == "still" else phase.label,
                        "start_fraction": float(np.interp(phase.start, np.arange(start, end + 1), values)),
                        "end_fraction": last})
        profiles.append(" → ".join(_percent(size) for size in sizes))
    return "; ".join(profiles), raw


def _observations(workspace, entities):
    observations, missing = [], []
    seen = set()
    for name in entities:
        canonical = workspace.canonical_name(name)
        if re.fullmatch(r'.+ \d+', canonical):
            # Resolve numbered labels before SAM; stale labels cannot become nouns.
            resolved = workspace.resolve(canonical)
            canonical = resolved.label
        group = workspace.ground(canonical)
        requested = workspace.canonical_name(name)
        selected = [entity for entity in group.entities if entity.label == requested]
        selected = selected or group.entities
        if not selected:
            missing.append(workspace.missing_message(name))
        for entity in selected:
            if entity.label in seen:
                continue
            seen.add(entity.label)
            mask = group.grounding.masks[list(entity.instances)].any(axis=0)
            observations.append((entity, mask))
    return observations, missing


def _mask_report(workspace, observations, missing, title="Segmentation masks"):
    rows, instances = [], []
    for entity, mask in observations:
        present = mask.any(axis=(1, 2))
        area = mask.mean(axis=(1, 2))
        size, phases = _size_profile(workspace, area, present, mask.shape[1] * mask.shape[2])
        edges = {"left": mask[:, :, 0].any(axis=1), "right": mask[:, :, -1].any(axis=1),
                 "top": mask[:, 0, :].any(axis=1), "bottom": mask[:, -1, :].any(axis=1)}
        edge_text = "; ".join(f"{edge} edge at {_spans(workspace, contact)}"
                              for edge, contact in edges.items() if contact.any()) or "—"
        rows.append((_spans(workspace, present), entity.label, size, edge_text))
        instances.append({"label": entity.label, "category": entity.category,
                          "visible": present.tolist(), "area_fraction": area.tolist(),
                          "size_phases": phases, "edges": {k: v.tolist() for k, v in edges.items()}})
    if not rows:
        return Report.unavailable(title, " ".join(missing) or "No segmentable entities were listed.",
                                  raw={"instances": [], "missing": missing})
    return Report(title, headline=f"{len(rows)} segmented instances; visibility and image-size changes are listed by object.", rows=rows, columns=COLUMNS,
                  status="partial" if missing else "ok", reason=" ".join(missing),
                  reading_full="Size is the share of image pixels in the mask. Edge contact marks possible cropping; inspect frames for occlusion.",
                  reading_short="Size is image share; edge contact marks possible cropping.",
                  raw={"instances": instances, "missing": missing})


def _colour(index, name=None):
    if name in COLOURS:
        return ImageColor.getrgb(name)
    if index < len(COLOURS):
        return ImageColor.getrgb(COLOURS[index])
    return tuple(round(255 * value) for value in colorsys.hsv_to_rgb((index * 0.618034) % 1, 0.7, 0.95))


def _overlay(workspace, index, observations, colours=None, legend=False):
    """V5's mask blend and box overlay, with V6's stable instance labels."""
    image = Image.fromarray(workspace.clip.frame(index)).convert("RGB")
    image.thumbnail((640, 640))
    pixels = np.array(image)
    boxes, entries = [], []
    for number, (entity, masks) in enumerate(observations):
        name = colours[number] if colours is not None else None
        rgb = _colour(number, name)
        mask = np.asarray(Image.fromarray(masks[index]).resize(image.size, Image.Resampling.NEAREST), bool)
        suffix = "" if mask.any() else " (not visible at this time)"
        entries.append((f"{name}: {entity.label}{suffix}" if name else entity.label + suffix, rgb))
        if not mask.any():
            continue
        pixels[mask] = (0.6 * pixels[mask] + 0.4 * np.asarray(rgb)).astype(np.uint8)
        y, x = np.nonzero(mask)
        boxes.append(((int(x.min()), int(y.min()), int(x.max()), int(y.max())), entity.label, rgb))
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for box, label, rgb in boxes:
        draw.rectangle(box, outline=rgb, width=2)
        draw.text(box[:2], label, fill=rgb)
    if legend:
        panel = Image.new("RGB", (max(image.width, 360), image.height + 18 * len(entries)), "white")
        panel.paste(image, (0, 0))
        draw = ImageDraw.Draw(panel)
        for number, (label, rgb) in enumerate(entries):
            y = image.height + number * 18
            draw.rectangle((2, y + 3, 12, y + 13), fill=rgb)
            draw.text((18, y + 2), label, fill="black")
        image = panel
    return image


def get_segmentation_mask(workspace, entities, times=None):
    if not entities:
        raise WorkspaceError("At least one entity is required for segmentation.")
    observations, missing = _observations(workspace, entities)
    report = _mask_report(workspace, observations, missing)
    if times is not None:
        views = workspace.frames(times=times, limit=8)
        report.images = [(view.label, _overlay(workspace, view.index, observations)) for view in views]
    return report


def _inventory_frames(workspace, observations):
    if not observations:
        return []
    presence = np.stack([mask.any(axis=(1, 2)) for _, mask in observations])
    occupancy = presence.sum(axis=0)
    # Start at peak occupancy, then cover instances absent from that frame.
    selected, unseen = [], np.ones(len(observations), dtype=bool)
    while unseen.any() and len(selected) < 8:
        coverage = presence[unseen].sum(axis=0)
        if not coverage.max():
            break
        candidates = np.flatnonzero(coverage == coverage.max())
        index = int(candidates[np.argmax(occupancy[candidates])])
        selected.append(index)
        unseen &= ~presence[:, index]
    return [(workspace.mark_shown(index), _overlay(workspace, index, observations)) for index in sorted(selected)]


def detect_all_entities(workspace, qwen, question, options, caption):
    instruction = (Path(__file__).parent.parent / "prompts" / "inventory.md").read_text().strip()
    context = {"question": question, "options": options, "caption": caption}
    response = qwen.generate([{"role": "system", "content": instruction},
                              {"role": "user", "content": json.dumps(context)}], max_new_tokens=256)
    try:
        nouns = parse_object(response)["entities"]
        if not isinstance(nouns, list) or not all(valid_entity(noun) for noun in nouns):
            raise ValueError("Invalid inventory noun list.")
    except (ValueError, KeyError, TypeError) as error:
        raise WorkspaceError("The inventory viewer did not return a list of short entity names.") from error
    nouns = list(dict.fromkeys(workspace.canonical_name(noun) for noun in nouns))
    nouns = [noun for noun in nouns if noun != "camera"]
    observations, missing = _observations(workspace, nouns)
    images = _inventory_frames(workspace, observations)
    report = _mask_report(workspace, observations, missing, title="Entity inventory")
    counts = {noun: sum(entity.category == noun for entity, _ in observations) for noun in nouns}
    report.headline = "; ".join(f"{noun}: {count} segmented instance{'s' if count != 1 else ''}"
                               for noun, count in counts.items())
    report.images = images
    report.raw.update(nouns=nouns, counts=counts)
    return report


def candidate_report(workspace, selection):
    group = workspace.ground(selection.name)
    entities = {entity.label: entity for entity in group.entities}
    observations = [(entities[choice.label], group.grounding.masks[list(choice.instances)].any(axis=0))
                    for choice in selection.choices]
    occupancy = sum((mask.any(axis=(1, 2)).astype(int) for _, mask in observations),
                    np.zeros(workspace.clip.length, dtype=int))
    colours = [choice.colour for choice in selection.choices]
    present = np.stack([mask.any(axis=(1, 2)) for _, mask in observations])
    indices, unseen = [], np.ones(len(observations), dtype=bool)
    while unseen.any():
        index = int(np.argmax(present[unseen].sum(axis=0)))
        if not present[unseen, index].any():
            raise WorkspaceError("A segmentation candidate has no visible frame.")
        indices.append(index)
        unseen &= ~present[:, index]
    tiles, labels = [], []
    for index in sorted(indices):
        tile = _overlay(workspace, index, observations, colours, legend=True)
        tile.thumbnail((480, 480))
        tiles.append(tile)
        labels.append(workspace.mark_shown(index))
    columns = min(3, len(tiles))
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles) + 20
    image = Image.new("RGB", (columns * width, ((len(tiles) + columns - 1) // columns) * height), "white")
    draw = ImageDraw.Draw(image)
    for number, (label, tile) in enumerate(zip(labels, tiles)):
        x, y = (number % columns) * width, (number // columns) * height
        draw.text((x + 2, y + 2), label, fill="black")
        image.paste(tile, (x, y + 20))
    return Report("Choose a segmentation candidate",
                  headline=f"Choose the intended {selection.name} by colour; every candidate is visible in this image.",
                  status="partial", reason="An instance choice is needed before measuring.",
                  images=[("; ".join(labels), image)],
                  raw={"name": selection.name, "frame_index": indices[0], "frame_indices": indices,
                       "candidates": [{"label": choice.label, "colour": choice.colour,
                                       "rgb": _colour(i, choice.colour)}
                                      for i, choice in enumerate(selection.choices)]})


def choose_segmentation_candidate(workspace, colour):
    entity = workspace.choose(colour)
    return Report("Segmentation choice", headline=f"Selected {entity.label}.",
                  raw={"label": entity.label, "category": entity.category})
