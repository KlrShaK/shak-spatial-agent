"""Annotated keyframes for the ``show`` tool: one 2x2 contact sheet, so a check costs one image.

* ``entities``: every grounded entity's mask outline and id on four keyframes;
* an entity id: that entity's smoothed 3D track projected into each keyframe's camera, with an
  arrow for where it goes next, plus its outline;
* ``camera``: the motion of a grid of scene points between consecutive keyframes (the image
  motion the camera's movement causes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

COLOURS = [(255, 64, 64), (64, 160, 255), (64, 220, 64), (255, 200, 0), (220, 64, 255), (0, 220, 220)]
TILE_LONG_SIDE = 448


def _project(scene: Any, X: np.ndarray, pos: int, size_hw: tuple[int, int]) -> tuple[float, float] | None:
    local = scene.to_camera(X, pos)
    if not np.isfinite(local).all() or local[2] <= 1e-6:
        return None
    K = scene.K[pos]
    u = K[0, 0] * local[0] / local[2] + K[0, 2]
    v = K[1, 1] * local[1] / local[2] + K[1, 2]
    h, w = size_hw
    return u * w / scene.model_hw[1], v * h / scene.model_hw[0]


def _outline(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> tuple[int, int] | None:
    import cv2

    full = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, colour, 2)
    ys, xs = np.nonzero(full)
    return (int(xs.mean()), int(ys.min())) if len(xs) else None


def _tile(image: np.ndarray, label: str) -> np.ndarray:
    import cv2

    scale = TILE_LONG_SIDE / max(image.shape[:2])
    tile = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return tile


def render(ws: Any, target: str) -> tuple[Any, str]:
    import cv2
    from PIL import Image

    scene = ws.scene()
    clip = ws.clip
    keys = clip.keyframes(4)
    positions = [int(np.argmin(np.abs(scene.gf_index - g))) for g in keys]
    tiles = []
    for n, (gf, pos) in enumerate(zip(keys, positions)):
        image = clip.frame(gf).copy()
        h, w = image.shape[:2]
        if target == "entities":
            for c, entity in enumerate(e for e in ws.program.entities if e.kind != "camera"):
                if entity.id not in ws.groundings:
                    continue
                for track in ws.tracks_of(entity.id):
                    mask = ws.masks_at(track.id)(pos)
                    if mask is None:
                        continue
                    anchor = _outline(image, mask, COLOURS[c % len(COLOURS)])
                    if anchor:
                        cv2.putText(image, track.id, (anchor[0], max(anchor[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, COLOURS[c % len(COLOURS)], 2)
        elif target == "camera":
            if n + 1 < len(positions):
                nxt = positions[n + 1]
                cells = scene.points[pos][::6, ::6].reshape(-1, 3)
                for X in cells[np.isfinite(cells).all(1)][:: max(1, len(cells) // 60)]:
                    a, b = _project(scene, X, pos, (h, w)), _project(scene, X, nxt, (h, w))
                    if a and b:
                        cv2.arrowedLine(image, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 200, 0), 2,
                                        tipLength=0.3)
        else:
            track = ws.track(target)
            if track is not None and not isinstance(track, str):
                mask = ws.masks_at(target)(pos)
                if mask is not None:
                    _outline(image, mask, COLOURS[0])
                pts = [_project(scene, track.smooth[p], pos, (h, w)) for p in range(scene.length)
                       if np.isfinite(track.smooth[p]).all()]
                pts = [p for p in pts if p is not None]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(image, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), COLOURS[1], 2)
                here = _project(scene, track.smooth[pos], pos, (h, w)) if np.isfinite(track.smooth[pos]).all() else None
                if here:
                    cv2.circle(image, (int(here[0]), int(here[1])), 7, COLOURS[2], -1)
        tiles.append(_tile(image, clip.label(gf)))
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    sheet = np.zeros((th * 2, tw * 2, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles[:4]):
        r, c = divmod(i, 2)
        sheet[r * th: r * th + tile.shape[0], c * tw: c * tw + tile.shape[1]] = tile
    captions = {
        "entities": "Four keyframes (left to right, top to bottom) with every grounded entity outlined and labelled "
                    "by its id.",
        "camera": "Four keyframes; arrows show where scene points move in the picture by the next keyframe (the "
                  "image motion the camera causes).",
    }
    caption = captions.get(target, f"Four keyframes with {target} outlined; the blue line is its smoothed 3D track "
                                   "projected into each keyframe, the green dot its position at that keyframe.")
    return Image.fromarray(sheet), caption
