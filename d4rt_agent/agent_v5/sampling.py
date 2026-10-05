"""Which depth cells of a mask may give an entity's 3D position: the boundary rule.

Draft rule (``v5-agent_draft.md``): never sample points close to the segmentation boundary. At a
silhouette the depth map blends foreground and background, and a SAM3 mask edge does not line up
exactly with the depth model's edge, so a cell there can sit metres behind the object. Two tests:

1. **Distance to the mask edge.** A pixel is interior when its distance to the nearest
   non-mask pixel is at least ``max(margin_px, margin_frac * inscribed radius)``: a fixed floor,
   and a margin that grows with the object so a large mask loses a proportionate rim. A depth
   cell is interior when at least half of it is covered by interior pixels.
2. **Depth edges.** An interior cell is dropped when its camera depth jumps by more than
   ``depth_edge`` (relative) to any of its 8 neighbours -- the depth map's own silhouette, which
   the mask edge alone does not find.

When fewer than ``min_cells`` survive, the margin is halved (down to one cell), never removed:
a mask too thin for an interior keeps its 2D evidence and gets no 3D position on that frame.
The runtime agent (``scene.lift_instances``) and the decomposition notebook (Step 4b) both call
this function, so the rule they show is the same one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from d4rt_agent.agent_v5.scene import _cell_coverage

MARGIN_PX = 4.0           # mask pixels (masks are stored at a long side of 512)
MARGIN_FRAC = 0.25        # of the mask's inscribed radius
DEPTH_EDGE = 0.10         # relative depth jump to a neighbour that marks a silhouette cell
MIN_CELLS = 3


@dataclass
class Interior:
    cells: np.ndarray | None      # (gh, gw) bool, or None when no interior survives
    margin_px: float              # the margin finally used (mask pixels)
    backed_off: bool              # a smaller margin than the first choice had to be used
    depth_edge_cells: int         # interior cells dropped as depth discontinuities
    reason: str = ""


def camera_depth(points: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """(gh, gw) depth along the camera's optical axis, from a world-frame cloud."""

    return ((np.asarray(points, dtype=np.float64) - cam_t) @ cam_R)[..., 2]


def depth_edges(depth: np.ndarray, tol: float = DEPTH_EDGE) -> np.ndarray:
    """(gh, gw) bool: cells whose depth differs from some 8-neighbour by more than ``tol``."""

    d = np.where(np.isfinite(depth) & (depth > 1e-9), depth, np.nan)
    padded = np.pad(d, 1, constant_values=np.nan)
    gh, gw = d.shape
    jumps = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = padded[1 + dy: 1 + dy + gh, 1 + dx: 1 + dx + gw]
            jumps.append(np.abs(d - nb) / d)
    with np.errstate(invalid="ignore"):
        worst = np.nanmax(np.stack(jumps), axis=0) if jumps else np.zeros_like(d)
    return np.nan_to_num(worst, nan=0.0) > tol


def interior_cells(mask: np.ndarray, points: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray, *,
                   margin_px: float = MARGIN_PX, margin_frac: float = MARGIN_FRAC,
                   depth_edge: float = DEPTH_EDGE, min_cells: int = MIN_CELLS) -> Interior:
    """The depth-grid cells of ``mask`` that are safe to read a 3D position from (see module doc).

    ``mask`` is at mask resolution, ``points`` the frame's (gh, gw, 3) world cloud.
    """

    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask, dtype=bool)
    gh, gw = points.shape[:2]
    if not mask.any():
        return Interior(None, float("nan"), False, 0, "empty mask")
    dist = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]   # the image border counts as an edge
    radius = float(dist.max())
    cell_px = max(mask.shape[0] / gh, mask.shape[1] / gw)
    # never less than one depth cell: the research pipeline's one-cell erosion is the floor
    first = max(float(margin_px), margin_frac * radius, cell_px)
    edges = depth_edges(camera_depth(points, cam_R, cam_t), depth_edge)

    margin, dropped = first, 0
    while True:
        cells = _cell_coverage(dist >= margin, (gh, gw)) >= 0.5
        on_edge = cells & edges
        cells &= ~edges
        dropped = int(on_edge.sum())
        if int(cells.sum()) >= min_cells:
            return Interior(cells, margin, margin < first, dropped)
        if margin <= cell_px:
            return Interior(None, margin, margin < first, dropped,
                            f"fewer than {min_cells} interior cells even at a {margin:.1f} px margin")
        margin = max(margin / 2.0, cell_px)
